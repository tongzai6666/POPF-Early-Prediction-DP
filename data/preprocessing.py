from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

STAGE_TO_ID = {
    "baseline": 0,
    "preoperative": 1,
    "intraoperative": 2,
    "postoperative_24h": 3,
}


def load_feature_schema(path: str | Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    features = obj["features"]
    for feat in features:
        if feat["stage"] not in STAGE_TO_ID:
            raise ValueError(f"Unknown stage {feat['stage']} for {feat['name']}")
        if feat["type"] not in {"numeric", "categorical"}:
            raise ValueError(f"Unknown type {feat['type']} for {feat['name']}")
    return features


@dataclass
class PreprocessorState:
    numeric_stats: Dict[str, Dict[str, float]]
    category_vocab: Dict[str, int]
    feature_names: List[str]
    feature_descriptions: List[str]
    feature_types: List[str]
    stage_ids: List[int]
    core_features: List[str]


class DataPreprocessor:
    """Training-only preprocessing for heterogeneous clinical variables.

    Numeric variables are z-standardized using mean and SD estimated only from the
    training cohort. Categorical values are mapped to a global vocabulary using
    variable-specific keys ("feature::category"). Missing values are *not* imputed
    as clinical observations: zero is only an internal tensor value and the observed
    mask remains 0 so the model can exclude the variable from effective attention.
    """

    def __init__(self, feature_schema: List[Dict]):
        self.schema = feature_schema
        self.state: PreprocessorState | None = None

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        numeric_stats: Dict[str, Dict[str, float]] = {}
        category_vocab: Dict[str, int] = {"<MISSING_OR_UNKNOWN>": 0}
        next_id = 1

        for feat in self.schema:
            name = feat["name"]
            if name not in df.columns:
                continue
            if feat["type"] == "numeric":
                x = pd.to_numeric(df[name], errors="coerce")
                mean = float(x.mean()) if x.notna().any() else 0.0
                std = float(x.std(ddof=0)) if x.notna().any() else 1.0
                if not np.isfinite(std) or std < 1e-8:
                    std = 1.0
                numeric_stats[name] = {"mean": mean, "std": std}
            else:
                values = df[name].dropna().astype(str).unique().tolist()
                for value in sorted(values):
                    key = f"{name}::{value}"
                    if key not in category_vocab:
                        category_vocab[key] = next_id
                        next_id += 1

        self.state = PreprocessorState(
            numeric_stats=numeric_stats,
            category_vocab=category_vocab,
            feature_names=[f["name"] for f in self.schema],
            feature_descriptions=[f["description"] for f in self.schema],
            feature_types=[f["type"] for f in self.schema],
            stage_ids=[STAGE_TO_ID[f["stage"]] for f in self.schema],
            core_features=[f["name"] for f in self.schema if f.get("core", False)],
        )
        return self

    def transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        if self.state is None:
            raise RuntimeError("Preprocessor must be fit or loaded before transform().")

        n, f = len(df), len(self.schema)
        numeric_values = np.zeros((n, f), dtype=np.float32)
        category_ids = np.zeros((n, f), dtype=np.int64)
        feature_type_ids = np.zeros(f, dtype=np.int64)  # 0 numeric, 1 categorical
        observed_mask = np.zeros((n, f), dtype=np.float32)

        for j, feat in enumerate(self.schema):
            name = feat["name"]
            feature_type_ids[j] = 0 if feat["type"] == "numeric" else 1
            if name not in df.columns:
                continue

            s = df[name]
            observed = ~s.isna()
            observed_mask[:, j] = observed.astype(np.float32).to_numpy()

            if feat["type"] == "numeric":
                x = pd.to_numeric(s, errors="coerce")
                stats = self.state.numeric_stats.get(name, {"mean": 0.0, "std": 1.0})
                z = (x - stats["mean"]) / stats["std"]
                numeric_values[:, j] = z.fillna(0.0).astype(np.float32).to_numpy()
            else:
                ids = []
                for val in s:
                    if pd.isna(val):
                        ids.append(0)
                    else:
                        ids.append(self.state.category_vocab.get(f"{name}::{str(val)}", 0))
                category_ids[:, j] = np.asarray(ids, dtype=np.int64)

        return {
            "numeric_values": numeric_values,
            "category_ids": category_ids,
            "feature_type_ids": feature_type_ids,
            "observed_mask": observed_mask,
            "stage_ids": np.asarray(self.state.stage_ids, dtype=np.int64),
        }

    @property
    def num_categories(self) -> int:
        if self.state is None:
            raise RuntimeError("Preprocessor not fit/loaded")
        return max(self.state.category_vocab.values()) + 1

    def save(self, path: str | Path) -> None:
        if self.state is None:
            raise RuntimeError("Nothing to save before fit().")
        payload = {
            "numeric_stats": self.state.numeric_stats,
            "category_vocab": self.state.category_vocab,
            "feature_names": self.state.feature_names,
            "feature_descriptions": self.state.feature_descriptions,
            "feature_types": self.state.feature_types,
            "stage_ids": self.state.stage_ids,
            "core_features": self.state.core_features,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path, feature_schema: List[Dict]) -> "DataPreprocessor":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        obj = cls(feature_schema)
        obj.state = PreprocessorState(**payload)
        return obj


def derive_class_weights(df: pd.DataFrame, leakage_col: str, cr_col: str) -> Tuple[float, float]:
    """Return BCE pos_weight for primary leakage and conditional CR-POPF tasks.

    Primary pos_weight = n(no leakage) / n(leakage).
    Severity pos_weight is computed only among leakage-positive patients:
    n(BL) / n(CR-POPF).
    """
    leakage = df[leakage_col].astype(int).to_numpy()
    cr = df[cr_col].astype(int).to_numpy()
    n_pos = max(int((leakage == 1).sum()), 1)
    n_neg = max(int((leakage == 0).sum()), 1)
    primary = n_neg / n_pos

    m = leakage == 1
    n_cr = max(int((cr[m] == 1).sum()), 1)
    n_bl = max(int((cr[m] == 0).sum()), 1)
    severity = n_bl / n_cr
    return float(primary), float(severity)
