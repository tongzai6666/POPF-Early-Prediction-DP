from __future__ import annotations

from typing import Dict

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .preprocessing import DataPreprocessor


class POPFDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        preprocessor: DataPreprocessor,
        patient_id_col: str = "patient_id",
        leakage_label_col: str = "label_leakage",
        cr_label_col: str = "label_cr_popf",
    ):
        self.df = df.reset_index(drop=True).copy()
        self.patient_id_col = patient_id_col
        self.leakage_label_col = leakage_label_col
        self.cr_label_col = cr_label_col
        self.arrays = preprocessor.transform(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        item = {
            "patient_id": str(row[self.patient_id_col]) if self.patient_id_col in self.df.columns else str(idx),
            "numeric_values": torch.tensor(self.arrays["numeric_values"][idx], dtype=torch.float32),
            "category_ids": torch.tensor(self.arrays["category_ids"][idx], dtype=torch.long),
            "feature_type_ids": torch.tensor(self.arrays["feature_type_ids"], dtype=torch.long),
            "observed_mask": torch.tensor(self.arrays["observed_mask"][idx], dtype=torch.bool),
            "stage_ids": torch.tensor(self.arrays["stage_ids"], dtype=torch.long),
        }
        if self.leakage_label_col in self.df.columns:
            item["label_leakage"] = torch.tensor(float(row[self.leakage_label_col]), dtype=torch.float32)
        if self.cr_label_col in self.df.columns:
            item["label_cr_popf"] = torch.tensor(float(row[self.cr_label_col]), dtype=torch.float32)
        return item


def collate_fn(batch):
    out = {"patient_id": [x["patient_id"] for x in batch]}
    tensor_keys = [k for k in batch[0] if k != "patient_id"]
    for key in tensor_keys:
        out[key] = torch.stack([x[key] for x in batch], dim=0)
    return out


def make_loader(dataset: POPFDataset, batch_size: int, shuffle: bool, num_workers: int = 2) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
