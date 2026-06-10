"""
Preprocessing Module
==================
Handles data preprocessing including missing value masking, feature normalization,
and clinical composite index calculation (PNI, NLR, CAR).

Paper Section 2.3: "Preoperative laboratory data included complete blood counts (CBC), 
albumin, and C-reactive protein (CRP), from which the preoperative prognostic nutritional 
index (PNI), neutrophil-to-lymphocyte ratio (NLR), and C-reactive protein-to-albumin 
ratio (CAR) were derived."
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import os


class ClinicalIndexCalculator:
    """
    Calculates clinical composite indices from raw laboratory values.

    PNI (Prognostic Nutritional Index) = 10 * serum albumin (g/dL) + 0.005 * total lymphocyte count (per mm3)
    NLR (Neutrophil-to-Lymphocyte Ratio) = neutrophil count / lymphocyte count
    CAR (C-reactive protein-to-Albumin Ratio) = CRP (mg/L) / albumin (g/dL)
    """

    @staticmethod
    def calculate_pni(albumin: float, lymphocyte: float) -> float:
        """
        Calculate PNI.

        Args:
            albumin: Serum albumin in g/dL
            lymphocyte: Total lymphocyte count per mm3

        Returns:
            PNI value
        """
        if albumin <= 0 or lymphocyte <= 0:
            return np.nan
        return 10 * albumin + 0.005 * lymphocyte

    @staticmethod
    def calculate_nlr(neutrophil: float, lymphocyte: float) -> float:
        """
        Calculate NLR.

        Args:
            neutrophil: Neutrophil count
            lymphocyte: Lymphocyte count

        Returns:
            NLR value
        """
        if lymphocyte <= 0:
            return np.nan
        return neutrophil / lymphocyte

    @staticmethod
    def calculate_car(crp: float, albumin: float) -> float:
        """
        Calculate CAR.

        Args:
            crp: C-reactive protein in mg/L
            albumin: Serum albumin in g/dL

        Returns:
            CAR value
        """
        if albumin <= 0:
            return np.nan
        return crp / albumin


class FeatureNormalizer:
    """
    Feature normalization handler for clinical variables.

    Uses different strategies for different variable types:
    - StandardScaler for approximately normal distributions (age, BMI, lab values)
    - Log transform + StandardScaler for skewed distributions (DFA, operative time)
    - MinMaxScaler for bounded ratios (P/S, P/L, P/PM)
    """

    def __init__(self, scaler_path: Optional[str] = None):
        """
        Args:
            scaler_path: Path to load pre-fitted scalers (for inference)
        """
        self.scalers = {}
        self.log_transform_vars = [
            'dfa_24h', 'operative_time', 'blood_loss', 'crp_preop', 'crp_24h',
            'car_preop', 'car_24h'
        ]
        self.minmax_vars = [
            'p_s_ratio', 'p_l_ratio', 'p_pm_ratio', 'sex', 'hypertension', 
            'diabetes', 'transfusion', 'surgical_approach', 'stump_management'
        ]

        if scaler_path and os.path.exists(scaler_path):
            self.load_scalers(scaler_path)

    def fit(self, data: pd.DataFrame, variable_names: List[str]):
        """
        Fit scalers on training data.

        Args:
            data: DataFrame with columns [variable_name, value]
            variable_names: List of all variable names to fit
        """
        for var_name in variable_names:
            var_data = data[data['variable_name'] == var_name]['value'].dropna()

            if len(var_data) == 0:
                continue

            if var_name in self.log_transform_vars:
                # Log transform then standardize
                log_data = np.log1p(var_data.values)
                scaler = StandardScaler()
                scaler.fit(log_data.reshape(-1, 1))
            elif var_name in self.minmax_vars:
                # MinMax scaling to [0, 1]
                scaler = MinMaxScaler()
                scaler.fit(var_data.values.reshape(-1, 1))
            else:
                # Standard scaling (z-score normalization)
                scaler = StandardScaler()
                scaler.fit(var_data.values.reshape(-1, 1))

            self.scalers[var_name] = scaler

    def transform(self, var_name: str, value: float) -> float:
        """
        Transform a single value.

        Args:
            var_name: Variable name
            value: Raw value

        Returns:
            Normalized value
        """
        if var_name not in self.scalers or np.isnan(value):
            return value

        scaler = self.scalers[var_name]

        if var_name in self.log_transform_vars:
            value = np.log1p(value)

        normalized = scaler.transform(np.array([[value]]))[0, 0]
        return float(normalized)

    def inverse_transform(self, var_name: str, normalized_value: float) -> float:
        """Inverse transform for interpretability."""
        if var_name not in self.scalers:
            return normalized_value

        scaler = self.scalers[var_name]
        raw = scaler.inverse_transform(np.array([[normalized_value]]))[0, 0]

        if var_name in self.log_transform_vars:
            raw = np.expm1(raw)

        return float(raw)

    def save_scalers(self, path: str):
        """Save fitted scalers to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'scalers': self.scalers,
                'log_transform_vars': self.log_transform_vars,
                'minmax_vars': self.minmax_vars
            }, f)

    def load_scalers(self, path: str):
        """Load fitted scalers from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.scalers = data['scalers']
            self.log_transform_vars = data['log_transform_vars']
            self.minmax_vars = data['minmax_vars']


class MissingValuePreprocessor:
    """
    Preprocessor for handling missing values in clinical data.

    Implements attention mask generation and missing value strategies:
    - Variables recorded: attention_mask = 1
    - Variables not recorded (center-specific missing): attention_mask = 0
    - Missing values are NOT imputed; handled by attention masking in model
    """

    def __init__(self, standard_variables: List[str]):
        """
        Args:
            standard_variables: Complete list of expected variables
        """
        self.standard_variables = standard_variables
        self.var_to_idx = {var: idx for idx, var in enumerate(standard_variables)}

    def create_attention_mask(
        self,
        available_variables: List[str],
        max_length: int
    ) -> torch.Tensor:
        """
        Create binary attention mask for available variables.

        Args:
            available_variables: List of variable names present for this patient
            max_length: Maximum sequence length (padding)

        Returns:
            attention_mask: (max_length,) binary tensor
        """
        mask = torch.zeros(max_length, dtype=torch.float32)

        for var_name in available_variables:
            if var_name in self.var_to_idx:
                idx = self.var_to_idx[var_name]
                if idx < max_length:
                    mask[idx] = 1.0

        return mask

    def align_variables(
        self,
        patient_data: Dict[str, float],
        max_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Align patient variables to standard order with padding.

        Args:
            patient_data: Dict mapping variable_name -> value
            max_length: Maximum sequence length

        Returns:
            values: (max_length,) tensor with aligned values (0 for missing)
            attention_mask: (max_length,) binary mask
            ordered_names: List of variable names in standard order
        """
        values = torch.zeros(max_length, dtype=torch.float32)
        attention_mask = torch.zeros(max_length, dtype=torch.float32)
        ordered_names = [""] * max_length

        for var_name, value in patient_data.items():
            if var_name in self.var_to_idx:
                idx = self.var_to_idx[var_name]
                if idx < max_length and not np.isnan(value):
                    values[idx] = value
                    attention_mask[idx] = 1.0
                    ordered_names[idx] = var_name

        return values, attention_mask, ordered_names


class DataPreprocessor:
    """
    Main data preprocessing pipeline combining all preprocessing steps.

    Pipeline:
    1. Calculate composite indices (PNI, NLR, CAR) from raw labs
    2. Normalize features using fitted scalers
    3. Create attention masks for missing variables
    4. Align to standard variable order
    """

    def __init__(
        self,
        standard_variables: List[str],
        scaler_path: Optional[str] = None,
        fit_scalers: bool = True
    ):
        """
        Args:
            standard_variables: Complete list of expected variables
            scaler_path: Path to load/save scalers
            fit_scalers: Whether to fit scalers (True for training, False for inference)
        """
        self.standard_variables = standard_variables
        self.index_calculator = ClinicalIndexCalculator()
        self.normalizer = FeatureNormalizer(scaler_path if not fit_scalers else None)
        self.missing_handler = MissingValuePreprocessor(standard_variables)
        self.scaler_path = scaler_path
        self.fit_scalers = fit_scalers

    def compute_composite_indices(self, patient_data: Dict) -> Dict:
        """
        Compute PNI, NLR, CAR from raw laboratory values if not already present.

        Args:
            patient_data: Dictionary of patient variables

        Returns:
            Updated patient data with composite indices
        """
        data = patient_data.copy()

        # PNI: needs albumin and lymphocyte
        if 'pni_preop' not in data or np.isnan(data['pni_preop']):
            if 'albumin_preop' in data and 'lymphocyte_preop' in data:
                if not np.isnan(data['albumin_preop']) and not np.isnan(data['lymphocyte_preop']):
                    data['pni_preop'] = self.index_calculator.calculate_pni(
                        data['albumin_preop'], data['lymphocyte_preop']
                    )

        if 'pni_24h' not in data or np.isnan(data['pni_24h']):
            if 'albumin_24h' in data and 'lymphocyte_24h' in data:
                if not np.isnan(data['albumin_24h']) and not np.isnan(data['lymphocyte_24h']):
                    data['pni_24h'] = self.index_calculator.calculate_pni(
                        data['albumin_24h'], data['lymphocyte_24h']
                    )

        # NLR: needs neutrophil and lymphocyte
        if 'nlr_preop' not in data or np.isnan(data['nlr_preop']):
            if 'neutrophil_preop' in data and 'lymphocyte_preop' in data:
                if not np.isnan(data['neutrophil_preop']) and not np.isnan(data['lymphocyte_preop']):
                    data['nlr_preop'] = self.index_calculator.calculate_nlr(
                        data['neutrophil_preop'], data['lymphocyte_preop']
                    )

        if 'nlr_24h' not in data or np.isnan(data['nlr_24h']):
            if 'neutrophil_24h' in data and 'lymphocyte_24h' in data:
                if not np.isnan(data['neutrophil_24h']) and not np.isnan(data['lymphocyte_24h']):
                    data['nlr_24h'] = self.index_calculator.calculate_nlr(
                        data['neutrophil_24h'], data['lymphocyte_24h']
                    )

        # CAR: needs CRP and albumin
        if 'car_preop' not in data or np.isnan(data['car_preop']):
            if 'crp_preop' in data and 'albumin_preop' in data:
                if not np.isnan(data['crp_preop']) and not np.isnan(data['albumin_preop']):
                    data['car_preop'] = self.index_calculator.calculate_car(
                        data['crp_preop'], data['albumin_preop']
                    )

        if 'car_24h' not in data or np.isnan(data['car_24h']):
            if 'crp_24h' in data and 'albumin_24h' in data:
                if not np.isnan(data['crp_24h']) and not np.isnan(data['albumin_24h']):
                    data['car_24h'] = self.index_calculator.calculate_car(
                        data['crp_24h'], data['albumin_24h']
                    )

        return data

    def preprocess_patient(
        self,
        patient_data: Dict[str, float],
        timestamps: Dict[str, str],
        max_length: int = 35
    ) -> Dict:
        """
        Complete preprocessing pipeline for a single patient.

        Args:
            patient_data: Raw patient variables
            timestamps: Dict mapping variable_name -> timestamp
            max_length: Maximum sequence length

        Returns:
            Preprocessed data dictionary ready for model input
        """
        # Step 1: Compute composite indices
        data = self.compute_composite_indices(patient_data)

        # Step 2: Normalize features
        normalized_data = {}
        for var_name, value in data.items():
            if not np.isnan(value):
                normalized_data[var_name] = self.normalizer.transform(var_name, value)
            else:
                normalized_data[var_name] = np.nan

        # Step 3: Align and create mask
        values, attention_mask, ordered_names = self.missing_handler.align_variables(
            normalized_data, max_length
        )

        # Get timestamps in aligned order
        aligned_timestamps = [timestamps.get(name, "preop") for name in ordered_names]

        return {
            'values': values,
            'attention_mask': attention_mask,
            'variable_names': ordered_names,
            'timestamps': aligned_timestamps
        }

    def fit_normalizers(self, dataset: pd.DataFrame):
        """
        Fit feature normalizers on training dataset.

        Args:
            dataset: DataFrame with all training data
        """
        self.normalizer.fit(dataset, self.standard_variables)

        if self.scaler_path:
            self.normalizer.save_scalers(self.scaler_path)

    def save(self, path: str):
        """Save preprocessor state."""
        if self.scaler_path:
            self.normalizer.save_scalers(self.scaler_path)

    def load(self, path: str):
        """Load preprocessor state."""
        self.normalizer.load_scalers(path)


if __name__ == "__main__":
    # Test clinical index calculations
    calc = ClinicalIndexCalculator()

    pni = calc.calculate_pni(albumin=4.0, lymphocyte=2000)
    print(f"PNI test: albumin=4.0, lymphocyte=2000 -> PNI={pni:.2f}")

    nlr = calc.calculate_nlr(neutrophil=5.0, lymphocyte=2.0)
    print(f"NLR test: neutrophil=5.0, lymphocyte=2.0 -> NLR={nlr:.2f}")

    car = calc.calculate_car(crp=10.0, albumin=4.0)
    print(f"CAR test: CRP=10.0, albumin=4.0 -> CAR={car:.4f}")

    # Test missing value handler
    standard_vars = ['age', 'bmi', 'dfa_24h', 'pt', 'pni_preop', 'nlr_preop', 'car_preop']
    handler = MissingValuePreprocessor(standard_vars)

    available = ['age', 'bmi', 'dfa_24h', 'pni_preop']
    mask = handler.create_attention_mask(available, max_length=7)
    print(f"\nAttention mask for available vars {available}: {mask}")

    # Test data preprocessor
    preprocessor = DataPreprocessor(
        standard_variables=standard_vars,
        fit_scalers=True
    )

    patient_data = {
        'age': 62.0,
        'bmi': 24.5,
        'dfa_24h': 2984.0,
        'pt': 14.8,
        'albumin_preop': 3.8,
        'lymphocyte_preop': 1800,
        'neutrophil_preop': 6.5,
        'crp_preop': 15.0
    }
    timestamps = {
        'age': 'preop', 'bmi': 'preop', 'dfa_24h': 'postop_24h',
        'pt': 'preop', 'albumin_preop': 'preop', 'lymphocyte_preop': 'preop',
        'neutrophil_preop': 'preop', 'crp_preop': 'preop'
    }

    result = preprocessor.preprocess_patient(patient_data, timestamps, max_length=7)
    print(f"\nPreprocessed patient data:")
    print(f"  Values shape: {result['values'].shape}")
    print(f"  Attention mask: {result['attention_mask']}")
    print(f"  Variables: {result['variable_names']}")

    print("\nPreprocessing test passed!")
