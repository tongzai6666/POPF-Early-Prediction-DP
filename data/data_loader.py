"""
Data Loader Module
==================
Handles loading and batching of multicenter perioperative EHR data.
Supports variable-length sequences due to missing variables across centers.

Paper Section 2.3: "All clinical data were extracted from the electronic medical record 
systems of each center." Missing data were missing at random (MAR) with no observed systematic bias.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import json


class POPFVariable:
    """
    Represents a single clinical variable observation.

    Attributes:
        timestamp: Time stage ("preop", "intraop", "postop_24h")
        variable_name: Clinical variable name (e.g., "DFA", "PT", "PNI")
        value: Numerical or categorical value
        center_id: Hospital center identifier
    """

    def __init__(self, timestamp: str, variable_name: str, value: float, center_id: int = 1):
        self.timestamp = timestamp
        self.variable_name = variable_name
        self.value = value
        self.center_id = center_id

    def to_text(self) -> str:
        """Convert to text format for BERT encoding."""
        return f"{self.timestamp} {self.variable_name} {self.value:.4f}"


class POPFDataset(Dataset):
    """
    Dataset for perioperative EHR data with variable-length sequences.

    Handles:
    - Variable number of recorded variables per patient (due to center heterogeneity)
    - Missing variable handling through attention masking
    - Temporal ordering of variables
    - Label encoding for dual tasks
    """

    # Standard variable list for consistent ordering
    # Based on paper Section 2.3 Data Collection
    STANDARD_VARIABLES = {
        # Baseline
        "age": "baseline",
        "bmi": "baseline",
        "sex": "baseline",
        "hypertension": "baseline",
        "diabetes": "baseline",

        # Preoperative Imaging
        "mpdd": "preop",
        "pt": "preop",
        "ct_pancreas": "preop",
        "ct_spleen": "preop",
        "ct_liver": "preop",
        "ct_psoas": "preop",
        "p_s_ratio": "preop",
        "p_l_ratio": "preop",
        "p_pm_ratio": "preop",

        # Preoperative Lab
        "wbc_preop": "preop",
        "rbc_preop": "preop",
        "hgb_preop": "preop",
        "plt_preop": "preop",
        "albumin_preop": "preop",
        "crp_preop": "preop",
        "pni_preop": "preop",
        "nlr_preop": "preop",
        "car_preop": "preop",

        # Intraoperative
        "surgical_approach": "intraop",
        "operative_time": "intraop",
        "blood_loss": "intraop",
        "transfusion": "intraop",
        "stump_management": "intraop",
        "mpdd_transection": "intraop",
        "pt_transection": "intraop",

        # Postoperative 24h
        "dfa_24h": "postop_24h",
        "wbc_24h": "postop_24h",
        "rbc_24h": "postop_24h",
        "hgb_24h": "postop_24h",
        "plt_24h": "postop_24h",
        "albumin_24h": "postop_24h",
        "crp_24h": "postop_24h",
        "pni_24h": "postop_24h",
        "nlr_24h": "postop_24h",
        "car_24h": "postop_24h",
    }

    def __init__(
        self,
        data_path: str,
        center_id: Optional[int] = None,
        max_variables: int = 35,
        is_training: bool = True
    ):
        """
        Args:
            data_path: Path to CSV/JSON data file
            center_id: If specified, only load data from this center (for external validation)
            max_variables: Maximum number of variables per patient (for padding)
            is_training: Whether this is training set (for data augmentation)
        """
        self.data_path = data_path
        self.center_id = center_id
        self.max_variables = max_variables
        self.is_training = is_training

        self.data = self._load_data()
        self.patient_ids = list(self.data.keys())

    def _load_data(self) -> Dict:
        """
        Load data from file and structure as patient-level records.

        Expected input format (CSV):
        patient_id, center_id, timestamp, variable_name, value, label_popf, label_severity

        Returns:
            Dictionary mapping patient_id -> patient record dict
        """
        df = pd.read_csv(self.data_path)

        # Filter by center if specified
        if self.center_id is not None:
            df = df[df['center_id'] == self.center_id]

        data = {}

        for patient_id in df['patient_id'].unique():
            patient_df = df[df['patient_id'] == patient_id]

            # Extract labels (same for all rows of same patient)
            labels = {
                'label_popf': patient_df['label_popf'].iloc[0],
                'label_severity': patient_df['label_severity'].iloc[0] if 'label_severity' in patient_df.columns else -1
            }

            # Extract variables
            variables = []
            for _, row in patient_df.iterrows():
                if pd.notna(row['value']):  # Skip truly missing values
                    var = POPFVariable(
                        timestamp=row['timestamp'],
                        variable_name=row['variable_name'],
                        value=float(row['value']),
                        center_id=int(row['center_id'])
                    )
                    variables.append(var)

            # Sort by timestamp order: preop -> intraop -> postop_24h
            timestamp_order = {"preop": 0, "intraop": 1, "postop_24h": 2}
            variables.sort(key=lambda v: timestamp_order.get(v.timestamp, 99))

            data[patient_id] = {
                'patient_id': patient_id,
                'center_id': int(patient_df['center_id'].iloc[0]),
                'variables': variables,
                'labels': labels
            }

        return data

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single patient record formatted for model input.

        Returns:
            Dictionary containing:
                - patient_id: str
                - timestamps: list of str
                - variable_names: list of str
                - values: torch.Tensor (max_variables,)
                - attention_mask: torch.Tensor (max_variables,) - 1 for valid, 0 for missing/padded
                - center_id: int
                - label_popf: torch.Tensor (1,)
                - label_severity: torch.Tensor (1,) - -1 if not applicable
        """
        patient_id = self.patient_ids[idx]
        record = self.data[patient_id]

        variables = record['variables']
        num_vars = len(variables)

        # Pad or truncate to max_variables
        timestamps = [""] * self.max_variables
        variable_names = [""] * self.max_variables
        values = torch.zeros(self.max_variables, dtype=torch.float32)
        attention_mask = torch.zeros(self.max_variables, dtype=torch.float32)

        for i in range(min(num_vars, self.max_variables)):
            var = variables[i]
            timestamps[i] = var.timestamp
            variable_names[i] = var.variable_name
            values[i] = var.value
            attention_mask[i] = 1.0

        return {
            'patient_id': patient_id,
            'timestamps': timestamps,
            'variable_names': variable_names,
            'values': values,
            'attention_mask': attention_mask,
            'center_id': record['center_id'],
            'label_popf': torch.tensor([record['labels']['label_popf']], dtype=torch.float32),
            'label_severity': torch.tensor([record['labels']['label_severity']], dtype=torch.float32)
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for DataLoader to batch variable-length sequences.

    Args:
        batch: List of patient records from __getitem__

    Returns:
        Batched dictionary with padded tensors
    """
    batch_size = len(batch)
    max_vars = max(len(b['timestamps']) for b in batch)

    # Initialize batched tensors
    batched_timestamps = []
    batched_variable_names = []
    batched_values = torch.zeros(batch_size, max_vars, dtype=torch.float32)
    batched_attention_mask = torch.zeros(batch_size, max_vars, dtype=torch.float32)
    batched_center_ids = torch.zeros(batch_size, dtype=torch.long)
    batched_label_popf = torch.zeros(batch_size, 1, dtype=torch.float32)
    batched_label_severity = torch.full((batch_size, 1), -1.0, dtype=torch.float32)

    for i, sample in enumerate(batch):
        num_vars = int(sample['attention_mask'].sum().item())

        batched_timestamps.append(sample['timestamps'][:num_vars])
        batched_variable_names.append(sample['variable_names'][:num_vars])
        batched_values[i, :num_vars] = sample['values'][:num_vars]
        batched_attention_mask[i, :num_vars] = 1.0
        batched_center_ids[i] = sample['center_id']
        batched_label_popf[i] = sample['label_popf']
        batched_label_severity[i] = sample['label_severity']

    return {
        'patient_ids': [b['patient_id'] for b in batch],
        'timestamps': batched_timestamps,
        'variable_names': batched_variable_names,
        'values': batched_values,
        'attention_mask': batched_attention_mask,
        'center_ids': batched_center_ids,
        'label_popf': batched_label_popf,
        'label_severity': batched_label_severity
    }


class MulticenterDataLoader:
    """
    High-level data loader manager for multicenter training and validation.

    Handles:
    - Center 1: Training/Internal validation split (7:3)
    - Centers 2-6: External validation (each center separately + pooled)
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        train_val_split: float = 0.7,
        random_seed: int = 42
    ):
        """
        Args:
            data_dir: Directory containing center-specific CSV files
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for data loading
            train_val_split: Ratio for train/validation split (Center 1 only)
            random_seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.random_seed = random_seed

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def get_center_loaders(self, center_id: int) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Get data loaders for a specific center.

        For Center 1: Returns (train_loader, val_loader) with 7:3 split
        For Centers 2-6: Returns (test_loader, None)

        Args:
            center_id: Center identifier (1-6)

        Returns:
            Tuple of (train/test loader, validation loader or None)
        """
        data_path = f"{self.data_dir}/center_{center_id}.csv"

        dataset = POPFDataset(
            data_path=data_path,
            center_id=center_id if center_id > 1 else None,  # Center 1 uses all data then split
            is_training=(center_id == 1)
        )

        if center_id == 1:
            # Split into training and internal validation (7:3)
            total_size = len(dataset)
            train_size = int(total_size * self.train_val_split)
            val_size = total_size - train_size

            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(self.random_seed)
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=collate_fn
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collate_fn
            )

            return train_loader, val_loader

        else:
            # External validation: use all data
            test_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collate_fn
            )

            return test_loader, None

    def get_all_loaders(self) -> Dict[str, DataLoader]:
        """
        Get all data loaders for the complete experiment.

        Returns:
            Dictionary with keys:
                - 'train': Training loader (Center 1)
                - 'val': Internal validation loader (Center 1)
                - 'center_2' to 'center_6': External validation loaders
                - 'external_pooled': Pooled external validation loader
        """
        loaders = {}

        # Center 1: Training and internal validation
        train_loader, val_loader = self.get_center_loaders(1)
        loaders['train'] = train_loader
        loaders['val'] = val_loader

        # Centers 2-6: External validation
        external_datasets = []
        for center_id in range(2, 7):
            test_loader, _ = self.get_center_loaders(center_id)
            loaders[f'center_{center_id}'] = test_loader
            external_datasets.append(test_loader.dataset)

        # Pooled external validation
        if len(external_datasets) > 0:
            pooled_dataset = torch.utils.data.ConcatDataset(external_datasets)
            loaders['external_pooled'] = DataLoader(
                pooled_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collate_fn
            )

        return loaders


def create_synthetic_data(
    output_path: str,
    num_patients: int = 100,
    center_id: int = 1,
    popf_rate: float = 0.4,
    cr_popf_rate: float = 0.45  # Among POPF-positive, proportion of CR-POPF
):
    """
    Create synthetic demonstration data matching the paper's variable structure.

    This is for code demonstration only - does NOT contain real patient data.

    Args:
        output_path: Path to save synthetic CSV
        num_patients: Number of synthetic patients
        center_id: Center identifier
        popf_rate: Rate of POPF occurrence
        cr_popf_rate: Rate of CR-POPF among POPF-positive cases
    """
    np.random.seed(42)

    records = []

    for patient_id in range(num_patients):
        patient_id_str = f"P{center_id}_{patient_id:04d}"

        # Generate labels
        has_popf = np.random.random() < popf_rate
        if has_popf:
            is_cr_popf = np.random.random() < cr_popf_rate
            label_popf = 1
            label_severity = 1 if is_cr_popf else 0
        else:
            label_popf = 0
            label_severity = -1

        # Generate variables with some missingness (simulating multicenter heterogeneity)
        all_vars = list(POPFDataset.STANDARD_VARIABLES.keys())

        # Randomly drop some variables to simulate missingness (10-30% missing)
        available_vars = [v for v in all_vars if np.random.random() > np.random.uniform(0.1, 0.3)]

        for var_name in available_vars:
            timestamp = POPFDataset.STANDARD_VARIABLES[var_name]

            # Generate realistic clinical values based on variable type
            if var_name in ['sex', 'hypertension', 'diabetes', 'transfusion', 'surgical_approach', 'stump_management']:
                value = np.random.randint(0, 2)  # Binary
            elif 'age' in var_name:
                value = np.random.normal(62, 12)
            elif 'bmi' in var_name:
                value = np.random.normal(24, 4)
            elif 'time' in var_name:
                value = np.random.normal(200, 60)
            elif 'dfa' in var_name:
                # DFA higher for POPF patients
                if has_popf:
                    value = np.random.lognormal(8, 0.5)
                else:
                    value = np.random.lognormal(5, 0.5)
            elif 'pt' in var_name:
                value = np.random.normal(15, 5)
            elif 'ct_' in var_name and 'ratio' not in var_name:
                value = np.random.normal(45, 10)
            elif 'pni' in var_name:
                value = np.random.normal(48, 8)
            elif 'nlr' in var_name:
                value = np.random.normal(5, 3)
            elif 'car' in var_name:
                value = np.random.exponential(0.1)
            else:
                value = np.random.normal(0, 1)

            records.append({
                'patient_id': patient_id_str,
                'center_id': center_id,
                'timestamp': timestamp,
                'variable_name': var_name,
                'value': round(value, 4),
                'label_popf': label_popf,
                'label_severity': label_severity
            })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Synthetic data saved to {output_path}: {num_patients} patients, {len(df)} records")


if __name__ == "__main__":
    # Create synthetic demonstration data
    import tempfile
    temp_dir = tempfile.mkdtemp()

    # Create data for all 6 centers with different POPF rates (simulating heterogeneity)
    center_configs = [
        (1, 100, 0.409),  # Center 1: training, POPF rate 40.9% (from paper)
        (2, 50, 0.35),   # Center 2: external, lower rate
        (3, 40, 0.42),   # Center 3
        (4, 80, 0.578),  # Center 4: higher rate (from paper range 30-57.8%)
        (5, 30, 0.30),   # Center 5: lower rate
        (6, 35, 0.45),   # Center 6
    ]

    for center_id, num_patients, popf_rate in center_configs:
        create_synthetic_data(
            f"{temp_dir}/center_{center_id}.csv",
            num_patients=num_patients,
            center_id=center_id,
            popf_rate=popf_rate
        )

    # Test data loader
    loader_manager = MulticenterDataLoader(
        data_dir=temp_dir,
        batch_size=8,
        num_workers=0
    )

    loaders = loader_manager.get_all_loaders()

    print(f"\nData loaders created:")
    for name, loader in loaders.items():
        if loader is not None:
            print(f"  {name}: {len(loader)} batches")

    # Test a single batch
    train_loader = loaders['train']
    batch = next(iter(train_loader))

    print(f"\nSample batch shapes:")
    print(f"  values: {batch['values'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  label_popf: {batch['label_popf'].shape}")
    print(f"  label_severity: {batch['label_severity'].shape}")
    print(f"  center_ids: {batch['center_ids'].shape}")

    print("\nData Loader test passed!")
