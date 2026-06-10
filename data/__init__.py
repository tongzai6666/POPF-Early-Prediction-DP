"""
Data Processing Modules
========================
Handles loading, preprocessing, and batching of multicenter perioperative EHR data.

Modules:
    data_loader: Multi-center dataset loading with variable-length sequences
    preprocessing: Feature normalization, composite index calculation, missing value handling
"""

from .data_loader import (
    POPFVariable,
    POPFDataset,
    collate_fn,
    MulticenterDataLoader,
    create_synthetic_data
)
from .preprocessing import (
    ClinicalIndexCalculator,
    FeatureNormalizer,
    MissingValuePreprocessor,
    DataPreprocessor
)

__all__ = [
    'POPFVariable',
    'POPFDataset',
    'collate_fn',
    'MulticenterDataLoader',
    'create_synthetic_data',
    'ClinicalIndexCalculator',
    'FeatureNormalizer',
    'MissingValuePreprocessor',
    'DataPreprocessor'
]
