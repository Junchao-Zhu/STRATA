"""
Configuration for the Triple-Branch 3D gene expression predictor.
"""
import os
import numpy as np

# Data roots (override via environment variables or edit before running).
DATA_ROOT = os.environ.get("DATA_ROOT", "./data/data_infor")
IMAGE_ROOT = os.environ.get("IMAGE_ROOT", "./data/patches")
GRAPH_ROOT = os.environ.get("GRAPH_ROOT", "./data/3d_map")
CONCH_ROOT = os.path.join(DATA_ROOT, "conch_features")

# Dataset configuration (HER2 example).
DATASET_CONFIG = {
    "HER2": {
        "n_genes": 967,
        "n_cell_types": 67,
    },
}

MASK_SEED = 42


def get_fixed_target_genes(dataset_name, mask_ratio):
    n_genes = DATASET_CONFIG[dataset_name]["n_genes"]
    if mask_ratio >= 1:
        n_target = int(mask_ratio)
    else:
        n_target = int(n_genes * mask_ratio)
    n_target = min(n_target, n_genes)
    target_indices = np.arange(n_target)
    input_indices = np.arange(n_target, n_genes)
    return target_indices, input_indices


def get_model_dims(dataset_name, mask_ratio):
    target_indices, input_indices = get_fixed_target_genes(dataset_name, mask_ratio)
    return len(input_indices), len(target_indices)
