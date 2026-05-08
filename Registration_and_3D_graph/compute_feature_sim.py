"""Compute pairwise cosine similarity between spot features across slices."""
import os
import glob
import re
import numpy as np
from itertools import combinations
from tqdm import tqdm

FEATURES_ROOT = os.environ.get("FEATURES_ROOT", "./data/conch_features")
OUTPUT_ROOT   = os.environ.get("SIM_ROOT", "./data/feature_sim")
DATASETS      = ['HER2']


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def cosine_sim(a, b):
    """a: (N, D), b: (M, D) -> (N, M) cosine similarity"""
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return (a @ b.T).astype(np.float32)


if __name__ == '__main__':
    for dataset in DATASETS:
        feat_dataset_dir = os.path.join(FEATURES_ROOT, dataset)
        cases = sorted(os.listdir(feat_dataset_dir))

        for case in tqdm(cases, desc=dataset):
            out_path = os.path.join(OUTPUT_ROOT, dataset, case + '.npz')
            if os.path.exists(out_path):
                continue

            case_dir = os.path.join(feat_dataset_dir, case)
            npy_files = sorted(glob.glob(os.path.join(case_dir, '*.npy')),
                               key=natural_keys)
            slice_names = [os.path.basename(f).replace('.npy', '') for f in npy_files]

            result = {}
            for i, j in combinations(range(len(slice_names)), 2):
                name_i = slice_names[i]
                name_j = slice_names[j]
                feat_i = np.load(npy_files[i]).astype(np.float32)
                feat_j = np.load(npy_files[j]).astype(np.float32)
                result[f'{name_i}_{name_j}'] = cosine_sim(feat_i, feat_j)

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.savez(out_path, **result)

    print('Done.')
