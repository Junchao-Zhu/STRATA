"""Run scGPT embedding extraction over a directory of h5ad files (HER2 example)."""
import sys
from pathlib import Path
import os
import numpy as np
import scanpy as sc
import warnings
import scipy.sparse as sp

sys.path.insert(0, "../")
import scgpt as scg

warnings.filterwarnings("ignore", category=ResourceWarning)


def fill_zero_rows(adata):
    """Replace all-zero expression rows with the previous non-zero row."""
    X = adata.X
    if sp.issparse(X):
        row_sum = X.sum(axis=1).A1
    else:
        row_sum = X.sum(axis=1)
    if np.all(row_sum == 0):
        return None
    nonzero_idx = np.where(row_sum != 0)[0]
    zero_idx = np.where(row_sum == 0)[0]
    for idx in zero_idx:
        rep_idx = nonzero_idx[0] if idx == 0 else idx - 1
        X[idx] = X[rep_idx]
    adata.X = X
    return adata


model_dir = Path(os.environ.get("SCGPT_CKPT", "./checkpoints/scgpt"))
datasets = ['HER2']

for data in datasets:
    data_path = os.path.join(os.environ.get("ST_H5AD_ROOT", "./data/st_h5ad"), data)
    if not os.path.exists(data_path):
        print(f"[Skip] Not found: {data_path}")
        continue

    for temp_file in os.listdir(data_path):
        file_path = os.path.join(data_path, temp_file)
        print(f"\n===== Processing: {file_path} =====")

        try:
            adata = sc.read_h5ad(file_path)
        except Exception as e:
            print(f"[Error] Failed to read {file_path}: {e}")
            continue

        try:
            adata = fill_zero_rows(adata)
            if adata is None:
                print(f"[Skip] All-zero expression in {file_path}")
                continue
        except Exception as e:
            print(f"[Error] Failed to fix zero rows {file_path}: {e}")
            continue

        try:
            ref_embed_adata = scg.tasks.embed_data(
                adata,
                model_dir,
                gene_col="gene_symbol",
                obs_to_save=None,
                batch_size=64,
                return_new_adata=True,
            )
            ex_embed = ref_embed_adata.X
            print(f"[OK] Embedding shape: {ex_embed.shape}")
        except Exception as e:
            print(f"[Error] Failed during embedding {file_path}: {e}")
            continue

        try:
            save_dir = os.path.join(os.environ.get("ST_EMBED_ROOT", "./data/st_embeddings"), data)
            os.makedirs(save_dir, exist_ok=True)
            save_path = f"{save_dir}/{temp_file}.npy"
            np.save(save_path, ex_embed)
            print(f"[Saved] {save_path}")
        except Exception as e:
            print(f"[Error] Failed to save embedding {file_path}: {e}")
            continue
