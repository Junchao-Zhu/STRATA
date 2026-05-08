"""scGPT embedding for SC data, using obs['cell_type'] as label."""
import os, sys, warnings
import numpy as np
from pathlib import Path
import scanpy as sc
import scipy.sparse as sp

sys.path.insert(0, os.environ.get("SCGPT_REPO", "./data/scgpt"))
import scgpt as scg

warnings.filterwarnings("ignore")


def fill_zero_rows(adata):
    X = adata.X
    row_sum = X.sum(axis=1).A1 if sp.issparse(X) else X.sum(axis=1)
    if np.all(row_sum == 0):
        return None
    nz = np.where(row_sum != 0)[0]
    for idx in np.where(row_sum == 0)[0]:
        rep = nz[0] if idx == 0 else idx - 1
        X[idx] = X[rep]
    adata.X = X
    return adata


model_dir = Path(os.environ.get("SCGPT_CKPT", "./checkpoints/scgpt"))
CELL_TYPE_COL = "cell_type"

datasets = [os.environ.get("SC_DATASET_NAME", "her2_sc_data")]

for dataset_name in datasets:
    data_path = os.path.join(os.environ.get("SC_DATA_ROOT", "./data/sc_data_updated"), dataset_name)
    save_dir = os.path.join(os.environ.get("SC_EMBED_ROOT", "./data/sc_embeddings"), dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Processing {dataset_name}")
    for fname in sorted(os.listdir(data_path)):
        if not fname.endswith(".h5ad"):
            continue
        file_path = f"{data_path}/{fname}"
        print(f"=== {fname} ===")

        adata = sc.read_h5ad(file_path)
        adata = fill_zero_rows(adata)
        if adata is None:
            print(f"Skip {fname}: all zero.")
            continue

        if CELL_TYPE_COL not in adata.obs.columns:
            print(f"Skip {fname}: no obs['{CELL_TYPE_COL}']")
            continue
        raw_types = adata.obs[CELL_TYPE_COL].astype(str).values
        print(f"Cells: {len(raw_types)}, unique types: {len(set(raw_types))}")

        emb_adata = scg.tasks.embed_data(
            adata, model_dir,
            gene_col="gene_symbol",
            obs_to_save=None,
            batch_size=128,
            return_new_adata=True,
        )
        embeddings = emb_adata.X
        print(f"Embedding shape: {embeddings.shape}")

        out_list = [
            {"embedding": embeddings[i].astype(np.float32), "cell_type": raw_types[i]}
            for i in range(len(raw_types))
        ]
        out_arr = np.array(out_list, dtype=object)

        save_path = f"{save_dir}/{fname.replace('.h5ad', '')}.npy"
        np.save(save_path, out_arr)
        print(f"Saved: {save_path}  ({len(out_arr)} cells)")
