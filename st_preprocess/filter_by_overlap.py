"""Filter both ST HVG and SC data to the gene overlap produced by compute_gene_overlap.py."""
import os
import logging
from pathlib import Path

import numpy as np
import scanpy as sc
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

BASE_DIR  = Path(os.environ.get("PREPROCESS_ROOT", "./data/preprocess"))
OVERLAP_DIR = BASE_DIR / "gene_overlap"
HVG_DIR    = BASE_DIR / "hvg_expr_npy"
SC_DIR     = BASE_DIR / "sc_data_gene_mapped"
HVG_OUT    = BASE_DIR / "filter_hvg_expression"
SC_OUT     = BASE_DIR / "filter_sc_expression"


def filter_hvg_data(input_dir: Path, output_dir: Path,
                    overlap_genes: np.ndarray, tissue: str):
    """Filter HVG .npy files to overlap genes (preserves overlap gene order)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_files = sorted(input_dir.glob("*.npy"))
    n_ok = n_skip = 0
    for npy_file in tqdm(npy_files, desc=f"{tissue} HVG"):
        data = np.load(npy_file, allow_pickle=True).item()
        gene_names = data["gene_names"]
        expression = data["expression"]
        gene_indices, kept = [], []
        for g in overlap_genes:
            idx = np.where(gene_names == g)[0]
            if len(idx) > 0:
                gene_indices.append(idx[0])
                kept.append(g)
        if not gene_indices:
            n_skip += 1
            continue
        np.save(output_dir / npy_file.name,
                {"gene_names": np.array(kept),
                 "expression": expression[:, gene_indices]})
        n_ok += 1
    logging.info(f"{tissue} HVG: ok={n_ok}, skipped={n_skip}")


def filter_sc_data(input_dir: Path, output_dir: Path,
                   overlap_genes: np.ndarray, tissue: str):
    """Filter SC .h5ad files to overlap genes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    h5ad_files = sorted(input_dir.glob("*.h5ad"))
    n_ok = n_skip = 0
    for h5ad_file in tqdm(h5ad_files, desc=f"{tissue} SC"):
        adata = sc.read_h5ad(h5ad_file)
        in_data = adata.var_names.tolist()
        keep = [g for g in overlap_genes if g in in_data]
        if not keep:
            n_skip += 1
            continue
        adata[:, keep].copy().write_h5ad(output_dir / h5ad_file.name)
        n_ok += 1
    logging.info(f"{tissue} SC: ok={n_ok}, skipped={n_skip}")


# Each entry: (tissue, overlap_filename, hvg_subdir, sc_subdir, sc_out_subdir).
DATASETS = [
    ("HER2", "HER2_hvg_in_sc.npy", "HER2", "her2_sc_data", "her2_sc_filtered"),
]


def main():
    for tissue, ofile, hvg_sub, sc_sub, sc_out_sub in DATASETS:
        opath = OVERLAP_DIR / ofile
        if not opath.exists():
            logging.warning(f"missing overlap file: {opath}")
            continue
        overlap_genes = np.load(opath, allow_pickle=True)
        logging.info(f"{tissue}: {len(overlap_genes)} overlap genes")
        filter_hvg_data(HVG_DIR / hvg_sub,  HVG_OUT / hvg_sub,
                        overlap_genes, tissue)
        filter_sc_data(SC_DIR / sc_sub,    SC_OUT / sc_out_sub,
                       overlap_genes, tissue)


if __name__ == "__main__":
    main()
