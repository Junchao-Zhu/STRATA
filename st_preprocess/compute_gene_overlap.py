"""Compute the gene overlap between an ST HVG list and an SC h5ad gene set."""
import os
import numpy as np
import scanpy as sc
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def get_hvg_genes(hvg_dir):
    npy_files = list(hvg_dir.glob("*.npy"))
    if not npy_files:
        return None
    
    data = np.load(npy_files[0], allow_pickle=True).item()
    return set(data['gene_names'])


def get_sc_genes(sc_dir):
    h5ad_files = list(sc_dir.glob("*.h5ad"))
    if not h5ad_files:
        return None
    
    all_genes = None
    for f in tqdm(h5ad_files, desc="Reading SC files", unit="file"):
        adata = sc.read_h5ad(f)
        genes = set(adata.var_names)
        if all_genes is None:
            all_genes = genes
        else:
            all_genes &= genes
    
    return all_genes


def main():
    BASE_DIR = Path(os.environ.get("PREPROCESS_ROOT", "./data/preprocess"))
    HVG_DIR = BASE_DIR / "hvg_expr_npy"
    SC_DIR = BASE_DIR / "sc_data_gene_mapped"
    OUTPUT_DIR = BASE_DIR / "gene_overlap"
    
    print("="*60)
    print("Compute Gene Overlap (ST HVG ∩ SC genes)")
    print(f"HVG dir: {HVG_DIR}")
    print(f"SC dir:  {SC_DIR}")
    print(f"Output:  {OUTPUT_DIR}")
    print("="*60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    datasets = [
        ("HER2", "her2_sc_data", "HER2_hvg_in_sc.npy"),
    ]
    
    results = []
    
    for hvg_subdir, sc_subdir, output_name in datasets:
        print(f"\n{'-'*60}")
        print(f"Processing: {hvg_subdir} vs {sc_subdir}")
        print("-"*60)
        
        hvg_path = HVG_DIR / hvg_subdir
        sc_path = SC_DIR / sc_subdir
        
        if not hvg_path.exists():
            print(f"[WARN] HVG directory not found: {hvg_path}")
            continue
        if not sc_path.exists():
            print(f"[WARN] SC directory not found: {sc_path}")
            continue
        
        print("Loading HVG genes...")
        hvg_genes = get_hvg_genes(hvg_path)
        if hvg_genes is None:
            print(f"[WARN] No HVG files found in {hvg_path}")
            continue
        print(f"  HVG genes: {len(hvg_genes)}")
        
        print("Loading SC genes...")
        sc_genes = get_sc_genes(sc_path)
        if sc_genes is None:
            print(f"[WARN] No SC files found in {sc_path}")
            continue
        print(f"  SC genes: {len(sc_genes)}")
        
        overlap_genes = hvg_genes & sc_genes
        overlap_genes = sorted(list(overlap_genes))
        
        print(f"\n  Overlap: {len(overlap_genes)} genes")
        print(f"  Coverage: {len(overlap_genes)/len(hvg_genes)*100:.1f}% of HVG")
        
        output_path = OUTPUT_DIR / output_name
        np.save(output_path, np.array(overlap_genes))
        print(f"  Saved: {output_path}")
        
        results.append({
            'dataset': hvg_subdir,
            'hvg_genes': len(hvg_genes),
            'sc_genes': len(sc_genes),
            'overlap': len(overlap_genes),
            'coverage': len(overlap_genes)/len(hvg_genes)*100
        })
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Dataset':<12} {'HVG':>8} {'SC':>10} {'Overlap':>10} {'Coverage':>10}")
    print("-"*60)
    for r in results:
        print(f"{r['dataset']:<12} {r['hvg_genes']:>8} {r['sc_genes']:>10} {r['overlap']:>10} {r['coverage']:>9.1f}%")
    
    print("\n" + "="*60)
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
