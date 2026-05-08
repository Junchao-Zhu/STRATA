"""
Extract top-1000 HVG expression from h5ad files.
Pipeline: normalize_total -> log1p -> highly_variable_genes (n=1000)
HVGs are selected jointly across all slides per dataset.
Output: hvg_expression/{dataset}/{slide_id}.npy
  each file: dict with 'gene_names' (1000,) and 'expression' (N_spots, 1000)
"""
import os
import numpy as np
import scanpy as sc
from tqdm import tqdm

ST_DIR = os.environ.get("ST_H5AD_ROOT", "./data/st_h5ad")
OUTPUT_BASE = os.environ.get("HVG_EXPR_ROOT", "./data/hvg_expression")
N_HVG = 1000

DATASETS = {
    'HER2': [f'SPA{i}' for i in range(119, 155)],
}


def load_and_preprocess(slide_ids):
    """Load slides, normalize, log1p, return concatenated adata."""
    adatas = []
    valid_ids = []
    for sid in tqdm(slide_ids, desc='  loading'):
        path = os.path.join(ST_DIR, f'{sid}.h5ad')
        if not os.path.exists(path):
            print(f'  [SKIP] {sid} not found')
            continue
        adata = sc.read_h5ad(path)
        adata.obs['_slide_id'] = sid
        adatas.append(adata)
        valid_ids.append(sid)

    combined = sc.concat(adatas, label='_slide_id', keys=valid_ids)
    sc.pp.normalize_total(combined, target_sum=1e4)
    sc.pp.log1p(combined)
    return combined, valid_ids


def select_hvg(combined, n_top=1000):
    """Select HVGs on combined data."""
    sc.pp.highly_variable_genes(combined, n_top_genes=n_top, flavor='seurat')
    hvg_names = combined.var_names[combined.var['highly_variable']].tolist()

    disp = combined.var.loc[combined.var['highly_variable'], 'dispersions_norm']
    hvg_names = disp.sort_values(ascending=False).index.tolist()
    print(f'  Selected {len(hvg_names)} HVGs')
    return hvg_names


def save_per_slide(combined, hvg_names, valid_ids, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    gene_idx = [combined.var_names.get_loc(g) for g in hvg_names]
    X = combined.X.toarray() if hasattr(combined.X, 'toarray') else combined.X

    for sid in tqdm(valid_ids, desc='  saving'):
        mask = combined.obs['_slide_id'] == sid
        expr = X[mask][:, gene_idx].astype(np.float32)
        out = {'gene_names': np.array(hvg_names), 'expression': expr}
        np.save(os.path.join(out_dir, f'{sid}.npy'), out)
        tqdm.write(f'  {sid}: {expr.shape}')


if __name__ == '__main__':
    for dataset, slide_ids in DATASETS.items():
        print(f'\n{"="*60}')
        print(f'Dataset: {dataset} ({len(slide_ids)} slides)')
        print('='*60)

        out_dir = os.path.join(OUTPUT_BASE, dataset)

        combined, valid_ids = load_and_preprocess(slide_ids)
        print(f'  Combined shape: {combined.shape}')

        hvg_names = select_hvg(combined, n_top=N_HVG)

        # Save HVG gene list
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, 'hvg_genes.npy'), np.array(hvg_names))
        print(f'  Saved gene list: {out_dir}/hvg_genes.npy')

        save_per_slide(combined, hvg_names, valid_ids, out_dir)

    print('\nDone.')
