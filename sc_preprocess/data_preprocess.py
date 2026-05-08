"""Convert raw SC h5ad gene identifiers to official human gene symbols.

Strips Ensembl version suffix, queries MyGene to map Ensembl -> Symbol, falls
back to a valid original symbol if mapping fails, and drops the rest.
"""
import os, sys, warnings, re
from pathlib import Path
import pandas as pd
import scanpy as sc
import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, "../")

import mygene
mg = mygene.MyGeneInfo()


def is_valid_symbol(x):
    """A valid human gene symbol starts with a letter."""
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9\-]+$", str(x)))


SRC_DIR = Path(os.environ.get("SC_SRC_DIR", "./data/sc_data"))
DST_DIR = Path(os.environ.get("SC_DST_DIR", "./data/sc_data_updated"))
DST_DIR.mkdir(exist_ok=True, parents=True)


print(f"Converting SC gene names from {SRC_DIR} to {DST_DIR}")

for file in sorted(SRC_DIR.glob("*.h5ad")):
    print(f"\nLoading {file.name} ...")
    adata = sc.read_h5ad(str(file))

    # 1. Strip Ensembl version postfix.
    var = adata.var_names.to_series().astype(str)
    stripped = var.str.replace(r"\.\d+$", "", regex=True)

    # 2. Force map all genes using MyGene.
    print(f"Mapping {len(stripped)} Ensembl IDs -> official symbols ...")
    results = []
    chunk = 1000
    ids_to_query = stripped.tolist()

    for i in tqdm.tqdm(range(0, len(ids_to_query), chunk)):
        res = mg.querymany(
            ids_to_query[i:i+chunk],
            scopes="ensembl.gene",
            fields="symbol",
            species="human",
            as_dataframe=True,
            verbose=False,
        )
        results.append(res)

    df_map = pd.concat(results).reset_index().rename(
        columns={"query": "ensembl_stripped", "symbol": "mapped_symbol"},
    )
    df_map = df_map.drop_duplicates(subset="ensembl_stripped")

    # 3. Merge mapping back.
    var_df = pd.DataFrame({"ensembl_stripped": stripped})
    var_df = var_df.merge(
        df_map[["ensembl_stripped", "mapped_symbol"]],
        on="ensembl_stripped", how="left"
    )

    # 4. Keep valid original symbol if mapping fails.
    var_df["gene_symbol"] = var_df["mapped_symbol"]
    orig_ok = var_df["mapped_symbol"].isna() & var.apply(is_valid_symbol)
    var_df.loc[orig_ok, "gene_symbol"] = var[orig_ok]

    # 5. Drop unmapped genes.
    before = adata.n_vars
    keep_mask = var_df["gene_symbol"].notna()
    adata = adata[:, keep_mask].copy()
    after = adata.n_vars

    adata.var["gene_symbol"] = var_df.loc[keep_mask, "gene_symbol"].values
    print(f"{file.name}: kept {after} genes (removed {before-after})")

    out_path = DST_DIR / file.name
    adata.write_h5ad(out_path)
    print(f"Saved cleaned -> {out_path}")

print("Done.")
