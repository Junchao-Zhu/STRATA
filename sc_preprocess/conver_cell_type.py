"""Merge per-file SC embeddings and assign integer ids to cell types."""
import os
from pathlib import Path
import numpy as np
import pandas as pd

data_root = Path(os.environ.get("SC_EMBED_ROOT", "./data/sc_embeddings"))
save_dir = Path(os.environ.get("SC_OUT_ROOT", "./data/sc_embedding_out"))
save_dir.mkdir(parents=True, exist_ok=True)

save_data_path = save_dir / "merged_data.npy"
save_map_path = save_dir / "celltype_map.npy"
save_map_csv = save_dir / "celltype_map.csv"

npy_files = list(data_root.glob("*.npy"))
print(f"Found {len(npy_files)} files")

merged = []
for f in npy_files:
    try:
        arr = np.load(f, allow_pickle=True)
        merged.extend(arr.tolist())
        print(f"  Loaded {f.name} ({len(arr)} rows)")
    except Exception as e:
        print(f"  Error reading {f}: {e}")

print(f"Total rows = {len(merged)}")

# Map cell_type strings to integer ids.
all_types = [d["cell_type"] for d in merged]
unique_types = sorted(set(all_types))
type2id = {t: idx for idx, t in enumerate(unique_types)}
print(f"Unique cell types = {len(unique_types)}")

for d in merged:
    d["cell_type_id"] = type2id[d["cell_type"]]

np.save(save_map_path, type2id)
pd.DataFrame(list(type2id.items()), columns=["cell_type", "cell_type_id"]).to_csv(save_map_csv, index=False)
print(f"Saved mapping (npy) -> {save_map_path}")
print(f"Saved mapping (csv) -> {save_map_csv}")

merged_arr = np.array(merged, dtype=object)
np.save(save_data_path, merged_arr)
print(f"Saved merged data -> {save_data_path}")
