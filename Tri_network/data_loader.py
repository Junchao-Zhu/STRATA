"""
Data loader for 3D gene expression prediction.

Supports two modes:
1. Standalone training: all training spots' expressions are available.
2. Active sampling: only sampled spots have revealed expression, controlled
   by a `sampled_set` of pool indices.
"""
import os
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import get_fixed_target_genes, IMAGE_ROOT, DATA_ROOT, GRAPH_ROOT, CONCH_ROOT, DATASET_CONFIG
from cross_fold import parse_file_path, get_case_slices


# 3D neighbor precomputation.

def build_neighbor_lookup(dataset_name, train_files, max_k=25):
    """Build neighbor and query-feature lookups for 3D interpolation."""
    case_slices_all = get_case_slices(dataset_name)

    # Restrict neighbor source to training slices.
    train_slice_set = set()
    for fpath in train_files:
        case_id, slice_name = parse_file_path(fpath)
        train_slice_set.add((case_id, slice_name))

    # pool_index_map follows the order of train_files.
    pool_index_map = {}
    pool_idx = 0
    for fpath in train_files:
        case_id, slice_name = parse_file_path(fpath)
        npy_path = os.path.join(DATA_ROOT, dataset_name, fpath)
        if os.path.exists(npy_path):
            data = np.load(npy_path, allow_pickle=True)
            for local_idx in range(len(data)):
                pool_index_map[(case_id, slice_name, local_idx)] = pool_idx
                pool_idx += 1

    neighbor_data = {}
    query_conch_map = {}

    for case_id, slices in case_slices_all.items():
        graph_path = os.path.join(GRAPH_ROOT, dataset_name, f'{case_id}.npz')
        if not os.path.exists(graph_path):
            continue

        graph = np.load(graph_path)
        slice_names = list(graph['slice_names'])
        slice_offsets = list(graph['slice_offsets'])
        cross_indices = graph['cross_indices']
        cross_weights = graph['cross_weights']

        # Number of spots per slice.
        n_spots_per_slice = []
        for s_idx in range(len(slice_names)):
            if s_idx + 1 < len(slice_offsets):
                n_spots_per_slice.append(int(slice_offsets[s_idx + 1]) - int(slice_offsets[s_idx]))
            else:
                n_spots_per_slice.append(cross_indices.shape[0] - int(slice_offsets[s_idx]))

        # Load CONCH features and expression for this case.
        slice_data = {}
        slice_conch = {}
        for s_name in slice_names:
            npy_path = os.path.join(DATA_ROOT, dataset_name, case_id, f'{s_name}.npy')
            if os.path.exists(npy_path):
                slice_data[s_name] = np.load(npy_path, allow_pickle=True)
            conch_path = os.path.join(CONCH_ROOT, dataset_name, case_id, f'{s_name}_conch.npy')
            if os.path.exists(conch_path):
                slice_conch[s_name] = np.load(conch_path)

        # In-patient global to (slice_name, local_idx) mapping.
        global_to_local = {}
        for s_idx, s_name in enumerate(slice_names):
            offset = int(slice_offsets[s_idx])
            for li in range(n_spots_per_slice[s_idx]):
                global_to_local[offset + li] = (s_name, li)

        # query_conch_map covers train and val spots (val needs it at inference).
        for s_name, conch_mat in slice_conch.items():
            for li in range(conch_mat.shape[0]):
                query_conch_map[(case_id, s_name, li)] = conch_mat[li].astype(np.float32)

        # Build neighbor entries for every spot, restricted to training slices.
        for s_idx, s_name in enumerate(slice_names):
            offset = int(slice_offsets[s_idx])
            for local_idx in range(n_spots_per_slice[s_idx]):
                global_idx = offset + local_idx

                nbr_global  = cross_indices[global_idx]
                nbr_weights = cross_weights[global_idx]

                valid_expr, valid_vis, valid_w, valid_pool_idx = [], [], [], []
                for nbr_g, nbr_w in zip(nbr_global, nbr_weights):
                    if nbr_g < 0 or nbr_w <= 0:
                        continue
                    if nbr_g not in global_to_local:
                        continue
                    nbr_slice, nbr_local = global_to_local[nbr_g]

                    # Only training-slice neighbors have measured expression.
                    if (case_id, nbr_slice) not in train_slice_set:
                        continue
                    if nbr_slice not in slice_data or nbr_slice not in slice_conch:
                        continue

                    nbr_key = (case_id, nbr_slice, nbr_local)
                    if nbr_key not in pool_index_map:
                        continue

                    nbr_spot = slice_data[nbr_slice][nbr_local]
                    valid_expr.append(nbr_spot['expression_norm'])
                    valid_vis.append(slice_conch[nbr_slice][nbr_local])
                    valid_w.append(float(nbr_w))
                    valid_pool_idx.append(pool_index_map[nbr_key])

                if valid_expr:
                    neighbor_data[(case_id, s_name, local_idx)] = {
                        'neighbor_expr':     np.stack(valid_expr).astype(np.float32),
                        'neighbor_vis':      np.stack(valid_vis).astype(np.float32),
                        'neighbor_weights':  np.array(valid_w,       dtype=np.float32),
                        'neighbor_pool_idx': np.array(valid_pool_idx, dtype=np.int64),
                    }

    return neighbor_data, pool_index_map, query_conch_map


# Dataset.

class GeneExpressionDataset3D(Dataset):
    """Dataset for 3D gene expression prediction."""

    def __init__(self, samples, case_slice_info, dataset_name, mask_ratio,
                 neighbor_data, query_conch_map,
                 transform=None, max_k=25, sampled_set=None):
        self.samples = samples
        self.case_slice_info = case_slice_info
        self.dataset_name = dataset_name
        self.transform = transform
        self.max_k = max_k
        self.neighbor_data = neighbor_data
        self.query_conch_map = query_conch_map
        self.sampled_set = sampled_set

        self.target_indices, _ = get_fixed_target_genes(dataset_name, mask_ratio)
        self.n_target_genes = len(self.target_indices)

    def set_sampled(self, sampled_set):
        """Update which pool indices have been sampled."""
        self.sampled_set = sampled_set

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        case_id, slice_name, local_idx = self.case_slice_info[idx]

        image_path = os.path.join(IMAGE_ROOT, sample['sample_id'], f"{sample['patch_id']}.png")
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        target_genes = torch.tensor(
            sample['expression_norm'][self.target_indices], dtype=torch.float32
        )

        key = (case_id, slice_name, local_idx)
        if key in self.query_conch_map:
            query_conch = torch.tensor(self.query_conch_map[key], dtype=torch.float32)
        else:
            query_conch = torch.zeros(512, dtype=torch.float32)

        nbr = self.neighbor_data.get(key)

        neighbor_vis     = torch.zeros(self.max_k, 512,               dtype=torch.float32)
        neighbor_expr    = torch.zeros(self.max_k, self.n_target_genes, dtype=torch.float32)
        neighbor_mask    = torch.zeros(self.max_k,                    dtype=torch.bool)
        neighbor_weights = torch.zeros(self.max_k,                    dtype=torch.float32)

        if nbr is not None:
            total_n = len(nbr['neighbor_vis'])
            pool_indices = nbr['neighbor_pool_idx']

            j = 0
            for i in range(total_n):
                if self.sampled_set is not None:
                    if int(pool_indices[i]) not in self.sampled_set:
                        continue
                neighbor_vis[j]     = torch.from_numpy(nbr['neighbor_vis'][i])
                neighbor_expr[j]    = torch.from_numpy(
                    nbr['neighbor_expr'][i][self.target_indices]
                )
                neighbor_mask[j]    = True
                neighbor_weights[j] = float(nbr['neighbor_weights'][i])
                j += 1
                if j >= self.max_k:
                    break

        return {
            'image':             image,
            'target_genes':      target_genes,
            'query_conch':       query_conch,
            'neighbor_vis':      neighbor_vis,
            'neighbor_expr':     neighbor_expr,
            'neighbor_mask':     neighbor_mask,
            'neighbor_weights':  neighbor_weights,
            'sample_id':         sample['sample_id'],
            'patch_id':          sample['patch_id'],
            'case_id':           case_id,
        }


# Transforms.

def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


# Helpers.

def load_samples_with_info(npy_files, dataset_name):
    samples, case_slice_info = [], []
    for fpath in npy_files:
        case_id, slice_name = parse_file_path(fpath)
        full_path = os.path.join(DATA_ROOT, dataset_name, fpath)
        data = np.load(full_path, allow_pickle=True)
        for local_idx, spot in enumerate(data):
            samples.append(spot)
            case_slice_info.append((case_id, slice_name, local_idx))
    return samples, case_slice_info


def create_dataloaders(train_npy_files, val_npy_files, dataset_name,
                       mask_ratio, batch_size=32, num_workers=8):
    """Create data loaders with 3D neighbor info (standalone mode)."""
    print(f"[DataLoader] Building 3D neighbor lookup for {dataset_name}...")
    neighbor_data, pool_index_map, query_conch_map = build_neighbor_lookup(
        dataset_name, train_npy_files
    )

    n_with = sum(1 for v in neighbor_data.values() if v is not None)
    if n_with > 0:
        avg_k = np.mean([len(v['neighbor_vis']) for v in neighbor_data.values()])
        print(f"  Spots with 3D neighbors: {n_with}, avg k: {avg_k:.1f}")
    print(f"  query_conch entries: {len(query_conch_map)}")

    train_samples, train_info = load_samples_with_info(train_npy_files, dataset_name)
    train_dataset = GeneExpressionDataset3D(
        samples=train_samples, case_slice_info=train_info,
        dataset_name=dataset_name, mask_ratio=mask_ratio,
        neighbor_data=neighbor_data, query_conch_map=query_conch_map,
        transform=get_transforms(is_train=True),
        sampled_set=None,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )

    val_loaders_by_slide = {}
    for fpath in val_npy_files:
        case_id, slice_name = parse_file_path(fpath)
        slide_key = f"{case_id}/{slice_name}"
        samples, info = load_samples_with_info([fpath], dataset_name)
        val_dataset = GeneExpressionDataset3D(
            samples=samples, case_slice_info=info,
            dataset_name=dataset_name, mask_ratio=mask_ratio,
            neighbor_data=neighbor_data, query_conch_map=query_conch_map,
            transform=get_transforms(is_train=False),
            sampled_set=None,
        )
        val_loaders_by_slide[slide_key] = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

    print(f"  Training samples: {len(train_samples)}")
    print(f"  Validation slides: {len(val_loaders_by_slide)}")
    print(f"  Target genes: {train_dataset.n_target_genes}")

    return (train_loader, val_loaders_by_slide, train_samples,
            neighbor_data, pool_index_map, query_conch_map)
