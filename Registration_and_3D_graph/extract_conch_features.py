"""Extract CONCH (ViT-B/16) features per spot for HER2 patches."""

import os
import sys
import glob
import re
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from conch.open_clip_custom import create_model_from_pretrained

CONCH_WEIGHTS = os.environ.get("CONCH_CKPT", "./checkpoints/conch_vit_b_16.bin")
PATCH_ROOT    = os.environ.get("PATCH_ROOT", "./data/patches")
COORDS_ROOT   = os.environ.get("SPOT_COORDS_ROOT", "./data/spot_coords")
OUTPUT_ROOT   = os.environ.get("CONCH_FEAT_ROOT", "./data/conch_features")

BATCH_SIZE  = 256
NUM_WORKERS = 4
DATASETS    = ['HER2']


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


class PatchDataset(Dataset):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def __init__(self, slice_id, n_spots):
        self.patch_dir = os.path.join(PATCH_ROOT, slice_id)
        self.n_spots = n_spots

    def __len__(self):
        return self.n_spots

    def __getitem__(self, idx):
        path = os.path.join(self.patch_dir, f'patch_{idx}.png')
        return self.transform(Image.open(path).convert('RGB'))


def load_conch(device):
    model = create_model_from_pretrained(
        'conch_ViT-B-16',
        checkpoint_path=CONCH_WEIGHTS,
        force_image_size=224,
        return_transform=False,
    )
    trunk = model.visual.trunk

    class ConchEncoder(torch.nn.Module):
        def __init__(self, trunk):
            super().__init__()
            self.patch_embed = trunk.patch_embed
            self.cls_token   = trunk.cls_token
            self.pos_embed   = trunk.pos_embed
            self.pos_drop    = trunk.pos_drop
            self.blocks      = trunk.blocks
            self.norm        = trunk.norm

        def forward(self, x):
            B = x.shape[0]
            x = self.patch_embed(x)
            if x.dim() == 4:
                x = x.reshape(B, -1, x.shape[-1])
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)
            for block in self.blocks:
                x = block(x)
            x = self.norm(x)
            return x[:, 0]  # CLS token -> (B, 768)

    encoder = ConchEncoder(trunk)
    encoder.eval()
    encoder.to(device)
    return encoder


@torch.no_grad()
def extract_slice_features(encoder, device, slice_id, n_spots):
    dataset = PatchDataset(slice_id, n_spots)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                         num_workers=NUM_WORKERS, pin_memory=True)
    feats = []
    for batch in loader:
        feats.append(encoder(batch.to(device)).cpu().float().numpy())
    return np.concatenate(feats, axis=0)  # (N_spots, 768)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    encoder = load_conch(device)

    for dataset in DATASETS:
        cases = sorted(os.listdir(os.path.join(COORDS_ROOT, dataset)))

        for case in tqdm(cases, desc=dataset):
            out_path = os.path.join(OUTPUT_ROOT, dataset, case + '.npz')
            if os.path.exists(out_path):
                continue

            coords_case_dir = os.path.join(COORDS_ROOT, dataset, case)
            csvs = sorted(glob.glob(os.path.join(coords_case_dir, '*.csv')),
                          key=natural_keys)

            case_feats = {}
            for csv_path in tqdm(csvs, desc=case, leave=False):
                slice_id = os.path.basename(csv_path).replace('.csv', '')
                patch_dir = os.path.join(PATCH_ROOT, slice_id)
                if not os.path.isdir(patch_dir):
                    print(f'  WARNING: patch dir not found for {slice_id}, skip')
                    continue
                n_spots = sum(1 for _ in open(csv_path)) - 1
                case_feats[slice_id] = extract_slice_features(
                    encoder, device, slice_id, n_spots)

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.savez(out_path, **case_feats)

    print('Done.')
