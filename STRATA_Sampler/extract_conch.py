"""Extract CONCH ViT-B-16 features for every ST spot.

Writes one npy per slide at
    <DATA_ROOT>/conch_features/<dataset>/<patient>/<slide>_conch.npy
aligned 1:1 with samples in <DATA_ROOT>/<dataset>/<patient>/<slide>.npy.
"""
import os, argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from conch.open_clip_custom import create_model_from_pretrained

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "./data/data_infor"))
IMG_ROOT  = Path(os.environ.get("IMAGE_ROOT", "./data/patches"))
CONCH_CKPT = os.environ.get("CONCH_CKPT", "./checkpoints/conch_vit_b_16.bin")
OUT_ROOT = DATA_ROOT / "conch_features"


def build_model(device: str):
    m = create_model_from_pretrained(
        "conch_ViT-B-16",
        checkpoint_path=CONCH_CKPT,
        force_image_size=224,
        return_transform=False,
    )
    return m.visual.to(device).eval()


TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@torch.no_grad()
def extract_slide(vis, samples, device: str, batch_size: int = 128) -> np.ndarray:
    feats = np.zeros((len(samples), 512), dtype=np.float32)
    batch_imgs, batch_idx = [], []
    for i, s in enumerate(samples):
        img_path = IMG_ROOT / s["sample_id"] / f"{s['patch_id']}.png"
        img = Image.open(img_path).convert("RGB")
        batch_imgs.append(TF(img))
        batch_idx.append(i)
        if len(batch_imgs) == batch_size:
            x = torch.stack(batch_imgs).to(device, non_blocking=True)
            y = vis(x)[0].cpu().numpy()
            feats[batch_idx] = y
            batch_imgs, batch_idx = [], []
    if batch_imgs:
        x = torch.stack(batch_imgs).to(device, non_blocking=True)
        y = vis(x)[0].cpu().numpy()
        feats[batch_idx] = y
    return feats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="HER2")
    p.add_argument("--gpu", default="0")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vis = build_model(device)
    print(f"[extract_conch] loaded CONCH ViT-B-16 -> {device}")

    ds_root = DATA_ROOT / args.dataset
    out_root = OUT_ROOT / args.dataset
    total = 0
    for pdir in sorted(ds_root.iterdir()):
        if not pdir.is_dir():
            continue
        out_pdir = out_root / pdir.name
        out_pdir.mkdir(parents=True, exist_ok=True)
        for slide_npy in sorted(pdir.glob("*.npy")):
            slide = slide_npy.stem
            out_file = out_pdir / f"{slide}_conch.npy"
            if out_file.exists() and not args.force:
                arr = np.load(out_file)
                print(f"  skip {pdir.name}/{slide} ({arr.shape})")
                total += arr.shape[0]
                continue
            samples = list(np.load(slide_npy, allow_pickle=True))
            feats = extract_slide(vis, samples, device, args.batch_size)
            np.save(out_file, feats)
            print(f"  {pdir.name}/{slide}: {feats.shape}")
            total += feats.shape[0]
    print(f"[extract_conch] done, total spots = {total}")


if __name__ == "__main__":
    main()
