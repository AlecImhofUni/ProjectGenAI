#!/usr/bin/env python3
"""
RIGID baseline (noise-only) with timm + DINOv2 (LVD-142M), per-dataset AUROC,
and CSV exports into a dedicated 'results/' directory.

What this script does
---------------------
- Loads a timm DINOv2 ViT (B/L/G) as a feature extractor (num_classes=0).
- Reads images from data_root/{real,fake} and resizes on-the-fly to 224×224
  in memory (Resize -> CenterCrop), no recompression on disk.
- Adds Gaussian noise in pixel space [0,1] (n times), re-normalizes,
  embeds with DINOv2, and computes the cosine distance vs. the clean embedding.
  The score per image is the mean distance across the n noises.
- Computes global AUROC and per-dataset AUROC (dataset parsed from filename).

Folder layout (new naming)
--------------------------
  data_root/
    real/  pair_0001_AMD.jpeg, pair_0002_CollabDiff.png, ...
    fake/  pair_0001_AMD.png,  pair_0002_CollabDiff.jpeg, ...

Dependencies
------------
  pip install timm torch torchvision scikit-learn pillow
  (pandas is optional; a CSV fallback is included if pandas is missing)

Typical run
-----------
  python rigid_baseline_timm.py \
    --data_root ~/data/pairs_1000_eval \
    --model dinov2-b14 \
    --batch_size 64 \
    --sigma 0.009 \
    --n_noise 3 \
    --results_dir results \
    --calibrate_on_same_set
"""

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import timm
from sklearn.metrics import roc_auc_score

# ------------------ Constants & Config ------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 224  # target resolution (paper-friendly; we force 224 here)

# Friendly aliases -> timm names
NAME_MAP = {
    "dinov2-b14": "vit_base_patch14_dinov2.lvd142m",
    "dinov2-l14": "vit_large_patch14_dinov2.lvd142m",
    "dinov2-g14": "vit_giant_patch14_dinov2.lvd142m",
}

# NEW: filenames are now like "pair_0001_AMD.jpg" or "pair_0002_CollabDiff.png"
DATASET_REGEX = re.compile(r"^pair_\d+_([^_]+)$", re.IGNORECASE)

def parse_dataset_from_path(path_str: str) -> str:
    """
    Parse dataset name from filenames like:
        pair_0001_AMD.jpg        -> "AMD"
        pair_1000_CollabDiff.png -> "CollabDiff"
    Fallback: last underscore token if regex doesn't match.
    """
    stem = Path(path_str).stem
    m = DATASET_REGEX.match(stem)
    if m:
        return m.group(1)
    parts = stem.split("_")
    return parts[-1] if len(parts) >= 2 else "UNKNOWN"

# ------------------ Dataset (on-the-fly resize) ------------------

class RealFakeFolder(Dataset):
    """
    Loads:
      <root>/real/*.jpg|png|...
      <root>/fake/*.jpg|png|...

    All resizing happens in memory via torchvision transforms.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root, transform=None):
        self.items = []
        root = Path(root)
        for name, y in [("real", 0), ("fake", 1)]:  # 0=real, 1=fake
            d = root / name
            if not d.is_dir():
                raise FileNotFoundError(f"Missing folder: {d}")
            for p in sorted(d.rglob("*")):
                if p.suffix.lower() in self.exts:
                    self.items.append((str(p), y))
        if not self.items:
            raise RuntimeError(f"No images found under {root}")
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img) if self.transform is not None else img
        dataset_name = parse_dataset_from_path(path)
        return x, y, path, dataset_name


def make_loader(data_root, batch_size=64, workers=4):
    """
    On-the-fly preprocessing:
      - Resize(224) bicubic (shorter side -> 224)
      - CenterCrop(224)
      - ToTensor + Normalize(ImageNet)
    """
    tfm = T.Compose([
        T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds = RealFakeFolder(data_root, transform=tfm)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=False
    )
    return loader, ds

# ------------------ Model / Embeddings (timm only) ------------------

def create_dinov2(model_name="dinov2-b14", device="cuda"):
    """
    Create a timm DINOv2 model as a feature extractor.
    num_classes=0 -> forward() returns features.
    We pass img_size=224 so timm resizes/interpolates pos-embeds accordingly.
    """
    model_name = NAME_MAP.get(model_name, model_name)  # resolve alias
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0,    # features only
        img_size=IMG_SIZE # ensure 224 grid at creation
    )
    model.eval().to(device)
    return model


@torch.no_grad()
def embed_cls(model, x):
    """
    Extract L2-normalized CLS embeddings (shape [B, D]) from a timm ViT.
    """
    feats = model.forward_features(x)
    if isinstance(feats, dict):
        if "x_norm_clstoken" in feats:
            cls = feats["x_norm_clstoken"]
        elif "cls_token" in feats:
            cls = feats["cls_token"]
        elif "x_norm" in feats:
            cls = feats["x_norm"][:, 0]
        else:
            first_tensor = next(v for v in feats.values() if torch.is_tensor(v))
            cls = first_tensor[:, 0] if first_tensor.dim() >= 2 else first_tensor
    else:
        cls = feats[:, 0]
    cls = F.normalize(cls, dim=-1)
    return cls

# ------------------ Pixel-space perturbations ------------------

def _mean_std(device, dtype):
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std

def denorm(x_norm, mean, std):
    """Inverse of Normalize: back to approx. pixel space [0..1]."""
    return x_norm * std + mean

def renorm(x_pix, mean, std):
    """Re-apply Normalize after pixel-space ops."""
    return (x_pix - mean) / std

def add_gaussian_noise_pixel(x_norm, sigma_pix, mean, std):
    """
    Add Gaussian noise in pixel space [0,1], then clip and re-normalize.
    Args:
        x_norm: [B,3,H,W] normalized
        sigma_pix: noise std in pixel units (e.g., 0.009)
    """
    x_pix = denorm(x_norm, mean, std)
    noise = torch.randn_like(x_pix) * sigma_pix
    x_noisy = torch.clamp(x_pix + noise, 0.0, 1.0)
    return renorm(x_noisy, mean, std)

# ------------------ Scoring & Metrics ------------------

def cosine_distance(e0, e1):
    """Cosine distance for L2-normalized embeddings (1 - cosine_sim)."""
    return 1.0 - torch.sum(e0 * e1, dim=-1)

@torch.no_grad()
def score_loader(model, loader, sigma=0.009, n_noise=3, device="cuda"):
    """
    Compute RIGID scores:
      - e0 = CLS clean
      - repeat n_noise: add Gaussian noise in [0,1], get e1, compute distance
      - final score = mean distance over n_noise
    Returns:
      scores [N], labels [N] (0=real,1=fake), paths [N], datasets [N]
    """
    mean, std = _mean_std(device, torch.float32)

    all_scores, all_labels, all_paths, all_dsets = [], [], [], []
    for x, y, paths, dsets in loader:
        x = x.to(device, non_blocking=True)
        e0 = embed_cls(model, x)

        dists = []
        for _ in range(n_noise):
            x_pert = add_gaussian_noise_pixel(x, sigma, mean, std)
            e1 = embed_cls(model, x_pert)
            dists.append(cosine_distance(e0, e1))
        d = torch.stack(dists, dim=0).mean(dim=0)  # [B]

        all_scores.append(d.cpu())
        all_labels.append(y)
        all_paths.extend(paths)
        all_dsets.extend(dsets)

    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    return scores, labels, all_paths, all_dsets

def auroc(scores, labels):
    """AUROC with sklearn (labels: 0=real, 1=fake; scores: higher=faker)."""
    return float(roc_auc_score(labels.numpy(), scores.numpy()))

def per_dataset_auroc(scores, labels, datasets):
    """AUROC per dataset (requires both classes present)."""
    idx_by_ds = defaultdict(list)
    for i, ds in enumerate(datasets):
        idx_by_ds[ds].append(i)

    results = {}
    for ds, idxs in sorted(idx_by_ds.items()):
        s = scores[idxs]
        y = labels[idxs]
        uniques = set(y.tolist())
        if len(uniques) < 2:
            results[ds] = (float("nan"), len(idxs))
        else:
            results[ds] = (float(roc_auc_score(y.numpy(), s.numpy())), len(idxs))
    return results

def threshold_at_fpr(real_scores, target_fpr=0.05):
    """Choose threshold so that FPR ~= target_fpr on REAL scores."""
    real_scores = torch.as_tensor(real_scores)
    if real_scores.numel() == 0:
        return float("nan")
    real_sorted = torch.sort(real_scores).values
    n = len(real_sorted)
    q = max(0.0, min(1.0, 1.0 - target_fpr))
    k = int(math.floor(q * (n - 1))) if n > 1 else 0
    return float(real_sorted[k])

# ------------------ Main (CLI) ------------------

def parse_args():
    ap = argparse.ArgumentParser("RIGID baseline (timm, noise-only, per-dataset AUROC, CSV to results/)")
    ap.add_argument("--data_root", required=True, type=str,
                    help="Folder with real/ and fake/ subfolders (e.g., ~/data/pairs_1000_eval)")
    ap.add_argument("--model", default="dinov2-b14", type=str,
                    help="timm model or alias: dinov2-b14 | dinov2-l14 | dinov2-g14 | "
                         "vit_base_patch14_dinov2.lvd142m, etc.")
    ap.add_argument("--batch_size", default=64, type=int)
    ap.add_argument("--workers", default=4, type=int)
    ap.add_argument("--sigma", default=0.009, type=float,
                    help="Gaussian noise std in pixel units [0,1] (try 0.008–0.010)")
    ap.add_argument("--n_noise", default=3, type=int,
                    help="How many noise samples to average")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    ap.add_argument("--calibrate_on_same_set", action="store_true",
                    help="Calibrate threshold@FPR=5% on the SAME REAL set (optimistic).")
    ap.add_argument("--results_dir", default="results", type=str,
                    help="Directory to write CSV results (will be created if missing).")
    return ap.parse_args()

def main():
    args = parse_args()

    # Prepare results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Data (on-the-fly resize/crop -> 224)
    loader, _ = make_loader(args.data_root, batch_size=args.batch_size, workers=args.workers)

    # Model (timm)
    model = create_dinov2(args.model, device=args.device)
    model_name_print = NAME_MAP.get(args.model, args.model)
    print(f"[Info] Using timm model: {model_name_print} | Device: {args.device}")

    # Score all images
    scores, labels, paths, datasets = score_loader(
        model, loader, sigma=args.sigma, n_noise=args.n_noise, device=args.device
    )

    # Global AUROC
    roc_global = auroc(scores, labels)
    print(f"[RIGID] Global AUROC = {roc_global:.4f}  (higher is better)")

    # AUROC per dataset
    per_ds = per_dataset_auroc(scores, labels, datasets)
    print("[RIGID] Per-dataset AUROC:")
    for ds, (roc, n) in per_ds.items():
        roc_str = f"{roc:.4f}" if not (roc != roc) else "NaN"
        print(f"  - {ds:<16} AUROC = {roc_str}  |  n = {n}")

    # Optional threshold @ FPR=5% (on same set if flag enabled)
    thr = fpr = tpr = None
    if args.calibrate_on_same_set:
        thr = threshold_at_fpr(scores[labels == 0], target_fpr=0.05)
        y_pred = (scores >= thr).int()
        fpr = ((y_pred[labels == 0] == 1).float().mean().item()) if (labels == 0).any() else float("nan")
        tpr = ((y_pred[labels == 1] == 1).float().mean().item()) if (labels == 1).any() else float("nan")
        print(f"[RIGID] thr@FPR=5% = {thr:.6f} | FPR={fpr:.3f} | TPR={tpr:.3f}")

    # ------------------ CSV exports to results/ ------------------

    # 1) Per-image scores
    rows_scores = [{"path": p, "label": int(l), "score": float(s), "dataset": d}
                   for p, l, s, d in zip(paths, labels.tolist(), scores.tolist(), datasets)]
    out_scores = results_dir / "rigid_scores.csv"

    # 2) Summary (global + per-dataset AUROC)
    rows_summary = [{
        "dataset": "GLOBAL",
        "n": int(len(labels)),
        "auroc": float(roc_global),
        "thr_at_fpr_0.05": (float(thr) if thr is not None else ""),
        "fpr_on_cal_set":  (float(fpr) if fpr is not None else ""),
        "tpr_on_cal_set":  (float(tpr) if tpr is not None else ""),
        "model": model_name_print,
        "sigma": float(args.sigma),
        "n_noise": int(args.n_noise),
    }]
    for ds, (roc, n) in per_ds.items():
        rows_summary.append({
            "dataset": ds,
            "n": int(n),
            "auroc": (float(roc) if not (roc != roc) else ""),
            "thr_at_fpr_0.05": "",
            "fpr_on_cal_set": "",
            "tpr_on_cal_set": "",
            "model": model_name_print,
            "sigma": float(args.sigma),
            "n_noise": int(args.n_noise),
        })
    out_summary = results_dir / "rigid_summary.csv"

    # Write CSVs
    try:
        import pandas as pd
    except Exception:
        pd = None

    if pd is not None:
        pd.DataFrame(rows_scores).to_csv(out_scores, index=False)
        pd.DataFrame(rows_summary).to_csv(out_summary, index=False)
        print(f"[Saved] Per-image scores -> {out_scores}")
        print(f"[Saved] Summary AUROC   -> {out_summary}")
    else:
        import csv
        with open(out_scores, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "label", "score", "dataset"])
            w.writeheader(); [w.writerow(r) for r in rows_scores]
        with open(out_summary, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "dataset","n","auroc","thr_at_fpr_0.05","fpr_on_cal_set","tpr_on_cal_set",
                "model","sigma","n_noise"
            ])
            w.writeheader(); [w.writerow(r) for r in rows_summary]
        print(f"[Saved] (fallback csv) Per-image scores -> {out_scores} | Summary -> {out_summary} (no pandas)")

if __name__ == "__main__":
    main()
