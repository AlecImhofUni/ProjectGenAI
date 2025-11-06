#!/usr/bin/env python3
"""
Training-free detection of AI-generated images with VFMs (DINOv2) — RIGID (Noise), Blur, and MINDER.
Computes global and per-dataset AUROC; writes per-image and summary CSVs to a dedicated 'results/' dir.

What this script does
---------------------
- Loads a timm DINOv2 ViT (B/L/G) as a **feature extractor** (num_classes=0); returns L2-normalized CLS embeddings.
- Reads images from data_root/{real,fake} and resizes on-the-fly to 224×224 (Resize -> CenterCrop); no disk recompression.
- Applies **pixel-space perturbations** on [0,1] images, then re-normalizes for the encoder:
    • **Noise (RIGID)**: add Gaussian noise σ ∈ [0,1] (repeat n_noise, average distance).
    • **Blur**: Gaussian blur with σ_blur in **pixels** (deterministic, 1 pass).
    • **MINDER**: per-image min between Noise and Blur scores (either computed in-memory or from CSVs).
- Scores each image by **cosine distance**: 1 − cos(embedding_clean, embedding_perturbed).
  Higher score ⇒ more sensitive ⇒ more likely **fake**.
- Reports **GLOBAL** AUROC and **per-dataset** AUROC (dataset parsed from filename or CSV tag).
- (Optional) Calibrates a threshold at **FPR=5%** on the same REAL set (inspection only).

Folder layout (expected)
------------------------
  data_root/
    real/  <images...>
    fake/  <images...>
  # Filenames or an index CSV should carry/encode the dataset tag (e.g., ADM, CollabDiff).

Outputs (CSV)
-------------
  results/
    rigid_scores.csv    | per-image (Noise) : path,label,score,dataset
    rigid_summary.csv   | GLOBAL + per-dataset AUROC (+ hyperparams)
    blur_scores.csv     | per-image (Blur)
    blur_summary.csv    | summary (Blur)
    minder_scores.csv   | per-image (MINDER = max of Noise/Blur distances ≡ min of similarities)
    minder_summary.csv  | summary (MINDER)

Key CLI flags
-------------
  --perturb {noise,blur,both,minder}
      noise  : compute Noise only
      blur   : compute Blur only
      both   : compute Noise, then Blur, then MINDER in the same run
      minder : compute Noise+Blur in-memory and export MINDER only
  --sigma <float>         Gaussian **noise** std in pixel units [0,1]  (e.g., 0.009)
  --n_noise <int>         Number of noise samples to average (e.g., 3)
  --sigma_blur <float>    Gaussian **blur** std in **pixels** at 224×224 (e.g., 1.6)
  --calibrate_on_same_set Calibrate thr@FPR=5% on REALs from this eval set (optimistic; for inspection)

CSV-only (no recompute) mode
----------------------------
  --minder_from_csv                 Merge existing per-image CSVs and compute MINDER only
  --rigid_csv <path>                Path to Noise (rigid_scores.csv)
  --blur_csv  <path>                Path to Blur  (blur_scores.csv)
  # In this mode, --data_root/--model are ignored.

Dependencies
------------
  pip install timm torch torchvision scikit-learn pillow
  (pandas is optional; a CSV fallback is used if pandas is missing)

Typical runs
------------
  # Noise (RIGID)
  python training_free_detect.py \
    --data_root ~/data/pairs_1000_eval \
    --model dinov2-l14 \
    --batch_size 64 \
    --sigma 0.009 --n_noise 3 \
    --perturb noise \
    --results_dir results

  # Blur
  python training_free_detect.py \
    --data_root ~/data/pairs_1000_eval \
    --model dinov2-l14 \
    --batch_size 64 \
    --sigma_blur 0.55 \
    --perturb blur \
    --results_dir results

  # All three in one go: Noise, Blur, MINDER
  python training_free_detect.py \
    --data_root ~/data/pairs_1000_eval \
    --model dinov2-l14 \
    --batch_size 64 \
    --sigma 0.009 --n_noise 3 \
    --sigma_blur 0.55 \
    --perturb both \
    --results_dir results

  # Compute MINDER from existing CSVs (no embedding recompute)
  python training_free_detect.py \
    --minder_from_csv \
    --rigid_csv results/rigid_scores.csv \
    --blur_csv  results/blur_scores.csv \
    --results_dir results
"""


import argparse
import math
import re
import sys
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

# ------------------ Pixel-space perturbations ------------------

def gaussian_blur_pixel(x_norm: torch.Tensor, sigma_pix: float, mean, std, kernel_size: int | None = None):
    """
    Fixed 3×3 Gaussian blur in pixel space [0,1], then re-normalize.
    x_norm: [B,3,H,W] normalized (ImageNet)
    sigma_pix: Gaussian std in pixels (default 0.55 matches paper).
    """
    if sigma_pix <= 0:
        return x_norm

    x_pix = denorm(x_norm, mean, std)              # -> [0,1]
    B, C, H, W = x_pix.shape
    device, dtype = x_pix.device, x_pix.dtype

    # Fixed 3×3 separable kernel
    k = 3
    half = 1
    t = torch.tensor([-1.0, 0.0, 1.0], device=device, dtype=dtype)  # length-3 axis
    g1d = torch.exp(-0.5 * (t / max(sigma_pix, 1e-6))**2)
    g1d = g1d / g1d.sum()  # normalize to sum=1

    g_x = g1d.view(1, 1, 1, k).repeat(C, 1, 1, 1)   # (C,1,1,3)
    g_y = g1d.view(1, 1, k, 1).repeat(C, 1, 1, 1)   # (C,1,3,1)

    # reflect padding of 1 pixel
    x_pad = F.pad(x_pix, (half, half, half, half), mode="reflect")

    # depthwise conv: groups=C
    out = F.conv2d(x_pad, g_x, groups=C)
    out = F.conv2d(out,   g_y, groups=C)

    out = out.clamp(0.0, 1.0)
    return renorm(out, mean, std)

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

@torch.no_grad()
def score_loader_blur(model, loader, sigma_blur=0.55, device="cuda"):
    """
    Compute RIGID-style scores with *Gaussian blur* (single pass, fixed 3×3 kernel):
      - e0 = CLS clean
      - x_blur = gaussian_blur_pixel(x, sigma_blur) in [0,1], then re-normalize
      - distance = 1 - cosine(e0, e_blur)
    Returns:
      scores [N], labels [N], paths [N], datasets [N]
    """
    mean, std = _mean_std(device, torch.float32)

    all_scores, all_labels, all_paths, all_dsets = [], [], [], []
    for x, y, paths, dsets in loader:
        x = x.to(device, non_blocking=True)
        e0 = embed_cls(model, x)

        x_blur = gaussian_blur_pixel(x, sigma_blur, mean, std)
        e1 = embed_cls(model, x_blur)

        d = cosine_distance(e0, e1)  # [B]

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

# ------------------ CSV Export Helpers ------------------

def _save_csvs(results_dir: Path, paths, labels, scores, datasets, model_name_print, args, method_tag: str):
    """Write per-image and summary CSVs for a given method_tag: 'noise' | 'blur' | 'minder'."""
    if method_tag == "noise":
        out_scores = results_dir / "rigid_scores.csv"
        out_summary = results_dir / "rigid_summary.csv"
    elif method_tag == "blur":
        out_scores = results_dir / "blur_scores.csv"
        out_summary = results_dir / "blur_summary.csv"
    else:
        out_scores = results_dir / "minder_scores.csv"
        out_summary = results_dir / "minder_summary.csv"

    rows_scores = [{"path": p, "label": int(l), "score": float(s), "dataset": d}
                   for p, l, s, d in zip(paths, labels.tolist(), scores.tolist(), datasets)]

    # Global AUROC + per-dataset
    roc_global = auroc(scores, labels)
    rows_summary = [{
        "dataset": "GLOBAL",
        "n": int(len(labels)),
        "auroc": float(roc_global),
        "thr_at_fpr_0.05": "",
        "fpr_on_cal_set":  "",
        "tpr_on_cal_set":  "",
        "model": model_name_print,
        "sigma":      (float(args.sigma)      if method_tag in ("noise","minder") else ""),
        "sigma_blur": (float(args.sigma_blur) if method_tag in ("blur","minder")  else ""),
        "n_noise":    (int(args.n_noise)      if method_tag in ("noise","minder") else 1),
    }]
    per_ds = per_dataset_auroc(scores, labels, datasets)
    for ds, (roc, n) in per_ds.items():
        rows_summary.append({
            "dataset": ds,
            "n": int(n),
            "auroc": (float(roc) if not (roc != roc) else ""),
            "thr_at_fpr_0.05": "",
            "fpr_on_cal_set":  "",
            "tpr_on_cal_set":  "",
            "model": model_name_print,
            "sigma":      (float(args.sigma)      if method_tag in ("noise","minder") else ""),
            "sigma_blur": (float(args.sigma_blur) if method_tag in ("blur","minder")  else ""),
            "n_noise":    (int(args.n_noise)      if method_tag in ("noise","minder") else 1),
        })

    try:
        import pandas as pd
        pd.DataFrame(rows_scores).to_csv(out_scores, index=False)
        pd.DataFrame(rows_summary).to_csv(out_summary, index=False)
    except Exception:
        import csv
        with open(out_scores, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path","label","score","dataset"])
            w.writeheader(); [w.writerow(r) for r in rows_scores]
        with open(out_summary, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "dataset","n","auroc","thr_at_fpr_0.05","fpr_on_cal_set","tpr_on_cal_set",
                "model","sigma","sigma_blur","n_noise"
            ])
            w.writeheader(); [w.writerow(r) for r in rows_summary]
    print(f"[Saved] Per-image scores -> {out_scores}")
    print(f"[Saved] Summary AUROC    -> {out_summary}")

def _load_scores_csv(path: Path):
    import pandas as pd
    df = pd.read_csv(path)
    for col in ("path","label","score","dataset"):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path}")
    return df

# ------------------ Main (CLI) ------------------

def parse_args():
    ap = argparse.ArgumentParser("RIGID baseline (timm, noise-only, per-dataset AUROC, CSV to results/)")
    ap.add_argument("--data_root", default=None, type=str,
                    help="Folder with real/ and fake/ (required unless --minder_from_csv)")
    ap.add_argument("--model", default="dinov2-l14", type=str,
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
    ap.add_argument("--sigma_blur", default=0.55, type=float,
                    help="Gaussian blur σ (fixed 3×3 kernel in pixel space; default 0.55).")    
    ap.add_argument("--perturb", default="both",
                    choices=["noise", "blur", "both", "minder"],
                    help="Perturbation to use: noise | blur | both (noise+blur [+minder]) | minder (max of noise/blur distances ≡ min of similarities))")
    ap.add_argument("--minder_from_csv", action="store_true",
                    help="Compute MINDER by merging existing rigid_scores.csv and blur_scores.csv (no recompute).")
    ap.add_argument("--rigid_csv", default="results/rigid_scores.csv", type=str)
    ap.add_argument("--blur_csv",  default="results/blur_scores.csv",  type=str)
    return ap.parse_args()

def main():
    args = parse_args()

    # Prepare results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # MINDER from existing CSVs
    if args.minder_from_csv:
        rigid_df = _load_scores_csv(Path(args.rigid_csv)).rename(columns={"score":"score_noise"})
        blur_df  = _load_scores_csv(Path(args.blur_csv)).rename(columns={"score":"score_blur"})
        # Merge on 'path' (intersection)
        df = rigid_df.merge(blur_df[["path","score_blur"]], on="path", how="inner")
        if df.empty:
            raise SystemExit("[Error] Merge produced 0 rows. Check that both CSVs refer to the same images/paths.")
        # Build tensors for metrics/save helper
        paths    = df["path"].tolist()
        datasets = df["dataset"].tolist()   # dataset label from rigid_df
        labels   = torch.tensor(df["label"].values, dtype=torch.long)
        scores_n = torch.tensor(df["score_noise"].values, dtype=torch.float32)
        scores_b = torch.tensor(df["score_blur"].values,  dtype=torch.float32)
        scores   = torch.minimum(scores_n, scores_b)  # MINDER

        model_name_print = "from-csv"

        roc_global = auroc(scores, labels)
        print(f"[RIGID] Global AUROC (MINDER-from-CSV) = {roc_global:.4f}")
        per_ds = per_dataset_auroc(scores, labels, datasets)
        print("[RIGID] Per-dataset AUROC (MINDER-from-CSV):")
        for ds, (roc, n) in per_ds.items():
            roc_str = f"{roc:.4f}" if not (roc != roc) else "NaN"
            print(f"  - {ds:<16} AUROC = {roc_str}  |  n = {n}")

        _save_csvs(results_dir, paths, labels, scores, datasets,
                model_name_print, args, method_tag="minder")
        return

    # ---- Normal path requires data_root ----
    if not args.data_root:
        raise SystemExit("Error: --data_root is required unless --minder_from_csv is set.")

    # Data (on-the-fly resize/crop -> 224)
    loader, _ = make_loader(args.data_root, batch_size=args.batch_size, workers=args.workers)

    # Model (timm)
    model = create_dinov2(args.model, device=args.device)
    model_name_print = NAME_MAP.get(args.model, args.model)
    print(f"[Info] Using timm model: {model_name_print} | Device: {args.device}")

        # Score all images according to --perturb
    method = args.perturb
    if method == "noise":
        scores, labels, paths, datasets = score_loader(
            model, loader, sigma=args.sigma, n_noise=args.n_noise, device=args.device
        )
        method_print = f"RIGID/Noise (sigma={args.sigma}, n_noise={args.n_noise})"

    elif method == "blur":
        scores, labels, paths, datasets = score_loader_blur(
            model, loader, sigma_blur=args.sigma_blur, device=args.device
        )
        method_print = f"Blur (sigma_blur={args.sigma_blur})"

    elif method == "minder":
        # Compute noise and blur scores (same alignment/order)
        scores_n, labels, paths, datasets = score_loader(
            model, loader, sigma=args.sigma, n_noise=args.n_noise, device=args.device
        )
        scores_b, _, _, _ = score_loader_blur(
            model, loader, sigma_blur=args.sigma_blur, device=args.device
        )
        # MINDER = per-image max(distance) ≡ min(similarity)
        scores = torch.minimum(scores_n, scores_b)

        method_print = f"MINDER (max(noise,blur) distances; sigma={args.sigma}, n_noise={args.n_noise}, sigma_blur={args.sigma_blur})"

        # Print metrics (global + per-dataset)
        roc_global = auroc(scores, labels)
        print(f"[RIGID] Global AUROC (MINDER) = {roc_global:.4f}")
        per_ds = per_dataset_auroc(scores, labels, datasets)
        print("[RIGID] Per-dataset AUROC (MINDER):")
        for ds, (roc, n) in per_ds.items():
            roc_str = f"{roc:.4f}" if not (roc != roc) else "NaN"
            print(f"  - {ds:<16} AUROC = {roc_str}  |  n = {n}")

        # threshold on same set 
        if args.calibrate_on_same_set:
            thr = threshold_at_fpr(scores[labels == 0], target_fpr=0.05)
            y_pred = (scores >= thr).int()
            fpr = ((y_pred[labels == 0] == 1).float().mean().item()) if (labels == 0).any() else float("nan")
            tpr = ((y_pred[labels == 1] == 1).float().mean().item()) if (labels == 1).any() else float("nan")
            print(f"[RIGID/MINDER] thr@FPR=5% = {thr:.6f} | FPR={fpr:.3f} | TPR={tpr:.3f}")

        # Save MINDER CSVs
        _save_csvs(
            results_dir, paths, labels, scores, datasets,
            model_name_print, args, method_tag="minder"
        )
        return  # done

    else:  # both
        # Run noise
        scores_n, labels, paths, datasets = score_loader(
            model, loader, sigma=args.sigma, n_noise=args.n_noise, device=args.device
        )
        # Run blur (re-use labels/paths/datasets from above)
        scores_b, _, _, _ = score_loader_blur(
            model, loader, sigma_blur=args.sigma_blur, device=args.device
        )

        # Export both below; printing summaries separately
        # First: NOISE
        roc_global = auroc(scores_n, labels)
        print(f"[RIGID] Global AUROC (NOISE) = {roc_global:.4f}")
        per_ds = per_dataset_auroc(scores_n, labels, datasets)
        print("[RIGID] Per-dataset AUROC (NOISE):")
        for ds, (roc, n) in per_ds.items():
            roc_str = f"{roc:.4f}" if not (roc != roc) else "NaN"
            print(f"  - {ds:<16} AUROC = {roc_str}  |  n = {n}")
        _save_csvs(
            results_dir, paths, labels, scores_n, datasets,
            model_name_print, args, method_tag="noise"
        )

        # Then: BLUR
        roc_global = auroc(scores_b, labels)
        print(f"[RIGID] Global AUROC (BLUR)  = {roc_global:.4f}")
        per_ds = per_dataset_auroc(scores_b, labels, datasets)
        print("[RIGID] Per-dataset AUROC (BLUR):")
        for ds, (roc, n) in per_ds.items():
            roc_str = f"{roc:.4f}" if not (roc != roc) else "NaN"
            print(f"  - {ds:<16} AUROC = {roc_str}  |  n = {n}")
        _save_csvs(
            results_dir, paths, labels, scores_b, datasets,
            model_name_print, args, method_tag="blur"
        )

        # Finally: MINDER = min(noise, blur)
        scores_m = torch.minimum(scores_n, scores_b)
        roc_global = auroc(scores_m, labels)
        print(f"[RIGID] Global AUROC (MINDER) = {roc_global:.4f}")
        per_ds = per_dataset_auroc(scores_m, labels, datasets)
        print("[RIGID] Per-dataset AUROC (MINDER):")
        for ds, (roc, n) in per_ds.items():
            roc_str = f"{roc:.4f}" if not (roc != roc) else "NaN"
            print(f"  - {ds:<16} AUROC = {roc_str}  |  n = {n}")
        _save_csvs(
            results_dir, paths, labels, scores_m, datasets,
            model_name_print, args, method_tag="minder"
        )        
        return  # done

    # === Single-method path (noise OR blur) continues ===
    roc_global = auroc(scores, labels)
    print(f"[RIGID] Global AUROC = {roc_global:.4f}  ({method_print})")

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

    # Save CSVs
    method_tag = "noise" if method == "noise" else "blur"
    _save_csvs(
        results_dir, paths, labels, scores, datasets,
        model_name_print, args, method_tag=method_tag
    )

if __name__ == "__main__":
    main()
