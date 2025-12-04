#!/usr/bin/env python3
"""
Training-free detection of AI-generated images with VFMs (DINOv2) — RIGID (Noise), Contrastive Blur, and MINDER.
Computes global and per-dataset AUROC; writes per-image and summary CSVs to a dedicated 'results/' dir.

What this script does
---------------------
- Loads a timm DINOv2 / DINOv3 ViT (B/L/G) as a **feature extractor** (num_classes=0); returns L2-normalized CLS embeddings.
- Reads images from data_root/{real,fake} and resizes on-the-fly to 224×224 (Resize -> CenterCrop); no disk recompression.
- Applies **pixel-space perturbations** on [0,1] images, then re-normalizes for the encoder:
    • **Noise (RIGID)**: add Gaussian noise σ ∈ [0,1] (repeat n_noise, average distance).
    • **Contrastive Blur**: compare a **blurred** version and a **sharpened** version of the image
      (Gaussian blur with σ_blur in pixels, then contrastive sharpening: x_sharp = clamp(2·x - x_blur)).
    • **MINDER**: per-image **min** between Noise and Contrastive Blur scores
      (either computed in-memory or from CSVs).
- Scores each image by **cosine distance**: 1 − cos(embedding_clean, embedding_perturbed) (or between blur vs sharp).
  Higher score ⇒ more sensitive ⇒ more likely **fake**.
- Reports **GLOBAL** AUROC and **per-dataset** AUROC (dataset parsed from filename or CSV tag).
- Exports per-image CSVs (path, label, score, dataset) and summary CSVs (GLOBAL + per-dataset AUROC).

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
    blur_scores.csv     | per-image (Contrastive Blur)
    blur_summary.csv    | summary (Contrastive Blur)
    minder_scores.csv   | per-image (MINDER = min(Noise, Blur) distance)
    minder_summary.csv  | summary (MINDER)

Key CLI flags
-------------
  --perturb {noise,blur,both,minder}
      noise  : compute Noise only
      blur   : compute Contrastive Blur only
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

"""

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
import pandas as pd


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
    #DINOv2
    "dinov2-b14": "vit_base_patch14_dinov2.lvd142m",
    "dinov2-l14": "vit_large_patch14_dinov2.lvd142m",
    "dinov2-g14": "vit_giant_patch14_dinov2.lvd142m",

    #DINOv3
    "dinov3-s16": "vit_small_patch16_dinov3.lvd1689m",
    "dinov3-l16": "vit_large_patch16_dinov3.lvd1689m",

}

# filenames now like "pair_0001_ADM.jpg", "pair_0002_CollabDiff.png", "pair_0003_SID.jpeg"
DATASET_REGEX = re.compile(r"^pair_\d+_([^_]+)$", re.IGNORECASE)

def _normalize_tag(tag: str) -> str:
    """
    Normalize raw dataset tag strings to a canonical set.

    Args:
        tag (str): Raw dataset tag string.

    Returns:
        str: Canonical tag ("ADM", "CollabDiff", "SID", or original tag).
    """
    t = tag.strip()
    if t.upper() in {"ADM"}:
        return "ADM"
    if t.lower().startswith("collab"):
        return "CollabDiff"
    if t.upper() == "SID":
        return "SID"
    return t

def parse_dataset_from_path(path_str: str) -> str:
    """
    Infer the dataset name from a file path.

    Priority:
    1. Use the filename stem pattern:
       - pair_0001_ADM / pair_0002_CollabDiff / pair_0003_SID
    2. Fallback to directory names:
       - "sid_dataset", "/authentic/", "/fully_synthetic/" → SID
       - "collab" in path → CollabDiff
       - "adm" in path → ADM
    3. If nothing matches, returns "UNKNOWN".

    Args:
        path_str (str): File path as string.

    Returns:
        str: Canonical dataset name.
    """
    stem = Path(path_str).stem
    m = DATASET_REGEX.match(stem)
    if m:
        return _normalize_tag(m.group(1))

    # Fallback: infer from directory names
    p = str(path_str).lower()
    if "sid_dataset" in p or "/authentic/" in p or "/fully_synthetic/" in p:
        return "SID"
    if "collab" in p:
        return "CollabDiff"
    if "/adm/" in p:
        return "ADM"
    return "UNKNOWN"
# ------------------ Dataset (on-the-fly resize) ------------------

class RealFakeFolder(Dataset):
    """
    Dataset that loads images from:
      <root>/real/*.jpg|png|...
      <root>/fake/*.jpg|png|...

    The class:
    - Recursively scans subfolders under real/ and fake/.
    - Stores (path, label) tuples, where label=0 for real and 1 for fake.
    - Applies a torchvision transform (resize, crop, normalization) on-the-fly.
    - Additionally parses a dataset tag from the file path (ADM / CollabDiff / SID / UNKNOWN).
    """
    # Accepted image extensions
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root, transform=None):
        """
        Build the list of all images under <root>/real and <root>/fake.

        Args:
            root: Root directory that contains 'real/' and 'fake/' subfolders.
            transform: Optional torchvision transform applied to each PIL image.
        """
        self.items = []
        root = Path(root)
        for name, y in [("real", 0), ("fake", 1)]:  # 0 = real, 1 = fake
            d = root / name
            if not d.is_dir():
                raise FileNotFoundError(f"Missing folder: {d}")
            # rglob to also support nested folders if needed
            for p in sorted(d.rglob("*")):
                if p.suffix.lower() in self.exts:
                    self.items.append((str(p), y))
        if not self.items:
            raise RuntimeError(f"No images found under {root}")
        self.transform = transform

    def __len__(self) -> int:
        """
        Return the total number of images (real + fake).

        Returns:
            int: Number of items in the dataset.
        """
        return len(self.items)

    def __getitem__(self, idx: int):
        """
        Load an image and return its transformed tensor, label, path, and dataset tag.        
        Args:
            idx: Index of the image to load.
        
        Returns:
            tuple[torch.Tensor, int, str, str]:
            x: Transformed image tensor [3, H, W].
            y: Label (0=real, 1=fake).
            path: Original image path as a string.
            dataset_name: Dataset tag inferred from the path.
        """
        path, y = self.items[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img) if self.transform is not None else img
        dataset_name = parse_dataset_from_path(path)
        return x, y, path, dataset_name


def make_loader(data_root, batch_size: int=64, workers: int=4):
    """
    Create a DataLoader over RealFakeFolder with DINO-friendly preprocessing.

    Preprocessing steps:
      - Resize(IMG_SIZE) using bicubic interpolation (shorter side -> IMG_SIZE).
      - CenterCrop(IMG_SIZE) to get a square.
      - ToTensor() to convert to [0,1] floats.
      - Normalize with ImageNet mean/std.

    Args:
        data_root (str | Path): Directory containing 'real/' and 'fake/'.
        batch_size (int): Batch size for the DataLoader.
        workers (int): Number of worker processes.

    Returns:
        tuple[DataLoader, RealFakeFolder]: The DataLoader and the underlying dataset.
    """
    tfm = T.Compose([
        T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds = RealFakeFolder(data_root, transform=tfm)
    loader = DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=workers, 
        pin_memory=True, 
        drop_last=False
    )
    return loader, ds

# ------------------ Model / Embeddings (timm only) ------------------

def create_backbone(model_name: str="dinov2-l14", device: str="cuda"):
    """
    Create a timm ViT backbone (DINOv2 / DINOv3) as a feature extractor.

    - Resolves short aliases (dinov2-b14, etc.) to full timm names.
    - Uses num_classes=0 so forward() returns features instead of logits.
    - Forces img_size=IMG_SIZE so positional embeddings match 224×224.

    Args:
        model_name (str): Alias or full timm model name.
        device (str): Target device, usually "cuda" or "cpu".

    Returns:
        torch.nn.Module: A ViT backbone in eval mode on the specified device.
    """
    model_name = NAME_MAP.get(model_name, model_name)  # resolve alias
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0,    # features only (CLS embedding)
        img_size=IMG_SIZE # ensure 224 grid at creation
    )
    model.eval().to(device)
    return model


@torch.no_grad()
def embed_cls(model, x: torch.Tensor) -> torch.Tensor:
    """
    Extract L2-normalized CLS embeddings (shape [B, D]) from a timm ViT.

    Handles various timm output formats:
    - Some models return a dict with keys like 'x_norm_clstoken', 'cls_token'.
    - Others return a tensor [B, N, D] where N includes the CLS token at index 0.
    - We always return a tensor [B, D], normalized along the last dimension.

    Args:
        model (torch.nn.Module): ViT backbone from timm (num_classes=0).
        x (torch.Tensor): Input batch tensor [B, 3, H, W] already normalized.

    Returns:
        torch.Tensor: L2-normalized CLS embedding tensor [B, D].
    """
    feats = model.forward_features(x)

    # timm models may return dictionaries or tensors depending on the backbone
    if isinstance(feats, dict):
        if "x_norm_clstoken" in feats:
            cls = feats["x_norm_clstoken"]
        elif "cls_token" in feats:
            cls = feats["cls_token"]
        elif "x_norm" in feats:
            cls = feats["x_norm"][:, 0]
        else:
            # Fallback: take first tensor value and assume CLS at index 0 if 2D+
            first_tensor = next(v for v in feats.values() if torch.is_tensor(v))
            cls = first_tensor[:, 0] if first_tensor.dim() >= 2 else first_tensor
    else:
        # Typical ViT output: [B, N, D] with CLS token at position 0
        cls = feats[:, 0]

    # L2-normalize to make cosine distance well-defined and stable
    cls = F.normalize(cls, dim=-1)
    return cls

# ------------------ Pixel-space perturbations ------------------

def _mean_std(device: str, dtype: torch.dtype):
    """
    Create broadcastable ImageNet mean/std tensors on the given device and dtype.

    Args:
        device (str): Target device (e.g., "cuda" or "cpu").
        dtype (torch.dtype): Target data type (e.g., torch.float32).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (mean, std) each shaped [1, 3, 1, 1].
    """
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std

def denorm(x_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Invert torchvision.Normalize to get back to approx. pixel space [0..1].

    Args:
        x_norm (torch.Tensor): Normalized tensor [B, 3, H, W].
        mean (torch.Tensor): Mean tensor [1, 3, 1, 1].
        std (torch.Tensor): Std tensor [1, 3, 1, 1].

    Returns:
        torch.Tensor: Tensor in approximate [0,1] (pixel space).
    """
    return x_norm * std + mean

def renorm(x_pix: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Apply torchvision.Normalize-like operation to a [0,1] pixel tensor.

    Args:
        x_pix (torch.Tensor): Tensor in [0,1] (pixel space) [B, 3, H, W].
        mean (torch.Tensor): Mean tensor [1, 3, 1, 1].
        std (torch.Tensor): Std tensor [1, 3, 1, 1].

    Returns:
        torch.Tensor: Normalized tensor suitable for the encoder [B, 3, H, W].
    """
    return (x_pix - mean) / std

def add_gaussian_noise_pixel(x_norm: torch.Tensor, sigma_pix: float, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Add Gaussian noise in pixel space [0,1], then clip and re-normalize.

    The noise is applied in pixel units, not in normalized space, to make
    sigma interpretable as a change in the actual image intensities.

    Args:
        x_norm (torch.Tensor): Normalized tensor [B, 3, H, W].
        sigma_pix (float): Noise standard deviation in pixel units (e.g., 0.009).
        mean (torch.Tensor): Mean tensor used to denormalize [1, 3, 1, 1].
        std (torch.Tensor): Std tensor used to denormalize [1, 3, 1, 1].

    Returns:
        torch.Tensor: Perturbed, re-normalized tensor [B, 3, H, W].
    """
    # Back to [0,1] pixel space
    x_pix = denorm(x_norm, mean, std)
    # Add Gaussian noise
    noise = torch.randn_like(x_pix) * sigma_pix
    x_noisy = torch.clamp(x_pix + noise, 0.0, 1.0)
    # Re-apply normalization for the encoder
    return renorm(x_noisy, mean, std)

# ------------------ Pixel-space perturbations ------------------

def gaussian_blur_pixel(x_norm: torch.Tensor, sigma_pix: float,  mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Apply a fixed 3×3 Gaussian blur in pixel space [0,1], then re-normalize.

    Blur is applied channel-wise using depthwise convolutions with a separable
    1D Gaussian kernel (g_x and g_y). Padding is reflection-based to avoid
    dark borders.

    Args:
        x_norm (torch.Tensor): Normalized tensor [B, 3, H, W] (ImageNet stats).
        sigma_pix (float): Gaussian std in pixels. If <= 0, the input is returned.
        mean (torch.Tensor): Mean tensor for de/normalization [1, 3, 1, 1].
        std (torch.Tensor): Std tensor for de/normalization [1, 3, 1, 1].

    Returns:
        torch.Tensor: Blurred and re-normalized tensor with the same shape as x_norm.
    """

    if sigma_pix <= 0:
        # No blur requested, return original tensor
        return x_norm

    # Convert to [0,1] pixel space
    x_pix = denorm(x_norm, mean, std)
    B, C, H, W = x_pix.shape
    device, dtype = x_pix.device, x_pix.dtype

    # Fixed 3×3 separable kernel
    k = 3
    half = 1

    # 1D coordinate axis: [-1, 0, 1]
    t = torch.tensor([-1.0, 0.0, 1.0], device=device, dtype=dtype)
    # Gaussian over that axis
    g1d = torch.exp(-0.5 * (t / max(sigma_pix, 1e-6))**2)
    g1d = g1d / g1d.sum()  # normalize to sum = 1

    # Construct depthwise 3×3 kernels for x and y directions
    g_x = g1d.view(1, 1, 1, k).repeat(C, 1, 1, 1)   # (C,1,1,3)
    g_y = g1d.view(1, 1, k, 1).repeat(C, 1, 1, 1)   # (C,1,3,1)

    # Reflect padding of 1 pixel on all sides
    x_pad = F.pad(x_pix, (half, half, half, half), mode="reflect")

    # Depthwise convolution: blur horizontally then vertically
    out = F.conv2d(x_pad, g_x, groups=C)
    out = F.conv2d(out,   g_y, groups=C)

    out = out.clamp(0.0, 1.0)
    return renorm(out, mean, std)

# ------------------ Scoring & Metrics ------------------

def cosine_distance(e0: torch.Tensor, e1: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine distance for L2-normalized embeddings.

    Because embeddings are normalized, this is equivalent to:
        1 - cosine_similarity(e0, e1)

    Args:
        e0 (torch.Tensor): Embedding tensor [B, D].
        e1 (torch.Tensor): Embedding tensor [B, D].

    Returns:
        torch.Tensor: Distance tensor [B], where larger means more different.
    """
    return 1.0 - torch.sum(e0 * e1, dim=-1)

@torch.no_grad()
def score_loader(model, loader: DataLoader, sigma: float = 0.009, n_noise: int = 3, device: str = "cuda", allowed_datasets=None):
    """
    Compute RIGID-style scores with Gaussian noise (Noise baseline).

    For each batch:
      - e0 = CLS embedding of the clean image.
      - For n_noise times:
          * Add Gaussian noise in pixel space.
          * Re-normalize and re-embed → e1.
          * Compute cosine distance d_i = 1 - cos(e0, e1).
      - Final score = mean(d_i) over n_noise.

    Args:
        model (torch.nn.Module): ViT backbone (timm).
        loader (DataLoader): DataLoader over RealFakeFolder.
        sigma (float): Noise std in pixel units [0,1].
        n_noise (int): Number of independent noise samples to average.
        device (str): Device on which to run the model ("cuda" or "cpu").
        allowed_datasets (set[str] | None): If provided, only keep samples from these datasets.

    Returns:
        tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
            - scores: 1D tensor [N] of scores (higher = more unstable).
            - labels: 1D tensor [N] (0 = real, 1 = fake).
            - paths: List of file paths (str).
            - datasets: List of dataset tags (str).
    """
    mean, std = _mean_std(device, torch.float32)
    allowed_set = None
    if allowed_datasets:
        # case-insensitive match
        allowed_set = {ds.lower() for ds in allowed_datasets}

    all_scores, all_labels, all_paths, all_dsets = [], [], [], []
    for x, y, paths, dsets in loader:
        # Filter batch by dataset if needed
        if allowed_set is not None:
            mask = [ds.lower() in allowed_set for ds in dsets]
            if not any(mask):
                continue
            idx = [i for i, keep in enumerate(mask) if keep]
            x = x[idx]
            y = y[idx]
            paths = [paths[i] for i in idx]
            dsets = [dsets[i] for i in idx]

        x = x.to(device, non_blocking=True)
        # Embedding of the clean image
        e0 = embed_cls(model, x)

        dists = []
        for _ in range(n_noise):
            # Perturb in pixel space and re-embed
            x_pert = add_gaussian_noise_pixel(x, sigma, mean, std)
            e1 = embed_cls(model, x_pert)
            dists.append(cosine_distance(e0, e1))

        # Average distance across noise samples → robustness score
        d = torch.stack(dists, dim=0).mean(dim=0)  # [B]

        all_scores.append(d.cpu())
        all_labels.append(y)
        all_paths.extend(paths)
        all_dsets.extend(dsets)

    if not all_scores:
        raise RuntimeError(
            "No samples left after applying dataset filter "
            f"(allowed={allowed_datasets})."
        )

    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    return scores, labels, all_paths, all_dsets

@torch.no_grad()
def score_loader_blur(model, loader: DataLoader, sigma_blur: float = 0.55, device: str = "cuda", allowed_datasets=None):
    """
    Compute contrastive blur scores using Gaussian blur + contrastive sharpening.

    For each batch:
      - Start from normalized images x_norm.
      - Convert to pixel space x_pix in [0,1].
      - Apply gaussian_blur_pixel(x_norm, sigma_blur, ...) to get a **blurred** normalized tensor x_blur.
      - Convert x_blur back to pixels: x_blur_pix = denorm(x_blur, mean, std).
      - Build a "sharpened" version in pixel space:
            x_sharp_pix = clamp(2 * x_pix - x_blur_pix, 0, 1)
        (this increases local contrast relative to the blurred version).
      - Normalize x_sharp_pix to get x_sharp (x is already normalized).
      - Embed to e_blur and e_sharp.
      - Score = cosine_distance(e_blur, e_sharp).

    This is a **contrastive blur** setup: we directly compare a blurred and a sharpened
    version of the image to measure how sensitive the representation is to degradations.

    Args:
        model (torch.nn.Module): ViT backbone (timm).
        loader (DataLoader): DataLoader over RealFakeFolder.
        sigma_blur (float): Blur std in pixel units.
        device (str): Device on which to run the model ("cuda" or "cpu").
        allowed_datasets (set[str] | None): If provided, only keep samples from these datasets.

    Returns:
        tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
            - scores: 1D tensor [N] of blur-based scores.
            - labels: 1D tensor [N] (0 = real, 1 = fake).
            - paths: List of file paths (str).
            - datasets: List of dataset tags (str).
    """
    mean, std = _mean_std(device, torch.float32)
    allowed_set = None
    if allowed_datasets:
        allowed_set = {ds.lower() for ds in allowed_datasets}

    all_scores, all_labels, all_paths, all_dsets = [], [], [], []
    for x, y, paths, dsets in loader:
        # Filter batch by dataset if needed
        if allowed_set is not None:
            mask = [ds.lower() in allowed_set for ds in dsets]
            if not any(mask):
                continue
            idx = [i for i, keep in enumerate(mask) if keep]
            x = x[idx]
            y = y[idx]
            paths = [paths[i] for i in idx]
            dsets = [dsets[i] for i in idx]

        x = x.to(device, non_blocking=True)

        # Convert to [0,1] for blur/sharpen operations
        x_pix = denorm(x, mean, std)
        x_blur = gaussian_blur_pixel(x, sigma_blur, mean, std)

        x_sharp_pix = torch.clamp(2.0 * x_pix - denorm(x_blur, mean, std), 0.0, 1.0)
        x_sharp = renorm(x_sharp_pix, mean, std)

        e_blur  = embed_cls(model, x_blur)
        e_sharp = embed_cls(model, x_sharp)

        d = cosine_distance(e_blur, e_sharp)

        all_scores.append(d.cpu())
        all_labels.append(y)
        all_paths.extend(paths)
        all_dsets.extend(dsets)

    if not all_scores:
        raise RuntimeError(
            "No samples left after applying dataset filter "
            f"(allowed={allowed_datasets})."
        )

    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    return scores, labels, all_paths, all_dsets

def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute AUROC with sklearn.

    Args:
        scores (torch.Tensor): 1D tensor [N] of anomaly scores (higher = more likely fake).
        labels (torch.Tensor): 1D tensor [N] of binary labels (0 = real, 1 = fake).

    Returns:
        float: AUROC score.
    """
    return float(roc_auc_score(labels.numpy(), scores.numpy()))

def per_dataset_auroc(scores: torch.Tensor, labels: torch.Tensor, datasets: list[str]) -> dict:
    """
    Compute AUROC per dataset, handling single-class edge cases.

    If a dataset only contains a single class (all real or all fake),
    AUROC is undefined; we return (NaN, n_samples) in that case.

    Args:
        scores (torch.Tensor): 1D tensor [N] of scores.
        labels (torch.Tensor): 1D tensor [N] (0 or 1).
        datasets (list[str]): List of dataset names, one per sample.

    Returns:
        dict[str, tuple[float, int]]:
            Mapping from dataset name to (auroc, n_samples).
    """
    idx_by_ds = defaultdict(list)
    for i, ds in enumerate(datasets):
        idx_by_ds[ds].append(i)

    results = {}
    for ds, idxs in sorted(idx_by_ds.items()):
        s = scores[idxs]
        y = labels[idxs]
        uniques = set(y.tolist())
        if len(uniques) < 2:
            # Only one class present → AUROC not defined
            results[ds] = (float("nan"), len(idxs))
        else:
            results[ds] = (float(roc_auc_score(y.numpy(), s.numpy())), len(idxs))
    return results

def threshold_at_fpr(real_scores, target_fpr: float = 0.05) -> float:
    """
    Choose a threshold such that FPR ≈ target_fpr on REAL scores.

    Implementation:
      - Sort REAL scores ascending.
      - Keep (1 - target_fpr) quantile as threshold.
      - Under the assumption that fake scores are larger, this yields the
        desired false positive rate for a rule "score >= thr means fake".

    Args:
        real_scores (torch.Tensor | list[float] | numpy.ndarray): Scores for real images.
        target_fpr (float): Desired false positive rate (0..1).

    Returns:
        float: Threshold value on the score.
    """
    real_scores = torch.as_tensor(real_scores)
    if real_scores.numel() == 0:
        return float("nan")
    real_sorted = torch.sort(real_scores).values
    n = len(real_sorted)
    q = max(0.0, min(1.0, 1.0 - target_fpr))
    k = int(math.floor(q * (n - 1))) if n > 1 else 0
    return float(real_sorted[k])

# ------------------ CSV Export Helpers ------------------

def _save_csvs(results_dir: Path, paths, labels: torch.Tensor, scores: torch.Tensor, datasets, model_name_print: str, args, method_tag: str) -> None:
    """
    Write per-image and summary CSVs for a given method.

    This helper:
      - Resolves output filenames based on method_tag ('noise'|'blur'|'minder').
      - Writes per-image rows with columns: path, label, score, dataset.
      - Computes global and per-dataset AUROC and writes a summary CSV.
      - Uses pandas to write per-image and summary CSVs.

    Args:
        results_dir (Path): Root directory where CSVs will be written.
        paths (list[str]): List of image paths.
        labels (torch.Tensor): 1D tensor of labels (0 or 1).
        scores (torch.Tensor): 1D tensor of scores.
        datasets (list[str]): List of dataset tags.
        model_name_print (str): Human-readable model name for logging/CSV.
        args (argparse.Namespace): Parsed CLI arguments (store hyperparams in summary).
        method_tag (str): One of 'noise', 'blur', or 'minder'.

    Returns:
        None
    """
    if method_tag == "noise":
        out_scores = results_dir / "rigid_scores.csv"
        out_summary = results_dir / "rigid_summary.csv"
    elif method_tag == "blur":
        out_scores = results_dir / "blur_scores.csv"
        out_summary = results_dir / "blur_summary.csv"
    else:
        out_scores = results_dir / "minder_scores.csv"
        out_summary = results_dir / "minder_summary.csv"

    # Build per-image rows
    rows_scores = [{"path": p, "label": int(l), "score": float(s), "dataset": d}
                   for p, l, s, d in zip(paths, labels.tolist(), scores.tolist(), datasets)]

    # Global AUROC
    roc_global = auroc(scores, labels)
    rows_summary = [{
        "dataset": "GLOBAL",
        "n": int(len(labels)),
        "auroc": float(roc_global),
        "model": model_name_print,
        "sigma":      (float(args.sigma)      if method_tag in ("noise","minder") else ""),
        "sigma_blur": (float(args.sigma_blur) if method_tag in ("blur","minder")  else ""),
        "n_noise":    (int(args.n_noise)      if method_tag in ("noise","minder") else 1),
    }]

    # Per-dataset AUROCs
    per_ds = per_dataset_auroc(scores, labels, datasets)
    for ds, (roc, n) in per_ds.items():
        rows_summary.append({
            "dataset": ds,
            "n": int(n),
            "auroc": (float(roc) if not (roc != roc) else ""),
            "model": model_name_print,
            "sigma":      (float(args.sigma)      if method_tag in ("noise","minder") else ""),
            "sigma_blur": (float(args.sigma_blur) if method_tag in ("blur","minder")  else ""),
            "n_noise":    (int(args.n_noise)      if method_tag in ("noise","minder") else 1),
        })

    # Prefer pandas if available (nicer CSVs)
    pd.DataFrame(rows_scores).to_csv(out_scores, index=False)
    pd.DataFrame(rows_summary).to_csv(out_summary, index=False)

    print(f"[Saved] Per-image scores -> {out_scores}")
    print(f"[Saved] Summary AUROC    -> {out_summary}")

def _load_scores_csv(path: Path):
    """
    Load a scores CSV and validate that the required columns are present.

    Required columns:
        - 'path'
        - 'label'
        - 'score'
        - 'dataset'

    Args:
        path (Path): Path to the CSV file.

    Returns:
        pandas.DataFrame: DataFrame with the CSV contents.

    Raises:
        ValueError: If any required column is missing.
    """
    df = pd.read_csv(path)
    for col in ("path","label","score","dataset"):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path}")
    return df

# ------------------ Main (CLI) ------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the RIGID baseline script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    ap = argparse.ArgumentParser("RIGID baseline (timm, noise + contrastive blur + MINDER, per-dataset AUROC, CSV to results/)")

    # Data / model
    ap.add_argument("--data_root", default=None, type=str,
                    help="Folder with real/ and fake/ (required unless --minder_from_csv)")
    ap.add_argument("--model", default="dinov2-l14",
                type=str,
                help=(
                    "timm model or alias: "
                    "dinov2-s14|dinov2-l14|dinov2-g14, "
                    "dinov3-s16|dinov3-l16, "
                    "or any timm model name "
                    "(ex: vit_small_patch14_dinov2.lvd142m, vit_small_patch16_dinov3.lvd1689m)."
                ),)

    # Loader hyperparameters
    ap.add_argument("--batch_size", default=64, type=int)
    ap.add_argument("--workers", default=4, type=int)

    # Noise hyperparameters
    ap.add_argument("--sigma", default=0.009, 
                    type=float,
                    help="Gaussian noise std in pixel units [0,1] (try 0.008–0.010)"
                    )
    ap.add_argument("--n_noise", default=3, type=int,
                    help="How many noise samples to average")
    
    # Device
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    
    # Threshold calibration
    ap.add_argument("--calibrate_on_same_set", action="store_true",
                    help="Calibrate threshold@FPR=5% on the SAME REAL set (optimistic).")
    
    # Output directory
    ap.add_argument("--results_dir", default="results", type=str,
                    help="Directory to write CSV results (will be created if missing).")
    
    # Blur hyperparameters
    ap.add_argument("--sigma_blur", default=0.55, type=float,
                    help="Gaussian blur σ (fixed 3×3 kernel in pixel space; default 0.55).")    
    
    # Perturbation mode
    ap.add_argument("--perturb", default="both",
                choices=["noise", "blur", "both", "minder"],
                help=(
                    "Perturbation to use: "
                    "noise | blur | both (noise + contrastive blur + minder) | "
                    "minder (MINDER = min(noise, blur) distances)."
                ))
    
    # CSV-only MINDER mode
    ap.add_argument("--minder_from_csv", action="store_true",
                    help="Compute MINDER by merging existing rigid_scores.csv and blur_scores.csv (no recompute).")
    ap.add_argument("--rigid_csv", default="results/rigid_scores.csv", type=str)
    ap.add_argument("--blur_csv",  default="results/blur_scores.csv",  type=str)

    # Dataset selection (optional)
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "Optional list of dataset tags to include, e.g.: "
            "--datasets ADM CollabDiff. "
            "Tags are matched after normalization (ADM, CollabDiff, SID, UNKNOWN)."
        ),
    )

    return ap.parse_args()

def main() -> None:
    """
    Entry point for the RIGID / Blur / MINDER evaluation script.

    High-level flow:
      - Parse CLI arguments.
      - If --minder_from_csv:
            * Load noise & blur CSVs.
            * Merge on path, compute MINDER = min(noise, blur) per image.
            * Compute AUROCs and export CSVs.
        Else:
            * Build DataLoader from data_root (real/fake).
            * Create ViT backbone (DINOv2/v3).
            * Depending on --perturb:
                - 'noise': only RIGID (noise).
                - 'blur' : only contrastive blur.
                - 'minder': compute noise+blur then MINDER in-memory.
                - 'both'  : export all three: noise, blur, minder.
            * Optionally calibrate thr@FPR=5% on REAL scores.
            * Write scores + summary CSVs under results_dir.

    Returns:
        None
    """
    args = parse_args()

    # Prepare results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------- MINDER-from-CSV mode -------------------
    if args.minder_from_csv:
        # Load RIGID (noise) and Contrastive Blur scores
        rigid_df = _load_scores_csv(Path(args.rigid_csv)).rename(columns={"score": "score_noise"})
        blur_df  = _load_scores_csv(Path(args.blur_csv)).rename(columns={"score": "score_blur"})

        # Merge on 'path' to keep only common images
        df = rigid_df.merge(blur_df[["path", "score_blur"]], on="path", how="inner")
        if args.datasets:
            df = df[df["dataset"].isin(args.datasets)]
        if df.empty:
            raise SystemExit(
                "[Error] Merge produced 0 rows after merge/dataset filter. "
                f"Check --rigid_csv, --blur_csv and --datasets={args.datasets}."
            )

        # Build tensors for metrics/save helper
        paths    = df["path"].tolist()
        datasets = df["dataset"].tolist()   # dataset label from rigid_df
        labels   = torch.tensor(df["label"].values, dtype=torch.long)
        scores_n = torch.tensor(df["score_noise"].values, dtype=torch.float32)
        scores_b = torch.tensor(df["score_blur"].values,  dtype=torch.float32)

        # MINDER score = min(noise_score, blur_score) in distance space
        scores   = torch.minimum(scores_n, scores_b)

        model_name_print = "from-csv"

        # Print metrics
        roc_global = auroc(scores, labels)
        print(f"[MINDER/CSV] Global AUROC = {roc_global:.4f}")
        per_ds = per_dataset_auroc(scores, labels, datasets)
        print("[MINDER/CSV] Per-dataset AUROC:")
        for ds, (roc, n) in per_ds.items():
            roc_str = f"{roc:.4f}" if not (roc != roc) else "NaN"
            print(f"  - {ds:<16} AUROC = {roc_str}  |  n = {n}")

        # Save MINDER CSVs
        _save_csvs(results_dir, paths, labels, scores, datasets,
                   model_name_print, args, method_tag="minder")
        return

    # ------------------- Normal evaluation path -------------------

    # data_root is mandatory here
    if not args.data_root:
        raise SystemExit("Error: --data_root is required unless --minder_from_csv is set.")

    # Data (on-the-fly resize/crop -> 224)
    loader, _ = make_loader(args.data_root, batch_size=args.batch_size, workers=args.workers)

    # Model (timm backbone)
    model = create_backbone(args.model, device=args.device)
    model_name_print = NAME_MAP.get(args.model, args.model)
    print(f"[Info] Using timm model: {model_name_print} | Device: {args.device}")

    # Score all images according to --perturb
    method = args.perturb
    if method == "noise":
        # Noise-only RIGID
        scores, labels, paths, datasets = score_loader(
            model, loader, sigma=args.sigma, n_noise=args.n_noise, device=args.device, allowed_datasets=args.datasets
        )
        method_print = f"RIGID/Noise (sigma={args.sigma}, n_noise={args.n_noise})"

    elif method == "blur":
        # Contrastive blur-only
        scores, labels, paths, datasets = score_loader_blur(
            model, loader, sigma_blur=args.sigma_blur, device=args.device, allowed_datasets=args.datasets
        )
        method_print = f"Blur (sigma_blur={args.sigma_blur})"

    elif method == "minder":
        # Compute noise and blur scores (same alignment/order) then combine

        # Noise scores
        scores_n, labels, paths, datasets = score_loader(
            model, loader, sigma=args.sigma, n_noise=args.n_noise, device=args.device, allowed_datasets=args.datasets
        )
        # Blur scores
        scores_b, _, _, _ = score_loader_blur(
            model, loader, sigma_blur=args.sigma_blur, device=args.device, allowed_datasets=args.datasets
        )
        # MINDER = per-image min(distance) = max(similarity degradation)
        scores = torch.minimum(scores_n, scores_b)

        method_print = (
            f"MINDER (min(noise, blur) distances; "
            f"sigma={args.sigma}, n_noise={args.n_noise}, sigma_blur={args.sigma_blur})"
        )

        # Print metrics (global + per-dataset)
        roc_global = auroc(scores, labels)
        print(f"[RIGID] Global AUROC (MINDER) = {roc_global:.4f}")
        per_ds = per_dataset_auroc(scores, labels, datasets)
        print("[RIGID] Per-dataset AUROC (MINDER):")
        for ds, (roc, n) in per_ds.items():
            roc_str = f"{roc:.4f}" if not (roc != roc) else "NaN"
            print(f"  - {ds:<16} AUROC = {roc_str}  |  n = {n}")

        # Optional threshold on same set (REAL scores only) 
        if args.calibrate_on_same_set:
            thr = threshold_at_fpr(scores[labels == 0], target_fpr=0.05)
            y_pred = (scores >= thr).int()
            fpr = ((y_pred[labels == 0] == 1).float().mean().item()) if (labels == 0).any() else float("nan")
            tpr = ((y_pred[labels == 1] == 1).float().mean().item()) if (labels == 1).any() else float("nan")
            print(f"[RIGID/MINDER] thr@FPR=5% = {thr:.6f} | FPR={fpr:.3f} | TPR={tpr:.3f}")

        # Save MINDER CSVs and stop
        _save_csvs(
            results_dir, paths, labels, scores, datasets,
            model_name_print, args, method_tag="minder"
        )
        return  # done

    else:  # method == "both"
        # Run noise
        scores_n, labels, paths, datasets = score_loader(
            model, loader, sigma=args.sigma, n_noise=args.n_noise, device=args.device, allowed_datasets=args.datasets
        )
        # Run blur (re-use labels/paths/datasets from above)
        scores_b, _, _, _ = score_loader_blur(
            model, loader, sigma_blur=args.sigma_blur, device=args.device, allowed_datasets=args.datasets
        )

        # First: NOISE summaries + CSVs
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

        # Then: BLUR summaries + CSVs
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

        # Finally: MINDER = min(noise, blur) summaries + CSVs
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

    # Save CSVs for noise/blur
    method_tag = "noise" if method == "noise" else "blur"
    _save_csvs(
        results_dir, paths, labels, scores, datasets,
        model_name_print, args, method_tag=method_tag
    )

if __name__ == "__main__":
    main()
