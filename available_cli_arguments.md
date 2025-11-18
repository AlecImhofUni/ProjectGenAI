# Command-line parameters

Below is a summary of all available CLI arguments for `training_free_detect.py`.

## Core data & model parameters

- `--data_root <str>`  
  Path to the dataset root containing two subfolders:
  - `<data_root>/real/` — real images (label = 0)  
  - `<data_root>/fake/` — fake images (label = 1)  
  **Required**, unless you run in `--minder_from_csv` mode (CSV-only).

- `--model <str>` (default: `dinov2-l14`)  
  Backbone to use as feature extractor (timm model name or alias).  
  Supported aliases in the script:
  - `dinov2-b14`, `dinov2-l14`, `dinov2-g14`
  - `dinov3-s16`, `dinov3-l16`  
  You can also pass any valid timm ViT name (e.g. `vit_large_patch14_dinov2.lvd142m`).

- `--device <str>` (default: `cuda` if available, otherwise `cpu`)  
  Device on which to run the model:
  - `cuda` (GPU + CUDA)
  - `cpu`  
  Use `--device cpu` if you don’t have a GPU or CUDA.

## Dataloader parameters

- `--batch_size <int>` (default: `64`)  
  Batch size used by the PyTorch `DataLoader`.  
  If you get out-of-memory errors, reduce this value (e.g. `32` or `16`).

- `--workers <int>` (default: `4`)  
  Number of worker processes for the `DataLoader` (`num_workers`).  
  Increase to speed up data loading if your disk/CPU can handle it.

## Noise (RIGID) parameters

These control the **Noise / RIGID** perturbation:

- `--sigma <float>` (default: `0.009`)  
  Standard deviation of the **Gaussian noise in pixel space** `[0,1]`.  
  Typical range: `0.008–0.010`.  
  Higher values = stronger perturbation, larger distances.

- `--n_noise <int>` (default: `3`)  
  Number of independent noise samples per image.  
  For each image, the script:
  - applies noise `n_noise` times,
  - embeds each noisy version,
  - averages the cosine distance over these runs.  
  Larger values are more stable but slower.

These parameters are used when:
- `--perturb noise`
- `--perturb both`
- `--perturb minder`
- or `--minder_from_csv` combined with a noise CSV.

## Blur parameters

These control the **Blur** perturbation:

- `--sigma_blur <float>` (default: `0.55`)  
  Standard deviation of the Gaussian blur in **pixel units**, applied with a fixed 3×3 separable kernel in pixel space.  
  Used when:
  - `--perturb blur`
  - `--perturb both`
  - `--perturb minder`
  - or `--minder_from_csv` combined with a blur CSV.

## Perturbation mode (`--perturb`)

- `--perturb <str>` (default: `both`)  
  Chooses which perturbation(s) to run when evaluating from images:

  - `noise`  
    Run **Noise (RIGID)** only and save:
    - `rigid_scores.csv`, `rigid_summary.csv`.

  - `blur`  
    Run **Blur** only and save:
    - `blur_scores.csv`, `blur_summary.csv`.

  - `minder`  
    Compute **Noise + Blur** internally, then:
    - build **MINDER = min(noise_distance, blur_distance)** per image,
    - print AUROCs,
    - save only:
      - `minder_scores.csv`, `minder_summary.csv`.

  - `both`  
    Run **Noise**, **Blur**, and **MINDER** in one pass and save **all three**:
    - `rigid_scores.csv`, `rigid_summary.csv`
    - `blur_scores.csv`, `blur_summary.csv`
    - `minder_scores.csv`, `minder_summary.csv`

> Note: `--perturb` is ignored when you use `--minder_from_csv` (CSV-only mode).

## Threshold calibration

- `--calibrate_on_same_set` (flag, default: off)  
  If set, the script will:
  - compute a threshold at **FPR = 5%** on REAL scores of the current eval set,
  - print the threshold, FPR and TPR obtained on this same set.  

This is mainly for inspection / analysis; the threshold is **not** saved into the CSV (only printed).

## Results directory

- `--results_dir <str>` (default: `results`)  
  Directory where all CSV outputs are written.  
  The script will create it if it does not exist.

You will typically find:

```text
results/
  rigid_scores.csv
  rigid_summary.csv
  blur_scores.csv
  blur_summary.csv
  minder_scores.csv
  minder_summary.csv
