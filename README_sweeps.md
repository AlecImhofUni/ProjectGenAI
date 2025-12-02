# Choice of parameters for DINOv3

## Sigma sweep scripts
### Noise $\sigma$ sweep: sweep_sigma_dinov3.sh

Sweep Gaussian noise σ for the RIGID baseline and log AUROC.

```bash
SIGMAS=(0.004 0.006 0.008 0.010 0.012 0.014 0.016 0.018 0.020 0.022 0.024 0.026 0.028 0.030 0.032 0.034 0.036 0.038 0.040)
```

Run:
```bash
    sweep_sigma_dinov3.sh
```

Once the script is complete, you get:
- one folder per value of $\sigma$ (e.g. `sigma_0.010/`)
- inside each folder:
    - per-image scores and summary CSVs for that $\sigma$
- two aggregate CSVs under results_sigma_dinov3/:
    - `sigma_sweep_global.csv` (GLOBAL AUROC per $\sigma$)
    - `sigma_sweep_per_dataset.csv` (AUROC per dataset per $\sigma$)

### Blur $\sigma_{blur}$ sweep: sweep_sigma_blur_dinov3.sh

Sweep Gaussian blur σ_blur for the Contrastive Blur baseline and log AUROC.

```bash
BLURS=(0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00)
```

Run:
```bash
    sweep_sigma_blur_dinov3.sh
```

Once the script is complete, you get:
- one folder per value of $\sigma_{\text{blur}}$ (e.g. `sigma_blur_0.55/`)
- inside each folder:
    - per-image scores and summary CSVs for that $\sigma_{blur}$
- two aggregate CSVs under results_sigma_blur_dinov3/:
    - `sigma_blur_sweep_global.csv` (GLOBAL AUROC per $\sigma_{blur}$)
    - `sigma_blur_sweep_per_dataset.csv` (AUROC per dataset per $\sigma_{blur}$)


### MINDER grid sweep: sweep_minder_dinov3.sh

Explore all combinations ($\sigma$, $\sigma_{blur}$) for MINDER by reusing Noise/Blur scores from the two sweeps (no recomputation).

Assumptions:
- `results_sigma_dinov3/sigma_*/rigid_scores.csv` already exist (from `sweep_sigma_dinov3.sh`).
- `results_sigma_blur_dinov3/sigma_blur_*/blur_scores.csv` already exist (from `sweep_sigma_blur_dinov3.sh`).

Run:
```bash 
    sweep_minder_dinov3.sh
```

Once the script is complete, you get:
- one folder per $(\sigma, \sigma_{\text{blur}})$ combination, e.g. `minder_sigma_0.010__sblur_0.55/`
    - containing `minder_scores.csv` + `minder_summary.csv`
- two aggregate CSVs under `results_minder_dinov3/`:
    - `minder_sweep_global.csv` (GLOBAL AUROC per pair $(\sigma, \sigma_{\text{blur}})$)
    - `minder_sweep_per_dataset.csv` (AUROC per dataset per pair)

## Global AUROC without SID

These three scripts recompute global AUROC after removing SID samples, by going back to per-image `*_scores.csv` files.

### `build_noise_global_no_sid.sh`

Input:
- `results_sigma_dinov3/sigma_sweep_global.csv`
- all `results_sigma_dinov3/sigma_*/rigid_scores.csv`

Output:
- `results_sigma_dinov3/sigma_sweep_global_no_sid.csv`

For each row in the original global CSV, it:
- reads the `results_dir` for that $\sigma$ (e.g. `sigma_0.010/`)
- loads `rigid_scores.csv` (per-image scores)
- filters out all rows with `dataset == "SID"`
- recomputes global AUROC on the remaining samples
It writes a new wide CSV with AUROC without SID.

### `build_blur_global_no_sid.sh`

Input:
- `results_sigma_blur_dinov3/sigma_blur_sweep_global.csv`
- all `results_sigma_blur_dinov3/sigma_blur_*/blur_scores.csv`

Output:
- `results_sigma_blur_dinov3/sigma_blur_sweep_global_no_sid.csv`

Same logic as above, but:
- reads `blur_scores.csv`
- recomputes AUROC after filtering out SID
- stores results in:
    - `global_auroc_wo_sid`
    - `total_n` (number of non-SID samples used)

### MINDER: `build_minder_global_no_sid.sh`

Input:
- `results_minder_dinov3/minder_sweep_global.csv`
- all `results_minder_dinov3/minder_sigma_*__sblur_*/minder_scores.csv`

Output:
- `results_minder_dinov3/minder_sweep_global_no_sid.csv`

For each $(\sigma, \sigma_{\text{blur}})$ combination, it:
- reads `minder_scores.csv` in the corresponding `results_dir`
- filters out SID rows
- recomputes global AUROC
- stores results in new columns:
    - `total_n_wo_sid`
    - `global_auroc_wo_sid`



## Table + Top-10 builder: make_table_and_top10.py

Build a wide table indexed by $(\sigma, \sigma_{\text{blur}})$ with:
- global AUROC
- optional `global_auroc_wo_sid`
- AUROC per dataset (`ADM`, `CollabDiff`, `SID`)
- plus a second CSV with the union of Top-10 combinations across metrics.

### Typical usage
```bash
    python make_table_and_top10.py \
    --global-file results_minder_dinov3/minder_sweep_global_no_sid.csv \
    --dataset-file results_minder_dinov3/minder_sweep_per_dataset.csv \
    --out-dir out_tables
```

### What it does

The script creates two CSV tables:
1. out_tables/table_by_sigma.csv
    - One row per pair $(\sigma, \sigma_{\text{blur}})$.
    - Columns:
        - `sigma`, `sigma_blur`
        - `global_auroc` (and, if available, `global_auroc_wo_sid`)
        - one column per dataset, e.g. `ADM AUROC`, `CollabDiff AUROC`, `SID AUROC`
2. out_tables/top_union_table_by_sigma.csv
    - Keeps only the Top-10 configurations for each metric:
        - `global_auroc`
        - `global_auroc_wo_sid` (if present)
        - each per-dataset AUROC (`ADM AUROC`, `CollabDiff AUROC`, `SID AUROC`)
    - Takes the union of those rows.
    - Adds a `top10_for` column listing which metric(s) selected each row (e.g. `global_wo_sid; ADM`).

This table is what we use to select the most promising combinations of $(\sigma, \sigma_{\text{blur}})$.
