# ProjectGenAI
Project about Training-Free Detection of AI-Generated Images with Performance Modeling & Acceleration

## General description : 
AI-generated images are increasingly hard to distinguish by eye, and the political and safety consequences pose significant risks to trust in digital media, elections, and public discourse. This topic studies training-free detection of AI-generated images using vision foundation models to improve our ability to detect synthetic content.

**Paper 1.1:** Further improving training-free AI-generated image detection.

**Link:** [Understanding and Improving Training-Free AI-Generated Image Detections with Vision Foundation Models](https://arxiv.org/abs/2411.19117)

**How to reproduce :** You will be given a set of 1000 image pairs (real vs. fake). 
Reproduce the key experiments as follows:
1. Implement the baseline detector (RIGID) by extracting and computing the embedding similarity.
2. Recreate the Gaussian noise and Gaussian blur perturbation experiment.
3. Implement the MINDER approach from the paper by taking the minimum of similarities.

## Startup

```bash
# 1) Instalation of miniconda and activate environment
#instalation in $HOME
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

    # Activate conda in the shell
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash
exec $SHELL
    #Create and active the environment
conda create -y -n NAME python=3.12 gdown unzip rsync -c conda-forge
conda activate  NAME # any Python 3.10+ env

```

## Training-Free Detection with DINOv2 / DINOv3 (RIGID, Blur, MINDER)

### Basic usage of `training_free_detect.py`

Run 
```bash
    python training_free_detect.py --data_root /path/to/pairs_1000_eval
```
By default:
- model: `dinov2-l14`
- perturbation: `--perturb both` (Noise + Contrastive Blur + MINDER)
- results are saved under `./results/`

Find all available CLI arguments in `available_cli_arguments.md`

### Noise only (RIGID baseline)

 Run 
```bash
    python training_free_detect.py \
  --data_root /path/to/pairs_1000_eval \
  --perturb noise \
  --sigma 0.009 \
  --n_noise 3 \
  --device cuda
```

This will:
- run **Noise (RIGID)** only
- print global and per-dataset AUROC
- write:
    - `results/rigid_scores.csv`
    - `results/rigid_summary.csv`

### Constractive Blur only

Run 
```bash
    python training_free_detect.py \
  --data_root /path/to/pairs_1000_eval \
  --perturb blur \
  --sigma_blur 0.55 \
  --device cuda
```

This will:
- run **Contrastive Blur** only
- print global and per-dataset AUROC
- write:
    - `results/blur_scores.csv`
    - `results/blur_summary.csv`

### Noise + Blur + MINDER (all in one run)
Run 
```bash
    python training_free_detect.py \
  --data_root /path/to/pairs_1000_eval \
  --perturb both \
  --sigma 0.009 \
  --n_noise 3 \
  --sigma_blur 0.55 \
  --device cuda

```

This will:
- compute **Noise, Contrastive Blur, and MINDER = min(Noise, Blur)** scores
- print global and per-dataset for Noise, Blur, MINDER
- save:
    - `results/rigid_scores.csv` / `results/rigid_summary.csv`
    - `results/blur_scores.csv` / `results/blur_summary.csv`
    - `results/minder_scores.csv` / `minder_summary.csv`

### MINDER only (compute Noise + Contrastive Blur in memory)
Run 
```bash
    python training_free_detect.py \
  --data_root /path/to/pairs_1000_eval \
  --perturb minder \
  --sigma 0.009 \
  --n_noise 3 \
  --sigma_blur 0.55 \
  --device cuda
```

This will:
- compute Noise and Blur internally
- **only exports** MINDER CSVs:
    - `results/minder_scores.csv`
    - `minder_summary.csv`

### CSV-only mode (no recompute, --minder_from_csv)
Run 
```bash
    python training_free_detect.py \
  --minder_from_csv \
  --rigid_csv results/rigid_scores.csv \
  --blur_csv  results/blur_scores.csv

```

In this mode:
- --data_root and --model are ignored
- the script:
    - loads both CSVs
    - merges on path
    - computes MINDER = min(score_noise, score_blur)
    - prints AUROCs
    - saves `results/minder_scores.csv` and `results/minder_summary.csv`

### Patch-wise top-k scoring (optional)

By default, scores are computed from the CLS token:
- `--score_mode_noise` defaults to `cls`
- `--score_mode_blur` defaults to `cls`

So the commands above all use CLS-based scoring unless you explicitly switch to patch-wise scoring.

In addition to CLS-based scores, the script supports patch-wise top-k aggregation:
- `--score_mode_noise {cls, topk_patches}`
- `--score_mode_blur  {cls, topk_patches}`
- `--topk_patches_noise K` and `--topk_patches_blur K` set the value of \(k\).

**Example: Noise with patch-wise top-k (k = 8)**
Run
```bash
python training_free_detect.py \
  --data_root /path/to/pairs_1000_eval \
  --perturb noise \
  --sigma 0.009 \
  --n_noise 3 \
  --score_mode_noise topk_patches \
  --topk_patches_noise 8 \
  --device cuda
```

### Restricting to specific datasets (e.g. exclude SID)
All modes support an optional --datasets argument to restrict which datasets are included
in the evaluation. Dataset tags are inferred from filenames/paths and normalized to:
- ADM
- CollabDiff
- SID
- UNKNOWN
Example: run Noise + Contrastive Blur + MINDER on **ADM + CollabDiff only**

Run 
```bash
    python training_free_detect.py \
  --data_root /path/to/pairs_1000_eval \
  --perturb both \
  --sigma 0.009 \
  --n_noise 3 \
  --sigma_blur 0.55 \
  --datasets ADM CollabDiff \
  --device cuda
```

This will:
- drop any sample whose dataset tag is not in {ADM, CollabDiff},
- compute **Noise, Contrastive Blur, and MINDER = min(Noise, Blur)** scores
- print global and per-dataset for Noise, Blur, MINDER
- save:
    - `results/rigid_scores.csv` / `results/rigid_summary.csv`
    - `results/blur_scores.csv` / `results/blur_summary.csv`
    - `results/minder_scores.csv` / `minder_summary.csv`
