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
