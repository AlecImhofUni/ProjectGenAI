## Collect Data
### ADM
Retrieve the various zip files located on Google Drive: (https://drive.google.com/drive/folders/1-9o163XaC-7L8Ch9-r7dLxbY_yjN8SVj)
Extract images:
``` bash
7z x /home/group-1/data/_adm_zip/imagenet_ai_0508_adm.zip -o/home/group-1/data/_adm_extracted_ai
```
Sort fake/real images:
```bash
AI_VAL=$(find /home/group-1/data/_adm_extracted_ai -type d -path "*/val/ai" -print -quit)
NATURE_VAL=$(find /home/group-1/data/_adm_extracted_ai -type d -path "*/val/nature" -print -quit)

echo "AI_VAL=$AI_VAL"
echo "NATURE_VAL=$NATURE_VAL"

mkdir -p /home/group-1/data/ADM/val/{fake,real}

# copy AI -> fake
rsync -a "$AI_VAL"/ /home/group-1/data/ADM/val/fake/

# copy NATURE -> real (if find in the zip)
if [ -n "$NATURE_VAL" ]; then
  rsync -a "$NATURE_VAL"/ /home/group-1/data/ADM/val/real/
fi

# control
echo "FAKE (ADM/val):" $(find /home/group-1/data/ADM/val/fake -type f | wc -l)
echo "REAL (ADM/val):" $(find /home/group-1/data/ADM/val/real -type f | wc -l)
```
### CollabDiff
Follows the instruction in the git of Collaborative Diffusion (CVPR 2023):
- Clone the repository:
```bash
git clone https://github.com/ziqihuangg/Collaborative-Diffusion.git

```
- Download the dataset (actual images) and checkpoints from Google Drive:
```bash
# Datasets (pre-processed) -> contains dataset/image/... = actual images
gdown --fuzzy --folder "https://drive.google.com/drive/folders/1rLcdN-VctJpW4k9AfSXWk0kqxh329xc4" -O dataset

# Checkpoints (for generating fakes)
gdown --fuzzy --folder "https://drive.google.com/drive/folders/13MdDea8eI8P4ygeIyfy8krlTb8Ty0mAP" -O pretrained

```
Creation of fake images (script): `gen_cd_fakes_gpu.sh`
```bash
bash ~/gen_cd_fakes_gpu.sh 500 8 10 "A portrait photo of a person."
```

### Create the set of 1000 image pairs (real vs. fake)
Using the script: `make_pairs.sh`
```bash
TAKE_ADM=500 TAKE_CD=500 bash ~/make_pairs.sh
```

### SID
Download 2000 images using the python file: `sid_download.py`
#### Create a set of 2000 image pairs (real vs. fake)
Using the script: `make_pairs_sid.sh`
```bash
TAKE_SID=1000 bash ~/make_pairs_sid.sh
```



