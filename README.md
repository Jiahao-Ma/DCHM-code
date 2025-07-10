<div align="center">

# DCHM: Depth-Consistent Human Modeling for Multiview Detection

<a href=""><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://jiahao-ma.github.io/DCHM/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>


</div>

**Depth-Consistent Human Modeling** (DCHM) framework enhances multiview pedestrian detection by achieving annotation-free 3D human modeling through superpixel-wise Gaussian Splatting, outperforming existing methods  across challenging crowded scenarios. Note: The code is still under release.

***Check our [website](https://jiahao-ma.github.io/DCHM/) for videos and reconstruction results!***

## TODO List
- [x] Release source code of the inference and training.
- [ ] Release the spatially consistent pseudo-deph label.
- [ ] Release the checkpoint of model.

## Installation Guide
Follow these steps to set up the **DCHM** codebase on your system.



### 1. Clone this repository
```bash
git clone https://github.com/Jiahao-Ma/DCHM-code
cd DCHM-code
```

### 2.  Create conda environment

```bash
conda create -n DCHM python=3.10
pip3 install torch torchvision torchaudio # use the correct version of cuda for your system
```

### 3. Install necessary libraries

- **Grounding-SAM**

    ```bash
    # Clone the Grounded-SAM and install dependency
    mkdir submodules && cd submodules
    git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
    cd Grounded-Segment-Anything
    python -m pip install -e segment_anything
    pip install --no-build-isolation -e GroundingDINO
    pip install --upgrade "diffusers[torch]"
    git submodule update --init --recursive
    cd grounded-sam-osx && bash install.sh
    git clone https://github.com/xinyu1205/recognize-anything.git
    pip install -r ./recognize-anything/requirements.txt
    pip install -e ./recognize-anything/
    
    # Download the pre-trained model
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

    ```

- **GSplat** 
    
    We customize Gaussian Splatting (GSplat) for rendering, with the implementation available in `submodules/customized_gsplat/rendering.py`.

    ```bash
    pip install gsplat
    ```


### 4. Download the checkpoints and our pseudo-depth label
```bash
# The code and data are coming soon ...
```

## Inference
### Supervised-version 
```bash
python supervised_cluster.py \
    --root, 'path\to\wildtrack_data_gt', \
    --round, 2_1, \
    --n-segments, 30, \
    --start-with, 0, \
    --end-with, -1, \
    --fun_type 'Inference'
```
### Unsupervised-version 
```bash
python unsupervised_cluster.py \
    --root, 'path\to\wildtrack_data_gt', \
    --round, 2_1, \
    --n-segments, 30, \
    --start-with, 0, \
    --end-with, 10, \
    --min_gs_threshold, 10
```

## Training - Stage1. Generating Pseudo-depth Label

### First Round Training
We provide the initial training pipeline as follows. For the complete iterative process, please refer to [`pipeline.sh`](./pipeline.sh).

### Alternative Option
If you prefer to skip the training process for generating consistent pseudo-depth labels:  
⬇️ [Download pre-generated labels]() *(Coming soon)*  

> **Note**: The pseudo-depth labels are currently in preparation and will be available shortly. We recommend checking back later or following our [release updates](https://jiahao-ma.github.io/DCHM/).

---
## Pipeline Implementation

### 1. Preprocessing
```bash
# Step 1: Extract video frames
DATA_DIR=/path/to/wildtrack_data_gt
python s01_data_download.py \
    --data-dir ${DATA_DIR} \
    --duration 35 \
    --fps 2 \
    --output-folder Image_subsets
```

### 2. Foreground/Background Segmentation

```bash
# Step2: segment the foreground and background using grounded-sam
mkdir submodules && cd submodules
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything
# Step2.1: Generate the foreground (human) mask
TARGET1="people"
python s02_sam2_wgt.py \
        --root ${DATA_DIR} \
        --config submodules/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
        --grounded_checkpoint submodules/Grounded-Segment-Anything/groundingdino_swint_ogc.pth \
        --sam_checkpoint submodules/Grounded-Segment-Anything/sam_vit_h_4b8939.pth \
        --input_image None \
        --output_dir ${DATA_DIR}/masks/${TARGET1} \
        --box_threshold 0.3 \
        --text_threshold 0.25 \
        --text_prompt ${TARGET1} \
        --device "cuda:0"

python s02_sam2.py \
        --root ${DATA_DIR} \
        --config submodules/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
        --grounded_checkpoint submodules/Grounded-Segment-Anything/groundingdino_swint_ogc.pth \
        --sam_checkpoint submodules/Grounded-Segment-Anything/sam_vit_h_4b8939.pth \
        --input_image None \
        --output_dir ${DATA_DIR}/masks/${TARGET1} \
        --box_threshold 0.3 \
        --text_threshold 0.25 \
        --text_prompt ${TARGET1} \
        --device "cuda:0"

# Step2.2: Generate the background (ground) mask
TARGET2="ground"
python s02_sam2.py \
        --root ${DATA_DIR} \
        --config submodules/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
        --grounded_checkpoint submodules/Grounded-Segment-Anything/groundingdino_swint_ogc.pth \
        --sam_checkpoint submodules/Grounded-Segment-Anything/sam_vit_h_4b8939.pth \
        --input_image None \
        --output_dir ${DATA_DIR}/masks/${TARGET2} \
        --box_threshold 0.3 \
        --text_threshold 0.25 \
        --text_prompt ${TARGET2} \
        --device "cuda:0"
```
### 3. Gaussian Splatting Initialization
```bash
# Step3: Initialization for GS. Generate super-pixel for the foreground
python s03_super_pixel.py \
    --root ${DATA_DIR} \
    --n_segments "30, 60"
```

### 4. 1st Round - Generate Pseudo-Depth Label 
```bash
# --- Iterative Matching Label --- #
# --- First Rround --- #
# Step4: per frame training for GS
python s04_gs_us.py --root ${DATA_DIR} --round 1_1 --n-segments 30 --start-with 1 --end-with -1
```
### 5. Fine Tuning Mono-Depth
```bash
# Step5: fine-tuning mono-depth estimation
python s05_depth_finetune.py \
        --epochs 200 \
        --encoder vits \
        --bs 1 \
        --lr 0.000005 \
        --save-path "output/1_1_w_dc_200" \
        --dataset wildtrack \
        --img-size 518 \
        --min-depth 0.001 \
        --max-depth 40 \
        --pretrained-from checkpoints/depth_anything_v2_vits.pth \
        --port 20596 \
        --data_root ${DATA_DIR} \
        --round 1_1

```
> **Note**: Please refer to the [`pipeline.sh`](./pipeline.sh) for the all iterative training process. The iterative training process design for achieving consistent pseudo-depth label and fining the mono-depth label.

## Training - Stage2. Fine Tuning on Downstream Task 
The fine-tuning is only for supervised method only.
```bash
python supervised_cluster.py \
    --root, 'path\to\wildtrack_data_gt', \
    --round, 2_1, \
    --n-segments, 30, \
    --start-with, 0, \
    --end-with, -1, \
    --fun_type 'Train'
```


## Evaluate
### Supervised evaluation
The trained checkpoint of supervised deocoder will be released soon.
```bash
python supervised_evaluate.py \
    --pr_dir_pred "output/exp_sup/pr_dir_pred.txt" \
    --pr_dir_gt "output/exp_sup/pr_dir_gt.txt" \
    --sup_decoder_checkpoint "path/to/checkpoint.pth"
```

### Unsupervised evaluation
The trained GS and the depth prediction model checkpoints will be released soon.
```bash
python unsupervised_evaluate.py \
    --root 'path/to/wildtrack_data_gt' \
    --round 1_2 \
    --n-segments 30 \
    --start-with 0 \
    --end-with 10 \
    --min_gs_threshold 10
```


## Citation
If you find our code or paper useful for your research, please consider citing:
```bibtex
    # Coming soon ...
```
