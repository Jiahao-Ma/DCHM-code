<div align="center">

# DCHM: Depth-Consistent Human Modeling for Multiview Detection

<a href=""><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://jiahao-ma.github.io/DCHM/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>


</div>

**Depth-Consistent Human Modeling** (DCHM) framework enhances multiview pedestrian detection by achieving annotation-free 3D human modeling through superpixel-wise Gaussian Splatting, outperforming existing methods  across challenging crowded scenarios.

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

- **GSplat** 
    
    We customize Gaussian Splatting (GSplat) for rendering, with the implementation available in `submodules/customized_gsplat/rendering.py`.

    ```bash
    pip install gsplat
    ```
- **Grounding-SAM**

    Follow the [official Grounding-SAM guide](https://github.com/IDEA-Research/Grounded-Segment-Anything).


### 4. Download the checkpoints
```bash
# TOOD: coming soon ...
```

## Inference

## Training


## Citation
If you find our code or paper useful for your research, please consider citing:
```bibtex
    # Coming soon ...
```
