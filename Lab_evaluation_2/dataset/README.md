# DIV2K Dataset Specification & Preparation Guide

## 1. Dataset Overview

The **DIV2K (Diverse 2K Resolution Images)** dataset is a standard academic benchmark in high-resolution computer vision and image processing. In this project, the DIV2K high-resolution training set is evaluated as a large-scale real-world workload for parallel histogram computation on the GPU.

- **Dataset Source:** Computer Vision Lab, ETH Zurich
- **Official Download URL:** [DIV2K_train_HR.zip](https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip)
- **Image Count:** 800 high-resolution images (`0001.png` through `0800.png`)
- **Total Pixels:** 2,230,421,760 pixels
- **Total RGB Values:** 6,691,265,280 bytes (~6.23 GiB uncompressed binary)
- **Data Format:** Raw 8-bit unsigned integer (UINT8) RGB channels (values 0–255)

---

## 2. Dataset Preparation Workflow

Due to GitHub repository size limits, raw images and the 6.23 GiB binary are intentionally excluded via `.gitignore`. To reproduce the binary locally:

### Step 1: Download & Extract
Download the official archive and extract it so the images are located at:
```text
dataset/DIV2K/DIV2K_train_HR/
  0001.png
  0002.png
  ...
  0800.png
```

### Step 2: Binary Conversion
Run the preprocessing script using Python (requires `numpy` and `pillow`):
```cmd
python -m pip install numpy pillow
python scripts/prepare_div2k.py
```

This generates:
1. `dataset/div2k_rgb_u8.bin`: A contiguous raw binary stream of 6,691,265,280 bytes.
2. `dataset/div2k_manifest.csv`: Image metadata manifest (filename, width, height, channel count).

---

## 3. Streaming Benchmark Architecture

To process 6.69 billion values without exhausting GPU memory (VRAM), the CUDA benchmark (`div2k_histogram_benchmark.cu`):
- Memory-maps or streams the binary in **64 MiB contiguous chunks** (67,108,864 bytes).
- Uses 32-bit atomic counters for block/grid local aggregation.
- Accumulates chunk totals into **64-bit host accumulators** (`uint64_t`) to prevent counter overflow.

---

## 4. Citation and Attribution

```bibtex
@InProceedings{Agustsson_2017_CVPR_Workshops,
  author = {Agustsson, Eirikur and Timofte, Radu},
  title = {NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {July},
  year = {2017}
}
```
