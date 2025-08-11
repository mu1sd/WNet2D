# WNet2D

**WNet2D** is a lightweight **dual-path** medical image segmentation model that integrates:
- **MS-LSB (Multi-Scale Local Scope Block):** dilated convolutions to enhance shallow edge/texture details.
- **E-GSB (Enhanced Global Scope Block):** parallel 3×3 / 5×5 / 7×7 average pooling branches followed by a 1×1 conv and a lightweight MLP (GELU) in a residual design to improve global semantic consistency.
- **Mamba SSM (deepest stage):** efficient and stable long-range dependency modeling with low compute.

> **Staged release.** This repository currently provides a **reference implementation** of WNet2D and a **minimal inference demo** for inspection and sanity checks. **Training/evaluation scripts and pretrained checkpoints will follow** (see *Roadmap*).  
> Paper repo: https://github.com/mu1sd/WNet2D

---

## Model Overview
<p align="center">
  <img src="architecture.png" alt="WNet2D Architecture" width="760">
</p>

---

## Repository Layout
WNet2D/
├─ README.md
├─ model.py # WNet2D with MS-LSB / E-GSB / Mamba SSM (reference)
├─ demo_infer.py # minimal forward pass demo (1×3×512×512)
└─ architecture.png # model diagram used in README/paper

---

## Environment

**Reference environment (from the paper):**  
- Hardware: NVIDIA GeForce RTX 4090 (22.15 GB VRAM), x86_64 CPU (8 cores), 15.57 GB RAM  
- OS: Ubuntu 22.04 (Linux 6.5.0-28)  
- Software: Python 3.9.23, PyTorch 2.1.2 (CUDA 11.8), cuDNN 8.7

**Quick install**
```bash
# optional virtual env
conda create -n wnet2d python=3.9 -y
conda activate wnet2d

# pytorch (choose the CUDA build matching your system)
pip install torch==2.1.2

# common deps (minimal)
pip install numpy opencv-python scikit-image scipy tqdm einops
# optional: Mamba SSM
# pip install mamba-ssm
Minimal Inference
bash

python demo_infer.py
# expected: Input (1,3,512,512) -> Output (1,1,512,512) or (1,num_classes,512,512)
(Optional) When checkpoints are available:

python

ckpt = torch.load("wnet2d_xxx.pth", map_location="cpu")
model.load_state_dict(ckpt, strict=False)
Datasets
The paper uses four public datasets: Kvasir-SEG, ISIC 2017, DRIVE, and PH2.
This repository does not redistribute raw data. Please obtain them from the official sources and follow their licenses/terms.

Reproducibility Protocol (Measurement)
To ensure consistency with the paper:

FLOPs are reported as 2×MACs for input 1×3×512×512.

Latency / Peak Memory are measured in model.eval() with batch size = 1, FP32.
Use 50 warm-up runs and report the average of 200 timed runs using CUDA events, with torch.cuda.synchronize() between runs, on a single RTX 4090.

Efficiency Summary (from the paper)
Parameters: ~4M (≈57% of nnWNet), fewer than most Conv/Hybrid baselines (e.g., BCU-Net, CMU-Net, UCTransNet).

FLOPs: ~30M per 512×512 image — 30% lower than nnWNet (43M), far below BCU-Net (454M) and TransAttUNet (356M).

Latency / Memory: nnU-Net shows the lowest latency (3.77 ms) and peak memory (0.82 GB) but lower segmentation accuracy; WNet2D achieves 14.19 ms latency and 2.54 GB peak memory with higher segmentation accuracy.

Conclusion. WNet2D provides strong parameter/compute efficiency with competitive latency/memory, suitable for resource-constrained deployments.

Roadmap
v1.0.0 (current): reference model + minimal inference + measurement protocol

v1.1.0 (planned): training/evaluation scripts and configs for Kvasir-SEG / ISIC 2017 / DRIVE / PH2

v1.2.0 (planned): pretrained checkpoints + reproducibility logs + FLOPs/latency scripts

If you need a particular script first, please open an issue and we will prioritize it.

Citation
If this repository is useful, please cite the paper (or temporarily cite the repo):

bibtex

@misc{WNet2D_Repo_2025,
  title        = {WNet2D: Enhanced Dual-Path Architecture with Multi-Scale LSB, Improved GSB, and Mamba SSM for Efficient Medical Image Segmentation},
  author       = {Li, Dianyuan and Xin, Yixuan and Li, Qinghua and Chao, Zhen},
  year         = {2025},
  note         = {Code repository, staged release},
  howpublished = {\url{https://github.com/mu1sd/WNet2D}}
}
(Replace with the IEEE Access entry once the paper is published.)

License & Disclaimer
Released under MIT (or Apache-2.0). See LICENSE.
Disclaimer: research use only; not for clinical decision-making.

Contact
Questions and issues: open a GitHub Issue or email xyxaidushu@163.com.

markdown

---

### Method
- *The minimal implementation of WNet2D is publicly available at* **https://github.com/mu1sd/WNet2D** *to facilitate inspection and reuse.*  


### Experimental Setup
-*Code (model + minimal inference) and the exact measurement protocol (FLOPs=2×MACs @ 1×3×512×512; latency/memory in eval, bs=1, FP32, 50 warm-ups, 200 runs with CUDA events and `torch.cuda.synchronize()`) are provided at* **https://github.com/mu1sd/WNet2D**.  


###  Code Availability
- *To foster transparency and reuse, we release WNet2D at* **https://github.com/mu1sd/WNet2D**. *Training/evaluation scripts and checkpoints will be added in a subsequent update.*  

---

