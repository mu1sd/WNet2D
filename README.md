WNet2D
WNet2D is a lightweight dual-path medical image segmentation model featuring:

MS-LSB (Multi-Scale Local Scope Block): Dilated convolutions for enhanced shallow edge and texture details.

E-GSB (Enhanced Global Scope Block): Parallel 3×3 / 5×5 / 7×7 average pooling branches, followed by a 1×1 convolution and a lightweight MLP (GELU), in a residual design to improve global semantic consistency.

Mamba SSM (deepest stage): Efficient and stable long-range dependency modeling with low computational cost.

Status: Minimal release — includes the model implementation and a simple inference demo. Training and full evaluation scripts may be released later.

📂 Project Structure

WNet2D/
├─ README.md
├─ model.py    
└─ demo_infer.py     
💻 Environment & Installation
Reference environment (from the paper):

Hardware: NVIDIA GeForce RTX 4090 (22.15 GB VRAM), x86_64 CPU (8 cores), 15.57 GB RAM

OS: Ubuntu 22.04 (Linux 6.5.0-28)

Software: Python 3.9.23, PyTorch 2.1.2 (CUDA 11.8), cuDNN 8.7

Quick install:

bash

# (Optional) create a virtual environment
conda create -n wnet2d python=3.9 -y
conda activate wnet2d

# Install PyTorch (select CUDA version matching your system from the official PyTorch site)
pip install torch==2.1.2

# Install common dependencies
pip install numpy opencv-python scikit-image scipy tqdm einops mamba-ssm
📊 Datasets
This paper uses four public datasets: Kvasir-SEG, ISIC 2017, DRIVE, and PH2.
We do not redistribute raw datasets. Please download them from their official sources and follow their respective licenses and terms of use.

📏 Reproducibility Protocol
To ensure consistency with the paper's results:

FLOPs: Calculated as 2 × MACs for an input of size 1×3×512×512.

Latency & Peak Memory: Measured in model.eval() mode with batch size = 1, FP32 precision. Perform 50 warm-up runs, then average over 200 timed runs using CUDA events, with torch.cuda.synchronize() between runs.

⚡ Efficiency Summary (from the paper)
As reported in Table 3 and Figure 7:

Parameters: ~4M (≈57% of nnWNet), fewer than most Conv/Hybrid baselines (e.g., BCU-Net, CMU-Net, UCTransNet).

FLOPs: ~30M per 512×512 image — 30% lower than nnWNet (43M), far below BCU-Net (454M) and TransAttUNet (356M).

Latency / Memory:

nnU-Net: Lowest latency (3.77 ms) and peak memory (0.82 GB) but lower segmentation accuracy.

WNet2D: Latency 14.19 ms, peak memory 2.54 GB, offering a balanced trade-off between speed, memory, and segmentation accuracy.

In summary, WNet2D combines strong parameter/compute efficiency with competitive latency and memory usage, making it highly suitable for resource-constrained medical image segmentation deployments.
