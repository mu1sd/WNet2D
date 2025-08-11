# -*- coding: utf-8 -*-
import time
import torch
from model import WNet2D

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WNet2D(in_channels=3, num_classes=1, base_ch=32).eval().to(device)

    # 1x3x512x512 输入（与论文一致）
    x = torch.randn(1, 3, 512, 512, device=device)

    # 前向一次，打印形状
    with torch.no_grad():
        y = model(x)
    print(f"Input: {tuple(x.shape)} -> Output: {tuple(y.shape)}")

    # 轻量计时（不耗时）：warmup=10，avg=50
    warmup, runs = 10, 50
    if device.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(runs):
            with torch.no_grad():
                _ = model(x)
        end.record()
        torch.cuda.synchronize()
        latency_ms = start.elapsed_time(end) / runs
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    else:
        t0 = time.perf_counter()
        for _ in range(runs):
            with torch.no_grad():
                _ = model(x)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0 / runs
        peak_gb = 0.0

    print(f"Avg latency: {latency_ms:.2f} ms  |  Peak memory: {peak_gb:.2f} GB  |  Device: {device}")

if __name__ == "__main__":
    main()
