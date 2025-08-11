# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Basic Blocks ----------

BN = nn.BatchNorm2d
ACT = nn.GELU

class MSLSB(nn.Module):
    """Multi-Scale Local Scope Block (简化版):
    三个 depthwise 卷积 (k=3/5/7, dilation=2) + concat + 1x1 fuse + 残差
    """
    def __init__(self, in_ch, out_ch=None):
        super().__init__()
        out_ch = out_ch or in_ch
        self.dw3 = nn.Conv2d(in_ch, in_ch, 3, padding=2, dilation=2, groups=in_ch, bias=False)
        self.dw5 = nn.Conv2d(in_ch, in_ch, 5, padding=4, dilation=2, groups=in_ch, bias=False)
        self.dw7 = nn.Conv2d(in_ch, in_ch, 7, padding=6, dilation=2, groups=in_ch, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch * 3, out_ch, 1, bias=False),
            BN(out_ch),
            ACT()
        )
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        f = torch.cat([self.dw3(x), self.dw5(x), self.dw7(x)], dim=1)
        y = self.fuse(f)
        return y + self.skip(x)

class EGSB(nn.Module):
    """Enhanced Global Scope Block (简化版):
    多尺度 AvgPool(3/5/7) + 每支 1x1 conv + concat + 1x1 fuse + MLP(1x1) 残差
    """
    def __init__(self, in_ch, out_ch=None, mlp_ratio=4.0):
        super().__init__()
        out_ch = out_ch or in_ch
        self.p3 = nn.AvgPool2d(3, stride=1, padding=1)
        self.p5 = nn.AvgPool2d(5, stride=1, padding=2)
        self.p7 = nn.AvgPool2d(7, stride=1, padding=3)
        self.c3 = nn.Conv2d(in_ch, in_ch, 1, bias=False)
        self.c5 = nn.Conv2d(in_ch, in_ch, 1, bias=False)
        self.c7 = nn.Conv2d(in_ch, in_ch, 1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch * 3, out_ch, 1, bias=False),
            BN(out_ch),
            ACT()
        )
        hidden = int(out_ch * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(out_ch, hidden, 1, bias=False),
            ACT(),
            nn.Conv2d(hidden, out_ch, 1, bias=False)
        )
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        idn = self.skip(x)
        f3 = self.c3(self.p3(x))
        f5 = self.c5(self.p5(x))
        f7 = self.c7(self.p7(x))
        y = self.fuse(torch.cat([f3, f5, f7], dim=1))
        y = y + self.mlp(y)
        return y + idn

# ---------- Optional Mamba (fallback to conv if not installed) ----------

def _try_import_mamba():
    try:
        from mamba_ssm.modules.mamba_simple import Mamba
        return Mamba
    except Exception:
        return None

_Mamba = _try_import_mamba()

class Mamba2D(nn.Module):
    """将 (B,C,H,W) 展平为序列做 Mamba；若无 mamba-ssm，则退化为 3x3 depthwise + 1x1。"""
    def __init__(self, ch):
        super().__init__()
        self.ch = ch
        if _Mamba is None:
            # fallback: depthwise separable conv
            self.fallback = nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),
                BN(ch),
                ACT(),
                nn.Conv2d(ch, ch, 1, bias=False)
            )
        else:
            self.proj = nn.Conv2d(ch, ch, 1, bias=False)
            self.mamba = _Mamba(d_model=ch)

    def forward(self, x):
        if _Mamba is None:
            return self.fallback(x) + x
        b, c, h, w = x.shape
        z = self.proj(x).flatten(2).transpose(1, 2)   # [B,HW,C]
        z = self.mamba(z)                              # [B,HW,C]
        z = z.transpose(1, 2).view(b, c, h, w)
        return z + x

# ---------- Simple Down/Up ----------

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            BN(out_ch),
            ACT()
        )
    def forward(self, x): return self.block(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            BN(out_ch),
            ACT()
        )
    def forward(self, x): return self.block(x)

# ---------- Minimal WNet2D (3-level) ----------

class WNet2D(nn.Module):
    """极简三层版：MS-LSB -> (Down+EGSB)×2 -> Deep Mamba -> Up ×2 -> Head"""
    def __init__(self, in_channels=3, num_classes=1, base_ch=32):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch*2, base_ch*4

        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1, bias=False), BN(c1), ACT(),
            MSLSB(c1, c1)
        )

        # encoder
        self.enc1_down = Down(c1, c2)     # 256->128
        self.enc1_gsb  = EGSB(c1, c1)

        self.enc2_down = Down(c2, c3)     # 128->64
        self.enc2_gsb  = EGSB(c2, c2)

        # deep global (Mamba / fallback conv)
        self.deep_mamba = Mamba2D(c3)

        # decoder
        self.dec2_up  = Up(c3, c2)
        self.dec2_fuse = nn.Conv2d(c2 + c2, c2, 1, bias=False)  # fuse with enc2_gsb

        self.dec1_up  = Up(c2, c1)
        self.dec1_fuse = nn.Conv2d(c1 + c1, c1, 1, bias=False)  # fuse with enc1_gsb

        # head
        self.head = nn.Conv2d(c1, num_classes, 1)

    def forward(self, x):
        s0 = self.stem(x)                  # [B,c1,512,512]

        e1 = self.enc1_down(s0)            # [B,c2,256,256]
        g1 = self.enc1_gsb(s0)             # [B,c1,512,512]

        e2 = self.enc2_down(e1)            # [B,c3,128,128]
        g2 = self.enc2_gsb(e1)             # [B,c2,256,256]

        d  = self.deep_mamba(e2)           # [B,c3,128,128]

        u2 = self.dec2_up(d)               # [B,c2,256,256]
        u2 = torch.cat([u2, g2], dim=1)    # 与 enc2 的 global 分支融合
        u2 = self.dec2_fuse(u2)

        u1 = self.dec1_up(u2)              # [B,c1,512,512]
        u1 = torch.cat([u1, g1], dim=1)    # 与 enc1 的 global 分支融合
        u1 = self.dec1_fuse(u1)

        logits = self.head(u1)             # [B,num_classes,512,512]
        return logits
