"""
LSD-HybridViT (Gou et al., IEEE Access 2025) re-implemented on timm's MobileViT-S,
plus our enhancements:
  * ImageNet-pretrained backbone          (--pretrained)
  * Ordinal (CORAL-style) classification  (--loss ordinal)
  * Monte-Carlo dropout uncertainty       (evaluate.py --mc-samples)

Base-paper components:
  LMA  - Lightweight Mixed-domain Attention (paper Sec. III-D, Eqs. 6-10), placed
         inside four inverted residual blocks (IRB+A).
  JDPM - Frequency multi-scale dilated conv block (paper Sec. III-C, Eqs. 1-5), used
         as the local representation of every MobileViT block (EMB).
"""
import math
import torch
import torch.nn as nn
import timm


# --------------------------------------------------------------------------------------
# LMA: Lightweight Mixed-domain Attention
# --------------------------------------------------------------------------------------
def adaptive_kernel(channels, lam=1.5, gamma=1.0):
    """Eq. (8): K = | log2(C)/lambda - gamma/lambda |_odd"""
    k = int(abs(math.log2(channels) / lam - gamma / lam))
    return max(k if k % 2 else k + 1, 3)


class LMA(nn.Module):
    def __init__(self, channels, kernel_size=None):
        super().__init__()
        k = kernel_size or adaptive_kernel(channels)
        self.k = k
        # 1D convs across the channel axis, one for each direction (Eq. 9)
        self.conv_h = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)
        self.conv_w = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        d_h = x.mean(dim=3)                      # Eq. (6): (B, C, H) pooled along W
        d_w = x.mean(dim=2)                      # Eq. (7): (B, C, W) pooled along H
        # permute so the conv slides over channels at each row / column
        a_h = self.conv_h(d_h.permute(0, 2, 1).reshape(b * h, 1, c))
        a_w = self.conv_w(d_w.permute(0, 2, 1).reshape(b * w, 1, c))
        a_h = torch.sigmoid(a_h).reshape(b, h, c).permute(0, 2, 1).unsqueeze(3)   # (B,C,H,1)
        a_w = torch.sigmoid(a_w).reshape(b, w, c).permute(0, 2, 1).unsqueeze(2)   # (B,C,1,W)
        return x * a_h * a_w                     # Eq. (10)


# --------------------------------------------------------------------------------------
# JDPM: frequency multi-scale dilated convolution block
# --------------------------------------------------------------------------------------
def conv_bn_act(cin, cout, k=1, dilation=1, act=True):
    layers = [nn.Conv2d(cin, cout, k, padding=dilation * (k // 2), dilation=dilation, bias=False),
              nn.BatchNorm2d(cout)]
    if act:
        layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)


class FrequencyFilter(nn.Module):
    """Eq. (1): J^f = | IFFT( sigma(FFT(J^s)) * FFT(J^s) ) |
    sigma = conv-BN-ReLU-conv-sigmoid applied to the amplitude spectrum, giving a
    learned per-frequency gate that suppresses redundant frequencies."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x):
        h, w = x.shape[-2:]
        with torch.autocast(device_type=x.device.type, enabled=False):   # FFT needs fp32
            xf = x.float()
            spec = torch.fft.rfft2(xf, norm="ortho")
            weight = self.gate(spec.abs())
            out = torch.fft.irfft2(spec * weight, s=(h, w), norm="ortho").abs()
        return out.to(x.dtype)


class JDPM(nn.Module):
    """Four cascaded branches with dilation 3,5,7,9 (Fig. 3). Each branch receives the
    reduced input plus the previous branch's output, extracts spatial features J^s with a
    dilated 3x3 conv, adds frequency features J^f (Eq. 4), and the concatenated result is
    fused with a residual connection (Eq. 5)."""
    def __init__(self, channels, dilations=(3, 5, 7, 9)):
        super().__init__()
        mid = max(channels // 2, 16)
        self.reduce = conv_bn_act(channels, mid, 1)
        self.branch_in = nn.ModuleList([conv_bn_act(mid, mid, 1) for _ in dilations])
        self.dilated = nn.ModuleList([conv_bn_act(mid, mid, 3, dilation=d) for d in dilations])
        self.freq = nn.ModuleList([FrequencyFilter(mid) for _ in dilations])
        self.merge = conv_bn_act(mid * len(dilations), channels, 1, act=False)
        self.out = nn.Sequential(conv_bn_act(channels, channels, 3), conv_bn_act(channels, channels, 1))

    def forward(self, x):
        r = self.reduce(x)
        feats, prev = [], 0
        for bin_, dil, freq in zip(self.branch_in, self.dilated, self.freq):
            js = dil(bin_(r + prev))
            jn = js + freq(js)                       # Eq. (4)
            feats.append(jn); prev = jn
        return self.out(self.merge(torch.cat(feats, 1)) + x)   # Eq. (5)


# --------------------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------------------
# Inverted residual blocks that receive LMA ("IRB+A"): (stage, block index)
LMA_POSITIONS = [(1, 2), (2, 0), (3, 0), (4, 0)]


class LSDHybridViT(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, use_lma=True, use_jdpm=True,
                 head="ce", dropout=0.3, lma_kernel=None, backbone="mobilevit_s"):
        super().__init__()
        assert head in ("ce", "ordinal")
        self.head_type, self.num_classes = head, num_classes
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="")

        if use_lma:
            for s, b in LMA_POSITIONS:
                blk = self.backbone.stages[s][b]
                mid = blk.conv2_kxk.conv.out_channels
                blk.attn = LMA(mid, lma_kernel)
        if use_jdpm:
            for stage in self.backbone.stages:
                for blk in stage:
                    if type(blk).__name__ == "MobileVitBlock":
                        c = blk.conv_kxk.conv.in_channels
                        blk.conv_kxk = JDPM(c)

        c = self.backbone.num_features
        self.cam_layer = nn.Identity()        # Grad-CAM hooks here (final feature map)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)    # kept active at test time for MC dropout
        if head == "ce":
            self.fc = nn.Linear(c, num_classes)
        else:
            # CORAL: one shared weight vector + (K-1) thresholds -> rank-consistent
            self.fc = nn.Linear(c, 1, bias=False)
            self.thresholds = nn.Parameter(torch.linspace(1.0, -1.0, num_classes - 1))

    def forward(self, x):
        f = self.cam_layer(self.backbone(x))
        z = self.dropout(torch.flatten(self.pool(f), 1))
        if self.head_type == "ce":
            return self.fc(z)
        return self.fc(z) + self.thresholds        # (B, K-1) logits for P(y > k)


def build_model(pretrained=True, use_lma=True, use_jdpm=True, head="ce", dropout=0.3,
                lma_kernel=None, num_classes=5):
    return LSDHybridViT(num_classes, pretrained, use_lma, use_jdpm, head, dropout, lma_kernel)


# --------------------------------------------------------------------------------------
# Output helpers shared by train / evaluate / gradcam
# --------------------------------------------------------------------------------------
def ordinal_targets(y, num_classes=5):
    """grade y -> [y>0, y>1, ..., y>K-2] as floats."""
    ks = torch.arange(num_classes - 1, device=y.device)
    return (y.unsqueeze(1) > ks).float()


def outputs_to_probs(out, head):
    """Class probabilities (B, K) from model outputs for either head type."""
    if head == "ce":
        return out.softmax(1)
    s = torch.sigmoid(out)                                   # P(y > k)
    ones = torch.ones_like(s[:, :1]); zeros = torch.zeros_like(s[:, :1])
    cum = torch.cat([ones, s, zeros], 1)
    p = (cum[:, :-1] - cum[:, 1:]).clamp(min=1e-8)
    return p / p.sum(1, keepdim=True)


def outputs_to_pred(out, head):
    if head == "ce":
        return out.argmax(1)
    return (torch.sigmoid(out) > 0.5).sum(1)                 # number of thresholds passed


def cam_score(out, head, cls):
    """Scalar that Grad-CAM differentiates: the predicted class logit (CE head) or, for the
    ordinal head, the expected grade sum_k P(y>k), so the heatmap shows the regions that
    increase predicted severity (lesions)."""
    if head == "ce":
        return out[0, cls]
    return torch.sigmoid(out[0]).sum()


def enable_mc_dropout(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
