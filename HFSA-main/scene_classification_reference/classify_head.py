"""改进的分类头模块 — 多尺度特征 + 空间注意力池化 + GeM + ASL Loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================== 池化模块 ==================

class GeMPool(nn.Module):
    """Generalized Mean Pooling: 在 Avg 和 Max 之间自适应.

    p=1 等价于 avg pooling, p→∞ 趋近 max pooling.
    可学习的 p 让模型自行决定对强激活信号的敏感程度.
    """
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # x: [B, C, H, W]
        p = self.p.clamp(min=1.0, max=10.0)
        x_p = x.clamp(min=self.eps).pow(p)
        x_avg = self.gap(x_p)
        return x_avg.pow(1.0 / p).flatten(1)  # [B, C]


class SpatialAttentionPool(nn.Module):
    """空间注意力池化: 学习每个空间位置对分类的重要性权重.

    替代全局平均池化 —— 不同类别关注不同区域。
    aircraft 关注跑道上的一小块, bridge 关注跨越水域的一长条。
    """
    def __init__(self, in_dim, reduction=4):
        super().__init__()
        hidden = max(in_dim // reduction, 32)
        self.attention = nn.Sequential(
            nn.Conv2d(in_dim, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, x):
        # x: [B, C, H, W]
        attn = self.attention(x)           # [B, 1, H, W]
        attn = attn.sigmoid()              # [B, 1, H, W]
        weighted = (x * attn).sum(dim=[2, 3])   # [B, C], 加权和
        norm = attn.sum(dim=[2, 3]).clamp_min(1e-6)  # [B, 1], 归一化因子
        return weighted / norm             # [B, C]


def conv_bn_relu(in_ch, out_ch, kernel=1):
    """1×1 卷积 + BN + ReLU 辅助函数."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ================== 多尺度分类头 ==================

class ClassifyHeadV2(nn.Module):
    """多尺度场景分类头.

    P3 [B, 256, 80, 80] → proj → attn_pool + gem → [B, 512]
    P4 [B, 512, 40, 40] → proj → attn_pool + gem → [B, 512]
    P5 [B, 512, 20, 20] → proj → attn_pool + gem → [B, 512]
                concat → [B, 1536] → MLP → [B, num_classes]
    """
    def __init__(self, num_classes=26, proj_dim=256, hidden_dim=512,
                 dropout=0.3, use_gem=True):
        super().__init__()
        self.use_gem = use_gem

        # 通道投影: 所有尺度 → proj_dim
        self.proj_p3 = conv_bn_relu(256, proj_dim)
        self.proj_p4 = conv_bn_relu(512, proj_dim)
        self.proj_p5 = conv_bn_relu(512, proj_dim)

        # 每种尺度独立的空间注意力
        self.attn_p3 = SpatialAttentionPool(proj_dim)
        self.attn_p4 = SpatialAttentionPool(proj_dim)
        self.attn_p5 = SpatialAttentionPool(proj_dim)

        # 每种尺度独立的 GeM 池化
        self.gem_p3 = GeMPool() if use_gem else None
        self.gem_p4 = GeMPool() if use_gem else None
        self.gem_p5 = GeMPool() if use_gem else None

        # 融合后的特征维度
        per_scale_dim = proj_dim * 2 if use_gem else proj_dim
        fusion_dim = per_scale_dim * 3  # 3 个尺度

        # 分类 MLP
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.7),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def _pool_scale(self, x, proj, attn, gem):
        """对单个尺度做投影 + 双路池化."""
        x = proj(x)
        parts = [attn(x)]
        if gem is not None:
            parts.append(gem(x))
        return torch.cat(parts, dim=1)

    def forward(self, features):
        """features: (p3, p4, p5) 三个特征图 tuple."""
        p3, p4, p5 = features

        f3 = self._pool_scale(p3, self.proj_p3, self.attn_p3, self.gem_p3)
        f4 = self._pool_scale(p4, self.proj_p4, self.attn_p4, self.gem_p4)
        f5 = self._pool_scale(p5, self.proj_p5, self.attn_p5, self.gem_p5)

        fused = torch.cat([f3, f4, f5], dim=1)  # [B, 1536]
        return self.classifier(fused)


# ================== Asymmetric Loss ==================

class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification.

    专为极端类别不均衡设计的 BCE 变体:
    - gamma_pos=1: 对正样本做轻度聚焦 (罕见类正样本本来就少, 不压制)
    - gamma_neg=4: 对负样本做强聚焦 (大量 easy negatives 被压制)
    - clip=0.05: 概率偏移, 给概率适中的负样本更大容错空间
    """
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        p_m = (p - self.clip).clamp(min=0.0)  # 概率偏移

        pos = -targets * (1 - p).pow(self.gamma_pos) * torch.log(p.clamp(min=self.eps))
        neg = -(1 - targets) * p_m.pow(self.gamma_neg) * torch.log((1 - p_m).clamp(min=self.eps))

        return (pos + neg).sum(dim=1).mean()
