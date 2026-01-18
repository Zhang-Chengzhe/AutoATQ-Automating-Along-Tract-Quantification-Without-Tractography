# model.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================
# PointNet 编码器：逐点 MLP + mask-aware max pooling
# 输入: (B, N, in_dim)
# 输出: (B, feat_dim)
# =====================================================
class PointNetEncoder(nn.Module):
    def __init__(self, in_dim=3, feat_dim=1024, act=nn.ReLU):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),  act(True),
            nn.Linear(64, 128),     act(True),
            nn.Linear(128, feat_dim), act(True),
        )

    def forward(self, x, mask=None):  # x: (B,N,in_dim)
        h = self.mlp(x)  # (B,N,feat_dim)
        if mask is not None:
            m = mask.float().unsqueeze(-1)          # (B,N,1)
            h = h * m + (1 - m) * (-1e9)            # padding位置置为极小值以便max
        g = h.max(dim=1).values                      # (B,feat_dim)
        return g


# =====================================================
# 单头模型：输入坐标 → 输出 100 个中心线坐标
# 输入：
#   vox_xyz : (B, N, 3)
#   vox_mask: (B, N)  可选
# 输出：
#   P_pred  : (B, 100, 3)
# =====================================================
class CenterlinePointNetMLP(nn.Module):
    def __init__(self,
                 g_dim: int = 1024,
                 pos_dim: int = 128,
                 K: int = 100,
                 act=nn.ReLU):
        super().__init__()
        self.K = K

        # 编码器：仅用坐标 (in_dim=3)
        self.encoder = PointNetEncoder(in_dim=3, feat_dim=g_dim, act=act)

        # 位置编码：固定 100 个 t∈[0,1]，映射到 pos_emb
        self.pos_mlp = nn.Sequential(
            nn.Linear(1, 32), act(True),
            nn.Linear(32, pos_dim), act(True),
        )
        t = torch.linspace(0., 1., K)
        self.register_buffer('t_grid', t)  # (K,)

        # 解码干路（共享）
        dec_in_dim = g_dim + pos_dim
        self.dec_trunk = nn.Sequential(
            nn.Linear(dec_in_dim, 128), act(True),
            nn.Linear(128, 64),         act(True),
        )

        # 坐标回归头
        self.coord_head = nn.Linear(64, 3)

    def forward(self, vox_xyz, vox_mask=None):
        """
        返回:
          P_pred: (B,100,3)
        """
        # 仅坐标编码
        g = self.encoder(vox_xyz, vox_mask)                    # (B,g_dim)

        # 位置编码
        B = vox_xyz.size(0)
        t_emb = self.pos_mlp(self.t_grid.expand(B, self.K).unsqueeze(-1))  # (B,100,pos_dim)

        # 解码：拼接全局特征与位置嵌入
        g_rep = g.unsqueeze(1).expand(-1, self.K, -1)          # (B,100,g_dim)
        dec_in = torch.cat([g_rep, t_emb], dim=-1)             # (B,100,g+pos)
        h = self.dec_trunk(dec_in)                             # (B,100,64)

        # 输出 100×3 节点
        P_pred = self.coord_head(h)                            # (B,100,3)
        return P_pred


# =====================================================
#（可选）坐标损失：Smooth L1
# =====================================================
def loss_coord_smoothl1(a, b, beta: float = 0.5):
    """
    a, b: (B,100,3)
    """
    return F.smooth_l1_loss(a, b, beta=beta, reduction="mean")
