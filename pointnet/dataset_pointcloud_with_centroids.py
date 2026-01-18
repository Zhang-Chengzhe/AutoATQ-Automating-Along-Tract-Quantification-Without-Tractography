# dataloader.py
# -*- coding: utf-8 -*-
"""
读取 fold_root/<subject>/Pointdata/<bundle>/ 结构下的数据：
  - bundle_points.npz:  points (N, Dv) -> [x,y,z, 体素属性...]
  - centroids_init.npz: centroids_init (100, Dn) -> [x,y,z, 节点属性...]

支持：
  1) 返回体素与节点的坐标 + 属性
  2) 可变 N 的 batch padding，并提供 vox_mask
  3) Dataset 内部的简单随机抽样（SRS）：每次迭代为同一束抽不同子集
  4) 可选几何增强 transform：对 vox_xyz 与 node_xyz 同变换
  5) __main__ 测试入口
"""

from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# =========================
# Dataset
# =========================
class BundleCenterlineDataset(Dataset):
    """
    读取一个“束样本”：(vox_xyz, vox_feat, node_xyz, node_feat, subject, bundle)

    返回（均为 float32 张量，除 subject/bundle 为 str）：
      vox_xyz   : (N, 3)
      vox_feat  : (N, C_v)   —— 若无属性则 (N, 0)
      node_xyz  : (100, 3)
      node_feat : (100, C_n) —— 若无属性则 (100, 0)
      subject   : str
      bundle    : str
    """
    def __init__(
        self,
        fold_root: str,
        subjects: Optional[List[str]] = None,
        bundles: Optional[List[str]] = None,
        min_points: int = 8,
        strict_100: bool = True,
        # --- 新增：增强 & 子集采样 ---
        transform=None,                      # 几何增强：对 (vox_xyz, node_xyz) 同变换
        sample_points: Optional[int] = None, # 简单随机抽样的目标点数 M（优先级高于 ratio）
        sample_ratio: Optional[float] = None,# 简单随机抽样的比例（0~1）
        sample_with_replacement: bool = False, # 当 M>N 是否放回补足（默认不放回，直接用全体点）
    ):
        self.fold_root = Path(fold_root)
        if not self.fold_root.exists():
            raise FileNotFoundError(f"fold_root not found: {self.fold_root}")

        # 扫描 subjects
        all_subjects = sorted([p.name for p in self.fold_root.iterdir() if p.is_dir()])
        self.subjects = subjects if subjects is not None else all_subjects
        missing = set(self.subjects) - set(all_subjects)
        if missing:
            raise FileNotFoundError(f"Subjects not found: {sorted(list(missing))}")

        self.bundles_filter = set(bundles) if bundles is not None else None
        self.min_points = int(min_points)
        self.strict_100 = strict_100

        # 增强 & 采样设置
        self.transform = transform
        self.sample_points = sample_points
        self.sample_ratio = sample_ratio
        self.sample_with_replacement = sample_with_replacement

        # 构建样本索引列表
        self.samples: List[Tuple[str, str, Path, Path]] = []
        for sub in self.subjects:
            subj_dir = self.fold_root / sub / "Pointdata"
            if not subj_dir.exists():
                continue
            for bd in sorted([p for p in subj_dir.iterdir() if p.is_dir()]):
                bname = bd.name
                if self.bundles_filter and bname not in self.bundles_filter:
                    continue
                p_points = bd / "bundle_points.npz"
                p_cent = bd / "centroids_init.npz"
                if not p_points.exists() or not p_cent.exists():
                    continue
                # 轻量检查
                try:
                    with np.load(p_points, allow_pickle=False) as d:
                        pts = d["points"]
                    if pts.shape[0] < self.min_points:
                        continue
                    with np.load(p_cent, allow_pickle=False) as d:
                        cen = d["centroids_init"]
                    if self.strict_100 and cen.shape[0] != 100:
                        continue
                except Exception:
                    continue
                self.samples.append((sub, bname, p_points, p_cent))

        if not self.samples:
            raise RuntimeError("No valid samples found.")

    # ---- 内部：简单随机抽样（SRS） ----
    def _subsample_uniform(self, vox_xyz_np: np.ndarray, vox_feat_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        对体素点做等概率无放回抽样；当 M>=N 且不放回时，直接返回全体点。
        """
        N = vox_xyz_np.shape[0]
        # 决定 M
        if self.sample_points is not None:
            M = int(self.sample_points)
        elif self.sample_ratio is not None:
            M = max(1, int(round(N * float(self.sample_ratio))))
        else:
            return vox_xyz_np, vox_feat_np  # 不采样

        if M >= N and not self.sample_with_replacement:
            return vox_xyz_np, vox_feat_np

        if self.sample_with_replacement and M > N:
            idx = torch.randint(low=0, high=N, size=(M,), dtype=torch.long).numpy()
        else:
            idx = torch.randperm(N)[:M].numpy()

        if vox_feat_np.shape[1] > 0:
            return vox_xyz_np[idx], vox_feat_np[idx]
        else:
            return vox_xyz_np[idx], vox_feat_np

    # ---- Dataset protocol ----
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        subj, bname, p_points, p_cent = self.samples[idx]

        with np.load(p_points, allow_pickle=False) as d:
            pts = d["points"].astype(np.float32)          # (N, Dv)

        with np.load(p_cent, allow_pickle=False) as d:
            cen = d["centroids_init"].astype(np.float32)  # (100, Dn)

        assert pts.shape[1] >= 3 and cen.shape[1] >= 3, "前3列必须为 xyz"

        # 拆分
        vox_xyz_np = pts[:, :3]
        vox_feat_np = pts[:, 3:] if pts.shape[1] > 3 else np.zeros((pts.shape[0], 0), np.float32)

        node_xyz_np = cen[:100, :3]
        node_feat_np = cen[:100, 3:] if cen.shape[1] > 3 else np.zeros((100, 0), np.float32)

        # --- SRS 随机子集采样（每次迭代不同） ---
        vox_xyz_np, vox_feat_np = self._subsample_uniform(vox_xyz_np, vox_feat_np)

        # 转张量
        vox_xyz = torch.from_numpy(vox_xyz_np)           # (N,3)
        vox_feat = torch.from_numpy(vox_feat_np)         # (N,C_v)
        node_xyz = torch.from_numpy(node_xyz_np)         # (100,3)
        node_feat = torch.from_numpy(node_feat_np)       # (100,C_n)

        # --- 几何增强：需对 (vox_xyz, node_xyz) 同变换 ---
        if self.transform is not None:
            vox_xyz, node_xyz = self.transform(vox_xyz, node_xyz)

        return {
            "vox_xyz": vox_xyz.float(),
            "vox_feat": vox_feat.float(),
            "node_xyz": node_xyz.float(),
            "node_feat": node_feat.float(),
            "subject": subj,
            "bundle": bname,
        }


# =========================
# Collate: padding + mask
# =========================
def pad_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    将可变长度的体素点云右侧 padding 到同一长度 N_max，并返回 vox_mask。
    节点固定 100，无需 padding。
    """
    B = len(batch)
    N_list = [b["vox_xyz"].shape[0] for b in batch]
    N_max = max(N_list)

    # 体素：坐标/属性/掩码
    vox_xyz = torch.zeros(B, N_max, 3, dtype=torch.float32)
    vox_mask = torch.zeros(B, N_max, dtype=torch.bool)

    C_v_max = max(b["vox_feat"].shape[1] for b in batch)
    # 允许 0 列属性（C_v_max=0）
    vox_feat = torch.zeros(B, N_max, C_v_max, dtype=torch.float32)

    # 节点：固定 100
    node_xyz = torch.stack([b["node_xyz"] for b in batch], dim=0)  # (B,100,3)
    C_n_max = max(b["node_feat"].shape[1] for b in batch)
    node_feat = torch.zeros(B, 100, C_n_max, dtype=torch.float32)

    subjects: List[str] = []
    bundles: List[str] = []

    for i, b in enumerate(batch):
        n = b["vox_xyz"].shape[0]
        vox_xyz[i, :n] = b["vox_xyz"]
        vox_mask[i, :n] = True

        if C_v_max > 0 and b["vox_feat"].shape[1] > 0:
            Cv_i = b["vox_feat"].shape[1]
            vox_feat[i, :n, :Cv_i] = b["vox_feat"]

        if C_n_max > 0 and b["node_feat"].shape[1] > 0:
            Cn_i = b["node_feat"].shape[1]
            node_feat[i, :, :Cn_i] = b["node_feat"]

        subjects.append(b["subject"])
        bundles.append(b["bundle"])

    return {
        "vox_xyz": vox_xyz,     # (B, N_max, 3)
        "vox_feat": vox_feat,   # (B, N_max, C_v_max)
        "vox_mask": vox_mask,   # (B, N_max)
        "node_xyz": node_xyz,   # (B, 100, 3)
        "node_feat": node_feat, # (B, 100, C_n_max)
        "subject": subjects,
        "bundle": bundles,
    }


# =========================
# Loader helper
# =========================
def make_loader(
    fold_root: str,
    subjects: Optional[List[str]] = None,
    bundles: Optional[List[str]] = None,
    batch_size: int = 2,
    num_workers: int = 4,
    shuffle: bool = True,
    # 透传给 Dataset 的增强与采样参数
    transform=None,
    sample_points: Optional[int] = None,
    sample_ratio: Optional[float] = None,
    sample_with_replacement: bool = False,
):
    ds = BundleCenterlineDataset(
        fold_root=fold_root,
        subjects=subjects,
        bundles=bundles,
        transform=transform,
        sample_points=sample_points,
        sample_ratio=sample_ratio,
        sample_with_replacement=sample_with_replacement,
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        collate_fn=pad_collate_fn
    )
    return loader, ds


if __name__ == '__main__':

    fold_root = "/data/zcz/test_1010/tractometry/Pointdata2/fold4"

    train_ds = BundleCenterlineDataset(
        fold_root=fold_root,
        subjects=None,
        bundles=["AF_left"],
        sample_points=2048,  # 每次迭代从该束随机采 16k 点
        sample_with_replacement=True  # N<16k 时：直接用全体点
        # transform=train_tf,         # 若已配置几何增强
    )

    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True, num_workers=0,
        pin_memory=True, collate_fn=pad_collate_fn
    )

    batch = next(iter(train_loader))
    print(batch["vox_xyz"].shape)  # -> (2, N_max≈16000, 3)（按每样本实际 N 与 M 的上界 pad）
