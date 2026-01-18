# predict.py
# -*- coding: utf-8 -*-
import os
import gc
import csv
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn

from dataset_pointcloud_with_centroids import make_loader
from model import CenterlinePointNetMLP, loss_coord_smoothl1  # loss仅在需要时才会用到（可留作调试）


# =========================
# 辅助函数
# =========================
def list_all_bundles(fold_root: str) -> List[str]:
    """从第一个 subject 推断 bundle 名单"""
    root = Path(fold_root)
    subjects = [p for p in root.iterdir() if p.is_dir()]
    if not subjects:
        return []
    first = subjects[0] / "Pointdata"
    if not first.exists():
        return []
    return sorted([p.name for p in first.iterdir() if p.is_dir()])


def calculate_mpjpe(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    计算每个节点的平均位置误差 MPJPE
    predictions, targets: (B, N, 3) 或 (N, 3)
    """
    errors = torch.norm(predictions - targets, dim=-1)  # (B,N) 或 (N,)
    return errors.mean().item()


def infer_sample_id(batch: Dict[str, Any], fallback_idx: int) -> str:
    """
    从 batch 中尽力获取一个样本标识，用于保存文件命名。
    常见键位：'sid', 'subject', 'name', 'id'，也支持字符串列表/标量。
    """
    candidate_keys = ["sid", "subject", "name", "id", "uid"]
    for k in candidate_keys:
        if k in batch:
            v = batch[k]
            # 处理张量 / 标量 / 列表 / 字符串
            if isinstance(v, torch.Tensor):
                try:
                    v = v.item()
                except Exception:
                    v = v.detach().cpu().tolist()
            if isinstance(v, (list, tuple)):
                # batch_size=1 时取第一个
                if len(v) > 0:
                    return str(v[0])
            else:
                return str(v)
    return f"idx{fallback_idx:06d}"


def load_model(weights_path: str, device: torch.device) -> nn.Module:
    """
    构建模型并加载权重到单卡模型（训练时保存的是 base 或 module 子模块的 state_dict 均可兼容）。
    """
    model = CenterlinePointNetMLP(g_dim=256, pos_dim=32, K=100)
    state = torch.load(weights_path, map_location=device)
    # 某些情况下可能带有 'module.' 前缀，做下兼容处理
    if any(k.startswith("module.") for k in state.keys()):
        new_state = {}
        for k, v in state.items():
            new_state[k.replace("module.", "", 1)] = v
        state = new_state
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# =========================
# 推理主函数
# =========================
def run_inference(
    fold_root: str,
    weights_dir: str,
    out_dir: str,
    bundles: Optional[List[str]],
    batch_size: int = 1,
    num_workers: int = 0,
    save_as: str = "npy",  # "npy" 或 "pt"
    compute_mpjpe: bool = True,
    device_str: Optional[str] = None,
):
    """
    对fold_root中的数据逐bundle推理，使用weights_dir中的{bundle}_best.pth权重。
    """
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    ensure_dir(Path(out_dir))

    if bundles is None:
        bundles = list_all_bundles(fold_root)
    print(f"Inference bundles: {bundles}")

    # 准备总体 metrics.csv
    metrics_path = Path(out_dir) / "metrics.csv"
    metrics_f = open(metrics_path, "w", newline="")
    csv_writer = csv.writer(metrics_f)
    # 表头
    header = ["bundle", "sample_id", "mpjpe"]
    csv_writer.writerow(header)

    for bundle in bundles:
        print(f"\n=== Inference Bundle: {bundle} ===")

        # ---- 查找权重 ----
        weight_path = Path(weights_dir) / f"{bundle}_best.pth"
        if not weight_path.exists():
            print(f"[WARN] Weights not found for bundle '{bundle}': {weight_path}. Skip this bundle.")
            continue

        # ---- DataLoader ----
        # 对fold5做推理，不需要采样（保持与验证一致），batch_size=1 以便逐样本保存
        loader, _ = make_loader(
            fold_root=fold_root,
            subjects=None,
            bundles=[bundle],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            sample_points=None,
            sample_ratio=None,
            sample_with_replacement=False,
            transform=None,
        )

        # ---- Load model weights ----
        model = load_model(str(weight_path), device=device)

        # ---- 输出路径 ----
        bundle_out_dir = Path(out_dir) / bundle
        ensure_dir(bundle_out_dir)

        with torch.no_grad():
            for idx, batch in enumerate(loader):
                vox_xyz = batch["vox_xyz"].to(device)
                vox_mask = batch["vox_mask"].to(device)

                # 推理
                P_pred = model(vox_xyz, vox_mask)  # 期望形状 (B, N, 3)

                # 保存预测
                sample_id = infer_sample_id(batch, fallback_idx=idx)
                pred_path = bundle_out_dir / f"{sample_id}_pred.{save_as}"

                if save_as.lower() == "npy":
                    np.save(pred_path, P_pred.detach().cpu().numpy())
                else:
                    torch.save(P_pred.detach().cpu(), pred_path)

                # 可选：计算并记录 MPJPE（若 batch 中有 node_xyz）
                if compute_mpjpe and ("node_xyz" in batch):
                    node_xyz = batch["node_xyz"].to(device)
                    mpjpe = calculate_mpjpe(P_pred, node_xyz)
                    csv_writer.writerow([bundle, sample_id, f"{mpjpe:.6f}"])
                else:
                    # 若无真值，记空
                    csv_writer.writerow([bundle, sample_id, ""])

                if (idx + 1) % 50 == 0:
                    print(f"[{bundle}] processed {idx + 1} samples...")

        # —— 清理显存/内存 —— #
        for obj_name in ["model", "loader", "batch", "vox_xyz", "vox_mask", "P_pred"]:
            if obj_name in locals():
                del locals()[obj_name]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics_f.close()
    print(f"\nDone. Predictions saved to: {out_dir}")
    print(f"Metrics (if available) saved to: {metrics_path}")


# =========================
# CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser(description="Per-bundle inference on fold data using trained weights.")
    ap.add_argument("--fold_root", type=str, required=True,
                    help="用于推理的数据根目录（例如 fold5 路径）")
    ap.add_argument("--weights_dir", type=str, required=True,
                    help="训练权重目录，内含 {bundle}_best.pth")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="输出目录，保存预测和metrics")
    ap.add_argument("--bundles", type=str, nargs="*", default=None,
                    help="指定需要推理的bundle列表；缺省则自动从fold_root推断")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--save_as", type=str, default="npy", choices=["npy", "pt"],
                    help="预测保存格式：npy 或 pt")
    ap.add_argument("--no_mpjpe", action="store_true",
                    help="不计算MPJPE（即使存在node_xyz）")
    ap.add_argument("--device", type=str, default=None,
                    help="指定设备，例如 'cuda:0' 或 'cpu'；缺省自动选择")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        fold_root=args.fold_root,
        weights_dir=args.weights_dir,
        out_dir=args.out_dir,
        bundles=args.bundles,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_as=args.save_as,
        compute_mpjpe=(not args.no_mpjpe),
        device_str=args.device,
    )
