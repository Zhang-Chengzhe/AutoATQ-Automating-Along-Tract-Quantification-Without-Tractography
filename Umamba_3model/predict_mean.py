import os
import argparse
from pathlib import Path
import glob
import numpy as np
import nibabel as nib
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_loadingv2 import MyDataset
from train_3model_1loss import predict_inmem  # 复用你的三向推理函数

# ----------------- 工具函数 -----------------
def restore_to_orig_space(crop_vol_cdhw: np.ndarray, meta: dict):
    """裁剪空间 [C,144,144,144] → 原空间 4D [X,Y,Z,C]，返回 (arr4d, affine)"""
    C, Dc, Hc, Wc = crop_vol_cdhw.shape
    D0, H0, W0 = meta["orig_shape"]
    d0, d1, h0, h1, w0, w1 = meta["bbox"]
    affine = meta.get("affine", np.eye(4))
    full = np.zeros((C, D0, H0, W0), dtype=np.float32)
    full[:, d0:d1, h0:h1, w0:w1] = crop_vol_cdhw.astype(np.float32)
    arr4d = np.transpose(full, (1, 2, 3, 0))  # [D,H,W,C]
    return arr4d, affine

def save_4d(arr_4d_xyzc: np.ndarray, affine: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr_4d_xyzc, affine), str(out_path))

def find_label_file(labels_dir: Path, base: str):
    """在 labels_dir 中搜索与 base 同名的标签文件（.nii.gz / .nii / .npy）"""
    cand = []
    cand += glob.glob(str(labels_dir / f"{base}.nii.gz"))
    cand += glob.glob(str(labels_dir / f"{base}.nii"))
    cand += glob.glob(str(labels_dir / f"{base}.npy"))
    return cand[0] if len(cand) > 0 else None

def load_label_as_onehot(label_path: str, C: int):
    """
    读取标签；支持：
      - NIfTI: 若为整型 3D，转 one-hot 为 [X,Y,Z,C]
      - NPY:   若为 4D [...,C] 直接用；若为 3D 整型则 one-hot
    """
    if label_path.endswith((".nii", ".nii.gz")):
        lab = np.asanyarray(nib.load(label_path).dataobj)
    elif label_path.endswith(".npy"):
        lab = np.load(label_path)
    else:
        raise ValueError(f"Unsupported label file: {label_path}")

    if lab.ndim == 4 and lab.shape[-1] == C:
        onehot = lab.astype(np.uint8)
    elif lab.ndim == 3:
        # 整型标签 → one-hot
        onehot = np.zeros(lab.shape + (C,), dtype=np.uint8)
        # 允许标签值 0..C-1；若有越界将被忽略
        for c in range(C):
            onehot[..., c] = (lab == c).astype(np.uint8)
    else:
        raise ValueError(f"Label shape not compatible: {lab.shape}, expected 3D int or 4D with C={C}")
    return onehot

def dice_per_class(pred_oh: np.ndarray, gt_oh: np.ndarray, eps: float = 1e-6):
    """
    pred_oh, gt_oh: [X,Y,Z,C] 的 0/1 掩模
    返回:
      dices: [C]（若该类 GT 为空 → np.nan）
      valid: [C]（bool，有 GT 体素）
    """
    C = pred_oh.shape[-1]
    dices = np.zeros(C, dtype=np.float32)
    valid = np.zeros(C, dtype=bool)
    for c in range(C):
        p = pred_oh[..., c].astype(np.float32)
        g = gt_oh[..., c].astype(np.float32)
        g_sum = g.sum()
        if g_sum < 0.5:  # 无 GT
            dices[c] = np.nan
            valid[c] = False
            continue
        inter = (p * g).sum()
        denom = p.sum() + g_sum
        dices[c] = (2.0 * inter + eps) / (denom + eps)
        valid[c] = True
    return dices, valid

# ----------------- 主流程 -----------------
def main():
    parser = argparse.ArgumentParser(description="Three-plane inference + fusion + 3D Dice eval + Excel export")
    parser.add_argument("--images_dir", required=True, help="图像目录（.npy 或 .nii.gz）")
    parser.add_argument("--labels_dir", required=True, help="标签目录（要求与图像同名）")
    parser.add_argument("--save_dir", required=True, help="权重目录：best_axial.pth / best_coronal.pth / best_sagittal.pth")
    parser.add_argument("--out_dir", required=True, help="输出目录（保存 fused 概率/掩模 与 评估表）")
    parser.add_argument("--excel_path", default=None, help="结果 Excel 路径（默认：out_dir/dice_per_subject.xlsx）")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--slices_per_subject", type=int, default=144)
    parser.add_argument("--mask_thr", type=float, default=0.5)
    parser.add_argument("--save_each_plane", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    excel_path = Path(args.excel_path) if args.excel_path else (out_dir / "dice_per_subject.xlsx")

    # 数据集（不增强，固定顺序）
    ds_ax = MyDataset(args.images_dir, args.labels_dir, transform=None, direction="axial", augment=False)
    ds_cor = MyDataset(args.images_dir, args.labels_dir, transform=None, direction="coronal", augment=False)
    ds_sag = MyDataset(args.images_dir, args.labels_dir, transform=None, direction="sagittal", augment=False)

    dl_ax  = DataLoader(ds_ax,  batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    dl_cor = DataLoader(ds_cor, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    dl_sag = DataLoader(ds_sag, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # 三向推理
    probs_ax, probs_cor, probs_sag = predict_inmem(
        save_dir=args.save_dir,
        device=device,
        loader_axial=dl_ax,
        loader_coronal=dl_cor,
        loader_sagittal=dl_sag,
        slices_per_subject=args.slices_per_subject,
        use_amp=args.use_amp
    )

    # 输出子目录
    out_fused_prob = out_dir / "fused_prob_4d"
    out_fused_mask = out_dir / f"fused_mask_thr{args.mask_thr:.2f}_4d"
    out_eachprob   = out_dir / "each_plane_prob_4d"
    out_eachmask   = out_dir / f"each_plane_mask_thr{args.mask_thr:.2f}_4d"
    out_fused_prob.mkdir(parents=True, exist_ok=True)
    out_fused_mask.mkdir(parents=True, exist_ok=True)
    if args.save_each_plane:
        out_eachprob.mkdir(parents=True, exist_ok=True)
        out_eachmask.mkdir(parents=True, exist_ok=True)

    # 基名（用于匹配标签与输出命名）
    base_names = []
    for p in ds_ax.data_path:
        name = os.path.basename(p).replace(".npy", "").replace(".nii.gz", "").replace(".nii", "")
        base_names.append(name)

    num_subjects = len(ds_ax) // args.slices_per_subject
    print(f"[INFO] Subjects: {num_subjects}")

    # 用于 Excel 的记录
    rows = []
    C_guess = probs_ax[0].shape[0] if len(probs_ax) > 0 else None
    per_class_cols = [f"class_{i:02d}" for i in range(C_guess)] if C_guess is not None else []

    for sid in range(num_subjects):
        base = base_names[sid] if sid < len(base_names) else f"subj{sid:03d}"
        meta = ds_ax.meta[sid]

        # 融合概率（裁剪空间）
        pv_fused = (probs_ax[sid].astype(np.float32)
                    + probs_cor[sid].astype(np.float32)
                    + probs_sag[sid].astype(np.float32)) / 3.0  # [C,144,144,144]

        # 回原空间 → 概率与掩模
        fused_prob_4d, aff = restore_to_orig_space(pv_fused, meta)
        fused_mask_4d = (fused_prob_4d >= args.mask_thr).astype(np.uint8)

        # 保存 fused
        save_4d(fused_prob_4d, aff, out_fused_prob / f"{base}_fused_prob.nii.gz")
        save_4d(fused_mask_4d, aff, out_fused_mask / f"{base}_fused_mask.nii.gz")

        # 可选保存各向
        if args.save_each_plane:
            ax4d, _  = restore_to_orig_space(probs_ax[sid],  meta)
            cor4d, _ = restore_to_orig_space(probs_cor[sid], meta)
            sag4d, _ = restore_to_orig_space(probs_sag[sid], meta)
            save_4d(ax4d,  aff, out_eachprob / f"{base}_axial_prob.nii.gz")
            save_4d(cor4d, aff, out_eachprob / f"{base}_coronal_prob.nii.gz")
            save_4d(sag4d, aff, out_eachprob / f"{base}_sagittal_prob.nii.gz")
            save_4d((ax4d  >= args.mask_thr).astype(np.uint8), aff, out_eachmask / f"{base}_axial_mask.nii.gz")
            save_4d((cor4d >= args.mask_thr).astype(np.uint8), aff, out_eachmask / f"{base}_coronal_mask.nii.gz")
            save_4d((sag4d >= args.mask_thr).astype(np.uint8), aff, out_eachmask / f"{base}_sagittal_mask.nii.gz")

        # ------------ 评估：读取标签并计算 Dice ------------
        label_file = find_label_file(Path(args.labels_dir), base)
        if label_file is None:
            print(f"[WARN] Label not found for {base}, skip Dice.")
            # 仍然输出一行，记为 NaN
            row = {"subject": base}
            for k in per_class_cols:
                row[k] = np.nan
            row["valid_classes"] = 0
            row["macro_mean_dice"] = np.nan
            rows.append(row)
            continue

        gt_oh_4d = load_label_as_onehot(label_file, fused_mask_4d.shape[-1])  # [X,Y,Z,C]
        # 确保空间一致（必要时可在此处做重采样；默认假设已对齐）
        if gt_oh_4d.shape != fused_mask_4d.shape:
            raise ValueError(f"Shape mismatch for {base}: pred {fused_mask_4d.shape}, label {gt_oh_4d.shape}")

        dices, valid = dice_per_class(fused_mask_4d, gt_oh_4d)
        macro_mean = np.nanmean(dices) if np.any(valid) else np.nan
        row = {"subject": base, "valid_classes": int(valid.sum()), "macro_mean_dice": float(macro_mean)}
        # 填 per-class
        for i, d in enumerate(dices):
            row[f"class_{i:02d}"] = (np.nan if np.isnan(d) else float(d))
        rows.append(row)
        print(f"[OK] {base}: macro-mean dice = {macro_mean:.4f} over {valid.sum()} valid classes")

    # ------------ 导出 Excel ------------
    df = pd.DataFrame(rows)
    # 列顺序：subject, macro_mean_dice, valid_classes, class_00..class_C-1
    ordered_cols = ["subject", "macro_mean_dice", "valid_classes"] + [c for c in df.columns if c.startswith("class_")]
    df = df.reindex(columns=ordered_cols)
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="dice_per_subject")
    print(f"[DONE] Excel saved to: {excel_path}")

if __name__ == "__main__":
    main()
