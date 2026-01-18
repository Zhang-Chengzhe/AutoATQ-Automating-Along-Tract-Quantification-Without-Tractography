#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用版本：评估预测与真实的沿束指标（FA / MD / AD / ...）

示例：
  python eval_metric_profiles.py --root /data/project --metric FA
  python eval_metric_profiles.py --root /data/project --metric MD

结构（每个被试目录）：
<root>/<subject>/
  pred_FA.xlsx   或 pred_MD.xlsx
  FA_2.csv       或 MD_2.csv
"""

import os
import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===============================
# 1️⃣ 指标计算
# ===============================
def evaluate_profile(true_arr: np.ndarray, pred_arr: np.ndarray):
    true_arr = np.asarray(true_arr).astype(float).flatten()
    pred_arr = np.asarray(pred_arr).astype(float).flatten()
    if true_arr.shape != pred_arr.shape:
        n = min(len(true_arr), len(pred_arr))
        true_arr, pred_arr = true_arr[:n], pred_arr[:n]
        warnings.warn(f"长度不一致，已截齐到 {n}")

    mask = np.isfinite(true_arr) & np.isfinite(pred_arr)
    if not np.any(mask):
        return {k: np.nan for k in ["MAE", "RMSE", "nRMSE", "R2", "Pearson_r", "Spearman_r"]}

    t, p = true_arr[mask], pred_arr[mask]
    mae = mean_absolute_error(t, p)
    rmse = np.sqrt(mean_squared_error(t, p))
    rng = (np.nanmax(t) - np.nanmin(t)) + 1e-8
    nrmse = rmse / rng
    r2 = r2_score(t, p)
    r_pearson, _ = pearsonr(t, p)
    r_spearman, _ = spearmanr(t, p)
    return dict(MAE=mae, RMSE=rmse, nRMSE=nrmse, R2=r2, Pearson_r=r_pearson, Spearman_r=r_spearman)


# ===============================
# 2️⃣ 智能读取真实 CSV（自动识别分隔符）
# ===============================
def read_true_csv_auto(csv_path: Path) -> pd.DataFrame:
    for sep_try in (None, ";", ",", "\t", "|"):
        try:
            df = pd.read_csv(csv_path, sep=sep_try, engine="python", encoding="utf-8-sig")
            if df.shape[1] == 1 and isinstance(df.iloc[0, 0], str) and (";" in df.iloc[0, 0] or "," in df.iloc[0, 0]):
                continue
            return df
        except Exception:
            continue
    return pd.read_csv(csv_path, sep=";", engine="python", encoding="utf-8-sig")


# ===============================
# 3️⃣ 画叠加图
# ===============================
def plot_overlay(true_series, pred_series, title, save_path):
    plt.figure(figsize=(7, 4))
    plt.plot(true_series.values, label="True", color="black", linewidth=1.5)
    plt.plot(pred_series.values, label="Predicted", color="red", linestyle="--", linewidth=1.8)
    plt.xlabel("Segment index")
    plt.ylabel("Metric value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=160)
    plt.close()


# ===============================
# 4️⃣ 核心处理函数（单个被试）
# ===============================
def process_subject(
    subj_dir: Path,
    metric: str = "FA",
):
    pred_name = f"pred_{metric}.xlsx"
    true_name = f"{metric}_2.csv"
    out_metrics_name = f"accuracy_{metric}.xlsx"
    plot_dirname = f"plots_{metric}"

    pred_path = subj_dir / pred_name
    true_path = subj_dir / true_name
    if not pred_path.exists() or not true_path.exists():
        print(f"[SKIP] {subj_dir.name}: 缺少 {pred_name} 或 {true_name}")
        return

    # 读预测
    pred_df = pd.read_excel(pred_path, engine="openpyxl")

    # 读真实
    true_df_raw = read_true_csv_auto(true_path)

    # 对齐列顺序
    pred_cols = list(pred_df.columns)
    missing = [c for c in pred_cols if c not in true_df_raw.columns]
    if missing:
        raise ValueError(f"[{subj_dir.name}] 真实文件缺少这些列: {missing}")
    true_df = true_df_raw.reindex(columns=pred_cols)

    # 对齐行数
    n = min(len(pred_df), len(true_df))
    pred_df = pred_df.iloc[:n, :]
    true_df = true_df.iloc[:n, :]

    # 计算每列指标
    records = []
    for bundle in pred_cols:
        metrics = evaluate_profile(true_df[bundle].values, pred_df[bundle].values)
        metrics["Bundle"] = bundle
        records.append(metrics)
    df_metrics = pd.DataFrame(records).set_index("Bundle")

    # 加整体均值
    overall = df_metrics.mean(numeric_only=True)
    overall.name = "Overall_mean"
    df_metrics_out = pd.concat([df_metrics, overall.to_frame().T], axis=0)

    # 写 Excel
    out_path = subj_dir / out_metrics_name
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_metrics_out.to_excel(writer, sheet_name="metrics")

        # 构造对齐后的数据（两层列：('True', bundle), ('Pred', bundle)）
        aligned = pd.concat({"True": true_df, "Pred": pred_df}, axis=1)

        # ⚠️ 关键：拍平 MultiIndex 列，避免 “MultiIndex columns + index=False” 报错
        aligned.columns = [
            f"{lvl0}_{lvl1}" if isinstance(col, tuple) else str(col)
            for col in aligned.columns.to_flat_index()
            for lvl0, lvl1 in [col if isinstance(col, tuple) else ("", col)]
        ]
        # 现在列是单层了，可以放心 index=False
        aligned.to_excel(writer, sheet_name="aligned_profiles", index=False)
    print(f"[OK] {subj_dir.name}: 指标写入 {out_path}")

    # 绘图
    plot_dir = subj_dir / plot_dirname
    for bundle in pred_cols:
        save_path = plot_dir / f"{bundle}.png"
        plot_overlay(true_df[bundle], pred_df[bundle], f"{subj_dir.name} - {bundle} ({metric})", save_path)
    print(f"[OK] {subj_dir.name}: 曲线图保存到 {plot_dir}")


# ===============================
# 5️⃣ 主程序
# ===============================
def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate along-tract metric profiles (FA / MD / etc.)")
    ap.add_argument("--root", type=str, required=True, help="根目录，包含多个被试文件夹")
    ap.add_argument("--metric", type=str, default="FA", help="指标名，如 FA / MD / AD / NDI 等")
    return ap.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    subs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not subs:
        print(f"[ERR] 在 {root} 下未找到任何被试目录")
        return

    for subj in subs:
        try:
            process_subject(subj, metric=args.metric)
        except Exception as e:
            print(f"[FAIL] {subj.name}: {e}")


if __name__ == "__main__":
    main()
