"""
情绪波动性分析与可视化脚本
--------------------------------
功能：
1. 从 CSV 文件中读取随时间记录的情绪标签
2. 将情绪映射为数值，计算情绪波动性指标：
   - 标准差（std）
   - 平均情绪水平（mean）
   - 变异系数（CV = std / mean）
   - 取值跨度（range）
3. 生成情绪随时间变化折线图，并保存为 PNG 图片

使用方式：
1. 准备一个 CSV 文件，例如 ../data/emotion_log.csv
   要求至少包含两列：
       time    —— 时间戳（字符串，例如 "2025-11-24 18:00:00"）
       emotion —— 模型预测的情绪标签（例如 "Happy", "Sad"...）

2. 直接运行本脚本：
   python emotion_fluctuation_analysis.py

3. 运行后：
   - 在控制台输出统计指标
   - 在 ../result/ 目录下生成 emotion_fluctuation_curve.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 和 real_time_detection.py 中保持一致的情绪标签顺序
EMOTION_LABELS = ['Angry', 'Disgusted', 'Scared', 'Happy', 'Sad', 'Surprised', 'Calm']

# 构造情绪 -> 数值映射表
EMOTION_TO_INT = {name: idx for idx, name in enumerate(EMOTION_LABELS)}


def load_emotion_series(csv_path: str) -> pd.DataFrame:
    """
    从 CSV 中读取情绪时间序列，并做基础预处理。
    期望 CSV 至少包含列：time, emotion
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到数据文件：{csv_path}")

    df = pd.read_csv(csv_path)

    # 检查必要列
    if "time" not in df.columns or "emotion" not in df.columns:
        raise ValueError("CSV 中必须至少包含 'time' 和 'emotion' 两列")

    # 解析时间，并排序
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])  # 去掉无法解析时间的行
    df = df.sort_values("time").reset_index(drop=True)

    # 将情绪标签映射为数值，为后续计算做准备
    df["emotion_code"] = df["emotion"].map(EMOTION_TO_INT)

    # 去掉无法映射的情绪（可能是拼写错误等）
    df = df.dropna(subset=["emotion_code"])
    df["emotion_code"] = df["emotion_code"].astype(int)

    return df


def compute_fluctuation_metrics(df: pd.DataFrame) -> dict:
    """
    计算情绪波动性相关指标。
    输入：
        df —— 必须包含列 emotion_code
    输出：
        一个字典，包含 std / mean / cv / range
    """
    codes = df["emotion_code"].astype(float)

    std = float(codes.std(ddof=1))     # 样本标准差
    mean = float(codes.mean())
    value_range = float(codes.max() - codes.min())
    cv = float(std / mean) if mean != 0 else np.nan

    metrics = {
        "std": std,
        "mean": mean,
        "cv": cv,
        "range": value_range,
    }
    return metrics


def plot_emotion_curve(df: pd.DataFrame, save_dir: str = "../result") -> str:
    """
    绘制情绪变化曲线（横轴用采集顺序，而不是时间戳），并按点数自动拆分多张图。

    满足两个需求：
    1. 同一秒采集的 2~4 帧，不会再挤在一起，而是按采集顺序依次展开在横轴上；
    2. 如果点太多，会自动按采集顺序拆成多张图片。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # === 1. 确保有 emotion_code 数值列 ===
    if "emotion_code" not in df.columns:
        emo_to_int = {name: idx for idx, name in enumerate(EMOTION_LABELS)}
        df = df.copy()
        df["emotion_code"] = df["emotion"].map(emo_to_int)

    # === 2. 用“采集顺序”当横轴，而不是时间 ===
    # 先按原始时间排序，保证顺序正确
    df = df.copy()
    df["time_dt"] = pd.to_datetime(df["time"])
    df = df.sort_values("time_dt").reset_index(drop=True)

    # x 轴：0,1,2,... 每一帧一个点
    df["frame_idx"] = np.arange(len(df))

    # === 3. 按点数拆成多张图 ===
    MAX_POINTS_PER_FIG = 800  # 一张图最多画多少个点，可以自己调
    n = len(df)
    n_fig = int(np.ceil(n / MAX_POINTS_PER_FIG))

    save_paths = []

    for fig_idx in range(n_fig):
        start = fig_idx * MAX_POINTS_PER_FIG
        end = min((fig_idx + 1) * MAX_POINTS_PER_FIG, n)
        sub = df.iloc[start:end]

        plt.figure(figsize=(12, 5))

        # 核心：横轴是 frame_idx —— 永远严格递增，不会竖线重叠
        plt.plot(sub["frame_idx"], sub["emotion_code"], marker="o", linewidth=1)

        # y 轴显示中文标签
        plt.yticks(
            ticks=list(range(len(EMOTION_LABELS))),
            labels=EMOTION_LABELS
        )

        plt.xlabel("帧序号（随采集时间递增）")
        if n_fig == 1:
            title_suffix = ""
        else:
            title_suffix = f"（第 {fig_idx + 1} 段，共 {n_fig} 段）"
        plt.title("情绪随采集顺序变化曲线" + title_suffix)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()

        # 文件名：单图或多段
        if n_fig == 1:
            fname = "emotion_fluctuation_curve.png"
        else:
            fname = f"emotion_fluctuation_curve_part{fig_idx + 1}.png"

        save_path = os.path.join(save_dir, fname)
        plt.savefig(save_path, dpi=150)
        plt.close()

        save_paths.append(save_path)

    # 为兼容原来的调用逻辑，这里返回第一张图路径
    return save_paths[0] if save_paths else ""



def main():
    # 默认的数据路径，可根据需要修改
    default_csv_path = "../data/emotion_log.csv"

    csv_path = default_csv_path
    print(f"[INFO] 使用的数据文件：{csv_path}")

    df = load_emotion_series(csv_path)
    if df.empty:
        print("[WARN] 数据为空，请检查 emotion_log.csv 是否有有效内容。")
        return

    # 1. 计算情绪波动性指标
    metrics = compute_fluctuation_metrics(df)

    print("\n===== 情绪波动性分析结果 =====")
    print(f"样本数量: {len(df)}")
    print(f"情绪标准差 (std): {metrics['std']:.4f}")
    print(f"平均情绪水平 (mean): {metrics['mean']:.4f}")
    print(f"变异系数 (cv = std / mean): {metrics['cv']:.4f}" if not np.isnan(metrics['cv']) else "变异系数 (cv): NaN（平均值为 0）")
    print(f"情绪编码取值跨度 (max - min): {metrics['range']:.4f}")

    # 2. 画图并保存
    save_path = plot_emotion_curve(df)
    print(f"\n[INFO] 情绪随时间变化折线图已保存至：{save_path}")


if __name__ == "__main__":
    main()
