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
    绘制情绪随时间变化的折线图，并保存为 PNG 文件。
    返回：保存的文件路径
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "emotion_fluctuation_curve.png")

    plt.figure(figsize=(12, 5))

    # 画出 emotion_code 的折线
    plt.plot(df["time"], df["emotion_code"], marker="o", linewidth=1)

    # 设置坐标轴与标题
    plt.title("情绪随时间变化曲线", fontproperties="SimHei")
    plt.xlabel("时间", fontproperties="SimHei")
    plt.ylabel("情绪（数值编码）", fontproperties="SimHei")

    # 将 y 轴刻度标成情绪名称，方便阅读
    yticks = sorted(df["emotion_code"].unique())
    ylabels = [EMOTION_LABELS[int(v)] for v in yticks]
    plt.yticks(yticks, ylabels, fontproperties="SimHei")

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return save_path


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
