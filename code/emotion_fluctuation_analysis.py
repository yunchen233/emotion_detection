"""
情绪波动性分析与可视化脚本
--------------------------------
功能：
1. 从 CSV 文件中读取随时间记录的情绪标签
2.生成情绪随时间变化折线图，并保存为 PNG 图片
4.在 ../result/ 目录下生成 emotion_fluctuation_curve.png
改为只支持400帧的，太麻烦了
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

EMOTION_LABELS = ['Angry', 'Disgusted', 'Scared', 'Happy', 'Sad', 'Surprised', 'Calm','Contempt']

# 构造情绪 -> 数值映射表，映射到0-7
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

def plot_emotion_curve(df: pd.DataFrame, save_dir: str = "../result") -> str:
    """
    情绪变化可视化：
    - 按采集顺序排序绘制散点图；
    - 按帧数自动拆分为多张图。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # 1. 数值编码
    if "emotion_code" not in df.columns:
        emo_to_int = {name: idx for idx, name in enumerate(EMOTION_LABELS)}
        df = df.copy()
        df["emotion_code"] = df["emotion"].map(emo_to_int)

    # 2. 按时间排序 + 建立帧序号
    df = df.copy()
    df["time_dt"] = pd.to_datetime(df["time"])
    df = df.sort_values("time_dt").reset_index(drop=True)
    df["frame_idx"] = np.arange(len(df))

    #3. 按帧索引范围拆成多张图
    MAX_FRAMES_PER_FIG = 400
    n_frames = len(df)
    n_fig = int(np.ceil(n_frames / MAX_FRAMES_PER_FIG))#计算需要分成多少张图

    save_paths = []

    for fig_idx in range(n_fig):
        fig_start = fig_idx * MAX_FRAMES_PER_FIG
        fig_end = min((fig_idx + 1) * MAX_FRAMES_PER_FIG - 1, n_frames - 1)#循环生成每张图，计算帧范围

        # 获取当前图的数据子集
        fig_df = df[(df["frame_idx"] >= fig_start) & (df["frame_idx"] <= fig_end)]


        # 散点图
        plt.scatter(
            fig_df["frame_idx"],
            fig_df["emotion_code"],
            color='blue',
            alpha=0.6,
            s=2,  # 点的大小
        )

        # y 轴标签
        plt.yticks(
            ticks=list(range(len(EMOTION_LABELS))),
            labels=EMOTION_LABELS,
        )

        plt.xlabel("帧序号（随采集时间递增）")
        if n_fig == 1:
            title_suffix = ""
        else:
            title_suffix = f"（第 {fig_idx + 1} 段，共 {n_fig} 段）"
        plt.title("情绪随采集顺序变化曲线" + title_suffix)

        plt.grid(True, linestyle="--", alpha=0.2, axis="x")
        plt.tight_layout()

        # 保存
        if n_fig == 1:
            fname = "emotion_fluctuation_curve.png"
        else:
            fname = f"emotion_fluctuation_curve_part{fig_idx + 1}.png"

        save_path = os.path.join(save_dir, fname)
        plt.savefig(save_path, dpi=150)
        plt.close()
        save_paths.append(save_path)

    return save_paths[0] if save_paths else ""



def main():
    default_csv_path =("../data/emotion_log.csv")

    csv_path = default_csv_path
    print(f"[INFO] 使用的数据文件：{csv_path}")

    df = load_emotion_series(csv_path)
    if df.empty:
        print("[WARN] 数据为空，请检查 emotion_log.csv 是否有有效内容。")
        return

    print("\n===== 情绪波动性分析结果 =====")
    print(f"样本数量: {len(df)}")

    # 2. 画图并保存
    save_path = plot_emotion_curve(df)
    print(f"\n[INFO] 情绪随时间变化折线图已保存至：{save_path}")


if __name__ == "__main__":
    main()