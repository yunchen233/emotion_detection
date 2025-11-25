"""
情绪转移分析（马尔可夫链）与可视化
------------------------------------
功能：
1) 从 CSV 读取按时间记录的情绪序列
2) 计算一阶马尔可夫链转移矩阵（计数 & 概率）
3) 生成两个可视化并保存到一张图里：
   - 左侧：情绪时间轴散点（横轴时间，纵轴情绪类别，离散点表示）
   - 右侧：情绪转移概率热力图

默认输入：项目根目录下 data/emotion_log.csv（自动用脚本位置推断，无需依赖当前工作目录）
默认输出：项目根目录下 result/emotion_transition_analysis.png
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 字体回退：无 SimHei 时优先用系统中文字体，避免中文乱码
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'STHeiti', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# 统一使用与实时检测一致的情绪顺序
EMOTION_LABELS = ['Angry', 'Disgusted', 'Scared', 'Happy', 'Sad', 'Surprised', 'Calm']
EMOTION_TO_INT = {name: idx for idx, name in enumerate(EMOTION_LABELS)}
# 基于脚本位置推断项目根目录，避免 cwd 不一致导致路径出错
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_emotion_series(csv_path: str) -> pd.DataFrame:
    """
    读取并预处理情绪时间序列。
    返回的 DataFrame 至少包含：time（datetime）、emotion、emotion_code（int）
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到数据文件：{csv_path}")

    df = pd.read_csv(csv_path)
    if "time" not in df.columns or "emotion" not in df.columns:
        raise ValueError("CSV 中必须包含 'time' 和 'emotion' 两列")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    df["emotion_code"] = df["emotion"].map(EMOTION_TO_INT)
    df = df.dropna(subset=["emotion_code"])
    df["emotion_code"] = df["emotion_code"].astype(int)

    return df


def compute_transition_matrices(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    基于一阶马尔可夫链计算转移矩阵。
    返回：(计数矩阵, 概率矩阵)，索引与列均为情绪标签。
    """
    codes = df["emotion_code"].to_numpy()
    if len(codes) < 2:
        raise ValueError("样本数量不足以计算转移（至少需要两条时间序列记录）")

    n = len(EMOTION_LABELS)
    count_matrix = np.zeros((n, n), dtype=int)

    for i in range(len(codes) - 1):
        src = codes[i]
        dst = codes[i + 1]
        count_matrix[src, dst] += 1

    prob_matrix = np.zeros_like(count_matrix, dtype=float)
    row_sums = count_matrix.sum(axis=1, keepdims=True)
    # 避免除零；无转出样本时保持 0
    np.divide(
        count_matrix,
        row_sums,
        out=prob_matrix,
        where=row_sums != 0
    )

    count_df = pd.DataFrame(count_matrix, index=EMOTION_LABELS, columns=EMOTION_LABELS)
    prob_df = pd.DataFrame(prob_matrix, index=EMOTION_LABELS, columns=EMOTION_LABELS)
    return count_df, prob_df


def plot_timeline(ax, df: pd.DataFrame) -> None:
    """绘制离散情绪时间轴散点图。"""
    ax.scatter(df["time"], df["emotion_code"], c=df["emotion_code"], cmap="tab20", s=24)
    ax.set_title("情绪时间轴（离散）")
    ax.set_xlabel("时间")
    ax.set_ylabel("情绪类别")
    ax.set_yticks(range(len(EMOTION_LABELS)))
    ax.set_yticklabels(EMOTION_LABELS)
    ax.grid(True, linestyle="--", alpha=0.3)

    # 调细时间刻度：更密的刻度、更短的可视范围
    span_seconds = max((df["time"].max() - df["time"].min()).total_seconds(), 0)
    target_ticks = 15  # 目标刻度数量，更细的刻度
    if span_seconds <= 900:  # 15 分钟内用秒刻度
        interval = max(1, int(np.ceil(span_seconds / target_ticks)))
        ax.xaxis.set_major_locator(mdates.SecondLocator(interval=interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    else:  # 更长的用分钟刻度
        span_minutes = span_seconds / 60
        interval = max(1, int(np.ceil(span_minutes / target_ticks)))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig = ax.get_figure()
    fig.autofmt_xdate()

    # 缩短跨度：两端留极少的空白，进一步“放大”视图
    pad = max(span_seconds * 0.005, 0.5)  # 至少 0.5 秒留白
    ax.set_xlim(df["time"].min() - pd.Timedelta(seconds=pad),
                df["time"].max() + pd.Timedelta(seconds=pad))


def plot_transition_heatmap(ax, prob_df: pd.DataFrame) -> None:
    """绘制情绪转移概率热力图。"""
    im = ax.imshow(prob_df, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("情绪转移概率 (一阶马尔可夫链)")
    ax.set_xlabel("下一步情绪")
    ax.set_ylabel("当前情绪")
    ax.set_xticks(range(len(EMOTION_LABELS)))
    ax.set_xticklabels(EMOTION_LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(EMOTION_LABELS)))
    ax.set_yticklabels(EMOTION_LABELS)

    # 在格子内标数值，便于快速解读
    for i in range(len(EMOTION_LABELS)):
        for j in range(len(EMOTION_LABELS)):
            prob = prob_df.iloc[i, j]
            ax.text(
                j, i, f"{prob:.2f}",
                ha="center", va="center",
                fontsize=9,
                color="black" if prob < 0.6 else "white"
            )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def analyze_and_plot(csv_path: str = None, save_dir: str = None) -> str:
    """主入口：计算转移矩阵并生成可视化，返回保存路径。"""
    # 若未传入，使用基于脚本位置推断的默认路径，避免 cwd 影响
    if csv_path is None:
        csv_path = os.path.join(PROJECT_ROOT, "data", "emotion_log.csv")
    if save_dir is None:
        save_dir = os.path.join(PROJECT_ROOT, "result")

    df = load_emotion_series(csv_path)
    if df.shape[0] < 2:
        raise ValueError("数据不足，无法计算情绪转移。请采集更多时间序列记录。")

    count_df, prob_df = compute_transition_matrices(df)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "emotion_transition_analysis.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_timeline(axes[0], df)
    plot_transition_heatmap(axes[1], prob_df)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    # 控制台输出：关键转移摘要
    print("===== 情绪转移统计（前 5 个最常见转移） =====")
    transitions = list(zip(df["emotion"].iloc[:-1], df["emotion"].iloc[1:]))
    transition_series = pd.Series(transitions)
    for (src, dst), cnt in transition_series.value_counts().head(5).items():
        prob = prob_df.loc[src, dst]
        print(f"{src} -> {dst}: {cnt} 次, 概率 {prob:.2f}")

    print("\n===== 每个情绪的最可能后继 =====")
    for emo in EMOTION_LABELS:
        row = prob_df.loc[emo]
        top_next = row.idxmax()
        top_prob = row.max()
        print(f"{emo} -> {top_next}: {top_prob:.2f}")

    return save_path


if __name__ == "__main__":
    output_path = analyze_and_plot()
    print(f"\n[INFO] 情绪转移可视化已保存到：{output_path}")
