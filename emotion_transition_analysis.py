import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# 字体设置
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

EMOTION_LABELS = ['Angry', 'Disgusted', 'Scared', 'Happy', 'Sad', 'Surprised', 'Calm','Contempt']
EMOTION_TO_INT = {name: idx for idx, name in enumerate(EMOTION_LABELS)}

# --- 【关键修改】锁定当前项目根目录 ---
# 既然脚本就在根目录下，dirname(__file__) 就是根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_emotion_series(csv_path: str) -> pd.DataFrame:
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
    codes = df["emotion_code"].to_numpy()
    # 放宽限制：只要有数据就行，不需要强制 >=2 条（防止极短测试报错）
    if len(codes) < 2:
        # 如果数据太少，生成一个空的单位矩阵防止报错
        n = len(EMOTION_LABELS)
        return pd.DataFrame(np.zeros((n,n)), index=EMOTION_LABELS, columns=EMOTION_LABELS), \
               pd.DataFrame(np.zeros((n,n)), index=EMOTION_LABELS, columns=EMOTION_LABELS)

    n = len(EMOTION_LABELS)
    count_matrix = np.zeros((n, n), dtype=int)

    for i in range(len(codes) - 1):
        src = codes[i]
        dst = codes[i + 1]
        count_matrix[src, dst] += 1

    prob_matrix = np.zeros_like(count_matrix, dtype=float)
    row_sums = count_matrix.sum(axis=1, keepdims=True)
    np.divide(
        count_matrix,
        row_sums,
        out=prob_matrix,
        where=row_sums != 0
    )

    count_df = pd.DataFrame(count_matrix, index=EMOTION_LABELS, columns=EMOTION_LABELS)
    prob_df = pd.DataFrame(prob_matrix, index=EMOTION_LABELS, columns=EMOTION_LABELS)
    return count_df, prob_df

def plot_transition_heatmap(ax, prob_df: pd.DataFrame) -> None:
    im = ax.imshow(prob_df, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("情绪转移概率 (一阶马尔可夫链)")
    ax.set_xlabel("下一步情绪")
    ax.set_ylabel("当前情绪")
    ax.set_xticks(range(len(EMOTION_LABELS)))
    ax.set_xticklabels(EMOTION_LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(EMOTION_LABELS)))
    ax.set_yticklabels(EMOTION_LABELS)

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
    # --- 【关键修改】使用绝对路径 ---
    if csv_path is None:
        csv_path = os.path.join(PROJECT_ROOT, "data", "emotion_log.csv")
    if save_dir is None:
        save_dir = os.path.join(PROJECT_ROOT, "result")

    df = load_emotion_series(csv_path)
    
    # 防止空数据绘图报错
    if df.shape[0] < 2:
        print("数据过少，跳过转移矩阵绘图")
        return ""

    count_df, prob_df = compute_transition_matrices(df)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "emotion_transition_analysis.png")
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plot_transition_heatmap(ax, prob_df)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    return save_path


if __name__ == "__main__":
    analyze_and_plot()