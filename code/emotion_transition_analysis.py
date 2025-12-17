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
    if len(codes) < 2:
        # 如果数据太少，生成一个空的单位矩阵防止报错
        n = len(EMOTION_LABELS)
        return pd.DataFrame(np.zeros((n,n)), index=EMOTION_LABELS, columns=EMOTION_LABELS), \
               pd.DataFrame(np.zeros((n,n)), index=EMOTION_LABELS, columns=EMOTION_LABELS)

    n = len(EMOTION_LABELS)
    count_matrix = np.zeros((n, n), dtype=int)#转移矩阵

    for i in range(len(codes) - 1):
        src = codes[i]
        dst = codes[i + 1]
        count_matrix[src, dst] += 1#从src转移到dst的次数加1

    #创建与计数矩阵相同形状的浮点矩阵，计算每行的总和，将计数矩阵除以行总和，得到转移概率，只有当行总和不等于0时才进行除法
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
    #绘制概率热力图
    im = ax.imshow(prob_df, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("情绪转移概率 ")
    ax.set_xlabel("下一步情绪")
    ax.set_ylabel("当前情绪")
    ax.set_xticks(range(len(EMOTION_LABELS)))
    ax.set_xticklabels(EMOTION_LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(EMOTION_LABELS)))
    ax.set_yticklabels(EMOTION_LABELS)

    #在所有单元格中显示概率值，显示两位小数
    for i in range(len(EMOTION_LABELS)):
        for j in range(len(EMOTION_LABELS)):
            prob = prob_df.iloc[i, j]
            ax.text(
                j, i, f"{prob:.2f}",
                ha="center", va="center",
                fontsize=9,
                color="black" if prob < 0.6 else "white"#概率小于0.6用黑色，否则用白色（提高可读性）
            )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def analyze_and_plot(csv_path: str = None, save_dir: str = None) -> str:
    if csv_path is None:
        csv_path = os.path.join(PROJECT_ROOT, "..","data", "emotion_log.csv")
    if save_dir is None:
        save_dir = os.path.join(PROJECT_ROOT,"..", "result")

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

def get_pro_df():#api调用接口
    csv_path = os.path.join(PROJECT_ROOT, "..", "data", "emotion_log.csv")
    df = load_emotion_series(csv_path)
    count_df,prob_df=compute_transition_matrices(df)
    return count_df, prob_df

if __name__ == "__main__":
    analyze_and_plot()