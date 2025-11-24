import os
import pandas as pd
import matplotlib.pyplot as plt

EMOTION_LABELS = ['Angry', 'Disgusted', 'Scared', 'Happy', 'Sad', 'Surprised', 'Calm']

def load_emotion_series(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到数据文件：{csv_path}")

    df = pd.read_csv(csv_path)

    if "time" not in df.columns or "emotion" not in df.columns:
        raise ValueError("CSV 必须包含 time 和 emotion 列")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    return df


def plot_single_emotion(df: pd.DataFrame, emotion: str, save_dir: str):
    df_em = df[df["emotion"] == emotion]
    if df_em.empty:
        print(f"[WARN] 情绪 {emotion} 在数据中未出现，跳过生成图像。")
        return

    plt.figure(figsize=(12, 4))
    plt.plot(df_em["time"], [1] * len(df_em), "o-", markersize=6)

    plt.title(f"{emotion} 随时间出现曲线", fontproperties="SimHei")
    plt.xlabel("时间", fontproperties="SimHei")
    plt.yticks([])
    plt.grid(True, linestyle="--", alpha=0.4)

    save_path = os.path.join(save_dir, f"{emotion}_curve.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[INFO] 已生成图像：{save_path}")


def main():
    csv_path = "../data/emotion_log.csv"
    save_dir = "../result"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    df = load_emotion_series(csv_path)

    print("===== 开始生成七类情绪的独立折线图 =====")
    for emo in EMOTION_LABELS:
        plot_single_emotion(df, emo, save_dir)

    print("===== 全部图像生成完毕 =====")


if __name__ == "__main__":
    main()
