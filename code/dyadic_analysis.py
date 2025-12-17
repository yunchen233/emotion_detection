import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# --- 预定义颜色方案 ---
FIXED_COLOR_MAP = {
    'Happy': '#FFD700', 'Surprised': '#FF8C00', 'Calm': '#32CD32',
    'Neutral': '#D3D3D3', 'Sad': '#4682B4', 'Scared': '#8A2BE2',
    'Angry': '#DC143C', 'Disgusted': '#8B4513', 'Contempt': '#FF69B4'
}


def perform_strict_matching(df, id_a, id_b):
    """严格序列配对逻辑"""
    df = df.sort_values('time')
    aligned_rows = []
    cache_a, cache_b = None, None
    time_a, time_b = None, None

    for _, row in df.iterrows():
        curr_id = row['face_id']
        curr_emo = row['emotion']
        curr_time = row['time']

        if curr_id == id_a:
            cache_a = curr_emo
            time_a = curr_time
        elif curr_id == id_b:
            cache_b = curr_emo
            time_b = curr_time

        if cache_a is not None and cache_b is not None:
            aligned_rows.append({
                'time': max(time_a, time_b) if time_a and time_b else curr_time,
                'emotion_a': cache_a,
                'emotion_b': cache_b
            })
            cache_a, cache_b = None, None
            time_a, time_b = None, None

    return pd.DataFrame(aligned_rows)

def analyze_dyadic_relationship(save_dir):
    csv_path = "../data/double_emotion_log.csv"
    if not os.path.exists(csv_path): return None

    try:
        df = pd.read_csv(csv_path)
    except:
        return None

    if 'face_id' not in df.columns: return None
    df['time'] = pd.to_datetime(df['time'])

    top_ids = df['face_id'].value_counts().head(2).index.tolist()
    if len(top_ids) < 2:
        print("[INFO] 人数不足，跳过分析")
        return None

    id_a, id_b = top_ids[0], top_ids[1]

    # 1. 执行配对
    df_filtered = df[df['face_id'].isin([id_a, id_b])].copy()
    data = perform_strict_matching(df_filtered, id_a, id_b)

    if len(data) < 5:
        print("[WARN] 有效数据太少")
        return None

    series_a = data['emotion_a']
    series_b = data['emotion_b']

    # 2. 计算指标
    sync_series = (series_a == series_b)
    overlap_rate = sync_series.sum() / len(data)

    crosstab_a_b = pd.crosstab(series_a, series_b, rownames=[f'ID {id_a}'], colnames=[f'ID {id_b}'])
    prob_a_to_b = crosstab_a_b.div(crosstab_a_b.sum(axis=1), axis=0).fillna(0)

    crosstab_b_a = pd.crosstab(series_b, series_a, rownames=[f'ID {id_b}'], colnames=[f'ID {id_a}'])
    prob_b_to_a = crosstab_b_a.div(crosstab_b_a.sum(axis=1), axis=0).fillna(0)

    # 3. 准备绘图数据
    unique_emotions = sorted(list(set(series_a.unique()) | set(series_b.unique())))
    color_dict = {emo: FIXED_COLOR_MAP.get(emo, "#808080") for emo in unique_emotions}

    # ==========================================
    # 全景仪表盘绘制 (3行布局)
    # ==========================================
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.2, 1.2])  # 第一行(时间轴)稍矮一点
    fig.suptitle(f"双人情绪交互全景报告 (ID {id_a} vs ID {id_b})\n总体同步率: {overlap_rate * 100:.1f}%", fontsize=18,
                 y=0.96)

    # --- Row 0: 时间轴彩条 (Timeline) ---
    ax_timeline = fig.add_subplot(gs[0, :])

    # 策略：如果数据超过300帧，为了显示清晰，只取最后300帧展示细节
    display_len = 300
    if len(data) > display_len:
        plot_data = data.tail(display_len)
        plot_sync = sync_series.tail(display_len)
        title_suffix = f" (展示最后 {display_len} 次交互)"
    else:
        plot_data = data
        plot_sync = sync_series
        title_suffix = " (全量数据)"

    idx_range = range(len(plot_data))

    # 画条带
    ax_timeline.barh(y=[3] * len(plot_data), width=1, left=idx_range, height=0.8,
                     color=[color_dict.get(e, '#808080') for e in plot_data['emotion_a']], edgecolor='none')
    ax_timeline.barh(y=[1] * len(plot_data), width=1, left=idx_range, height=0.8,
                     color=[color_dict.get(e, '#808080') for e in plot_data['emotion_b']], edgecolor='none')

    sync_colors = ['#FFD700' if s else '#F0F0F0' for s in plot_sync]
    ax_timeline.barh(y=[2] * len(plot_data), width=1, left=idx_range, height=0.4,
                     color=sync_colors, edgecolor='none')

    ax_timeline.set_yticks([1, 2, 3])
    ax_timeline.set_yticklabels([f"ID {id_b}", "同步", f"ID {id_a}"], fontsize=11, fontweight='bold')
    ax_timeline.set_title(f"情绪交互时间轴{title_suffix}", fontsize=14)
    ax_timeline.set_xlabel("交互次序 (Sequence)", fontsize=10)
    ax_timeline.set_xlim(0, len(plot_data))

    # 图例
    patches = [mpatches.Patch(color=color_dict[e], label=e) for e in unique_emotions]
    patches.append(mpatches.Patch(color='#FFD700', label='SYNC'))
    ax_timeline.legend(handles=patches, loc='upper left', bbox_to_anchor=(1.0, 1.0), title="情绪图例")

    # --- Row 1 Left: 分布对比 ---
    ax1 = fig.add_subplot(gs[1, 0])
    counts_a = series_a.value_counts(normalize=True).sort_index()
    counts_b = series_b.value_counts(normalize=True).sort_index()
    all_emos = sorted(list(set(counts_a.index) | set(counts_b.index)))

    x = np.arange(len(all_emos))
    ax1.bar(x - 0.175, [counts_a.get(e, 0) for e in all_emos], 0.35, label=f'ID {id_a}', color='skyblue', alpha=0.9)
    ax1.bar(x + 0.175, [counts_b.get(e, 0) for e in all_emos], 0.35, label=f'ID {id_b}', color='orange', alpha=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_emos, rotation=15)
    ax1.set_title("情绪占比分布", fontsize=14)
    ax1.legend()

    # --- Row 1 Right: 共现计数 ---
    ax2 = fig.add_subplot(gs[1, 1])
    sns.heatmap(crosstab_a_b, annot=True, fmt='d', cmap="Purples", ax=ax2)
    ax2.set_title("共现计数矩阵 (次数)", fontsize=14)
    ax2.set_ylabel(f"ID {id_a}")
    ax2.set_xlabel(f"ID {id_b}")

    # --- Row 2 Left: 概率 A->B ---
    ax3 = fig.add_subplot(gs[2, 0])
    sns.heatmap(prob_a_to_b, annot=True, fmt='.2f', cmap="Greens", ax=ax3)
    ax3.set_title(f"条件概率 P(B|A): 已知 {id_a} -> 预测 {id_b}", fontsize=14)
    ax3.set_ylabel(f"ID {id_a} (Condition)")
    ax3.set_xlabel(f"ID {id_b} (Response)")

    # --- Row 2 Right: 概率 B->A ---
    ax4 = fig.add_subplot(gs[2, 1])
    sns.heatmap(prob_b_to_a, annot=True, fmt='.2f', cmap="Blues", ax=ax4)
    ax4.set_title(f"条件概率 P(A|B): 已知 {id_b} -> 预测 {id_a}", fontsize=14)
    ax4.set_ylabel(f"ID {id_b} (Condition)")
    ax4.set_xlabel(f"ID {id_a} (Response)")

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # 留出右侧给图例

    if not os.path.exists(save_dir): os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "relationship_dashboard.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"[SUCCESS] 全景图表已保存: {save_path}")

    return save_path,id_a,id_b


def get_data():
    csv_path = "../data/double_emotion_log.csv"
    if not os.path.exists(csv_path): return None

    try:
        df = pd.read_csv(csv_path)
    except:
        return None

    if 'face_id' not in df.columns: return None
    df['time'] = pd.to_datetime(df['time'])

    top_ids = df['face_id'].value_counts().head(2).index.tolist()
    if len(top_ids) < 2:
        print("[INFO] 人数不足，跳过分析")
        return None

    id_a, id_b = top_ids[0], top_ids[1]

    # 1. 执行配对
    df_filtered = df[df['face_id'].isin([id_a, id_b])].copy()
    data = perform_strict_matching(df_filtered, id_a, id_b)

    if len(data) < 5:
        print("[WARN] 有效数据太少")
        return None

    series_a = data['emotion_a']
    series_b = data['emotion_b']

    # 2. 计算指标
    sync_series = (series_a == series_b)
    overlap_rate = sync_series.sum() / len(data)

    crosstab_a_b = pd.crosstab(series_a, series_b, rownames=[f'ID {id_a}'], colnames=[f'ID {id_b}'])
    prob_a_to_b = crosstab_a_b.div(crosstab_a_b.sum(axis=1), axis=0).fillna(0)

    crosstab_b_a = pd.crosstab(series_b, series_a, rownames=[f'ID {id_b}'], colnames=[f'ID {id_a}'])
    prob_b_to_a = crosstab_b_a.div(crosstab_b_a.sum(axis=1), axis=0).fillna(0)

    return series_a, series_b, sync_series, overlap_rate, prob_a_to_b, prob_b_to_a

if __name__ == "__main__":
    analyze_dyadic_relationship("../result")