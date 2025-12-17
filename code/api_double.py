from openai import OpenAI
# 需要安装，直接复制这个pip install -U openai
import os
import csv
from datetime import datetime
import dyadic_analysis

# 初始化OpenAI客户端（兼容阿里云百炼）
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 直接读取你的环境变量里面的apikey
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# 这里是csv文件的信息读取
def read_emotion_csv(csv_path="../data/double_emotion_log.csv"):
    """读取情绪CSV文件，返回统计数据和关键信息"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件不存在：{csv_path}")
        # 分别存储两个人的数据
    person0_data = []
    person1_data = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {
                "time": row["time"],
                "emotion": row["emotion"],
                "face_id": int(row["face_id"])
            }

            if item["face_id"] == 0:
                person0_data.append(item)
            else:  # face_id == 1
                person1_data.append(item)

        # 处理第一个人（face_id=0）的数据

    def process_person_data(data, person_name):
        if not data:
            return None

        total_frames = len(data)
        start_time = datetime.fromisoformat(data[0]["time"])
        end_time = datetime.fromisoformat(data[-1]["time"])
        duration = (end_time - start_time).total_seconds()

        # 情绪分布统计
        emotion_counts = {}
        for item in data:
            emo = item["emotion"]
            emotion_counts[emo] = emotion_counts.get(emo, 0) + 1

        # 情绪序列
        emotion_sequence = [item["emotion"] for item in data]

        return {
            "person_name": person_name,
            "total_frames": total_frames,
            "duration": f"{duration:.1f}s",
            "emotion_counts": emotion_counts,
            "emotion_sequence": emotion_sequence,
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # 处理两个人数据

    person0_stats = process_person_data(person0_data, "人物A (face_id=0)")
    person1_stats = process_person_data(person1_data, "人物B (face_id=1)")

    return {
        "person0": person0_stats,
        "person1": person1_stats,
    }


# prompt在下面修改
def generate_analysis_prompt(series_a, series_b, sync_series, overlap_rate, prob_a_to_b, prob_b_to_a):  # 我这里prompt直接让AI生成的，主要是prompt里面的接口，具体功能未设计
    """生成模型分析用的Prompt"""
    return f"""
   【身份定位】你是一位拥有10年以上情绪心理学与人际关系研究经验的资深专家，擅长从双人情绪数据中解读互动模式和关系质量，能精准捕捉两人情绪协调性与相互影响，沟通时语气如亲和的心理顾问，既有专业严谨性又不失人文温度。

【核心任务】基于以下完整的双人情绪检测数据，完成两重目标：1. 输出专业、精准的双人情绪互动解读报告，需结合数据逻辑与心理学理论；2. 以关怀式语言沟通，让用户感受到对双方关系的深度理解，而非单纯的数据分析。

【警告】文字中不要出现如"#"，"*"这些非人类在正常说话时会使用的符号，请模仿人类用正常的语气说话，并且在给出关怀引导的时候不要直接说出来你在"关怀引导"，否则程序会崩溃。

【基础数据信息】
- 情绪同步率：{overlap_rate}（{sync_series}为同步情绪序列）
- id为0的人的情绪分布：{series_a}
- id为1的人情绪分布：{series_b}
- 条件概率矩阵：{prob_a_to_b}当id_0处于某种情绪时，id_1出现各种情绪的概率，需突出概率超过50%的高关联转换;{prob_b_to_a}当id_1处于某种情绪时，id_0出现各种情绪的概率，需突出概率超过50%的高关联转换

【关键发现】
- 同步性分析：
- 主导情绪对比：
- 关键情绪组合
- 高概率情绪影响：
【输出要求】严格遵循以下结构，语言风格统一为"专业结论+关怀引导"，避免生硬的数据罗列，每部分均需结合具体数据支撑观点，同时自然融入对双方关系状态的关注。

1. 双人情绪协调性分析（约200字）：
- 从情绪同步率出发，分析两人的情感协调程度（如"情绪同步率达到65%，表明你们在多数时刻能感知到彼此的情绪状态，形成良好的情感共鸣"）；
- 对比两人主导情绪的异同，分析情绪基调是否匹配（如"A的快乐主导与B的平静主导形成积极互补，而非冲突对立"）；
- 关怀衔接：基于协调性给出观察反馈，如"这样的情绪协调模式反映出你们之间有着不错的默契基础，是不是平时也比较注重彼此的情绪感受？"

2. 情绪互动模式解读（约250字）：
- 分析关键情绪组合背后的互动意义（如"当A焦虑时B平静的组合出现了6次，这可能意味着B在A焦虑时扮演了情绪稳定器的角色"）；
- 解读高概率情绪转换反映的影响关系（如"A的情绪状态对B有明显影响，当A快乐时有70%概率B也快乐，显示出较强的情绪传染效应"）；
- 结合同步序列，描述两人情绪是如何随时间互动的（如"在检测的前半段情绪同步率较低，后半段逐渐提高，可能反映出随互动深入情感协调性在改善"）；
- 关怀衔接：如"这种互动模式其实挺有意思的，是否在某些时刻你们能感觉到彼此情绪的相互影响，或者一方在主动调整以适应另一方？"

3. 关系质量与改善建议（约200字）：
- 基于所有数据，综合评估这段关系的情绪健康度（如"整体来看，情绪同步性中等但存在积极的互补模式，关系基础良好但仍有深化协调的空间"）；
- 从心理学角度给出关系发展的可能方向（如"如果希望进一步提升情感协调，可以尝试更多情绪表达的练习，或建立更明确的情绪信号系统"）；
- 关怀收尾：给出建设性小提示，如"每段关系都有其独特的情绪节奏，重要的是找到适合双方的协调方式。如果你们在某些时刻感受到情绪不同步，那或许正是深入了解对方的好机会。"

【禁忌提醒】避免使用"异常""问题""不良"等负面定性词汇，用"协调性""互动模式""调整空间"等中性表述；数据解读需精准，不夸大也不淡化，同时确保每句关怀语都自然衔接前文分析，不生硬割裂。
    """


# 保存结果，写入txt文件的函数
def save_analysis_to_txt(analysis_content, csv_path="../data/double_emotion_log.csv"):
    """将分析结果保存到TXT文件"""
    txt_filename = "api2分析结果.txt"

    results_dir = "../result"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 完整的文件路径
    full_path = os.path.join(results_dir, txt_filename)
    # 写入内容（包含基础信息+分析结果）
    with open(full_path, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write(f"情绪分析报告 - 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"分析的CSV文件：{csv_path}\n")
        f.write("=" * 50 + "\n\n")
        f.write(analysis_content)

    print(f"\n\n分析结果已保存至TXT文件：{txt_filename},路径为：{full_path}")
    return txt_filename


# here！主要的api运用分析在这里
def analyze_emotion_from_csv(csv_path="../data/double_emotion_log.csv"):
    """主函数：读取CSV→生成Prompt→调用API分析"""
    try:
        # 1. 读取CSV统计数据
        emotion_stats = read_emotion_csv(csv_path)
        series_a, series_b, sync_series, overlap_rate, prob_a_to_b, prob_b_to_a=dyadic_analysis.get_data()
        # 2. 调用Prompt
        prompt = generate_analysis_prompt(series_a, series_b, sync_series, overlap_rate, prob_a_to_b, prob_b_to_a)
        # 3. 调用阿里云百炼API
        messages = [{"role": "user", "content": prompt}]
        completion = client.chat.completions.create(
            model="deepseek-v3.2",  # 这里可以修改模型选择
            messages=messages,
            extra_body={"enable_thinking": False},  # 这里可以展示or不展示思考过程，True可以展示思考过程
            stream=True,
            stream_options={"include_usage": True}
        )

        # 4. 接收并打印结果
        print("\n" + "=" * 20 + "情绪分析结果" + "=" * 20 + "\n")
        answer_content = ""
        for chunk in completion:
            if not chunk.choices:
                print("\n" + "=" * 20 + "Token 消耗" + "=" * 20 + "\n")
                # 这个是官网上原始文献上复制的tokens数输出，用于测试时自己看资源消耗，不会保存在txt文档里面
                print(chunk.usage)
                continue
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                print(delta.content, end="", flush=True)
                answer_content += delta.content
        if answer_content:
            save_analysis_to_txt(answer_content, csv_path)

        return answer_content
    except Exception as e:
        print(f"分析失败：{str(e)}")
        return None


# 直接运行时执行分析
if __name__ == "__main__":
    # 可修改csv_path为具体文件路径
    analyze_emotion_from_csv(csv_path="../data/double_emotion_log.csv")