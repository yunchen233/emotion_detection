
from openai import OpenAI
#需要安装，直接复制这个pip install -U openai
import os
import csv
from datetime import datetime
import emotion_transition_analysis

# 初始化OpenAI客户端（兼容阿里云百炼）
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"), #直接读取你的环境变量里面的apikey
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
#这里是csv文件的信息读取
def read_emotion_csv(csv_path="..data/emotion_log.csv"):
    """读取情绪CSV文件，返回统计数据和关键信息"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件不存在：{csv_path}")
    
    emotion_data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            emotion_data.append({
                "time": row["time"],
                "emotion": row["emotion"]
            })
    
    # 核心统计信息（精简数据，避免prompt过长）
    total_frames = len(emotion_data)
    start_time = datetime.fromisoformat(emotion_data[0]["time"])
    end_time = datetime.fromisoformat(emotion_data[-1]["time"])
    duration = (end_time - start_time).total_seconds()  # 总时长（秒）
    
    # 情绪分布统计
    emotion_counts = {}
    for item in emotion_data:
        emo = item["emotion"]
        emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
    
    # 情绪序列 
    emotion_sequence = [item["emotion"] for item in emotion_data]
    
    return {
        "total_frames": total_frames,
        "duration": f"{duration:.1f}s",
        "emotion_counts": emotion_counts,
        "emotion_sequence": emotion_sequence,
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S")
    }
#prompt在下面修改
def generate_analysis_prompt(emotion_stats,count_df,prob_df):#我这里prompt直接让AI生成的，主要是prompt里面的接口，具体功能未设计
    """生成模型分析用的Prompt"""
    return f"""
   【身份定位】你是一位拥有10年以上情绪心理学与数据解读经验的资深专家，不仅精通情绪数据的专业分析，更擅长用共情视角将复杂数据转化为温暖易懂的解读，能精准捕捉数据背后的心理需求，沟通时语气如亲和的心理顾问，既有专业严谨性又不失人文温度。
【核心任务】基于以下完整的情绪检测数据，完成两重目标：1. 输出专业、精准的情绪解读报告，需结合数据逻辑与心理学理论；2. 以关怀式语言与用户沟通，让用户感受到被理解，而非单纯的数据分析。
【警告】文字中不要出现如“#”，“*”这些非人类在正常说话时会使用的符号，请模仿人类用正常的语气说话，并且在给出关怀引导的时候不要直接说出来你在“关怀引导”，否则程序会崩溃
【基础数据信息】
- 检测时段：{emotion_stats['start_time']} 至 {emotion_stats['end_time']}
- 检测总时长：{emotion_stats['duration']}
- 数据采集规模：共采集 {emotion_stats['total_frames']} 帧有效数据
- 核心数据支撑：
  - 1. 情绪分布数据：{emotion_stats['emotion_counts']}（请重点关注占比超20%的主导情绪及占比低于5%的稀有情绪）
  - 2. 情绪时间序列：{emotion_stats['emotion_sequence']}（请按时间线梳理情绪流转规律）
  - 3. 情绪转变次数矩阵：{count_df}（矩阵含义为"从行情绪转变为列情绪的实际次数"，需标注转变次数超3次的关键组合）
  - 4. 情绪转变概率矩阵：{prob_df}（矩阵含义为"从行情绪转变为列情绪的概率值"，需突出概率超50%的高关联转变）
【输出要求】严格遵循以下结构，语言风格统一为"专业结论+关怀引导"，避免生硬的数据罗列，每部分均需结合具体数据支撑观点，同时自然融入对用户状态的关注。
1. 整体情绪倾向（约200字）：
- 明确指出检测时段内的主导情绪（需标注具体占比，如"愉悦情绪占比42%，为核心主导情绪"）；
- 分析情绪稳定性（结合情绪转变总次数，如"全程情绪转变仅2次，整体状态稳定"或"1小时内转变7次，情绪活跃度高但稳定性较弱"）；
- 总结情绪变化整体趋势（如"从焦虑逐步过渡到平静，呈现积极好转态势"）；
- 关怀衔接：基于整体状态给出温和反馈，如"这样的情绪状态其实很贴合近期可能的生活节奏，你是否也觉得这段时间心态在慢慢调整？"。
2. 关键变化节点（约150字）：
- 精准定位情绪突变时间点（结合情绪序列的时间标记，如"(某个具体时间点)出现明显突变，从'烦躁'直接转为'平静'"）；
- 分析突变前后的情绪特征及转变逻辑（结合转变矩阵，如"此次转变对应矩阵中'烦躁→平静'的高概率组合，符合情绪自我调节的规律"）；
- 关怀衔接：如"这个时间点的情绪变化还挺明显的，是不是当时发生了什么让你心态松动的小事？"。
3. 心理状态解读（约200字）：
- 基于情绪波动特征，结合心理学常识给出合理推测（如"主导情绪为'平静'但伴随3次短暂'焦虑'波动，推测可能处于'稳定但有轻微压力'的心理状态"）；
- 关联情绪转变规律分析深层需求（如"'专注→疲惫'的高频转变，可能反映出近期有集中精力处理的事务，身体在发出需要休息的信号"）；
- 关怀收尾：给出建设性小提示，如"这样的心理状态很正常，要是觉得压力偶尔冒出来，不妨试试短暂暂停手头的事，做个深呼吸，或许会更舒服一些"。
【禁忌提醒】避免使用"异常""问题""不良"等负面定性词汇，用"波动""变化""调整"等中性表述；数据解读需精准，不夸大也不淡化，同时确保每句关怀语都自然衔接前文分析，不生硬割裂。
    """
#保存结果，写入txt文件的函数
def save_analysis_to_txt(analysis_content, csv_path="../data/emotion_log.csv"):
    """将分析结果保存到TXT文件"""
    txt_filename = "api1分析结果.txt"

    results_dir = "../result"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 完整的文件路径
    full_path = os.path.join(results_dir, txt_filename)
    # 写入内容（包含基础信息+分析结果）
    with open(full_path, "w", encoding="utf-8") as f:
        f.write("="*50 + "\n")
        f.write(f"情绪分析报告 - 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"分析的CSV文件：{csv_path}\n")
        f.write("="*50 + "\n\n")
        f.write(analysis_content)
    
    print(f"\n\n分析结果已保存至TXT文件：{txt_filename},路径为：{full_path}")
    return txt_filename
#here！主要的api运用分析在这里
def analyze_emotion_from_csv(csv_path="../data/emotion_log.csv"):
    """主函数：读取CSV→生成Prompt→调用API分析"""
    try:
        # 1. 读取CSV统计数据
        emotion_stats = read_emotion_csv(csv_path)
        count_df,prob_df=emotion_transition_analysis.get_pro_df()
        # 2. 调用Prompt
        prompt = generate_analysis_prompt(emotion_stats,count_df,prob_df)
        # 3. 调用阿里云百炼API
        messages = [{"role": "user", "content": prompt}]
        completion = client.chat.completions.create(
            model="deepseek-v3.2",#这里可以修改模型选择
            messages=messages,
            extra_body={"enable_thinking": False},  # 这里可以展示or不展示思考过程，True可以展示思考过程
            stream=True,
            stream_options={"include_usage": True}
        )
        
        # 4. 接收并打印结果
        print("\n" + "="*20 + "情绪分析结果" + "="*20 + "\n")
        answer_content = ""
        for chunk in completion:
            if not chunk.choices:
                print("\n" + "="*20 + "Token 消耗" + "="*20 + "\n")
            #这个是官网上原始文献上复制的tokens数输出，用于测试时自己看资源消耗，不会保存在txt文档里面
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
    analyze_emotion_from_csv(csv_path="../data/emotion_log.csv")