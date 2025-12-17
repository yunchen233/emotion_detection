from openai import OpenAI
#需要安装，直接复制这个pip install -U openai
import os
import csv
from datetime import datetime

# 初始化OpenAI客户端（兼容阿里云百炼）
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"), #直接读取你的环境变量里面的apikey
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
#这里是csv文件的信息读取
def read_emotion_csv(csv_path="emotion_log.csv"):
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
def generate_analysis_prompt(emotion_stats):#我这里prompt直接让AI生成的，主要是prompt里面的接口，具体功能未设计
    """生成模型分析用的Prompt"""
    return f"""
    你是一名自身心理咨询师，请分析以下情绪检测数据，输出专业的解读报告，同时温和地与用户沟通：
    1. 基础信息：
    -检测时段 {emotion_stats['start_time']} 至 {emotion_stats['end_time']}，
    -总时长 {emotion_stats['duration']}，
    -共采集 {emotion_stats['total_frames']} 帧数据
    2. 情绪分布：{emotion_stats['emotion_counts']}
    3. 情绪序列：{emotion_stats['emotion_sequence']}

    请按以下结构输出，以有人文关怀的语气输出：
    - 整体情绪倾向：总结主导情绪、情绪稳定性
    - 关键变化节点：指出情绪突变的时间点及变化趋势
    - 心理状态解读：基于情绪波动给出合理的心理状态推测
    """
#保存结果，写入txt文件的函数
def save_analysis_to_txt(analysis_content, csv_path="emotion_log.csv"):
    """将分析结果保存到TXT文件"""
    # 生成带时间戳的文件名，格式：emotion_analysis_20251217_153020.txt
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.basename(csv_path).replace(".csv", "")
    txt_filename = f"{csv_filename}_analysis_{timestamp}.txt"
    
    # 写入内容（包含基础信息+分析结果）
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write("="*50 + "\n")
        f.write(f"情绪分析报告 - 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"分析的CSV文件：{csv_path}\n")
        f.write("="*50 + "\n\n")
        f.write(analysis_content)
    
    print(f"\n\n分析结果已保存至TXT文件：{txt_filename}")
    return txt_filename
#here！主要的api运用分析在这里
def analyze_emotion_from_csv(csv_path="emotion_log.csv"):
    """主函数：读取CSV→生成Prompt→调用API分析"""
    try:
        # 1. 读取CSV统计数据
        emotion_stats = read_emotion_csv(csv_path)
        # 2. 调用Prompt
        prompt = generate_analysis_prompt(emotion_stats)
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
    analyze_emotion_from_csv(csv_path="emotion_log.csv")