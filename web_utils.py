import os
from fpdf import FPDF
from datetime import datetime

# 引入你的分析脚本
from emotion_fluctuation_analysis import main as fluctuation_main
from emotion_transition_analysis import analyze_and_plot as transition_analyze

# --- 【关键修改】定义绝对路径 ---
# 获取当前脚本(web_utils.py)所在的文件夹路径，即项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 定义 result 文件夹的绝对路径
RESULT_DIR = os.path.join(BASE_DIR, 'result')


def generate_web_report(start_time, end_time, mode="webcam", video_path=None):
    """
    生成 Web 端报告
    """
    # 确保文件夹存在
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    # 切换工作目录到项目根目录，防止分析脚本找不到 data 文件夹
    os.chdir(BASE_DIR)

    print("Web端触发：生成情绪波动分析图...")
    fluctuation_main()

    print("Web端触发：生成情绪转移矩阵图...")
    transition_analyze()

    # --- PDF生成逻辑 ---
    # 字体路径也改成绝对路径
    font_path = os.path.join(BASE_DIR, "fonts", "STFANGSO.TTF")

    pdf = FPDF()
    pdf.add_page()

    font_name = "STFANGSO"
    if os.path.exists(font_path):
        pdf.add_font(font_name, "", font_path, uni=True)
    else:
        font_name = "Arial"
        print(f"Warning: 字体文件不存在于 {font_path}")

    # 标题
    pdf.set_font(font_name, "", 16)
    title = "实时情绪检测分析报告" if font_name == "STFANGSO" else "Emotion Analysis Report"
    pdf.cell(0, 15, title, 0, 1, "C")
    pdf.ln(2)

    # 信息
    pdf.set_font(font_name, "", 12)
    pdf.cell(0, 10, f"采集时间：{start_time} 至 {end_time}", 0, 1)

    if mode == "video" and video_path:
        # 只提取文件名，不带路径
        v_name = os.path.basename(video_path)
        pdf.cell(0, 10, f"输入源：视频文件 {v_name}", 0, 1)
    else:
        pdf.cell(0, 10, "输入源：实时摄像头", 0, 1)
    pdf.ln(2)

    # 图片1：波动图 (使用绝对路径)
    fluctuation_img_path = os.path.join(RESULT_DIR, "emotion_fluctuation_curve.png")
    pdf.set_font(font_name, "", 14)
    pdf.cell(0, 12, "情绪波动趋势图", 0, 1)
    if os.path.exists(fluctuation_img_path):
        pdf.image(fluctuation_img_path, x=10, w=190)
    pdf.ln(4)

    # 图片2：转移图 (使用绝对路径)
    transition_img_path = os.path.join(RESULT_DIR, "emotion_transition_analysis.png")
    pdf.set_font(font_name, "", 14)
    pdf.cell(0, 12, "情绪转移概率矩阵", 0, 1)
    if os.path.exists(transition_img_path):
        pdf.image(transition_img_path, x=10, w=190)

    # 保存 PDF (使用绝对路径)
    report_filename = f"report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    pdf_save_path = os.path.join(RESULT_DIR, report_filename)

    pdf.output(pdf_save_path)

    print(f"DEBUG: PDF 已保存到绝对路径: {pdf_save_path}")

    # 【重点】只返回文件名，不要返回路径
    return report_filename