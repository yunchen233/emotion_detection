文件说明  
train_model.py 模型主体，调用即可训练模型  
confusion_matrix.py 模型评估代码-混淆矩阵（看test中每类样本模型的分拣情况）  
real_time_detection.py 单人模式下的实时摄像头文件+pdf报告生成文件  
ssnapshot_manager.py  抓拍器：记录每类情绪置信度最高的时刻照片截图   
emotion_fluctuatation_analysis.py 情绪变化散点图(只支持前400帧),属于单人模式报告中  
emotion_transition_analysis.py 情绪转移概率矩阵，属于单人模式报告中  
api.py llm接入数据分析，单人报告模式  
real_time_detection_double.py 双人模式下的实时摄像头+pdf报告生成文件  
simple_tracker.py 区分用户的追踪器，按照移动距离  
dyadic_analysis.py 双人模式的两者关系分析图表(包括时间轴彩条与重复区间的highlight，情绪占比分布柱状图，情绪共线次数矩阵，两个用户之间情绪相互影响的条件概率矩阵)  
api_double.py  双人模式下接入的llm数据分析  
upload_video.py  视频上传接口，只做label标注，不会产生报告（调用时直接加 --input+视频路径）  

网页运行方法  
命令行在code路径里运行python app.py  
浏览器打开http://127.0.0.1:5000  

api需要在系统变量里面配置  
变量名：DASHSCOPE_API_KEY  
变量值：自己配一下

命令行终端运行pip install -r requirements.txt一键安装所有所需库  

！！！运行网页是请务必务必保证网络通畅且不要挂梯子，否则api接的阿里云，登不上阿里云网站就无法正常显示llm的返回结果
