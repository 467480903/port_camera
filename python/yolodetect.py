#!/usr/bin/env python3
"""
YOLO检测显示程序 - 显示所有检测框
"""
import torch
from ultralytics import YOLO
import cv2

# 加载模型
print("正在加载模型...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("../yoloTrain3/runs/detect/train/weights/best.pt", task="detect").to(device)

# 打开视频
print("正在打开视频...")
cap = cv2.VideoCapture('/home/yy/10.10.95.219_001M_202601091536527BD2.mp4')

if not cap.isOpened():
    print("错误: 无法打开视频文件 my.mp4")
    exit()

# 获取视频信息
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"视频信息: {width}x{height}, {fps} FPS, 共 {total_frames} 帧")
print("开始检测... (按 'q' 退出, 按 'p' 暂停/继续)")

frame_count = 0
paused = False

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        
        if not ret:
            print("视频播放完毕")
            break
        
        frame_count += 1
        
        # YOLO检测
        results = model(frame, conf=0.25, verbose=False)
        
        # 获取检测结果
        boxes = results[0].boxes
        
        # 在原图上绘制所有检测框
        for box in boxes:
            # 获取边界框坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 获取置信度和类别
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]
            
            # 绘制边界框（绿色，线宽2）
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 准备标签文本
            label = f'{class_name} {conf:.2f}'
            
            # 获取文本大小
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # 绘制标签背景
            cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), (0, 255, 0), -1)
            
            # 绘制标签文字
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # 显示检测统计信息
        info_text = f'Frame: {frame_count}/{total_frames} | Objects: {len(boxes)}'
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # 显示图像
    cv2.imshow('YOLO Detection - All Boxes', frame)
    
    # 控制播放速度和交互
    key = cv2.waitKey(int(1000/fps)) & 0xFF
    
    if key == ord('q'):
        print("用户退出")
        break
    elif key == ord('p'):
        paused = not paused
        print("暂停" if paused else "继续")

# 释放资源
cap.release()
cv2.destroyAllWindows()

print(f"完成! 共处理 {frame_count} 帧")