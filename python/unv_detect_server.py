#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import threading
import cv2
import numpy as np
import json
import time
import os
from datetime import datetime
import glob
import os

# ==================== 配置区 ====================
RTSP_URL = "rtsp://user1:h7Hsu3ULLnLTs*M@192.168.1.13:554/media/video1"
TCP_HOST = "0.0.0.0"
TCP_PORT = 1991
TIMEOUT = 10  # 抓图超时（秒）
# ================================================

# 强制使用 TCP 传输
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

def grab_frame():

    # 查找所有 .jpg 文件
    jpg_files = glob.glob("*.jpg")

    # 删除所有找到的文件
    for file in jpg_files:
        os.remove(file)
        print(f"已删除: {file}")

    print(f"共删除 {len(jpg_files)} 个 .jpg 文件")

    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    start = time.time()
    while time.time() - start < TIMEOUT:
        ret, frame = cap.read()
        if ret:
            cap.release()
            return frame
        time.sleep(0.05)
    cap.release()
    return None

def detect_circle(frame):
    # 1. 转为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. 过滤掉灰度值在 50-255 范围内的像素点
    # 创建掩码：保留灰度值 < 50 的像素，其他设为 255（白色）
    filtered = np.where(gray < 50, gray, 255).astype(np.uint8)
    
    # 保存灰度过滤图
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filtered_filename = f"debug_filtered_{timestamp}.jpg"
    cv2.imwrite(filtered_filename, filtered)
    print(f"[调试] 已保存灰度过滤图: {filtered_filename}")
    
    # 3. 对过滤后的图像进行高斯模糊
    blurred = cv2.GaussianBlur(filtered, (9, 9), 2)
    
    # 保存高斯模糊图
    blurred_filename = f"debug_blurred_{timestamp}.jpg"
    cv2.imwrite(blurred_filename, blurred)
    print(f"[调试] 已保存高斯模糊图: {blurred_filename}")
    
    # 4. 霍夫圆检测
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=200,        # 增加最小圆心距离，避免检测重叠圆（原100→200）
        param1=100,         # Canny边缘检测高阈值
        param2=120,          # 累加器阈值，越大检测越严格（原40→60）
        minRadius=210,      # 增加最小半径，过滤小圆（原50→100）
        maxRadius=450       # 减小最大半径，过滤超大圆（原500→400）
    )

    print(circles)
    
    if circles is None:
        return None
    
    circles = np.round(circles[0, :]).astype("int")
    
    # 5. 在原图上绘制所有检测到的圆
    result_img = frame.copy()
    
    # 遍历所有圆并绘制
    for i, (x, y, r) in enumerate(circles):
        # 绘制圆形外圈（绿色，线宽2）
        cv2.circle(result_img, (x, y), r, (0, 255, 0), 2)
        # 绘制圆心（红色，填充）
        cv2.circle(result_img, (x, y), 5, (0, 0, 255), -1)
        # 添加文本标注（显示编号和半径）
        text = f"#{i+1} R:{r}"
        cv2.putText(result_img, text, (x - 30, y - r - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 在图片左上角显示检测到的圆的总数
    summary_text = f"Total Circles: {len(circles)}"
    cv2.putText(result_img, summary_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    # 保存标注后的结果图
    result_filename = f"debug_result_{timestamp}.jpg"
    cv2.imwrite(result_filename, result_img)
    print(f"[调试] 已保存圆心检测结果图: {result_filename} (检测到 {len(circles)} 个圆)")
    
    # 返回最大的圆（保持原有逻辑）
    x, y, r = max(circles, key=lambda c: c[2])
    return x, y, r

def handle_client(conn, addr):
    print(f"[连接] {addr}")
    try:
        buffer = ""
        while True:
            data = conn.recv(1024).decode('utf-8')
            if not data:
                break
            buffer += data

            # 按换行符分割完整消息
            while '\n' in buffer:
                message, buffer = buffer.split('\n', 1)
                message = message.strip()
                if not message:
                    continue

                print(f"[收到] {addr} -> {message}")

                # 解析 JSON 命令
                try:
                    cmd = json.loads(message)
                except json.JSONDecodeError as e:
                    response = {
                        "success": False,
                        "error": "invalid_json",
                        "message": f"JSON 解析失败: {e}",
                        "timestamp": datetime.now().isoformat()
                    }
                    conn.sendall((json.dumps(response, ensure_ascii=False) + "\n").encode('utf-8'))
                    continue

                # 提取 cmd 字段
                cmd_name = cmd.get("cmd", "").strip().lower()

                # 支持 detecet 和 detect（兼容拼写错误）
                if cmd_name in ["detecet", "detect"]:
                    frame = grab_frame()
                    if frame is None:
                        result = {
                            "success": False,
                            "error": "timeout",
                            "message": "无法从 RTSP 流获取图像",
                            "timestamp": datetime.now().isoformat()
                        }
                    else:
                        circle = detect_circle(frame)
                        h, w = frame.shape[:2]
                        if circle:
                            x, y, r = circle
                            result = {
                                "success": True,
                                "cmd": cmd_name,
                                "circle": {
                                    "center": {"x": int(x), "y": int(y)},
                                    "radius": int(r),
                                    "diameter": int(r * 2)
                                },
                                "image": {"width": int(w), "height": int(h)},
                                "timestamp": datetime.now().isoformat()
                            }
                        else:
                            result = {
                                "success": False,
                                "error": "no_circle",
                                "cmd": cmd_name,
                                "message": "未检测到有效圆形",
                                "image": {"width": int(w), "height": int(h)},
                                "timestamp": datetime.now().isoformat()
                            }
                    response_str = json.dumps(result, ensure_ascii=False) + "\n"
                    conn.sendall(response_str.encode('utf-8'))
                    print(f"[响应] {addr} -> {response_str.strip()[:100]}...")

                elif cmd_name == "ping":
                    response = {
                        "success": True,
                        "cmd": "ping",
                        "message": "pong",
                        "timestamp": datetime.now().isoformat()
                    }
                    conn.sendall((json.dumps(response, ensure_ascii=False) + "\n").encode('utf-8'))

                else:
                    response = {
                        "success": False,
                        "error": "unknown_command",
                        "cmd": cmd_name,
                        "message": f"未知命令: {cmd_name}",
                        "timestamp": datetime.now().isoformat()
                    }
                    conn.sendall((json.dumps(response, ensure_ascii=False) + "\n").encode('utf-8'))

    except Exception as e:
        print(f"[错误] {addr} -> {e}")
    finally:
        conn.close()
        print(f"[断开] {addr}")

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((TCP_HOST, TCP_PORT))
    server.listen(5)
    print(f"[服务启动] 监听 {TCP_HOST}:{TCP_PORT}")
    print(f"   发送: {{\"cmd\": \"detect\"}} → 执行检测")
    print(f"   发送: {{\"cmd\": \"ping\"}} → 测试连接")

    while True:
        try:
            conn, addr = server.accept()
            thread = threading.Thread(target=handle_client, args=(conn, addr))
            thread.daemon = True
            thread.start()
        except KeyboardInterrupt:
            print("\n[停止] 服务关闭")
            break

if __name__ == "__main__":
    start_server()
    # frame = grab_frame()
    # circle = detect_circle(frame)