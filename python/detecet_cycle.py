#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import sys
import argparse
import time
import json
import os
import numpy as np   
from datetime import datetime

def grab_rtsp_frame(rtsp_url, timeout=10):
    """抓取一帧，返回 frame 或 None"""
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    start_time = time.time()
    while time.time() - start_time < timeout:
        ret, frame = cap.read()
        if ret:
            cap.release()
            return frame
        time.sleep(0.05)
    cap.release()
    return None

def detect_largest_circle(frame):
    """返回最大圆信息: (x, y, radius) 或 None"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
        param1=100, param2=40, minRadius=50, maxRadius=500
    )
    if circles is None:
        return None
    circles = np.round(circles[0, :]).astype("int")
    x, y, r = max(circles, key=lambda c: c[2])
    return x, y, r

def main():
    parser = argparse.ArgumentParser(description="RTSP 抓图 + 检测最大圆 → JSON 输出")
    parser.add_argument("url", nargs="?", 
                        default="rtsp://user1:h7Hsu3ULLnLTs*M@192.168.1.13:554/media/video1",
                        help="RTSP 地址")
    parser.add_argument("-t", "--timeout", type=int, default=10, help="超时时间（秒）")
    parser.add_argument("--pretty", action="store_true", help="美化 JSON 输出（调试用）")
    args = parser.parse_args()

    # 1. 抓图
    frame = grab_rtsp_frame(args.url, args.timeout)
    if frame is None:
        result = {
            "success": False,
            "error": "timeout",
            "message": "无法从 RTSP 流获取图像",
            "timestamp": datetime.now().isoformat()
        }
    else:
        # 2. 检测圆
        circle = detect_largest_circle(frame)
        h, w = frame.shape[:2]
        if circle:
            x, y, r = circle
            result = {
                "success": True,
                "circle": {
                    "center": {"x": int(x), "y": int(y)},
                    "radius": int(r),
                    "diameter": int(r * 2)
                },
                "image": {
                    "width": int(w),
                    "height": int(h)
                },
                "timestamp": datetime.now().isoformat()
            }
        else:
            result = {
                "success": False,
                "error": "no_circle",
                "message": "未检测到有效圆形",
                "image": {"width": int(w), "height": int(h)},
                "timestamp": datetime.now().isoformat()
            }

    # 3. 输出 JSON
    if args.pretty:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()