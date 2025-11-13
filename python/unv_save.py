#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import sys
import argparse
import time

def grab_rtsp_frame(rtsp_url, output_path, timeout=10):
    """
    从 RTSP 流中抓取一帧图像并保存。

    参数:
        rtsp_url:     RTSP 地址
        output_path:  保存路径（如 snapshot.jpg）
        timeout:      最大等待时间（秒）
    """
    # 建议使用 TCP 传输，避免 UDP 丢包导致黑屏
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    # 强制使用 TCP（对大部分摄像机有效）
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 可选：设置环境变量强制 TCP（更彻底）
    # import os
    # os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    start_time = time.time()
    frame = None

    print(f"正在连接 RTSP 流: {rtsp_url}")
    while time.time() - start_time < timeout:
        ret, img = cap.read()
        if ret:
            frame = img
            break
        else:
            print("未读取到帧，稍后重试...", end="\r")
        time.sleep(0.05)

    cap.release()

    if frame is None:
        print("\n错误：超时未获取到图像！")
        sys.exit(1)

    success = cv2.imwrite(output_path, frame)
    if success:
        print(f"图像已保存: {output_path}")
    else:
        print(f"保存失败: {output_path}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="从 RTSP 流抓取一张图片并保存")
    parser.add_argument("url", nargs="?", default="rtsp://user1:h7Hsu3ULLnLTs*M@192.168.1.13:554/media/video1",
                        help="RTSP 地址（默认使用代码中的地址）")
    parser.add_argument("-o", "--output", default="snapshot.jpg",
                        help="输出图片路径（默认: snapshot.jpg）")
    parser.add_argument("-t", "--timeout", type=int, default=10,
                        help="最大等待时间（秒，默认 10）")

    args = parser.parse_args()

    grab_rtsp_frame(args.url, args.output, args.timeout)


if __name__ == "__main__":
    main()