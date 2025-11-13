#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import aiohttp
import cv2
import argparse
import sys
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.rtcrtpparameters import RTCRtpCapabilities
import json

class WHEPClient:
    def __init__(self, whep_url, output_path):
        self.whep_url = whep_url  # 例如: http://192.168.1.100:8889/cam/webrtc/
        self.output_path = output_path
        self.pc = RTCPeerConnection()
        self.frame = None
        self.got_frame = asyncio.Event()

    async def run(self):
        # 1. 创建 offer
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        # 2. POST offer 到 WHEP 端点
        async with aiohttp.ClientSession() as session:
            print(f"发送 SDP offer 到: {self.whep_url}")
            async with session.post(
                self.whep_url,
                headers={"Content-Type": "application/sdp"},
                data=self.pc.localDescription.sdp
            ) as resp:
                if resp.status != 201:
                    text = await resp.text()
                    print(f"WHEP 错误 {resp.status}: {text}")
                    sys.exit(1)
                answer_sdp = await resp.text()
                print("收到 SDP answer")

        # 3. 设置远程描述
        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer_sdp, type="answer")
        )

        # 4. 处理接收的轨道
        @self.pc.on("track")
        def on_track(track):
            if track.kind == "video":
                print("收到视频轨道，开始接收帧...")
                asyncio.create_task(self.handle_video_track(track))

        # 5. 等待抓到一帧
        try:
            await asyncio.wait_for(self.got_frame.wait(), timeout=15)
        except asyncio.TimeoutError:
            print("超时：未接收到视频帧")
            sys.exit(1)
        finally:
            await self.pc.close()

        # 6. 保存图像
        if self.frame is not None:
            success = cv2.imwrite(self.output_path, self.frame)
            print(f"图像已保存: {self.output_path}" if success else "保存失败")
        else:
            print("未获取到图像帧")
            sys.exit(1)

    async def handle_video_track(self, track):
        """接收第一帧后停止"""
        try:
            while not self.got_frame.is_set():
                frame = await track.recv()
                img = frame.to_ndarray(format="bgr24")
                if img.mean() > 10:  # 过滤全黑帧
                    self.frame = img
                    self.got_frame.set()
                    print("已抓取一帧")
                    break
                await asyncio.sleep(0.01)
        except Exception as e:
            print(f"接收帧错误: {e}")

async def main():
    parser = argparse.ArgumentParser(description="从 MediaMTX WHEP 流抓取一张图片")
    parser.add_argument("url", nargs="?", 
                        default="http://192.168.0.13:8889/cam/webrtc/whep/",
                        help="WHEP 端点 URL（默认: http://192.168.0.13:8889/cam/webrtc/whep/）")
    parser.add_argument("-o", "--output", default="whep_snapshot.jpg",
                        help="输出图片路径（默认: whep_snapshot.jpg）")
    args = parser.parse_args()

    client = WHEPClient(args.url, args.output)
    await client.run()

if __name__ == "__main__":
    asyncio.run(main())