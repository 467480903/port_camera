import cv2
import tkinter as tk
from PIL import Image, ImageTk

RTSP_URL = "rtsp://admin:gene2025@192.168.1.64:554/ISAPI/Streaming/channels/101"

RTSP_URL = "rtsp://user1:h7Hsu3ULLnLTs*M@192.168.1.13:554/media/video1"

class RTSPPlayer:
    def __init__(self, root, rtsp_url):
        self.root = root
        self.root.title("RTSP Video Player (200ms/frame)")

        # 创建显示区域
        self.label = tk.Label(root)
        self.label.pack()

        # 打开RTSP流
        self.cap = cv2.VideoCapture(rtsp_url)

        # 避免缓冲区堆积延迟
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 定时更新帧
        self.update_frame()

    def read_latest_frame(cap):
        frame = None
        while True:
            ret, temp = cap.read()
            if not ret:
                break
            frame = temp
        return frame

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # BGR 转 RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)


        self.root.after(10, self.update_frame)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    player = RTSPPlayer(root, RTSP_URL)
    root.mainloop()
