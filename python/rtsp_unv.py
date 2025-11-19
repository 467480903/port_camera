import cv2
import threading
import tkinter as tk
from PIL import Image, ImageTk

RTSP_URL = "rtsp://user1:h7Hsu3ULLnLTs*M@192.168.1.13:554/media/video2"

class RTSPPlayer:
    def __init__(self, root, rtsp_url):
        self.root = root
        self.root.title("RTSP Video Player - Smooth")

        self.label = tk.Label(root)
        self.label.pack()

        self.frame = None
        self.cap = cv2.VideoCapture(rtsp_url)

        # 启动后台线程读取最新帧
        self.running = True
        threading.Thread(target=self.fetch_frames, daemon=True).start()

        # UI更新
        self.update_ui()

    def fetch_frames(self):
        """后台线程：循环读取 RTSP 最新帧，覆盖旧帧，避免延迟堆积"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def update_ui(self):
        """只显示 self.frame（最新帧）"""
        if self.frame is not None:
            img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=im)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

        self.root.after(15, self.update_ui)  # ~60 FPS UI刷新

    def __del__(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    player = RTSPPlayer(root, RTSP_URL)
    root.mainloop()
