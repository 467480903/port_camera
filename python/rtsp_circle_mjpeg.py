import cv2
import threading
import tkinter as tk
from PIL import Image, ImageTk
import socket
import json
import numpy as np
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import io

RTSP_URL = "rtsp://user1:h7Hsu3ULLnLTs*M@192.168.1.13:554/media/video2"
TCP_HOST = "0.0.0.0"
TCP_PORT = 1991



# =============================
#        MJPEG HTTP Handler
# =============================
class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/video":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Age", 0)
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        while True:
            frame = self.server.player.get_mjpeg_frame()
            if frame is None:
                continue
            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            except:
                break


# =============================
#        主类：RTSP + 检测
# =============================
class RTSPPlayer:
    def __init__(self, root, rtsp_url):
        self.root = root
        self.root.title("RTSP Video Player - Circle Detection")

        # Tkinter 显示区域（可保留）
        self.label = tk.Label(root)
        self.label.pack()

        # 按钮区域
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        self.detect_btn = tk.Button(button_frame, text="识别圆", command=self.detect_circle,
                                    bg="#4CAF50", fg="white", padx=20, pady=5)
        self.detect_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = tk.Button(button_frame, text="删除圆", command=self.clear_circles,
                                   bg="#f44336", fg="white", padx=20, pady=5)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(root, text="就绪", fg="blue")
        self.status_label.pack()

        self.frame = None
        self.cap = cv2.VideoCapture(rtsp_url)
        self.detected_circles = []

        # 多线程
        self.running = True
        threading.Thread(target=self.fetch_frames, daemon=True).start()
        threading.Thread(target=self.tcp_server, daemon=True).start()
        threading.Thread(target=self.start_mjpeg_server, daemon=True).start()

        self.update_ui()

    # ----------------------------------------------------------

    def fetch_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    # ----------------------------------------------------------

    def tcp_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((TCP_HOST, TCP_PORT))
        server.listen(5)
        print(f"[TCP] 服务器启动: {TCP_HOST}:{TCP_PORT}")

        while self.running:
            try:
                client, addr = server.accept()
                threading.Thread(target=self.handle_client, args=(client, addr), daemon=True).start()
            except:
                break

    def handle_client(self, conn, addr):
        print(f"[连接] {addr}")
        buffer = ""
        try:
            while True:
                data = conn.recv(1024).decode('utf-8')
                if not data:
                    break
                buffer += data
                while '\n' in buffer:
                    msg, buffer = buffer.split('\n', 1)
                    msg = msg.strip()
                    if not msg:
                        continue

                    try:
                        cmd = json.loads(msg)
                    except:
                        conn.sendall(b'{"success":false,"error":"invalid_json"}\n')
                        continue

                    if cmd.get("cmd") == "detect":
                        result = self.detect_circle()
                        conn.sendall((json.dumps(result, ensure_ascii=False) + "\n").encode('utf-8'))

        finally:
            conn.close()
            print(f"[断开] {addr}")

    def is_black_edge_circle(self, gray_img, x, y, r, threshold=80):
        """检查圆边缘是否为黑色"""
        # 采样圆周上的点
        angles = np.linspace(0, 2 * np.pi, 16)
        edge_values = []
        
        for angle in angles:
            px = int(x + r * np.cos(angle))
            py = int(y + r * np.sin(angle))
            
            if 0 <= px < gray_img.shape[1] and 0 <= py < gray_img.shape[0]:
                edge_values.append(gray_img[py, px])
        
        # 如果边缘平均灰度值较低，认为是黑边框
        return len(edge_values) > 0 and np.mean(edge_values) < threshold

    def detect_circle(self):
        """检测黑边框圆"""
        if self.frame is None:
            self.status_label.config(text="无视频帧", fg="red")
            return {"status": "error", "message": "无视频帧"}

        # 复制当前帧进行处理
        frame_copy = self.frame.copy()
        img = frame_copy


            # 1. 转灰度 + 轻微模糊（保留粗黑边）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)        # 中值模糊最抗盐椒噪声
        cv2.imwrite(f"blurred.jpg", blurred)
        
        # 2. 关键：只提取极暗的区域（黑胶带通常 < 60）
        #    比 Canny 更稳！因为你的圆是纯黑粗线
        _, dark = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite(f"dark.jpg", dark)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))    # 5×5 矩形
        # 常用三种形状：
        # cv2.MORPH_RECT   → 矩形（最常用）
        # cv2.MORPH_ELLIPSE → 圆形（最自然）
        # cv2.MORPH_CROSS   → 十字形

        # 2. 腐蚀
        eroded = cv2.erode(
            src=dark,
            kernel=kernel,
            anchor=(-1, -1),      # 默认中心
            iterations=4,         # 腐蚀次数，次数越多“瘦”得越厉害
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        cv2.imwrite(f"eroded.jpg", eroded)
        
        

        circles = cv2.HoughCircles(
            eroded,
            cv2.HOUGH_GRADIENT,
            dp=1.4,                    # 精度高一点
            minDist=150,               # 你的圆很大，最小圆心距设大点防重检
            param1=100,
            param2=50,                 # 关键！调低到 30~45，专门检测不完整圆
            minRadius=10,             # 根据你的图片，圆半径大概 150~300 像素
            maxRadius=900
        )
        




        self.detected_circles = []
        results = []

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
                h, w = gray.shape
                # 验证是否为黑边框圆（检查边缘像素）
                # if (self.is_center_region(x, y, w, h, grid_size=3) and 
                #     self.is_black_edge_circle(gray, x, y, r)):
                if self.is_black_edge_circle(gray, x, y, r):
                    self.detected_circles.append((x, y, r))
                    diameter = 2 * r
                    results.append({
                        "center_x": x,
                        "center_y": y,
                        "radius": r,
                        "diameter": diameter,
                        "image": {"width": int(w), "height": int(h)},
                    })
                    print(f"检测到圆: 圆心({x}, {y}), 半径={r}, 直径={diameter}")

        if results:
            self.status_label.config(text=f"检测到 {len(results)} 个圆", fg="green")
            return {"status": "success", "circles": results}
        else:
            self.status_label.config(text="未检测到黑边框圆", fg="orange")
            return {"status": "success", "circles": [], "message": "未检测到圆"}

    def clear_circles(self):
        self.detected_circles = []
        self.status_label.config(text="已清除所有圆", fg="blue")

    # ----------------------------------------------------------

    def update_ui(self):
        """Tkinter 显示画面（可保留或删除）"""
        if self.frame is not None:
            img = self.frame.copy()
            for (x, y, r) in self.detected_circles:
                cv2.circle(img, (x, y), r, (0, 255, 0), 2)
                cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=im)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

        self.root.after(20, self.update_ui)

    # ----------------------------------------------------------

    def get_mjpeg_frame(self):
        """返回 JPEG bytes"""
        if self.frame is None:
            return None
        img = self.frame.copy()
        for (x, y, r) in self.detected_circles:
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

        ret, jpeg = cv2.imencode('.jpg', img)
        if not ret:
            return None
        return jpeg.tobytes()

    # ----------------------------------------------------------

    def start_mjpeg_server(self, host="0.0.0.0", port=1992):
        server = HTTPServer((host, port), MJPEGHandler)
        server.player = self
        print(f"[MJPEG] 服务器已启动: http://{host}:{port}/video")
        server.serve_forever()


# =============================
#              Main
# =============================
if __name__ == "__main__":
    root = tk.Tk()
    player = RTSPPlayer(root, RTSP_URL)
    root.mainloop()
