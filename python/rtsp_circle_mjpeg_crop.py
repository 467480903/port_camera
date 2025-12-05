import cv2
import threading
import tkinter as tk
from tkinter import ttk
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

        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Tab 1: Original view
        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text='Circle Detection')
        
        # Tab 2: Crop view
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text='Video Crop')

        # ===== Tab 1 Content (Original) =====
        self.label = tk.Label(self.tab1)
        self.label.pack()

        # 按钮区域
        button_frame = tk.Frame(self.tab1)
        button_frame.pack(pady=10)

        self.detect_btn = tk.Button(button_frame, text="识别圆", command=self.detect_circle,
                                    bg="#4CAF50", fg="white", padx=20, pady=5)
        self.detect_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = tk.Button(button_frame, text="删除圆", command=self.clear_circles,
                                   bg="#f44336", fg="white", padx=20, pady=5)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(self.tab1, text="就绪", fg="blue")
        self.status_label.pack()

        # ===== Tab 2 Content (Crop) =====
        self.label_crop = tk.Label(self.tab2)
        self.label_crop.pack()

        # Crop controls frame
        crop_controls = tk.Frame(self.tab2)
        crop_controls.pack(pady=10)

        # Row 1: Position X and Y
        row1 = tk.Frame(crop_controls)
        row1.pack(pady=5)
        
        tk.Label(row1, text="Crop Position X:").pack(side=tk.LEFT, padx=5)
        self.crop_x_entry = tk.Entry(row1, width=10)
        self.crop_x_entry.insert(0, "0")
        self.crop_x_entry.pack(side=tk.LEFT, padx=5)
        
        self.crop_x_inc_btn = tk.Button(row1, text="+", command=lambda: self.increment(self.crop_x_entry, 5))
        self.crop_x_inc_btn.pack(side=tk.LEFT, padx=5)
        
        self.crop_x_dec_btn = tk.Button(row1, text="-", command=lambda: self.decrement(self.crop_x_entry, 5))
        self.crop_x_dec_btn.pack(side=tk.LEFT, padx=5)
        
        tk.Label(row1, text="Crop Position Y:").pack(side=tk.LEFT, padx=5)
        self.crop_y_entry = tk.Entry(row1, width=10)
        self.crop_y_entry.insert(0, "0")
        self.crop_y_entry.pack(side=tk.LEFT, padx=5)
        
        self.crop_y_inc_btn = tk.Button(row1, text="+", command=lambda: self.increment(self.crop_y_entry, 5))
        self.crop_y_inc_btn.pack(side=tk.LEFT, padx=5)
        
        self.crop_y_dec_btn = tk.Button(row1, text="-", command=lambda: self.decrement(self.crop_y_entry, 5))
        self.crop_y_dec_btn.pack(side=tk.LEFT, padx=5)

        # Row 2: Width and Height
        row2 = tk.Frame(crop_controls)
        row2.pack(pady=5)
        
        tk.Label(row2, text="Crop Width:").pack(side=tk.LEFT, padx=5)
        self.crop_width_entry = tk.Entry(row2, width=10)
        self.crop_width_entry.insert(0, "640")
        self.crop_width_entry.pack(side=tk.LEFT, padx=5)
        
        self.crop_width_inc_btn = tk.Button(row2, text="+", command=lambda: self.increment(self.crop_width_entry, 5))
        self.crop_width_inc_btn.pack(side=tk.LEFT, padx=5)
        
        self.crop_width_dec_btn = tk.Button(row2, text="-", command=lambda: self.decrement(self.crop_width_entry, 5))
        self.crop_width_dec_btn.pack(side=tk.LEFT, padx=5)
        
        tk.Label(row2, text="Crop Height:").pack(side=tk.LEFT, padx=5)
        self.crop_height_entry = tk.Entry(row2, width=10)
        self.crop_height_entry.insert(0, "480")
        self.crop_height_entry.pack(side=tk.LEFT, padx=5)
        
        self.crop_height_inc_btn = tk.Button(row2, text="+", command=lambda: self.increment(self.crop_height_entry, 5))
        self.crop_height_inc_btn.pack(side=tk.LEFT, padx=5)
        
        self.crop_height_dec_btn = tk.Button(row2, text="-", command=lambda: self.decrement(self.crop_height_entry, 5))
        self.crop_height_dec_btn.pack(side=tk.LEFT, padx=5)

        # Row 3: Buttons
        row3 = tk.Frame(crop_controls)
        row3.pack(pady=10)
        
        self.set_crop_btn = tk.Button(row3, text="Set Crop", command=self.set_crop,
                                      bg="#2196F3", fg="white", padx=20, pady=5)
        self.set_crop_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_crop_btn = tk.Button(row3, text="Reset", command=self.reset_crop,
                                        bg="#FF9800", fg="white", padx=20, pady=5)
        self.reset_crop_btn.pack(side=tk.LEFT, padx=5)

        self.crop_status_label = tk.Label(self.tab2, text="No crop applied", fg="blue")
        self.crop_status_label.pack()

        row4 = tk.Frame(crop_controls)
        row4.pack(pady=5)
        
        tk.Label(row4, text="Increment Step:").pack(side=tk.LEFT, padx=5)
        self.increment_step_entry = tk.Entry(row4, width=10)
        self.increment_step_entry.insert(0, "5")
        self.increment_step_entry.pack(side=tk.LEFT, padx=5)

        # Crop parameters
        self.crop_enabled = False
        self.crop_x = 0
        self.crop_y = 0
        self.crop_w = 640
        self.crop_h = 480

        self.frame = None
        self.cap = cv2.VideoCapture(rtsp_url)
        self.caculated_circles = []
        self.detected_circles = []

        # 多线程
        self.running = True
        threading.Thread(target=self.fetch_frames, daemon=True).start()
        threading.Thread(target=self.tcp_server, daemon=True).start()
        threading.Thread(target=self.start_mjpeg_server, daemon=True).start()

        self.update_ui()


    def increment(self, entry, step):
        try:
            current_value = int(entry.get())
            entry.delete(0, tk.END)
            entry.insert(0, str(current_value + step))
        except ValueError:
            entry.delete(0, tk.END)
            entry.insert(0, "0")

        self.set_crop()

    def decrement(self, entry, step):
        try:
            current_value = int(entry.get())
            entry.delete(0, tk.END)
            entry.insert(0, str(current_value - step))
        except ValueError:
            entry.delete(0, tk.END)
            entry.insert(0, "0")

        self.set_crop()
    # ----------------------------------------------------------

    def set_crop(self):
        """Set crop parameters from user input"""
        try:
            x = int(self.crop_x_entry.get())
            y = int(self.crop_y_entry.get())
            w = int(self.crop_width_entry.get())
            h = int(self.crop_height_entry.get())
            
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                self.crop_status_label.config(text="Invalid values! Must be positive.", fg="red")
                return
            
            self.crop_x = x
            self.crop_y = y
            self.crop_w = w
            self.crop_h = h
            self.crop_enabled = True
            
            self.crop_status_label.config(
                text=f"Crop applied: ({x}, {y}) - {w}x{h}", 
                fg="green"
            )
        except ValueError:
            self.crop_status_label.config(text="Invalid input! Please enter numbers.", fg="red")
    
    def reset_crop(self):
        """Reset crop to show original video"""
        self.crop_enabled = False
        self.crop_x_entry.delete(0, tk.END)
        self.crop_x_entry.insert(0, "0")
        self.crop_y_entry.delete(0, tk.END)
        self.crop_y_entry.insert(0, "0")
        self.crop_width_entry.delete(0, tk.END)
        self.crop_width_entry.insert(0, "640")
        self.crop_height_entry.delete(0, tk.END)
        self.crop_height_entry.insert(0, "480")
        self.crop_status_label.config(text="Crop reset - showing original video", fg="blue")
    
    def get_cropped_frame(self, frame):
        """Get cropped frame if crop is enabled and resize to original size"""
        if not self.crop_enabled or frame is None:
            return frame
        
        h, w = frame.shape[:2]
        
        # Ensure crop area is within frame bounds
        x1 = max(0, min(self.crop_x, w))
        y1 = max(0, min(self.crop_y, h))
        x2 = max(0, min(self.crop_x + self.crop_w, w))
        y2 = max(0, min(self.crop_y + self.crop_h, h))
        
        if x2 <= x1 or y2 <= y1:
            return frame
        
        # Crop the frame
        cropped = frame[y1:y2, x1:x2]
        
        # Resize the cropped frame back to original video dimensions
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return resized

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
    
    def is_circle_overlap_eroded(self, eroded_img, x, y, r, min_ratio=0.5):
        """
        检查圆是否与 eroded 图重合
        使用 3x3 区域判断命中，只要该区域有任意白点，就认为该点通过
        """
        angles = np.linspace(0, 2 * np.pi, 32)  # 圆周采样点
        hit = 0
        total = 0

        h, w = eroded_img.shape

        for angle in angles:
            px = int(x + r * np.cos(angle))
            py = int(y + r * np.sin(angle))

            # 该采样点是否在图像内
            if 0 <= px < w and 0 <= py < h:
                total += 1

                # ---- 新逻辑：检查 3×3 邻域 ----
                found = False
                for dx in [-2,-1, 0, 1,2]:
                    for dy in [-2,-1, 0, 1,2]:
                        nx = px + dx
                        ny = py + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            if eroded_img[ny, nx] > 127:
                                found = True
                                break
                    if found:
                        break

                if found:
                    hit += 1

        if total == 0:
            return False

        ratio = hit / total
        # print("eroded overlap ratio =", ratio)

        return ratio >= min_ratio

    def detect_circle(self):
        """基于轮廓检测黑边空心圆"""
        if self.frame is None:
            self.status_label.config(text="无视频帧", fg="red")
            return {"status": "error", "message": "无视频帧"}

        frame_copy = self.frame.copy()
        img = frame_copy

        # 1. 转灰度 + 轻微模糊
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 1)
        cv2.imwrite(f"blurred.jpg", blurred)

        scaleabs = cv2.convertScaleAbs(blurred, alpha=1.5, beta=0)
        cv2.imwrite(f"scaleabs.jpg", scaleabs)


        # 2. 提取极暗区域（黑胶带）
        _, dark = cv2.threshold(scaleabs, 60, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite(f"dark.jpg", dark)

        # 3. 先膨胀后腐蚀
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

        # 先膨胀：连接相邻区域，填充小孔
        dilated = cv2.dilate(dark, kernel, iterations=4)
        cv2.imwrite(f"dilated.jpg", dilated)

        # 后腐蚀：恢复大致形状，去除毛刺
        eroded = cv2.erode(dilated, kernel, iterations=4)
        cv2.imwrite(f"eroded.jpg", eroded)

        # 4. 找轮廓
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(contours)

        self.detected_circles = []
        self.caculated_circles = []
        results = []

        for cnt in contours:
            # 跳过太小的噪声
            if cv2.contourArea(cnt) < 500:  # 根据你的圆大小调整
                continue

            # 计算轮廓面积和周长
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            
            # 圆形度判断
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.8:  # 接近1表示更接近圆形

                    # 计算最小外接圆
                    (x, y), r = cv2.minEnclosingCircle(cnt)
                    x, y, r = int(x), int(y), int(r)

                    self.caculated_circles.append((x, y, r))

                    h, w = gray.shape

                    # # --- 黑边判断 ---
                    # if not self.is_black_edge_circle(gray, x, y, r):
                    #     continue

                    # # --- 与 eroded 图重合度判断 ---
                    # if not self.is_circle_overlap_eroded(eroded, x, y, r, min_ratio=0.8):
                    #     continue

                    # 满足条件 -> 记录结果
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
        self.caculated_circles = []
        self.status_label.config(text="已清除所有圆", fg="blue")

    # ----------------------------------------------------------

    def update_ui(self):
        """Tkinter 显示画面（可保留或删除）"""
        if self.frame is not None:
            # Tab 1: Original view with circle detection
            img = self.frame.copy()

            for (x, y, r) in self.caculated_circles:
                cv2.circle(img, (x, y), r, (255, 0, 0), 1)     

            for (x, y, r) in self.detected_circles:
                cv2.circle(img, (x, y), r, (0, 255, 0), 2)
                cv2.circle(img, (x, y), 3, (0, 0, 255), -1)        

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(img_rgb)
            imgtk = ImageTk.PhotoImage(image=im)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

            # Tab 2: Cropped view
            crop_img = self.get_cropped_frame(self.frame.copy())
            if crop_img is not None:
                crop_img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                im_crop = Image.fromarray(crop_img_rgb)
                imgtk_crop = ImageTk.PhotoImage(image=im_crop)
                self.label_crop.imgtk_crop = imgtk_crop
                self.label_crop.configure(image=imgtk_crop)

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
