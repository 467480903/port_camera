import cv2
import threading
import tkinter as tk
from PIL import Image, ImageTk
import socket
import json
import numpy as np
from datetime import datetime

RTSP_URL = "rtsp://user1:h7Hsu3ULLnLTs*M@192.168.1.13:554/media/video2"
TCP_HOST = "0.0.0.0"
TCP_PORT = 1991


def is_good_circle(img_gray, center, radius, debug=False):
    """
    判断一个候选圆是否“真的连续黑圆”
    返回 True / False
    """
    x, y = center
    r = radius
    
    # 1. 创建掩码：只看这个圆环附近（内外环）
    mask = np.zeros(img_gray.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, thickness=15)   # 粗一点，覆盖整个胶带宽度
    
    # 2. 提取这个圆环区域的像素
    ring_pixels = img_gray[mask == 255]
    
    if len(ring_pixels) == 0:
        return False
    
    # 3. 计算该区域平均亮度（黑胶带应该很黑）
    mean_val = np.mean(ring_pixels)
    if mean_val > 80:  # 太亮了，肯定不是黑胶带
        if debug: print(f"× 太亮了 mean={mean_val:.1f}")
        return False
    
    # 4. 计算圆形度（完美圆=1）
    #    方法：实际黑像素面积 / 理论圆环面积
    actual_black = np.sum(mask == 255) - np.sum((mask == 255) & (img_gray > 80))
    theoretical_area = np.pi * r * 15 * 2  # 近似
    circularity = actual_black / theoretical_area
    
    if circularity < 0.4:  # 缺口太大，连续性差
        if debug: print(f"× 圆形度低 circularity={circularity:.3f}")
        return False
    
    # 5. 额外检查：轮廓连续性（最强保险）
    #    在圆环上提取边缘点，做最小外接圆拟合误差
    y_mask, x_mask = np.where(mask == 255)
    if len(x_mask) < 100:
        return False
        
    points = np.column_stack((x_mask, y_mask))
    
    # 用 OpenCV 自带的圆拟合
    (cx, cy), fitted_r = cv2.minEnclosingCircle(points)
    fit_error = abs(fitted_r - r) / r + abs(cx - x)/r + abs(cy - y)/r
    
    if fit_error > 0.15:  # 拟合误差太大，说明轮廓不连续
        if debug: print(f"× 拟合误差大 error={fit_error:.3f}")
        return False
    
    if debug:
        print(f"√ 好圆！亮度={mean_val:.1f}, 圆形度={circularity:.3f}, 误差={fit_error:.3f}")
    
    return True




class RTSPPlayer:
    def __init__(self, root, rtsp_url):
        self.root = root
        self.root.title("RTSP Video Player - Circle Detection")

        # 视频显示区域
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

        # 状态标签
        self.status_label = tk.Label(root, text="就绪", fg="blue")
        self.status_label.pack()

        self.frame = None
        self.cap = cv2.VideoCapture(rtsp_url)
        self.detected_circles = []  # 存储检测到的圆 [(x, y, radius), ...]

        # 启动后台线程
        self.running = True
        threading.Thread(target=self.fetch_frames, daemon=True).start()
        threading.Thread(target=self.tcp_server, daemon=True).start()

        # UI更新
        self.update_ui()

    def fetch_frames(self):
        """后台线程：循环读取 RTSP 最新帧"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def tcp_server(self):
        """TCP服务器：监听命令"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((TCP_HOST, TCP_PORT))
        server.listen(5)
        print(f"TCP服务器启动: {TCP_HOST}:{TCP_PORT}")

        while self.running:
            try:
                client, addr = server.accept()
                threading.Thread(target=self.handle_client, args=(client, addr), daemon=True).start()
            except:
                break

    def handle_client(self, conn, addr):
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
                    if cmd_name == "detect":
                        result = self.detect_circle()

                        response_str = json.dumps(result, ensure_ascii=False) + "\n"
                        conn.sendall(response_str.encode('utf-8'))
                        print(f"[响应] {addr} -> {response_str.strip()[:100]}...")



        except Exception as e:
            print(f"[错误] {addr} -> {e}")
        finally:
            conn.close()
            print(f"[断开] {addr}")            

    def handle_client_(self, client, addr):
        """处理客户端连接"""
        print(f"客户端连接: {addr}")
        try:
            data = client.recv(1024).decode('utf-8')
            command = json.loads(data)
            
            if command.get("command") == "detect":
                result = self.detect_circle()
                response = json.dumps(result, ensure_ascii=False)
                client.send(response.encode('utf-8'))
                print(f"发送响应: {response}")
        except Exception as e:
            print(f"处理客户端错误: {e}")
        finally:
            client.close()

    def is_center_region(self, x, y, img_width, img_height, grid_size=5):
            """
            判断圆心是否落在 5x5 网格的正中心方块内
            grid_size=5 → 25个方块，中间那个是第3行第3列（从0开始算）
            """
            # 计算每个格子的大小
            cell_w = img_width // grid_size
            cell_h = img_height // grid_size
            
            # 中心格子的索引（5x5网格中，中间是第2个，索引从0开始）
            center_idx = grid_size // 2  # = 2
            
            # 中心格子的坐标范围
            min_x = center_idx * cell_w
            max_x = (center_idx + 1) * cell_w
            min_y = center_idx * cell_h
            max_y = (center_idx + 1) * cell_h
            
            # 判断圆心是否落在中心格子内（允许边缘一点误差）
            return min_x <= x <= max_x and min_y <= y <= max_y            

    def detect_circle(self):
        """检测黑边框圆"""
        if self.frame is None:
            self.status_label.config(text="无视频帧", fg="red")
            return {"status": "error", "message": "无视频帧"}

        # 复制当前帧进行处理
        frame_copy = self.frame.copy()
        img = frame_copy


        # gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
        
        # # 高斯模糊减少噪声
        # blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # # 霍夫圆检测
        # circles = cv2.HoughCircles(
        #     blurred,
        #     cv2.HOUGH_GRADIENT,
        #     dp=1,
        #     minDist=50,
        #     param1=100,
        #     param2=30,
        #     minRadius=10,
        #     maxRadius=200
        # )


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
        
    def draw_grid(self, img, grid_size=5, color=(100,100,100), thickness=1):
            h, w = img.shape[:2]
            cw, ch = w // grid_size, h // grid_size
            for i in range(1, grid_size):
                cv2.line(img, (i*cw, 0), (i*cw, h), color, thickness)
                cv2.line(img, (0, i*ch), (w, i*ch), color, thickness)
            # 高亮中心格子
            c = grid_size // 2
            cv2.rectangle(img, (c*cw, c*ch), ((c+1)*cw-1, (c+1)*ch-1), (0,255,255), 3)        

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

    def clear_circles(self):
        """清除检测到的圆"""
        self.detected_circles = []
        self.status_label.config(text="已清除所有圆", fg="blue")
        print("已清除所有圆")

    def update_ui(self):
        """显示最新帧并绘制检测到的圆"""
        if self.frame is not None:
            display_frame = self.frame.copy()
            
            # 在帧上绘制检测到的圆
            for (x, y, r) in self.detected_circles:
                # 绘制圆周（绿色）
                cv2.circle(display_frame, (x, y), r, (0, 255, 0), 2)
                # 绘制圆心（红色）
                cv2.circle(display_frame, (x, y), 3, (0, 0, 255), -1)
                # 标注坐标和直径
                text = f"({x},{y}) D={2*r}"
                cv2.putText(display_frame, text, (x - 50, y - r - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=im)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

        self.root.after(15, self.update_ui)

    def __del__(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    player = RTSPPlayer(root, RTSP_URL)
    root.mainloop()