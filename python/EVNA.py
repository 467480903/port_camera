import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import requests
import io
import threading
import time
import base64

# 摄像头地址
CAMERA_URL = "http://192.168.0.250/cgi-bin/jpg/image.cgi?id=2&rand="
# Basic Auth (Admin:1234 -> base64)
AUTH_HEADER = {
    "Authorization": "Basic " + base64.b64encode(b"Admin:1234").decode()
}


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("浙江智昌-摄像头实时控制-18989304974")

        # 摄像头画面
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        # 输入框区域
        input_frame = tk.Frame(root)
        input_frame.pack(pady=5)

        tk.Label(input_frame, text="调焦量:").pack(side="left")
        self.focus_step_var = tk.StringVar(value="100")
        self.focus_step_entry = ttk.Entry(input_frame, width=8, textvariable=self.focus_step_var)
        self.focus_step_entry.pack(side="left", padx=5)

        tk.Label(input_frame, text="设置聚焦量:").pack(side="left")
        self.focus_set_var = tk.StringVar(value="1375")
        self.focus_set_entry = ttk.Entry(input_frame, width=8, textvariable=self.focus_set_var)
        self.focus_set_entry.pack(side="left", padx=5)

        # 新增：放大比例输入框
        tk.Label(input_frame, text="放大比例:").pack(side="left")
        self.zoom_ratio_var = tk.StringVar(value="2")
        self.zoom_ratio_entry = ttk.Entry(input_frame, width=8, textvariable=self.zoom_ratio_var)
        self.zoom_ratio_entry.pack(side="left", padx=5)

        # 新增：连续调焦按钮
        self.continuous_adjusting = False
        self.adjust_button = ttk.Button(input_frame, text="连续调焦", command=self.toggle_continuous_adjust)
        self.adjust_button.pack(side="left", padx=5)

        # 按钮区域
        frame = tk.Frame(root)
        frame.pack(pady=10)

        # 聚焦命令按钮
        ttk.Button(frame, text="禁用 自动聚焦", command=lambda: self.send_command(
            "http://192.168.0.250/cgi-bin/com/ptz.cgi?autofocus=off", "禁用 自动聚焦")).pack(side="left", padx=5)
        ttk.Button(frame, text="增加聚焦", command=self.increase_focus).pack(side="left", padx=5)
        ttk.Button(frame, text="减小聚焦", command=self.decrease_focus).pack(side="left", padx=5)
        ttk.Button(frame, text="设置聚焦", command=self.set_focus).pack(side="left", padx=5)
        ttk.Button(frame, text="查询当前聚焦", command=lambda: self.send_command(
            "http://192.168.0.250/cgi-bin/com/ptz.cgi?query=focus", "查询当前聚焦")).pack(side="left", padx=5)

        # 新增：设置放大比例按钮
        ttk.Button(frame, text="设置放大比例", command=self.set_zoom_ratio).pack(side="left", padx=5)

        # 放大-调焦按钮
        self.continuous_focus = False
        self.focus_button = ttk.Button(frame, text="放大-调焦", command=self.toggle_continuous_focus)
        self.focus_button.pack(side="left", padx=5)

        # 日志区域
        self.log_text = tk.Text(root, height=20, width=100, state="disabled", wrap="word")
        self.log_text.pack(pady=10)

        # 启动刷新线程
        self.running = True
        threading.Thread(target=self.update_image, daemon=True).start()

    # ---------- 日志 ----------
    def log(self, message):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    # ---------- 摄像头刷新 ----------
    def update_image(self):
        while self.running:
            try:
                url = CAMERA_URL + str(time.time())
                r = requests.get(url, headers=AUTH_HEADER, timeout=2)
                if r.status_code == 200:
                    image_data = r.content
                    image = Image.open(io.BytesIO(image_data))
                    image = image.resize((640, 480))
                    photo = ImageTk.PhotoImage(image)
                    self.image_label.after(0, lambda: self.image_label.config(image=photo))
                    self.image_label.image = photo
            except Exception as e:
                self.log(f"获取图像失败: {e}")
            time.sleep(0.2)

    # ---------- 发送命令 ----------
    def send_command(self, url, name="命令"):
        try:
            r = requests.get(url, headers=AUTH_HEADER, timeout=2)
            self.log(f"{name} -> {url} | 返回: {r.text.strip()}")
        except Exception as e:
            self.log(f"{name} 请求失败: {e}")

    # ---------- 增减聚焦 ----------
    def increase_focus(self):
        try:
            step = int(self.focus_step_var.get())
        except ValueError:
            self.log("调焦量必须是整数")
            return
        url = f"http://192.168.0.250/cgi-bin/com/ptz.cgi?rfocus={step}"
        self.send_command(url, "增加聚焦")

    def decrease_focus(self):
        try:
            step = int(self.focus_step_var.get())
        except ValueError:
            self.log("调焦量必须是整数")
            return
        url = f"http://192.168.0.250/cgi-bin/com/ptz.cgi?rfocus={-step}"
        self.send_command(url, "减小聚焦")

    # ---------- 设置聚焦 ----------
    def set_focus(self):
        try:
            value = int(self.focus_set_var.get())
        except ValueError:
            self.log("设置聚焦量必须是整数")
            return
        url = f"http://192.168.0.250/cgi-bin/com/ptz.cgi?focus={value}"
        self.send_command(url, "设置聚焦")

    # ---------- 设置放大比例 ----------
    def set_zoom_ratio(self):
        try:
            ratio = float(self.zoom_ratio_var.get())
        except ValueError:
            self.log("放大比例必须是数字")
            return
        url = f"http://192.168.0.250/cgi-bin/com/ptz.cgi?zoomratio={ratio}"
        self.send_command(url, "设置放大比例")

    # ---------- 放大-调焦 ----------
    def toggle_continuous_focus(self):
        if not self.continuous_focus:
            self.continuous_focus = True
            self.focus_button.config(text="停止放大-调焦")
            threading.Thread(target=self.continuous_focus_loop, daemon=True).start()
            self.log("开始放大-调焦")
        else:
            self.continuous_focus = False
            self.focus_button.config(text="放大-调焦")
            self.log("停止放大-调焦")

    def continuous_focus_loop(self):
        steps = 32
        focus_arr = [1471,1902,2156,2325,2440,
                     2514,2558,2568,2550,2502,
                     2430,2332,2220,2098,1965,
                     1834,1701,1572,1452,1332,
                     1230,1117,1033,942,839,
                     754,693,626,554,475,
                     432,386]
        start_zoom = 3
        end_zoom = 32

        for i in range(3,32):
            if not self.continuous_focus:
                break
            # 线性插值
            focus = focus_arr[i]
            zoom = i+1
            # 先调整放大比例
            #调整前先开启自动聚焦
            url_auto_focus = "http://192.168.0.250/cgi-bin/com/ptz.cgi?continuouspantiltmove=0,0"
            self.send_command(url_auto_focus, "开启 自动聚焦")
            time.sleep(0.2)
            url_zoom = f"http://192.168.0.250/cgi-bin/com/ptz.cgi?zoomratio={zoom}"
            self.send_command(url_zoom, f"放大-调焦-设置放大比例 {zoom:.2f}")
            self.zoom_ratio_var.set(str(zoom))  # 同步更新“设置聚焦量”输入框
            time.sleep(0.8)
            # 再调整聚焦
            url_focus = f"http://192.168.0.250/cgi-bin/com/ptz.cgi?focus={focus}"
            self.send_command(url_focus, f"放大-调焦-设置聚焦 {focus}")
            self.focus_set_var.set(str(focus))  # 同步更新“设置聚焦量”输入框
           
            time.sleep(0.7)
        self.continuous_focus = False
        self.focus_button.config(text="放大-调焦")
        self.log("放大-调焦完成")


    # ---------- 连续调焦 ----------
    def toggle_continuous_adjust(self):
        if not self.continuous_adjusting:
            self.continuous_adjusting = True
            self.adjust_button.config(text="停止连续调焦")
            threading.Thread(target=self.continuous_adjust_loop, daemon=True).start()
            self.log("开始连续调焦")
        else:
            self.continuous_adjusting = False
            self.adjust_button.config(text="连续调焦")
            self.log("停止连续调焦")

    def continuous_adjust_loop(self):
        start_focus = 2600
        end_focus = 386
        interval = 0.05  # 秒
        total_time = 4  # 秒
        steps = int(total_time / interval)
        for i in range(steps + 1):
            if not self.continuous_adjusting:
                break
            # 线性插值
            focus = int(start_focus + (end_focus - start_focus) * i / steps)
            url_focus = f"http://192.168.0.250/cgi-bin/com/ptz.cgi?focus={focus}"
            self.send_command(url_focus, f"连续调焦-设置聚焦 {focus}")
            self.focus_set_var.set(str(focus))
            time.sleep(interval)
        self.continuous_adjusting = False
        self.adjust_button.config(text="连续调焦")
        self.log("连续调焦完成")        

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()