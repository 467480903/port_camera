import cv2
import threading
import time
from queue import Queue

def video_reader_thread(video_path, frame_queue, delay=0):
    """
    负责读取视频帧的线程函数
    """
    # 模拟启动前的延迟
    if delay > 0:
        time.sleep(delay)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: 无法打开视频 {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            # 视频结束，存入一个 None 作为结束信号
            frame_queue.put(None)
            break
        
        # 将读取到的帧存入队列
        frame_queue.put(frame)
        
        # 根据视频 FPS 控制读取速度，避免队列无限堆积
        fps = cap.get(cv2.CAP_PROP_FPS)
        wait = 1.0 / fps if fps > 0 else 0.03
        time.sleep(wait)

    cap.release()

def main():
    path1 = "/home/yy/Q310_62.ts"
    path2 = "/home/yy/Q310_63.ts"

    # 创建两个队列用于线程间通信
    q1 = Queue(maxsize=10)
    q2 = Queue(maxsize=10)

    # 创建并启动两个线程
    # 视频 1 延迟 4 秒，视频 2 立即开始
    t1 = threading.Thread(target=video_reader_thread, args=(path1, q1, 7))
    t2 = threading.Thread(target=video_reader_thread, args=(path2, q2, 0))
    
    t1.daemon = True # 设置为守护线程，主程序退出时自动结束
    t2.daemon = True
    t1.start()
    t2.start()

    print("正在播放... 按 'q' 键退出")

    while True:
        # 从队列中尝试获取帧（不阻塞，防止互相等待）
        frame1 = None
        frame2 = None

        if not q1.empty():
            frame1 = q1.get()
        if not q2.empty():
            frame2 = q2.get()

        # 显示画面
        if frame1 is not None:
            cv2.imshow('Video 1 (Thread - Delayed)', frame1)
        if frame2 is not None:
            cv2.imshow('Video 2 (Thread - Normal)', frame2)

        # 这里的 waitKey 必须在主线程，且是控制 UI 刷新的关键
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()