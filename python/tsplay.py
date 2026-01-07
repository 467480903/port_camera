import cv2

def play_two_videos(path1, path2):
    # 1. 初始化两个视频捕获对象
    cap1 = cv2.VideoCapture(path1)
    cap2 = cv2.VideoCapture(path2)

    # 检查是否成功打开
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: 无法打开其中一个或两个视频文件。")
        return

    # 获取视频帧率（以第一路视频为准来参考播放速度）
    fps = cap1.get(cv2.CAP_PROP_FPS)
    wait_time = int(1000 / fps) if fps > 0 else 30

    print("正在播放两个视频...")
    print("提示: 点击任意图像窗口并按 'q' 键退出。")

    while True:
        # 2. 同时读取两个视频的帧
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # 3. 检查视频是否结束
        # 如果两个视频都结束了，则退出
        if not ret1 and not ret2:
            break
        
        # 4. 显示视频
        # 如果视频 1 还在播放，显示画面
        if ret1:
            cv2.imshow('Video 1 - TS Player', frame1)
        
        # 如果视频 2 还在播放，显示画面
        if ret2:
            cv2.imshow('Video 2 - TS Player', frame2)

        # 5. 按键监听：按下 'q' 键退出
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    # 6. 释放所有资源
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 替换为你实际的 .ts 文件路径
    video_path_1 = '/home/yy/Q310_63.ts'
    video_path_2 = '/home/yy/Q310_62.ts'
    
    play_two_videos(video_path_1, video_path_2)