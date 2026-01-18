import cv2
import torch
from ultralytics import YOLO
import av
import time

def main():
    # 检查CUDA是否可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载YOLO模型
    model = YOLO("/home/yy/port_camera/yoloTrain3/runs/detect/train/weights/best.pt", task="detect").to(device)
    print("YOLO model loaded")
    
    # RTSP视频源
    rtsp_url = "rtsp://user1:h7Hsu3ULLnLTs*M@10.10.95.216:554/media/video1"
    
    # 连接RTSP流并使用GPU解码
    print(f"Connecting to RTSP stream: {rtsp_url}")
    try:
        # 尝试使用GPU硬件加速
        options = {
            'rtsp_transport': 'tcp',
            'max_delay': '500000',
            'hwaccel': 'cuda',
            'hwaccel_device': '0',
            'hwaccel_output_format': 'cuda'
        }
        
        container = av.open(rtsp_url, options=options)
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        print(f"✓ RTSP connected: {stream.width}x{stream.height} @ {fps}fps [GPU]")
        
    except Exception as e:
        print(f"GPU decode failed: {e}")
        print("Trying CPU decode...")
        
        # CPU降级
        options = {'rtsp_transport': 'tcp', 'max_delay': '500000'}
        container = av.open(rtsp_url, options=options)
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        print(f"✓ RTSP connected: {stream.width}x{stream.height} @ {fps}fps [CPU]")
    
    # 处理视频流
    frame_count = 0
    start_time = time.time()
    
    print("\nStarting detection... Press Ctrl+C to stop\n")
    
    try:
        for frame_obj in container.decode(stream):
            # 转换为numpy数组
            frame = frame_obj.to_ndarray(format='bgr24')
            frame_count += 1
            
            frame = cv2.rotate(frame, cv2.ROTATE_180)

            # YOLO检测 - 只检测class 41（被子）
            results = model(frame, conf=0.15, verbose=False, classes=[41])
            
            # 绘制检测结果
            detection_count = 0
            if len(results) > 0:
                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    for box in boxes:
                        detection_count += 1
                        x1, y1, x2, y2 = box.xyxy[0].astype(int)
                        conf = box.conf[0]
                        
                        # 绘制边界框
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # 显示置信度
                        label = f"Class 41: {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 计算FPS
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            
            # 显示信息
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Detections: {detection_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示画面
            cv2.imshow('YOLO Detection - Class 41', frame)
            
            # 每30帧打印一次信息
            if frame_count % 30 == 0:
                print(f"Frame {frame_count} | FPS: {current_fps:.1f} | Detections: {detection_count}")
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nQuitting...")
                break
                
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        # 清理
        container.close()
        cv2.destroyAllWindows()
        
        # 打印统计
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\nTotal frames: {frame_count}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average FPS: {avg_fps:.1f}")

if __name__ == "__main__":
    main()