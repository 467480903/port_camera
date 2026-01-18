import cv2
import torch
from ultralytics import YOLO
import numpy as np
import threading
import time
import logging
import av
import subprocess
import os

def setup_logging():
    logging.getLogger('ultralytics').setLevel(logging.ERROR)
    logging.getLogger('opencv').setLevel(logging.ERROR)
    
    # Suppress OpenCV warnings
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
    os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

setup_logging()

# Global frames and data
frame_left = None
frame_right = None
frame_lock_left = threading.Lock()
frame_lock_right = threading.Lock()

# Global center points
centers_left = []
centers_right = []
centers_lock_left = threading.Lock()
centers_lock_right = threading.Lock()

# Combined frame
combined_frame = None
combined_frame_lock = threading.Lock()

# Control flags
running = True
display_mode = 1  # 0 = original, 1 = combined

# FFmpeg processes - direct streaming without queues
ffmpeg_processes = {}
ffmpeg_locks = {
    'left': threading.Lock(),
    'right': threading.Lock(),
    'combined': threading.Lock()
}

# Fixed dimensions for combined stream
FIXED_COMBINED_WIDTH = 2560
FIXED_COMBINED_HEIGHT = 720

modelpath = "../yoloTrain3/runs/detect/train/weights/best.pt"
modelpath = "/home/yy/port_camera/python/yolo11n.pt"

def process_video_left():
    global frame_left, centers_left, running

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(modelpath, task="detect").to(device)

    rtsp_url = "rtsp://localhost:8554/cam5"
    
    retry_count = 0
    max_retries = 5
    
    while running and retry_count < max_retries:
        try:
            # RTSP stream options for GPU decoding
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
            
            print(f"✓ Left RTSP connected: {stream.width}x{stream.height} @ {fps}fps [GPU]")
            retry_count = 0
            break
            
        except Exception as e:
            retry_count += 1
            print(f"✗ Left RTSP connection error (attempt {retry_count}/{max_retries}): {e}")
            if retry_count < max_retries:
                print(f"  Retrying in 3 seconds...")
                time.sleep(3)
            else:
                print(f"  Trying CPU fallback...")
                try:
                    options = {'rtsp_transport': 'tcp', 'max_delay': '500000'}
                    container = av.open(rtsp_url, options=options)
                    stream = container.streams.video[0]
                    fps = float(stream.average_rate) if stream.average_rate else 30.0
                    print(f"✓ Left RTSP connected (CPU): {stream.width}x{stream.height} @ {fps}fps")
                except Exception as e2:
                    print(f"✗ Left RTSP CPU fallback failed: {e2}")
                    return
    
    if not running:
        return
    
    frame_count = 0
    reconnect_attempts = 0
    
    while running:
        try:
            frame_obj = next(container.decode(stream))
            frame = frame_obj.to_ndarray(format='bgr24')
            frame_count += 1
            reconnect_attempts = 0
            
        except (StopIteration, av.error.EOFError) as e:
            print(f"Left RTSP stream ended or disconnected, attempting reconnection...")
            reconnect_attempts += 1
            
            if reconnect_attempts > 10:
                print(f"Left RTSP: Too many reconnection attempts, exiting")
                break
            
            try:
                container.close()
            except:
                pass
            
            time.sleep(2)
            
            try:
                options = {
                    'rtsp_transport': 'tcp',
                    'max_delay': '500000',
                    'hwaccel': 'cuda',
                    'hwaccel_device': '0',
                    'hwaccel_output_format': 'cuda'
                }
                container = av.open(rtsp_url, options=options)
                stream = container.streams.video[0]
                print(f"✓ Left RTSP reconnected")
            except Exception as reconnect_error:
                print(f"✗ Left RTSP reconnection failed: {reconnect_error}")
                time.sleep(3)
            continue
            
        except Exception as e:
            print(f"Left decode error: {e}")
            time.sleep(0.1)
            continue
        
        # 旋转并检测
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        results = model(frame, conf=0.25, verbose=False, classes=[41])
        
        current_centers = []
        detection_count = 0
        if len(results) > 0:
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    detection_count += 1
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    conf = box.conf[0]
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    current_centers.append((center_x, center_y))
                    
                    # 绘制检测框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 绘制中心点
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    
                    # 显示置信度
                    label = f"Class 41: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 添加帧信息
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Detections: {detection_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 显示窗口
        cv2.imshow('Left Camera - YOLO Detection', frame)
        
        with centers_lock_left:
            centers_left = current_centers.copy()
        
        with frame_lock_left:
            frame_left = frame.copy()
        
    
    cv2.destroyWindow('Left Camera - YOLO Detection')
    
    try:
        container.close()
    except:
        pass


def process_video_right():
    global frame_right, centers_right, running

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(modelpath, task="detect").to(device)
    
    rtsp_url = "rtsp://localhost:8554/cam4"
    
    retry_count = 0
    max_retries = 5
    
    while running and retry_count < max_retries:
        try:
            # RTSP stream options for GPU decoding
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
            delay = 1.0 / fps
            
            print(f"✓ Right RTSP connected: {stream.width}x{stream.height} @ {fps}fps [GPU]")
            retry_count = 0
            break
            
        except Exception as e:
            retry_count += 1
            print(f"✗ Right RTSP connection error (attempt {retry_count}/{max_retries}): {e}")
            if retry_count < max_retries:
                print(f"  Retrying in 3 seconds...")
                time.sleep(3)
            else:
                print(f"  Trying CPU fallback...")
                try:
                    options = {'rtsp_transport': 'tcp', 'max_delay': '500000'}
                    container = av.open(rtsp_url, options=options)
                    stream = container.streams.video[0]
                    fps = float(stream.average_rate) if stream.average_rate else 30.0
                    delay = 1.0 / fps
                    print(f"✓ Right RTSP connected (CPU): {stream.width}x{stream.height} @ {fps}fps")
                except Exception as e2:
                    print(f"✗ Right RTSP CPU fallback failed: {e2}")
                    return
    
    if not running:
        return
    
    frame_count = 0
    reconnect_attempts = 0
    
    while running:
        start_time = time.time()
        
        try:
            frame_obj = next(container.decode(stream))
            frame = frame_obj.to_ndarray(format='bgr24')
            frame_count += 1
            reconnect_attempts = 0
            
        except (StopIteration, av.error.EOFError) as e:
            print(f"Right RTSP stream ended or disconnected, attempting reconnection...")
            reconnect_attempts += 1
            
            if reconnect_attempts > 10:
                print(f"Right RTSP: Too many reconnection attempts, exiting")
                break
            
            try:
                container.close()
            except:
                pass
            
            time.sleep(2)
            
            try:
                options = {
                    'rtsp_transport': 'tcp',
                    'max_delay': '500000',
                    'hwaccel': 'cuda',
                    'hwaccel_device': '0',
                    'hwaccel_output_format': 'cuda'
                }
                container = av.open(rtsp_url, options=options)
                stream = container.streams.video[0]
                print(f"✓ Right RTSP reconnected")
            except Exception as reconnect_error:
                print(f"✗ Right RTSP reconnection failed: {reconnect_error}")
                time.sleep(3)
            continue
            
        except Exception as e:
            print(f"Right decode error: {e}")
            time.sleep(0.1)
            continue
        
        height, width = frame.shape[:2]
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        results = model(frame, conf=0.15, verbose=False, classes=[41])
        
        current_centers = []
        if len(results) > 0:
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    current_centers.append((center_x, center_y))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        with centers_lock_right:
            centers_right = current_centers.copy()
        
        with frame_lock_right:
            frame_right = frame.copy()
        
        # Direct streaming to FFmpeg (no queue)
        if 'right' in ffmpeg_processes and ffmpeg_processes['right'] is not None:
            with ffmpeg_locks['right']:
                try:
                    ffmpeg_processes['right'].stdin.write(frame.tobytes())
                except (BrokenPipeError, IOError):
                    pass
        
        elapsed = time.time() - start_time
        sleep_time = max(0, delay - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    try:
        container.close()
    except:
        pass


def calculate_y_alignment_offset(centers_left, centers_right):
    if not centers_left or not centers_right:
        return 0
    
    avg_left_y = np.mean([center[1] for center in centers_left])
    avg_right_y = np.mean([center[1] for center in centers_right])
    y_diff = int(avg_right_y - avg_left_y)
    
    return y_diff


def get_average_x_position(centers):
    if not centers:
        return None
    return int(np.mean([center[0] for center in centers]))


def process_combined_frame():
    """Generate combined frame"""
    global combined_frame, running, display_mode
    
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    
    expected_shape = None
    frame_generation_count = 0
    
    while running:
        with frame_lock_left:
            local_frame_left = frame_left.copy() if frame_left is not None else None
            
        with frame_lock_right:
            local_frame_right = frame_right.copy() if frame_right is not None else None
            
        with centers_lock_left:
            local_centers_left = centers_left.copy()
            
        with centers_lock_right:
            local_centers_right = centers_right.copy()
            
        if local_frame_left is None or local_frame_right is None:
            time.sleep(0.01)
            continue

        frame_generation_count += 1
        
        # Calculate FPS
        fps_counter += 1
        if fps_counter >= 30:
            elapsed = time.time() - fps_start_time
            current_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start_time = time.time()
        
        h_left, w_left = local_frame_left.shape[:2]
        h_right, w_right = local_frame_right.shape[:2]
        
        # Get average x positions
        avg_x_left = get_average_x_position(local_centers_left)
        avg_x_right = get_average_x_position(local_centers_right)
        
        # Cut left frame first
        if display_mode == 1 and avg_x_left is not None and avg_x_right is not None:
            left_cut = local_frame_left[:, 0:avg_x_left]
        else:
            left_cut = local_frame_left
        
        # Use left cut height as target height
        target_height = left_cut.shape[0]
        
        # Resize right frame to match left cut height if needed
        if h_right != target_height:
            scale = target_height / h_right
            new_width_right = int(w_right * scale)
            local_frame_right = cv2.resize(local_frame_right, (new_width_right, target_height))
            # Scale the centers accordingly
            scale_x_right = new_width_right / w_right
            scale_y_right = target_height / h_right
            local_centers_right = [(int(x * scale_x_right), int(y * scale_y_right)) for x, y in local_centers_right]
            # Update avg_x_right if it was calculated
            if avg_x_right is not None:
                avg_x_right = int(avg_x_right * scale_x_right)
        
        # Calculate Y-difference for alignment
        y_diff = calculate_y_alignment_offset(local_centers_left, local_centers_right)
        
        # Apply vertical roll
        if y_diff != 0:
            roll_amount = -y_diff
            rolled_frame_right = np.roll(local_frame_right, roll_amount, axis=0)
            
            if roll_amount > 0:
                rolled_frame_right[:roll_amount, :] = 0
            elif roll_amount < 0:
                rolled_frame_right[roll_amount:, :] = 0
        else:
            rolled_frame_right = local_frame_right
        
        # Create combined view
        if display_mode == 1 and avg_x_left is not None and avg_x_right is not None:
            right_cut = rolled_frame_right[:, avg_x_right:]
            
            if left_cut.shape[1] > 0 and right_cut.shape[1] > 0:
                combined = np.hstack((left_cut, right_cut))
                junction_x = left_cut.shape[1]
            else:
                combined = np.hstack((left_cut, rolled_frame_right))
            
            cv2.putText(combined, "COMBINED VIEW", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        else:
            combined = np.hstack((left_cut, rolled_frame_right))
            cv2.putText(combined, "ORIGINAL VIEW", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Resize to fixed dimensions
        original_shape = combined.shape
        if combined.shape[1] != FIXED_COMBINED_WIDTH or combined.shape[0] != FIXED_COMBINED_HEIGHT:
            combined = cv2.resize(combined, (FIXED_COMBINED_WIDTH, FIXED_COMBINED_HEIGHT))
        
        # Check size consistency (for debugging)
        if expected_shape is None:
            expected_shape = (FIXED_COMBINED_HEIGHT, FIXED_COMBINED_WIDTH, 3)
            print(f"[DEBUG] Combined frame fixed to: {expected_shape}")
            print(f"[DEBUG] Original variable size was: {original_shape}")
        
        # Add info overlay
        cv2.putText(combined, f"FPS: {current_fps:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(combined, f"Left: {len(local_centers_left)} | Right: {len(local_centers_right)}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if local_centers_left and local_centers_right:
            cv2.putText(combined, f"Y-Diff: {y_diff}px", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Store combined frame
        with combined_frame_lock:
            combined_frame = combined.copy()
        
        # Direct streaming to FFmpeg (no queue)
        if 'combined' in ffmpeg_processes and ffmpeg_processes['combined'] is not None:
            with ffmpeg_locks['combined']:
                try:
                    ffmpeg_processes['combined'].stdin.write(combined.tobytes())
                except (BrokenPipeError, IOError):
                    pass
        
        time.sleep(1/30)  # 30 FPS output


def start_ffmpeg_rtsp_stream(stream_name, width, height, fps=30):
    """
    Start FFmpeg process to push stream to MediaMTX RTSP server
    Returns the subprocess object
    """
    command = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-b:v', '2000k',
        '-maxrate', '2000k',
        '-bufsize', '4000k',
        '-pix_fmt', 'yuv420p',
        '-g', str(fps),
        '-f', 'rtsp',
        '-rtsp_transport', 'tcp',
        f'rtsp://localhost:8554/{stream_name}'
    ]
    
    print(f"[DEBUG] Starting FFmpeg for {stream_name}")
    print(f"[DEBUG] Resolution: {width}x{height} @ {fps}fps")
    
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8
        )
        
        time.sleep(1.0)
        
        if process.poll() is not None:
            print(f"[ERROR] FFmpeg for {stream_name} exited immediately with code {process.returncode}")
            return None
        
        print(f"✓ {stream_name} stream: FFmpeg RTSP publisher started (rtsp://localhost:8554/{stream_name})")
        return process
    except Exception as e:
        print(f"✗ Failed to start FFmpeg for {stream_name}: {e}")
        return None


def init_ffmpeg_streams():
    """Initialize all FFmpeg streams"""
    global ffmpeg_processes
    
    # Wait for first frames
    print("Waiting for video frames to initialize FFmpeg streams...")
    time.sleep(2)
    
    with frame_lock_left:
        if frame_left is not None:
            h, w = frame_left.shape[:2]
            ffmpeg_processes['left'] = start_ffmpeg_rtsp_stream('left', w, h, fps=30)
    
    with frame_lock_right:
        if frame_right is not None:
            h, w = frame_right.shape[:2]
            ffmpeg_processes['right'] = start_ffmpeg_rtsp_stream('right', w, h, fps=30)
    
    # Combined uses fixed dimensions
    ffmpeg_processes['combined'] = start_ffmpeg_rtsp_stream('combined', 
                                                            FIXED_COMBINED_WIDTH, 
                                                            FIXED_COMBINED_HEIGHT, 
                                                            fps=30)


def cleanup_ffmpeg_processes():
    """Clean up all FFmpeg processes"""
    for name, process in ffmpeg_processes.items():
        if process is not None:
            try:
                process.stdin.close()
                process.terminate()
                process.wait(timeout=2)
            except:
                process.kill()
            print(f"✓ Stopped {name} FFmpeg process")


def main():
    global running
    
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
    os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
    
    print("\n" + "="*70)
    print("  Dual RTSP Video Processing (Direct Streaming - No Queue)")
    print("="*70)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("✗ CUDA not available")
    
    print(f"\n✓ Fixed combined resolution: {FIXED_COMBINED_WIDTH}x{FIXED_COMBINED_HEIGHT}")
    
    print("\n" + "="*70)
    print("  IMPORTANT: MediaMTX RTSP Server Required")
    print("="*70)
    print("\nPlease ensure MediaMTX is running:")
    print("  1. In another terminal run: ./mediamtx mediamtx.yml")
    print("="*70)
    
    input("\nPress Enter when MediaMTX is running...")
    
    print("\nStarting RTSP video processing threads...")
    
    # Start video processing threads
    thread_left = threading.Thread(target=process_video_left, daemon=True)
    thread_right = threading.Thread(target=process_video_right, daemon=True)
    thread_combine = threading.Thread(target=process_combined_frame, daemon=True)
    
    thread_left.start()
    thread_right.start()
    thread_combine.start()
    
    # Initialize FFmpeg streams after frames are ready
    time.sleep(3)
    init_ffmpeg_streams()
    
    print("\n" + "="*70)
    print("  All Streams Running (Direct Mode)")
    print("="*70)
    print("\nView streams using VLC or FFplay:")
    print("  ffplay rtsp://localhost:8554/left")
    print("  ffplay rtsp://localhost:8554/right")
    print("  ffplay rtsp://localhost:8554/combined")
    print("\nPress Ctrl+C to stop...")
    print("="*70 + "\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        running = False
    
    # Clean up
    print("\nCleaning up...")
    cleanup_ffmpeg_processes()
    
    thread_left.join(timeout=2)
    thread_right.join(timeout=2)
    thread_combine.join(timeout=2)
    
    print("\nAll threads stopped.")
    print("="*70)


if __name__ == "__main__":
    main()
