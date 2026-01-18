import cv2
import torch
from ultralytics import YOLO
import numpy as np
import threading
import time
import logging
import av
import subprocess
import queue
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

# Frame queues for FFmpeg streaming
frame_queue_left = queue.Queue(maxsize=30)
frame_queue_right = queue.Queue(maxsize=30)
frame_queue_combined = queue.Queue(maxsize=30)

# FFmpeg processes
ffmpeg_processes = {}
ffmpeg_stderr_queues = {}

# Fixed dimensions for combined stream
FIXED_COMBINED_WIDTH = 2560
FIXED_COMBINED_HEIGHT = 720


def process_video_left():
    global frame_left, centers_left, running

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO("../yoloTrain3/runs/detect/train/weights/best.pt", task="detect").to(device)

    file = "/home/yy/10.10.95.219_001M_202601091536527BD2.mp4"
    
    try:
        container = av.open(file)
        stream = container.streams.video[0]
        
        stream.codec_context.options = {
            'hwaccel': 'cuda',
            'hwaccel_device': '0',
            'hwaccel_output_format': 'cuda'
        }
        
        stream.thread_type = 'AUTO'
        
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        delay = 1.0 / fps
        
        print(f"Left video: {stream.width}x{stream.height} @ {fps}fps [GPU]")
        
    except Exception as e:
        print(f"Left video GPU error: {e}, using CPU")
        container = av.open(file)
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        delay = 1.0 / fps
    
    frame_count = 0
    
    while running:
        start_time = time.time()
        
        try:
            frame_obj = next(container.decode(stream))
            frame = frame_obj.to_ndarray(format='bgr24')
            frame_count += 1
            
        except StopIteration:
            container.seek(0)
            stream = container.streams.video[0]
            frame_count = 0
            continue
        except Exception as e:
            print(f"Left decode error: {e}")
            time.sleep(0.1)
            continue
        
        height, width = frame.shape[:2]
        
        results = model(frame, conf=0.25, verbose=False)
        
        current_centers = []
        if len(results) > 0:
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    current_centers.append((center_x, center_y))
                    
                    #cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    #cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    #cv2.line(frame, (center_x, center_y), (width, center_y), (0, 255, 0), 2)
        
        with centers_lock_left:
            centers_left = current_centers.copy()
        
        with frame_lock_left:
            frame_left = frame.copy()
        
        # Add to queue for FFmpeg streaming
        try:
            frame_queue_left.put_nowait(frame.copy())
        except queue.Full:
            pass  # Skip frame if queue is full
        
        elapsed = time.time() - start_time
        sleep_time = max(0, delay - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    container.close()


def process_video_right():
    global frame_right, centers_right, running

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO("../yoloTrain3/runs/detect/train/weights/best.pt", task="detect").to(device)
    
    file = "/home/yy/10.10.95.219_002M_202601091536524732.mp4"
    
    try:
        container = av.open(file)
        stream = container.streams.video[0]
        
        stream.codec_context.options = {
            'hwaccel': 'cuda',
            'hwaccel_device': '0',
            'hwaccel_output_format': 'cuda'
        }
        
        stream.thread_type = 'AUTO'
        
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        delay = 1.0 / fps
        
        print(f"Right video: {stream.width}x{stream.height} @ {fps}fps [GPU]")
        
    except Exception as e:
        print(f"Right video GPU error: {e}, using CPU")
        container = av.open(file)
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        delay = 1.0 / fps
    
    frame_count = 0
    
    while running:
        start_time = time.time()
        
        try:
            frame_obj = next(container.decode(stream))
            frame = frame_obj.to_ndarray(format='bgr24')
            frame_count += 1
            
        except StopIteration:
            container.seek(0)
            stream = container.streams.video[0]
            frame_count = 0
            continue
        except Exception as e:
            print(f"Right decode error: {e}")
            time.sleep(0.1)
            continue
        
        height, width = frame.shape[:2]
        
        results = model(frame, conf=0.25, verbose=False, classes=[1])
        
        current_centers = []
        if len(results) > 0:
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    current_centers.append((center_x, center_y))
                    
                    #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    #cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
                    #cv2.line(frame, (center_x, center_y), (0, center_y), (255, 255, 0), 2)
        
        with centers_lock_right:
            centers_right = current_centers.copy()
        
        with frame_lock_right:
            frame_right = frame.copy()
        
        # Add to queue for FFmpeg streaming
        try:
            frame_queue_right.put_nowait(frame.copy())
        except queue.Full:
            pass  # Skip frame if queue is full
        
        elapsed = time.time() - start_time
        sleep_time = max(0, delay - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    container.close()


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
    shape_change_count = 0
    
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
        
        target_height = min(h_left, h_right)
        
        new_width_left = w_left
        new_width_right = w_right
        
        # Resize left frame if needed
        if h_left != target_height:
            scale = target_height / h_left
            new_width_left = int(w_left * scale)
            local_frame_left = cv2.resize(local_frame_left, (new_width_left, target_height))
            scale_x_left = new_width_left / w_left
            scale_y_left = target_height / h_left
            local_centers_left = [(int(x * scale_x_left), int(y * scale_y_left)) for x, y in local_centers_left]
        
        # Resize right frame if needed
        if h_right != target_height:
            scale = target_height / h_right
            new_width_right = int(w_right * scale)
            local_frame_right = cv2.resize(local_frame_right, (new_width_right, target_height))
            scale_x_right = new_width_right / w_right
            scale_y_right = target_height / h_right
            local_centers_right = [(int(x * scale_x_right), int(y * scale_y_right)) for x, y in local_centers_right]
        
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
        
        # Get average x positions
        avg_x_left = get_average_x_position(local_centers_left)
        avg_x_right = get_average_x_position(local_centers_right)
        
        # Create combined view
        if display_mode == 1 and avg_x_left is not None and avg_x_right is not None:
            
            left_cut = local_frame_left[:, 0:avg_x_left]
            right_cut = rolled_frame_right[:, avg_x_right:new_width_right]
            
            if left_cut.shape[1] > 0 and right_cut.shape[1] > 0:
                combined = np.hstack((left_cut, right_cut))
                junction_x = left_cut.shape[1]
                #cv2.line(combined, (junction_x, 0), (junction_x, target_height), (0, 255, 255), 2)
            else:
                combined = np.hstack((local_frame_left, rolled_frame_right))
            
            cv2.putText(combined, "COMBINED VIEW", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        else:
            combined = np.hstack((local_frame_left, rolled_frame_right))
            cv2.putText(combined, "ORIGINAL VIEW", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # CRITICAL: Resize to fixed dimensions FIRST before adding any text
        # This ensures all frames sent to FFmpeg have consistent size
        original_shape = combined.shape
        if combined.shape[1] != FIXED_COMBINED_WIDTH or combined.shape[0] != FIXED_COMBINED_HEIGHT:
            combined = cv2.resize(combined, (FIXED_COMBINED_WIDTH, FIXED_COMBINED_HEIGHT))
        
        # Check size consistency (for debugging)
        if expected_shape is None:
            expected_shape = (FIXED_COMBINED_HEIGHT, FIXED_COMBINED_WIDTH, 3)
            print(f"[DEBUG] Combined frame fixed to: {expected_shape}")
            print(f"[DEBUG] Original variable size was: {original_shape}")
        
        # Add info overlay AFTER resizing to fixed dimensions
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
        
        # Add to queue for FFmpeg streaming
        try:
            frame_queue_combined.put_nowait(combined.copy())
        except queue.Full:
            pass
        
        time.sleep(1/30)  # 30 FPS output


def start_ffmpeg_rtsp_stream(stream_name, width, height, fps=30):
    """
    Start FFmpeg process to push stream to MediaMTX RTSP server
    Returns the subprocess object
    """
    # FFmpeg command to push to MediaMTX RTSP server
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
    print(f"[DEBUG] Command: {' '.join(command)}")
    
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8
        )
        
        # Create queue to store stderr
        stderr_queue = queue.Queue()
        ffmpeg_stderr_queues[stream_name] = stderr_queue
        
        # Start thread to read stderr
        def read_stderr():
            try:
                for line in iter(process.stderr.readline, b''):
                    if not line:
                        break
                    decoded = line.decode('utf-8', errors='ignore').strip()
                    stderr_queue.put(decoded)
                    # Print FFmpeg output in real-time
                    if 'error' in decoded.lower() or 'warning' in decoded.lower():
                        print(f"[FFmpeg-{stream_name}] {decoded}")
            except Exception as e:
                print(f"[ERROR] stderr reader for {stream_name}: {e}")
        
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stderr_thread.start()
        
        # Wait to see if it starts successfully
        time.sleep(1.0)
        
        if process.poll() is not None:
            print(f"[ERROR] FFmpeg for {stream_name} exited immediately with code {process.returncode}")
            print(f"[ERROR] FFmpeg stderr for {stream_name}:")
            while not stderr_queue.empty():
                print(f"  {stderr_queue.get()}")
            return None
        
        print(f"✓ {stream_name} stream: FFmpeg RTSP publisher started (rtsp://localhost:8554/{stream_name})")
        return process
    except Exception as e:
        print(f"✗ Failed to start FFmpeg for {stream_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def stream_left_video():
    """Stream left video using FFmpeg"""
    global running
    
    # Wait for first frame
    print("Waiting for left video frames...")
    frame = frame_queue_left.get()  # Block until first frame
    if frame is None or not running:
        return
    
    # Get frame dimensions
    h, w = frame.shape[:2]
    
    print(f"[DEBUG] Left first frame shape: {frame.shape}, dtype: {frame.dtype}")
    
    # Start FFmpeg process
    process = start_ffmpeg_rtsp_stream('left', w, h, fps=30)
    if process is None:
        print("Failed to start left FFmpeg process")
        return
    
    ffmpeg_processes['left'] = process
    
    frame_count = 0
    
    # Write first frame
    try:
        process.stdin.write(frame.tobytes())
        process.stdin.flush()
        frame_count += 1
    except Exception as e:
        print(f"Left stream initial write error: {e}")
        return
    
    # Continue streaming
    while running:
        try:
            # Check if process is still alive
            poll_result = process.poll()
            if poll_result is not None:
                print(f"[ERROR] Left FFmpeg died! Exit code: {poll_result}")
                break
            
            frame = frame_queue_left.get(timeout=1)
            if frame is None:
                break
            
            process.stdin.write(frame.tobytes())
            process.stdin.flush()
            frame_count += 1
            
            if frame_count % 300 == 0:
                print(f"[INFO] Left: {frame_count} frames written")
            
        except queue.Empty:
            continue
        except (BrokenPipeError, IOError) as e:
            if running:
                print(f"Left stream pipe error: {e}")
                break
        except Exception as e:
            print(f"Left stream error: {e}")
            time.sleep(0.1)
    
    try:
        process.stdin.close()
        process.wait(timeout=2)
    except:
        process.kill()
    
    print(f"Left stream stopped (wrote {frame_count} frames)")


def stream_right_video():
    """Stream right video using FFmpeg"""
    global running
    
    # Wait for first frame
    print("Waiting for right video frames...")
    frame = frame_queue_right.get()  # Block until first frame
    if frame is None or not running:
        return
    
    # Get frame dimensions
    h, w = frame.shape[:2]
    
    print(f"[DEBUG] Right first frame shape: {frame.shape}, dtype: {frame.dtype}")
    
    # Start FFmpeg process
    process = start_ffmpeg_rtsp_stream('right', w, h, fps=30)
    if process is None:
        print("Failed to start right FFmpeg process")
        return
    
    ffmpeg_processes['right'] = process
    
    frame_count = 0
    
    # Write first frame
    try:
        process.stdin.write(frame.tobytes())
        process.stdin.flush()
        frame_count += 1
    except Exception as e:
        print(f"Right stream initial write error: {e}")
        return
    
    # Continue streaming
    while running:
        try:
            # Check if process is still alive
            poll_result = process.poll()
            if poll_result is not None:
                print(f"[ERROR] Right FFmpeg died! Exit code: {poll_result}")
                break
            
            frame = frame_queue_right.get(timeout=1)
            if frame is None:
                break
            
            process.stdin.write(frame.tobytes())
            process.stdin.flush()
            frame_count += 1
            
            if frame_count % 300 == 0:
                print(f"[INFO] Right: {frame_count} frames written")
            
        except queue.Empty:
            continue
        except (BrokenPipeError, IOError) as e:
            if running:
                print(f"Right stream pipe error: {e}")
                break
        except Exception as e:
            print(f"Right stream error: {e}")
            time.sleep(0.1)
    
    try:
        process.stdin.close()
        process.wait(timeout=2)
    except:
        process.kill()
    
    print(f"Right stream stopped (wrote {frame_count} frames)")


def stream_combined_video():
    """Stream combined video using FFmpeg"""
    global running
    
    # Wait for first frame
    print("Waiting for combined video frames...")
    frame = frame_queue_combined.get()  # Block until first frame
    if frame is None or not running:
        return
    
    # Get frame dimensions
    h, w = frame.shape[:2]
    
    print(f"[DEBUG] Combined first frame shape: {frame.shape}, dtype: {frame.dtype}")
    
    # Start FFmpeg process
    process = start_ffmpeg_rtsp_stream('combined', w, h, fps=30)
    if process is None:
        print("Failed to start combined FFmpeg process")
        return
    
    ffmpeg_processes['combined'] = process
    
    frame_count = 0
    last_shape = frame.shape
    shape_changes = 0
    
    # Write first frame
    try:
        process.stdin.write(frame.tobytes())
        process.stdin.flush()
        frame_count += 1
        print(f"[DEBUG] Combined: wrote first frame")
    except Exception as e:
        print(f"Combined stream initial write error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Continue streaming
    while running:
        try:
            # Check if process is still alive
            poll_result = process.poll()
            if poll_result is not None:
                print(f"[ERROR] Combined FFmpeg died! Exit code: {poll_result}")
                print("[ERROR] Last stderr messages:")
                stderr_queue = ffmpeg_stderr_queues.get('combined')
                if stderr_queue:
                    messages = []
                    while not stderr_queue.empty() and len(messages) < 20:
                        messages.append(stderr_queue.get())
                    for msg in messages[-10:]:  # Show last 10 messages
                        print(f"  {msg}")
                break
            
            frame = frame_queue_combined.get(timeout=1)
            if frame is None:
                break
            
            # Check for frame size changes
            if frame.shape != last_shape:
                shape_changes += 1
                print(f"[WARNING] Combined frame shape changed from {last_shape} to {frame.shape} (change #{shape_changes})")
                last_shape = frame.shape
                
                # If too many shape changes, something is wrong
                if shape_changes > 10:
                    print(f"[ERROR] Too many shape changes ({shape_changes}), stopping")
                    break
            
            process.stdin.write(frame.tobytes())
            process.stdin.flush()
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"[INFO] Combined: {frame_count} frames written")
            
        except queue.Empty:
            continue
        except (BrokenPipeError, IOError) as e:
            if running:
                print(f"Combined stream pipe error: {e}")
                print(f"[DEBUG] Frames written before error: {frame_count}")
                
                # Print FFmpeg stderr
                stderr_queue = ffmpeg_stderr_queues.get('combined')
                if stderr_queue:
                    print("[ERROR] FFmpeg stderr (last 10 messages):")
                    messages = []
                    while not stderr_queue.empty() and len(messages) < 20:
                        messages.append(stderr_queue.get())
                    for msg in messages[-10:]:
                        print(f"  {msg}")
                break
        except Exception as e:
            print(f"Combined stream error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)
    
    try:
        process.stdin.close()
        process.wait(timeout=2)
    except:
        process.kill()
    
    print(f"Combined stream stopped (wrote {frame_count} frames)")


def cleanup_ffmpeg_processes():
    """Clean up all FFmpeg processes"""
    for name, process in ffmpeg_processes.items():
        try:
            process.stdin.close()
            process.terminate()
            process.wait(timeout=2)
        except:
            process.kill()
        print(f"✓ Stopped {name} FFmpeg process")


def main():
    global running
    
    # Suppress OpenCV warnings at the very beginning
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
    os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
    
    print("\n" + "="*70)
    print("  Dual Video Processing with RTSP Output (FFmpeg)")
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
    
    print("\nStarting video processing threads...")
    
    # Start video processing threads
    thread_left = threading.Thread(target=process_video_left, daemon=True)
    thread_right = threading.Thread(target=process_video_right, daemon=True)
    thread_combine = threading.Thread(target=process_combined_frame, daemon=True)
    
    thread_left.start()
    thread_right.start()
    thread_combine.start()
    
    # Wait for frames to be ready
    print("Waiting for video processing to start...")
    time.sleep(3)
    
    print("\n" + "="*70)
    print("  Starting RTSP Publishers (FFmpeg)")
    print("="*70)
    
    # Start FFmpeg streaming threads
    stream_left = threading.Thread(target=stream_left_video, daemon=True)
    stream_right = threading.Thread(target=stream_right_video, daemon=True)
    stream_combined = threading.Thread(target=stream_combined_video, daemon=True)
    
    stream_left.start()
    stream_right.start()
    stream_combined.start()
    
    time.sleep(2)
    
    print("\n" + "="*70)
    print("  All Streams Running")
    print("="*70)
    print("\nView streams using VLC or FFplay:")
    print("  ffplay rtsp://localhost:8554/left      (Left video)")
    print("  ffplay rtsp://localhost:8554/right     (Right video)")
    print("  ffplay rtsp://localhost:8554/combined  (Combined video)")
    print("\nOr in VLC: Media → Open Network Stream → rtsp://localhost:8554/left")
    print("\nTest WebRTC access:")
    print("  Open test_webrtc.html in browser")
    print("  Or check: http://localhost:8889/left/webrtc")
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
    stream_left.join(timeout=2)
    stream_right.join(timeout=2)
    stream_combined.join(timeout=2)
    
    print("\nAll threads stopped.")
    print("="*70)


if __name__ == "__main__":
    main()
