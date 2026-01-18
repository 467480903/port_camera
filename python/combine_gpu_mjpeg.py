import cv2
import torch
from ultralytics import YOLO
import numpy as np
import threading
import time
import logging
import av
import os
import warnings
from flask import Flask, Response

def setup_logging():
    """Suppress all unnecessary warnings and logging"""
    logging.getLogger('ultralytics').setLevel(logging.ERROR)
    logging.getLogger('opencv').setLevel(logging.ERROR)
    logging.getLogger('libav').setLevel(logging.ERROR)
    
    # Suppress OpenCV warnings
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
    os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
    
    # Suppress Python warnings
    warnings.filterwarnings('ignore')
    
    # Suppress FFmpeg/libav warnings
    av.logging.set_level(av.logging.ERROR)

setup_logging()

# Global frames (raw from RTSP)
frame_left = None
frame_right = None
frame_lock_left = threading.Lock()
frame_lock_right = threading.Lock()

# Combined frame (processed with YOLO)
combined_frame = None
combined_frame_lock = threading.Lock()

# Control flags
running = True
display_mode = 1  # 0 = original, 1 = combined

# Fixed dimensions for combined stream
FIXED_COMBINED_WIDTH = 2560
FIXED_COMBINED_HEIGHT = 720

modelpath = "/home/yy/port_camera/best.pt"
# modelpath = "/home/yy/port_camera/python/yolo11n.pt"

# Flask app for MJPEG streaming
app = Flask(__name__)


def process_video_left():
    """Thread 1: Read left RTSP stream only"""
    global frame_left, running

    rtsp_url = "rtsp://localhost:8554/cam2"
    
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
            if "SEI" not in str(e):  # Don't print SEI truncation errors
                print(f"Left decode error: {e}")
            time.sleep(0.01)
            continue
        
        # frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        # No rotation, no YOLO - just store the raw frame
        with frame_lock_left:
            frame_left = frame.copy()
    
    try:
        container.close()
    except:
        pass
    
    print("✓ Left RTSP reader thread stopped")


def process_video_right():
    """Thread 2: Read right RTSP stream only"""
    global frame_right, running

    rtsp_url = "rtsp://localhost:8554/cam3"
    
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
                    print(f"✓ Right RTSP connected (CPU): {stream.width}x{stream.height} @ {fps}fps")
                except Exception as e2:
                    print(f"✗ Right RTSP CPU fallback failed: {e2}")
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
            if "SEI" not in str(e):  # Don't print SEI truncation errors
                print(f"Right decode error: {e}")
            time.sleep(0.01)
            continue
        
        # Rotate 180 degrees
        # frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        # No YOLO - just store the raw frame
        with frame_lock_right:
            frame_right = frame.copy()
    
    try:
        container.close()
    except:
        pass
    
    print("✓ Right RTSP reader thread stopped")


def calculate_y_alignment_offset(centers_left, centers_right):
    """Calculate vertical offset between two sets of centers"""
    if not centers_left or not centers_right:
        return 0
    
    try:
        avg_left_y = np.mean([center[1] for center in centers_left])
        avg_right_y = np.mean([center[1] for center in centers_right])
        y_diff = int(avg_right_y - avg_left_y)
        return y_diff
    except Exception as e:
        print(f"Error calculating Y alignment: {e}")
        return 0


def get_average_x_position(centers):
    """Get average x position of centers"""
    if not centers:
        return None
    try:
        return int(np.mean([center[0] for center in centers]))
    except Exception as e:
        print(f"Error calculating average X: {e}")
        return None


def process_combined_frame():
    """Thread 3: YOLO detection, alignment, and combination"""
    global combined_frame, running, display_mode
    
    try:
        # Load YOLO model once
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = YOLO(modelpath, task="detect").to(device)
        print(f"✓ YOLO model loaded on {device}")
    except Exception as e:
        print(f"✗ Failed to load YOLO model: {e}")
        return
    
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    
    expected_shape = None
    frame_generation_count = 0
    
    while running:
        try:
            # Get raw frames from global variables
            with frame_lock_left:
                local_frame_left = frame_left.copy() if frame_left is not None else None
                
            with frame_lock_right:
                local_frame_right = frame_right.copy() if frame_right is not None else None
                
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
            
            # Run YOLO detection on LEFT frame
            centers_left = []
            detection_count_left = 0
            
            try:
                results_left = model(local_frame_left, conf=0.25, verbose=False, classes=[0])
                
                if len(results_left) > 0:
                    for result in results_left:
                        boxes = result.boxes.cpu().numpy()
                        for box in boxes:
                            detection_count_left += 1
                            x1, y1, x2, y2 = box.xyxy[0].astype(int)
                            conf = box.conf[0]
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            centers_left.append((center_x, center_y))
                            
                            # Draw detection box
                            # cv2.rectangle(local_frame_left, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Draw center point
                            # cv2.circle(local_frame_left, (center_x, center_y), 5, (0, 0, 255), -1)
                            
                            # Show confidence
                            label = f"Class 41: {conf:.2f}"
                            # cv2.putText(local_frame_left, label, (x1, y1 - 10), 
                            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except Exception as e:
                print(f"Left YOLO detection error: {e}")
            
            # Add frame info to left
            # cv2.putText(local_frame_left, f"Left Detections: {detection_count_left}", (10, 30), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Run YOLO detection on RIGHT frame
            centers_right = []
            detection_count_right = 0
            
            try:
                results_right = model(local_frame_right, conf=0.15, verbose=False, classes=[0])
                
                if len(results_right) > 0:
                    for result in results_right:
                        boxes = result.boxes.cpu().numpy()
                        for box in boxes:
                            detection_count_right += 1
                            x1, y1, x2, y2 = box.xyxy[0].astype(int)
                            conf = box.conf[0]
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            centers_right.append((center_x, center_y))
                            
                            # Draw detection box
                            # cv2.rectangle(local_frame_right, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            
                            # Draw center point
                            # cv2.circle(local_frame_right, (center_x, center_y), 5, (0, 0, 255), -1)
                            
                            # Show confidence
                            label = f"Class 41: {conf:.2f}"
                            # cv2.putText(local_frame_right, label, (x1, y1 - 10), 
                            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            except Exception as e:
                print(f"Right YOLO detection error: {e}")
            
            # Add frame info to right
            # cv2.putText(local_frame_right, f"Right Detections: {detection_count_right}", (10, 30), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            h_left, w_left = local_frame_left.shape[:2]
            h_right, w_right = local_frame_right.shape[:2]
            
            # Get average x positions for cutting
            avg_x_left = get_average_x_position(centers_left)
            avg_x_right = get_average_x_position(centers_right)
            
            # Cut left frame
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
                centers_right = [(int(x * scale_x_right), int(y * scale_y_right)) for x, y in centers_right]
                # Update avg_x_right if it was calculated
                if avg_x_right is not None:
                    avg_x_right = int(avg_x_right * scale_x_right)
            
            # Calculate Y-difference for vertical alignment
            y_diff = calculate_y_alignment_offset(centers_left, centers_right)
            
            # Apply vertical roll to align
            if y_diff != 0:
                roll_amount = -y_diff
                rolled_frame_right = np.roll(local_frame_right, roll_amount, axis=0)
                
                if roll_amount > 0:
                    rolled_frame_right[:roll_amount, :] = 0
                elif roll_amount < 0:
                    rolled_frame_right[roll_amount:, :] = 0
            else:
                rolled_frame_right = local_frame_right
            
            # Create combined view (horizontal cutting and merging)
            if display_mode == 1 and avg_x_left is not None and avg_x_right is not None:
                right_cut = rolled_frame_right[:, avg_x_right:]
                
                if left_cut.shape[1] > 0 and right_cut.shape[1] > 0:
                    combined = np.hstack((left_cut, right_cut))
                    junction_x = left_cut.shape[1]
                else:
                    combined = np.hstack((left_cut, rolled_frame_right))
                
                # cv2.putText(combined, "COMBINED VIEW", (10, 60), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            else:
                combined = np.hstack((left_cut, rolled_frame_right))
                # cv2.putText(combined, "ORIGINAL VIEW", (10, 60), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
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
            # cv2.putText(combined, f"FPS: {current_fps:.1f}", (10, 90), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.putText(combined, f"Left: {len(centers_left)} | Right: {len(centers_right)}", 
            #            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # if centers_left and centers_right:
            #     cv2.putText(combined, f"Y-Diff: {y_diff}px", (10, 150), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Store combined frame
            with combined_frame_lock:
                combined_frame = combined.copy()
            
            time.sleep(1/30)  # 30 FPS output
            
        except Exception as e:
            print(f"Error in combined frame processing: {e}")
            time.sleep(0.1)
    
    print("✓ YOLO combine thread stopped")


def generate_mjpeg():
    """Thread 4: MJPEG server streaming (generator function)"""
    while True:
        with combined_frame_lock:
            if combined_frame is None:
                time.sleep(0.1)
                continue
            frame = combined_frame.copy()
        
        try:
            # Encode frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            
            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        except Exception as e:
            print(f"MJPEG encoding error: {e}")
            continue
        
        time.sleep(1/30)  # 30 FPS


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Home page with embedded video"""
    return """
    <html>
    <head>
        <title>Combined Video Stream</title>
        <style>
            body {
                margin: 0;
                padding: 20px;
                background-color: #1a1a1a;
                color: white;
                font-family: Arial, sans-serif;
            }
            h1 {
                text-align: center;
                color: #4CAF50;
            }
            .container {
                max-width: 100%;
                margin: 0 auto;
                text-align: center;
            }
            img {
                max-width: 100%;
                height: auto;
                border: 2px solid #4CAF50;
                border-radius: 5px;
            }
            .info {
                margin-top: 20px;
                padding: 10px;
                background-color: #2a2a2a;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <h1>Dual Camera Combined View (MJPEG Stream)</h1>
        <div class="container">
            <img src="/video_feed" alt="Combined Video Stream">
            <div class="info">
                <p>Resolution: 2560x720 @ 30fps</p>
                <p>Stream URL: http://localhost:5000/video_feed</p>
                <p>Thread 1: Left RTSP Reader | Thread 2: Right RTSP Reader</p>
                <p>Thread 3: YOLO Detection + Alignment + Combine</p>
                <p>Thread 4: MJPEG Server (Flask)</p>
            </div>
        </div>
    </body>
    </html>
    """


def run_flask_server():
    """Run Flask server in a thread"""
    # Suppress Flask startup messages
    import logging as flask_logging
    flask_log = flask_logging.getLogger('werkzeug')
    flask_log.setLevel(flask_logging.ERROR)
    
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)


def main():
    global running
    
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
    os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
    
    print("\n" + "="*70)
    print("  Dual RTSP Video Processing (4-Thread Architecture)")
    print("="*70)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("✗ CUDA not available")
    
    print(f"\n✓ Fixed combined resolution: {FIXED_COMBINED_WIDTH}x{FIXED_COMBINED_HEIGHT}")
    
    print("\nThread Architecture:")
    print("  Thread 1: Read Left RTSP stream")
    print("  Thread 2: Read Right RTSP stream (rotated 180°)")
    print("  Thread 3: YOLO detection + Vertical align + Horizontal cut/merge")
    print("  Thread 4: MJPEG HTTP server (Flask)")
    
    print("\nStarting threads...")
    
    # Start video processing threads
    thread_left = threading.Thread(target=process_video_left, daemon=True, name="RTSP-Left")
    thread_right = threading.Thread(target=process_video_right, daemon=True, name="RTSP-Right")
    thread_combine = threading.Thread(target=process_combined_frame, daemon=True, name="YOLO-Combine")
    
    thread_left.start()
    thread_right.start()
    thread_combine.start()
    
    # Wait a bit for RTSP streams to connect
    time.sleep(2)
    
    # Start Flask server in a thread
    flask_thread = threading.Thread(target=run_flask_server, daemon=True, name="Flask-MJPEG")
    flask_thread.start()
    
    print("\n" + "="*70)
    print("  MJPEG Server Running")
    print("="*70)
    print("\nAccess the stream:")
    print("  Web browser: http://localhost:5000")
    print("  Direct stream: http://localhost:5000/video_feed")
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
    
    thread_left.join(timeout=2)
    thread_right.join(timeout=2)
    thread_combine.join(timeout=2)
    
    print("\nAll threads stopped.")
    print("="*70)


if __name__ == "__main__":
    main()