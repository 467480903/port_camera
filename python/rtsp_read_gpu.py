import av
import cv2
import numpy as np
import threading
import time
from queue import Queue
from ultralytics import YOLO

# Configuration - Set your RTSP URLs here
RTSP_URL_1 = "rtsp://localhost:8554/cam3"
RTSP_URL_2 = "rtsp://localhost:8554/cam2"
HW_ACCEL = "cuda"  # Options: "cuda", "vaapi", "qsv", "dxva2", or None
YOLO_MODEL = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.5

class RTSPStream:
    def __init__(self, url, name="Stream", hw_accel=None):
        self.url = url
        self.name = name
        self.hw_accel = hw_accel
        self.container = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.fps = 0
        self.frame_count = 0
        self.last_frame_time = time.time()
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_thread, daemon=True)
        self.thread.start()
        print(f"{self.name}: Started processing")
        
    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        if self.container:
            self.container.close()
        print(f"{self.name}: Stopped")
        
    def _capture_thread(self):
        while self.running:
            try:
                if self.container is None:
                    print(f"{self.name}: Connecting to {self.url}")
                    
                    options = {
                        'rtsp_transport': 'tcp',
                        'max_delay': '500000',
                        'buffer_size': '1024000',
                    }
                    
                    self.container = av.open(self.url, options=options, timeout=10.0)
                    self.video_stream = self.container.streams.video[0]
                    
                    # Set up hardware acceleration
                    if self.hw_accel:
                        try:
                            codec_context = self.video_stream.codec_context
                            codec_context.thread_type = 'AUTO'
                            
                            if self.hw_accel == 'cuda':
                                codec_context.options = {'hwaccel': 'cuda'}
                            elif self.hw_accel == 'vaapi':
                                codec_context.options = {'hwaccel': 'vaapi'}
                            elif self.hw_accel == 'qsv':
                                codec_context.options = {'hwaccel': 'qsv'}
                            elif self.hw_accel == 'dxva2':
                                codec_context.options = {'hwaccel': 'dxva2'}
                            
                            print(f"{self.name}: Hardware acceleration ({self.hw_accel}) enabled")
                        except Exception as e:
                            print(f"{self.name}: Hardware acceleration failed: {e}")
                    
                    print(f"{self.name}: Connected - {self.video_stream.width}x{self.video_stream.height}")
                
                # Decode frames
                for packet in self.container.demux(self.video_stream):
                    if not self.running:
                        break
                        
                    for frame in packet.decode():
                        if not self.running:
                            break
                            
                        # Convert to numpy array
                        img = frame.to_ndarray(format='bgr24')
                        
                        # Calculate FPS
                        current_time = time.time()
                        self.frame_count += 1
                        if current_time - self.last_frame_time >= 1.0:
                            self.fps = self.frame_count
                            self.frame_count = 0
                            self.last_frame_time = current_time
                        
                        with self.lock:
                            self.frame = img
                        
                        break  # Process one frame per packet for lower latency
                    
            except Exception as e:
                print(f"{self.name}: Error - {e}")
                if self.container:
                    try:
                        self.container.close()
                    except:
                        pass
                    self.container = None
                time.sleep(1)
        
        if self.container:
            try:
                self.container.close()
            except:
                pass
            
    def get_frame(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
        return None
        
    def get_fps(self):
        return self.fps

def detect_cups(frame, model, conf_threshold):
    """Detect cups in frame and return annotated frame with detection info"""
    if frame is None:
        return None, 0
    
    # Run YOLO detection
    results = model(frame, conf=conf_threshold, verbose=False)
    
    cups_detected = 0
    
    # Process detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get class and confidence
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]
            
            # Check if detected object is a cup
            if class_name.lower() == 'cup':
                cups_detected += 1
                
                # Draw bounding box (green)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f'Cup {conf:.2f}'
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_y = max(y1, label_size[1] + 10)
                
                # Draw label background
                cv2.rectangle(frame, (x1, label_y - label_size[1] - 10),
                            (x1 + label_size[0] + 10, label_y), (0, 255, 0), -1)
                
                # Draw label text
                cv2.putText(frame, label, (x1 + 5, label_y - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return frame, cups_detected

def create_combined_frame(frame1, frame2, stream1, stream2, cups1, cups2, total_cups1, total_cups2):
    """Combine two frames horizontally with labels"""
    
    # Create placeholder if frames are None
    if frame1 is None:
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame1, "Waiting for Camera 1...", (100, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    if frame2 is None:
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame2, "Waiting for Camera 2...", (100, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Resize to same height
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]
    target_height = min(h1, h2)
    
    frame1 = cv2.resize(frame1, (int(w1 * target_height / h1), target_height))
    frame2 = cv2.resize(frame2, (int(w2 * target_height / h2), target_height))
    
    h, w = frame1.shape[:2]
    w2_resized = frame2.shape[1]
    
    # Create combined frame
    combined = np.zeros((h, w + w2_resized, 3), dtype=np.uint8)
    combined[0:h, 0:w] = frame1
    combined[0:h, w:w + w2_resized] = frame2
    
    # Add dividing line
    cv2.line(combined, (w, 0), (w, h), (255, 255, 255), 3)
    
    # Add labels for Camera 1
    cv2.putText(combined, f"Camera 1 - {stream1.fps}FPS", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(combined, f"Cups: {cups1} | Total: {total_cups1}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Add labels for Camera 2
    cv2.putText(combined, f"Camera 2 - {stream2.fps}FPS", (w + 10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(combined, f"Cups: {cups2} | Total: {total_cups2}", (w + 10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return combined

def main():
    print("=" * 60)
    print("Dual RTSP Stream Viewer with YOLO Cup Detection")
    print("=" * 60)
    print(f"Camera 1: {RTSP_URL_1}")
    print(f"Camera 2: {RTSP_URL_2}")
    print(f"Hardware Acceleration: {HW_ACCEL if HW_ACCEL else 'None (Software)'}")
    print(f"YOLO Model: {YOLO_MODEL}")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print("=" * 60)
    
    # Load YOLO model
    print("\nLoading YOLO model...")
    model = YOLO(YOLO_MODEL)
    print("Model loaded successfully!\n")
    
    # Create streams
    stream1 = RTSPStream(RTSP_URL_1, "Camera 1", HW_ACCEL)
    stream2 = RTSPStream(RTSP_URL_2, "Camera 2", HW_ACCEL)
    
    # Start streams
    stream1.start()
    stream2.start()
    
    # Wait for streams to initialize
    print("Waiting for streams to initialize...\n")
    time.sleep(2)
    
    # Create window
    window_name = "Dual RTSP Stream - Cup Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 480)
    
    print("Press 'q' or ESC to quit\n")
    
    total_cups1 = 0
    total_cups2 = 0
    
    try:
        while True:
            # Get frames from both streams
            frame1 = stream1.get_frame()
            frame2 = stream2.get_frame()
            
            # Detect cups in both frames
            if frame1 is not None:
                frame1, cups1 = detect_cups(frame1, model, CONFIDENCE_THRESHOLD)
                if cups1 > 0:
                    total_cups1 += cups1
            else:
                cups1 = 0
            
            if frame2 is not None:
                frame2, cups2 = detect_cups(frame2, model, CONFIDENCE_THRESHOLD)
                if cups2 > 0:
                    total_cups2 += cups2
            else:
                cups2 = 0
            
            # Create combined display
            combined = create_combined_frame(frame1, frame2, stream1, stream2, 
                                            cups1, cups2, total_cups1, total_cups2)
            
            # Display
            cv2.imshow(window_name, combined)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\nQuitting...")
                break
            
            # Check if window closed
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("\nWindow closed")
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        stream1.stop()
        stream2.stop()
        cv2.destroyAllWindows()
        print("\nProgram terminated")
        print(f"Total cups detected - Camera 1: {total_cups1}, Camera 2: {total_cups2}")

if __name__ == "__main__":
    main()