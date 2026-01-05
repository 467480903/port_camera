import cv2
import numpy as np
import threading
import time
from queue import Queue
import argparse
from datetime import datetime

class RTSPStream:
    def __init__(self, url, name="Stream"):
        self.url = url
        self.name = name
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.frame_queue = Queue(maxsize=1)
        self.last_frame_time = 0
        self.frame_count = 0
        self.fps = 0
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_thread, daemon=True)
        self.thread.start()
        print(f"Started {self.name} capture thread")
        
    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        print(f"Stopped {self.name}")
        
    def _capture_thread(self):
        # Add connection timeout
        cv2.setNumThreads(1)
        
        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    print(f"{self.name}: Attempting to connect to {self.url}")
                    self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                    
                    # Set buffer size to reduce latency
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    if not self.cap.isOpened():
                        print(f"{self.name}: Failed to connect. Retrying...")
                        time.sleep(2)
                        continue
                        
                    width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = self.cap.get(cv2.CAP_PROP_FPS)
                    print(f"{self.name}: Connected! Resolution: {width}x{height}, FPS: {fps}")
                
                ret, frame = self.cap.read()
                if ret:
                    current_time = time.time()
                    
                    # Calculate FPS
                    self.frame_count += 1
                    if current_time - self.last_frame_time >= 1.0:
                        self.fps = self.frame_count
                        self.frame_count = 0
                        self.last_frame_time = current_time
                    
                    with self.lock:
                        self.frame = frame.copy()
                        
                    # Try to put frame in queue (non-blocking)
                    if not self.frame_queue.full():
                        try:
                            self.frame_queue.put_nowait(frame.copy())
                        except:
                            pass
                else:
                    print(f"{self.name}: Failed to read frame. Reconnecting...")
                    self.cap.release()
                    self.cap = None
                    time.sleep(1)
                    
            except Exception as e:
                print(f"{self.name}: Error in capture thread: {e}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
                time.sleep(1)
                
        if self.cap:
            self.cap.release()
            
    def get_frame(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
        return None
        
    def get_fps(self):
        return self.fps

class DualRTSPViewer:
    def __init__(self, rtsp_url1, rtsp_url2, layout='horizontal', scale_factor=0.5):
        self.stream1 = RTSPStream(rtsp_url1, "Camera 1")
        self.stream2 = RTSPStream(rtsp_url2, "Camera 2")
        self.layout = layout  # 'horizontal', 'vertical', or 'pip'
        self.scale_factor = scale_factor
        self.running = False
        self.display_fps = 0
        self.display_count = 0
        self.last_display_time = time.time()
        
    def start(self):
        print("Starting Dual RTSP Viewer...")
        print(f"Camera 1: {self.stream1.url}")
        print(f"Camera 2: {self.stream2.url}")
        
        self.stream1.start()
        self.stream2.start()
        
        # Wait a bit for streams to initialize
        time.sleep(2)
        
        self.running = True
        self._main_loop()
        
    def stop(self):
        print("Stopping Dual RTSP Viewer...")
        self.running = False
        self.stream1.stop()
        self.stream2.stop()
        cv2.destroyAllWindows()
        
    def _create_combined_frame(self, frame1, frame2):
        if frame1 is None and frame2 is None:
            return None
            
        if frame1 is None:
            frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame1, "No Camera 1", (100, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        if frame2 is None:
            frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame2, "No Camera 2", (100, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Resize frames if needed
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        
        # Resize to same height for horizontal layout
        if self.layout == 'horizontal':
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
            cv2.line(combined, (w, 0), (w, h), (255, 255, 255), 2)
            
            # Add labels
            cv2.putText(combined, f"Camera 1 - {self.stream1.fps}FPS", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined, f"Camera 2 - {self.stream2.fps}FPS", (w + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        elif self.layout == 'vertical':
            target_width = min(w1, w2)
            frame1 = cv2.resize(frame1, (target_width, int(h1 * target_width / w1)))
            frame2 = cv2.resize(frame2, (target_width, int(h2 * target_width / w2)))
            
            h, w = frame1.shape[:2]
            h2_resized = frame2.shape[0]
            
            # Create combined frame
            combined = np.zeros((h + h2_resized, w, 3), dtype=np.uint8)
            combined[0:h, 0:w] = frame1
            combined[h:h + h2_resized, 0:w] = frame2
            
            # Add dividing line
            cv2.line(combined, (0, h), (w, h), (255, 255, 255), 2)
            
            # Add labels
            cv2.putText(combined, f"Camera 1 - {self.stream1.fps}FPS", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined, f"Camera 2 - {self.stream2.fps}FPS", (10, h + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        elif self.layout == 'pip':  # Picture-in-Picture
            # Resize frame2 for PIP
            pip_height = int(h1 * self.scale_factor)
            pip_width = int(w1 * self.scale_factor * (w2 / h2))
            frame2_pip = cv2.resize(frame2, (pip_width, pip_height))
            
            # Create combined frame (frame1 as background)
            combined = frame1.copy()
            
            # Add frame2 as PIP (top-right corner)
            combined[10:10 + pip_height, w1 - pip_width - 10:w1 - 10] = frame2_pip
            
            # Add PIP border
            cv2.rectangle(combined, 
                         (w1 - pip_width - 10, 10), 
                         (w1 - 10, 10 + pip_height), 
                         (0, 255, 0), 2)
            
            # Add labels
            cv2.putText(combined, f"Camera 1 - {self.stream1.fps}FPS", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined, f"PIP: Camera 2 - {self.stream2.fps}FPS", 
                       (w1 - pip_width, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(combined, timestamp, (10, combined.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add display FPS
        cv2.putText(combined, f"Display: {self.display_fps}FPS", 
                   (combined.shape[1] - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return combined
        
    def _main_loop(self):
        window_name = "Dual RTSP Stream Viewer"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 480)
        
        print("\nControls:")
        print("  q, ESC - Quit")
        print("  s - Save current frame")
        print("  f - Toggle fullscreen")
        print("  1 - Show Camera 1 only")
        print("  2 - Show Camera 2 only")
        print("  h - Horizontal layout")
        print("  v - Vertical layout")
        print("  p - Picture-in-Picture layout")
        print("  + - Increase PIP size")
        print("  - - Decrease PIP size")
        print("  r - Reset layout")
        
        show_both = True
        show_stream1_only = False
        show_stream2_only = False
        
        while self.running:
            # Get frames from both streams
            frame1 = self.stream1.get_frame()
            frame2 = self.stream2.get_frame()
            
            display_frame = None
            
            if show_both:
                display_frame = self._create_combined_frame(frame1, frame2)
            elif show_stream1_only and frame1 is not None:
                display_frame = frame1.copy()
                cv2.putText(display_frame, f"Camera 1 - {self.stream1.fps}FPS", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif show_stream2_only and frame2 is not None:
                display_frame = frame2.copy()
                cv2.putText(display_frame, f"Camera 2 - {self.stream2.fps}FPS", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if display_frame is not None:
                # Calculate display FPS
                self.display_count += 1
                current_time = time.time()
                if current_time - self.last_display_time >= 1.0:
                    self.display_fps = self.display_count
                    self.display_count = 0
                    self.last_display_time = current_time
                
                cv2.imshow(window_name, display_frame)
            else:
                # Show waiting message
                waiting_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(waiting_frame, "Waiting for streams...", (100, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow(window_name, waiting_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # q or ESC
                print("\nQuitting...")
                break
            elif key == ord('s'):  # Save frame
                if display_frame is not None:
                    filename = f"frame_{int(time.time())}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"Saved frame to {filename}")
            elif key == ord('f'):  # Toggle fullscreen
                fullscreen = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
                if fullscreen == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            elif key == ord('1'):  # Show Camera 1 only
                show_both = False
                show_stream1_only = True
                show_stream2_only = False
                print("Showing Camera 1 only")
            elif key == ord('2'):  # Show Camera 2 only
                show_both = False
                show_stream1_only = False
                show_stream2_only = True
                print("Showing Camera 2 only")
            elif key == ord('h'):  # Horizontal layout
                show_both = True
                self.layout = 'horizontal'
                print("Switched to Horizontal layout")
            elif key == ord('v'):  # Vertical layout
                show_both = True
                self.layout = 'vertical'
                print("Switched to Vertical layout")
            elif key == ord('p'):  # Picture-in-Picture
                show_both = True
                self.layout = 'pip'
                print("Switched to Picture-in-Picture layout")
            elif key == ord('+'):  # Increase PIP size
                self.scale_factor = min(0.8, self.scale_factor + 0.1)
                print(f"PIP size increased to {self.scale_factor:.1f}")
            elif key == ord('-'):  # Decrease PIP size
                self.scale_factor = max(0.2, self.scale_factor - 0.1)
                print(f"PIP size decreased to {self.scale_factor:.1f}")
            elif key == ord('r'):  # Reset layout
                show_both = True
                self.layout = 'horizontal'
                self.scale_factor = 0.5
                print("Layout reset to default")
            
            # Check if window was closed
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("\nWindow closed by user")
                break
        
        self.stop()

def main():
    parser = argparse.ArgumentParser(description='Dual RTSP Stream Viewer')
    parser.add_argument('--cam1', default='rtsp://localhost:8554/cam3',
                       help='RTSP URL for Camera 1 (default: rtsp://localhost:8554/cam3)')
    parser.add_argument('--cam2', default='rtsp://localhost:8554/cam2',
                       help='RTSP URL for Camera 2 (default: rtsp://localhost:8554/cam2)')
    parser.add_argument('--layout', choices=['horizontal', 'vertical', 'pip'], 
                       default='horizontal',
                       help='Layout for displaying streams (default: horizontal)')
    parser.add_argument('--scale', type=float, default=0.5,
                       help='Scale factor for PIP layout (default: 0.5)')
    
    args = parser.parse_args()
    
    try:
        viewer = DualRTSPViewer(args.cam1, args.cam2, args.layout, args.scale)
        viewer.start()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Program terminated")

if __name__ == "__main__":
    main()