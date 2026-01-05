import cv2
import torch
from ultralytics import YOLO
import threading
from flask import Flask, Response
import io
import time
import logging
import numpy as np  

# Global variables
global frame1, frame2, lock1, lock2
global cap1, cap2
global model
global combined_frame 

# Initialize with a blank frame
combined_frame = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(combined_frame, "Starting up...", 
           (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def setup_logging():
    logging.getLogger('ultralytics').setLevel(logging.ERROR)

setup_logging()

frame1 = [None]
frame2 = [None]
lock1 = threading.Lock()
lock2 = threading.Lock()
cap1 = None
cap2 = None
model = None

def detect_objects(frame, model):
    try:
        # Only detect cups (class 41)
        results = model(frame, save=False, classes=[41], conf=0.3)
        for result in results:
            boxes = result.boxes.cpu().numpy()
            if len(boxes) > 0:
                # Get the first (most confident) detection
                box = boxes[0]
                r = box.xyxy[0].astype(int)
                return r  # Return the bounding box of the cup
    except Exception as e:
        print(f"Exception in detect_objects: {e}")
    return None

def process_stream(cap, model, output_frame, lock, cam_name):
    try:
        while not cap.isOpened():
            print(f"Waiting for stream {cam_name} to open...")
            time.sleep(1)
        
        print(f"Stream {cam_name} started successfully")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not read frame from {cam_name}")
                time.sleep(0.1)
                continue

            cup_position = detect_objects(frame, model)

            with lock:
                output_frame[0] = (frame, cup_position)
    except Exception as e:
        print(f"Exception in process_stream for {cam_name}: {e}")
    finally:
        if cap:
            cap.release()

def combine_frames():
    global combined_frame
    frame_count = 0
    
    while True:
        frame_count += 1
        
        # Get frames from both cameras
        with lock1:
            frame1_data = frame1[0]
            if frame1_data is not None:
                frame1_copy, cup1 = frame1_data
            else:
                frame1_copy, cup1 = None, None
        
        with lock2:
            frame2_data = frame2[0]
            if frame2_data is not None:
                frame2_copy, cup2 = frame2_data
            else:
                frame2_copy, cup2 = None, None

        # Check if we have valid frames and cup detections
        if (frame1_copy is not None and frame2_copy is not None and 
            cup1 is not None and cup2 is not None):
            
            try:
                # Calculate cup center points
                cup1_center_x = (cup1[0] + cup1[2]) // 2  # x-center of cup in frame1
                cup2_center_x = (cup2[0] + cup2[2]) // 2  # x-center of cup in frame2
                
                # Calculate vertical positions (y-center) of both cups
                cup1_center_y = (cup1[1] + cup1[3]) // 2  # y-center of cup in frame1
                cup2_center_y = (cup2[1] + cup2[3]) // 2  # y-center of cup in frame2
                
                # Calculate vertical difference
                vertical_diff = cup1_center_y - cup2_center_y
                
                # Take left half of cup from frame1 (everything up to center of cup)
                left_part = frame1_copy[:, :cup1_center_x]
                left_height, left_width = left_part.shape[:2]
                
                # Take right half of cup from frame2 (from center of cup to end)
                right_part_raw = frame2_copy[:, cup2_center_x:]
                right_height, right_width = right_part_raw.shape[:2]
                
                # Create aligned right part
                right_part_aligned = np.zeros((left_height, right_width, 3), dtype=np.uint8)
                
                # Apply vertical alignment
                if vertical_diff >= 0:
                    # Right cup needs to move down
                    paste_y = min(vertical_diff, left_height)
                    copy_height = min(right_height, left_height - paste_y)
                    if copy_height > 0 and paste_y + copy_height <= left_height:
                        right_part_aligned[paste_y:paste_y + copy_height, :] = right_part_raw[:copy_height, :]
                else:
                    # Right cup needs to move up
                    cut_start = -vertical_diff
                    copy_height = min(right_height - cut_start, left_height)
                    if copy_height > 0 and cut_start < right_height:
                        right_part_aligned[:copy_height, :] = right_part_raw[cut_start:cut_start + copy_height, :]
                
                # Combine frames
                combined_frame = cv2.hconcat([left_part, right_part_aligned])

                cv2.imshow("combined_frame",combined_frame)
                
                # Draw cup bounding boxes on combined frame
                # Left cup (draw right half since we only took left part)
                cup1_right_x = cup1_center_x - cup1[0]  # Convert to combined frame coordinates
                cv2.rectangle(combined_frame, 
                            (cup1[0], cup1[1]), 
                            (cup1_right_x, cup1[3]), 
                            (255, 0, 0), 2)  # Blue for left cup
                
                # Right cup (draw left half since we only took right part)
                cup2_left_x = 0  # Starting from left edge of right part
                cup2_relative_y = max(0, vertical_diff) if vertical_diff >= 0 else 0
                cup2_draw_y1 = cup2[1] + cup2_relative_y
                cup2_draw_y2 = cup2[3] + cup2_relative_y
                cv2.rectangle(combined_frame, 
                            (left_width, cup2_draw_y1), 
                            (left_width + (cup2[2] - cup2_center_x), cup2_draw_y2), 
                            (0, 0, 255), 2)  # Red for right cup
                
                # Draw stitching line
                stitch_x = left_width
                height = combined_frame.shape[0]
                cv2.line(combined_frame, (stitch_x, 0), (stitch_x, height), 
                        (0, 255, 0), 3)  # Green line, thicker
                
                # Draw cup center lines
                cv2.line(combined_frame, (0, cup1_center_y), (stitch_x, cup1_center_y), 
                        (255, 255, 0), 2)  # Cyan line for left cup center
                
                right_cup_center_y = cup1_center_y  # After alignment
                cv2.line(combined_frame, (stitch_x, right_cup_center_y), 
                        (combined_frame.shape[1], right_cup_center_y), 
                        (255, 0, 255), 2)  # Magenta line for right cup center
                
                # Add info text
                info_y = 30
                cv2.putText(combined_frame, f"Vertical Alignment: {vertical_diff}px", 
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(combined_frame, f"Cup Centers: L({cup1_center_x},{cup1_center_y}) R({cup2_center_x},{cup2_center_y})", 
                           (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
                cv2.putText(combined_frame, f"Frame: {frame_count}", 
                           (10, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)
                
                logging.error("combined_frame success")
                
            except Exception as e:
                print(f"Error in combine_frames: {e}")
                # Create error frame
                combined_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(combined_frame, f"Processing Error: {str(e)[:50]}", 
                           (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        else:
            # Show status message
            status_msg = []
            if frame1_copy is None or frame2_copy is None:
                status_msg.append("Waiting for camera feeds...")
            elif cup1 is None or cup2 is None:
                status_msg.append("Searching for cups...")
                if cup1 is None:
                    status_msg.append("No cup in left camera")
                if cup2 is None:
                    status_msg.append("No cup in right camera")
            
            # Create status frame
            combined_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            for i, msg in enumerate(status_msg):
                cv2.putText(combined_frame, msg, 
                           (50, 200 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
            # Small delay to prevent high CPU usage
            time.sleep(0.01)
            logging.error('frame detect error')

        

def generate_frames():
    global combined_frame
    while True:
        # Ensure combined_frame is valid
        if combined_frame is None or combined_frame.size == 0:
            # Create a default frame
            default_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(default_frame, "No video feed", 
                       (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', default_frame)
        else:
            # Encode the current combined frame
            ret, buffer = cv2.imencode('.jpg', combined_frame)
        
        if ret:
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # Send a blank frame if encoding fails
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')

app = Flask(__name__)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    global frame1, frame2, lock1, lock2
    global cap1, cap2
    global model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        model = YOLO("./yolo11n.pt", task="detect").to(device)
        print("YOLO model loaded successfully")
    except Exception as e:
        print(f"Exception loading model: {e}")
        return

    # RTSP URLs
    rtsp_url1 = "rtsp://localhost:8554/cam3"
    rtsp_url2 = "rtsp://localhost:8554/cam2"

    # Configure video capture with timeouts
    cap1 = cv2.VideoCapture(rtsp_url1)
    cap2 = cv2.VideoCapture(rtsp_url2)
    
    # Set buffer size to reduce latency
    cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap2.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Set frame size if needed
    # cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap1.isOpened():
        print(f"Error: Could not open video stream from {rtsp_url1}")
        return
    if not cap2.isOpened():
        print(f"Error: Could not open video stream from {rtsp_url2}")
        return

    print("Both cameras opened successfully")

    frame1 = [None]
    frame2 = [None]

    lock1 = threading.Lock()
    lock2 = threading.Lock()

    # Start processing threads
    thread1 = threading.Thread(target=process_stream, args=(cap1, model, frame1, lock1, "Left Camera"))
    thread2 = threading.Thread(target=process_stream, args=(cap2, model, frame2, lock2, "Right Camera"))
    thread3 = threading.Thread(target=combine_frames)

    thread1.daemon = True
    thread2.daemon = True
    thread3.daemon = True

    thread1.start()
    thread2.start()
    thread3.start()

    print("Starting Flask server on http://0.0.0.0:5000")
    print("Video feed available at http://0.0.0.0:5000/video_feed")
    
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Release the VideoCapture objects
        if cap1:
            cap1.release()
        if cap2:
            cap2.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()