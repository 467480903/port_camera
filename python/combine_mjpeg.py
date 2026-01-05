import cv2
import torch
from ultralytics import YOLO
import threading
from flask import Flask, Response
import io
import time
import logging

# Global variables
global frame1, frame2, lock1, lock2
global cap1, cap2
global model
global combined_frame

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
        results = model(frame, save=False)
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                r = box.xyxy[0].astype(int)
                cls = box.cls[0].astype(int)
                conf = box.conf[0]

                if cls == 41:  # Assuming class 41 is 'cup'
                    
                    return r  # Return the bounding box of the cup
    except Exception as e:
        print(f"Exception in detect_objects: {e}")
    return None

def process_stream(cap, model, output_frame, lock):
    try:
        while not cap.isOpened():
            print(f"Waiting for stream to open...")
            time.sleep(1)
        
        while True:
            # time.sleep(0.03)
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not read frame")
                break

            cup_position = detect_objects(frame, model)

            with lock:
                output_frame[0] = (frame, cup_position)
    except Exception as e:
        print(f"Exception in process_stream: {e}")
    finally:
        cap.release()

def display_combined_frames():
    global combined_frame
    while True:
        with lock1:
            frame1_copy, cup1 = frame1[0] if frame1[0] is not None else (None, None)
        with lock2:
            frame2_copy, cup2 = frame2[0] if frame2[0] is not None else (None, None)

        if frame1_copy is not None and frame2_copy is not None and cup1 is not None and cup2 is not None:
            # Calculate cup center points
            cup1_center_x = (cup1[0] + cup1[2]) // 2  # x-center of cup in frame1
            cup2_center_x = (cup2[0] + cup2[2]) // 2  # x-center of cup in frame2
            
            # Take left half of cup from frame1 (everything up to center of cup)
            left_part = frame1_copy[:, :cup1_center_x]
            
            # Take right half of cup from frame2 (from center of cup to end)
            right_part = frame2_copy[:, cup2_center_x:]
            
            # Combine the left and right parts
            combined_frame = cv2.hconcat([left_part, right_part])
            
            # Get the stitching position (width of left part)
            stitch_x = left_part.shape[1]
            
            # Draw a vertical line at the stitching position
            height = combined_frame.shape[0]
            cv2.line(combined_frame, (stitch_x, 0), (stitch_x, height), (0, 255, 0), 2)  # Green line, thickness 2
            
            # Add label at the stitching line
            cv2.putText(combined_frame, "Stitching Line", 
                       (stitch_x - 100, 30),  # Position text near the line
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7,  # Font scale
                       (0, 255, 0),  # Green color
                       2)  # Thickness
            
            # Also visualize the cup halves on the original frames (for debugging)
            # Draw rectangle around left half of cup in frame1_copy
            cv2.rectangle(frame1_copy, (cup1[0], cup1[1]), (cup1_center_x, cup1[3]), (255, 0, 0), 2)
            cv2.putText(frame1_copy, "Left Half", (cup1[0], cup1[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Draw rectangle around right half of cup in frame2_copy
            cv2.rectangle(frame2_copy, (cup2_center_x, cup2[1]), (cup2[2], cup2[3]), (0, 0, 255), 2)
            cv2.putText(frame2_copy, "Right Half", (cup2_center_x, cup2[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Display individual frames for debugging (optional)
            # cv2.imshow('Left Camera', frame1_copy)
            # cv2.imshow('Right Camera', frame2_copy)
            
        else:
            # Create a blank frame if no cups detected
            combined_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(combined_frame, "Waiting for cup detection...", 
                       (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            time.sleep(0.1)

        # Display the combined frame in a window
        cv2.imshow('Combined Stream', combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def generate_frames():
    global combined_frame
    while True:


        # Encode the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', combined_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


app = Flask(__name__)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    global frame1, frame2, lock1, lock2
    global cap1, cap2
    global model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        model = YOLO("./yolo11n.pt", task="detect").to(device)
    except Exception as e:
        print(f"Exception loading model: {e}")
        return

    # rtsp_url1 = "rtsp://user1:h7Hsu3ULLnLTs*M@10.10.95.216:554/media/video2"
    # rtsp_url2 = "rtsp://user1:h7Hsu3ULLnLTs*M@10.10.95.218:554/media/video2"

    rtsp_url1 = "rtsp://localhost:8554/cam3"
    rtsp_url2 = "rtsp://localhost:8554/cam2"

    cap1 = cv2.VideoCapture(rtsp_url1)
    cap2 = cv2.VideoCapture(rtsp_url2)

    if not cap1.isOpened():
        print(f"Error: Could not open video stream from {rtsp_url1}")
        return
    if not cap2.isOpened():
        print(f"Error: Could not open video stream from {rtsp_url2}")
        return

    frame1 = [None]
    frame2 = [None]

    lock1 = threading.Lock()
    lock2 = threading.Lock()

    thread1 = threading.Thread(target=process_stream, args=(cap1, model, frame1, lock1))
    thread2 = threading.Thread(target=process_stream, args=(cap2, model, frame2, lock2))
    thread3 = threading.Thread(target=display_combined_frames)

    thread1.start()
    thread2.start()
    thread3.start()

    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        # Release the VideoCapture objects
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()

    thread1.join()
    thread2.join()
    thread3.join()

if __name__ == "__main__":
    main()