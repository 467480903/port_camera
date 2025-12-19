import cv2
import threading
from flask import Flask, Response, stream_with_context

app = Flask(__name__)

# Replace these with your RTSP URLs
RTSP_URLS = [
    "rtsp://user1:h7Hsu3ULLnLTs*M@10.10.95.216:554/media/video1",
    "rtsp://user1:h7Hsu3ULLnLTs*M@10.10.95.218:554/media/video1"
]

# Global variables
cap1 = cv2.VideoCapture(RTSP_URLS[0])
cap2 = cv2.VideoCapture(RTSP_URLS[1])
frame1 = None
frame2 = None
frame1_lock = threading.Lock()
frame2_lock = threading.Lock()

def save_initial_frame(rtsp_url, filename):
    cap = cv2.VideoCapture(rtsp_url)
    success, frame = cap.read()
    if success:
        cv2.imwrite(filename, frame)
    cap.release()

def generate_frames(cap, frame, frame_lock):
    global frame1, frame2
    while True:
        success, new_frame = cap.read()
        if not success:
            break
        with frame_lock:
            frame[:] = new_frame

def merge_videos():
    global frame1, frame2
    
    while True:
        with frame1_lock:
            frame1_copy = frame1.copy() if frame1 is not None else None
        with frame2_lock:
            frame2_copy = frame2.copy() if frame2 is not None else None
        
        if frame1_copy is None or frame2_copy is None:
            continue
        
        # Get the width and height of the frames
        height1, width1, _ = frame1_copy.shape
        height2, width2, _ = frame2_copy.shape
        
        # Resize frames if necessary to have the same height
        if height1 != height2:
            frame1_copy = cv2.resize(frame1_copy, (width1, height2))
        
        # Take left 2/3 of frame1
        left_frame1 = frame1_copy[:, :int(width1 * 60/100)]
        
        # Take right 2/3 of frame2
        right_frame2 = frame2_copy[:, int(width2 * 40/100):]
        
        # Concatenate the frames horizontally
        combined_frame = cv2.hconcat([left_frame1, right_frame2])
        
        ret, buffer = cv2.imencode('.jpg', combined_frame)
        combined_frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + combined_frame + b'\r\n')

def gen_frame_wrapper(frame, frame_lock):
    while True:
        with frame_lock:
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed2')
def video_feed2():
    global frame1, frame1_lock
    return Response(stream_with_context(gen_frame_wrapper(frame1, frame1_lock)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed3')
def video_feed3():
    global frame2, frame2_lock
    return Response(stream_with_context(gen_frame_wrapper(frame2, frame2_lock)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/merged_video_feed')
def merged_video_feed():
    return Response(stream_with_context(merge_videos()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Initialize frame buffers
    success1, frame1 = cap1.read()
    success2, frame2 = cap2.read()
    if not success1 or not success2:
        print("Failed to read initial frames")
        exit(1)
    
    # Start threads for frame generation
    frame_thread1 = threading.Thread(target=generate_frames, args=(cap1, frame1, frame1_lock))
    frame_thread2 = threading.Thread(target=generate_frames, args=(cap2, frame2, frame2_lock))
    frame_thread1.daemon = True
    frame_thread2.daemon = True
    frame_thread1.start()
    frame_thread2.start()
    
    try:
        app.run(host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        cap1.release()
        cap2.release()
        frame_thread1.join()
        frame_thread2.join()