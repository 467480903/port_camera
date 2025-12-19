import cv2
import threading
from flask import Flask, Response, stream_with_context

app = Flask(__name__)

# Replace these with your RTSP URLs
RTSP_URLS = [
    "rtsp://user1:h7Hsu3ULLnLTs*M@10.10.95.215:554/media/video2",
    "rtsp://user1:h7Hsu3ULLnLTs*M@10.10.95.219:554/media/video3",
    "rtsp://user1:h7Hsu3ULLnLTs*M@10.10.95.220:554/media/video3"
]

# Global variables
caps = [cv2.VideoCapture(url) for url in RTSP_URLS]
frames = [None] * len(RTSP_URLS)
locks = [threading.Lock()] * len(RTSP_URLS)

def generate_frames(cap, frame, frame_lock):
    while True:
        success, new_frame = cap.read()
        if not success:
            break
        with frame_lock:
            frame[:] = new_frame

def gen_frame_wrapper(index):
    global frames, locks
    while True:
        with locks[index]:
            if frames[index] is not None:
                ret, buffer = cv2.imencode('.jpg', frames[index])
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed1')
def video_feed1():
    return Response(stream_with_context(gen_frame_wrapper(0)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(stream_with_context(gen_frame_wrapper(1)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed3')
def video_feed3():
    return Response(stream_with_context(gen_frame_wrapper(2)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Initialize frames
    for i, cap in enumerate(caps):
        success, frame = cap.read()
        if not success:
            print(f"Failed to read initial frame from {RTSP_URLS[i]}")
            exit(1)
        frames[i] = frame
    
    # Start threads for frame generation
    threads = []
    for i in range(len(RTSP_URLS)):
        thread = threading.Thread(target=generate_frames, args=(caps[i], frames[i], locks[i]))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    try:
        app.run(host='0.0.0.0', port=5001)  # Use a different port to avoid conflict
    except KeyboardInterrupt:
        for cap in caps:
            cap.release()
        for thread in threads:
            thread.join()