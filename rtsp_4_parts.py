import cv2
import threading
from flask import Flask, Response, stream_with_context
app = Flask(__name__)
# Replace these with your RTSP URLs
RTSP_URLS = [
"rtsp://user1:h7Hsu3ULLnLTs*M@10.10.95.215:554/media/video2",
"rtsp://user1:h7Hsu3ULLnLTs*M@10.10.95.216:554/media/video2",
"rtsp://user1:h7Hsu3ULLnLTs*M@10.10.95.218:554/media/video2"
]

def save_initial_frame(rtsp_url, filename):
    cap = cv2.VideoCapture(rtsp_url)
    success, frame = cap.read()
    if success:
        cv2.imwrite(filename, frame)
    cap.release()

def generate_frames(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    while True:
        success, frame = cap.read()
        if not success:
            cap.release()
            return
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

def gen_frame_wrapper(rtsp_url):
    return stream_with_context(generate_frames(rtsp_url))

@app.route('/video_feed1')
def video_feed1():
    return Response(gen_frame_wrapper(RTSP_URLS[0]),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(gen_frame_wrapper(RTSP_URLS[1]),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed3')
def video_feed3():
    return Response(gen_frame_wrapper(RTSP_URLS[2]),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)