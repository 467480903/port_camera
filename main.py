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

def merge_videos(url1, url2):
    cap1 = cv2.VideoCapture(url1)
    cap2 = cv2.VideoCapture(url2)
    
    while True:
        success1, frame1 = cap1.read()
        success2, frame2 = cap2.read()
        
        if not success1 or not success2:
            cap1.release()
            cap2.release()
            return
        
        # Get the width and height of the frames
        height1, width1, _ = frame1.shape
        height2, width2, _ = frame2.shape
        
        # Resize frames if necessary to have the same height
        if height1 != height2:
            frame1 = cv2.resize(frame1, (width1, height2))
        
        # Take left 2/3 of frame1
        left_frame1 = frame1[:, :int(width1 * 60/100)]
        
        # Take right 2/3 of frame2
        right_frame2 = frame2[:, int(width2 * 40/100):]
        
        # Concatenate the frames horizontally
        combined_frame = cv2.hconcat([left_frame1, right_frame2])
        
        ret, buffer = cv2.imencode('.jpg', combined_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap1.release()
    cap2.release()

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

@app.route('/merged_video_feed')
def merged_video_feed():
    return Response(stream_with_context(merge_videos(RTSP_URLS[1], RTSP_URLS[2])),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)