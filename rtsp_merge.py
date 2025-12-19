import cv2
from flask import Flask, Response, stream_with_context

app = Flask(__name__)

# Replace these with your RTSP URLs
RTSP_URLS = [
    "rtsp://user1:h7Hsu3ULLnLTs*M@10.10.95.216:554/media/video2",
    "rtsp://user1:h7Hsu3ULLnLTs*M@10.10.95.218:554/media/video2"
]

# Global variables
cap1 = cv2.VideoCapture(RTSP_URLS[0])
cap2 = cv2.VideoCapture(RTSP_URLS[1])
frame1 = None
frame2 = None

def merge_videos():
    global frame1, frame2
    
    while True:
        success1, frame1 = cap1.read()
        success2, frame2 = cap2.read()
        
        if not success1 or not success2:
            break
        
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
        combined_frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + combined_frame + b'\r\n')

@app.route('/video_feed2')
def video_feed2():
    def gen_frame_wrapper1():
        global frame1
        while True:
            if frame1 is not None:
                ret, buffer = cv2.imencode('.jpg', frame1)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(stream_with_context(gen_frame_wrapper1()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed3')
def video_feed3():
    def gen_frame_wrapper2():
        global frame2
        while True:
            if frame2 is not None:
                ret, buffer = cv2.imencode('.jpg', frame2)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(stream_with_context(gen_frame_wrapper2()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/merged_video_feed')
def merged_video_feed():
    return Response(stream_with_context(merge_videos()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        cap1.release()
        cap2.release()