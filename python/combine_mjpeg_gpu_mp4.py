import cv2
import torch
from ultralytics import YOLO
import numpy as np
import threading
import time
import logging
import av
import asyncio
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

def setup_logging():
    logging.getLogger('ultralytics').setLevel(logging.ERROR)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
setup_logging()

# Global frames and data
frame_left = None
frame_right = None
frame_lock_left = threading.Lock()
frame_lock_right = threading.Lock()

# Global center points
centers_left = []
centers_right = []
centers_lock_left = threading.Lock()
centers_lock_right = threading.Lock()

# Combined frame for WebRTC
combined_frame = None
combined_frame_lock = threading.Lock()

# Control flags
running = True
display_mode = 1  # 0 = original, 1 = combined

def process_video_left():
    global frame_left, centers_left, running

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO("../yoloTrain3/runs/detect/train/weights/best.pt", task="detect").to(device)

    file = "/home/yy/10.10.95.219_001M_202601091536527BD2.mp4"
    
    try:
        container = av.open(file)
        stream = container.streams.video[0]
        
        stream.codec_context.options = {
            'hwaccel': 'cuda',
            'hwaccel_device': '0',
            'hwaccel_output_format': 'cuda'
        }
        
        stream.thread_type = 'AUTO'
        
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        delay = 1.0 / fps
        
        print(f"Left video: {stream.width}x{stream.height} @ {fps}fps [GPU]")
        
    except Exception as e:
        print(f"Left video GPU error: {e}, using CPU")
        container = av.open(file)
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        delay = 1.0 / fps
    
    frame_count = 0
    
    while running:
        start_time = time.time()
        
        try:
            frame_obj = next(container.decode(stream))
            frame = frame_obj.to_ndarray(format='bgr24')
            frame_count += 1
            
        except StopIteration:
            container.seek(0)
            stream = container.streams.video[0]
            frame_count = 0
            continue
        except Exception as e:
            print(f"Left decode error: {e}")
            time.sleep(0.1)
            continue
        
        height, width = frame.shape[:2]
        
        results = model(frame, conf=0.25, verbose=False)
        
        current_centers = []
        if len(results) > 0:
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    current_centers.append((center_x, center_y))
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    cv2.line(frame, (center_x, center_y), (width, center_y), (0, 255, 0), 2)
        
        with centers_lock_left:
            centers_left = current_centers.copy()
        
        with frame_lock_left:
            frame_left = frame
        
        elapsed = time.time() - start_time
        sleep_time = max(0, delay - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    container.close()


def process_video_right():
    global frame_right, centers_right, running

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO("../yoloTrain3/runs/detect/train/weights/best.pt", task="detect").to(device)
    
    file = "/home/yy/10.10.95.219_002M_202601091536524732.mp4"
    
    try:
        container = av.open(file)
        stream = container.streams.video[0]
        
        stream.codec_context.options = {
            'hwaccel': 'cuda',
            'hwaccel_device': '0',
            'hwaccel_output_format': 'cuda'
        }
        
        stream.thread_type = 'AUTO'
        
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        delay = 1.0 / fps
        
        print(f"Right video: {stream.width}x{stream.height} @ {fps}fps [GPU]")
        
    except Exception as e:
        print(f"Right video GPU error: {e}, using CPU")
        container = av.open(file)
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        delay = 1.0 / fps
    
    frame_count = 0
    
    while running:
        start_time = time.time()
        
        try:
            frame_obj = next(container.decode(stream))
            frame = frame_obj.to_ndarray(format='bgr24')
            frame_count += 1
            
        except StopIteration:
            container.seek(0)
            stream = container.streams.video[0]
            frame_count = 0
            continue
        except Exception as e:
            print(f"Right decode error: {e}")
            time.sleep(0.1)
            continue
        
        height, width = frame.shape[:2]
        
        results = model(frame, conf=0.25, verbose=False, classes=[1])
        
        current_centers = []
        if len(results) > 0:
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    current_centers.append((center_x, center_y))
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
                    cv2.line(frame, (center_x, center_y), (0, center_y), (255, 255, 0), 2)
        
        with centers_lock_right:
            centers_right = current_centers.copy()
        
        with frame_lock_right:
            frame_right = frame
        
        elapsed = time.time() - start_time
        sleep_time = max(0, delay - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    container.close()


def calculate_y_alignment_offset(centers_left, centers_right):
    if not centers_left or not centers_right:
        return 0
    
    avg_left_y = np.mean([center[1] for center in centers_left])
    avg_right_y = np.mean([center[1] for center in centers_right])
    y_diff = int(avg_right_y - avg_left_y)
    
    return y_diff


def get_average_x_position(centers):
    if not centers:
        return None
    return int(np.mean([center[0] for center in centers]))


def process_combined_frame():
    """Generate combined frame for WebRTC streaming"""
    global combined_frame, running, display_mode
    
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    
    while running:
        with frame_lock_left:
            local_frame_left = frame_left.copy() if frame_left is not None else None
            
        with frame_lock_right:
            local_frame_right = frame_right.copy() if frame_right is not None else None
            
        with centers_lock_left:
            local_centers_left = centers_left.copy()
            
        with centers_lock_right:
            local_centers_right = centers_right.copy()
            
        if local_frame_left is None or local_frame_right is None:
            time.sleep(0.01)
            continue
        
        # Calculate FPS
        fps_counter += 1
        if fps_counter >= 30:
            elapsed = time.time() - fps_start_time
            current_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start_time = time.time()
        
        h_left, w_left = local_frame_left.shape[:2]
        h_right, w_right = local_frame_right.shape[:2]
        
        target_height = min(h_left, h_right)
        
        new_width_left = w_left
        new_width_right = w_right
        
        # Resize left frame if needed
        if h_left != target_height:
            scale = target_height / h_left
            new_width_left = int(w_left * scale)
            local_frame_left = cv2.resize(local_frame_left, (new_width_left, target_height))
            scale_x_left = new_width_left / w_left
            scale_y_left = target_height / h_left
            local_centers_left = [(int(x * scale_x_left), int(y * scale_y_left)) for x, y in local_centers_left]
        
        # Resize right frame if needed
        if h_right != target_height:
            scale = target_height / h_right
            new_width_right = int(w_right * scale)
            local_frame_right = cv2.resize(local_frame_right, (new_width_right, target_height))
            scale_x_right = new_width_right / w_right
            scale_y_right = target_height / h_right
            local_centers_right = [(int(x * scale_x_right), int(y * scale_y_right)) for x, y in local_centers_right]
        
        # Calculate Y-difference for alignment
        y_diff = calculate_y_alignment_offset(local_centers_left, local_centers_right)
        
        # Apply vertical roll
        if y_diff != 0:
            roll_amount = -y_diff
            rolled_frame_right = np.roll(local_frame_right, roll_amount, axis=0)
            
            if roll_amount > 0:
                rolled_frame_right[:roll_amount, :] = 0
            elif roll_amount < 0:
                rolled_frame_right[roll_amount:, :] = 0
        else:
            rolled_frame_right = local_frame_right
        
        # Get average x positions
        avg_x_left = get_average_x_position(local_centers_left)
        avg_x_right = get_average_x_position(local_centers_right)
        
        # Create combined view
        if display_mode == 1 and avg_x_left is not None and avg_x_right is not None:
            left_cut = local_frame_left[:, 0:avg_x_left]
            right_cut = rolled_frame_right[:, avg_x_right:new_width_right]
            
            if left_cut.shape[1] > 0 and right_cut.shape[1] > 0:
                combined = np.hstack((left_cut, right_cut))
                junction_x = left_cut.shape[1]
                cv2.line(combined, (junction_x, 0), (junction_x, target_height), (0, 255, 255), 2)
            else:
                combined = np.hstack((local_frame_left, rolled_frame_right))
            
            cv2.putText(combined, "COMBINED VIEW", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            combined = np.hstack((local_frame_left, rolled_frame_right))
            cv2.putText(combined, "ORIGINAL VIEW", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add info overlay
        cv2.putText(combined, f"FPS: {current_fps:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(combined, f"Left: {len(local_centers_left)} | Right: {len(local_centers_right)}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if local_centers_left and local_centers_right:
            cv2.putText(combined, f"Y-Diff: {y_diff}px", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Store combined frame
        with combined_frame_lock:
            combined_frame = combined.copy()
        
        time.sleep(1/30)  # 30 FPS output


# WebRTC Video Track
class CombinedVideoTrack(VideoStreamTrack):
    """
    A video track that returns the combined video frame
    """
    kind = "video"
    
    def __init__(self):
        super().__init__()
        self.counter = 0
    
    async def recv(self):
        global combined_frame
        
        pts, time_base = await self.next_timestamp()
        
        # Get the combined frame
        with combined_frame_lock:
            if combined_frame is not None:
                frame = combined_frame.copy()
            else:
                # Return black frame if no frame available
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Convert BGR to RGB (WebRTC uses RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create VideoFrame
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        
        return video_frame


# WebRTC signaling
pcs = set()

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"Connection state: {pc.connectionState}")
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            await pc.close()
            pcs.discard(pc)

    # Add video track
    video_track = CombinedVideoTrack()
    pc.addTrack(video_track)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
    )


async def on_shutdown(app):
    # Close all peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


# HTML page for WebRTC client
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Dual Video Detection Stream</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            color: white;
            font-family: Arial, sans-serif;
        }
        h1 {
            text-align: center;
        }
        #video-container {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
        video {
            max-width: 90%;
            border: 2px solid #00ff00;
            border-radius: 8px;
        }
        .controls {
            text-align: center;
            margin-top: 20px;
        }
        button {
            padding: 10px 20px;
            margin: 5px;
            font-size: 16px;
            cursor: pointer;
            background: #00ff00;
            border: none;
            border-radius: 5px;
        }
        button:hover {
            background: #00cc00;
        }
        #status {
            text-align: center;
            margin-top: 10px;
            font-size: 14px;
            color: #00ff00;
        }
    </style>
</head>
<body>
    <h1>🎥 Dual Video Detection Stream</h1>
    
    <div id="video-container">
        <video id="video" autoplay playsinline muted></video>
    </div>
    
    <div class="controls">
        <button id="start-btn" onclick="start()">▶️ Start Stream</button>
        <button id="stop-btn" onclick="stop()" style="display:none;">⏹️ Stop Stream</button>
    </div>
    
    <div id="status">Ready to connect...</div>
    
    <script>
        let pc = null;
        
        async function start() {
            document.getElementById('status').textContent = 'Connecting...';
            document.getElementById('start-btn').style.display = 'none';
            
            const config = {
                iceServers: [{urls: 'stun:stun.l.google.com:19302'}]
            };
            
            pc = new RTCPeerConnection(config);
            
            // Add transceivers for receiving video
            pc.addTransceiver('video', {direction: 'recvonly'});
            
            pc.ontrack = (event) => {
                const video = document.getElementById('video');
                video.srcObject = event.streams[0];
                document.getElementById('status').textContent = '✓ Connected - Streaming';
                document.getElementById('stop-btn').style.display = 'inline-block';
            };
            
            pc.oniceconnectionstatechange = () => {
                console.log('ICE state:', pc.iceConnectionState);
                if (pc.iceConnectionState === 'disconnected' || 
                    pc.iceConnectionState === 'failed') {
                    document.getElementById('status').textContent = '✗ Connection lost';
                    stop();
                }
            };
            
            // Create offer
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);
            
            // Send offer to server
            const response = await fetch('/offer', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    sdp: pc.localDescription.sdp,
                    type: pc.localDescription.type
                })
            });
            
            const answer = await response.json();
            await pc.setRemoteDescription(new RTCSessionDescription(answer));
        }
        
        function stop() {
            if (pc) {
                pc.close();
                pc = null;
            }
            const video = document.getElementById('video');
            video.srcObject = null;
            document.getElementById('status').textContent = 'Stopped';
            document.getElementById('start-btn').style.display = 'inline-block';
            document.getElementById('stop-btn').style.display = 'none';
        }
    </script>
</body>
</html>
"""

import json

async def index(request):
    return web.Response(content_type="text/html", text=HTML)


def start_webrtc_server():
    """Start the WebRTC HTTP server"""
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.on_shutdown.append(on_shutdown)
    
    print("\n" + "="*60)
    print("  WebRTC Server Starting")
    print("="*60)
    print("  Open browser and navigate to: http://localhost:8080")
    print("="*60 + "\n")
    
    web.run_app(app, host="0.0.0.0", port=8089)


def main():
    global running
    
    print("\n" + "="*60)
    print("  Dual Video Processing with WebRTC Output")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("✗ CUDA not available")
    
    print("\nStarting video processing threads...")
    
    # Start video processing threads
    thread_left = threading.Thread(target=process_video_left, daemon=True)
    thread_right = threading.Thread(target=process_video_right, daemon=True)
    thread_combine = threading.Thread(target=process_combined_frame, daemon=True)
    
    thread_left.start()
    thread_right.start()
    thread_combine.start()
    
    # Wait for frames to be ready
    time.sleep(2)
    
    print("\nVideo processing started!")
    print("Starting WebRTC server...\n")
    
    try:
        # Start WebRTC server (blocking)
        start_webrtc_server()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        running = False
    
    thread_left.join(timeout=2)
    thread_right.join(timeout=2)
    thread_combine.join(timeout=2)
    
    print("All threads stopped.")
    print("="*60)


if __name__ == "__main__":
    main()