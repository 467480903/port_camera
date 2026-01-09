import cv2
import torch
from ultralytics import YOLO
import numpy as np
import threading
import time
from collections import deque

# Global frames and data
frame_62 = None
frame_63 = None
frame_lock_62 = threading.Lock()
frame_lock_63 = threading.Lock()

# Global center points (store multiple detections)
centers_62 = []  # List of (x, y) tuples for video 62
centers_63 = []  # List of (x, y) tuples for video 63
centers_lock_62 = threading.Lock()
centers_lock_63 = threading.Lock()

# Control flags
running = True

def process_video_62():
    global frame_62, centers_62, running
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO("./best.pt", task="detect").to(device)
    
    # Open video
    cap = cv2.VideoCapture("/home/yy/62.mp4")
    if not cap.isOpened():
        print("Error: Could not open 62.mp4")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = 1.0 / fps if fps > 0 else 1.0/30.0
    
    while running:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        height, width = frame.shape[:2]
        
        # Clear previous centers
        with centers_lock_62:
            centers_62 = []
            
        # Detect class 0
        results = model(frame, save=False, classes=[0])
        
        # Draw detections and collect centers
        current_centers = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                if box.cls[0] == 0:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    
                    # Calculate center point
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    current_centers.append((center_x, center_y))
                    
                    # Draw bounding box (blue)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # Draw center point (red circle)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    
                    # Draw line from center point to RIGHT border (green)
                    cv2.line(frame, (center_x, center_y), (width, center_y), (0, 255, 0), 2)
        
        # Update global centers
        with centers_lock_62:
            centers_62 = current_centers.copy()
        
        # Store in global frame
        with frame_lock_62:
            frame_62 = frame
            
        # Maintain timing
        elapsed = time.time() - start_time
        sleep_time = max(0, delay - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
            
    cap.release()

def process_video_63():
    global frame_63, centers_63, running
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO("./best.pt", task="detect").to(device)
    
    # Open video
    cap = cv2.VideoCapture("/home/yy/63_2.mp4")
    if not cap.isOpened():
        print("Error: Could not open 63.mp4")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = 1.0 / fps if fps > 0 else 1.0/30.0
    
    while running:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        height, width = frame.shape[:2]
        
        # Clear previous centers
        with centers_lock_63:
            centers_63 = []
            
        # Detect class 1
        results = model(frame, save=False, classes=[1])
        
        # Draw detections and collect centers
        current_centers = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                if box.cls[0] == 1:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    
                    # Calculate center point
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    current_centers.append((center_x, center_y))
                    
                    # Draw bounding box (red)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Draw center point (yellow circle)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
                    
                    # Draw line from center point to LEFT border (cyan)
                    cv2.line(frame, (center_x, center_y), (0, center_y), (255, 255, 0), 2)
        
        # Update global centers
        with centers_lock_63:
            centers_63 = current_centers.copy()
        
        # Store in global frame
        with frame_lock_63:
            frame_63 = frame
            
        # Maintain timing
        elapsed = time.time() - start_time
        sleep_time = max(0, delay - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
            
    cap.release()

def calculate_y_alignment_offset(centers_left, centers_right):
    """
    Calculate the vertical offset needed to align centers.
    Returns the average Y difference between detected centers.
    If no centers found, returns 0.
    """
    if not centers_left or not centers_right:
        return 0
    
    # Average Y position of all detections
    avg_left_y = np.mean([center[1] for center in centers_left])
    avg_right_y = np.mean([center[1] for center in centers_right])
    y_diff = int(avg_right_y - avg_left_y)
    
    return y_diff

def display_combined():
    global frame_62, frame_63, centers_62, centers_63, running
    
    cv2.namedWindow('Dual Video Detection', cv2.WINDOW_NORMAL)
    
    while running:
        # Get frames with locks
        with frame_lock_62:
            local_frame_62 = frame_62.copy() if frame_62 is not None else None
            
        with frame_lock_63:
            local_frame_63 = frame_63.copy() if frame_63 is not None else None
            
        with centers_lock_62:
            local_centers_62 = centers_62.copy()
            
        with centers_lock_63:
            local_centers_63 = centers_63.copy()
            
        if local_frame_62 is None or local_frame_63 is None:
            time.sleep(0.01)
            continue
            
        # Get original dimensions
        h62, w62 = local_frame_62.shape[:2]
        h63, w63 = local_frame_63.shape[:2]
        
        # Resize to same height (use min height)
        target_height = min(h62, h63)
        
        # Initialize variables
        new_width_62 = w62
        new_width_63 = w63
        rolled_frame_63 = local_frame_63
        
        # Resize left frame if needed
        if h62 != target_height:
            scale = target_height / h62
            new_width_62 = int(w62 * scale)
            local_frame_62 = cv2.resize(local_frame_62, (new_width_62, target_height))
            # Adjust centers for resized frame
            scale_x_62 = new_width_62 / w62
            scale_y_62 = target_height / h62
            local_centers_62 = [(int(x * scale_x_62), int(y * scale_y_62)) for x, y in local_centers_62]
        else:
            # Use original dimensions
            new_width_62 = w62
        
        # Resize right frame if needed
        if h63 != target_height:
            scale = target_height / h63
            new_width_63 = int(w63 * scale)
            local_frame_63 = cv2.resize(local_frame_63, (new_width_63, target_height))
            # Adjust centers for resized frame
            scale_x_63 = new_width_63 / w63
            scale_y_63 = target_height / h63
            local_centers_63 = [(int(x * scale_x_63), int(y * scale_y_63)) for x, y in local_centers_63]
        else:
            # Use original dimensions
            new_width_63 = w63
        
        # Calculate Y-difference for alignment
        y_diff = calculate_y_alignment_offset(local_centers_62, local_centers_63)
        
        # Apply vertical roll to right frame based on Y-difference
        if y_diff != 0 and local_frame_63 is not None:
            # Roll the right frame vertically
            roll_amount = -y_diff  # Negative because np.roll rolls downward with positive values
            
            # Create a copy of the right frame
            rolled_frame_63 = np.roll(local_frame_63, roll_amount, axis=0)
            
            # Fill the empty area with black
            if roll_amount > 0:
                # Rolled downward, fill top with black
                rolled_frame_63[:roll_amount, :] = 0
            elif roll_amount < 0:
                # Rolled upward, fill bottom with black
                rolled_frame_63[roll_amount:, :] = 0
        else:
            rolled_frame_63 = local_frame_63
        
        # Draw alignment line on left frame (for visualization)
        if local_centers_62:
            # Draw horizontal line at average Y position of left centers
            avg_left_y = int(np.mean([center[1] for center in local_centers_62]))
            cv2.line(local_frame_62, (0, avg_left_y), (new_width_62, avg_left_y), 
                    (255, 255, 0), 1)  # Light blue horizontal reference line
        
        # Draw alignment line on right frame (for visualization)
        if local_centers_63 and rolled_frame_63 is not None:
            # Draw horizontal line at average Y position of right centers (after adjustment)
            # Adjust right centers for roll
            adjusted_right_centers = [(x, (y - y_diff) % target_height) for x, y in local_centers_63]
            avg_right_y = int(np.mean([center[1] for center in adjusted_right_centers]))
            cv2.line(rolled_frame_63, (0, avg_right_y), (new_width_63, avg_right_y), 
                    (255, 255, 0), 1)  # Light blue horizontal reference line
        
        # Combine frames
        if rolled_frame_63 is not None:
            combined = np.hstack((local_frame_62, rolled_frame_63))
        else:
            combined = local_frame_62
        
        # Add info and legend
        cv2.putText(combined, f"Video 62 (Class 0) - Detections: {len(local_centers_62)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, f"Video 63 (Class 1) - Detections: {len(local_centers_63)}", 
                   (new_width_62 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add alignment info
        if local_centers_62 and local_centers_63:
            cv2.putText(combined, f"Y-Diff: {y_diff}px (Roll: {-y_diff}px)", 
                       (new_width_62 // 2 - 100, target_height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Add detailed legend
        legend_y = target_height - 10
        cv2.putText(combined, "Left: Blue box, Red center, Green line", (10, legend_y - 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(combined, "Right: Red box, Yellow center, Cyan line", (10, legend_y - 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(combined, "Light blue: Alignment reference lines", (10, legend_y - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Display center coordinates if needed (optional)
        # if local_centers_62:
        #     cv2.putText(combined, f"Centers 62: {local_centers_62}", (10, legend_y - 30), 
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        # if local_centers_63:
        #     cv2.putText(combined, f"Centers 63: {local_centers_63}", (new_width_62 + 10, legend_y - 30), 
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Display
        cv2.imshow('Dual Video Detection - Y-Aligned', combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
            break
        elif key == ord('r'):
            # Reset alignment (show without roll)
            print("Reset alignment")
            
    cv2.destroyAllWindows()

def main():
    global running
    
    # Start processing threads
    thread_62 = threading.Thread(target=process_video_62, daemon=True)
    thread_63 = threading.Thread(target=process_video_63, daemon=True)
    thread_display = threading.Thread(target=display_combined, daemon=True)
    
    print("Starting threads...")
    print("Features:")
    print("1. Left video detects Class 0, Right video detects Class 1")
    print("2. Global center points stored for alignment")
    print("3. Automatic Y-axis alignment using np.roll")
    print("4. Light blue lines show alignment reference")
    print("5. Press 'q' to quit")
    print("6. Press 'r' to reset alignment")
    
    thread_62.start()
    thread_63.start()
    thread_display.start()
    
    try:
        # Keep main thread alive
        while running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        running = False
        
    # Wait for threads to finish
    thread_62.join(timeout=1)
    thread_63.join(timeout=1)
    thread_display.join(timeout=1)
    print("All threads stopped.")

if __name__ == "__main__":
    main()