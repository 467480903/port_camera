import cv2
import torch
from ultralytics import YOLO
import numpy as np
import threading
import time
from collections import deque
import logging

def setup_logging():
    logging.getLogger('ultralytics').setLevel(logging.ERROR)
setup_logging()

# Global frames and data
frame_left = None
frame_right = None
frame_lock_left = threading.Lock()
frame_lock_right = threading.Lock()

# Global center points (store multiple detections)
centers_left = []  # List of (x, y) tuples for video 62
centers_right = []  # List of (x, y) tuples for video 63
centers_lock_left = threading.Lock()
centers_lock_right = threading.Lock()

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("../yoloTrain2/runs/detect/train/weights/best_left_right_2.pt", task="detect").to(device)

# Control flags
running = True

def process_video_left():
    global frame_left, centers_left, running, model

    # Open video
    file = "/home/yy/10.10.95.219_001M_202601091536527BD2.mp4"
    cap = cv2.VideoCapture(file)
    if not cap.isOpened():
        print("Error: Could not open "+file)
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
            
        # Detect class 0
        results = model(frame, save=False, classes=[0])
        
        # Draw detections and collect centers
        if(len(results)>0):
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
        with centers_lock_left:
            centers_left = current_centers.copy()
        
        # Store in global frame
        with frame_lock_left:
            frame_left = frame
        
        # Maintain timing
        elapsed = time.time() - start_time
        sleep_time = max(0, delay - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
            
    cap.release()

def process_video_right():
    global frame_right, centers_right, running, model
    
    file = "/home/yy/10.10.95.219_002M_202601091536524732.mp4"
    cap = cv2.VideoCapture(file)
    if not cap.isOpened():
        print("Error: Could not open "+file)
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
            
        # Detect class 1
        results = model(frame, save=False, classes=[1])
        if(len(results)>0):
            current_centers = []

        # Draw detections and collect centers
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
        with centers_lock_right:
            centers_right = current_centers.copy()
        
        # Store in global frame
        with frame_lock_right:
            frame_right = frame
            
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

def get_average_x_position(centers):
    """
    Get average x position from a list of centers.
    Returns average x position or None if no centers.
    """
    if not centers:
        return None
    return int(np.mean([center[0] for center in centers]))

def display_combined():
    global frame_left, frame_right, centers_left, centers_right, running
    
    cv2.namedWindow('Dual Video Detection', cv2.WINDOW_NORMAL)
    
    # Display mode: 0 = original, 1 = combined view
    display_mode = 1
    
    while running:
        # Get frames with locks
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
            
        # Get original dimensions
        h_left, w_left = local_frame_left.shape[:2]
        h_right, w_right = local_frame_right.shape[:2]
        
        # Resize to same height (use min height)
        target_height = min(h_left, h_right)
        
        # Initialize variables
        new_width_left = w_left
        new_width_right = w_right
        rolled_frame_right = local_frame_right
        
        # Resize left frame if needed
        if h_left != target_height:
            scale = target_height / h_left
            new_width_left = int(w_left * scale)
            local_frame_left = cv2.resize(local_frame_left, (new_width_left, target_height))
            # Adjust centers for resized frame
            scale_x_left = new_width_left / w_left
            scale_y_left = target_height / h_left
            local_centers_left = [(int(x * scale_x_left), int(y * scale_y_left)) for x, y in local_centers_left]
        else:
            # Use original dimensions
            new_width_left = w_left
        
        # Resize right frame if needed
        if h_right != target_height:
            scale = target_height / h_right
            new_width_right = int(w_right * scale)
            local_frame_right = cv2.resize(local_frame_right, (new_width_right, target_height))
            # Adjust centers for resized frame
            scale_x_right = new_width_right / w_right
            scale_y_right = target_height / h_right
            local_centers_right = [(int(x * scale_x_right), int(y * scale_y_right)) for x, y in local_centers_right]
        else:
            # Use original dimensions
            new_width_right = w_right
        
        # Calculate Y-difference for alignment
        y_diff = calculate_y_alignment_offset(local_centers_left, local_centers_right)
        
        # Apply vertical roll to right frame based on Y-difference
        if y_diff != 0 and local_frame_right is not None:
            # Roll the right frame vertically
            roll_amount = -y_diff  # Negative because np.roll rolls downward with positive values
            
            # Create a copy of the right frame
            rolled_frame_right = np.roll(local_frame_right, roll_amount, axis=0)
            
            # Fill the empty area with black
            if roll_amount > 0:
                # Rolled downward, fill top with black
                rolled_frame_right[:roll_amount, :] = 0
            elif roll_amount < 0:
                # Rolled upward, fill bottom with black
                rolled_frame_right[roll_amount:, :] = 0
        else:
            rolled_frame_right = local_frame_right
        
        # Get average x positions for cutting
        avg_x_left = get_average_x_position(local_centers_left)
        avg_x_right = get_average_x_position(local_centers_right)
        print( [y_diff, avg_x_left, avg_x_right] )
        # Prepare display based on mode
        if display_mode == 1 and avg_x_left is not None and avg_x_right is not None:
            # MODE 1: Combined view - cut and merge frames
            
            # Cut left frame: from left to center (inclusive of center)
            # We'll take the portion from the center to the right edge for left video
            left_cut = local_frame_left[:, 0:avg_x_left ]
            
            # Cut right frame: from center to right (inclusive of center)
            # We'll take the portion from the left edge to the center for right video
            right_cut = rolled_frame_right[:, avg_x_right: new_width_right]
            
            # Combine the two cuts
            if left_cut.shape[1] > 0 and right_cut.shape[1] > 0:
                combined = np.hstack((left_cut, right_cut))
            elif left_cut.shape[1] > 0:
                combined = left_cut
            elif right_cut.shape[1] > 0:
                combined = right_cut
            else:
                combined = np.hstack((local_frame_left, rolled_frame_right))
            
            # Add vertical line at the junction
            if left_cut.shape[1] > 0 and right_cut.shape[1] > 0:
                # Draw a vertical line at the junction point
                junction_x = left_cut.shape[1]
                cv2.line(combined, (junction_x, 0), (junction_x, target_height), (0, 255, 255), 2)
                
                # Add text at junction
                cv2.putText(combined, "JUNCTION", (junction_x - 60, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Add info text for combined mode
            cv2.putText(combined, "COMBINED VIEW - Left portion + Right portion", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        else:
            # MODE 0: Original view - both full videos side by side
            if rolled_frame_right is not None:
                combined = np.hstack((local_frame_left, rolled_frame_right))
            else:
                combined = local_frame_left
            
            # Add info text for original mode
            cv2.putText(combined, "ORIGINAL VIEW - Both videos side by side", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add detection info
        cv2.putText(combined, f"Video 62 (Class 0) - Detections: {len(local_centers_left)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(combined, f"Video 63 (Class 1) - Detections: {len(local_centers_right)}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add X position info if available
        if avg_x_left is not None:
            cv2.putText(combined, f"Left X: {avg_x_left}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 200), 1)
        if avg_x_right is not None:
            cv2.putText(combined, f"Right X: {avg_x_right}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
        
        # Add alignment info
        if local_centers_left and local_centers_right:
            cv2.putText(combined, f"Y-Diff: {y_diff}px (Roll: {-y_diff}px)", 
                       (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Add cut line visualization on original frames (in original mode)
        if display_mode == 0:
            # Draw cut lines on left frame
            if avg_x_left is not None:
                cv2.line(combined, (avg_x_left, 0), (avg_x_left, target_height), (255, 255, 0), 2)
                cv2.putText(combined, "Left cut", (avg_x_left + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Draw cut lines on right frame
            if avg_x_right is not None:
                right_frame_start = new_width_left
                right_cut_x = right_frame_start + avg_x_right
                cv2.line(combined, (right_cut_x, 0), (right_cut_x, target_height), (255, 200, 0), 2)
                cv2.putText(combined, "Right cut", (right_cut_x + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        
        # Add mode info
        mode_text = "Mode: COMBINED (cut & merge)" if display_mode == 1 else "Mode: ORIGINAL (side by side)"
        cv2.putText(combined, mode_text, (10, target_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display controls info
        cv2.putText(combined, "Press 'c' to toggle mode | 'q' to quit", 
                   (combined.shape[1] - 300, target_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display
        window_title = 'Dual Video - Combined View' if display_mode == 1 else 'Dual Video - Original View'
        cv2.imshow(window_title, combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
            break
        elif key == ord('c'):
            # Toggle between display modes
            display_mode = 1 - display_mode  # Toggle between 0 and 1
            print(f"Switched to {'COMBINED' if display_mode == 1 else 'ORIGINAL'} mode")
        elif key == ord('r'):
            # Reset alignment (show without roll)
            print("Reset alignment")
            
    cv2.destroyAllWindows()

def main():
    global running
    
    # Start processing threads
    thread_left = threading.Thread(target=process_video_left, daemon=True)
    thread_right = threading.Thread(target=process_video_right, daemon=True)
    thread_display = threading.Thread(target=display_combined, daemon=True)
    
    thread_left.start()
    thread_right.start()
    thread_display.start()
    
    print("\n=== Dual Video Processing ===")
    print("Controls:")
    print("  'c' - Toggle between Original and Combined view")
    print("  'q' - Quit the application")
    print("  'r' - Reset alignment\n")
    
    try:
        # Keep main thread alive
        while running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        running = False
        
    # Wait for threads to finish
    thread_left.join(timeout=1)
    thread_right.join(timeout=1)
    thread_display.join(timeout=1)
    print("All threads stopped.")

if __name__ == "__main__":
    main()