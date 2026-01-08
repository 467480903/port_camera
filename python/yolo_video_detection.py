import cv2
import torch
from ultralytics import YOLO
import os

def main(video_path, output_dir="output_frames"):
    # Load YOLO model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO("./best.pt", task="detect").to(device)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        if 2>1:
            # Perform inference
            results = model(frame, save=False, classes=[0])

            # Extract detections
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    r = box.xyxy[0].astype(int)
                    cls = box.cls[0].astype(int)
                    conf = box.conf[0]

                    if cls == 0:  
                        x1, y1, x2, y2 = r

                        # Estimate head region (simple heuristic: top 1/4 of person bounding box)
                        head_y1 = y1
                        head_y2 = y1 + (y2 - y1) // 4

                        head_crop = frame[head_y1:head_y2, x1:x2]

                        if head_crop.size == 0:
                            continue

                        # Perform inference on head crop
                        head_results = model(head_crop, save=True)

                        # Extract head detections
                        for head_result in head_results:
                            head_boxes = head_result.boxes.cpu().numpy()
                            for head_box in head_boxes:
                                hr = head_box.xyxy[0].astype(int)
                                hcls = head_box.cls[0].astype(int)
                                hconf = head_box.conf[0]

                                # Draw detection rectangle for head
                                cv2.rectangle(head_crop, (hr[0], hr[1]), (hr[2], hr[3]), (0, 255, 0), 2)

                        # Draw person detection rectangle with head crop
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        # cv2.imshow('Head Detection', head_crop)

        # Display the frame with detections
        cv2.imshow('RTSP Stream', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "/home/yy/62.mp4"
    main(video_path)