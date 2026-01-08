import cv2
import torch
from ultralytics import YOLO

def main(video_path):
    # Load YOLO model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO("./best.pt", task="detect").to(device)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video Info: {width}x{height} @ {fps:.2f} FPS")

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video ended or could not read frame.")
            break

        # Perform inference
        results = model(frame, save=False, classes=[0])

        # Extract detections
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                r = box.xyxy[0].astype(int)
                cls = box.cls[0].astype(int)
                conf = box.conf[0]

                if cls == 41:  # Assuming class 0 is 'person'
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
                    # Optional: Display head crop window
                    # cv2.imshow('Head Detection', head_crop)

        # Display the frame with detections
        cv2.imshow('Video Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace with your MP4 video file path
    video_path = "/home/yy/62.mp4"  # Change this to your video file path
    main(video_path)