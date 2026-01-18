import cv2
import os

def extract_images_from_video(video_path, output_folder, interval_seconds=3):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    image_count = 0
    
    # Calculate frames per interval
    frames_per_interval = int(fps * interval_seconds)
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Break the loop if we've reached the end of the video
        if not ret:
            break
        
        # Save the frame as an image file every interval_seconds
        if frame_count % frames_per_interval == 0:
            frame_path = os.path.join(output_folder, f'all_{image_count:04d}.jpg')
            cv2.imwrite(frame_path, frame)
            image_count += 1
            print(f'Saved {frame_path}')
        
        # Increment frame count
        frame_count += 1
    
    # Release the video capture object
    cap.release()
    print(f"Extraction complete. Total frames: {frame_count}, Images saved: {image_count}")

# Define video path and output folder
# video_path = '/home/yy/63.mp4'
# output_folder = 'label/63/'

# video_path = '/home/yy/62_2.mp4'
# output_folder = 'label/62/'

video_path = '/home/yy/snap/obs-studio/1316/2026-01-15 20-08-48.mp4'
output_folder = 'all/'

# video_path = '/home/yy/10.10.95.219_002M_202601091536524732.mp4'
# output_folder = 'right/'

# Extract images every 3 seconds
extract_images_from_video(video_path, output_folder, interval_seconds=1)                                                            