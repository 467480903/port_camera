import cv2
import os

def extract_images_from_video(video_path, output_folder):
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
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Break the loop if we've reached the end of the video
        if not ret:
            break
        
        # Save the frame as an image file every second
        if frame_count % int(fps) == 0:
            frame_path = os.path.join(output_folder, f'frame_{image_count:04d}.jpg')
            cv2.imwrite(frame_path, frame)
            image_count += 1
            print(f'Saved {frame_path}')
        
        # Increment frame count
        frame_count += 1
    
    # Release the video capture object
    cap.release()

# Define video path and output folder
video_path = '/home/yy/62.mp4'
output_folder = 'label/62/'

# Extract images
extract_images_from_video(video_path, output_folder)
