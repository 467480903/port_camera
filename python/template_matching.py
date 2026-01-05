import cv2
import numpy as np
import time


# Read the images
main_image = cv2.imread('c.jpg')
template = cv2.imread('template.png')

main_image =cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
template =cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

cv2.imwrite('main_image_gray.jpg', main_image)
cv2.imwrite('template_gray.jpg', template)

# Ensure the images are loaded properly
if main_image is None or template is None:
    print("Error: Images not loaded. Please check the file paths.")
    exit()

# Get the dimensions of the template
template_height, template_width = template.shape[:2]

# Perform template matching with different methods
methods = ['cv2.TM_CCOEFF_NORMED']

for method in methods:
    # Get the evaluation method
    eval_method = eval(method)


    # 当前时间戳（浮点数，单位秒）
    timestamp = time.time()
    print(f"时间戳: {timestamp}")

    # 格式化的本地时间
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"本地时间: {local_time}")

    # Perform template matching
    result = cv2.matchTemplate(main_image, template, eval_method)

    # Find the location of the best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Set a confidence threshold
    confidence_threshold = 0.8

    if eval_method in [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED]:
        if max_val >= confidence_threshold:
            print(f"Method: {method}, Best match location: {max_loc}, Confidence: {max_val}")
            # Draw a rectangle around the best match
            # top_left = max_loc
            # bottom_right = (top_left[0] + template_width, top_left[1] + template_height)
            # cv2.rectangle(main_image, top_left, bottom_right, (0, 255, 0), 2)

            timestamp = time.time()
            print(f"时间戳: {timestamp}")

            # 格式化的本地时间
            local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"本地时间: {local_time}")

# Display the main image with the match
# cv2.imshow('Detected Match', main_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
