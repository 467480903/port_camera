import cv2
import numpy as np

# 1. 加载图像
image1 = cv2.imread("/home/yy/port_camera/python/p1.png")
image2 = cv2.imread("/home/yy/port_camera/python/p2.png")

# 2. 转换为灰度图
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# 3. 特征点检测
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# 4. 特征点匹配
matcher = cv2.BFMatcher()
matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

# 5. 筛选匹配点
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 6. 计算单应性矩阵
if len(good_matches) > 10:
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
else:
    print("Not enough matches found.")
    exit()

# 7. 图像变换
height1, width1 = image1.shape[:2]
height2, width2 = image2.shape[:2]
warped_image = cv2.warpPerspective(image1, H, (width1 + width2, height1))

# 8. 图像拼接
warped_image[0:height2, 0:width2] = image2

# 9. 显示结果
cv2.imshow("Stitched Image", warped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()