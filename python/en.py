import cv2
import numpy as np
import matplotlib.pyplot as plt

def enhance_contrast_linear(img, alpha=1.5, beta=0):
    """
    alpha: 对比度系数 (>1 增强对比度)
    beta: 亮度调整
    """
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def gamma_correction(img, gamma=1.5):
    """
    gamma < 1: 变亮
    gamma > 1: 变暗，增强对比度
    """
    # 归一化
    img_normalized = img / 255.0
    # Gamma校正
    corrected = np.power(img_normalized, gamma)
    # 恢复范围
    return np.uint8(corrected * 255)

# 使用示例
img = cv2.imread('image.jpg', 0)  # 读取为灰度图
enhanced = enhance_contrast_linear(img, alpha=2.0, beta=10)

# 读取图像
img = cv2.imread('blurred.jpg', 0)

# 方法1：线性对比度增强
linear_enhanced = enhance_contrast_linear(img, alpha=1.8, beta=5)

# 方法2：Gamma校正
gamma_enhanced = gamma_correction(img, gamma=1.6)

# 显示结果
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')

plt.subplot(1, 3, 2)
plt.imshow(linear_enhanced, cmap='gray')
plt.title('Linear Enhanced')

plt.subplot(1, 3, 3)
plt.imshow(gamma_enhanced, cmap='gray')
plt.title('Gamma Enhanced')
plt.show()