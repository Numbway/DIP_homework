import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import cv2


def normalize_image(image):
    """将图像归一化到0-255灰度范围"""
    image = image.astype(np.float32)
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = 255 * (image - min_val) / (max_val - min_val)
    return normalized_image.astype(np.uint8)


def equalized_images(image):

    image = normalize_image(image)

    # 1. 计算灰度直方图
    histogram = np.zeros(256, dtype=int)
    for pixel in image.flatten():
        histogram[pixel] += 1

    # 2. 计算累积分布函数（CDF）
    cdf = np.cumsum(histogram)
    cdf_normalized = cdf / cdf[-1]  # 归一化，将CDF的最后一个值变成1

    # 3. 计算均衡化后的像素值映射表
    equalization_map = np.floor(255 * cdf_normalized).astype(np.uint8)

    # 4. 应用均衡化映射到图像
    equalized_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            equalized_image[i, j] = equalization_map[image[i, j]]

    return equalized_image


def gaussian_kernel(size, sigma):
    """生成高斯核矩阵"""
    kernel = np.zeros((size, size), dtype=np.float32)
    offset = size // 2
    for x in range(-offset, offset + 1):
        for y in range(-offset, offset + 1):
            kernel[x + offset, y + offset] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(
                -(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return kernel / np.sum(kernel)


def apply_convolution(image, kernel):
    """手动实现卷积操作"""
    k_size = kernel.shape[0]
    offset = k_size // 2
    padded_image = np.pad(image, pad_width=offset, mode='reflect')
    output = np.zeros_like(image, dtype=np.float32)

    for i in range(offset, padded_image.shape[0] - offset):
        for j in range(offset, padded_image.shape[1] - offset):
            region = padded_image[i - offset:i + offset + 1, j - offset:j + offset + 1]
            output[i - offset, j - offset] = np.sum(region * kernel)

    return output


def guassian_smooth(image, sigma):

    # 确定高斯核的大小
    kernel_size = int(6 * sigma + 1) if int(6 * sigma + 1) % 2 != 0 else int(6 * sigma + 1) + 1

    # 生成高斯核
    gaussian_k = gaussian_kernel(kernel_size, sigma)

    # 对图像进行高斯平滑
    smoothed_image = apply_convolution(image, gaussian_k)

    return smoothed_image


def laplacian(image):

    # 拉普拉斯算子定义
    laplacian_k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    # 图像应用拉普拉斯算子
    log_image = apply_convolution(image, laplacian_k)

    return log_image


def enhance_edges(gradient_magnitude, alpha=5.0, threshold=100):
    """强化边缘显示：增强梯度幅值和应用阈值化"""
    # 增强梯度幅值
    enhanced_edges = gradient_magnitude * alpha

    # 应用阈值化
    enhanced_edges[enhanced_edges > 255] = 255  # 防止像素值超过255
    enhanced_edges[enhanced_edges < threshold] = 0  # 将低于阈值的部分设置为0（黑色）

    return enhanced_edges.astype(np.uint8)


def sobel_edge_detection(image):

    # 定义Sobel算子
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    # 水平方向卷积
    sobel_x = apply_convolution(image, Gx)
    # 垂直方向卷积
    sobel_y = apply_convolution(image, Gy)

    # 计算梯度幅值
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # 归一化到0-255
    gradient_magnitude = normalize_image(gradient_magnitude)

    # 强化边缘
    gradient_magnitude = enhance_edges(gradient_magnitude)

    return gradient_magnitude


def compute_gradients(image):
    # 使用Sobel算子计算梯度
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return Ix, Iy


def compute_structure_tensor(Ix, Iy, sigma=2.0):
    # 使用高斯平滑来计算结构张量
    Ixx = guassian_smooth(Ix ** 2, sigma)
    Ixy = guassian_smooth(Ix * Iy, sigma)
    Iyy = guassian_smooth(Iy ** 2, sigma)
    return Ixx, Ixy, Iyy


def compute_eigenvalues(Ixx, Ixy, Iyy):
    # 计算结构张量的特征值
    trace = Ixx + Iyy
    det = Ixx * Iyy - Ixy ** 2
    lambda1 = 0.5 * (trace + np.sqrt(trace ** 2 - 4 * det))
    lambda2 = 0.5 * (trace - np.sqrt(trace ** 2 - 4 * det))
    return lambda1, lambda2


def enhance_tube_structures(image, sigma=1.0):
    # 计算图像梯度
    Ix, Iy = compute_gradients(image)

    # 计算结构张量
    Ixx, Ixy, Iyy = compute_structure_tensor(Ix, Iy, sigma)

    # 计算特征值
    lambda1, lambda2 = compute_eigenvalues(Ixx, Ixy, Iyy)

    # 通过选择合适的特征值来增强管状结构
    enhanced_image = np.abs(lambda1 - lambda2)  # 强调管状结构的方向性

    return enhanced_image


def show_image(raw_image, processed_image):

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.imshow(raw_image, cmap='gray'), plt.title('Original Image')
    plt.subplot(1, 2, 2), plt.imshow(processed_image, cmap='gray'), plt.title('Equalized Image')
    plt.show()
