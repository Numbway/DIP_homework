import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_fft(image):

    """计算傅里叶变换并返回频谱和转换后的数据"""
    dft = cv2.dft(image.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    return dft_shift, magnitude_spectrum


def create_notch_filter(shape, notch_centers, radius):

    """创建陷波滤波器掩模"""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2  # 中心点
    mask = np.ones((rows, cols, 2), np.uint8)

    # 在掩模上画出陷波滤波器
    for center in notch_centers:
        x, y = center
        cv2.circle(mask, (y, x), radius, (0, 0), -1)

    return mask


def apply_filter(dft_shift, mask):

    """应用陷波滤波器"""
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return img_back


def visualize_results(original, spectrum, mask, filtered):

    """可视化结果"""
    spectrum_normalized = (spectrum / spectrum.max() * 255).astype(np.uint8)
    mask_overlay = (mask[:, :, 0] * 255).astype(np.uint8)
    overlay = cv2.addWeighted(spectrum_normalized, 0.7, mask_overlay, 0.3, 0)

    plt.figure(figsize=(12, 8))
    plt.subplot(221), plt.imshow(original, cmap='gray'), plt.title('Original Image')
    plt.subplot(222), plt.imshow(spectrum, cmap='gray'), plt.title('Magnitude Spectrum')
    plt.subplot(223), plt.imshow(overlay, cmap='gray'), plt.title('Spectrum + Mask Overlay')
    plt.subplot(224), plt.imshow(filtered, cmap='gray'), plt.title('Filtered Image')
    plt.tight_layout()
    plt.show()
