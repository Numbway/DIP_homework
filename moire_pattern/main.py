import numpy as np
from PIL import Image
import utils
import glob
import os


def remove_moire(image_path, notch_centers, radius):

    # 读取图像为numpy数组
    with Image.open(image_path) as img:
        img = img.convert('L')  # 转换为灰度图像
        img_array = np.array(img)

    # 计算傅里叶变换
    dft_shift, magnitude_spectrum = utils.compute_fft(img_array)

    # 构建陷波滤波器
    mask = utils.create_notch_filter(img_array.shape, notch_centers, radius)

    # 应用滤波器
    filtered_image = utils.apply_filter(dft_shift, mask)

    # 可视化结果
    utils.visualize_results(img, magnitude_spectrum, mask, filtered_image)


# 使用示例
if __name__ == '__main__':

    image_path = r'D:\File\Postgraduate\First_Year\Digital_Image_Procesing\DIP_homework\moire_pattern\image\car-moire-pattern.tif'  # 替换为你的图像路径
    notch_centers = [(205, 55), (205, 115), (165, 55), (165, 115), (80, 55), (80, 115), (40, 55), (40, 115)]
    remove_moire(image_path, notch_centers, radius=8)
