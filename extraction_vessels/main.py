import numpy as np
from PIL import Image
import utils
import glob
import os


def extract(input_image_path):

    # 遍历指定文件夹中所有的 .png 文件
    image_paths = glob.glob(os.path.join(input_image_path, "*.png"))

    # 遍历每个图像路径进行处理
    for image_path in image_paths:
        with Image.open(image_path) as img:
            img = img.convert('L')  # 转换为灰度图像
            img_array = np.array(img)

        # 直方图均衡化
        processed_img = utils.equalized_images(img_array)

        # 高斯滤波器
        # processed_img = utils.guassian_smooth(img_array, 2)

        # 拉普拉斯滤波器
        # processed_img = utils.laplacian(processed_img)

        # 使用Sobel算子进行边缘提取
        # processed_img = utils.sobel_edge_detection(img_array)

        # Frangi滤波器
        processed_img = utils.enhance_tube_structures(processed_img)

        # 展示处理效果
        utils.show_image(img, processed_img)


if __name__ == '__main__':
    input_image_path = r'D:\File\Postgraduate\First_Year\Digital_Image_Procesing\DIP_homework\extraction_vessels\image\IMG'
    extract(input_image_path)
