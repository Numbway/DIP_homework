# utils.py
import nibabel as nib
import numpy as np
from skimage.morphology import opening, closing, disk, ball
import matplotlib.pyplot as plt
import SimpleITK as sitk


def load_nifti(file_path):
    """
    加载NIfTI文件
    :param file_path: 输入的nii文件路径
    :return: 图像数据和图像的仿射矩阵
    """
    img = nib.load(file_path)
    data = img.get_fdata()
    return data, img.affine


def save_nifti(data, affine, output_path):
    """
    保存处理后的数据为NIfTI文件
    :param data: 处理后的图像数据
    :param affine: 图像的仿射矩阵
    :param output_path: 输出的文件路径
    """
    processed_img = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(processed_img, output_path)


def apply_morphological_filter(binary_data, selem_size=3):
    """
    使用形态学开运算（腐蚀后膨胀）去除小的结构
    :param binary_data: 输入的二值化图像数据
    :param selem_size: 形态学操作的结构元素大小
    :return: 处理后的二值图像
    """
    cleaned_data = opening(binary_data, footprint=ball(selem_size))
    return cleaned_data


def display_image(data, slice_index=50):
    """
    可视化图像的一个切片
    :param data: 图像数据
    :param slice_index: 要显示的切片索引
    """
    plt.imshow(data[:, :, slice_index], cmap='gray')
    plt.axis('off')
    plt.show()


def binarize_image(data, threshold=0.3):
    """
    根据灰度值进行二值化处理
    :param data: 原始图像数据
    :param threshold: 二值化的阈值
    :return: 二值化图像
    """
    return data > threshold
