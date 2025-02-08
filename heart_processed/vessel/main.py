# main.py
import os
from utils import load_nifti, save_nifti, apply_morphological_filter, display_image, binarize_image


def main():
    # 设置输入输出路径
    input_file = r"/heart_processed/ct_atlas/ct_train_1001_imageROI.nii"
    output_file = r"/heart_processed/processed/ct_train_1001_imageROI_processed.nii"

    # 加载数据
    data, affine = load_nifti(input_file)

    # 进行二值化处理，假设血管区域的灰度值较高
    binary_data = binarize_image(data, threshold=0.3)

    # 应用形态学开运算进行肺部血管滤除
    cleaned_data = apply_morphological_filter(binary_data, selem_size=3)

    # 保存处理后的数据
    save_nifti(cleaned_data, affine, output_file)
    print(f"Processed image saved to {output_file}")


if __name__ == "__main__":
    main()
