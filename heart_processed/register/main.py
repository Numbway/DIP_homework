import os
import nibabel as nib
import numpy as np
from utils import rigid_register, apply_transform, nibabel_to_sitk, sitk_to_nibabel, resize_image, nonrigid_register

# 配置参数
TEMPLATE_IDS = list(range(1001, 1010))
TARGET_ID = 1010
CATEGORIES = ['Ao', 'Heart', 'LA', 'LV', 'LV_Myo', 'PA', 'RA', 'RV']


def main():
    # 加载目标图像
    target_nib = nib.load(rf'D:\File\Postgraduate\First_Year\Digital_Image_Procesing\DIP_homework\heart_processed\ct_atlas\ct_train_{TARGET_ID}_imageROI.nii')
    target_sitk = nibabel_to_sitk(target_nib)

    # 初始化存储配准结果的字典
    results = {category: [] for category in CATEGORIES}

    # 获取目标图像的形状
    target_shape = target_nib.shape

    # 遍历每个模板进行配准和变换
    for tid in TEMPLATE_IDS:
        print(f"Processing template {tid}...")

        # 加载模板图像
        template_nib = nib.load(rf'D:\File\Postgraduate\First_Year\Digital_Image_Procesing\DIP_homework\heart_processed\ct_atlas\ct_train_{tid}_imageROI.nii')
        template_sitk = nibabel_to_sitk(template_nib)

        # 执行配准
        transform_params = nonrigid_register(target_sitk, template_sitk)

        # 对每个解剖结构进行变换
        for category in CATEGORIES:
            label_nib = nib.load(rf'D:\File\Postgraduate\First_Year\Digital_Image_Procesing\DIP_homework\heart_processed\ct_atlas\ct_train_{tid}_{category}.nii')
            label_sitk = nibabel_to_sitk(label_nib)

            # 应用变换（使用最近邻插值）
            transformed_sitk = apply_transform(label_sitk, transform_params, is_label=True)
            transformed_nib = sitk_to_nibabel(transformed_sitk)
            transformed_data = transformed_nib.get_fdata().astype(np.uint8)

            # 确保所有图像具有相同的形状
            if transformed_data.shape != target_shape:
                transformed_data = resize_image(transformed_data, target_shape)

            results[category].append(transformed_data)

    # 融合结果（多数投票）
    final_seg = {}
    for category in CATEGORIES:
        stacked = np.stack(results[category], axis=0)
        fused = (np.mean(stacked, axis=0) > 0.5).astype(np.uint8)  # 阈值融合
        final_seg[category] = fused

    # 保存结果
    for category in CATEGORIES:
        seg_nii = nib.Nifti1Image(final_seg[category], target_nib.affine, header=target_nib.header)
        seg_nii.to_filename(rf'D:\File\Postgraduate\First_Year\Digital_Image_Procesing\DIP_homework\heart_processed\processed\ct_train_{TARGET_ID}_{category}_seg.nii')
        print(f"Saved segmentation for {category}")


if __name__ == "__main__":
    main()