import nibabel as nib
import SimpleITK as sitk
import numpy as np


def nibabel_to_sitk(nib_img):
    """将nibabel图像转换为SimpleITK图像，处理坐标系转换（RAS到LPS）"""
    data = nib_img.get_fdata()
    affine = nib_img.affine

    # 转换为LPS坐标系
    lps_affine = np.copy(affine)
    lps_affine[0:2, :] *= -1  # 翻转X和Y轴方向
    origin = lps_affine[:3, 3]
    spacing = list(map(float, nib_img.header.get_zooms()[:3]))
    sitk_img = sitk.GetImageFromArray(nib_img.get_fdata())
    direction = lps_affine[:3, :3].T.ravel()  # 方向矩阵转置

    sitk_img = sitk.GetImageFromArray(data.transpose())
    sitk_img.SetOrigin(origin)
    sitk_img.SetSpacing(spacing)
    sitk_img.SetDirection(direction)
    return sitk_img


def sitk_to_nibabel(sitk_img):
    """将SimpleITK图像转换回nibabel格式（LPS到RAS）"""
    data = sitk.GetArrayFromImage(sitk_img).transpose()
    origin = np.array(sitk_img.GetOrigin())
    spacing = np.array(sitk_img.GetSpacing())
    direction = np.array(sitk_img.GetDirection()).reshape(3, 3)

    # 转换回RAS坐标系
    ras_affine = np.eye(4)
    ras_affine[:3, :3] = direction.T * spacing
    ras_affine[:3, 3] = origin
    ras_affine[0:2, :3] *= -1  # 翻转X和Y轴方向

    return nib.Nifti1Image(data, ras_affine)


def rigid_register(fixed_sitk, moving_sitk):
    """使用SimpleITK自带的配准工具进行刚性配准"""
    # 初始化配准器
    registration_method = sitk.ImageRegistrationMethod()

    # 设置相似性度量（互信息）
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

    # 设置优化器（梯度下降）
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # 设置变换类型（刚性变换）
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_sitk, moving_sitk, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # 设置插值器
    registration_method.SetInterpolator(sitk.sitkLinear)

    # 执行配准
    final_transform = registration_method.Execute(fixed_sitk, moving_sitk)

    return final_transform


def nonrigid_register(fixed_sitk, moving_sitk):
    """使用SimpleITK自带的配准工具进行非刚性配准"""
    # 初始化配准器
    registration_method = sitk.ImageRegistrationMethod()

    # 设置相似性度量（互信息）
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

    # 设置优化器（梯度下降）
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # 设置变换类型（非刚性变换，使用B样条变换）
    transform_domain_mesh_size = [3, 3, 3]  # 根据需要调整网格大小
    initial_transform = sitk.BSplineTransformInitializer(fixed_sitk, transform_domain_mesh_size)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # 设置插值器
    registration_method.SetInterpolator(sitk.sitkLinear)

    # 执行配准
    final_transform = registration_method.Execute(fixed_sitk, moving_sitk)

    return final_transform


def apply_transform(image_sitk, transform, is_label=False):
    """应用配准变换到图像"""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image_sitk)
    resampler.SetTransform(transform)

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # 标签使用最近邻插值
    else:
        resampler.SetInterpolator(sitk.sitkLinear)  # 图像使用线性插值

    return resampler.Execute(image_sitk)

def resize_image(image, target_shape):
    """
    调整图像大小以匹配目标形状
    """
    import scipy.ndimage
    zoom_factors = [t / s for t, s in zip(target_shape, image.shape)]
    resized_image = scipy.ndimage.zoom(image, zoom_factors, order=0)  # 使用最近邻插值
    return resized_image