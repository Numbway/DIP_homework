import numpy as np
from PIL import Image

def shear_image_auto_adjust(input_path, output_path, matrix):
    # 打开TIF图像
    image = Image.open(input_path)

    # 提取图像的宽高
    width, height = image.size

    # 定义图像的四个角点
    corners = np.array([
        [0, 0, 1],        # 左上角
        [width, 0, 1],    # 右上角
        [0, height, 1],   # 左下角
        [width, height, 1] # 右下角
    ])

    # 仿射变换矩阵
    affine_matrix = np.array(matrix).reshape((3, 3))

    # 变换角点坐标
    new_corners = np.dot(corners, affine_matrix.T)

    # 获取新的坐标范围（最小和最大值）
    min_x = np.min(new_corners[:, 0])
    max_x = np.max(new_corners[:, 0])
    min_y = np.min(new_corners[:, 1])
    max_y = np.max(new_corners[:, 1])

    # 计算新的画幅尺寸
    new_width = int(np.ceil(max_x - min_x))
    new_height = int(np.ceil(max_y - min_y))

    # 平移量，使变换后的图像在新的画幅内不被裁剪
    offset_x = 0
    offset_y = 0

    # 更新仿射变换矩阵，将图像平移到新画幅中
    affine_matrix[0, 2] += offset_x
    affine_matrix[1, 2] += offset_y

    # PIL使用的是2x3的仿射变换矩阵，取前两行的前3列
    pil_affine_matrix = affine_matrix[:2, :].flatten()

    # 对图像进行仿射变换，使用自动调整后的画幅尺寸
    transformed_image = image.transform(
        (new_width, new_height),  # 使用计算出的新画幅尺寸
        Image.AFFINE,
        data=pil_affine_matrix[:6],  # 仿射矩阵只需要前6个元素
        resample=Image.BILINEAR
    )

    # 保存shear变换后的图像
    transformed_image.save(output_path)
    print(f"图像已保存至: {output_path}, 新尺寸: {new_width}x{new_height}")

# 定义输入和输出路径
input_path = 'image/2.42_512try1.tif'
output_path = 'image/2.42_512try2.tif'

# 定义shear矩阵
shear_matrix = [1, 0, 0,   # X方向扭曲
                -0.1, 1, 0,   # Y方向旋转
                0, 0, 1]      # 这是3x3矩阵

# 调用函数进行shear操作，并自动调整画幅尺寸
shear_image_auto_adjust(input_path, output_path, shear_matrix)
