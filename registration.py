import numpy as np
from PIL import Image
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

def bilinear_interpolation(v, w, coefficients):
    """
    使用双线性插值计算目标图像坐标
    v, w: 待配准图像的坐标
    coefficients: 双线性模型的系数 [c1, c2, c3, c4, c5, c6, c7, c8]
    返回目标图像坐标 (x, y)
    """
    c1, c2, c3, c4, c5, c6, c7, c8 = coefficients
    x = c1 * v + c2 * w + c3 * v * w + c4
    y = c5 * v + c6 * w + c7 * v * w + c8
    return x, y

def nearest_neighbor_interpolation(image_array, output_image_array):
    """
    使用最邻近插值填补目标图像中的空点
    image_array: 原始图像的像素数组
    output_image_array: 在配准映射后包含空点的目标图像数组
    返回填补后的目标图像数组
    """
    # 找出 output_image_array 中值为 0（即空点）的像素坐标
    empty_points = np.argwhere(output_image_array == 0)

    # 找出 output_image_array 中非空点的像素坐标
    filled_points = np.argwhere(output_image_array > 0)
    filled_values = output_image_array[output_image_array > 0]

    # 使用 KD-Tree 加速最近邻查找
    tree = cKDTree(filled_points)
    nearest_neighbors = tree.query(empty_points, k=1)[1]

    # 将最近邻点的值赋给空点
    for i, empty_point in enumerate(empty_points):
        output_image_array[empty_point[0], empty_point[1]] = filled_values[nearest_neighbors[i]]

    return output_image_array

def linear_interpolation(image_array, output_image_array):
    """
    使用线性插值填补目标图像中的空点
    image_array: 原始图像的像素数组
    output_image_array: 在配准映射后包含空点的目标图像数组
    返回填补后的目标图像数组
    """
    # 找出 output_image_array 中值为 0（即空点）的像素坐标
    empty_points = np.argwhere(output_image_array == 0)

    # 找出 output_image_array 中非空点的像素坐标
    filled_points = np.argwhere(output_image_array > 0)
    filled_values = output_image_array[output_image_array > 0]

    # 使用 griddata 进行线性插值
    interpolated_values = griddata(filled_points, filled_values, empty_points, method='linear')

    # 将插值结果赋给空点
    for i, empty_point in enumerate(empty_points):
        if not np.isnan(interpolated_values[i]):  # 确保插值结果有效
            output_image_array[empty_point[0], empty_point[1]] = interpolated_values[i]

    return output_image_array

def gaussian_smooth(image_array, sigma=1):

    smooth_array = gaussian_filter(image_array, sigma=sigma, radius=1)
    smooth_image = np.clip(smooth_array, 0, 255)
    return smooth_image

def apply_bilinear_registration(source_points, target_points, input_image_path, output_image_path, output_size=(512, 512)):
    # 计算双线性模型的系数
    A = np.array([
        [source_points[0][0], source_points[0][1], source_points[0][0] * source_points[0][1], 1, 0, 0, 0, 0],
        [0, 0, 0, 0, source_points[0][0], source_points[0][1], source_points[0][0] * source_points[0][1], 1],
        [source_points[1][0], source_points[1][1], source_points[1][0] * source_points[1][1], 1, 0, 0, 0, 0],
        [0, 0, 0, 0, source_points[1][0], source_points[1][1], source_points[1][0] * source_points[1][1], 1],
        [source_points[2][0], source_points[2][1], source_points[2][0] * source_points[2][1], 1, 0, 0, 0, 0],
        [0, 0, 0, 0, source_points[2][0], source_points[2][1], source_points[2][0] * source_points[2][1], 1],
        [source_points[3][0], source_points[3][1], source_points[3][0] * source_points[3][1], 1, 0, 0, 0, 0],
        [0, 0, 0, 0, source_points[3][0], source_points[3][1], source_points[3][0] * source_points[3][1], 1],
    ])

    b = np.array(target_points).flatten()
    coefficients = np.linalg.solve(A, b)

    print(bilinear_interpolation(482, 574, coefficients))
    with Image.open(input_image_path) as img:
        img = img.convert('L')  # 转换为灰度图像
        img_array = np.array(img)
        height, width = img_array.shape
        output_height, output_width = 512, 512
        transformed_img_array = np.zeros((output_height, output_width), dtype=img_array.dtype)

        # 遍历源图像的每个像素点
        for v in range(height):
            for w in range(width):
                x, y = bilinear_interpolation(v, w, coefficients)

                # 如果目标坐标在 [0, 512) 范围内，则进行映射
                if 0 <= x <= output_width and 0 <= y <= output_height:
                    x = int(x)
                    y = int(y)
                    transformed_img_array[x, y] = img_array[v, w]

        transformed_img = Image.fromarray(transformed_img_array)
        transformed_img.save(output_image_path, format='TIFF')
        print(f"已成功保存配准后的图像到 {output_image_path}")

# # 输入和输出图像路径
# input_image_path = 'image/2.42_512try2.tif'  # 输入图像路径
# output_image_path = 'image/2.42_512four_point_moved.tif'  # 输出图像路径
# 示例数据：源点和目标点（四组数据）
# source_points = [
#     (59, 46),    # 源点1
#     (73, 433),   # 源点2
#     (485, 170),   # 源点3
#     (485, 555),  # 源点4
# ]
# target_points = [
#     (55, 30),    # 目标点1
#     (29, 423),   # 目标点2
#     (469, 29),   # 目标点3
#     (430, 425),  # 目标点4
# ]
# # 运行配准函数
# apply_bilinear_registration(source_points, target_points, input_image_path, output_image_path)


def apply_nine_point_registration(source_points, target_points, input_image_path, output_image_path, output_size=(512, 512)):
    # 将源点和目标点转换为 numpy 数组
    source_points = np.array(source_points)
    target_points = np.array(target_points)

    # 打开图像并获取像素值
    with Image.open(input_image_path) as img:
        img = img.convert('L')  # 转为灰度图像
        img_array = np.array(img)
        height, width = 512, 512
        transformed_img_array = np.zeros((height, width), dtype=img_array.dtype)

        # 创建目标网格的坐标
        grid_x, grid_y = np.mgrid[0:width, 0:height]

        # 使用 griddata 对源图像的像素进行插值计算目标像素
        grid_z = griddata(source_points, target_points, (grid_x, grid_y), method='cubic')

        # 遍历源图像的每个像素点
        for v in range(width):
            for w in range(height):
                # 使用插值计算新位置
                x, y = grid_z[v, w]

                # 如果目标坐标在范围内，则进行映射
                if 0 <= x < width and 0 <= y < height:
                    transformed_img_array[int(x), int(y)] = img_array[v, w]

        # 保存结果图像
        transformed_img = Image.fromarray(transformed_img_array)
        transformed_img.save(output_image_path, format='TIFF')
        print(f"已成功保存配准后的图像到 {output_image_path}")


def quadratic_interpolation(v, w, coefficients):
    """
    使用二次插值计算目标图像坐标
    v, w: 待配准图像的坐标
    coefficients: 二次模型的系数 [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12]
    返回目标图像坐标 (x, y)
    """
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = coefficients
    x = c1 * v**2 + c2 * w**2 + c3 * v * w + c4 * v + c5 * w + c6
    y = c7 * v**2 + c8 * w**2 + c9 * v * w + c10 * v + c11 * w + c12
    return x, y


def apply_quadratic_registration(source_points, target_points, input_image_path, output_image_path,
                                 output_size=(512, 512)):
    # 构建12参数二次插值模型的方程组
    A = []
    b = []

    for (sp, tp) in zip(source_points, target_points):
        v, w = sp
        x, y = tp

        A.append([v ** 2, w ** 2, v * w, v, w, 1, 0, 0, 0, 0, 0, 0])
        A.append([0, 0, 0, 0, 0, 0, v ** 2, w ** 2, v * w, v, w, 1])

        b.append(x)
        b.append(y)

    A = np.array(A)
    b = np.array(b)

    # 使用最小二乘法求解
    coefficients, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    with Image.open(input_image_path) as img:
        img = img.convert('L')  # 转换为灰度图像
        img_array = np.array(img)
        height, width = img_array.shape
        output_height, output_width = output_size
        transformed_img_array = np.zeros((output_height, output_width), dtype=img_array.dtype)

        # 遍历源图像的每个像素点
        for v in range(height):
            for w in range(width):
                y, x = quadratic_interpolation(v, w, coefficients)

                # 如果目标坐标在 [0, 512) 范围内，则进行映射
                if 0 <= x < output_width and 0 <= y < output_height:
                    x = int(x)
                    y = int(y)
                    transformed_img_array[y, x] = img_array[v, w]

        # 锐化和其他处理
        output_img_array = linear_interpolation(img_array, transformed_img_array)
        # output_img_array = gaussian_smooth(output_img_array)

        # 保存结果图像
        transformed_img = Image.fromarray(output_img_array.astype(np.uint8))
        transformed_img.save(output_image_path, format='TIFF')
        print(f"已成功保存配准并锐化后的图像到 {output_image_path}")


# 定义九个源点和目标点
source_points = [
    (59, 46), (73, 433), (485, 170), (485, 555),
    (246, 349), (71, 308), (257, 101), (524, 438), (295, 555)
]
target_points = [
    (55, 30), (29, 423), (469, 29), (430, 425),
    (210, 285), (39, 296), (245, 28), (479, 295), (238, 483)
]
# 输入和输出图像路径
input_image_path = 'image/2.42_512try22.tif'  # 输入图像路径
output_image_path = 'image/2.42_512nine_point_moved.tif'  # 输出图像路径
apply_quadratic_registration(source_points, target_points, input_image_path, output_image_path)