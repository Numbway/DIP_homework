import numpy as np
from PIL import Image


def get_difference(fixed_image_path, moved_image_path, diff_image_path):
    with Image.open(fixed_image_path) as fixed_img:
        # fixed_img = fixed_img.convert('L')
        fixed_array = np.array(fixed_img)
        height, width = fixed_array.shape

    with Image.open(moved_image_path) as moved_img:
        # moved_img = moved_img.convert('L')
        moved_array = np.array(moved_img)
    diff_image_array = np.zeros((height, width), dtype=fixed_array.dtype)
    for x in range(height):
        for y in range(width):
            diff = abs(abs(fixed_array[x, y]) - abs(moved_array[x, y]))
            diff_image_array[x, y] = diff

    diff_img = Image.fromarray(diff_image_array)
    diff_img.save(diff_image_path, format='TIFF')
    print(f"已成功保存差异图像到 {diff_image_path}")


fixed_image_path = "image/2.42_512fixed.tif"
moved_image_path = "image/2.42_512nine_point_moved.tif"
diff_image_path = "image/2.42_512test.tif"
get_difference(fixed_image_path, moved_image_path, diff_image_path)


