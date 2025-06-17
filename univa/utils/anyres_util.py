import math
from PIL import Image

RATIO = {
    'any_15ratio': [(2, 1), (1, 2), (16, 9), (9, 16), (7, 6), (6, 7), (7, 5), (5, 7), (5, 4), (4, 5), (4, 3), (3, 4), (3, 2), (2, 3), (1, 1)],
    'any_11ratio': [(16, 9), (9, 16), (7, 5), (5, 7), (5, 4), (4, 5), (4, 3), (3, 4), (3, 2), (2, 3), (1, 1)],
    'any_9ratio': [(16, 9), (9, 16), (5, 4), (4, 5), (4, 3), (3, 4), (3, 2), (2, 3), (1, 1)],
    'any_7ratio': [(16, 9), (9, 16), (4, 3), (3, 4), (3, 2), (2, 3), (1, 1)],
    'any_5ratio': [(16, 9), (9, 16), (4, 3), (3, 4), (1, 1)],
    'any_1ratio': [(1, 1)],
}

def dynamic_resize(h, w, anyres='any_1ratio', anchor_pixels=1024 * 1024, stride=32):
    
    orig_ratio = w / h

    # 找到与原图比例最接近的候选比例
    target_ratio = min(RATIO[anyres], key=lambda x: abs((x[0] / x[1]) - orig_ratio))
    rw, rh = target_ratio

    # 计算 stride 对齐下的最小基准尺寸
    base_h = rh * stride
    base_w = rw * stride
    base_area = base_h * base_w

    # 计算在该比例和 stride 对齐下，能接近 anchor_pixels 的放缩因子
    scale = round(math.sqrt(anchor_pixels / base_area))

    new_h = base_h * scale
    new_w = base_w * scale

    return new_h, new_w



def concat_images_adaptive(images, bg_color=(255, 255, 255)):
    """
    将任意数量的 PIL.Image 对象自适应地拼接成一个近似正方形的网格图像。
    
    参数:
        images (list of PIL.Image): 要拼接的图像列表。
        bg_color (tuple of int): 背景颜色，默认为白色 (255, 255, 255)。
    
    返回:
        PIL.Image: 拼接后的大图。
    """
    if not images:
        raise ValueError("images 列表不能为空")

    n = len(images)
    
    # 计算网格行列数，尽可能接近正方形
    cols = int(n**0.5)
    if cols * cols < n:
        cols += 1
    rows = (n + cols - 1) // cols

    # 找到所有图像的最大宽度和最大高度
    widths, heights = zip(*(img.size for img in images))
    max_w = max(widths)
    max_h = max(heights)

    # 创建新的画布
    new_img = Image.new('RGB', (cols * max_w, rows * max_h), color=bg_color)

    # 逐行逐列粘贴图像，若某行列没有图像则留空白
    for idx, img in enumerate(images):
        row_idx = idx // cols
        col_idx = idx % cols
        # 如果图像尺寸不同，可选：在粘贴前将其居中放置于单元格，或调整为单元格大小
        offset_x = col_idx * max_w + (max_w - img.width) // 2
        offset_y = row_idx * max_h + (max_h - img.height) // 2
        new_img.paste(img, (offset_x, offset_y))

    return new_img