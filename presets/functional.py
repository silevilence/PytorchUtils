# coding=utf-8
from torchvision.transforms import functional as F
from PIL import Image


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def translate(img: Image.Image, right: int, down: int) -> Image.Image:
    """
    平移的实现函数。
    单位是像素，以右下为正。
    使用先裁剪后补0的方法来模拟平移操作。

    举例说明：

    假设向右下平移(x, y)。

    Step 0: 生成裁剪和补0函数需要的参数。

    Step 1: 裁去平移后移出的部分，即右边x像素，下边y像素。

    Step 2: 在空出来的部分补0，即左边x像素，上边y像素。

    裁剪和补0使用 `torchvision.transforms.functional` 中的相应方法。

    :param img: image to be translated
    :param right: pixels to move right
    :param down: pixels to move down
    """
    # type check
    if not _is_pil_image(img):
        raise TypeError(f'img should be PIL Image. Got {type(img)}')
    if not isinstance(right, int):
        raise TypeError(f'right should be int. Got {type(right)}')
    if not isinstance(down, int):
        raise TypeError(f'down should be int. Got {type(right)}')

    # crop 的参数为左上的起始点和宽度和高度，原点在左上，右下为正
    # 注意crop的参数中顺序为高、宽，PIL里的顺序为宽、高，反的……
    img_size = img.size

    crop_h = img_size[1] - abs(down)
    crop_h = 0 if crop_h < 0 else crop_h

    crop_w = img_size[0] - abs(right)
    crop_w = 0 if crop_w < 0 else crop_w

    # 如果向左上移动，需要设置左上坐标
    crop_i = abs(down) if down < 0 else 0
    crop_j = abs(right) if right < 0 else 0

    img = F.crop(img, crop_i, crop_j, crop_h, crop_w)

    # padding
    # pad 的参数按下面这个顺序，打包成元组
    pad_left = pad_top = pad_right = pad_bottom = 0
    pad_w = img_size[0] - crop_w
    pad_h = img_size[1] - crop_h
    if right > 0:  # 右移，补左边
        pad_left = pad_w
    else:
        pad_right = pad_w
    if down > 0:  # 下移，补上边
        pad_top = pad_h
    else:
        pad_bottom = pad_h

    return F.pad(img, (pad_left, pad_top, pad_right, pad_bottom))
