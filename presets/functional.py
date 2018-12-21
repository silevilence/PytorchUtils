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

    裁剪和补0使用 `PIL.Image.Image.crop` 自动完成。

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

    # crop
    img_size = img.size
    box_left = -right
    box_top = -down
    box_right = img_size[0] + box_left
    box_down = img_size[1] + box_top

    return img.crop((box_left, box_top, box_right, box_down))
