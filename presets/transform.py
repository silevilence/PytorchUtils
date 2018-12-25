import torchvision.transforms as T
import torch

import random
import numbers

from PIL import Image
from . import functional as F
from typing import Union
import collections


def PadImage(size: int):
    return [
        T.Pad(size // 2),
        T.CenterCrop(size)
    ]


def PostAlways(mean=None, std=None):
    if std is None:
        std = [.5, .5, .5]
    if mean is None:
        mean = [.5, .5, .5]
    return [
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ]


class ToRGB(object):
    """
    Convert a PIL Image to RGB
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image): Image to be converted to RGB.

        Returns:
            Tensor: Converted image.
        """
        if not (isinstance(pic, Image.Image)):
            raise TypeError(
                'pic should be PIL Image. Got {}'.format(type(pic)))

        return pic.convert('RGB')

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomTranslate(object):
    # noinspection PyTypeChecker
    def __init__(self, trans_range: Union[numbers.Number, tuple, list]):
        """Translate the given PIL Image randomly

        :param trans_range: range of translate. positive for right and down.
        """
        if isinstance(trans_range, numbers.Number):
            # 分别是左右上下四个界限
            self.range = (-abs(trans_range), abs(trans_range),
                          -abs(trans_range), abs(trans_range))
        elif len(trans_range) == 2:
            self.range = (-abs(trans_range[0]), abs(trans_range[0]
                                                    ), -abs(trans_range[1]), abs(trans_range[1]))
        else:
            self.range = (-abs(trans_range[0]), abs(trans_range[1]
                                                    ), -abs(trans_range[2]), abs(trans_range[3]))

    @staticmethod
    def get_params(trans_range):
        """Get parameters for `translate` for a random translate.

        :param trans_range: range of random translate. (l, r, t, b)
        :return: params to translate. (r, b)
        """
        r = random.randint(trans_range[0], trans_range[1])
        b = random.randint(trans_range[2], trans_range[3])
        return r, b

    def __call__(self, img):
        """

        :param img: image to be translated
        :return: translated PIL image
        """

        r, b = self.get_params(self.range)

        return F.translate(img, r, b)

    def __repr__(self):
        return f'{self.__class__.__name__}(range={self.range})'


class RandomCenterCrop(object):
    """Crops the given PIL Image at the center.
    Created by crh.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    # noinspection PyTypeChecker
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def get_params(self, img):
        i = random.randint(img.size[0] - self.size[0], img.size[0])
        j = random.randint(img.size[1] - self.size[1], img.size[1])
        return i, j

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        i, j = self.get_params(img)
        return F.center_crop(img, (j, i))

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomResize(object):
    # noinspection PyTypeChecker
    def __init__(self, size=(0.8, 1.2)):
        """Random resize by ratio

        :param size: ratio tuple or list
        """
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size

    def get_params(self, img):
        isize = img.size
        edge = min(isize)
        s = random.randint(edge * self.size[0], edge * self.size[1])
        return s

    def __call__(self, img):
        """random resize.
        by zxn.

        :param img: PIL image
        :return: resized pil image
        """
        s = self.get_params(img)
        return T.Resize(s)(img)
