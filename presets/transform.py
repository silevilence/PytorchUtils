import torchvision.transforms as T
import torch

from PIL import Image


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
            raise TypeError('pic should be PIL Image. Got {}'.format(type(pic)))

        return pic.convert('RGB')

    def __repr__(self):
        return self.__class__.__name__ + '()'
