import torchvision.transforms as T
import torch


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
