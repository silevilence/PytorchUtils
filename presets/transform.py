import torchvision.transforms as T


def PadImage(size: int):
    return [
        T.Pad(size // 2),
        T.CenterCrop(size)
    ]


def PostAlways(mean=[.5, .5, .5], std=[.5, .5, .5]):
    return [
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ]
