import torch
from torch.utils.data import Dataset
from PIL import Image
import os


def default_loader(path):
    return Image.open(path)


class MyDataset(Dataset):
    def __init__(self, txt, encoding='utf-8', data_root='.', transform=None, loader=default_loader, count=True,
                 title=''):
        if not isinstance(data_root, str):
            data_root = '.'
        with open(txt, 'r', encoding=encoding) as fh:
            if count:
                data_count = dict()
            imgs = []
            for line in fh.readlines():
                line = line.strip('\r\n')
                line = line.strip()
                words = line.split()
                img = os.path.join(data_root, os.path.split(words[0])[1])
                if os.path.exists(img):
                    tag = int(words[1])
                    imgs.append((img, tag))
                    if count:
                        if tag in data_count:
                            data_count[tag] += 1
                        else:
                            data_count[tag] = 1
                else:
                    print(f"'{img}' doesn't exist, skipped.")
        self.imgs = imgs
        self.transform = transform
        self.loader = loader if loader is not None else default_loader

        print(f'A total of {len(imgs)} images were loaded.')
        if count:
            print(title)
            for k, v in sorted(data_count.items()):
                print(f'{k}: {v}')

    def __getitem__(self, index):
        imgpath, label = self.imgs[index]
        img = self.loader(imgpath)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, imgpath

    def __len__(self):
        return len(self.imgs)
