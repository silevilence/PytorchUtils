from utils.Dataset import MyDataset
from utils.transformparser import TransformParser
from utils.Classifier import Classifier

import torch.nn as nn
import torch
from torch.autograd import Variable
from PIL import Image

import os


class CNNClassifier(Classifier):
    """
    Use CNN to classify images.
    """

    def __init__(self, net_config: dict):
        # load net
        net_module = __import__(net_config['module'], fromlist=[net_config['net']])
        net_class = getattr(net_module, net_config['net'])
        self.net: nn.Module = net_class(**net_config['net_params'])

        # gpu mode
        if net_config['gpu']:
            self.net = self.net.cuda()
            self.V = lambda x, **params: Variable(x.cuda(), **params)
        else:
            self.V = lambda x, **params: Variable(x, **params)

        # load state dict(weights)
        self.net.load_state_dict(torch.load(net_config['weights']))

        # transform
        parser = TransformParser(net_params=net_config)
        self.transforms = parser.parse(net_config['transforms'])

        # offset(first tag)
        self.first_tag = net_config['first_tag']

        self.net.eval()

    def classify(self, image_path: str) -> int:
        pil_img = Image.open(image_path)
        data = self.transforms(pil_img)
        data = data.unsqueeze(0)
        inputs = self.V(data)
        # if self.half:
        #     inputs = inputs.half()
        output = self.net(inputs)

        prediction_max = torch.max(output, 1)[1]
        predy_int = int(prediction_max.data.numpy())
        return predy_int + self.first_tag
