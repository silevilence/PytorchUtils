from utils.Dataset import MyDataset
from utils.transformparser import TransformParser

import torch.nn as nn
import torch
import progressbar
from torch.autograd import Variable
from PIL import Image
import json

import os


class Classifier(object):
    def __init__(self, config: str = ''):
        # load configs
        default_config_file = open(
            os.path.abspath(
                os.path.join(os.path.dirname(__file__), os.path.pardir, 'presets', 'default_classifier.json')), 'r')
        default_config = json.load(default_config_file)
        if not (isinstance(config, str) and os.path.exists(config)):
            user_config = default_config
        else:
            user_config_file = open(config, 'r')
            user_config: dict = json.load(user_config_file)

        # load net
        net_module = __import__(user_config['module'], fromlist=['wzx'])
        net_class = getattr(net_module, user_config['net'])
        self.net: nn.Module = net_class(**user_config['net_params'])
        # self.half = user_config['half']
        # if self.half:
        #     self.net.half()

        # gpu mode
        if user_config['gpu']:
            self.net = self.net.cuda()
            self.V = lambda x, **params: Variable(x.cuda(), **params)
        else:
            self.V = lambda x, **params: Variable(x, **params)

        # load state dict(weights)
        weights = user_config['weights']
        if not os.path.exists(weights):
            weights = os.path.abspath(os.path.join(os.path.split(config)[0], os.path.split(weights)[1]))

        if user_config['gpu']:
            self.net.load_state_dict(torch.load(weights))
        else:
            self.net.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))

        # transform
        parser = TransformParser(net_params=user_config)
        self.transforms = parser.parse(user_config['transforms'])

        # offset(first tag)
        self.first_tag = user_config['first_tag']

        self.net.eval()

        # close file stream
        if 'user_config_file' in locals() or 'user_config_file' in globals():
            user_config_file.close()
        default_config_file.close()

    def classify(self, image_path: str):
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
