import json
import os

import torch
import torch.nn as nn
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import tensorboardX
import progressbar

from utils.Dataset import MyDataset
from utils.transformparser import TransformParser


class Tester(object):
    def __init__(self, config: str = '', test_root: str = '.'):
        self.test_root = test_root

        # load configs
        default_config_file = open(
            os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'presets', 'default_tester.json')), 'r')
        default_config = json.load(default_config_file)
        if not (isinstance(config, str) and os.path.exists(config)):
            user_config = default_config
        else:
            user_config_file = open(config, 'r')
            user_config: dict = json.load(user_config_file)

        # load net
        net_config: dict = dict(
            default_config['net'], **user_config.get('net', default_config['net']))
        net_module = __import__(net_config['module'], fromlist=['wzx'])
        net_class = getattr(net_module, net_config['net'])
        # self.num_classes = net_config['num_classes']
        # self.first_tag = net_config['first_tag'] # to test_config
        self.net: nn.Module = net_class(**net_config['net_params'])
        # print(self.net)

        # gpu mode
        if net_config['gpu']:
            self.net = self.net.cuda()
            self.V = lambda x, **params: Variable(x.cuda(), **params)
        else:
            self.V = lambda x, **params: Variable(x, **params)

        # load test config
        test_config: dict = dict(
            default_config['test'], **user_config.get('test', default_config['test'])
        )

        parser = TransformParser(net_params=net_config)
        # transform
        test_transform = parser.parse(test_config['transforms'])

        # test dataset
        data_root = test_config['data_root'] if 'data_root' in test_config else net_config['data_root']
        self.test_data = MyDataset(txt=os.path.join(data_root, test_config['data']),
                                   transform=test_transform, data_root=net_config['image_root'],
                                   count=test_config['count'], title='Test data:')
        self.test_loader = DataLoader(dataset=self.test_data, **test_config['loader_params'])

        self.test_on_train = test_config['test_on_train']
        self.first_tag = test_config['first_tag']
        self.result_root = test_config['result_root']
        self.top1_prefix = test_config['top1_prefix']
        self.top2_prefix = test_config['top2_prefix']
        if 'num_classes' in net_config['net_params']:
            self.num_classes = net_config['net_params']['num_classes']
        else:
            self.num_classes = test_config['num_classes']

        # close file stream
        if 'user_config_file' in locals() or 'user_config_file' in globals():
            user_config_file.close()
        default_config_file.close()

    def test(self):
        weightses = os.listdir(self.test_root)
        weightses = list(filter(lambda x: os.path.splitext(x)[-1] == '.pkl', weightses))

        pb = progressbar.ProgressBar(widgets=[progressbar.Percentage(),
                                              '(', progressbar.SimpleProgress(), ')',
                                              progressbar.Bar(),
                                              progressbar.ETA(),
                                              ' ', progressbar.Timer()])
        pb.start(max_value=len(weightses))

        min_err = 10000
        best_epoch = 0
        for i, weights_file in enumerate(weightses):
            pb.update(i)

            # weights config
            weights_path = os.path.join(self.test_root, weights_file)
            epoch = int(weights_file.split('_')[-1].split('.')[0])

            # load weights
            self.net.load_state_dict(torch.load(weights_path))

            total, errors = self._test(epoch=epoch)
            if errors < min_err:
                min_err = errors
                best_epoch = epoch

        pb.finish()

        print(f'best epoch: epoch {best_epoch}, errors: {min_err} / {total}')

    def _test(self, epoch: int):
        compare_mat = dict()
        # initial for compare mat
        for r in range(self.num_classes):
            for c in range(self.num_classes):
                compare_mat[(r, c)] = 0

        self.net.eval()
        test_loss = 0.
        test_acc = 0
        total = 0
        results = []
        for imgs, labels, imgpathes in self.test_loader:
            with torch.no_grad():
                imgs, labels = self.V(imgs), self.V(labels)
                out = self.net(imgs)
                total += labels.size(0)

                p = nn.functional.softmax(out, dim=1)
                results_iter = torch.topk(p, 2, dim=1)
                # calculate result
                for img, label, possible, predict in zip(imgpathes, labels, results_iter[0], results_iter[1]):
                    results.append((os.path.split(img)[1], label.item(), predict[0].item(
                    ), possible[0].item(), predict[1].item(), possible[1].item()))
                    # count for mat
                    compare_mat[(label.item(), predict[0].item())] += 1

                pred = torch.max(out, 1)[1]
                num_correct = (pred == labels).sum()

                test_acc += num_correct.item()

        total = len(results)
        # top1 output
        errors = list(filter(lambda x: x[1] != x[2], results))
        acc = (total - len(errors)) / total
        filename = os.path.join(
            self.result_root, f'{self.top1_prefix}_epoch{epoch}({len(errors)}).txt')
        with open(filename, 'w', newline='') as t1f:
            for line in errors:
                t1f.write(
                    f'{line[0]} {line[1] + self.first_tag} {line[2] + self.first_tag} {line[3]}\n')
            # print result satisfy
            t1f.write(f'accuracy: {acc}({total - len(errors)}/{total})\n')
            # print compare mat
            t1f.write('\t')
            t1f.write(
                '\t'.join([str(x + self.first_tag) for x in range(self.num_classes)]))
            t1f.write('\n')
            for r in range(self.num_classes):
                t1f.write(str(r + self.first_tag))
                for c in range(self.num_classes):
                    t1f.write(f'\t{compare_mat[(r, c)]}')
                t1f.write('\n')

        rtotal = total
        rerrors = len(errors)

        # top2 output
        errors = list(filter(lambda x: x[1] != x[2] and x[1] != x[4], results))
        acc = (total - len(errors)) / total
        filename = os.path.join(
            self.result_root, f'{self.top2_prefix}_epoch{epoch}({len(errors)}).txt')
        with open(filename, 'w', newline='') as t2f:
            for line in errors:
                t2f.write(
                    f'{line[0]} {line[1] + self.first_tag} {line[2] + self.first_tag} {line[3]}\
 {line[4] + self.first_tag} {line[5]}\n')
            # print result satisfy
            t2f.write(f'accuracy: {acc}({total - len(errors)}/{total})\n')

        return rtotal, rerrors
