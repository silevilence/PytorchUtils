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

from Dataset import MyDataset


class Solver(object):
    def __init__(self, config: str = ''):
        # load configs
        default_config_file = open(
            os.path.join(os.path.split(os.path.realpath(__file__))[0], 'default.json'), 'r')
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
        self.net = net_class(**net_config['net_params'])
        # print(self.net)

        # optimizer
        optimizer = getattr(torch.optim, net_config['optimizer'])
        self.optimizer = optimizer(
            self.net.parameters(), **net_config['optimizer_params'])

        # lr_scheduler
        scheduler = getattr(lr_scheduler, net_config['lr_scheduler'])
        self.scheduler = scheduler(
            self.optimizer, **net_config['lr_scheduler_params'])

        # loss
        loss_func = getattr(nn, net_config['loss_func'])
        self.loss_func = loss_func()

        # tensorboardX
        self.log_root = net_config['log_root']
        self.log_file = net_config['log_file']

        # gpu mode
        if net_config['gpu']:
            self.net = self.net.cuda()
            self.V = lambda x, **params: Variable(x.cuda(), **params)
        else:
            self.V = lambda x, **params: Variable(x, **params)

        # transform
        image_size = net_config['image_size']
        transform = transforms.Compose([
            transforms.Pad(image_size // 2),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])

        # load train config
        train_config: dict = dict(
            default_config['train'], **user_config.get('train', default_config['train'])
        )
        # train dataset
        data_root = train_config['data_root'] if 'data_root' in train_config else net_config['data_root']
        self.train_data = MyDataset(txt=os.path.join(data_root, train_config['data']),
                                    transform=transform, data_root=net_config['image_root'],
                                    count=train_config['count'], title="Train data:")
        self.train_loader = DataLoader(
            dataset=self.train_data, **train_config['loader_params'])

        self.max_epoches = train_config['max_epoches']
        self.display_interval = train_config['display_interval']

        # snapshot
        self.snapshot_interval = train_config['snapshot_interval']
        self.snapshot_prefix = train_config['snapshot_prefix']
        self.snapshot_root = train_config['snapshot_root']

        # load eval config
        eval_config: dict = dict(
            default_config['eval'], **user_config.get('eval', default_config['eval'])
        )
        # eval dataset
        data_root = eval_config['data_root'] if 'data_root' in eval_config else net_config['data_root']
        self.eval_data = MyDataset(txt=os.path.join(data_root, eval_config['data']),
                                   transform=transform, data_root=net_config['image_root'],
                                   count=eval_config['count'], title='Eval data:')
        self.eval_loader = DataLoader(
            dataset=self.eval_data, **eval_config['loader_params'])

        # load test config
        test_config: dict = dict(
            default_config['test'], **user_config.get('test', default_config['test'])
        )
        # test dataset
        data_root = test_config['data_root'] if 'data_root' in test_config else net_config['data_root']
        self.test_data = MyDataset(txt=os.path.join(data_root, test_config['data']),
                                   transform=transform, data_root=net_config['image_root'],
                                   count=test_config['count'], title='Test data:')
        self.test_loader = DataLoader(
            dataset=self.test_data, **test_config['loader_params'])

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

    def train(self):
        niter = 0
        self.writer = tensorboardX.SummaryWriter(
            log_dir=os.path.join(self.log_root, self.log_file), comment=os.path.splitext(self.log_file)[0])
        for epoch in range(self.max_epoches):
            self.scheduler.step()
            for p in self.optimizer.param_groups:
                print(f'epoch {epoch + 1}: lr: {p["lr"]}')
                # log
                self.writer.add_scalar('lr', p['lr'], epoch)

            # training
            self.net.train()
            train_loss = 0.
            train_acc = 0
            total = 0

            running_loss = 0.
            running_acc = 0
            running_total = 0

            for imgs, labels, _ in self.train_loader:
                niter += 1

                imgs, labels = self.V(imgs), self.V(labels)

                self.optimizer.zero_grad()

                out = self.net(imgs)

                loss = self.loss_func(out, labels)
                train_loss += loss.item()
                running_loss += loss.item()
                pred = torch.max(out, 1)[1]

                train_correct = (pred == labels).sum()
                train_acc += train_correct.item()
                running_acc += train_correct.item()

                total += labels.size(0)
                running_total += labels.size(0)

                loss.backward()
                self.optimizer.step()

                # running display
                if niter % self.display_interval == 0:
                    print(
                        f'iter {niter}: Train Loss: {running_loss / running_total:.6f}, Acc: {running_acc / running_total:.6f}({running_acc}/{running_total})')
                    # log
                    self.writer.add_scalar(
                        'running/loss', running_loss / running_total, niter)
                    self.writer.add_scalar(
                        'running/acc', running_acc / running_total, niter)
                    running_loss = 0.
                    running_acc = 0
                    running_total = 0
            print(
                f'epoch {epoch + 1}: Train Loss: {train_loss / total:.6f}, Acc: {train_acc / total:.6f}({train_acc}/{total})')
            # log
            self.writer.add_scalar('train/loss', train_loss / total, epoch + 1)
            self.writer.add_scalar('train/acc', train_acc / total, epoch + 1)

            # evaluation
            self.net.eval()
            eval_loss = 0.
            eval_acc = 0
            total = 0
            for imgs, labels, _ in self.eval_loader:
                with torch.no_grad():
                    imgs, labels = self.V(imgs), self.V(labels)
                    out = self.net(imgs)
                    loss = self.loss_func(out, labels)
                    eval_loss += loss.item()
                    pred = torch.max(out, 1)[1]
                    num_correct = (pred == labels).sum()
                    eval_acc += num_correct.item()
                    total += labels.size(0)
            print(
                f'epoch {epoch + 1}: Eval Loss: {eval_loss / total:.6f}, Acc: {eval_acc / total:.6f}({eval_acc}/{total})')
            # log
            self.writer.add_scalar('eval/loss', eval_loss / total, epoch + 1)
            self.writer.add_scalar('eval/acc', eval_acc / total, epoch + 1)

            # save snapshot
            if (epoch + 1) % self.snapshot_interval == 0:
                filename = os.path.join(
                    self.snapshot_root, f'{self.snapshot_prefix}_epoch_{epoch + 1}.pkl')
                torch.save(self.net.state_dict(), filename)
                print(f"snapshot '{filename}' saved.")
                if self.test_on_train:
                    self._test(epoch + 1)

        # self.writer.export_scalars_to_json(self.log_file)
        self.writer.close()

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

                loss = self.loss_func(out, labels)
                test_loss += loss.item()

                pred = torch.max(out, 1)[1]
                num_correct = (pred == labels).sum()

                test_acc += num_correct.item()
        print(
            f'Test Loss: {test_loss / total:.6f}, Acc: {test_acc / total:.6f}({test_acc}/{total})')
        # log
        self.writer.add_scalar('test/loss', test_loss / total, epoch + 1)
        self.writer.add_scalar('test/acc', test_acc / total, epoch + 1)

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

        # top2 output
        errors = list(filter(lambda x: x[1] != x[2] and x[1] != x[4], results))
        acc = (total - len(errors)) / total
        filename = os.path.join(
            self.result_root, f'{self.top2_prefix}_epoch{epoch}({len(errors)}).txt')
        with open(filename, 'w', newline='') as t2f:
            for line in errors:
                t2f.write(
                    f'{line[0]} {line[1] + self.first_tag} {line[2] + self.first_tag} {line[3]} {line[4] + self.first_tag} {line[5]}\n')
            # print result satisfy
            t2f.write(f'accuracy: {acc}({total - len(errors)}/{total})\n')
