from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import math

from core.clearn import ContinualLearner
from utils.nn_utils import NeuralNetUtils as nnu


class NeuralNet(ContinualLearner):

    def __init__(self, net: nn.Module, loss: _Loss, optimizer: Optimizer, scheduler: _LRScheduler=None, device='cpu'):
        super().__init__()
        self.net = net.to(device)
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def predict(self, x_batch):
        return torch.max(self.predict_prob((x_batch.to(self.device))), 1)[1]

    def predict_prob(self, x_batch):
        self.net.eval()
        with torch.no_grad():
            return self.net(x_batch.to(self.device)).cpu()

    def update(self, x_batch, y_batch, **kwargs):
        y_batch = y_batch.long()

        self.net.train()
        self.optimizer.zero_grad()

        outputs = self.net(x_batch.to(self.device))
        loss = self.loss(outputs, y_batch.to(self.device))

        loss.backward()
        self.optimizer.step()

    def get_net(self):
        return self.net

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            size = m.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            torch.nn.init.normal_(m.weight, mean=0., std=math.sqrt(2 / (fan_in + fan_out)))
            m.bias.data.fill_(1)

    @staticmethod
    def create_nn(shape, dropout):
        classifier = nn.Sequential()

        for i in range(len(shape) - 2):
            layer = nn.Linear(shape[i], shape[i + 1])
            layer.apply(NeuralNet.init_weights)
            act = nn.ReLU()

            if i != len(shape) - 1:
                layer = nn.Sequential(layer, nn.BatchNorm1d(shape[i + 1]), act, nn.Dropout(p=dropout))
            else:
                layer = nn.Sequential(layer, nn.BatchNorm1d(shape[i + 1]), act)

            classifier.add_module(str(i), layer)

        layer = nn.Linear(shape[-2], shape[-1])
        layer.apply(NeuralNet.init_weights)
        classifier.add_module('output', layer)

        return classifier


class CifarResNet(nn.Module):

    # https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
    # Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, Deep Residual Learning for Image Recognition. arXiv:1512.03385

    def __init__(self, block, num_blocks, in_size=(3, 32, 32), out_size=10):
        super(CifarResNet, self).__init__()
        self.in_planes = 64

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            self._make_layer(block, 64, num_blocks[0], stride=1),
            self._make_layer(block, 128, num_blocks[1], stride=2),
            self._make_layer(block, 256, num_blocks[2], stride=2),
            self._make_layer(block, 512, num_blocks[3], stride=2),
            nn.AvgPool2d(4)
        )

        flat_num = nnu.flat_num(self.feature_extractor, in_size)
        self.fc1 = nn.Sequential(nn.Linear(flat_num, 128), nn.ReLU())
        self.fc2 = nn.Linear(128, out_size)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.feature_extractor(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


def CifarResNet18(in_size=(3, 32, 32), out_size=10):
    return CifarResNet(BasicBlock, [2, 2, 2, 2], in_size, out_size)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class MnistNet(nn.Module):

    def __init__(self, in_size=(1, 28, 28), out_size=10):
        super(MnistNet, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_size[0], 32, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2))

        flat_num = nnu.flat_num(self.feature_extractor, in_size)

        self.fc1 = nn.Sequential(nn.Linear(flat_num, 128), nn.ReLU())
        self.fc2 = nn.Linear(128, out_size)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.reshape(x.size(0), -1)
        out = self.fc1(x)
        out = self.fc2(out)

        return out


def mnistnet(model_path: str, device):
    net = MnistNet()
    net.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    return net


def cifar10_resnet(model_path: str, device):
    net = CifarResNet18(in_size=(3, 32, 32), out_size=10)
    net.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    return net
