'''
A slightly modified ResNet implementation from github repo [1] by Yerlan Idelbayev.
Original file header comment is left below as-is.
Reference:
[1] https://github.com/akamaster/pytorch_resnet_cifar10

Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
from typing import Dict, Union, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import optim


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(1, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, self.in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, self.in_planes * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, self.in_planes * 2, num_blocks[2], stride=2)
        self.linear1 = nn.Linear(self.in_planes, 26)
        self.linear2 = nn.Linear(self.in_planes, 26)
        self.linear3 = nn.Linear(self.in_planes, 26)
        self.linear4 = nn.Linear(self.in_planes, 26)
        self.linear5 = nn.Linear(self.in_planes, 27)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, option='B'))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = F.avg_pool2d(out, out.size()[2:])
        out = F.avg_pool2d(out, (10, 25), stride=(10, 25))
        out = out.view(out.size(0), -1)
        out1 = self.linear1(out)
        out2 = self.linear2(out)
        out3 = self.linear3(out)
        out4 = self.linear4(out)
        out5 = self.linear5(out)

        return out1, out2, out3, out4, out5


class Metrics:
    def __init__(self):
        self.correct = 0
        self.total = 0
        self.per_char_total = 0
        self.per_char_correct = 0

    def update(self, pred, target):
        predicted = torch.stack([tensor.max(1)[1] for tensor in pred], 1)
        targets_stacked = torch.stack(target, 1)
        self.total += target[0].size(0)
        self.correct += torch.all(predicted.eq(targets_stacked), 1).sum().item()
        self.per_char_total += target[0].size(0) * 5
        self.per_char_correct += predicted.eq(targets_stacked).sum().item()

    def reset(self):
        self.correct = 0
        self.total = 0
        self.per_char_total = 0
        self.per_char_correct = 0

    def accuracy(self):
        return self.correct / self.total

    def char_accuracy(self):
        return self.per_char_correct / self.per_char_total

    def log_dict(self, prefix=""):
        return {
            f"{prefix}acc": 100. * self.accuracy(),
            f"{prefix}correct": 1. * self.correct,
            f"{prefix}total": 1. * self.total,
            f"{prefix}char_acc": 100. * self.char_accuracy(),
            f"{prefix}char_correct": 1. * self.per_char_correct,
            f"{prefix}char_total": 1. * self.per_char_total
        }

    def log_str(self):
        return f"[S]{100. * self.accuracy():.2f}%[{self.correct}/{self.total}] [C]{100. * self.char_accuracy():.2f}%[{self.per_char_correct}/{self.per_char_total}]"

    @staticmethod
    def from_dict(d: Dict, prefix="", pop=True) -> Optional["Metrics"]:
        if f"{prefix}acc" not in d:
            return None
        this = Metrics()
        if pop:
            d.pop(f"{prefix}acc")
            d.pop(f"{prefix}char_acc")
            this.correct = d.pop(f"{prefix}correct")
            this.total = d.pop(f"{prefix}total")
            this.per_char_correct = d.pop(f"{prefix}char_correct")
            this.per_char_total = d.pop(f"{prefix}char_total")
        else:
            this.correct = d[f"{prefix}correct"]
            this.total = d[f"{prefix}total"]
            this.per_char_correct = d[f"{prefix}char_correct"]
            this.per_char_total = d[f"{prefix}char_total"]
        return this


class PLWrapper(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float):
        super().__init__()
        self.save_hyperparameters("lr")

        self.model = model
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.val_metrics = Metrics()
        self.train_metrics = Metrics()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)

        losses = []
        for idx in range(len(outputs)):
            losses.append(self.criterion(outputs[idx], targets[idx]))
        loss = sum(losses)

        # Calculate accuracy (sentence-based and char-based)
        self.train_metrics.update(outputs, targets)

        self.log("loss", loss)
        self.log_dict(self.train_metrics.log_dict(), prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)

        losses = []
        for idx in range(len(outputs)):
            losses.append(self.criterion(outputs[idx], targets[idx]))
        loss = sum(losses)

        self.log("val_loss", loss)

        # Calculate accuracy (sentence-based and char-based)
        self.val_metrics.update(outputs, targets)

    def on_train_epoch_start(self):
        self.train_metrics.reset()

    def on_validation_epoch_start(self):
        self.val_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.log_dict("val_"), prog_bar=True, logger=True)


class CustomProgressBar(pl.callbacks.RichProgressBar):
    def get_metrics(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> Dict[str, Union[int, str, float, Dict[str, float]]]:
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num")
        train_report = Metrics.from_dict(items)
        val_report = Metrics.from_dict(items, prefix="val_")
        if train_report is not None:
            items["train"] = train_report.log_str()
        if val_report is not None:
            items["val"] = val_report.log_str()
        return items


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])
