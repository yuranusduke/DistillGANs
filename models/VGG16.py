# -*- coding: utf-8 -*-
"""
This file contains all operations about building VGG16 model

Created by Kunhong Yu
Date: 2021/03/21
"""
import torch as t
from torch.nn import functional as F

class VGG16(t.nn.Module):
    """Define VGG16 model"""

    def __init__(self, num_classes = 10):
        super(VGG16, self).__init__()

        self.layer1 = t.nn.Sequential(
            t.nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            t.nn.BatchNorm2d(64),
            t.nn.ReLU(inplace = True),
            t.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            t.nn.BatchNorm2d(64),
            t.nn.ReLU(inplace = True),
            t.nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer2 = t.nn.Sequential(
            t.nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            t.nn.BatchNorm2d(128),
            t.nn.ReLU(inplace = True),
            t.nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            t.nn.BatchNorm2d(128),
            t.nn.ReLU(inplace = True),
            t.nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer3 = t.nn.Sequential(
            t.nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            t.nn.BatchNorm2d(256),
            t.nn.ReLU(inplace = True),
            t.nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            t.nn.BatchNorm2d(256),
            t.nn.ReLU(inplace = True),
            t.nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            t.nn.BatchNorm2d(256),
            t.nn.ReLU(inplace = True),
            t.nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer4 = t.nn.Sequential(
            t.nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            t.nn.BatchNorm2d(512),
            t.nn.ReLU(inplace = True),
            t.nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            t.nn.BatchNorm2d(512),
            t.nn.ReLU(inplace = True),
            t.nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            t.nn.BatchNorm2d(512),
            t.nn.ReLU(inplace = True),
            t.nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.fc = t.nn.Sequential(
            t.nn.Linear(512, 256),
            t.nn.ReLU(inplace = True),
            t.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        out = F.adaptive_avg_pool2d(x, (1, 1)) # We use global average pooling instead
        x = out.squeeze()
        x = self.fc(x)

        return x, out