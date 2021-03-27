# -*- coding: utf-8 -*-
"""
This file contains all operations about building LeNet5

Created by Kunhong Yu
Date: 2021/03/09
"""

import torch as t

class LeNet5(t.nn.Module):
    """Define LeNet5 model only for MNIST"""
    def __init__(self, **kwargs):
        super(LeNet5, self).__init__()

        self.conv_layers = t.nn.Sequential(
            t.nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5, stride = 1),
            t.nn.ReLU(inplace = True),
            t.nn.MaxPool2d(kernel_size = 2, stride = 2),

            t.nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1),
            t.nn.ReLU(inplace = True),
            t.nn.MaxPool2d(kernel_size = 2, stride = 2),
        )

        self.fc_layers = t.nn.Sequential(
            t.nn.Linear(256, 128),
            t.nn.ReLU(inplace = True),

        )

        self.final = t.nn.Linear(128, 10)


    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        out = self.fc_layers(x)# out for evaluation
        x = self.final(out)

        return x, out