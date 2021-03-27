# -*- coding: utf-8 -*-
"""
This file contains all operations about compressing GAN
using KD method
From paper: <Compressing GANs using Knowledge Distillation>

Created by Kunhong Yu
Date: 2021/03/01
"""
import torch as t

class Discriminator(t.nn.Module):
    """Define Discriminator, specifically, we use DCGAN's discriminator"""
    def __init__(self, in_channels = 3):
        super(Discriminator, self).__init__()

        self.layers = t.nn.Sequential(
            t.nn.Conv2d(in_channels = in_channels, out_channels = 256,
                        kernel_size = 4, stride = 2, padding = 1),
            t.nn.LeakyReLU(0.2, inplace = True),

            t.nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 4,
                        stride = 2, padding = 1),
            t.nn.BatchNorm2d(512),
            t.nn.LeakyReLU(0.2, inplace = True),

            t.nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 4,
                        stride = 2, padding = 1),
            t.nn.BatchNorm2d(1024),
            t.nn.LeakyReLU(0.2, inplace = True)
        )

        self.final = t.nn.Sequential(
            t.nn.Conv2d(in_channels = 1024, out_channels = 1,
                        kernel_size = 3, stride = 1, padding = 0),
            t.nn.AdaptiveAvgPool2d(1),
            t.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.final(x)

        return x

class GeneratorT(t.nn.Module):
    """Define Generator for teacher, specifically, we use DCGAN's generator"""
    def __init__(self, in_dimensions = 100, out_channels = 3):
        super(GeneratorT, self).__init__()

        self.layers = t.nn.Sequential(
            t.nn.ConvTranspose2d(in_channels = in_dimensions, out_channels = 1024, kernel_size = 3,
                                 stride = 1, padding = 0),# 3
            t.nn.BatchNorm2d(1024),
            t.nn.ReLU(inplace = True),

            t.nn.ConvTranspose2d(in_channels = 1024, out_channels = 512, kernel_size = 4,
                                 stride = 2, padding = 0),# 6
            t.nn.BatchNorm2d(512),
            t.nn.ReLU(inplace = True),

            t.nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = 4,
                                 stride = 2, padding = 1),# 14 # 16 for cifar10
            t.nn.BatchNorm2d(256),
            t.nn.ReLU(inplace = True),

            t.nn.ConvTranspose2d(in_channels = 256, out_channels = out_channels, kernel_size = 4,
                                 stride = 2, padding = 1),# 28
        )

        self.final = t.nn.Tanh()

    def forward(self, x):
        x = self.layers(x)
        x = self.final(x)

        return x

class GeneratorS(t.nn.Module):
    """Define Generator for student, specifically, we reduce some layers in GeneratorT"""
    def __init__(self, in_dimensions = 100, out_channels = 3, scale = 0.1):
        super(GeneratorS, self).__init__()

        self.layers = t.nn.Sequential(
            t.nn.ConvTranspose2d(in_channels = in_dimensions, out_channels = int(1024 * scale), kernel_size = 3,
                                 stride = 1, padding = 0),
            t.nn.BatchNorm2d(int(1024 * scale)),
            t.nn.ReLU(inplace = True),

            t.nn.ConvTranspose2d(in_channels = int(1024 * scale), out_channels = int(512 * scale), kernel_size = 4,
                                 stride = 2, padding = 0),
            t.nn.BatchNorm2d(int(512 * scale)),
            t.nn.ReLU(inplace = True),

            t.nn.ConvTranspose2d(in_channels = int(512 * scale), out_channels = int(256 * scale), kernel_size = 4,
                                 stride = 2, padding = 1),
            t.nn.BatchNorm2d(int(256 * scale)),
            t.nn.ReLU(inplace = True),

            t.nn.ConvTranspose2d(in_channels = int(256 * scale), out_channels = out_channels, kernel_size = 4,
                                 stride = 2, padding = 1)
        )

        self.final = t.nn.Tanh()

    def forward(self, x):
        return self.final(self.layers(x))