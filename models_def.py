# -*- coding: utf-8 -*-
"""
This file contains all models definitions

Created by Kunhong Yu
Date: 2021/03/02
"""
###############################KD For GAN###############################
from models import Discriminator, GeneratorS, GeneratorT

def kdforgan_def(**kwargs):
    """This function is used to define KD for GAN model
    return :
        --discriminator: Discriminator instance
        --generator_s: GeneratorS instance
        --generator_t: GeneratorT instance
    """
    discriminator = Discriminator(in_channels = kwargs['img_channels'])
    generator_s = GeneratorS(in_dimensions = kwargs['noise_dimension'],
                             out_channels = kwargs['img_channels'],
                             scale = kwargs['scale'])
    generator_t = GeneratorT(in_dimensions = kwargs['noise_dimension'],
                             out_channels = kwargs['img_channels'])

    print('Discriminator: \n', discriminator)
    print('GeneratorS : \n', generator_s)
    print('GeneratorT : \n', generator_t)

    return discriminator, generator_s, generator_t

###############################LeNet5 for MNIST###############################
from models import LeNet5

def LeNet5_def(**kwargs):
    """This function is used to define LeNet5 model
    return :
        --lenet5: LeNet5 model instance
    """
    lenet5 = LeNet5(**kwargs)

    print('LeNet5 model : \n', lenet5)

    return lenet5

###############################VGG16 for CIFAR10###############################
from models import VGG16

def VGG16_def(**kwargs):
    """This function is used to define VGG16 model
    return :
        --vgg16: VGG16 model instance
    """
    vgg16 = VGG16(**kwargs)

    print('VGG16 model : \n', vgg16)

    return vgg16