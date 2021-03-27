# -*- coding: utf-8 -*-
"""
This file contains all operations about all configurations in
all data, models, training & testing

Created by Kunhong Yu
Date: 2021/03/02
"""
import torchvision as tv
import torch as t
import os

class Config(object):
    """Define all hyperparameters in data, models, training & testing
    Args :
        --data_name: data set name, default is 'mnist', including 'cifar10'/'cifar100' as well
        --data_dir: data set directory, default is './data/'
        --img_height: image height, default is 32
        --img_width: image height, default is 32
        --img_channels: image channels, default is 3

        --noise_dimension: input noise's dimension, default is 100
        --noise_distribution: 'gaussian' or 'uniform', default is 'gaussian'
                    with mean 0 and 1 as std, 'uniform' with [-1, 1]
        --model_name: default is 'kdforgan'
        --alpha_kdforgan: hyperparameter in KDforGAN model to trade off between
                    teacher generator loss and student generator loss, default is 0.3
        --scale: scale factor for reducing parameters in the net,
                default is 0.2, for each layer, we multiply scale with original channels
                in the teacher model

        --device: training or testing device, tell by if cuda is available
        --epochs: training epochs, default is 100
        --batch_size: training & testing batch size, default is 32
        --plot_interval: how many epochs to visualize results, default is 10
        --save_model_dir: saved model directory, default is './checkpoints/'

        --only_test: default is False, if False, training will be executed
                else test only, but will first check if trained model exists
        --save_res_dir: saving results directory, default is './results/'
        --num_vis: number of visualize images, default is 25, must be complete-squared
    """
    #############
    #    Data   #
    #############
    data_name = 'mnist'
    data_dir = './data/'
    img_height = 32
    img_width = 32
    img_channels = 3

    #############
    #   Models  #
    #############
    noise_dimension = 100
    noise_distribution = 'gaussian'
    model_name = 'kdforgan'
    alpha_kdforgan = 0.3
    scale = 0.2

    #############
    #   Train   #
    #############
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    epochs = 100
    batch_size = 32
    plot_interval = 10
    save_model_dir = './checkpoints/'

    #############
    #   Test    #
    #############
    only_test = False
    save_res_dir = './results/'
    num_vis = 25

    def parse_kwargs(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                print('Object has no', k, 'which will be added!')

            setattr(self, k, v)

        if self.data_name == 'mnist':
            self.transform = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(0.5, 0.5)
            ])

        elif self.data_name == 'cifar10' or self.data_name == 'cifar100':
            self.transform = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(0.5, 0.5)
            ])

        self.save_s_dir = os.path.join(self.model_name,
                            'gen_s_' + self.data_name + '_' +
                             str(self.noise_dimension) + '_' +
                             str(self.noise_distribution) + '_' +
                             str(self.alpha_kdforgan) + '_' +
                             str(self.scale))

        self.save_t_dir = os.path.join(self.model_name,
                                       'gen_t_' + self.data_name + '_' +
                                       str(self.noise_dimension) + '_' +
                                       str(self.noise_distribution) + '_' +
                                       str(self.alpha_kdforgan) + '_' +
                                       str(self.scale))

    def print_kwargs(self):
        for k, _ in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, '......', getattr(self, k))
