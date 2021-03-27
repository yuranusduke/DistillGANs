# -*- coding: utf-8 -*-
"""
This file contains all operations about training models

Created by Kunhong Yu
Date: 2021/03/02
"""
###############################KD For GAN###############################
from train import train_kdforgan

def train_kdforgan_def(**kwargs):
    """This function is used to define training KD for GAN model"""
    train_kdforgan(**kwargs)

    print('\nTraining is done!\n')