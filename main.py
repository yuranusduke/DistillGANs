# -*- coding: utf-8 -*-
"""
This file contains all operations about main configurations

Created by Kunhong Yu
Date: 2021/03/08
"""
import fire
from train_def import *
from test_def import *

def main(**kwargs):
    if not kwargs['only_test']:
        #train & test
        if kwargs['model_name'] == 'kdforgan':
            train_kdforgan_def(**kwargs)
            test_kdforgan_def(**kwargs)

        else:
            raise Exception('No other models!')

    else:
        #test only
        if kwargs['model_name'] == 'kdforgan':
            test_kdforgan_def(**kwargs)

        else:
            raise Exception('No other models!')


if __name__ == '__main__':
    """
    USAGE : 
        python main.py main --data_name='mnist' --data_dir='./data/' --img_height=28 --img_width=28 --img_channels=1 --noise_dimension=100 --noise_distribution='gaussian' --model_name='kdforgan' --alpha_kdforgan=0.3 --scale=0.2 --epochs=2 --batch_size=64 --plot_interval=10 --save_model_dir='./checkpoints/' --only_test=False --save_res_dir='./results/' --num_vis=25
    """
    fire.Fire()
