# -*- coding: utf-8 -*-
"""
This file contains all operations about testing KD for GAN model

Created by Kunhong Yu
Date: 2021/03/03
"""
import torch as t
from configs import Config
import torchvision as tv
import os
import json
import numpy as np
import sys
from utils import generate_noise, visualize_gen, evaluation, train_eval_model

opt = Config()
def test_kdforgan(**kwargs):
    """This function is used to test KD for GAN model"""
    opt.parse_kwargs(**kwargs)
    opt.print_kwargs()

    device = opt.device

    # 1. We load test data
    if opt.data_name == 'mnist':
        test_data = tv.datasets.MNIST(root = opt.data_dir,
                                      download = True,
                                      train = False,
                                      transform = opt.transform)
        test_loader = t.utils.data.DataLoader(test_data,
                                              shuffle = False,
                                              batch_size = opt.batch_size)

        # train eval model first if it does not exist!
        eval_path = os.path.join(opt.save_model_dir, 'eval_models', 'mnist.pth')
        if not os.path.exists(eval_path):  # train lenet5 first
            train_eval_model(data_name = opt.data_name,
                             data_dir = opt.data_dir,
                             transform = opt.transform,
                             device = device, save_model_dir = opt.save_model_dir, img_channels = opt.img_channels,
                             img_height = opt.img_height, img_width = opt.img_width)
        else:
            print('Eval model already exists!')
        eval_model = t.load(eval_path)
        eval_model.to(device)
        eval_model.eval()

    elif opt.data_name == 'cifar10':
        test_data = tv.datasets.CIFAR10(root = opt.data_dir,
                                        download = True,
                                        train = False,
                                        transform = opt.transform)
        test_loader = t.utils.data.DataLoader(test_data,
                                              shuffle = False,
                                              batch_size = opt.batch_size)

        # train eval model first if it does not exist!
        eval_path = os.path.join(opt.save_model_dir, 'eval_models', 'cifar10.pth')
        if not os.path.exists(eval_path):  # train vgg16 first
            train_eval_model(data_name = opt.data_name,
                             data_dir = opt.data_dir,
                             transform = opt.transform,
                             device = device, save_model_dir = opt.save_model_dir, img_channels = opt.img_channels,
                             img_height = opt.img_height, img_width = opt.img_width)
        else:
            print('Eval model already exists!')
        eval_model = t.load(eval_path)
        eval_model.to(device)
        eval_model.eval()

    else:
        raise Exception('No other data sets!')

    # 2. We load trained model
    gen_s = t.load(os.path.join(opt.save_model_dir, opt.save_s_dir + '.pth'))
    gen_s.to(device)
    gen_t = t.load(os.path.join(opt.save_model_dir, opt.save_t_dir + '.pth'))
    gen_t.to(device)

    # 3. We do test
    visualize_gen(gen_t, gen_s,
                  num_vis = opt.num_vis,
                  save_res_dir = os.path.join(opt.save_res_dir, opt.model_name + os.sep),
                  noise_dimension = opt.noise_dimension,
                  noise_distribution = opt.noise_distribution,
                  device = device, tick = 'test_kdforgan' + opt.data_name + '_' + str(opt.alpha_kdforgan) + '_' + opt.model_name + '_' + str(opt.scale))

    # We need to calculate final score
    fids = []
    ists = []
    isses = []
    for i, (batch_x, _) in enumerate(test_loader):
        sys.stdout.write('\r>>Testing batch %d.' % (i + 1))
        sys.stdout.flush()
        batch_x = batch_x.view(batch_x.size(0), opt.img_channels, opt.img_height, opt.img_width)

        noise = generate_noise(noise_distribution = opt.noise_distribution,
                               noise_dimension = opt.noise_dimension,
                               batch_size = batch_x.size(0), device = device)
        noise = noise[:, :, None, None]

        fake_t = gen_t(noise)
        fake_s = gen_s(noise)
        fid, ist, iss = evaluation(fake_t, fake_s, eval_model, opt.data_name)
        fids.append(fid)
        ists.append(ist)
        isses.append(iss)

    final_fid = np.mean(fids)
    final_ist = np.mean(ists)
    final_iss = np.mean(isses)

    result = {'scale' : opt.scale, 'fid' : final_fid, 'ist' : final_ist , 'iss' : final_iss}
    json_str = json.dumps(str(result) + '\n')
    with open(os.path.join('./results/', opt.model_name, opt.data_name + '_scores.json'), 'a+') as f:
        f.write(json_str)

    print('\n\nDone!\n')