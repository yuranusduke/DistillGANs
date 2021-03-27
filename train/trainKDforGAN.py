# -*- coding: utf-8 -*-
"""
This file contains all operations about training compressed GAN
using KD method
From paper: <Compressing GANs using Knowledge Distillation>
We implement JOINT LOSS according to the paper

Created by Kunhong Yu
Date: 2021/03/02
"""
import torch as t
from tqdm import tqdm
from configs import Config
import torchvision as tv
from models_def import kdforgan_def
from utils import generate_noise, visualize_gen, gp, get_gradients, train_teacher_model
import os
import matplotlib.pyplot as plt

opt = Config()
def train_kdforgan(**kwargs):
    """This function is used to train KD for GAN model"""
    opt.parse_kwargs(**kwargs)
    opt.print_kwargs()

    # Step 0 Decide the structure of the model#
    device = opt.device
    # Step 1 Load the data set#
    if opt.data_name == 'mnist':
        train_data = tv.datasets.MNIST(root = opt.data_dir,
                                       download = True,
                                       train = True,
                                       transform = opt.transform)

        train_loader = t.utils.data.DataLoader(train_data,
                                               shuffle = True,
                                               batch_size = opt.batch_size)

        # train teacher model first if it does not exist!
        teacher_path = os.path.join(opt.save_model_dir, 'teacher_models', 'mnist.pth')
        if not os.path.exists(teacher_path):  # train teacher GAN first
            train_teacher_model(data_name = opt.data_name,
                                data_dir = opt.data_dir,
                                transform = opt.transform,
                                device = device,
                                img_channels = opt.img_channels, noise_dimension = opt.noise_dimension, noise_distribution = opt.noise_distribution,
                                batch_size = opt.batch_size,
                                scale = opt.scale, lr = 2e-4, save_res_dir = opt.save_res_dir, model_name = opt.model_name, save_model_dir = opt.save_model_dir,
                                img_height = opt.img_height, img_width = opt.img_width)
        else:
            print('Teacher model already exists!')
        generator_t = t.load(teacher_path)
        generator_t.to(device)
        generator_t.eval()

    elif opt.data_name == 'cifar10':
        train_data = tv.datasets.CIFAR10(root = opt.data_dir,
                                         download = True,
                                         train = True,
                                         transform = opt.transform)

        train_loader = t.utils.data.DataLoader(train_data,
                                               shuffle = True,
                                               batch_size = opt.batch_size)

        # train teacher model first if it does not exist!
        teacher_path = os.path.join(opt.save_model_dir, 'teacher_models', 'cifar10.pth')
        if not os.path.exists(teacher_path):  # train teacher GAN first
            train_teacher_model(data_name = opt.data_name,
                                data_dir = opt.data_dir,
                                transform = opt.transform,
                                device = device,
                                img_channels = opt.img_channels, noise_dimension = opt.noise_dimension, noise_distribution = opt.noise_distribution,
                                batch_size = opt.batch_size,
                                img_height = opt.img_height, img_width = opt.img_width,
                                scale = opt.scale, lr = 2e-4, save_res_dir = opt.save_res_dir, model_name = opt.model_name, save_model_dir = opt.save_model_dir)
        else:
            print('Teacher model already exists!')
        generator_t = t.load(teacher_path)
        generator_t.to(device)
        generator_t.eval()

    else:
        raise Exception('No other data sets!')

    # Step 2 Reshape the inputs#
    # Step 3 Normalize the inputs#
    # Step 4 Initialize parameters#
    # Step 5 Forward propagation(Vectorization/Activation functions)#
    params = {'img_channels' : opt.img_channels,
              'noise_dimension' : opt.noise_dimension,
              'scale' : opt.scale}
    discriminator, generator_s, _ = kdforgan_def(**params)
    discriminator.to(device)
    generator_s.to(device)

    # Step 6 Compute cost#
    '''
    In particular, we use WGAN loss for steady training according to the paper
    However, we still have to define loss function for distillation, i.e. MSE
    '''
    mse_loss = t.nn.MSELoss().to(device)

    # Step 7 Backward propagation(Vectorization/Activation functions gradients)#
    op_dis = t.optim.Adam(filter(lambda x : x.requires_grad, discriminator.parameters()), lr = 2e-4, betas = (0.5, 0.999))
    op_gen = t.optim.Adam(filter(lambda x : x.requires_grad, generator_s.parameters()), lr = 2e-4, betas = (0.5,  0.999))

    # Step 8 Update parameters#
    dis_loss = []
    gen_loss = []
    for epoch in tqdm(range(opt.epochs)):
        print('Epoch %d / %d.' % (epoch + 1, opt.epochs))
        for i, (batch_x, _) in enumerate(train_loader):
            batch_x = batch_x.view(batch_x.size(0), opt.img_channels, opt.img_height, opt.img_width)
            batch_x = batch_x.to(device)

            # 1. train discriminator
            for _ in range(3): # train discrminator/critic more
                op_dis.zero_grad()
                # 1.1 for discriminator
                real_out = discriminator(batch_x)

                # 1.2 for generator in teacher
                noise = generate_noise(noise_distribution = opt.noise_distribution,
                                       noise_dimension = opt.noise_dimension,
                                       batch_size = batch_x.size(0),
                                       device = device)
                noise = noise[:, :, None, None]
                gen_fake_s = generator_s(noise)
                fake_out_s = discriminator(gen_fake_s.detach())

                epsilon = t.rand(batch_x.size(0), 1, 1, 1, requires_grad = True).to(device)
                grads = get_gradients(batch_x, gen_fake_s.detach(), epsilon, discriminator)
                gp_term = gp(grads)
                dis_batch_loss = -t.mean(real_out) + t.mean(fake_out_s) + 10 * gp_term
                dis_batch_loss.backward()
                op_dis.step()

            # 2. train generator
            for p in discriminator.parameters():
                p.requires_grad = False

            op_gen.zero_grad()
            noise = generate_noise(noise_distribution = opt.noise_distribution,
                                   noise_dimension = opt.noise_dimension,
                                   batch_size = batch_x.size(0),
                                   device = device)
            noise = noise[:, :, None, None]
            gen_fake_t = generator_t(noise)

            gen_fake_s = generator_s(noise)
            fake_out_s = discriminator(gen_fake_s)

            gen_batch_loss = -(1 - opt.alpha_kdforgan) * t.mean(fake_out_s) + opt.alpha_kdforgan * mse_loss(gen_fake_s, gen_fake_t.detach())# distillation

            gen_batch_loss.backward()
            op_gen.step()

            for p in discriminator.parameters():
                p.requires_grad = True

            if i % opt.batch_size == 0:
                print('\tBatch %d has discriminator loss : %.2f. & generator loss : %.2f.' \
                      % (i + 1, dis_batch_loss.item(), gen_batch_loss.item()))

                gen_loss.append(gen_batch_loss.item())
                dis_loss.append(dis_batch_loss.item())

        if epoch % opt.plot_interval == 0:
            visualize_gen(generator_t, generator_s,
                          num_vis = opt.num_vis,
                          save_res_dir = os.path.join(opt.save_res_dir, opt.model_name + os.sep),
                          noise_dimension = opt.noise_dimension,
                          noise_distribution = opt.noise_distribution,
                          device = device, tick = opt.data_name + '_' + str(opt.alpha_kdforgan) + '_' + opt.model_name + '_' + str(opt.scale) + '_' + str(epoch + 1))
            generator_t.train()
            generator_s.train()

    print('Training is done!\n')

    t.save(generator_t, os.path.join(opt.save_model_dir, opt.save_t_dir + '.pth'))# we also save teacher model in another form for future use
    t.save(generator_s, os.path.join(opt.save_model_dir, opt.save_s_dir + '.pth'))

    plt.plot(range(len(dis_loss)), dis_loss, label = 'discriminator_loss')
    plt.plot(range(len(gen_loss)), gen_loss, label = 'generator_loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.title('Training losses')
    plt.legend(loc = 'best')
    plt.savefig(os.path.join(opt.save_res_dir, opt.save_s_dir + '_losses.png'))
    plt.close()