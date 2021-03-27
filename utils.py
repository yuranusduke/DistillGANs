# -*- coding: utf-8 -*-
"""
This file contains all utilities functions

Created by Kunhong Yu
Date: 2021/03/02
"""
import torch as t
import matplotlib.pyplot as plt
import torchvision as tv
import math
import os
import numpy as np
from torch.nn import functional as F
from scipy.stats import entropy
from scipy import linalg

########################################
#          Generate noise              #
########################################
def generate_noise(noise_distribution, noise_dimension, batch_size, device):
    """
    This function is used to generate noise distributions
    Args :
        --noise_distribution: 'gaussian' or 'uniform'
        --noise_dimension: default is 100
        --batch_size
        --device
    return :
        --noise: generated noise
    """
    if noise_distribution == 'gaussian':
        noise = t.randn(batch_size, noise_dimension, device = device, requires_grad = True)

    elif noise_distribution == 'uniform':
        noise = t.rand(batch_size, noise_dimension, device = device, requires_grad = True) * (-2) + 1  # [-1, 1]

    else:
        raise Exception('No other noise distributions!')

    return noise

########################################
#          Visualize results           #
########################################
def visualize_gen(*models, num_vis, save_res_dir, noise_dimension,
                  noise_distribution, device, tick):
    """This function is used to visualized generated results
    Args :
        --num_vis: number of visualized images
        --save_res_dir: saved results directory
        --noise_distribution: 'gaussian' or 'uniform'
        --noise_dimension: default is 100
        --device
        --tick: tick for saving results
        --*models: [teacher, student]
    """
    assert math.sqrt(num_vis) ** 2 == num_vis

    if len(models) == 2: # teacher and student
        gen_t, gen_s = models[0], models[1]
        gen_t.eval()
        gen_s.eval()
        with t.no_grad():
            noise = generate_noise(noise_distribution, noise_dimension, num_vis, device)
            noise = noise[:, :, None, None]
            fake_t = gen_t(noise).detach()
            fake_s = gen_s(noise).detach()

            each_row = int(math.sqrt(num_vis))
            fake_t = fake_t.view(each_row, each_row, *fake_t.shape[1:])
            fake_s = fake_s.view(each_row, each_row, *fake_s.shape[1:])
            fake = t.cat((fake_t, fake_s), dim = 1)
            fake = fake.data.cpu()
            fake = fake.view(-1, *fake.shape[2:])
            grid = tv.utils.make_grid(fake, nrow = each_row * 2)
            plt.imshow(grid.permute(1, 2, 0).squeeze())
            plt.axis('off')
            plt.title('Left : teacher || Right : student')

        plt.savefig(save_res_dir + tick + '.png', dpi = 300)
        plt.close()

    else: # only teacher
        gen_t = models[0]
        gen_t.eval()

        with t.no_grad():
            noise = generate_noise(noise_distribution, noise_dimension, num_vis, device)
            noise = noise[:, :, None, None]
            fake_t = gen_t(noise).detach()

            each_row = int(math.sqrt(num_vis))
            fake = fake_t.data.cpu()
            grid = tv.utils.make_grid(fake, nrow = each_row)
            plt.imshow(grid.permute(1, 2, 0).squeeze())
            plt.axis('off')
            plt.title('Teacher')

########################################
#             Evaluation               #
########################################
def evaluation(fake_t, fake_s, eval_model, data_name):
    """
    This function is used to evaluate the model using outputs of teacher model
    and student model, according to the paper, we use FID score and Inception Score(IS)
    Inspired from :
    https://github.com/xml94/open/blob/master/compute_FID_for_GAN
    Args :
        --fake_t: fake teacher output
        --fake_s: fake student output
        --eval_model: we use inception v3 as default with cutting output layer
        --data_name: data set name
    return :
        --fid: FID score
        --ist: teacher inception score
        --iss: student inception score
    """
    eval_model.eval()
    def _get_is(fake_t, fake_s, eval_model, data_name):
        """This inside function is used to get inception score
         Args :
            --fake_t: fake teacher output
            --fake_s: fake student output
            --eval_model: we use inception v3 as default with cutting output layer
            --data_name: data set name
        return :
            --ist: teacher inception score
            --iss: student inception score
        """
        with t.no_grad():

            t_out, _ = eval_model(fake_t)
            s_out, _ = eval_model(fake_s)

            t_out = F.softmax(t_out, dim = 1).data.cpu().numpy()
            s_out = F.softmax(s_out, dim = 1).data.cpu().numpy()

            # for teacher
            scores = []
            pyt = np.mean(t_out, axis = 0)
            for q in range(t_out.shape[0]):
                pyx = t_out[q, :]
                scores.append(entropy(pyx, pyt))

            ist = np.exp(np.mean(scores))

            # for student
            scores = []
            pys = np.mean(s_out, axis = 0)
            for q in range(s_out.shape[0]):
                pyx = s_out[q, :]
                scores.append(entropy(pyx, pys))

            iss = np.exp(np.mean(scores))

        return ist, iss

    def _get_fid(fake_t, fake_s, eval_model, data_name):
        """This inside function is used to get FID score
        NOTE: in this repo, we only calculate FID between
        generated images from teacher and generated images from student,
        we don't consider real images
        Args :
            --fake_t: fake teacher output
            --fake_s: fake student output
            --eval_model: we use inception v3 as default with cutting output layer
            --data_name: data set name
        return :
            --fid
        """
        model = eval_model
        _, t_out = model(fake_t)
        t_out = t_out.cpu().data.numpy().reshape(fake_t.size(0), -1)
        _, s_out = model(fake_s)
        s_out = s_out.cpu().data.numpy().reshape(fake_s.size(0), -1)
        with t.no_grad():
            t_mu = np.mean(t_out, axis = 0, keepdims = True)
            t_sigma = np.cov(t_out, rowvar = False)

            s_mu = np.mean(s_out, axis = 0, keepdims = True)
            s_sigma = np.cov(s_out, rowvar = False)

            diff = t_mu - s_mu
            t_sigma = np.atleast_2d(t_sigma)
            s_sigma = np.atleast_2d(s_sigma)
            covmean, _ = linalg.sqrtm(t_sigma.dot(s_sigma), disp = False)

            if not np.isfinite(covmean).all():
                eps = 1e-6
                msg = ('fid calculation produces singular product; '
                       'adding %s to diagonal of cov estimates') % eps
                print(msg)
                offset = np.eye(t_sigma.shape[0]) * eps
                covmean = linalg.sqrtm((t_sigma + offset).dot(s_sigma + offset))

            # Numerical error might give slight imaginary component
            if np.iscomplexobj(covmean):
                if not np.allclose(np.diagonal(covmean).imag, 0, atol = 1e-3):
                    m = np.max(np.abs(covmean.imag))
                    #raise ValueError('Imaginary component {}'.format(m))
                covmean = covmean.real

            tr_covmean = np.trace(covmean)

            fid = diff.dot(diff.T) + np.trace(t_sigma) + np.trace(s_sigma) - 2 * tr_covmean

        return fid

    ist, iss = _get_is(fake_t, fake_s, eval_model, data_name)
    fid = _get_fid(fake_t, fake_s, eval_model, data_name)# we choose the last second layer

    return fid, ist, iss

########################################
#          Train eval models           #
########################################
def train_eval_model(data_name, data_dir, transform, device, save_model_dir, img_channels, img_height, img_width):
    """This function is used to train evaluation model in the naive way
    Args :
        --data_name: data set name
        --data_dir: data set directory
        --transform: data preprocessing
        --device: learning device
        --save_model_dir: model saving directory
    """
    if data_name == 'mnist':
        # Step 0 Decide the structure of the model
        # Step 1 Load the data set
        from models_def import LeNet5_def

        dataset = tv.datasets.MNIST(root = data_dir,
                                    download = True,
                                    train = True,
                                    transform = transform)
        dataloader = t.utils.data.DataLoader(dataset,
                                             shuffle = True, batch_size = 32)

        # Step 2 Reshape the inputs
        # Step 3 Normalize the inputs
        # Step 4 Initialize parameters
        # Step 5 Forward propagation(Vectorization/Activation functions)
        model = LeNet5_def()

    elif data_name == 'cifar10':
        # Step 0 Decide the structure of the model
        # Step 1 Load the data set
        from models_def import VGG16_def

        dataset = tv.datasets.CIFAR10(root = data_dir,
                                      download = True,
                                      train = True,
                                      transform = transform)
        dataloader = t.utils.data.DataLoader(dataset,
                                             shuffle = True, batch_size = 32)

        # Step 2 Reshape the inputs
        # Step 3 Normalize the inputs
        # Step 4 Initialize parameters
        # Step 5 Forward propagation(Vectorization/Activation functions)
        model = VGG16_def(num_classes = 10)

    else:
        raise Exception('No other data sets!')

    model.to(device)
    # Step 6 Compute cost
    cost = t.nn.CrossEntropyLoss().to(device)
    # Step 7 Backward propagation(Vectorization/Activation functions gradients)
    optimizer = t.optim.Adam(filter(lambda x : x.requires_grad, model.parameters()), amsgrad = True)

    lr_schedule = t.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [20, 40, 50],
                                                   gamma = 0.1)

    # Step 8 Update parameters
    print('First train evaluation models because it does not exist!\n')
    for epoch in range(50):# 20 for MNIST, 50 epochs are enough for CIFAR10
        print('Epoch : %d / %d.' % (epoch + 1, 50))
        for i, (batch_x, batch_y) in enumerate(dataloader):
            batch_x = batch_x.view(batch_x.size(0), img_channels, img_height, img_width)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            out, _ = model(batch_x)
            batch_loss = cost(out, batch_y)
            batch_loss.backward()
            optimizer.step()

            if i % 32 == 0:
                preds = t.argmax(out, dim = 1)
                correct = t.sum(preds == batch_y).float()
                acc = correct / batch_x.size(0)

                print('\tThis batch %d has cost : %.3f. || acc : %.2f%%.' % (i + 1, batch_loss.item(), acc * 100.))
        lr_schedule.step()

    print('Training evaluation model is done!')
    t.save(model, os.path.join(save_model_dir, 'eval_models', 'mnist.pth' if data_name == 'mnist' else 'cifar10.pth'))

########################################
#          Train teacher models        #
########################################
def train_teacher_model(data_name, data_dir, transform, device, save_model_dir, **kwargs):
    """This function is used to train teacher model in the naive way
    Args :
        --data_name: data set name
        --data_dir: data set directory
        --transform: data preproecessing
        --device: learning device
        --save_model_dir: model saving directory
    """
    if data_name == 'mnist':
        # Step 0 Decide the structure of the model
        # Step 1 Load the data set
        from models_def import kdforgan_def

        dataset = tv.datasets.MNIST(root = data_dir,
                                    download = True,
                                    train = True,
                                    transform = transform)
        dataloader = t.utils.data.DataLoader(dataset,
                                             shuffle = True, batch_size = 32)

    elif data_name == 'cifar10':
        from models_def import kdforgan_def

        dataset = tv.datasets.CIFAR10(root = data_dir,
                                      download = True,
                                      train = True,
                                      transform = transform)
        dataloader = t.utils.data.DataLoader(dataset,
                                             shuffle = True, batch_size = 32)

    else:
        raise Exception('No other data sets!')

    # Step 2 Reshape the inputs
    # Step 3 Normalize the inputs
    # Step 4 Initialize parameters
    # Step 5 Forward propagation(Vectorization/Activation functions)
    discriminator, _, generator_t = kdforgan_def(**kwargs)
    discriminator.to(device)
    generator_t.to(device)
    # Step 6 Compute cost
    # Step 7 Backward propagation(Vectorization/Activation functions gradients)
    dis_opt = t.optim.Adam(filter(lambda x : x.requires_grad, discriminator.parameters()), lr = kwargs['lr'], betas = (0.5, 0.999))
    gen_opt = t.optim.Adam(filter(lambda x: x.requires_grad, generator_t.parameters()), lr = kwargs['lr'],
                           betas = (0.5, 0.999))

    # Step 8 Update parameters
    gen_loss = []
    dis_loss = []
    print('Train teacher model first because it does not exist!\n')
    for epoch in range(50):# 50 epochs are enough for MNIST and CIFAR10
        print('Epoch : %d / %d.' % (epoch + 1, 50))
        for i, (batch_x, _) in enumerate(dataloader):
            batch_x = batch_x.view(batch_x.size(0), kwargs['img_channels'], kwargs['img_height'], kwargs['img_width'])
            batch_x = batch_x.to(device)

            # train critic first for 3 times
            for _ in range(5):
                dis_opt.zero_grad()
                # 1.1 for discriminator
                real_out = discriminator(batch_x)

                # 1.2 for generator in teacher
                noise = generate_noise(noise_distribution = kwargs['noise_distribution'],
                                       noise_dimension = kwargs['noise_dimension'],
                                       batch_size = batch_x.size(0),
                                       device = device)
                noise = noise[:, :, None, None]
                gen_fake_t = generator_t(noise)
                fake_out_t = discriminator(gen_fake_t.detach())

                epsilon = t.rand(batch_x.size(0), 1, 1, 1, requires_grad = True).to(device)
                grads = get_gradients(batch_x, gen_fake_t.detach(), epsilon, discriminator)
                gp_term = gp(grads)
                dis_batch_loss = -t.mean(real_out) + t.mean(fake_out_t) + 10 * gp_term
                dis_batch_loss.backward()
                dis_opt.step()

            # 2. train generator
            for p in discriminator.parameters():
                p.requires_grad = False

            gen_opt.zero_grad()
            noise_t = generate_noise(noise_distribution = kwargs['noise_distribution'],
                                     noise_dimension = kwargs['noise_dimension'],
                                     batch_size = batch_x.size(0),
                                     device = device)
            noise_t = noise_t[:, :, None, None]
            gen_fake_t = generator_t(noise_t)
            fake_out_t = discriminator(gen_fake_t)

            gen_batch_loss = -t.mean(fake_out_t)

            gen_batch_loss.backward()
            gen_opt.step()

            for p in discriminator.parameters():
                p.requires_grad = True

            if i % kwargs['batch_size'] == 0:
                print('\tBatch %d has discriminator loss : %.2f. & generator loss : %.2f.' \
                      % (i + 1, dis_batch_loss.item(), gen_batch_loss.item()))

                gen_loss.append(gen_batch_loss.item())
                dis_loss.append(dis_batch_loss.item())

        if epoch % 5 == 0:
            visualize_gen(generator_t, num_vis = 25, save_res_dir = kwargs['save_res_dir'], noise_dimension = kwargs['noise_dimension'],
                          noise_distribution = kwargs['noise_distribution'], device = device, tick = None)
            plt.savefig(kwargs['save_res_dir'] + os.sep + kwargs['model_name'] + os.sep + data_name + '_teacher_in_step_%d' % (epoch + 1) + '.png')
            plt.close()
            generator_t.train()

    print('Training teacher model is done!')
    t.save(generator_t, os.path.join(save_model_dir, 'teacher_models', 'mnist.pth' if data_name == 'mnist' else 'cifar10.pth'))

    plt.plot(range(len(dis_loss)), dis_loss, label = 'discriminator_loss')
    plt.plot(range(len(gen_loss)), gen_loss, label = 'generator_loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.title('Training losses for teacher')
    plt.legend(loc = 'best')
    plt.savefig(os.path.join(kwargs['save_res_dir'], kwargs['model_name'] + os.sep + data_name + '_teacher_losses.png'))
    plt.close()

########################################
#          Gradient penalty            #
########################################
def get_gradients(real, faked, epsilon, cri):
    """This function is used to get gradients of inputs
    Args :
        --real: real image tensors
        --faked: faked image tensors
        --epsilon: a vector for each element in images
        --cri: critic instance
    return :
        --grads: gradient tensors
    """
    inputs = real * epsilon + faked * (1 - epsilon)
    cri_scores = cri(inputs)
    grads = t.autograd.grad(inputs = inputs, outputs = cri_scores,
                            grad_outputs = t.ones_like(cri_scores),
                            create_graph = True,
                            retain_graph = True)[0]

    return grads

def gp(grads):
    """This function is used to get gradient penalty term
    Args :
        --grads: gradients
    return :
        --gp_term
    """
    grads = grads.view(grads.size(0), -1)
    gp_term = t.norm(grads, 2, dim = 1)
    gp_term = t.mean((gp_term - 1.) ** 2)

    return gp_term
