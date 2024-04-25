#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy
from PIL import Image

import os

import pytorchwt_haar as tf
import torch
import torch.nn as nn
import math
import math
import unet.unet as unet
import unet.red_net as red_net
import scipy.misc
from scipy import signal

L1 = 4
L2 = 4
L3 = 2
L4 = 2


class Framelets(nn.Module):
    def __init__(self, in_channels=1, num_features=64, stride=1, padding=1):
        super(Framelets, self).__init__()
        # stage 0

        self.stage_0_0 = self.conv_bn_relu(num_features_in=in_channels, num_features_out=num_features)
        self.stage_0_1 = self.conv_bn_relu(num_features_in=num_features, num_features_out=num_features)
        if L1 == 2:
            self.stage_0_wt = tf.pytorchwt2(feature_num=num_features)
        else:
            self.stage_0_wt = tf.pytorchwt(feature_num=num_features)
        self.stage_0_HH_0 = self.conv_bn_relu(num_features_in=num_features, num_features_out=num_features * 2)
        self.stage_0_HH_1 = self.conv_bn_relu(num_features_in=num_features * 2, num_features_out=num_features)
        self.stage_0_HL_0 = self.conv_bn_relu(num_features_in=num_features, num_features_out=num_features * 2)
        self.stage_0_HL_1 = self.conv_bn_relu(num_features_in=num_features * 2, num_features_out=num_features)
        self.stage_0_LH_0 = self.conv_bn_relu(num_features_in=num_features, num_features_out=num_features * 2)
        self.stage_0_LH_1 = self.conv_bn_relu(num_features_in=num_features * 2, num_features_out=num_features)

        # stage 1
        self.stage_1_0 = self.conv_bn_relu(num_features_in=num_features, num_features_out=num_features * 2)
        self.stage_1_1 = self.conv_bn_relu(num_features_in=num_features * 2, num_features_out=num_features * 2)

        if L2 == 2:
            self.stage_1_wt = tf.pytorchwt2(feature_num=num_features * 2)
        else:
            self.stage_1_wt = tf.pytorchwt(feature_num=num_features * 2)
        self.stage_1_HH_0 = self.conv_bn_relu(num_features_in=num_features * 2, num_features_out=num_features * 4)
        self.stage_1_HH_1 = self.conv_bn_relu(num_features_in=num_features * 4, num_features_out=num_features * 2)
        self.stage_1_HL_0 = self.conv_bn_relu(num_features_in=num_features * 2, num_features_out=num_features * 4)
        self.stage_1_HL_1 = self.conv_bn_relu(num_features_in=num_features * 4, num_features_out=num_features * 2)
        self.stage_1_LH_0 = self.conv_bn_relu(num_features_in=num_features * 2, num_features_out=num_features * 4)
        self.stage_1_LH_1 = self.conv_bn_relu(num_features_in=num_features * 4, num_features_out=num_features * 2)

        # stage 2
        self.stage_2_0 = self.conv_bn_relu(num_features_in=num_features * 2, num_features_out=num_features * 4)
        self.stage_2_1 = self.conv_bn_relu(num_features_in=num_features * 4, num_features_out=num_features * 4)

        if L3 == 2:
            self.stage_2_wt = tf.pytorchwt2(feature_num=num_features * 4)
        else:
            self.stage_2_wt = tf.pytorchwt(feature_num=num_features * 4)
        self.stage_2_HH_0 = self.conv_bn_relu(num_features_in=num_features * 4, num_features_out=num_features * 8)
        self.stage_2_HH_1 = self.conv_bn_relu(num_features_in=num_features * 8, num_features_out=num_features * 4)
        self.stage_2_HL_0 = self.conv_bn_relu(num_features_in=num_features * 4, num_features_out=num_features * 8)
        self.stage_2_HL_1 = self.conv_bn_relu(num_features_in=num_features * 8, num_features_out=num_features * 4)
        self.stage_2_LH_0 = self.conv_bn_relu(num_features_in=num_features * 4, num_features_out=num_features * 8)
        self.stage_2_LH_1 = self.conv_bn_relu(num_features_in=num_features * 8, num_features_out=num_features * 4)

        # stage 3
        self.stage_3_0 = self.conv_bn_relu(num_features_in=num_features * 4, num_features_out=num_features * 8)
        self.stage_3_1 = self.conv_bn_relu(num_features_in=num_features * 8, num_features_out=num_features * 8)
        if L4 == 2:
            self.stage_3_wt = tf.pytorchwt2(feature_num=num_features * 8)
        else:
            self.stage_3_wt = tf.pytorchwt(feature_num=num_features * 8)
        self.stage_3_HH_0 = self.conv_bn_relu(num_features_in=num_features * 8, num_features_out=num_features * 16)
        self.stage_3_HH_1 = self.conv_bn_relu(num_features_in=num_features * 16, num_features_out=num_features * 8)
        self.stage_3_HL_0 = self.conv_bn_relu(num_features_in=num_features * 8, num_features_out=num_features * 16)
        self.stage_3_HL_1 = self.conv_bn_relu(num_features_in=num_features * 16, num_features_out=num_features * 8)
        self.stage_3_LH_0 = self.conv_bn_relu(num_features_in=num_features * 8, num_features_out=num_features * 16)
        self.stage_3_LH_1 = self.conv_bn_relu(num_features_in=num_features * 16, num_features_out=num_features * 8)
        self.stage_3_LL_0 = self.conv_bn_relu(num_features_in=num_features * 8, num_features_out=num_features * 16)
        self.stage_3_LL_1 = self.conv_bn_relu(num_features_in=num_features * 16, num_features_out=num_features * 8)

        # reconstruction
        self.stage_3_reconstruction_0 = self.conv_bn_relu(num_features_in=num_features * 8,
                                                          num_features_out=num_features * 8)
        self.stage_3_reconstruction_1 = self.conv_bn_relu(num_features_in=num_features * 8,
                                                          num_features_out=num_features * 4)

        self.stage_2_reconstruction_0 = self.conv_bn_relu(num_features_in=num_features * 4,
                                                          num_features_out=num_features * 4)
        self.stage_2_reconstruction_1 = self.conv_bn_relu(num_features_in=num_features * 4,
                                                          num_features_out=num_features * 2)

        self.stage_1_reconstruction_0 = self.conv_bn_relu(num_features_in=num_features * 2,
                                                          num_features_out=num_features * 2)
        self.stage_1_reconstruction_1 = self.conv_bn_relu(num_features_in=num_features * 2,
                                                          num_features_out=num_features)

        self.stage_0_reconstruction_0 = self.conv_bn_relu(num_features_in=num_features, num_features_out=num_features)
        self.stage_0_reconstruction_1 = self.conv_bn_relu(num_features_in=num_features, num_features_out=num_features)

        self.reconstruction_output = nn.Conv2d(num_features, in_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def conv_bn_relu(self, num_features_in, num_features_out):
        layers = []
        layers.append(nn.Conv2d(num_features_in, num_features_out, kernel_size=3, stride=1, padding=1, bias=True))
        layers.append(nn.BatchNorm2d(num_features_out, momentum=0.5))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x, target=None):
        # stage 0
        stage_0_0 = self.stage_0_0(x)
        stage_0_1 = self.stage_0_1(stage_0_0)
        # stage_0_transform = self.stage_0_wt.wavelet_n_dec(stage_0_1)
        # stage_0_LL = stage_0_transform[0]
        # [stage_0_LH, stage_0_HL, stage_0_HH] = stage_0_transform[-1]
        if L1 == 2:
            [stage_0_LL, stage_0_LH, stage_0_HL, stage_0_HH] = self.stage_0_wt.wavelet_dec2(stage_0_1)
        else:
            [stage_0_LL, stage_0_LH, stage_0_HL, stage_0_HH] = self.stage_0_wt.wavelet_dec(stage_0_1)
        stage_0_LH_0 = self.stage_0_LH_0(stage_0_LH)
        stage_0_LH_1 = self.stage_0_LH_1(stage_0_LH_0)
        stage_0_HL_0 = self.stage_0_HL_0(stage_0_HL)
        stage_0_HL_1 = self.stage_0_HL_1(stage_0_HL_0)
        stage_0_HH_0 = self.stage_0_HH_0(stage_0_HH)
        stage_0_HH_1 = self.stage_0_HH_1(stage_0_HH_0)

        # stage 1
        stage_1_0 = self.stage_1_0(stage_0_LL)
        stage_1_1 = self.stage_1_1(stage_1_0)
        # stage_1_transform = self.stage_1_wt.wavelet_n_dec(stage_1_1)
        # stage_1_LL = stage_1_transform[0]
        # [stage_1_LH, stage_1_HL, stage_1_HH] = stage_1_transform[-1]
        if L2 == 2:
            [stage_1_LL, stage_1_LH, stage_1_HL, stage_1_HH] = self.stage_1_wt.wavelet_dec2(stage_1_1)
        else:
            [stage_1_LL, stage_1_LH, stage_1_HL, stage_1_HH] = self.stage_1_wt.wavelet_dec(stage_1_1)
        stage_1_LH_0 = self.stage_1_LH_0(stage_1_LH)
        stage_1_LH_1 = self.stage_1_LH_1(stage_1_LH_0)
        stage_1_HL_0 = self.stage_1_HL_0(stage_1_HL)
        stage_1_HL_1 = self.stage_1_HL_1(stage_1_HL_0)
        stage_1_HH_0 = self.stage_1_HH_0(stage_1_HH)
        stage_1_HH_1 = self.stage_1_HH_1(stage_1_HH_0)

        # stage 2
        stage_2_0 = self.stage_2_0(stage_1_LL)
        stage_2_1 = self.stage_2_1(stage_2_0)
        # stage_2_transform = self.stage_2_wt.wavelet_n_dec(stage_2_1)
        # stage_2_LL = stage_2_transform[0]
        # [stage_2_LH, stage_2_HL, stage_2_HH] = stage_2_transform[-1]
        if L3 == 2:
            [stage_2_LL, stage_2_LH, stage_2_HL, stage_2_HH] = self.stage_2_wt.wavelet_dec2(stage_2_1)
        else:
            [stage_2_LL, stage_2_LH, stage_2_HL, stage_2_HH] = self.stage_2_wt.wavelet_dec(stage_2_1)
        stage_2_LH_0 = self.stage_2_LH_0(stage_2_LH)
        stage_2_LH_1 = self.stage_2_LH_1(stage_2_LH_0)
        stage_2_HL_0 = self.stage_2_HL_0(stage_2_HL)
        stage_2_HL_1 = self.stage_2_HL_1(stage_2_HL_0)
        stage_2_HH_0 = self.stage_2_HH_0(stage_2_HH)
        stage_2_HH_1 = self.stage_2_HH_1(stage_2_HH_0)

        # stage 3
        stage_3_0 = self.stage_3_0(stage_2_LL)
        stage_3_1 = self.stage_3_1(stage_3_0)
        # stage_3_transform = self.stage_3_wt.wavelet_n_dec(stage_3_1)
        # stage_3_LL = stage_3_transform[0]
        # [stage_3_LH, stage_3_HL, stage_3_HH] = stage_3_transform[-1]
        if L4 == 2:
            [stage_3_LL, stage_3_LH, stage_3_HL, stage_3_HH] = self.stage_3_wt.wavelet_dec2(stage_3_1)
        else:
            [stage_3_LL, stage_3_LH, stage_3_HL, stage_3_HH] = self.stage_3_wt.wavelet_dec(stage_3_1)
        stage_3_LH_0 = self.stage_3_LH_0(stage_3_LH)
        stage_3_LH_1 = self.stage_3_LH_1(stage_3_LH_0)
        stage_3_HL_0 = self.stage_3_HL_0(stage_3_HL)
        stage_3_HL_1 = self.stage_3_HL_1(stage_3_HL_0)
        stage_3_HH_0 = self.stage_3_HH_0(stage_3_HH)
        stage_3_HH_1 = self.stage_3_HH_1(stage_3_HH_0)
        stage_3_LL_0 = self.stage_3_LL_0(stage_3_LL)
        stage_3_LL_1 = self.stage_3_LL_1(stage_3_LL_0)

        # reconstruction
        if L4 == 2:
            stage_3_reconstruction = self.stage_3_wt.wavelet_rec2(stage_3_LL_1, stage_3_LH_1, stage_3_HL_1,
                                                                  stage_3_HH_1)
        else:
            stage_3_reconstruction = self.stage_3_wt.wavelet_rec(stage_3_LL_1, stage_3_LH_1, stage_3_HL_1,
                                                                 stage_3_HH_1)
        # stage_3_transform[0] = stage_3_LL_1
        # stage_3_reconstruction = self.stage_3_wt.wavelet_n_rec(stage_3_transform)
        stage_3_LL_reconstruction_0 = self.stage_3_reconstruction_0(stage_3_reconstruction + stage_3_1)
        stage_3_LL_reconstruction_1 = self.stage_3_reconstruction_1(stage_3_LL_reconstruction_0)

        if L3 == 2:
            stage_2_reconstruction = self.stage_2_wt.wavelet_rec2(stage_3_LL_reconstruction_1, stage_2_LH_1,
                                                                  stage_2_HL_1,
                                                                  stage_2_HH_1)
        else:
            stage_2_reconstruction = self.stage_2_wt.wavelet_rec(stage_3_LL_reconstruction_1, stage_2_LH_1,
                                                                 stage_2_HL_1,
                                                                 stage_2_HH_1)
        # stage_2_transform[0] = stage_2_LL_1
        # stage_2_reconstruction = self.stage_2_wt.wavelet_n_rec(stage_2_transform)
        stage_2_LL_reconstruction_0 = self.stage_2_reconstruction_0(stage_2_reconstruction + stage_2_1)
        stage_2_LL_reconstruction_1 = self.stage_2_reconstruction_1(stage_2_LL_reconstruction_0)

        if L2 == 2:
            stage_1_reconstruction = self.stage_1_wt.wavelet_rec2(stage_2_LL_reconstruction_1, stage_1_LH_1,
                                                                  stage_1_HL_1,
                                                                  stage_1_HH_1)
        else:
            stage_1_reconstruction = self.stage_1_wt.wavelet_rec(stage_2_LL_reconstruction_1, stage_1_LH_1,
                                                                 stage_1_HL_1,
                                                                 stage_1_HH_1)
        # stage_1_transform[0] = stage_1_LL_1
        # stage_1_reconstruction = self.stage_1_wt.wavelet_n_rec(stage_1_transform)
        stage_1_LL_reconstruction_0 = self.stage_1_reconstruction_0(stage_1_reconstruction + stage_1_1)
        stage_1_LL_reconstruction_1 = self.stage_1_reconstruction_1(stage_1_LL_reconstruction_0)

        if L1 == 2:
            stage_0_reconstruction = self.stage_0_wt.wavelet_rec2(stage_1_LL_reconstruction_1, stage_0_LH_1,
                                                                  stage_0_HL_1,
                                                                  stage_0_HH_1)
        else:
            stage_0_reconstruction = self.stage_0_wt.wavelet_rec(stage_1_LL_reconstruction_1, stage_0_LH_1,
                                                                 stage_0_HL_1,
                                                                 stage_0_HH_1)
        # stage_0_transform[0] = stage_0_LL_1
        # stage_0_reconstruction = self.stage_0_wt.wavelet_n_rec(stage_0_transform)
        stage_0_LL_reconstruction_0 = self.stage_0_reconstruction_0(stage_0_reconstruction + stage_0_1)
        stage_0_LL_reconstruction_1 = self.stage_0_reconstruction_1(stage_0_LL_reconstruction_0)

        # out = self.reconstruction_output(stage_0_LL_reconstruction_1) + x
        return self.reconstruction_output(stage_0_LL_reconstruction_1) + x

        if target is not None:
            pairs = {'out': (out, target)}
            return pairs, self.exports(x, out, target)
        else:
            return self.exports(x, out, target)

    def exports(self, x, output, target):
        result = {'input': x, 'output': output}
        if target is not None:
            result['target'] = target
        return result


def weights_init(self):
    for idx, m in enumerate(self.modules()):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


def psnr(img1, img2):
    mse = np.sum((img1 - img2) ** 2)
    mse = (1 / (float(img1.shape[2] * img1.shape[3]))) * mse
    max_i = 255.0
    return 20 * math.log10(max_i / math.sqrt(mse))
    # return 10 * math.log10(max_i / mse)


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = numpy.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = numpy.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def ssim(img1, img2, cs_map=False):
    img1 = img1.astype(numpy.float32)
    img2 = img2.astype(numpy.float32)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255  # bitdepth of image
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(window, img1 * img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2 * img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1 * img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                             (sigma1_sq + sigma2_sq + C2)),
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))


class Images_class(object):
    def __init__(self, master=None, noise=None, name=None, psnr=None, ssim=None, index=None, best_denoise=None):
        self.master = master
        self.noise = noise
        self.name = name
        self.psnr = psnr
        self.ssim = ssim
        self.index = index
        self.best_denoise = best_denoise


def noisy(image):
    row, col = image.shape
    mean = 0
    sigma = 30
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    return image + gauss


def create_mini_batches(_i, batch_size):
    m = []
    items = []
    for i in range(len(_i)):
        items.append(i)
    np.random.shuffle(items)
    for i in range(batch_size):
        m.append(Images_class(_i[items[i]].master,
                              Variable(torch.from_numpy(np.expand_dims(np.expand_dims(noisy(_i[items[i]].noise), axis=0)
                                                                       , axis=0)).float())
                              , _i[items[i]].name
                              , _i[items[i]].psnr
                              , _i[items[i]].ssim
                              , _i[items[i]].index))
    return m


# function to create a list containing mini-batches
def create_mini_batches2(data, batch_size):
    mini_batches = []
    np.random.shuffle(data)
    n_minibatches = len(data) // batch_size
    i = 0

    for i in range(n_minibatches + 1):
        mini_batch = data[i * batch_size:(i + 1) * batch_size]
        X_mini = mini_batch[:]
        mini_batches.append((X_mini))
    if len(data) % batch_size != 0:
        mini_batch = data[i * batch_size]
        X_mini = mini_batch[:]
        mini_batches.append((X_mini))
    return mini_batches


if __name__ == '__main__':
    import numpy as np
    from torch.autograd import Variable
    from skimage.measure import compare_ssim

    xx=512
    ImagesList = []
    for i in range(19):
        im = Image.open('data/' + '{0:02d}'.format(i + 1) + '.jpg').convert('LA')
        k = 0
        for ii in range(6):
            for jj in range(3):
                new_im = Image.new('I', (xx, xx), 'black')
                new_im.paste(im.crop(((ii * xx) + 200, (jj * xx) + 80, ((ii + 1) * xx) +200, ((jj + 1) * xx) + 80)))
                ImagesList.append(Images_class(
                    Variable(
                        torch.from_numpy(np.expand_dims(np.expand_dims(np.array(new_im), axis=0), axis=0)).float()),
                    np.array(new_im), 'data/out/' + '{0:02d}'.format(i + 1)
                                      + '_0.jpg', 0, 0, i + (k*19)))
                k = k + 1

    model = Framelets(in_channels=1, num_features=128)

    model.cuda()

    learning_rate = 0.0001

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)

    loss_list = []

    for epoch in range(200):

        mini_batches = create_mini_batches(ImagesList, 19 * 3*6)
        for mini_batch in mini_batches:
            optimizer.zero_grad()
            outputs = model(mini_batch.noise.cuda())
            loss = criterion(outputs, mini_batch.master.cuda())
            loss.backward()
            optimizer.step()
            mini_batch.noise.cpu()
            mini_batch.master.cpu()

            _p = psnr(outputs.cpu().detach().numpy(), mini_batch.master.detach().numpy())


            ImagesList[mini_batch.index].psnr = _p

        print("%8.3f" % (
            np.mean([c.psnr for c in ImagesList])))

        if epoch > 25 and (epoch % 25) == 0 and (learning_rate / 2) >= 0.00001:
            learning_rate = learning_rate / 2
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    torch.save(model.cpu().state_dict(), 'f_haar_4422.p')
