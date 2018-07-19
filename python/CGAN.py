#! /usr/bin/python
# -*- coding: utf-8 -*-

import sys
import numpy
import scipy
import theano
import theano.tensor as T
import struct
import math
import pickle
import time
import os
import chainer
import chainer.functions as F
import chainer.links as L
import sklearn
import sklearn.utils as U
import scipy.io
import h5py 
from chainer.datasets import tuple_dataset
from chainer import training
from chainer.training import extensions
from chainer.cuda import cupy as xp




class Relulayer(chainer.link.Chain): 
    ###A layer with batch normalization using rectified linear unit###
    def __init__(self, in_dim, out_dim, **kwargs):
        super(Relulayer, self).__init__()
        with self.init_scope():
            self.linear = L.Linear(in_dim, out_dim)

    def __call__(self, x):
        return F.leaky_relu(self.linear(x))


class DNN(chainer.Chain):
    ###Multi Network definition###
    def __init__(self, layer_shape):
        super(DNN, self).__init__()
        self._n_layer = len(layer_shape)-1

        for idx1 in range(self._n_layer-1) :
            self.add_link("l{0}".format(idx1 + 1),
            Relulayer(layer_shape[idx1], layer_shape[idx1 + 1]))
        self.add_link("l{0}".format(idx1 + 2),
        L.Linear(layer_shape[idx1 + 1], layer_shape[idx1 + 2]))
 
    def __call__(self, x, y):
        w_h=x
        for idx1 in range(self._n_layer) :
            w_h = self.__getitem__("l{0}".format(idx1 + 1))(w_h)
        loss = F.mean_squared_error(y, w_h)
        chainer.report({'loss': loss}, self)
        return loss

    def convert(self, x):
        w_h=x
        for idx1 in range(self._n_layer) :
            w_h = self.__getitem__("l{0}".format(idx1 + 1))(w_h)
        return w_h


class Unet(chainer.Chain):
    def __init__(self):
        super(Unet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(12,64,3,1,1)
            #self.norm1 = L.BatchNormalization(64)
            self.conv2 = L.Convolution2D(64,128,4,2,1)
            self.norm2 = L.BatchNormalization(128)
            self.conv3 = L.Convolution2D(128,256,4,2,1)
            self.norm3 = L.BatchNormalization(256)
            self.conv4 = L.Convolution2D(256,512,4,2,1)
            self.norm4 = L.BatchNormalization(512)
            self.conv5 = L.Convolution2D(512,512,4,2,1)
            self.norm5 = L.BatchNormalization(512)
            self.conv6 = L.Convolution2D(512,512,4,2,1)
            self.norm6 = L.BatchNormalization(512)
            self.conv7 = L.Convolution2D(512,512,4,2,1)
            self.norm7 = L.BatchNormalization(512)
            self.conv8 = L.Convolution2D(512,512,4,2,1)
            self.norm8 = L.BatchNormalization(512)

            self.deconv1 = L.Deconvolution2D(512,512,4,2,1)
            self.denorm1 = L.BatchNormalization(512)
            self.deconv2 = L.Deconvolution2D(1024,512,4,2,1)
            self.denorm2 = L.BatchNormalization(512)
            self.deconv3 = L.Deconvolution2D(1024,512,4,2,1)
            self.denorm3 = L.BatchNormalization(512)
            self.deconv4 = L.Deconvolution2D(1024,512,4,2,1)
            self.denorm4 = L.BatchNormalization(512)
            self.deconv5 = L.Deconvolution2D(1024,256,4,2,1)
            self.denorm5 = L.BatchNormalization(256)
            self.deconv6 = L.Deconvolution2D(512,128,4,2,1)
            self.denorm6 = L.BatchNormalization(128)
            self.deconv7 = L.Deconvolution2D(256,64,4,2,1)
            self.denorm7 = L.BatchNormalization(64)
            #self.deconv8 = L.Deconvolution2D(128,3,4,2,1)
            #self.denorm8 = L.BatchNormalization(3)
            self.deconv8 = L.Convolution2D(128,3,3,1,1)

    def convert(self, x):
        w_h=x

        w_h1 = F.relu(self.conv1(w_h))
        w_h2 = F.relu(self.norm2(self.conv2(w_h1)))
        w_h3 = F.relu(self.norm3(self.conv3(w_h2)))
        w_h4 = F.relu(self.norm4(self.conv4(w_h3)))
        w_h5 = F.relu(self.norm5(self.conv5(w_h4)))
        w_h6 = F.relu(self.norm6(self.conv6(w_h5)))
        w_h7 = F.relu(self.norm7(self.conv7(w_h6)))
        w_h8 = F.relu(self.norm8(self.conv8(w_h7)))

        w_hd1 = F.leaky_relu(F.dropout(self.denorm1(self.deconv1(w_h8))))
        w_hd1 = F.concat([w_hd1, w_h7])
        w_hd2 = F.leaky_relu(F.dropout(self.denorm2(self.deconv2(w_hd1))))
        w_hd2 = F.concat([w_hd2, w_h6])
        w_hd3 = F.leaky_relu(F.dropout(self.denorm3(self.deconv3(w_hd2))))
        w_hd3 = F.concat([w_hd3, w_h5])

        w_hd4 = F.leaky_relu(self.denorm4(self.deconv4(w_hd3)))
        w_hd4 = F.concat([w_hd4, w_h4])
        w_hd5 = F.leaky_relu(self.denorm5(self.deconv5(w_hd4)))
        w_hd5 = F.concat([w_hd5, w_h3])
        w_hd6 = F.leaky_relu(self.denorm6(self.deconv6(w_hd5)))
        w_hd6 = F.concat([w_hd6, w_h2])
        w_hd7 = F.leaky_relu(self.denorm7(self.deconv7(w_hd6)))
        w_hd7 = F.concat([w_hd7, w_h1])
        #w_hd8 = F.sigmoid(self.denorm8(self.deconv8(w_hd7)))
        #w_hd8 = F.sigmoid(self.deconv8(w_hd7))
        w_hd8 = self.deconv8(w_hd7)

        return w_hd8

 
    def __call__(self, x, y):
        w_h=x

        w_h1 = F.relu(self.conv1(w_h))
        w_h2 = F.relu(self.norm2(self.conv2(w_h1)))
        w_h3 = F.relu(self.norm3(self.conv3(w_h2)))
        w_h4 = F.relu(self.norm4(self.conv4(w_h3)))
        w_h5 = F.relu(self.norm5(self.conv5(w_h4)))
        w_h6 = F.relu(self.norm6(self.conv6(w_h5)))
        w_h7 = F.relu(self.norm7(self.conv7(w_h6)))
        w_h8 = F.relu(self.norm8(self.conv8(w_h7)))


        w_hd1 = F.leaky_relu(F.dropout(self.denorm1(self.deconv1(w_h8))))
        w_hd1 = F.concat([w_hd1, w_h7])
        w_hd2 = F.leaky_relu(F.dropout(self.denorm2(self.deconv2(w_hd1))))
        w_hd2 = F.concat([w_hd2, w_h6])
        w_hd3 = F.leaky_relu(F.dropout(self.denorm3(self.deconv3(w_hd2))))
        w_hd3 = F.concat([w_hd3, w_h5])

        w_hd4 = F.leaky_relu(self.denorm4(self.deconv4(w_hd3)))
        w_hd4 = F.concat([w_hd4, w_h4])
        w_hd5 = F.leaky_relu(self.denorm5(self.deconv5(w_hd4)))
        w_hd5 = F.concat([w_hd5, w_h3])
        w_hd6 = F.leaky_relu(self.denorm6(self.deconv6(w_hd5)))
        w_hd6 = F.concat([w_hd6, w_h2])
        w_hd7 = F.leaky_relu(self.denorm7(self.deconv7(w_hd6)))
        w_hd7 = F.concat([w_hd7, w_h1])
        #w_hd8 = F.sigmoid(self.denorm8(self.deconv8(w_hd7)))
        #w_hd8 = F.sigmoid(self.deconv8(w_hd7))
        w_hd8 = self.deconv8(w_hd7)

        loss = F.mean_squared_error(y, w_hd8)
        chainer.report({'loss': loss}, self)
        return loss



class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(12,32,4,2,1)
            self.conv1_2 = L.Convolution2D(3,32,4,2,1)
            #self.norm1 = L.BatchNormalization(64)
            self.conv2 = L.Convolution2D(64,128,4,2,1)
            self.norm2 = L.BatchNormalization(128)
            self.conv3 = L.Convolution2D(128,256,4,2,1)
            self.norm3 = L.BatchNormalization(256)
            self.conv4 = L.Convolution2D(256,512,4,2,1)
            self.norm4 = L.BatchNormalization(512)
            self.conv5 = L.Convolution2D(512,512,4,2,1)
            self.norm5 = L.BatchNormalization(512)
            self.conv6 = L.Convolution2D(512,512,4,2,1)
            self.norm6 = L.BatchNormalization(512)

            self.conv7 = L.Convolution2D(512,1,4,2,1)


 
    def __call__(self, x1, x2):
        ###x1:label data, x2:converted or raw data

        w_h1   = F.leaky_relu(self.conv1(x1))
        w_h1_2 = F.leaky_relu(self.conv1_2(x2))
        w_h1 = F.concat([w_h1, w_h1_2])

        w_h2 = F.leaky_relu(self.norm2(self.conv2(w_h1)))
        w_h3 = F.leaky_relu(self.norm3(self.conv3(w_h2)))
        w_h4 = F.leaky_relu(self.norm4(self.conv4(w_h3)))
        w_h5 = F.leaky_relu(self.norm5(self.conv5(w_h4)))
        w_h6 = F.leaky_relu(self.norm6(self.conv6(w_h5)))

        #w_h6 = F.sigmoid(self.denorm8(self.conv7(w_h6)))

        return w_h6




class Converter(chainer.Chain):
    def __init__(self):
        super(Converter, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(12,64,3,1,1)
            #self.conv1 = L.Convolution2D(12,64,4,2,1)
            #self.norm1 = L.BatchNormalization(64)
            self.conv2 = L.Convolution2D(64,128,4,2,1)
            self.norm2 = L.BatchNormalization(128)
            self.conv3 = L.Convolution2D(128,256,4,2,1)
            self.norm3 = L.BatchNormalization(256)
            self.conv4 = L.Convolution2D(256,512,4,2,1)
            self.norm4 = L.BatchNormalization(512)
            self.conv5 = L.Convolution2D(512,512,4,2,1)
            self.norm5 = L.BatchNormalization(512)
            self.conv6 = L.Convolution2D(512,512,4,2,1)
            self.norm6 = L.BatchNormalization(512)
            self.conv7 = L.Convolution2D(512,512,4,2,1)
            self.norm7 = L.BatchNormalization(512)
            self.conv8 = L.Convolution2D(512,512,4,2,1)
            self.norm8 = L.BatchNormalization(512)

            self.deconv1 = L.Deconvolution2D(512,512,4,2,1)
            self.denorm1 = L.BatchNormalization(512)
            self.deconv2 = L.Deconvolution2D(1024,512,4,2,1)
            self.denorm2 = L.BatchNormalization(512)
            self.deconv3 = L.Deconvolution2D(1024,512,4,2,1)
            self.denorm3 = L.BatchNormalization(512)
            self.deconv4 = L.Deconvolution2D(1024,512,4,2,1)
            self.denorm4 = L.BatchNormalization(512)
            self.deconv5 = L.Deconvolution2D(1024,256,4,2,1)
            self.denorm5 = L.BatchNormalization(256)
            self.deconv6 = L.Deconvolution2D(512,128,4,2,1)
            self.denorm6 = L.BatchNormalization(128)
            self.deconv7 = L.Deconvolution2D(256,64,4,2,1)
            self.denorm7 = L.BatchNormalization(64)
            #self.deconv8 = L.Deconvolution2D(128,3,4,2,1)
            #self.denorm8 = L.BatchNormalization(3)
            self.deconv8 = L.Convolution2D(128,3,3,1,1)
 
    def __call__(self, x):
        w_h=x

        w_h1 = F.relu(self.conv1(w_h))
        w_h2 = F.relu(self.norm2(self.conv2(w_h1)))
        w_h3 = F.relu(self.norm3(self.conv3(w_h2)))
        w_h4 = F.relu(self.norm4(self.conv4(w_h3)))
        w_h5 = F.relu(self.norm5(self.conv5(w_h4)))
        w_h6 = F.relu(self.norm6(self.conv6(w_h5)))
        w_h7 = F.relu(self.norm7(self.conv7(w_h6)))
        w_h8 = F.relu(self.norm8(self.conv8(w_h7)))

        w_hd1 = F.leaky_relu(F.dropout(self.denorm1(self.deconv1(w_h8))))
        w_hd1 = F.concat([w_hd1, w_h7])
        w_hd2 = F.leaky_relu(F.dropout(self.denorm2(self.deconv2(w_hd1))))
        w_hd2 = F.concat([w_hd2, w_h6])
        w_hd3 = F.leaky_relu(F.dropout(self.denorm3(self.deconv3(w_hd2))))
        w_hd3 = F.concat([w_hd3, w_h5])

        w_hd4 = F.leaky_relu(self.denorm4(self.deconv4(w_hd3)))
        w_hd4 = F.concat([w_hd4, w_h4])
        w_hd5 = F.leaky_relu(self.denorm5(self.deconv5(w_hd4)))
        w_hd5 = F.concat([w_hd5, w_h3])
        w_hd6 = F.leaky_relu(self.denorm6(self.deconv6(w_hd5)))
        w_hd6 = F.concat([w_hd6, w_h2])
        w_hd7 = F.leaky_relu(self.denorm7(self.deconv7(w_hd6)))
        w_hd7 = F.concat([w_hd7, w_h1])
        #w_hd8 = F.sigmoid(self.denorm8(self.deconv8(w_hd7)))
        w_hd8 = self.deconv8(w_hd7)

        return w_hd8




class CGANUpdater(chainer.training.updaters.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.conv, self.dis = kwargs.pop('models')
        super(CGANUpdater, self).__init__(*args, **kwargs)

    def loss_dis(self, dis, y_conv, y_real):
        batchsize = len(y_conv)
        hoge, hoge, w, h = y_conv.data.shape
        #L1 = F.sum(F.softplus(-y_real))/ batchsize
        #L2 = F.sum(F.softplus(y_conv))/ batchsize
        L1 = F.sum(F.softplus(-y_real))/ (batchsize*w*h)
        L2 = F.sum(F.softplus(y_conv))/ (batchsize*w*h)
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        chainer.report({'loss_real': L1}, dis)
        chainer.report({'loss_conv': L2}, dis)
        return loss


    def loss_conv(self, conv, y_conv_d, y_conv, y_real, w1=1, w2=1000):
        batchsize = len(y_conv)
        hoge, hoge, w, h = y_conv_d.data.shape
        L1 = (F.sum(F.softplus(-y_conv_d))/ (batchsize*w*h))*w1
        L2 = (F.mean_squared_error(y_real, y_conv))*w2
        loss = L1 + L2
        chainer.report({'loss_conv2dis': L1}, conv)
        chainer.report({'loss_conv_MSE': L2}, conv)
        return loss


    def update_core(self):
        conv_optimizer = self.get_optimizer('conv')
        dis_optimizer = self.get_optimizer('dis')

        batch_s, batch_t = chainer.dataset.concat_examples(self.get_iterator('main').next())

        source = chainer.Variable(self.converter(batch_s, self.device))
        target = chainer.Variable(self.converter(batch_t, self.device))

        conv = self.conv
        dis = self.dis

        y_conv   = conv(source)
        y_conv_d = dis(source, y_conv)
        y_target_d = dis(source, target)


        conv_optimizer.update(self.loss_conv, conv, y_conv_d, y_conv, target)
        dis_optimizer.update(self.loss_dis, dis, y_conv_d, y_target_d)

















