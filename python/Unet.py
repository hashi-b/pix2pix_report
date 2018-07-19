#! /usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import sys
import numpy
import time
import os
import sklearn
import sklearn.utils as U
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.datasets import tuple_dataset
from chainer import training
from chainer.training import extensions
from chainer.cuda import cupy as xp
import PIL
from PIL import Image
from CGAN import *

def make_optimizer(model, alpha=0.0002, beta1=0.5):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
    return optimizer

class Jitterdataset(chainer.dataset.DatasetMixin):
    def __init__(self, in_img, out_img):
        self.dataset = []
        for i in range(len(in_img)):
            self.dataset.append((in_img[i], out_img[i]))

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i, width=256):
        hoge, h, w = self.dataset[i][0].shape
        xl = numpy.random.randint(0,w-width)
        xr = xl+width
        yl = numpy.random.randint(0,h-width)
        yr = yl+width
        return self.dataset[i][0][:,yl:yr,xl:xr], self.dataset[i][1][:,yl:yr,xl:xr]

def facade_image_resize(imgfile, width):
    ###resizing facade image###
    #ret_imgs = []
    ws = []
    hs = []
    ret_imgs = numpy.zeros((len(imgfile),12,width,width), dtype=numpy.float32)
    idx=0
    for i in imgfile:
        img = Image.open(i)
        orig_w, orig_h = img.size
        if orig_w>orig_h:
            new_w = width
            new_h = (width*orig_h)//orig_w
            img = numpy.asarray(img.resize((new_w,new_h))).astype(numpy.float32)-1
            img = img.reshape((1,new_h,new_w))
            #tmp= numpy.zeros((1,12,width,width), dtype=numpy.float32)
            #tmp= numpy.zeros((12,width,width), dtype=numpy.float32)
            for j in range(12):
                #tmp[0,j,width-new_h:,:] = img[0,:,:] == j
                #tmp[j,width-new_h:,:] = img[0,:,:] == j
                ret_imgs[idx, j,width-new_h:,:] = img[0,:,:] == j
            #img = tmp
        else:
            new_w = (width*orig_w)//orig_h
            new_h = width
            img = numpy.asarray(img.resize((new_w,new_h))).astype(numpy.float32)-1
            img = img.reshape((1,new_h,new_w))
            #tmp= numpy.zeros((1,12,width,width), dtype=numpy.float32)
            #tmp= numpy.zeros((12,width,width), dtype=numpy.float32)
            for j in range(12):
                #tmp[0,j,:,width-new_w:] = img[0,:,:] == j
                #tmp[j,:,width-new_w:] = img[0,:,:] == j
                ret_imgs[idx, j,:,width-new_w:] = img[0,:,:] == j
            #img = tmp
        #ret_imgs.append(xp.asarray(img))
        idx+=1
        ws.append(new_w)
        hs.append(new_h)
    return ret_imgs, ws, hs


def image_resize(imgfile, width):
    ###resizing and normalizeing color image###
    #ret_imgs = []
    ws = []
    hs = []
    ret_imgs = numpy.zeros((len(imgfile),3,width,width), dtype=numpy.float32)
    idx=0
    for i in imgfile:
        img = Image.open(i)
        orig_w, orig_h = img.size
        if orig_w>orig_h:
            new_w = width
            new_h = (width*orig_h)//orig_w
            img = numpy.asarray(img.resize((new_w,new_h),Image.BILINEAR)).transpose(2, 0, 1).astype(numpy.float32)/128 -1
            #img = img.reshape((1,3,new_h,new_w))
            img = img.reshape((3,new_h,new_w))
            #tmp= numpy.zeros((1,3,width,width), dtype=numpy.float32)
            #tmp= numpy.zeros((3,width,width), dtype=numpy.float32)
            #tmp[0,:,width-new_h:,:] = img[0,:,:,:]
            #tmp[:,width-new_h:,:] = img[:,:,:]
            ret_imgs[idx,:,width-new_h:,:] = img[:,:,:]
            #img = tmp
        else:
            new_w = (width*orig_w)//orig_h
            new_h = width
            img = numpy.asarray(img.resize((new_w,new_h),Image.BILINEAR)).transpose(2, 0, 1).astype(numpy.float32)/128 -1
            #img = img.reshape((1,3,new_h,new_w))
            img = img.reshape((3,new_h,new_w))
            #tmp= numpy.zeros((1,3,width,width), dtype=numpy.float32)
            #tmp= numpy.zeros((3,width,width), dtype=numpy.float32)
            #tmp[0,:,:,width-new_w:] = img[0,:,:,:]
            #tmp[:,:,width-new_w:] = img[:,:,:]
            ret_imgs[idx,:,:,width-new_w:] = img[:,:,:]
            #img = tmp
        #ret_imgs.append(xp.asarray(img))
        idx+=1
        ws.append(new_w)
        hs.append(new_h)
    return ret_imgs, ws, hs

def clip(a):
    return 0 if a<0 else (255 if a>255 else a)

def save_image(img, new_w, new_h, it, outdir, width=256):
    img_cpu = (img+1)*128

    print(img_cpu.shape)

    print("new_w",new_w)
    print("new_h",new_h)
    
    if width==new_w:
        img_cpu = img_cpu[:,width-new_h:,:]
    else:
        img_cpu = img_cpu[:,:,width-new_w:]

    print(img_cpu.shape)
    #img_cpu.transpose()

    im = numpy.zeros((new_h,new_w,3))
    #im = numpy.zeros((3, new_h, new_w))
    print(im.shape)

    im[:,:,2] = img_cpu[2,:,:]
    im[:,:,1] = img_cpu[1,:,:]
    im[:,:,0] = img_cpu[0,:,:]
    #im = im.transpose(1,2,0)

    im = numpy.vectorize(clip)(im).astype(numpy.uint8)
    Image.fromarray(im).save(outdir+"/im_%05d.jpg"%it)


def save_facade_image(img, new_w, new_h, it, outdir, gpu=0, width=286):
    if gpu>=0:
        img_cpu = img.get()

    print(img_cpu.shape)

    print("new_w",new_w)
    print("new_h",new_h)
    
    if width==new_w:
        img_cpu = img_cpu[:,:,width-new_h:,:]
    else:
        img_cpu = img_cpu[:,:,:,width-new_w:]

    print(img_cpu.shape)
    #img_cpu.transpose()

    im = numpy.zeros((new_h,new_w,1))
    #im = numpy.zeros((3, new_h, new_w))
    print(im.shape)

    for i in range(12):
        im[:,:,0] += img_cpu[0,i,:,:]*(i+1)
    #im[:,:,1] = img_cpu[0,1,:,:]
    #im[:,:,0] = img_cpu[0,0,:,:]
    #im = im.transpose(1,2,0)

    im = numpy.vectorize(clip)(im).astype(numpy.uint8)
    Image.fromarray(im).save(outdir+"/im_%05d.jpg"%it)




def main():
    parser = argparse.ArgumentParser(description="Clasical DNN:")
    parser.add_argument("--batchsize", "-b", type=int, default=2, help="Number of features in each mini-batch")
    parser.add_argument("--epoch", "-e", type=int, default=200, help="Number of sweeps over the dataset to train")
    parser.add_argument("--dataset", "-i", default="./base/", help="Directory of image files.")
    parser.add_argument("--outdir", "-o", default="", help="Directory for outputs.")
    parser.add_argument("--width", "-w", type=int, default=1024, help="Size of Images")
    parser.add_argument("--gpu", "-g", type=int, default=0, help="GPU ID (Negative value indicates CPU)")
    parser.add_argument("--seed", "-se", type=int, default="1234", help="random seed")
    args = parser.parse_args()

    lshape_conv_enc = [64, 128, 256, 512, 512, 512, 512, 512]
    lshape_conv_dec = [512, 512, 512, 512, 256, 128, 64]
    ### the first layer of the encoder does not use batch norm. 
    ### first 2 layers of the decoder use dropout. 

    print("GPU: {}".format(args.gpu))
    print("# Layer_shape (Converter): {}".format(lshape_conv_enc))
    print("# Layer_shape (Converter): {}".format(lshape_conv_dec))
    print("# epoch: {}".format(args.epoch))
    print("# batch: {}".format(args.batchsize))
    print("")


    ###read_train/test_data###
    #all_files = os.listdir(args.dataset)
    #images_in = [args.dataset + f for f in all_files if ("png" in f)]
    #images_out = []
    #for i in images_in:
    #    images_out.append(i.replace("png", "jpg"))
    images_in = []
    images_out = []
    for i in range(378):
        images_in.append(args.dataset + "/cmp_b0%03d.png"%(i+1))
        images_out.append(args.dataset + "/cmp_b0%03d.jpg"%(i+1))

    print('{} contains {} png(input) files'.format(args.dataset, len(images_in)))
    print('{} contains {} jpg(output) files'.format(args.dataset, len(images_out)))
    

    ###resize_data###

    test_fr = len(images_in)/10

    train_images_in  = images_in[test_fr:]
    test_images_in   = images_in[:test_fr]
    train_images_out  = images_out[test_fr:]
    test_images_out   = images_out[:test_fr]


    train_out_re, train_out_w, train_out_h = image_resize(train_images_out, 286)
    train_in_re, train_in_w, train_in_h = facade_image_resize(train_images_in, 286)

    test_out_re, test_out_w, test_out_h = image_resize(test_images_out, 256)
    test_in_re, test_in_w, test_in_h = facade_image_resize(test_images_in, 256)


    train_in_re  = xp.asarray(train_in_re)
    test_in_re   = xp.asarray(test_in_re)
    train_out_re = xp.asarray(train_out_re)
    test_out_re  = xp.asarray(test_out_re)


    #train_in_w = in_w[test_fr:]
    #train_in_h = in_h[test_fr:]
    #test_in_w = in_w[:test_fr]
    #test_in_h = in_h[:test_fr]
    

    #train_out_w = out_w[test_fr:]
    #train_out_h = out_h[test_fr:]
    #test_out_w = out_w[:test_fr]
    #test_out_h = out_h[:test_fr]


    print(len(train_in_re[0]))
    print(len(train_in_re[0][0]))


    #source_norm, target_norm = U.shuffle(source_norm, target_norm, random_state=0)
    #train = tuple_dataset.TupleDataset(train_in_re, train_out_re)
    train = Jitterdataset(train_in_re, train_out_re)
    test = tuple_dataset.TupleDataset(test_in_re, test_out_re)

    model = Unet()

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    opt_model = make_optimizer(model)


    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, opt_model, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.outdir)
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(model, 'model_{.updater.epoch}'), trigger=(1,'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()

    #chainer.serializers.save_npz(str(args.outdir)+'/Unet.model', model)


    #if args.gpu >= 0:
    model.to_cpu()
    test_in_re = chainer.cuda.to_cpu(test_in_re)
    with chainer.using_config('train', False):
        test_conv = model.convert(test_in_re)
    test_conv = test_conv.data

    for i in range(len(test_conv)):
        save_image(test_conv[i], test_in_w[i], test_in_h[i], i, args.outdir)



if __name__ == "__main__":
    main()


































