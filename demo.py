from __future__ import print_function
import numpy
import math
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from misc import *
import sys
import models.dehaze1113  as net
import torchvision.models as models
import h5py
import torch.nn.functional as F
from skimage import measure
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
  default='pix2pix',  help='')
parser.add_argument('--valDataroot', required=False,default="", help='path to val dataset')
parser.add_argument('--mode', type=str, default='B2A', help='B2A: facade, A2B: edges2shoes')
parser.add_argument('--valBatchSize', type=int, default=1, help='input batch size')
parser.add_argument('--originalSize', type=int,
  default=1024, help='the height / width of the original input image')
parser.add_argument('--imageSize', type=int,
  default=1024, help='the height / width of the cropped input image to network')
parser.add_argument('--inputChannelSize', type=int,
  default=3, help='size of the input channels')
parser.add_argument('--outputChannelSize', type=int,
  default=3, help='size of the output channels')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--display', type=int, default=5, help='interval for displaying train-logs')
parser.add_argument('--evalIter', type=int, default=500, help='interval for evauating(generating) images from valDataroot')
opt = parser.parse_args()
print(opt)


# get dataloader

valDataloader = getLoader(opt.dataset,
                          opt.valDataroot,
                          opt.imageSize, #opt.originalSize,
                          opt.imageSize,
                          opt.valBatchSize,
                          opt.workers,
                          mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                          split='Train',
                          shuffle=False,
                          seed=None)



inputChannelSize = opt.inputChannelSize
outputChannelSize= opt.outputChannelSize

netG = net.FDGAN()

if opt.netG != '':
  netG.load_state_dict(torch.load(opt.netG))

# original saved file with DataParallel

# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
netG.load_state_dict(new_state_dict)
#netG.load_state_dict(state_dict)

netG = nn.DataParallel(netG).cuda()

netG.train()

target= torch.FloatTensor(opt.batchSize, outputChannelSize, opt.imageSize, opt.imageSize)
input = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)

val_target= torch.FloatTensor(opt.valBatchSize, outputChannelSize, opt.imageSize, opt.imageSize)
val_input = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)
# label_d = torch.FloatTensor(opt.batchSize)

target = torch.FloatTensor(opt.batchSize, outputChannelSize, opt.imageSize, opt.imageSize)
input = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)

val_target = torch.FloatTensor(opt.valBatchSize, outputChannelSize, opt.imageSize, opt.imageSize)
val_input = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)

target, input = target.cuda(), input.cuda()
val_target, val_input = val_target.cuda(), val_input.cuda()

target = Variable(target, volatile=True)
input = Variable(input,volatile=True)

import time

# NOTE training loop
ganIterations = 0
index=-1
iteration = 0
for epoch in range(1):
  for i, data in enumerate(valDataloader, 0):
    # t0 = time.time()

    if opt.mode == 'B2A':
        input_cpu, target_cpu = data
    elif opt.mode == 'A2B' :
        input_cpu, target_cpu = data
    batch_size = target_cpu.size(0)
    
    target_cpu, input_cpu = target_cpu.float().cuda(), input_cpu.float().cuda()
    # get paired data
    target.data.resize_as_(target_cpu).copy_(target_cpu)
    input.data.resize_as_(input_cpu).copy_(input_cpu)

    start = time.time()
    x_hat = netG(input)
    end = time.time()
    a = end-start
    print(a)
   
    x_hat1 = x_hat.data
    
    iteration=iteration+1
	#
    index2 = 0
    directory='./result_AAAI20/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(opt.valBatchSize):
        index=index+1
        print(index)
        x_hat2=x_hat1[index2,:,:,:]
      
 
        vutils.save_image(x_hat2, directory+str(index)+'.png', normalize=True, scale_each=False)
        
# trainLogger.close()
