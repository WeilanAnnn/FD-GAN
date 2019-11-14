import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models
from torch.autograd import Variable
#import torch.utils.checkpoint as cp
import functools


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

def conv_block(in_dim,out_dim):
  return nn.Sequential(nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(in_dim,out_dim,kernel_size=1,stride=1,padding=0),
                       nn.AvgPool2d(kernel_size=2,stride=2))
def deconv_block(in_dim,out_dim):
  return nn.Sequential(nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.UpsamplingNearest2d(scale_factor=2))


def blockUNet1(in_c, out_c, name, transposed=False, bn=False, relu=True, dropout=False):
  block = nn.Sequential()
  if relu:
    block.add_module('%s.relu' % name, nn.ReLU(inplace=True))
  else:
    block.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
  if not transposed:
    block.add_module('%s.conv' % name, nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False))
  else:
    block.add_module('%s.tconv' % name, nn.ConvTranspose2d(in_c, out_c, 3, 1, 1, bias=False))
  if bn:
    block.add_module('%s.bn' % name, nn.BatchNorm2d(out_c))
  if dropout:
    block.add_module('%s.dropout' % name, nn.Dropout2d(0.5, inplace=True))
  return block

def blockUNet(in_c, out_c, name, transposed=False, bn=False, relu=True, dropout=False):
  block = nn.Sequential()
  if relu:
    block.add_module('%s.relu' % name, nn.ReLU(inplace=True))
  else:
    block.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
  if not transposed:
    block.add_module('%s.conv' % name, nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
  else:
    block.add_module('%s.tconv' % name, nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
  if bn:
    block.add_module('%s.bn' % name, nn.BatchNorm2d(out_c))
  if dropout:
    block.add_module('%s.dropout' % name, nn.Dropout2d(0.5, inplace=True))
  return block

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class BasicBlock_res(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_res, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class D1(nn.Module):
  def __init__(self, nc, ndf, hidden_size):
    super(D1, self).__init__()

    # 256
    self.conv1 = nn.Sequential(nn.Conv2d(nc,ndf,kernel_size=3,stride=1,padding=1),
                               nn.ELU(True))
    # 256
    self.conv2 = conv_block(ndf,ndf)
    # 128
    self.conv3 = conv_block(ndf, ndf*2)
    # 64
    self.conv4 = conv_block(ndf*2, ndf*3)
    # 32
    self.encode = nn.Conv2d(ndf*3, hidden_size, kernel_size=1,stride=1,padding=0)
    self.decode = nn.Conv2d(hidden_size, ndf, kernel_size=1,stride=1,padding=0)
    # 32
    self.deconv4 = deconv_block(ndf, ndf)
    # 64
    self.deconv3 = deconv_block(ndf, ndf)
    # 128
    self.deconv2 = deconv_block(ndf, ndf)
    # 256
    self.deconv1 = nn.Sequential(nn.Conv2d(ndf,ndf,kernel_size=3,stride=1,padding=1),
                                 nn.ELU(True),
                                 nn.Conv2d(ndf,ndf,kernel_size=3,stride=1,padding=1),
                                 nn.ELU(True),
                                 nn.Conv2d(ndf, nc, kernel_size=3, stride=1, padding=1),
                                 nn.Tanh())
    """
    self.deconv1 = nn.Sequential(nn.Conv2d(ndf,nc,kernel_size=3,stride=1,padding=1),
                                 nn.Tanh())
    """
  def forward(self,x):
    out1 = self.conv1(x)
    out2 = self.conv2(out1)
    out3 = self.conv3(out2)
    out4 = self.conv4(out3)
    out5 = self.encode(out4)
    dout5= self.decode(out5)
    dout4= self.deconv4(dout5)
    dout3= self.deconv3(dout4)
    dout2= self.deconv2(dout3)
    dout1= self.deconv1(dout2)
    return dout1

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=True):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)



class D(nn.Module):
  def __init__(self, nc, nf):
    super(D, self).__init__()

    main = nn.Sequential()
    # 256
    layer_idx = 1
    name = 'layer%d' % layer_idx
    main.add_module('%s.conv' % name, nn.Conv2d(nc, nf, 4, 2, 1, bias=False))

    # 128
    layer_idx += 1
    name = 'layer%d' % layer_idx
    main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

    # 64
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

    # 32
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s.conv' % name, nn.Conv2d(nf, nf*2, 4, 1, 1, bias=False))
    main.add_module('%s.bn' % name, nn.BatchNorm2d(nf*2))

    # 31
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s.conv' % name, nn.Conv2d(nf, 1, 4, 1, 1, bias=False))
    main.add_module('%s.sigmoid' % name , nn.Sigmoid())
    # 30 (sizePatchGAN=30)

    self.main = main

  def forward(self, x):
    output = self.main(x)
    return output


class D_33(nn.Module):
  def __init__(self, nc, nf):
    super(D_33, self).__init__()

    main = nn.Sequential()
    # 256
    layer_idx = 1
    name = 'layer%d' % layer_idx
    main.add_module('%s.conv' % name, nn.Conv2d(nc, nf, 4, 2, 1, bias=False))

    # 128
    layer_idx += 1
    name = 'layer%d' % layer_idx
    main.add_module(name, blockUNet1(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

    # 64
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module(name, blockUNet1(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

    # 32
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s.conv' % name, nn.Conv2d(nf, nf*2, 4, 1, 1, bias=False))
    #main.add_module('%s.bn' % name, nn.BatchNorm2d(nf*2))

    # 31
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s.conv' % name, nn.Conv2d(nf, 1, 4, 1, 1, bias=False))
    main.add_module('%s.sigmoid' % name , nn.Sigmoid())
    # 30 (sizePatchGAN=30)

    self.main = main

  def forward(self, x):
    output = self.main(x)
    return output

class D_3(nn.Module):
  def __init__(self, nc, nf):
    super(D_3, self).__init__()

    main = nn.Sequential()
    # 256
    layer_idx = 1
    name = 'layer%d' % layer_idx
    main.add_module('%s.conv' % name, nn.Conv2d(nc, nf, 3, 1, 1, bias=False))

    # 128
    layer_idx += 1
    name = 'layer%d' % layer_idx
    main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

    # 64
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

    # 32
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s.conv' % name, nn.Conv2d(nf, nf*2, 3, 1, 1, bias=False))
    main.add_module('%s.bn' % name, nn.BatchNorm2d(nf*2))

    # 31
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s.conv' % name, nn.Conv2d(nf, 3, 3, 1, 1, bias=False))
    main.add_module('%s.sigmoid' % name , nn.Sigmoid())
    # 30 (sizePatchGAN=30)

    self.main = main

  def forward(self, x):
    output = self.main(x)
    return output


class D_3_withoutBN(nn.Module):
  def __init__(self, nc, nf):
    super(D_3_withoutBN, self).__init__()

    main = nn.Sequential()
    # 256
    layer_idx = 1
    name = 'layer%d' % layer_idx
    main.add_module('%s.conv' % name, nn.Conv2d(nc, nf, 4, 2, 1, bias=False))

    # 128
    layer_idx += 1
    name = 'layer%d' % layer_idx
    main.add_module(name, blockUNet1(nf, nf*2, name, transposed=False, bn=False, relu=False, dropout=False))

    # 64
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module(name, blockUNet1(nf, nf*2, name, transposed=False, bn=False, relu=False, dropout=False))

    # 32
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s.conv' % name, nn.Conv2d(nf, nf*2, 4, 1, 1, bias=False))
    #main.add_module('%s.bn' % name, nn.BatchNorm2d(nf*2))

    # 31
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    #main.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s.conv' % name, nn.Conv2d(nf, 3, 4, 1, 1, bias=False))
    #main.add_module('%s.sigmoid' % name , nn.Sigmoid())
    # 30 (sizePatchGAN=30)

    self.main = main

  def forward(self, x):
    output = self.main(x)
    return output


class D_withoutBN(nn.Module):
  def __init__(self, nc, nf):
    super(D_withoutBN, self).__init__()

    main = nn.Sequential()
    # 256
    layer_idx = 1
    name = 'layer%d' % layer_idx
    main.add_module('%s.conv' % name, nn.Conv2d(nc, nf, 4, 2, 1, bias=False))

    # 128
    layer_idx += 1
    name = 'layer%d' % layer_idx
    main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=False, relu=False, dropout=False))

    # 64
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=False, relu=False, dropout=False))

    # 32
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s.conv' % name, nn.Conv2d(nf, nf*2, 4, 1, 1, bias=False))
    #main.add_module('%s.bn' % name, nn.BatchNorm2d(nf*2))

    # 31
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s.conv' % name, nn.Conv2d(nf, 1, 4, 1, 1, bias=False))
    main.add_module('%s.sigmoid' % name , nn.Sigmoid())
    # 30 (sizePatchGAN=30)

    self.main = main

  def forward(self, x):
    output = self.main(x)
    return output

class D_withoutBNsame(nn.Module):
  def __init__(self, nc, nf):
    super(D_withoutBNsame, self).__init__()

    main = nn.Sequential()
    # 256
    layer_idx = 1
    name = 'layer%d' % layer_idx
    main.add_module('%s.conv' % name, nn.Conv2d(nc, nf, 3, 1, 1, bias=False))

    # 128
    layer_idx += 1
    name = 'layer%d' % layer_idx
    main.add_module(name, blockUNet1(nf, nf*2, name, transposed=False, bn=False, relu=False, dropout=False))

    # 64
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module(name, blockUNet1(nf, nf*2, name, transposed=False, bn=False, relu=False, dropout=False))

    # 32
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s.conv' % name, nn.Conv2d(nf, nf*2, 3, 1, 1, bias=False))
    #main.add_module('%s.bn' % name, nn.BatchNorm2d(nf*2))

    # 31
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s.conv' % name, nn.Conv2d(nf, 1, 3, 1, 1, bias=False))
    main.add_module('%s.sigmoid' % name , nn.Sigmoid())
    # 30 (sizePatchGAN=30)

    self.main = main

  def forward(self, x):
    output = self.main(x)
    return output

class D_withoutBNrelu(nn.Module):
  def __init__(self, nc, nf):
    super(D_withoutBNrelu, self).__init__()

    main = nn.Sequential()
    # 256
    layer_idx = 1
    name = 'layer%d' % layer_idx
    main.add_module('%s.conv' % name, nn.Conv2d(nc, nf, 4, 2, 1, bias=False))

    # 128
    layer_idx += 1
    name = 'layer%d' % layer_idx
    main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=False, relu=False, dropout=False))

    # 64
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=False, relu=False, dropout=False))

    # 32
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    #main.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s.conv' % name, nn.Conv2d(nf, nf*2, 4, 1, 1, bias=False))
    #main.add_module('%s.bn' % name, nn.BatchNorm2d(nf*2))

    # 31
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    #main.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s.conv' % name, nn.Conv2d(nf, 1, 4, 1, 1, bias=False))
    main.add_module('%s.sigmoid' % name , nn.Sigmoid())
    # 30 (sizePatchGAN=30)

    self.main = main

  def forward(self, x):
    output = self.main(x)
    return output
class D_withoutsigmoidBN(nn.Module):
  def __init__(self, nc, nf):
    super(D_withoutsigmoidBN, self).__init__()

    main = nn.Sequential()
    # 256
    layer_idx = 1
    name = 'layer%d' % layer_idx
    main.add_module('%s.conv' % name, nn.Conv2d(nc, nf, 4, 2, 1, bias=False))

    # 128
    layer_idx += 1
    name = 'layer%d' % layer_idx
    main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=False, relu=False, dropout=False))

    # 64
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=False, relu=False, dropout=False))

    # 32
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s.conv' % name, nn.Conv2d(nf, nf*2, 4, 1, 1, bias=False))
    #main.add_module('%s.bn' % name, nn.BatchNorm2d(nf*2))

    # 31
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s.conv' % name, nn.Conv2d(nf, 1, 4, 1, 1, bias=False))
    #main.add_module('%s.sigmoid' % name , nn.Sigmoid())
    # 30 (sizePatchGAN=30)

    self.main = main

  def forward(self, x):
    output = self.main(x)
    return output


class D_withoutactivation(nn.Module):
  def __init__(self, nc, nf):
    super(D_withoutactivation, self).__init__()

    main = nn.Sequential()
    # 256
    layer_idx = 1
    name = 'layer%d' % layer_idx
    main.add_module('%s.conv' % name, nn.Conv2d(nc, nf, 4, 2, 1, bias=False))

    # 128
    layer_idx += 1
    name = 'layer%d' % layer_idx
    main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

    # 64
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

    # 32
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    #main.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s.conv' % name, nn.Conv2d(nf, nf*2, 4, 1, 1, bias=False))
    main.add_module('%s.bn' % name, nn.BatchNorm2d(nf*2))

    # 31
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    #main.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s.conv' % name, nn.Conv2d(nf, 1, 4, 1, 1, bias=False))
    #main.add_module('%s.sigmoid' % name , nn.Sigmoid())
    # 30 (sizePatchGAN=30)

    self.main = main

  def forward(self, x):
    output = self.main(x)
    return output

class D_tran(nn.Module):
  def __init__(self, nc, nf):
    super(D_tran, self).__init__()

    main = nn.Sequential()
    # 256
    layer_idx = 1
    name = 'layer%d' % layer_idx
    main.add_module('%s.conv' % name, nn.Conv2d(nc, nf, 4, 2, 1, bias=False))

    # 128
    layer_idx += 1
    name = 'layer%d' % layer_idx
    main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

    # 64
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

    # 32
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s.conv' % name, nn.Conv2d(nf, nf*2, 4, 1, 1, bias=False))
    main.add_module('%s.bn' % name, nn.BatchNorm2d(nf*2))

    # 31
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s.conv' % name, nn.Conv2d(nf, 1, 4, 1, 1, bias=False))
    main.add_module('%s.sigmoid' % name , nn.Sigmoid())
    # 30 (sizePatchGAN=30)

    self.main = main

  def forward(self, x):
    output = self.main(x)
    return output



class G(nn.Module):
  def __init__(self, input_nc, output_nc, nf):
    super(G, self).__init__()
    # input is 256 x 256
    layer_idx = 1
    name = 'layer%d' % layer_idx
    layer1 = nn.Sequential()
    layer1.add_module(name, nn.Conv2d(input_nc, nf, 4, 2, 1, bias=False))
    # input is 128 x 128
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer2 = blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 64 x 64
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer3 = blockUNet(nf*2, nf*4, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 32
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer4 = blockUNet(nf*4, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 16
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer5 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 8
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer6 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 4
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer7 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 2 x  2
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer8 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)

    ## NOTE: decoder
    # input is 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8
    dlayer8 = blockUNet(d_inc, nf*8, name, transposed=True, bn=False, relu=True, dropout=True)

    # input is 2
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer7 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)
    # input is 4
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer6 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)
    # input is 8
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer5 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 16
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer4 = blockUNet(d_inc, nf*4, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 32
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*4*2
    dlayer3 = blockUNet(d_inc, nf*2, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 64
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*2*2
    dlayer2 = blockUNet(d_inc, nf, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 128
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    dlayer1 = nn.Sequential()
    d_inc = nf*2
    dlayer1.add_module('%s.relu' % name, nn.ReLU(inplace=True))
    dlayer1.add_module('%s.tconv' % name, nn.ConvTranspose2d(d_inc, 20, 4, 2, 1, bias=False))

    dlayerfinal = nn.Sequential()

    dlayerfinal.add_module('%s.conv' % name, nn.Conv2d(24, output_nc, 3, 1, 1, bias=False))
    dlayerfinal.add_module('%s.tanh' % name, nn.Tanh())

    self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
    self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
    self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
    self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

    self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)

    self.upsample = F.upsample_nearest

    self.layer1 = layer1
    self.layer2 = layer2
    self.layer3 = layer3
    self.layer4 = layer4
    self.layer5 = layer5
    self.layer6 = layer6
    self.layer7 = layer7
    self.layer8 = layer8
    self.dlayer8 = dlayer8
    self.dlayer7 = dlayer7
    self.dlayer6 = dlayer6
    self.dlayer5 = dlayer5
    self.dlayer4 = dlayer4
    self.dlayer3 = dlayer3
    self.dlayer2 = dlayer2
    self.dlayer1 = dlayer1
    self.dlayerfinal = dlayerfinal
    self.relu=nn.LeakyReLU(0.2, inplace=True)

  def forward(self, x):
    out1 = self.layer1(x)
    out2 = self.layer2(out1)
    out3 = self.layer3(out2)
    out4 = self.layer4(out3)
    out5 = self.layer5(out4)
    out6 = self.layer6(out5)
    out7 = self.layer7(out6)
    out8 = self.layer8(out7)
    dout8 = self.dlayer8(out8)
    dout8_out7 = torch.cat([dout8, out7], 1)
    dout7 = self.dlayer7(dout8_out7)
    dout7_out6 = torch.cat([dout7, out6], 1)
    dout6 = self.dlayer6(dout7_out6)
    dout6_out5 = torch.cat([dout6, out5], 1)
    dout5 = self.dlayer5(dout6_out5)
    dout5_out4 = torch.cat([dout5, out4], 1)
    dout4 = self.dlayer4(dout5_out4)
    dout4_out3 = torch.cat([dout4, out3], 1)
    dout3 = self.dlayer3(dout4_out3)
    dout3_out2 = torch.cat([dout3, out2], 1)
    dout2 = self.dlayer2(dout3_out2)
    dout2_out1 = torch.cat([dout2, out1], 1)
    dout1 = self.dlayer1(dout2_out1)

    shape_out = dout1.data.size()
    # print(shape_out)
    shape_out = shape_out[2:4]

    x101 = F.avg_pool2d(dout1, 16)
    x102 = F.avg_pool2d(dout1, 8)
    x103 = F.avg_pool2d(dout1, 4)
    x104 = F.avg_pool2d(dout1, 2)

    x1010 = self.upsample(self.relu(self.conv1010(x101)),size=shape_out)
    x1020 = self.upsample(self.relu(self.conv1020(x102)),size=shape_out)
    x1030 = self.upsample(self.relu(self.conv1030(x103)),size=shape_out)
    x1040 = self.upsample(self.relu(self.conv1040(x104)),size=shape_out)

    dehaze = torch.cat((x1010, x1020, x1030, x1040, dout1), 1)

    dout1 = self.dlayerfinal(dehaze)

    return dout1

class G2(nn.Module):
  def __init__(self, input_nc, output_nc, nf):
    super(G2, self).__init__()
    # input is 256 x 256
    layer_idx = 1
    name = 'layer%d' % layer_idx
    layer1 = nn.Sequential()
    layer1.add_module(name, nn.Conv2d(input_nc, nf, 4, 2, 1, bias=False))
    # input is 128 x 128
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer2 = blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 64 x 64
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer3 = blockUNet(nf*2, nf*4, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 32
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer4 = blockUNet(nf*4, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 16
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer5 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 8
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer6 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 4
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer7 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
    # input is 2 x  2
    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer8 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)

    ## NOTE: decoder
    # input is 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8
    dlayer8 = blockUNet(d_inc, nf*8, name, transposed=True, bn=False, relu=True, dropout=True)

    #import pdb; pdb.set_trace()
    # input is 2
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer7 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)
    # input is 4
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer6 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)
    # input is 8
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer5 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 16
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer4 = blockUNet(d_inc, nf*4, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 32
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*4*2
    dlayer3 = blockUNet(d_inc, nf*2, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 64
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*2*2
    dlayer2 = blockUNet(d_inc, nf, name, transposed=True, bn=True, relu=True, dropout=False)
    # input is 128
    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    dlayer1 = nn.Sequential()
    d_inc = nf*2
    dlayer1.add_module('%s.relu' % name, nn.ReLU(inplace=True))
    dlayer1.add_module('%s.tconv' % name, nn.ConvTranspose2d(d_inc, output_nc, 4, 2, 1, bias=False))
    dlayer1.add_module('%s.tanh' % name, nn.LeakyReLU(0.2, inplace=True))

    self.layer1 = layer1
    self.layer2 = layer2
    self.layer3 = layer3
    self.layer4 = layer4
    self.layer5 = layer5
    self.layer6 = layer6
    self.layer7 = layer7
    self.layer8 = layer8
    self.dlayer8 = dlayer8
    self.dlayer7 = dlayer7
    self.dlayer6 = dlayer6
    self.dlayer5 = dlayer5
    self.dlayer4 = dlayer4
    self.dlayer3 = dlayer3
    self.dlayer2 = dlayer2
    self.dlayer1 = dlayer1

  def forward(self, x):
    out1 = self.layer1(x)
    out2 = self.layer2(out1)
    out3 = self.layer3(out2)
    out4 = self.layer4(out3)
    out5 = self.layer5(out4)
    out6 = self.layer6(out5)
    out7 = self.layer7(out6)
    out8 = self.layer8(out7)
    dout8 = self.dlayer8(out8)
    dout8_out7 = torch.cat([dout8, out7], 1)
    dout7 = self.dlayer7(dout8_out7)
    dout7_out6 = torch.cat([dout7, out6], 1)
    dout6 = self.dlayer6(dout7_out6)
    dout6_out5 = torch.cat([dout6, out5], 1)
    dout5 = self.dlayer5(dout6_out5)
    dout5_out4 = torch.cat([dout5, out4], 1)
    dout4 = self.dlayer4(dout5_out4)
    dout4_out3 = torch.cat([dout4, out3], 1)
    dout3 = self.dlayer3(dout4_out3)
    dout3_out2 = torch.cat([dout3, out2], 1)
    dout2 = self.dlayer2(dout3_out2)
    dout2_out1 = torch.cat([dout2, out1], 1)
    dout1 = self.dlayer1(dout2_out1)
    return dout1


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckBlockdy(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlockdy, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(x))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(out))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckBlockdy1(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlockdy1, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(x))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(out))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return  out

class BottleneckBlock1(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock1, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=5, stride=1,
                               padding=2, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)



class BottleneckBlock2(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock2, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=7, stride=1,
                               padding=3, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=2)


class TransitionBlockdy(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlockdy, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(x))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=2)

class TransitionBlockdy1(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlockdy1, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(x))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=2)

class TransitionBlockdy2(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlockdy2, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(x))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=4)

class TransitionBlock1(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock1, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)



class TransitionBlock3(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock3, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out


class Dense(nn.Module):
    def __init__(self):
        super(Dense, self).__init__()




        ############# 256-256  ##############
        haze_class = models.densenet121(pretrained=True)

        self.conv0=haze_class.features.conv0
        self.norm0=haze_class.features.norm0
        self.relu0=haze_class.features.relu0
        self.pool0=haze_class.features.pool0

        ############# Block1-down 64-64  ##############
        self.dense_block1=haze_class.features.denseblock1
        self.trans_block1=haze_class.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2=haze_class.features.denseblock2
        self.trans_block2=haze_class.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3=haze_class.features.denseblock3
        self.trans_block3=haze_class.features.transition3

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock(512,256)
        self.trans_block4=TransitionBlock(768,128)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock(384,256)
        self.trans_block5=TransitionBlock(640,128)

        ############# Block6-up 32-32   ##############
        self.dense_block6=BottleneckBlock(256,128)
        self.trans_block6=TransitionBlock(384,64)


        ############# Block7-up 64-64   ##############
        self.dense_block7=BottleneckBlock(64,64)
        self.trans_block7=TransitionBlock(128,32)

        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8=BottleneckBlock(32,32)
        self.trans_block8=TransitionBlock(64,16)

        self.conv_refin=nn.Conv2d(19,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(20, 3, kernel_size=3,stride=1,padding=1)
        #self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        # self.upsample = F.upsample_nearest
        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)



    def forward(self, x):
        ## 256x256
        x0=self.pool0(self.relu0(self.norm0(self.conv0(x))))

        ## 64 X 64
        x1=self.dense_block1(x0)
        # print x1.size()
        x1=self.trans_block1(x1)

        ###  32x32
        x2=self.trans_block2(self.dense_block2(x1))
        # print  x2.size()


        ### 16 X 16
        x3=self.trans_block3(self.dense_block3(x2))

        # x3=Variable(x3.data,requires_grad=False)

        ## 8 X 8
        x4=self.trans_block4(self.dense_block4(x3))

        x42=torch.cat([x4,x2],1)
        ## 16 X 16
        x5=self.trans_block5(self.dense_block5(x42))

        x52=torch.cat([x5,x1],1)
        ##  32 X 32
        x6=self.trans_block6(self.dense_block6(x52))

        ##  64 X 64
        x7=self.trans_block7(self.dense_block7(x6))

        ##  128 X 128
        x8=self.trans_block8(self.dense_block8(x7))

        # print x8.size()
        # print x.size()

        x8=torch.cat([x8,x],1)

        # print x8.size()
		# x91 = self.conv_refin(x8)
		# x9 = self.relu(x91)
		
        # x9=self.relu(self.batchnorm20(self.conv_refin(x8)))
        x9 = self.relu(self.batchnorm20(self.conv_refin(x8)))

        # shape_out = x9.data.size()
        # # print(shape_out)
        # shape_out = shape_out[2:4]
		#
        # x101 = F.avg_pool2d(x9, 32)
        # x102 = F.avg_pool2d(x9, 16)
        # x103 = F.avg_pool2d(x9, 8)
        # x104 = F.avg_pool2d(x9, 4)
		#
        # x1010 = self.upsample(self.relu(self.batchnorm1(self.conv1010(x101))), shape_out)
        # x1020 = self.upsample(self.relu(self.batchnorm1(self.conv1020(x102))), shape_out)
        # x1030 = self.upsample(self.relu(self.batchnorm1(self.conv1030(x103))), shape_out)
        # x1040 = self.upsample(self.relu(self.batchnorm1(self.conv1040(x104))), shape_out)

        # dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        # dehaze = self.tanh(self.refine3(dehaze))
        dehaze = self.tanh(self.refine3(x9))


        return dehaze



class Dense2(nn.Module):
    def __init__(self):
        super(Dense2, self).__init__()




        ############# 256-256  ##############
        haze_class = models.densenet121(pretrained=True)

        self.conv0=haze_class.features.conv0
        self.norm0=haze_class.features.norm0
        self.relu0=haze_class.features.relu0
        self.pool0=haze_class.features.pool0

        ############# Block1-down 64-64  ##############
        self.dense_block1=haze_class.features.denseblock1
        self.trans_block1=haze_class.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2=haze_class.features.denseblock2
        self.trans_block2=haze_class.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3=haze_class.features.denseblock3
        self.trans_block3=haze_class.features.transition3

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock(512,256)
        self.trans_block4=TransitionBlock(768,128)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock(384,256)
        self.trans_block5=TransitionBlock(640,128)

        ############# Block6-up 32-32   ##############
        self.dense_block6=BottleneckBlock(256,128)
        self.trans_block6=TransitionBlock(384,64)


        ############# Block7-up 64-64   ##############
        self.dense_block7=BottleneckBlock(64,64)
        self.trans_block7=TransitionBlock(128,32)

        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8=BottleneckBlock(32,32)
        self.trans_block8=TransitionBlock(64,16)

        self.conv_refin=nn.Conv2d(19,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        ## 256x256
        x0=self.pool0(self.relu0(self.norm0(self.conv0(x))))

        ## 64 X 64
        x1=self.dense_block1(x0)
        # print x1.size()
        x1=self.trans_block1(x1)

        ###  32x32
        x2=self.trans_block2(self.dense_block2(x1))
        # print  x2.size()


        ### 16 X 16
        x3=self.trans_block3(self.dense_block3(x2))

        # x3=Variable(x3.data,requires_grad=False)

        ## 8 X 8
        x4=self.trans_block4(self.dense_block4(x3))

        x42=torch.cat([x4,x2],1)
        ## 16 X 16
        x5=self.trans_block5(self.dense_block5(x42))

        x52=torch.cat([x5,x1],1)
        ##  32 X 32
        x6=self.trans_block6(self.dense_block6(x52))

        ##  64 X 64
        x7=self.trans_block7(self.dense_block7(x6))

        ##  128 X 128
        x8=self.trans_block8(self.dense_block8(x7))

        # print x8.size()
        # print x.size()

        x8=torch.cat([x8,x],1)

        # print x8.size()

        x9=self.relu(self.conv_refin(x8))

        shape_out = x9.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)

        x1010 = self.upsample(self.relu(self.conv1010(x101)), size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)), size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)), size=shape_out)
        x1040 = self.upsample(self.relu(self.conv1040(x104)), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        dehaze = self.tanh(self.refine3(dehaze))

        return dehaze


class derain(nn.Module):
  def __init__(self):
    super(derain, self).__init__()
    self.tran_est=G(input_nc=3,output_nc=3, nf=32)
    self.tran_est1=G2(input_nc=3,output_nc=3, nf=32)

    self.tran_dense=Dense()

    self.relu=nn.LeakyReLU(0.2, inplace=True)
    # self.relu5=nn.ReLU6()

    self.tanh=nn.Tanh()

    self.refine1= nn.Conv2d(6, 20, kernel_size=3,stride=1,padding=1)
    self.refine2= nn.Conv2d(20, 20, kernel_size=3,stride=1,padding=1)
    self.threshold=nn.Threshold(0.1, 0.1)

    self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
    self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
    self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
    self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

    self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)

    self.upsample = F.upsample_nearest

    self.batch1= nn.BatchNorm2d(20)

    # self.batch2 = nn.InstanceNorm2d(100, affine=True)
  def forward(self, x):
    tran=self.tran_est(x)
    tran1=self.tran_dense(x)


    dehaze=torch.cat([tran,tran1],1)
    # dehaze=dehaze/(tran+(10**-10))
    # dehaze=self.relu(self.batch1(self.refine1(dehaze)))
    # dehaze=self.relu(self.batch1(self.refine2(dehaze)))

    dehaze=self.relu((self.refine1(dehaze)))
    dehaze=self.relu((self.refine2(dehaze)))
    shape_out = dehaze.data.size()
    # print(shape_out)
    shape_out = shape_out[2:4]

    x101 = F.avg_pool2d(dehaze, 32)
    x102 = F.avg_pool2d(dehaze, 16)
    x103 = F.avg_pool2d(dehaze, 8)
    x104 = F.avg_pool2d(dehaze, 4)

    x1010 = self.upsample(self.relu(self.conv1010(x101)),size=shape_out)
    x1020 = self.upsample(self.relu(self.conv1020(x102)),size=shape_out)
    x1030 = self.upsample(self.relu(self.conv1030(x103)),size=shape_out)
    x1040 = self.upsample(self.relu(self.conv1040(x104)),size=shape_out)

    dehaze = torch.cat((x1010, x1020, x1030, x1040, dehaze), 1)
    rain_streak= self.tanh(self.refine3(dehaze))
    dehaze=x-rain_streak

    return dehaze,rain_streak


class dehaze(nn.Module):
  def __init__(self, input_nc, output_nc, nf):
    super(dehaze, self).__init__()
    self.tran_est=G(input_nc=3,output_nc=3, nf=64)
    self.atp_est=G2(input_nc=3,output_nc=3, nf=8)

    self.tran_dense=Dense()
    self.relu=nn.LeakyReLU(0.2, inplace=True)
    # self.relu5=nn.ReLU6()

    self.tanh=nn.Tanh()

    self.refine1= nn.Conv2d(6, 20, kernel_size=3,stride=1,padding=1)
    self.refine2= nn.Conv2d(20, 20, kernel_size=3,stride=1,padding=1)
    self.threshold=nn.Threshold(0.1, 0.1)

    self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
    self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
    self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
    self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

    self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)

    self.upsample = F.upsample_nearest

    self.batch1= nn.BatchNorm2d(20)

    # self.batch2 = nn.InstanceNorm2d(100, affine=True)
  def forward(self, x):
    tran=self.tran_dense(x)
    # x = Variable(x.data, requires_grad=True)
    atp= self.atp_est(x)

    # zz= torch.abs(self.threshold(tran))
    zz= torch.abs((tran))+(10**-10)
    shape_out1 = atp.data.size()
    # print(shape_out)
    # shape_out = shape_out[0:5]

    # atp_mean=torch.mean(atp)

    # threshold = nn.Threshold(10, 0.95)

    shape_out = shape_out1[2:4]
    atp = F.avg_pool2d(atp, shape_out1[2])
    atp = self.upsample(self.relu(atp),size=shape_out)


    # print atp.data
    # atp = threshold(atp)



    dehaze= (x-atp)/zz+ atp
    dehaze2=dehaze
    dehaze=torch.cat([dehaze,x],1)
    # dehaze=dehaze/(tran+(10**-10))
    # dehaze=self.relu(self.batch1(self.refine1(dehaze)))
    # dehaze=self.relu(self.batch1(self.refine2(dehaze)))

    dehaze=self.relu((self.refine1(dehaze)))
    dehaze=self.relu((self.refine2(dehaze)))
    shape_out = dehaze.data.size()
    # print(shape_out)
    shape_out = shape_out[2:4]

    x101 = F.avg_pool2d(dehaze, 32)
    x102 = F.avg_pool2d(dehaze, 16)
    x103 = F.avg_pool2d(dehaze, 8)
    x104 = F.avg_pool2d(dehaze, 4)

    x1010 = self.upsample(self.relu(self.conv1010(x101)),size=shape_out)
    x1020 = self.upsample(self.relu(self.conv1020(x102)),size=shape_out)
    x1030 = self.upsample(self.relu(self.conv1030(x103)),size=shape_out)
    x1040 = self.upsample(self.relu(self.conv1040(x104)),size=shape_out)

    dehaze = torch.cat((x1010, x1020, x1030, x1040, dehaze), 1)
    dehaze= self.tanh(self.refine3(dehaze))

    return dehaze, tran, atp, dehaze2

class Dense_rain(nn.Module):
    def __init__(self):
        super(Dense_rain, self).__init__()




        ############# 256-256  ##############
        haze_class = models.densenet121(pretrained=True)

        self.conv0=haze_class.features.conv0
        self.norm0=haze_class.features.norm0
        self.relu0=haze_class.features.relu0
        self.pool0=haze_class.features.pool0

        ############# Block1-down 64-64  ##############
        self.dense_block1=haze_class.features.denseblock1
        self.trans_block1=haze_class.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2=haze_class.features.denseblock2
        self.trans_block2=haze_class.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3=haze_class.features.denseblock3
        self.trans_block3=haze_class.features.transition3

        # ############# Block31-down  xx-xx ##############
        self.dense_block31=haze_class.features.denseblock4
        # self.trans_block31=haze_class.features.transition4

        self.dense_norm31=haze_class.features.norm5

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock(512,256)
        self.trans_block4=TransitionBlock(768,128)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock(384,256)
        self.trans_block5=TransitionBlock(640,128)

        ############# Block6-up 32-32   ##############
        self.dense_block6=BottleneckBlock(256,128)
        self.trans_block6=TransitionBlock(384,64)


        ############# Block7-up 64-64   ##############
        self.dense_block7=BottleneckBlock(64,64)
        self.trans_block7=TransitionBlock(128,32)

        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8=BottleneckBlock(32,32)
        self.trans_block8=TransitionBlock(64,16)

        self.conv_refin=nn.Conv2d(19,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(4, 3, kernel_size=3,stride=1,padding=1)

        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)


        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)



    def forward(self, x):
        ## 256x256
        x0=self.pool0(self.relu0(self.norm0(self.conv0(x))))

        ## 64 X 64
        x1=self.dense_block1(x0)
        # print x1.size()
        x1=self.trans_block1(x1)

        ###  32x32
        x2=self.trans_block2(self.dense_block2(x1))
        # print  x2.size()


        ### 16 X 16
        x3=self.trans_block3(self.dense_block3(x2))
        ## Classifier  ##

        ## 8 X 8
        x4=self.trans_block4(self.dense_block4(x3))

        x42=torch.cat([x4,x2],1)
        ## 16 X 16
        x5=self.trans_block5(self.dense_block5(x42))

        x52=torch.cat([x5,x1],1)
        ##  32 X 32
        x6=self.trans_block6(self.dense_block6(x52))

        ##  64 X 64
        x7=self.trans_block7(self.dense_block7(x6))

        ##  128 X 128
        x8=self.trans_block8(self.dense_block8(x7))

        # print x8.size()
        # print x.size()

        x8=torch.cat([x8,x],1)

        # print x8.size()

        x9=self.relu(self.batchnorm20(self.conv_refin(x8)))

        shape_out = x9.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)

        x1010 = self.upsample(self.relu((self.conv1010(x101))), size=shape_out)
        x1020 = self.upsample(self.relu((self.conv1020(x102))), size=shape_out)
        x1030 = self.upsample(self.relu((self.conv1030(x103))), size=shape_out)
        x1040 = self.upsample(self.relu((self.conv1040(x104))), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        # dehaze = torch.cat((x1010, x1020, x1030, x1040), 1)

        residual = self.tanh(self.refine3(dehaze))
        # dehaze=x - residual

        return residual



class Dense_rain_cvprw(nn.Module):
    def __init__(self):
        super(Dense_rain_cvprw, self).__init__()




        ############# 256-256  ##############
        haze_class = models.densenet121(pretrained=True)

        self.conv0=haze_class.features.conv0
        self.norm0=haze_class.features.norm0
        self.relu0=haze_class.features.relu0
        self.pool0=haze_class.features.pool0

        ############# Block1-down 64-64  ##############
        self.dense_block1=haze_class.features.denseblock1
        self.trans_block1=haze_class.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2=haze_class.features.denseblock2
        self.trans_block2=haze_class.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3=haze_class.features.denseblock3
        self.trans_block3=haze_class.features.transition3

        # ############# Block31-down  xx-xx ##############
        self.dense_block31=haze_class.features.denseblock4
        # self.trans_block31=haze_class.features.transition4

        self.dense_norm31=haze_class.features.norm5

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock(512,256)
        self.trans_block4=TransitionBlock(768,128)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock(384,256)
        self.trans_block5=TransitionBlock(640,128)

        ############# Block6-up 32-32   ##############
        self.dense_block6=BottleneckBlock(256,128)
        self.trans_block6=TransitionBlock(384,64)


        ############# Block7-up 64-64   ##############
        self.dense_block7=BottleneckBlock(64,64)
        self.trans_block7=TransitionBlock(128,32)

        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8=BottleneckBlock(32,32)
        self.trans_block8=TransitionBlock(64,16)

        self.conv_refin=nn.Conv2d(19,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(4, 3, kernel_size=3,stride=1,padding=1)

        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)


        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)


        self.res31=BasicBlock_res(512,512)
        self.res32 = BasicBlock_res(512, 512)


        self.res41=BasicBlock_res(384,384)
        self.res42 = BasicBlock_res(384, 384)


        self.res51=BasicBlock_res(256,256)
        self.res52 = BasicBlock_res(256, 256)


        self.res61=BasicBlock_res(64,64)
        self.res62 = BasicBlock_res(64, 64)

        self.res71=BasicBlock_res(32,32)
        self.res72 = BasicBlock_res(32, 32)


        self.resref1 =BasicBlock_res(44,44)
        self.resref2 = BasicBlock_res(44, 44)


    def forward(self, x):
        ## 256x256
        x0=self.pool0(self.relu0(self.norm0(self.conv0(x))))

        ## 64 X 64
        x1=self.dense_block1(x0)
        # print x1.size()
        x1=self.trans_block1(x1)

        ###  32x32
        x2=self.trans_block2(self.dense_block2(x1))
        # print  x2.size()


        ### 16 X 16
        x3=self.trans_block3(self.dense_block3(x2))
        ## Classifier  ##

        ## 8 X 8
        x3 = self.res31(x3)
        x3 = self.res32(x3)
        # print  x3.size()

        ## 8 X 8
        x4 = self.trans_block4(self.dense_block4(x3))
        # print(x4.size())
        x42 = torch.cat([x4,x2], 1)
        # print(x42.size())

        x42 = self.res41(x42)
        x42 = self.res42(x42)

        ## 16 X 16
        x5 = self.trans_block5(self.dense_block5(x42))
        x52 = torch.cat([x5,x1], 1)
        # print(x52.size())



        x52 = self.res51(x52)
        x52 = self.res52(x52)

        ##  32 X 32
        x6 = self.trans_block6(self.dense_block6(x52))

        x6 = self.res61(x6)
        x6 = self.res62(x6)
        # print(x6.size())

        ##  64 X 64
        x7 = self.trans_block7(self.dense_block7(x6))
        x7 = self.res71(x7)
        x7 = self.res72(x7)

        ##  128 X 128
        x8 = self.trans_block8(self.dense_block8(x7))

        # print x8.size()
        # print x.size()

        x8=torch.cat([x8,x],1)

        # print x8.size()

        x9=self.relu(self.batchnorm20(self.conv_refin(x8)))

        shape_out = x9.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)

        x1010 = self.upsample(self.relu((self.conv1010(x101))), size=shape_out)
        x1020 = self.upsample(self.relu((self.conv1020(x102))), size=shape_out)
        x1030 = self.upsample(self.relu((self.conv1030(x103))), size=shape_out)
        x1040 = self.upsample(self.relu((self.conv1040(x104))), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        # dehaze = torch.cat((x1010, x1020, x1030, x1040), 1)

        residual = self.tanh(self.refine3(dehaze))
        # dehaze=x - residual

        return residual







class Dense_rain_cvprw2(nn.Module):
    def __init__(self):
        super(Dense_rain_cvprw2, self).__init__()




        ############# 256-256  ##############
        haze_class = models.densenet121(pretrained=True)

        self.conv0=haze_class.features.conv0
        self.norm0=haze_class.features.norm0
        self.relu0=haze_class.features.relu0
        self.pool0=haze_class.features.pool0

        ############# Block1-down 64-64  ##############
        self.dense_block1=haze_class.features.denseblock1
        self.trans_block1=haze_class.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2=haze_class.features.denseblock2
        self.trans_block2=haze_class.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3=haze_class.features.denseblock3
        self.trans_block3=haze_class.features.transition3

        # ############# Block31-down  xx-xx ##############
        self.dense_block31=haze_class.features.denseblock4
        # self.trans_block31=haze_class.features.transition4

        self.dense_norm31=haze_class.features.norm5

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock(512,256)
        self.trans_block4=TransitionBlock(768,128)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock(384,256)
        self.trans_block5=TransitionBlock(640,128)

        ############# Block6-up 32-32   ##############
        self.dense_block6=BottleneckBlock(256,128)
        self.trans_block6=TransitionBlock(384,64)


        ############# Block7-up 64-64   ##############
        self.dense_block7=BottleneckBlock(64,64)
        self.trans_block7=TransitionBlock(128,32)

        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8=BottleneckBlock(32,32)
        self.trans_block8=TransitionBlock(64,16)

        self.conv_refin=nn.Conv2d(19,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(4, 3, kernel_size=3,stride=1,padding=1)

        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)


        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)


        self.res31=BasicBlock_res(512,512)
        self.res32 = BasicBlock_res(512, 512)


        self.res41=BasicBlock_res(384,384)
        self.res42 = BasicBlock_res(384, 384)


        self.res51=BasicBlock_res(256,256)
        self.res52 = BasicBlock_res(256, 256)


        self.res61=BasicBlock_res(64,64)
        self.res62 = BasicBlock_res(64, 64)

        self.res71=BasicBlock_res(32,32)
        self.res72 = BasicBlock_res(32, 32)


        self.resref1 =BasicBlock_res(44,44)
        self.resref2 = BasicBlock_res(44, 44)


    def forward(self, x):
        ## 256x256
        x0=self.pool0(self.relu0(self.norm0(self.conv0(x))))

        ## 64 X 64
        x1=self.dense_block1(x0)
        # print x1.size()
        x1=self.trans_block1(x1)

        ###  32x32
        x2=self.trans_block2(self.dense_block2(x1))
        # print  x2.size()


        ### 16 X 16
        x3=self.trans_block3(self.dense_block3(x2))
        ## Classifier  ##

        ## 8 X 8
        x3 = self.res31(x3)
        x3 = self.res32(x3)
        # print  x3.size()

        ## 8 X 8
        x4 = self.trans_block4(self.dense_block4(x3))
        # print(x4.size())
        x42 = torch.cat([x4,x2], 1)
        # print(x42.size())

        x42 = self.res41(x42)
        x42 = self.res42(x42)

        ## 16 X 16
        x5 = self.trans_block5(self.dense_block5(x42))
        x52 = torch.cat([x5,x1], 1)
        # print(x52.size())



        x52 = self.res51(x52)
        x52 = self.res52(x52)

        ##  32 X 32
        x6 = self.trans_block6(self.dense_block6(x52))

        x6 = self.res61(x6)
        x6 = self.res62(x6)
        # print(x6.size())

        ##  64 X 64
        x7 = self.trans_block7(self.dense_block7(x6))
        x7 = self.res71(x7)
        x7 = self.res72(x7)

        ##  128 X 128
        x8 = self.trans_block8(self.dense_block8(x7))

        # print x8.size()
        # print x.size()

        x8=torch.cat([x8,x],1)

        # print x8.size()

        x9=self.relu(self.batchnorm20(self.conv_refin(x8)))

        shape_out = x9.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)

        x1010 = self.upsample(self.relu((self.conv1010(x101))), size=shape_out)
        x1020 = self.upsample(self.relu((self.conv1020(x102))), size=shape_out)
        x1030 = self.upsample(self.relu((self.conv1030(x103))), size=shape_out)
        x1040 = self.upsample(self.relu((self.conv1040(x104))), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        # dehaze = torch.cat((x1010, x1020, x1030, x1040), 1)

        residual = self.tanh(self.refine3(dehaze))
        # dehaze=x - residual

        return residual


class Dense_rain_cvprw1(nn.Module):
    def __init__(self):
        super(Dense_rain_cvprw1, self).__init__()




        ############# 256-256  ##############
        haze_class = models.densenet121(pretrained=True)

        self.conv0=haze_class.features.conv0
        self.norm0=haze_class.features.norm0
        self.relu0=haze_class.features.relu0
        self.pool0=haze_class.features.pool0

        ############# Block1-down 64-64  ##############
        self.dense_block1=haze_class.features.denseblock1
        self.trans_block1=haze_class.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2=haze_class.features.denseblock2
        self.trans_block2=haze_class.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3=haze_class.features.denseblock3
        self.trans_block3=haze_class.features.transition3

        # ############# Block31-down  xx-xx ##############
        self.dense_block31=haze_class.features.denseblock4
        # self.trans_block31=haze_class.features.transition4

        self.dense_norm31=haze_class.features.norm5

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock(512,256)
        self.trans_block4=TransitionBlock(768,128)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock(384,256)
        self.trans_block5=TransitionBlock(640,128)

        ############# Block6-up 32-32   ##############
        self.dense_block6=BottleneckBlock(256,128)
        self.trans_block6=TransitionBlock(384,64)


        ############# Block7-up 64-64   ##############
        self.dense_block7=BottleneckBlock(64,64)
        self.trans_block7=TransitionBlock(128,32)

        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8=BottleneckBlock(32,32)
        self.trans_block8=TransitionBlock(64,16)

        self.conv_refin=nn.Conv2d(19,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(4, 3, kernel_size=3,stride=1,padding=1)

        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)


        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)


        self.res31=BasicBlock_res(512,512)
        self.res32 = BasicBlock_res(512, 512)


        self.res41=BasicBlock_res(384,384)
        self.res42 = BasicBlock_res(384, 384)


        self.res51=BasicBlock_res(256,256)
        self.res52 = BasicBlock_res(256, 256)


        self.res61=BasicBlock_res(64,64)
        self.res62 = BasicBlock_res(64, 64)

        self.res71=BasicBlock_res(32,32)
        self.res72 = BasicBlock_res(32, 32)


        self.resref1 =BasicBlock_res(44,44)
        self.resref2 = BasicBlock_res(44, 44)


    def forward(self, x):
        ## 256x256
        x0=self.pool0(self.relu0(self.norm0(self.conv0(x))))

        ## 64 X 64
        x1=self.dense_block1(x0)
        # print x1.size()
        x1=self.trans_block1(x1)

        ###  32x32
        x2=self.trans_block2(self.dense_block2(x1))
        # print  x2.size()


        ### 16 X 16
        x3=self.trans_block3(self.dense_block3(x2))
        ## Classifier  ##

        ## 8 X 8
        x3 = self.res31(x3)
        x3 = self.res32(x3)
        # print  x3.size()

        ## 8 X 8
        x4 = self.trans_block4(self.dense_block4(x3))
        # print(x4.size())
        x42 = torch.cat([x4,x2], 1)
        # print(x42.size())

        x42 = self.res41(x42)
        x42 = self.res42(x42)

        ## 16 X 16
        x5 = self.trans_block5(self.dense_block5(x42))
        x52 = torch.cat([x5,x1], 1)
        # print(x52.size())



        x52 = self.res51(x52)
        x52 = self.res52(x52)

        ##  32 X 32
        x6 = self.trans_block6(self.dense_block6(x52))

        x6 = self.res61(x6)
        x6 = self.res62(x6)
        # print(x6.size())

        ##  64 X 64
        x7 = self.trans_block7(self.dense_block7(x6))
        x7 = self.res71(x7)
        x7 = self.res72(x7)

        ##  128 X 128
        x8 = self.trans_block8(self.dense_block8(x7))

        # print x8.size()
        # print x.size()

        x8=torch.cat([x8,x],1)

        # print x8.size()

        x9=self.relu((self.conv_refin(x8)))

        shape_out = x9.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)

        x1010 = self.upsample(self.relu((self.conv1010(x101))), size=shape_out)
        x1020 = self.upsample(self.relu((self.conv1020(x102))), size=shape_out)
        x1030 = self.upsample(self.relu((self.conv1030(x103))), size=shape_out)
        x1040 = self.upsample(self.relu((self.conv1040(x104))), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        # dehaze = torch.cat((x1010, x1020, x1030, x1040), 1)

        residual = self.tanh(self.refine3(dehaze))
        # dehaze=x - residual

        return residual




class Dense_rain_cvprw3(nn.Module):
    def __init__(self):
        super(Dense_rain_cvprw3, self).__init__()




        ############# 256-256  ##############
        haze_class = models.densenet121(pretrained=True)

        self.conv0=haze_class.features.conv0
        self.norm0=haze_class.features.norm0
        self.relu0=haze_class.features.relu0
        self.pool0=haze_class.features.pool0

        ############# Block1-down 64-64  ##############
        self.dense_block1=haze_class.features.denseblock1
        self.trans_block1=haze_class.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2=haze_class.features.denseblock2
        self.trans_block2=haze_class.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3=haze_class.features.denseblock3
        self.trans_block3=haze_class.features.transition3

        # ############# Block31-down  xx-xx ##############
        self.dense_block31=haze_class.features.denseblock4
        # self.trans_block31=haze_class.features.transition4

        self.dense_norm31=haze_class.features.norm5

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock(512,256)
        self.trans_block4=TransitionBlock(768,128)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock(387,256)
        self.trans_block5=TransitionBlock(643,128)

        ############# Block6-up 32-32   ##############
        self.dense_block6=BottleneckBlock(259,128)
        self.trans_block6=TransitionBlock(387,64)


        ############# Block7-up 64-64   ##############
        self.dense_block7=BottleneckBlock(67,64)
        self.trans_block7=TransitionBlock(131,32)

        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8=BottleneckBlock(35,32)
        self.trans_block8=TransitionBlock(67,16)

        self.conv_refin=nn.Conv2d(19,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        #self.refine3= nn.Conv2d(20, 3, kernel_size=3,stride=1,padding=1)

        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.pool1=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)



        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)


        self.res31=BasicBlock_res(512,512)
        self.res32 = BasicBlock_res(512, 512)


        self.res41=BasicBlock_res(387,387)
        self.res42 = BasicBlock_res(387, 387)


        self.res51=BasicBlock_res(259,259)
        self.res52 = BasicBlock_res(259, 259)


        self.res61=BasicBlock_res(67,67)
        self.res62 = BasicBlock_res(67, 67)

        self.res71=BasicBlock_res(35,35)
        self.res72 = BasicBlock_res(35, 35)


        self.resref1 =BasicBlock_res(44,44)
        self.resref2 = BasicBlock_res(44, 44)


    def forward(self, x):
        ## 256x256
        x0=self.pool0(self.relu0(self.norm0(self.conv0(x))))

        ## 64 X 64
        x1=self.dense_block1(x0)
        # print x1.size()
        x1=self.trans_block1(x1)

        ###  32x32
        x2=self.trans_block2(self.dense_block2(x1))
        # print  x2.size()


        ### 16 X 16
        x3=self.trans_block3(self.dense_block3(x2))
        ## Classifier  ##

        ## 8 X 8
        x3 = self.res31(x3)
        x3 = self.res32(x3)
        # print  x3.size()

        ## 8 X 8
        x4 = self.trans_block4(self.dense_block4(x3))
        # print('x4:',x4.size())
        x43=F.avg_pool2d(x,16)
        # print('x43:',x43.size())


        x42 = torch.cat([x4,x2,x43], 1)
        # print(x42.size())

        x42 = self.res41(x42)
        x42 = self.res42(x42)

        ## 16 X 16
        x5 = self.trans_block5(self.dense_block5(x42))
        x53=F.avg_pool2d(x,8)
        x52 = torch.cat([x5,x1,x53], 1)
        # print('x52.size())



        x52 = self.res51(x52)
        x52 = self.res52(x52)

        ##  32 X 32
        x6 = self.trans_block6(self.dense_block6(x52))
        x63=F.avg_pool2d(x,4)

        x62 = torch.cat([x6,x63], 1)
        x62 = self.res61(x62)
        x6 = self.res62(x62)
        # print('x6:',x6.size())
        # print('x63:',x63.size())

        ##  64 X 64
        x7 = self.trans_block7(self.dense_block7(x6))
        x73=F.avg_pool2d(x,2)
        x72 = torch.cat([x7,x73], 1)

        x72 = self.res71(x72)
        x7 = self.res72(x72)

        ##  128 X 128
        x8 = self.trans_block8(self.dense_block8(x7))

        # print x8.size()
        # print x.size()

        x8=torch.cat([x8,x],1)

        # print x8.size()

        # x9=self.relu((self.conv_refin(x8)))
        x9=self.relu(self.batchnorm20(self.conv_refin(x8)))

        shape_out = x9.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)

        x1010 = self.upsample(self.relu((self.conv1010(x101))), size=shape_out)
        x1020 = self.upsample(self.relu((self.conv1020(x102))), size=shape_out)
        x1030 = self.upsample(self.relu((self.conv1030(x103))), size=shape_out)
        x1040 = self.upsample(self.relu((self.conv1040(x104))), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        #dehaze = torch.cat((x1010, x1020, x1030, x1040), 1)

        residual = self.tanh(self.refine3(dehaze))
        #residual = self.tanh(self.refine3(x9))

        # dehaze=x - residual

        return residual

class Dense_rain_cvprw3crop(nn.Module):
    def __init__(self):
        super(Dense_rain_cvprw3crop, self).__init__()




        ############# 256-256  ##############
        haze_class = models.densenet121(pretrained=True)

        self.conv0=haze_class.features.conv0
        self.norm0=haze_class.features.norm0
        self.relu0=haze_class.features.relu0
        self.pool0=haze_class.features.pool0

        ############# Block1-down 64-64  ##############
        self.dense_block1=haze_class.features.denseblock1
        self.trans_block1=haze_class.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2=haze_class.features.denseblock2
        self.trans_block2=haze_class.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3=haze_class.features.denseblock3
        self.trans_block3=haze_class.features.transition3

        # ############# Block31-down  xx-xx ##############
        self.dense_block31=haze_class.features.denseblock4
        # self.trans_block31=haze_class.features.transition4

        self.dense_norm31=haze_class.features.norm5

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock(512,256)
        self.trans_block4=TransitionBlock(768,128)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock(387,256)
        self.trans_block5=TransitionBlock(643,128)

        ############# Block6-up 32-32   ##############
        self.dense_block6=BottleneckBlock(259,128)
        self.trans_block6=TransitionBlock(387,64)


        ############# Block7-up 64-64   ##############
        self.dense_block7=BottleneckBlock(67,64)
        self.trans_block7=TransitionBlock(131,32)

        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8=BottleneckBlock(35,32)
        self.trans_block8=TransitionBlock(67,16)

        self.conv_refin=nn.Conv2d(19,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        self.refine3= nn.Conv2d(20, 3, kernel_size=3,stride=1,padding=1)

        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.pool1=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)



        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)


        self.res31=BasicBlock_res(512,512)
        self.res32 = BasicBlock_res(512, 512)


        self.res41=BasicBlock_res(387,387)
        self.res42 = BasicBlock_res(387, 387)


        self.res51=BasicBlock_res(259,259)
        self.res52 = BasicBlock_res(259, 259)


        self.res61=BasicBlock_res(67,67)
        self.res62 = BasicBlock_res(67, 67)

        self.res71=BasicBlock_res(35,35)
        self.res72 = BasicBlock_res(35, 35)


        self.resref1 =BasicBlock_res(44,44)
        self.resref2 = BasicBlock_res(44, 44)


    def forward(self, x):
        ## 256x256
        x0=self.pool0(self.relu0(self.norm0(self.conv0(x))))

        ## 64 X 64
        x1=self.dense_block1(x0)
        # print x1.size()
        x1=self.trans_block1(x1)

        ###  32x32
        x2=self.trans_block2(self.dense_block2(x1))
        # print  x2.size()


        ### 16 X 16
        x3=self.trans_block3(self.dense_block3(x2))
        ## Classifier  ##

        ## 8 X 8
        x3 = self.res31(x3)
        x3 = self.res32(x3)
        # print  x3.size()

        ## 8 X 8
        x4 = self.trans_block4(self.dense_block4(x3))
        # print('x4:',x4.size())
        x43=F.avg_pool2d(x,16)
        # print('x43:',x43.size())


        x42 = torch.cat([x4,x2,x43], 1)
        # print(x42.size())

        x42 = self.res41(x42)
        x42 = self.res42(x42)

        ## 16 X 16
        x5 = self.trans_block5(self.dense_block5(x42))
        x53=F.avg_pool2d(x,8)
        x52 = torch.cat([x5,x1,x53], 1)
        # print('x52.size())



        x52 = self.res51(x52)
        x52 = self.res52(x52)

        ##  32 X 32
        x6 = self.trans_block6(self.dense_block6(x52))
        x63=F.avg_pool2d(x,4)

        x62 = torch.cat([x6,x63], 1)
        x62 = self.res61(x62)
        x6 = self.res62(x62)
        # print('x6:',x6.size())
        # print('x63:',x63.size())

        ##  64 X 64
        x7 = self.trans_block7(self.dense_block7(x6))
        x73=F.avg_pool2d(x,2)
        x72 = torch.cat([x7,x73], 1)

        x72 = self.res71(x72)
        x7 = self.res72(x72)

        ##  128 X 128
        x8 = self.trans_block8(self.dense_block8(x7))

        # print x8.size()
        # print x.size()

        x8=torch.cat([x8,x],1)

        # print x8.size()

        # x9=self.relu((self.conv_refin(x8)))
        x9=self.relu(self.conv_refin(x8))

        # shape_out = x9.data.size()
        # # print(shape_out)
        # shape_out = shape_out[2:4]
		#
        # x101 = F.avg_pool2d(x9, 32)
        # x102 = F.avg_pool2d(x9, 16)
        # x103 = F.avg_pool2d(x9, 8)
        # x104 = F.avg_pool2d(x9, 4)
		#
        # x1010 = self.upsample(self.relu((self.conv1010(x101))), size=shape_out)
        # x1020 = self.upsample(self.relu((self.conv1020(x102))), size=shape_out)
        # x1030 = self.upsample(self.relu((self.conv1030(x103))), size=shape_out)
        # x1040 = self.upsample(self.relu((self.conv1040(x104))), size=shape_out)
		#
        # dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        #dehaze = torch.cat((x1010, x1020, x1030, x1040), 1)

        # residual = self.tanh(self.refine3(dehaze))
        residual = self.tanh(self.refine3(x9))

        # dehaze=x - residual

        return residual

class Dense_rain_cvprw3dy(nn.Module):
    def __init__(self):
        super(Dense_rain_cvprw3dy, self).__init__()




        ############# 256-256  ##############
        haze_class = models.densenet121(pretrained=True)

        self.conv0=haze_class.features.conv0
        self.norm0=haze_class.features.norm0
        self.relu0=haze_class.features.relu0
        self.pool0=haze_class.features.pool0

        ############# Block1-down 64-64  ##############
        self.dense_block1=haze_class.features.denseblock1
        self.trans_block1=haze_class.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2=haze_class.features.denseblock2
        self.trans_block2=haze_class.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3=haze_class.features.denseblock3
        self.trans_block3=haze_class.features.transition3

        # ############# Block31-down  xx-xx ##############
        self.dense_block31=haze_class.features.denseblock4
        # self.trans_block31=haze_class.features.transition4

        self.dense_norm31=haze_class.features.norm5

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock(512,256)
        self.trans_block4=TransitionBlock(768,128)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock(387,256)
        self.trans_block5=TransitionBlock(643,128)

        ############# Block6-up 32-32   ##############
        self.dense_block6=BottleneckBlock(259,128)
        self.trans_block6=TransitionBlock(387,64)


        ############# Block7-up 64-64   ##############
        self.dense_block7=BottleneckBlock(67,64)
        self.trans_block7=TransitionBlock(131,32)

        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8=BottleneckBlock(35,32)
        self.trans_block8=TransitionBlock(67,16)

        self.conv_refin=nn.Conv2d(19,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        self.refine3= nn.Conv2d(20, 3, kernel_size=3,stride=1,padding=1)

        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.pool1=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)



        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)


        self.res31=BasicBlock_res(512,512)
        self.res32 = BasicBlock_res(512, 512)


        self.res41=BasicBlock_res(387,387)
        self.res42 = BasicBlock_res(387, 387)


        self.res51=BasicBlock_res(259,259)
        self.res52 = BasicBlock_res(259, 259)


        self.res61=BasicBlock_res(67,67)
        self.res62 = BasicBlock_res(67, 67)

        self.res71=BasicBlock_res(35,35)
        self.res72 = BasicBlock_res(35, 35)


        self.resref1 =BasicBlock_res(44,44)
        self.resref2 = BasicBlock_res(44, 44)


    def forward(self, x):
        ## 256x256
        x0=self.pool0(self.relu0(self.norm0(self.conv0(x))))

        ## 64 X 64
        x1=self.dense_block1(x0)
        # print x1.size()
        x1=self.trans_block1(x1)

        ###  32x32
        x2=self.trans_block2(self.dense_block2(x1))
        # print  x2.size()


        ### 16 X 16
        x3=self.trans_block3(self.dense_block3(x2))
        ## Classifier  ##

        ## 8 X 8
        # x3 = self.res31(x3)
        # x3 = self.res32(x3)
        # print  x3.size()

        ## 8 X 8
        x4 = self.trans_block4(self.dense_block4(x3))
        # print('x4:',x4.size())
        x43=F.avg_pool2d(x,16)
        # print('x43:',x43.size())


        x42 = torch.cat([x4,x2,x43], 1)
        # print(x42.size())

        # x42 = self.res41(x42)
        # x42 = self.res42(x42)

        ## 16 X 16
        x5 = self.trans_block5(self.dense_block5(x42))
        x53=F.avg_pool2d(x,8)
        x52 = torch.cat([x5,x1,x53], 1)
        # print('x52.size())



        # x52 = self.res51(x52)
        # x52 = self.res52(x52)

        ##  32 X 32
        x6 = self.trans_block6(self.dense_block6(x52))
        x63=F.avg_pool2d(x,4)

        x62 = torch.cat([x6,x63], 1)
        # x62 = self.res61(x62)
        # x6 = self.res62(x62)
        # print('x6:',x6.size())
        # print('x63:',x63.size())

        ##  64 X 64
        x7 = self.trans_block7(self.dense_block7(x62))
        x73=F.avg_pool2d(x,2)
        x72 = torch.cat([x7,x73], 1)

        # x72 = self.res71(x72)
        # x7 = self.res72(x72)

        ##  128 X 128
        x8 = self.trans_block8(self.dense_block8(x72))

        # print x8.size()
        # print x.size()

        x8=torch.cat([x8,x],1)

        # print x8.size()

        # x9=self.relu((self.conv_refin(x8)))
        x9=self.relu(self.conv_refin(x8))

        shape_out = x9.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        # x101 = F.avg_pool2d(x9, 32)
        # x102 = F.avg_pool2d(x9, 16)
        # x103 = F.avg_pool2d(x9, 8)
        # x104 = F.avg_pool2d(x9, 4)

        # x1010 = self.upsample(self.relu((self.conv1010(x101))), size=shape_out)
        # x1020 = self.upsample(self.relu((self.conv1020(x102))), size=shape_out)
        # x1030 = self.upsample(self.relu((self.conv1030(x103))), size=shape_out)
        # x1040 = self.upsample(self.relu((self.conv1040(x104))), size=shape_out)

        # dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        #dehaze = torch.cat((x1010, x1020, x1030, x1040), 1)

        # residual = self.tanh(self.refine3(dehaze))
        residual = self.tanh(self.refine3(x9))

        # dehaze=x - residual

        return residual


class Dense_rain_cvprw3dy1(nn.Module):
    def __init__(self):
        super(Dense_rain_cvprw3dy1, self).__init__()




        ############# 256-256  ##############
        haze_class = models.densenet121(pretrained=True)

        self.conv0=haze_class.features.conv0
        self.norm0=haze_class.features.norm0
        self.relu0=haze_class.features.relu0
        self.pool0=haze_class.features.pool0

        ############# Block1-down 64-64  ##############
        self.dense_block1=haze_class.features.denseblock1
        self.trans_block1=haze_class.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2=haze_class.features.denseblock2
        self.trans_block2=haze_class.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3=haze_class.features.denseblock3
        self.trans_block3=haze_class.features.transition3

        # ############# Block31-down  xx-xx ##############
        self.dense_block31=haze_class.features.denseblock4
        # self.trans_block31=haze_class.features.transition4

        self.dense_norm31=haze_class.features.norm5

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlockdy(512,256)
        self.trans_block4=TransitionBlockdy(768,128)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlockdy(387,256)
        self.trans_block5=TransitionBlockdy(643,128)

        ############# Block6-up 32-32   ##############
        self.dense_block6=BottleneckBlockdy(259,128)
        self.trans_block6=TransitionBlockdy(387,64)


        ############# Block7-up 64-64   ##############
        self.dense_block7=BottleneckBlockdy(67,64)
        self.trans_block7=TransitionBlockdy(131,32)

        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8=BottleneckBlockdy(35,32)
        self.trans_block8=TransitionBlockdy(67,16)

        self.conv_refin=nn.Conv2d(19,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        self.refine3= nn.Conv2d(20, 3, kernel_size=3,stride=1,padding=1)

        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.pool1=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)



        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)


        self.res31=BasicBlock_res(512,512)
        self.res32 = BasicBlock_res(512, 512)


        self.res41=BasicBlock_res(387,387)
        self.res42 = BasicBlock_res(387, 387)


        self.res51=BasicBlock_res(259,259)
        self.res52 = BasicBlock_res(259, 259)


        self.res61=BasicBlock_res(67,67)
        self.res62 = BasicBlock_res(67, 67)

        self.res71=BasicBlock_res(35,35)
        self.res72 = BasicBlock_res(35, 35)


        self.resref1 =BasicBlock_res(44,44)
        self.resref2 = BasicBlock_res(44, 44)


    def forward(self, x):
        ## 256x256
        x0=self.pool0(self.relu0(self.norm0(self.conv0(x))))

        ## 64 X 64
        x1=self.dense_block1(x0)
        # print x1.size()
        x1=self.trans_block1(x1)

        ###  32x32
        x2=self.trans_block2(self.dense_block2(x1))
        # print  x2.size()


        ### 16 X 16
        x3=self.trans_block3(self.dense_block3(x2))
        ## Classifier  ##

        ## 8 X 8
        # x3 = self.res31(x3)
        # x3 = self.res32(x3)
        # print  x3.size()

        ## 8 X 8
        x4 = self.trans_block4(self.dense_block4(x3))
        # print('x4:',x4.size())
        x43=F.avg_pool2d(x,16)
        # print('x43:',x43.size())


        x42 = torch.cat([x4,x2,x43], 1)
        # print(x42.size())

        # x42 = self.res41(x42)
        # x42 = self.res42(x42)

        ## 16 X 16
        x5 = self.trans_block5(self.dense_block5(x42))
        x53=F.avg_pool2d(x,8)
        x52 = torch.cat([x5,x1,x53], 1)
        # print('x52.size())



        # x52 = self.res51(x52)
        # x52 = self.res52(x52)

        ##  32 X 32
        x6 = self.trans_block6(self.dense_block6(x52))
        x63=F.avg_pool2d(x,4)

        x62 = torch.cat([x6,x63], 1)
        # x62 = self.res61(x62)
        # x6 = self.res62(x62)
        # print('x6:',x6.size())
        # print('x63:',x63.size())

        ##  64 X 64
        x7 = self.trans_block7(self.dense_block7(x62))
        x73=F.avg_pool2d(x,2)
        x72 = torch.cat([x7,x73], 1)

        # x72 = self.res71(x72)
        # x7 = self.res72(x72)

        ##  128 X 128
        x8 = self.trans_block8(self.dense_block8(x72))

        # print x8.size()
        # print x.size()

        x8=torch.cat([x8,x],1)

        # print x8.size()

        # x9=self.relu((self.conv_refin(x8)))
        x9=self.relu(self.conv_refin(x8))

        shape_out = x9.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        # x101 = F.avg_pool2d(x9, 32)
        # x102 = F.avg_pool2d(x9, 16)
        # x103 = F.avg_pool2d(x9, 8)
        # x104 = F.avg_pool2d(x9, 4)

        # x1010 = self.upsample(self.relu((self.conv1010(x101))), size=shape_out)
        # x1020 = self.upsample(self.relu((self.conv1020(x102))), size=shape_out)
        # x1030 = self.upsample(self.relu((self.conv1030(x103))), size=shape_out)
        # x1040 = self.upsample(self.relu((self.conv1040(x104))), size=shape_out)

        # dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        #dehaze = torch.cat((x1010, x1020, x1030, x1040), 1)

        # residual = self.tanh(self.refine3(dehaze))
        residual = self.tanh(self.refine3(x9))

        # dehaze=x - residual

        return residual


class Dense_rain_cvprw3dy2(nn.Module):
    def __init__(self):
        super(Dense_rain_cvprw3dy2, self).__init__()




        ############# 256-256  ##############
        haze_class = models.densenet121(pretrained=True)

        self.conv0=haze_class.features.conv0
        self.norm0=haze_class.features.norm0
        self.relu0=haze_class.features.relu0
        self.pool0=haze_class.features.pool0

        ############# Block1-down 64-64  ##############
        self.dense_block1=haze_class.features.denseblock1
        self.trans_block1=haze_class.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2=haze_class.features.denseblock2
        self.trans_block2=haze_class.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3=haze_class.features.denseblock3
        self.trans_block3=haze_class.features.transition3

        # ############# Block31-down  xx-xx ##############
        self.dense_block31=haze_class.features.denseblock4
        # self.trans_block31=haze_class.features.transition4

        self.dense_norm31=haze_class.features.norm5

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlockdy(512,256)
        self.trans_block4=TransitionBlockdy(768,128)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlockdy(387,256)
        self.trans_block5=TransitionBlockdy(643,128)

        ############# Block6-up 32-32   ##############
        self.dense_block6=BottleneckBlockdy(259,128)
        self.trans_block6=TransitionBlockdy(387,64)


        ############# Block7-up 64-64   ##############
        self.dense_block7=BottleneckBlockdy(67,64)
        self.trans_block7=TransitionBlockdy(131,32)

        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8=BottleneckBlockdy(35,32)
        self.trans_block8=TransitionBlockdy(67,16)

        self.conv_refin=nn.Conv2d(19,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        self.refine3= nn.Conv2d(22, 3, kernel_size=3,stride=1,padding=1)

        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.pool1=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)



        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)


        self.res31=BasicBlock_res(512,512)
        self.res32 = BasicBlock_res(512, 512)


        self.res41=BasicBlock_res(387,387)
        self.res42 = BasicBlock_res(387, 387)


        self.res51=BasicBlock_res(259,259)
        self.res52 = BasicBlock_res(259, 259)


        self.res61=BasicBlock_res(67,67)
        self.res62 = BasicBlock_res(67, 67)

        self.res71=BasicBlock_res(35,35)
        self.res72 = BasicBlock_res(35, 35)


        self.resref1 =BasicBlock_res(44,44)
        self.resref2 = BasicBlock_res(44, 44)


    def forward(self, x):
        ## 256x256
        x0=self.pool0(self.relu0(self.norm0(self.conv0(x))))

        ## 64 X 64
        x1=self.dense_block1(x0)
        # print x1.size()
        x1=self.trans_block1(x1)

        ###  32x32
        x2=self.trans_block2(self.dense_block2(x1))
        # print  x2.size()


        ### 16 X 16
        x3=self.trans_block3(self.dense_block3(x2))
        ## Classifier  ##

        ## 8 X 8
        # x3 = self.res31(x3)
        # x3 = self.res32(x3)
        # print  x3.size()

        ## 8 X 8
        x4 = self.trans_block4(self.dense_block4(x3))
        # print('x4:',x4.size())
        x43=F.avg_pool2d(x,16)
        # print('x43:',x43.size())


        x42 = torch.cat([x4,x2,x43], 1)
        # print(x42.size())

        # x42 = self.res41(x42)
        # x42 = self.res42(x42)

        ## 16 X 16
        x5 = self.trans_block5(self.dense_block5(x42))
        x53=F.avg_pool2d(x,8)
        x52 = torch.cat([x5,x1,x53], 1)
        # print('x52.size())



        # x52 = self.res51(x52)
        # x52 = self.res52(x52)

        ##  32 X 32
        x6 = self.trans_block6(self.dense_block6(x52))
        x63=F.avg_pool2d(x,4)

        x62 = torch.cat([x6,x63], 1)
        # x62 = self.res61(x62)
        # x6 = self.res62(x62)
        # print('x6:',x6.size())
        # print('x63:',x63.size())

        ##  64 X 64
        x7 = self.trans_block7(self.dense_block7(x62))
        x73=F.avg_pool2d(x,2)
        x72 = torch.cat([x7,x73], 1)

        # x72 = self.res71(x72)
        # x7 = self.res72(x72)

        ##  128 X 128
        x8 = self.trans_block8(self.dense_block8(x72))

        # print x8.size()
        # print x.size()

        x8=torch.cat([x8,x],1)

        # print x8.size()

        # x9=self.relu((self.conv_refin(x8)))
        x9=self.relu(self.conv_refin(x8))

        shape_out = x9.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        # x101 = F.avg_pool2d(x9, 32)
        # x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)

        # x1010 = self.upsample(self.relu((self.conv1010(x101))), size=shape_out)
        # x1020 = self.upsample(self.relu((self.conv1020(x102))), size=shape_out)
        x1030 = self.upsample(self.relu((self.conv1030(x103))), size=shape_out)
        x1040 = self.upsample(self.relu((self.conv1040(x104))), size=shape_out)

        dehaze = torch.cat((x1030, x1040, x9), 1)
        #dehaze = torch.cat((x1010, x1020, x1030, x1040), 1)

        # residual = self.tanh(self.refine3(dehaze))
        residual = self.tanh(self.refine3(dehaze))

        # dehaze=x - residual

        return residual




class Dense_rain_cvprw3dy3(nn.Module):
    def __init__(self):
        super(Dense_rain_cvprw3dy3, self).__init__()




        ############# 256-256  ##############
        haze_class = models.densenet121(pretrained=True)

        self.conv0=haze_class.features.conv0
        self.norm0=haze_class.features.norm0
        self.relu0=haze_class.features.relu0
        self.pool0=haze_class.features.pool0

        ############# Block1-down 64-64  ##############
        self.dense_block1=haze_class.features.denseblock1
        self.trans_block1=haze_class.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2=haze_class.features.denseblock2
        self.trans_block2=haze_class.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3=haze_class.features.denseblock3
        self.trans_block3=haze_class.features.transition3

        # ############# Block31-down  xx-xx ##############
        self.dense_block31=haze_class.features.denseblock4
        # self.trans_block31=haze_class.features.transition4

        self.dense_norm31=haze_class.features.norm5

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlockdy(512,256)
        self.trans_block4=TransitionBlockdy(768,128)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlockdy(387,256)
        self.trans_block5=TransitionBlockdy(643,128)

        ############# Block6-up 32-32   ##############
        self.dense_block6=BottleneckBlockdy(259,128)
        self.trans_block6=TransitionBlockdy(387,64)


        ############# Block7-up 64-64   ##############
        self.dense_block7=BottleneckBlockdy(67,64)
        self.trans_block7=TransitionBlockdy(131,32)

        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8=BottleneckBlockdy(35,32)
        self.trans_block8=TransitionBlockdy(67,16)

        self.conv_refin=nn.Conv2d(19,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        self.refine3= nn.Conv2d(22, 3, kernel_size=3,stride=1,padding=1)

        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.pool1=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)



        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)


        self.res31=BasicBlock_res(512,512)
        self.res32 = BasicBlock_res(512, 512)


        self.res41=BasicBlock_res(387,387)
        self.res42 = BasicBlock_res(387, 387)


        self.res51=BasicBlock_res(259,259)
        self.res52 = BasicBlock_res(259, 259)


        self.res61=BasicBlock_res(67,67)
        self.res62 = BasicBlock_res(67, 67)

        self.res71=BasicBlock_res(35,35)
        self.res72 = BasicBlock_res(35, 35)


        self.resref1 =BasicBlock_res(44,44)
        self.resref2 = BasicBlock_res(44, 44)


    def forward(self, x):
        ## 256x256
        x0=self.pool0(self.relu0(self.conv0(x)))

        ## 64 X 64
        x1=self.dense_block1(x0)
        # print x1.size()
        x1=self.trans_block1(x1)

        ###  32x32
        x2=self.trans_block2(self.dense_block2(x1))
        # print  x2.size()


        ### 16 X 16
        x3=self.trans_block3(self.dense_block3(x2))
        ## Classifier  ##

        ## 8 X 8
        # x3 = self.res31(x3)
        # x3 = self.res32(x3)
        # print  x3.size()

        ## 8 X 8
        x4 = self.trans_block4(self.dense_block4(x3))
        # print('x4:',x4.size())
        x43=F.avg_pool2d(x,16)
        # print('x43:',x43.size())


        x42 = torch.cat([x4,x2,x43], 1)
        # print(x42.size())

        # x42 = self.res41(x42)
        # x42 = self.res42(x42)

        ## 16 X 16
        x5 = self.trans_block5(self.dense_block5(x42))
        x53=F.avg_pool2d(x,8)
        x52 = torch.cat([x5,x1,x53], 1)
        # print('x52.size())



        # x52 = self.res51(x52)
        # x52 = self.res52(x52)

        ##  32 X 32
        x6 = self.trans_block6(self.dense_block6(x52))
        x63=F.avg_pool2d(x,4)

        x62 = torch.cat([x6,x63], 1)
        # x62 = self.res61(x62)
        # x6 = self.res62(x62)
        # print('x6:',x6.size())
        # print('x63:',x63.size())

        ##  64 X 64
        x7 = self.trans_block7(self.dense_block7(x62))
        x73=F.avg_pool2d(x,2)
        x72 = torch.cat([x7,x73], 1)

        # x72 = self.res71(x72)
        # x7 = self.res72(x72)

        ##  128 X 128
        x8 = self.trans_block8(self.dense_block8(x72))

        # print x8.size()
        # print x.size()

        x8=torch.cat([x8,x],1)

        # print x8.size()

        # x9=self.relu((self.conv_refin(x8)))
        x9=self.relu(self.conv_refin(x8))

        shape_out = x9.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        # x101 = F.avg_pool2d(x9, 32)
        # x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)

        # x1010 = self.upsample(self.relu((self.conv1010(x101))), size=shape_out)
        # x1020 = self.upsample(self.relu((self.conv1020(x102))), size=shape_out)
        x1030 = self.upsample(self.relu((self.conv1030(x103))), size=shape_out)
        x1040 = self.upsample(self.relu((self.conv1040(x104))), size=shape_out)

        dehaze = torch.cat((x1030, x1040, x9), 1)
        #dehaze = torch.cat((x1010, x1020, x1030, x1040), 1)

        # residual = self.tanh(self.refine3(dehaze))
        residual = self.tanh(self.refine3(dehaze))

        # dehaze=x - residual

        return residual


class Dense_rain_cvprw3dy4(nn.Module):
    def __init__(self):
        super(Dense_rain_cvprw3dy4, self).__init__()

        ############# 256-256  ##############
        haze_class = models.densenet121(pretrained=True)

        self.conv0 = haze_class.features.conv0
        self.relu0 = haze_class.features.relu0

        ############# Block1-down 256-256  ##############
        self.dense_block1 = haze_class.features.denseblock1
        self.trans_block1 = haze_class.features.transition1

        ############# Block2-down 128-128  ##############
        self.dense_block2 = haze_class.features.denseblock2
        self.trans_block2 = haze_class.features.transition2

        ############# Block3-down  64-64 ##############
        self.dense_block3 = haze_class.features.denseblock3
        self.trans_block3 = haze_class.features.transition3

        # ############# Block31-down  xx-xx ##############
        self.dense_block31 = haze_class.features.denseblock4
        # self.trans_block31=haze_class.features.transition4

        self.dense_norm31 = haze_class.features.norm5

        ############# Block4-up  8-8  ##############
        self.dense_block4 = BottleneckBlockdy(512, 256)
        self.trans_block4 = TransitionBlockdy(768, 128)

        ############# Block5-up  16-16 ##############
        self.dense_block5 = BottleneckBlockdy(384, 128)
        self.trans_block5 = TransitionBlockdy(512, 64)

        ############# Block6-up 32-32   ##############
        self.dense_block6 = BottleneckBlockdy(192, 32)
        self.trans_block6 = TransitionBlockdy(224, 16)

        ############# Block7-up 64-64   ##############

        self.conv_refin1 = nn.Conv2d(3, 64, 3, 1, 1)

        self.tanh = nn.Tanh()
        self.refine3 = nn.Conv2d(19, 3, kernel_size=3, stride=1, padding=1)

        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        ## 256*256
        x0 = self.relu0(self.conv_refin1(x))
        # print ('x0.size:',x0.size())

        ## 256*256
        x1 = self.dense_block1(x0)
        # print x1.size()
        x1 = self.trans_block1(x1)
        # print ('x1.size:', x1.size())

        ###  128*128
        x2 = self.trans_block2(self.dense_block2(x1))
        # print  x2.size()
        # print ('x2.size:', x2.size())

        ### 64X 64
        x3 = self.trans_block3(self.dense_block3(x2))
        # print ('x3.size:', x3.size())

        ## 32 X 32
        x4 = self.trans_block4(self.dense_block4(x3))
        # print('x4:',x4.size())

        x42 = torch.cat([x4, x2], 1)
        # print(x42.size())

        ## 64 X 64
        x5 = self.trans_block5(self.dense_block5(x42))
        x52 = torch.cat([x5, x1], 1)
        # print('x52.size())

        ##  128 X 128
        x6 = self.trans_block6(self.dense_block6(x52))

        x62 = torch.cat([x6, x], 1)

        # print('x62:',x62.size())

        ##  256 X 256

        dehaze = self.tanh(self.refine3(x62))

        # dehaze=x - residual

        return dehaze


class Dense_rain_cvprw3dy5(nn.Module):
	def __init__(self):
		super(Dense_rain_cvprw3dy5, self).__init__()

		############# 256-256  ##############
		haze_class = models.densenet121(pretrained=True)

		self.conv0 = haze_class.features.conv0
		self.relu0 = haze_class.features.relu0

		############# Block1-down 256-256  ##############
		self.dense_block1 = haze_class.features.denseblock1
		self.trans_block1 = haze_class.features.transition1

		############# Block2-down 128-128  ##############
		self.dense_block2 = haze_class.features.denseblock2
		self.trans_block2 = haze_class.features.transition2

		############# Block3-down  64-64 ##############
		self.dense_block3 = haze_class.features.denseblock3
		self.trans_block3 = haze_class.features.transition3

		# ############# Block31-down  xx-xx ##############
		self.dense_block31 = haze_class.features.denseblock4
		# self.trans_block31=haze_class.features.transition4

		self.dense_norm31 = haze_class.features.norm5

		############# Block4-up  8-8  ##############
		self.dense_block4 = BottleneckBlockdy(512, 256)
		self.trans_block4 = TransitionBlockdy(768, 128)

		############# Block5-up  16-16 ##############
		self.dense_block5 = BottleneckBlockdy(384, 128)
		self.trans_block5 = TransitionBlockdy(512, 64)

		############# Block6-up 32-32   ##############
		self.dense_block6 = BottleneckBlockdy(192, 32)
		self.trans_block6 = TransitionBlockdy(224, 16)

		############# Block7-up 64-64   ##############

		self.conv_refin1 = nn.Conv2d(3, 64, 3, 1, 1)

		self.tanh = nn.Tanh()
		self.refine3 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

		# self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

		self.upsample = F.upsample_nearest

		self.relu = nn.LeakyReLU(0.2, inplace=True)

	def forward(self, x):
		## 256*256
		x0 = self.relu0(self.conv_refin1(x))
		# print ('x0.size:',x0.size())

		## 256*256
		x1 = self.dense_block1(x0)
		# print x1.size()
		x1 = self.trans_block1(x1)
		# print ('x1.size:', x1.size())

		###  128*128
		x2 = self.trans_block2(self.dense_block2(x1))
		# print  x2.size()
		# print ('x2.size:', x2.size())

		### 64X 64
		x3 = self.trans_block3(self.dense_block3(x2))
		# print ('x3.size:', x3.size())

		## 32 X 32
		x4 = self.trans_block4(self.dense_block4(x3))
		# print('x4:',x4.size())

		x42 = torch.cat([x4, x2], 1)
		# print(x42.size())

		## 64 X 64
		x5 = self.trans_block5(self.dense_block5(x42))
		x52 = torch.cat([x5, x1], 1)
		# print('x52.size())

		##  128 X 128
		x6 = self.trans_block6(self.dense_block6(x52))

		##  256 X 256

		dehaze = self.tanh(self.refine3(x6))

		# dehaze=x - residual

		return dehaze


class Dense_rain_cvprw3dy6(nn.Module):
	def __init__(self):
		super(Dense_rain_cvprw3dy6, self).__init__()

		############# 256-256  ##############
		haze_class = models.densenet121(pretrained=True)

		self.conv0 = haze_class.features.conv0
		self.norm0 = haze_class.features.norm0
		self.relu0 = haze_class.features.relu0
		self.pool0 = haze_class.features.pool0

		############# Block1-down 64-64  ##############
		self.dense_block1 = haze_class.features.denseblock1
		self.trans_block1 = haze_class.features.transition1

		############# Block2-down 32-32  ##############
		self.dense_block2 = haze_class.features.denseblock2
		self.trans_block2 = haze_class.features.transition2

		############# Block3-down  16-16 ##############
		self.dense_block3 = haze_class.features.denseblock3
		self.trans_block3 = haze_class.features.transition3

		# ############# Block31-down  xx-xx ##############
		self.dense_block31 = haze_class.features.denseblock4
		# self.trans_block31=haze_class.features.transition4

		self.dense_norm31 = haze_class.features.norm5

		############# Block4-up  8-8  ##############
		self.dense_block4 = BottleneckBlockdy(512, 256)
		self.trans_block4 = TransitionBlockdy(768, 128)

		############# Block5-up  16-16 ##############
		self.dense_block5 = BottleneckBlockdy(387, 256)
		self.trans_block5 = TransitionBlockdy(643, 128)

		############# Block6-up 32-32   ##############
		self.dense_block6 = BottleneckBlockdy(259, 128)
		self.trans_block6 = TransitionBlockdy(387, 64)

		############# Block7-up 64-64   ##############
		self.dense_block7 = BottleneckBlockdy(67, 64)
		self.trans_block7 = TransitionBlockdy(131, 32)

		## 128 X  128
		############# Block8-up c  ##############
		self.dense_block8 = BottleneckBlockdy(35, 32)
		self.trans_block8 = TransitionBlockdy(67, 16)

		self.conv_refin = nn.Conv2d(19, 20, 3, 1, 1)
		self.tanh = nn.Tanh()

		self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
		self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
		self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
		self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm

		# self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
		self.refine3 = nn.Conv2d(22, 3, kernel_size=3, stride=1, padding=1)

		# self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

		self.upsample = F.upsample_nearest

		self.relu = nn.LeakyReLU(0.2, inplace=True)
		self.pool1 = nn.AvgPool2d(3, stride=2)
		self.pool2 = nn.AvgPool2d(3, stride=2)
		self.pool2 = nn.AvgPool2d(3, stride=2)
		self.pool2 = nn.AvgPool2d(3, stride=2)

		self.batchnorm20 = nn.BatchNorm2d(20)
		self.batchnorm1 = nn.BatchNorm2d(1)

		self.res31 = BasicBlock_res(512, 512)
		self.res32 = BasicBlock_res(512, 512)

		self.res41 = BasicBlock_res(387, 387)
		self.res42 = BasicBlock_res(387, 387)

		self.res51 = BasicBlock_res(259, 259)
		self.res52 = BasicBlock_res(259, 259)

		self.res61 = BasicBlock_res(67, 67)
		self.res62 = BasicBlock_res(67, 67)

		self.res71 = BasicBlock_res(35, 35)
		self.res72 = BasicBlock_res(35, 35)

		self.resref1 = BasicBlock_res(44, 44)
		self.resref2 = BasicBlock_res(44, 44)

	def forward(self, x):
		## 256x256
		x0 = self.pool0(self.relu0(self.conv0(x)))

		## 64 X 64
		x1 = self.dense_block1(x0)
		# print x1.size()
		x1 = self.trans_block1(x1)

		###  32x32
		x2 = self.trans_block2(self.dense_block2(x1))
		# print  x2.size()

		### 16 X 16
		x3 = self.trans_block3(self.dense_block3(x2))
		## Classifier  ##

		## 8 X 8
		# x3 = self.res31(x3)
		# x3 = self.res32(x3)
		# print  x3.size()

		## 8 X 8
		x4 = self.trans_block4(self.dense_block4(x3))
		# print('x4:',x4.size())
		x43 = F.avg_pool2d(x, 16)
		# print('x43:',x43.size())

		x42 = torch.cat([x4, x2, x43], 1)
		# print(x42.size())

		# x42 = self.res41(x42)
		# x42 = self.res42(x42)

		## 16 X 16
		x5 = self.trans_block5(self.dense_block5(x42))
		x53 = F.avg_pool2d(x, 8)
		x52 = torch.cat([x5, x1, x53], 1)
		# print('x52.size())

		# x52 = self.res51(x52)
		# x52 = self.res52(x52)

		##  32 X 32
		x6 = self.trans_block6(self.dense_block6(x52))
		x63 = F.avg_pool2d(x, 4)

		x62 = torch.cat([x6, x63], 1)
		# x62 = self.res61(x62)
		# x6 = self.res62(x62)
		# print('x6:',x6.size())
		# print('x63:',x63.size())

		##  64 X 64
		x7 = self.trans_block7(self.dense_block7(x62))
		x73 = F.avg_pool2d(x, 2)
		x72 = torch.cat([x7, x73], 1)

		# x72 = self.res71(x72)
		# x7 = self.res72(x72)

		##  128 X 128
		x8 = self.trans_block8(self.dense_block8(x72))

		# print x8.size()
		# print x.size()

		x8 = torch.cat([x8, x], 1)

		# print x8.size()

		# x9=self.relu((self.conv_refin(x8)))
		x9 = self.relu(self.conv_refin(x8))

		shape_out = x9.data.size()
		# print(shape_out)
		shape_out = shape_out[2:4]

		# x101 = F.avg_pool2d(x9, 32)
		# x102 = F.avg_pool2d(x9, 16)
		x103 = F.avg_pool2d(x9, 4)
		x104 = F.avg_pool2d(x9, 2)

		# x1010 = self.upsample(self.relu((self.conv1010(x101))), size=shape_out)
		# x1020 = self.upsample(self.relu((self.conv1020(x102))), size=shape_out)
		x1030 = self.upsample(self.relu((self.conv1030(x103))), size=shape_out)
		x1040 = self.upsample(self.relu((self.conv1040(x104))), size=shape_out)

		dehaze = torch.cat((x1030, x1040, x9), 1)
		# dehaze = torch.cat((x1010, x1020, x1030, x1040), 1)

		# residual = self.tanh(self.refine3(dehaze))
		residual = self.tanh(self.refine3(dehaze))

		# dehaze=x - residual

		return residual


class Dense_rain_cvprw3dy7(nn.Module):
	def __init__(self):
		super(Dense_rain_cvprw3dy7, self).__init__()

		############# 256-256  ##############
		haze_class = models.densenet121(pretrained=True)

		self.conv0 = haze_class.features.conv0
		self.relu0 = haze_class.features.relu0

		############# Block1-down 256-256  ##############
		self.dense_block1 = haze_class.features.denseblock1
		self.trans_block1 = haze_class.features.transition1

		############# Block2-down 128-128  ##############
		self.dense_block2 = haze_class.features.denseblock2
		self.trans_block2 = haze_class.features.transition2

		############# Block3-down  64-64 ##############
		self.dense_block3 = haze_class.features.denseblock3
		self.trans_block3 = haze_class.features.transition3

		# ############# Block31-down  xx-xx ##############
		self.dense_block31 = haze_class.features.denseblock4
		# self.trans_block31=haze_class.features.transition4

		self.dense_norm31 = haze_class.features.norm5

		############# Block4-up  8-8  ##############
		self.dense_block4 = BottleneckBlockdy(512, 256)
		self.trans_block4 = TransitionBlockdy(768, 128)

		############# Block5-up  16-16 ##############
		self.dense_block5 = BottleneckBlockdy(384, 128)
		self.trans_block5 = TransitionBlockdy(512, 64)

		############# Block6-up 32-32   ##############
		self.dense_block6 = BottleneckBlockdy(192, 32)
		self.trans_block6 = TransitionBlockdy(224, 64)

		############# Block7-up 64-64   ##############

		self.conv_refin1 = nn.Conv2d(3, 64, 3, 1, 1)

		self.tanh = nn.Tanh()
		self.refine3 = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1)

		# self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

		self.upsample = F.upsample_nearest

		self.relu = nn.LeakyReLU(0.2, inplace=True)

	def forward(self, x):
		## 256*256
		x0 = self.relu0(self.conv_refin1(x))
		#print ('x0.size:',x0.size())

		## 256*256
		x1 = self.dense_block1(x0)
		#print x1.size()
		x1 = self.trans_block1(x1)
		#print ('x1.size:', x1.size())

		###  128*128
		x2 = self.trans_block2(self.dense_block2(x1))
		# print  x2.size()
		# print ('x2.size:', x2.size())

		### 64X 64
		x3 = self.trans_block3(self.dense_block3(x2))
		# print ('x3.size:', x3.size())

		## 32 X 32
		x4 = self.trans_block4(self.dense_block4(x3))
		# print('x4:',x4.size())

		x42 = torch.cat([x4, x2], 1)
		# print(x42.size())

		## 64 X 64
		x5 = self.trans_block5(self.dense_block5(x42))
		x52 = torch.cat([x5, x1], 1)
		# print('x52.size:',x52.size())

		##  128 X 128
		x6 = self.trans_block6(self.dense_block6(x52))
		# print ('x6.size:', x6.size())
		##  256 X 256
		x62 = torch.cat([x6, x0], 1)
		# print ('x62.size:', x62.size())
		dehaze = self.tanh(self.refine3(x62))

		# dehaze=x - residual

		return dehaze



class Dense_rain_cvprw3dy8(nn.Module):
	def __init__(self):
		super(Dense_rain_cvprw3dy8, self).__init__()

		############# 256-256  ##############
		haze_class = models.densenet121(pretrained=True)

		self.conv0 = haze_class.features.conv0
		self.relu0 = haze_class.features.relu0

		############# Block1-down 256-256  ##############
		self.dense_block1 = haze_class.features.denseblock1
		self.trans_block1 = haze_class.features.transition1

		############# Block2-down 128-128  ##############
		self.dense_block2 = haze_class.features.denseblock2
		self.trans_block2 = haze_class.features.transition2



		############# Block4-up  8-8  ##############
		self.dense_block3 = BottleneckBlockdy(256, 128)
		self.trans_block3 = TransitionBlockdy(384, 64)

		############# Block5-up  16-16 ##############
		self.dense_block4 = BottleneckBlockdy(192, 32)
		self.trans_block4 = TransitionBlockdy(224, 16)



		self.conv_refin1 = nn.Conv2d(3, 64, 3, 1, 1)

		self.tanh = nn.Tanh()
		self.refine3 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

		# self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

		self.upsample = F.upsample_nearest

		self.relu = nn.LeakyReLU(0.2, inplace=True)

	def forward(self, x):
		## 256*256
		x0 = self.relu0(self.conv_refin1(x))
		# print ('x0.size:',x0.size())


		x1 = self.dense_block1(x0)
		# print x1.size()
		x1 = self.trans_block1(x1)
		# print ('x1.size:', x1.size())

		x2 = self.trans_block2(self.dense_block2(x1))
		# print  x2.size()
		# print ('x2.size:', x2.size())


		x3 = self.trans_block3(self.dense_block3(x2))
		# print ('x3.size:', x3.size())



		x4= torch.cat([x3, x1], 1)
		# print('x4:',x4.size())


		x4 = self.trans_block4(self.dense_block4(x4))


		dehaze = self.tanh(self.refine3(x4))

		# dehaze=x - residual

		return dehaze


class Dense_rain_cvprw3dy9(nn.Module):
    def __init__(self):
        super(Dense_rain_cvprw3dy9, self).__init__()

        ############# 256-256  ##############
        haze_class = models.densenet121(pretrained=True)

        self.conv0 = haze_class.features.conv0
        self.relu0 = haze_class.features.relu0

        ############# Block1-down 256-256  ##############
        self.dense_block1 = haze_class.features.denseblock1
        self.trans_block1 = haze_class.features.transition1

        ############# Block2-down 128-128  ##############
        self.dense_block2 = haze_class.features.denseblock2
        self.trans_block2 = haze_class.features.transition2

        ############# Block4-up  8-8  ##############
        self.dense_block3 = BottleneckBlockdy(256, 128)
        self.trans_block3 = TransitionBlockdy(384, 64)

        ############# Block5-up  16-16 ##############
        self.dense_block4 = BottleneckBlockdy(192, 32)
        self.trans_block4 = TransitionBlockdy(224, 64)

        self.conv_refin1 = nn.Conv2d(3, 64, 3, 1, 1)

        self.tanh = nn.Tanh()
        self.refine3 = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1)

        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu = nn.LeakyReLU(0.2, inplace=True)


    def forward(self, x):
        ## 256*256
        x0 = self.relu0(self.conv_refin1(x))
        # print ('x0.size:',x0.size())

        x1 = self.dense_block1(x0)
        # print x1.size()
        x1 = self.trans_block1(x1)
        # print ('x1.size:', x1.size())

        x2 = self.trans_block2(self.dense_block2(x1))
        # print  x2.size()
        # print ('x2.size:', x2.size())

        x3 = self.trans_block3(self.dense_block3(x2))
        # print ('x3.size:', x3.size())

        x4 = torch.cat([x3, x1], 1)
        # print('x4:',x4.size())

        x4 = self.trans_block4(self.dense_block4(x4))

        x42 = torch.cat([x4, x0], 1)

        dehaze = self.tanh(self.refine3(x42))
        return dehaze


class Dense_rain_cvprw3dy10(nn.Module):
	def __init__(self):
		super(Dense_rain_cvprw3dy10, self).__init__()

		############# 256-256  ##############
		haze_class = models.densenet121(pretrained=True)

		self.conv0 = haze_class.features.conv0
		self.relu0 = haze_class.features.relu0

		############# Block1-down 256-256  ##############
		self.dense_block1 = haze_class.features.denseblock1
		self.trans_block1 = haze_class.features.transition1

		############# Block2-down 128-128  ##############
		self.dense_block2 = haze_class.features.denseblock2
		self.trans_block2 = haze_class.features.transition2



		############# Block4-up  8-8  ##############
		self.dense_block3 = BottleneckBlockdy(256, 128)
		self.trans_block3 = TransitionBlockdy(384, 64)

		############# Block5-up  16-16 ##############
		self.dense_block4 = BottleneckBlockdy(64, 32)
		self.trans_block4 = TransitionBlockdy(96, 16)



		self.conv_refin1 = nn.Conv2d(3, 64, 3, 1, 1)

		self.tanh = nn.Tanh()
		self.refine3 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

		# self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

		self.upsample = F.upsample_nearest

		self.relu = nn.LeakyReLU(0.2, inplace=True)

	def forward(self, x):
		## 256*256
		x0 = self.relu0(self.conv_refin1(x))
		# print ('x0.size:',x0.size())


		x1 = self.dense_block1(x0)
		# print x1.size()
		x1 = self.trans_block1(x1)
		# print ('x1.size:', x1.size())

		x2 = self.trans_block2(self.dense_block2(x1))
		# print  x2.size()
		# print ('x2.size:', x2.size())


		x3 = self.trans_block3(self.dense_block3(x2))
		# print ('x3.size:', x3.size())

		x4 = self.trans_block4(self.dense_block4(x3))


		dehaze = self.tanh(self.refine3(x4))

		# dehaze=x - residual

		return dehaze

class Dense_rain_cvprw3dy11(nn.Module):
	def __init__(self):
		super(Dense_rain_cvprw3dy11, self).__init__()

		############# 256-256  ##############
		haze_class = models.densenet121(pretrained=True)

		self.conv0 = haze_class.features.conv0
		self.relu0 = haze_class.features.relu0

		############# Block1-down 256-256  ##############
		self.dense_block1 = haze_class.features.denseblock1
		self.trans_block1 = haze_class.features.transition1

		############# Block2-down 128-128  ##############
		self.dense_block3 = haze_class.features.denseblock2
		self.trans_block3 = haze_class.features.transition2



		############# Block4-up  8-8  ##############
		self.dense_block2 = BottleneckBlockdy1(128, 128)
		self.trans_block2 = TransitionBlockdy1(128, 128)

		############# Block5-up  16-16 ##############
		self.dense_block4 = BottleneckBlockdy1(256, 128)
		self.trans_block4 = TransitionBlockdy1(128, 32)



		self.conv_refin1 = nn.Conv2d(3, 64, 3, 1, 1)

		self.tanh = nn.Tanh()
		self.refine3 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

		# self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

		self.upsample = F.upsample_nearest

		self.relu = nn.LeakyReLU(0.2, inplace=True)

	def forward(self, x):
		## 256*256
		x0 = self.relu0(self.conv_refin1(x))
		# print ('x0.size:',x0.size())


		x1 = self.dense_block1(x0)
		# print x1.size()
		x1 = self.trans_block1(x1)
		# print ('x1.size:', x1.size())

		x2 = self.trans_block2(self.dense_block2(x1))
		# print  x2.size()
		# print ('x2.size:', x2.size())


		x3 = self.trans_block3(self.dense_block3(x2))
		# print ('x3.size:', x3.size())

		x4 = self.trans_block4(self.dense_block4(x3))


		dehaze = self.tanh(self.refine3(x4))

		# dehaze=x - residual

		return dehaze

class Dense_rain_cvprw3dy12(nn.Module):
	def __init__(self):
		super(Dense_rain_cvprw3dy12, self).__init__()

		############# 256-256  ##############
		haze_class = models.densenet121(pretrained=True)

		self.conv0 = haze_class.features.conv0
		self.relu0 = haze_class.features.relu0

		############# Block1-down 256-256  ##############
		self.dense_block1 = haze_class.features.denseblock1
		self.trans_block1 = haze_class.features.transition1

		############# Block2-down 128-128  ##############
		self.dense_block2 = haze_class.features.denseblock2
		self.trans_block2 = haze_class.features.transition2

		############# Block3-down  64-64 ##############
		self.dense_block3 = haze_class.features.denseblock3
		self.trans_block3 = haze_class.features.transition3

		# ############# Block31-down  xx-xx ##############
		self.dense_block31 = haze_class.features.denseblock4
		# self.trans_block31=haze_class.features.transition4

		self.dense_norm31 = haze_class.features.norm5

		############# Block4-up  8-8  ##############
		self.dense_block4 = BottleneckBlockdy(512, 256)
		self.trans_block4 = TransitionBlockdy(768, 128)

		############# Block5-up  16-16 ##############
		self.dense_block5 = BottleneckBlockdy(384, 128)
		self.trans_block5 = TransitionBlockdy(512, 64)

		############# Block6-up 32-32   ##############
		self.dense_block6 = BottleneckBlockdy(64, 32)
		self.trans_block6 = TransitionBlockdy(96, 16)

		############# Block7-up 64-64   ##############

		self.conv_refin1 = nn.Conv2d(3, 64, 3, 1, 1)

		self.tanh = nn.Tanh()
		self.refine3 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

		# self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

		self.upsample = F.upsample_nearest

		self.relu = nn.LeakyReLU(0.2, inplace=True)

	def forward(self, x):
		## 256*256
		x0 = self.relu0(self.conv_refin1(x))
		# print ('x0.size:',x0.size())

		## 256*256
		x1 = self.dense_block1(x0)
		# print x1.size()
		x1 = self.trans_block1(x1)
		# print ('x1.size:', x1.size())

		###  128*128
		x2 = self.trans_block2(self.dense_block2(x1))
		# print  x2.size()
		# print ('x2.size:', x2.size())

		### 64X 64
		x3 = self.trans_block3(self.dense_block3(x2))
		# print ('x3.size:', x3.size())

		## 32 X 32
		x4 = self.trans_block4(self.dense_block4(x3))
		# print('x4:',x4.size())

		x42 = torch.cat([x4, x2], 1)
		# print(x42.size())

		## 64 X 64
		x5 = self.trans_block5(self.dense_block5(x42))
		# x52 = torch.cat([x5, x1], 1)
		# print('x52.size())

		##  128 X 128
		x6 = self.trans_block6(self.dense_block6(x5))

		##  256 X 256

		dehaze = self.tanh(self.refine3(x6))

		# dehaze=x - residual

		return dehaze

class Dense_rain_cvprw3dy12_R(nn.Module):
	def __init__(self):
		super(Dense_rain_cvprw3dy12_R, self).__init__()

		############# 256-256  ##############
		haze_class = models.densenet121(pretrained=True)

		self.conv0 = haze_class.features.conv0
		self.relu0 = haze_class.features.relu0

		############# Block1-down 256-256  ##############
		self.dense_block1 = haze_class.features.denseblock1
		self.trans_block1 = haze_class.features.transition1

		############# Block2-down 128-128  ##############
		self.dense_block2 = haze_class.features.denseblock2
		self.trans_block2 = haze_class.features.transition2

		############# Block3-down  64-64 ##############
		self.dense_block3 = haze_class.features.denseblock3
		self.trans_block3 = haze_class.features.transition3

		# ############# Block31-down  xx-xx ##############
		self.dense_block31 = haze_class.features.denseblock4
		# self.trans_block31=haze_class.features.transition4

		self.dense_norm31 = haze_class.features.norm5

		############# Block4-up  8-8  ##############
		self.dense_block4 = BottleneckBlockdy(512, 256)
		self.trans_block4 = TransitionBlockdy(768, 128)

		############# Block5-up  16-16 ##############
		self.dense_block5 = BottleneckBlockdy(384, 128)
		self.trans_block5 = TransitionBlockdy(512, 64)

		############# Block6-up 32-32   ##############
		self.dense_block6 = BottleneckBlockdy(64, 32)
		self.trans_block6 = TransitionBlockdy(96, 16)

		############# Block7-up 64-64   ##############

		self.conv_refin1 = nn.Conv2d(3, 64, 3, 1, 1)

		self.conv_refin6 = nn.Conv2d(640,512,3,1,1)
		self.conv_refin5 = nn.Conv2d(256,128,1,1,0)
		self.tanh = nn.Tanh()
		self.conv_refin3 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

		self.conv_refin2= nn.Conv2d(64, 32, kernel_size=1,stride=1,padding=0)

		self.upsample = F.upsample_nearest

		self.conv_refine4 = nn.Conv2d(160, 128, kernel_size=3, stride=1, padding=1)
        #self.relu = nn.LeakyReLU(0.2, inplace=True)

	def forward(self, x):
		## 256*256
		x0 = self.relu0(self.conv_refin1(x))

		#print('x0.size:',x0.size())
		x01 = self.conv_refin2(F.avg_pool2d(x0,2))
        #  print ('x0.size:',x0.size())

		## 256*256
		x1 = self.dense_block1(x0)
		#print ('x01.size:',x01.size())
		x1 = self.trans_block1(x1)
		#print ('x1.size:', x1.size())

		###  128*128
		x10 =self.conv_refine4(torch.cat([x01,x1],1))
		x2=self.trans_block2(self.dense_block2(x10))
		#print ('x2.size:', x2.size())

		### 64X 64
		x3 = self.trans_block3(self.dense_block3(x2))
		#print('x3.size:', x3.size())
		x22=self.conv_refin5(F.avg_pool2d(x2,2))

		## 32 X 32
		x4 = self.trans_block4(self.dense_block4(self.conv_refin6(torch.cat([x3, x22], 1))))
		#print('x4:',x4.size())

		x42 = torch.cat([x4, x2], 1)
		# print(x42.size())

		## 64 X 64
		x5 = self.trans_block5(self.dense_block5(x42))
		# x52 = torch.cat([x5, x1], 1)
		# print('x52.size())

		##  128 X 128
		x6 = self.trans_block6(self.dense_block6(x5))

		##  256 X 256

		dehaze = self.tanh(self.conv_refin3(x6))

		# dehaze=x - residual

		return dehaze


class Dense_rain_cvprw3dy13(nn.Module):
	def __init__(self):
		super(Dense_rain_cvprw3dy13, self).__init__()

		############# 256-256  ##############
		haze_class = models.densenet121(pretrained=True)

		self.conv0 = haze_class.features.conv0
		self.relu0 = haze_class.features.relu0

		############# Block1-down 256-256  ##############
		self.dense_block1 = haze_class.features.denseblock1
		self.trans_block1 = haze_class.features.transition1

		############# Block2-down 128-128  ##############
		self.dense_block3 = haze_class.features.denseblock2
		self.trans_block3 = haze_class.features.transition2



		############# Block4-up  8-8  ##############
		self.dense_block2 = BottleneckBlockdy1(128, 128)
		self.trans_block2 = TransitionBlockdy2(128, 128)

		############# Block5-up  16-16 ##############
		self.dense_block4 = BottleneckBlockdy1(256, 128)
		self.trans_block4 = TransitionBlockdy2(128, 32)



		self.conv_refin1 = nn.Conv2d(3, 64, 3, 1, 1)

		self.tanh = nn.Tanh()
		self.refine3 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

		# self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

		self.upsample = F.upsample_nearest

		self.relu = nn.LeakyReLU(0.2, inplace=True)

	def forward(self, x):
		## 256*256
		x0 = self.relu0(self.conv_refin1(x))
		# print ('x0.size:',x0.size())


		x1 = self.dense_block1(x0)
		# print x1.size()
		x1 = self.trans_block1(x1)
		x1 = F.avg_pool2d(x1,2)
		print ('x1.size:', x1.size())

		x2 = self.trans_block2(self.dense_block2(x1))
		# print  x2.size()
		# print ('x2.size:', x2.size())


		x3 = self.trans_block3(self.dense_block3(x2))
		x3 =F.avg_pool2d(x3,2)
		print ('x3.size:', x3.size())

		x4 = self.trans_block4(self.dense_block4(x3))


		dehaze = self.tanh(self.refine3(x4))

		# dehaze=x - residual

		return dehaze



class Dense_rain_cvprw3dy14(nn.Module):
    def __init__(self):
        super(Dense_rain_cvprw3dy14, self).__init__()

        ############# 256-256  ##############
        haze_class = models.densenet121(pretrained=True)

        self.conv0 = haze_class.features.conv0
        self.relu0 = haze_class.features.relu0

        ############# Block1-down 256-256  ##############
        self.dense_block1 = haze_class.features.denseblock1
        self.trans_block1 = haze_class.features.transition1

        ############# Block2-down 128-128  ##############
        self.dense_block2 = haze_class.features.denseblock2
        self.trans_block2 = haze_class.features.transition2

        ############# Block3-down  64-64 ##############
        self.dense_block3 = haze_class.features.denseblock3
        self.trans_block3 = haze_class.features.transition3

        # ############# Block31-down  xx-xx ##############
        self.dense_block4 = haze_class.features.denseblock4
        self.trans_block4 = nn.Conv2d(1024, 768, kernel_size=3,stride=1,padding=1)


        ############# Block5-up  16-16 ##############
        self.dense_block5 = BottleneckBlockdy1(768, 512)
        self.trans_block5 = TransitionBlockdy1(512, 256)

        ############# Block6-up 32-32   ##############
        self.dense_block6 = BottleneckBlockdy1(768, 512)
        self.trans_block6 = TransitionBlockdy1(512, 384)

        ############# Block7-up 64-64   ##############

        self.dense_block7 = BottleneckBlockdy1(640, 384)
        self.trans_block7 = TransitionBlockdy1(384, 256)

        self.dense_block8 = BottleneckBlockdy1(256, 128)
        self.trans_block8 = TransitionBlockdy1(128, 32)
        
        self.conv_refin1 = nn.Conv2d(3, 64, 3, 1, 1)

        self.tanh = nn.Tanh()
        self.refine3 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

        #self= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        ## 256*256
        x0 = self.relu0(self.conv_refin1(x))
        # print ('x0.size:',x0.size())

        ## 256*256
        x1 = self.dense_block1(x0)
        # print x1.size()
        x1 = self.trans_block1(x1)

        ###  128*128
        x2 = self.trans_block2(self.dense_block2(x1))
        # print  x2.size()

        ### 64X 64
        x3 = self.trans_block3(self.dense_block3(x2))

        ## 32 X 32
        x4 = self.trans_block4(self.dense_block4(x3))
        x4 = F.avg_pool2d(x4, 2)

        x5 = self.trans_block5(self.dense_block5(x4))
        x53 = torch.cat([x5, x3], 1)

        ## 64 X 64
        x6 = self.trans_block6(self.dense_block6(x53))
        x62 = torch.cat([x6, x2], 1)

        ##  128 X 128
        x7 = self.trans_block7(self.dense_block7(x62))

        x8 = self.trans_block8(self.dense_block8(x7))

        ##  256 X 256

        dehaze = self.tanh(self.refine3(x8))

        # dehaze=x - residual

        return dehaze


class Dense_rain_cvprw4(nn.Module):
    def __init__(self):
        super(Dense_rain_cvprw4, self).__init__()




        ############# 256-256  ##############
        haze_class = models.densenet121(pretrained=True)

        self.conv0=haze_class.features.conv0
        self.norm0=haze_class.features.norm0
        self.relu0=haze_class.features.relu0
        self.pool0=haze_class.features.pool0

        ############# Block1-down 64-64  ##############
        self.dense_block1=haze_class.features.denseblock1
        self.trans_block1=haze_class.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2=haze_class.features.denseblock2
        self.trans_block2=haze_class.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3=haze_class.features.denseblock3
        self.trans_block3=haze_class.features.transition3

        # ############# Block31-down  xx-xx ##############
        self.dense_block31=haze_class.features.denseblock4
        # self.trans_block31=haze_class.features.transition4

        self.dense_norm31=haze_class.features.norm5

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock(512,256)
        self.trans_block4=TransitionBlock(768,128)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock(387,256)
        self.trans_block5=TransitionBlock(643,128)

        ############# Block6-up 32-32   ##############
        self.dense_block6=BottleneckBlock(259,128)
        self.trans_block6=TransitionBlock(387,64)


        ############# Block7-up 64-64   ##############
        self.dense_block7=BottleneckBlock(67,64)
        self.trans_block7=TransitionBlock(131,32)

        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8=BottleneckBlock(35,32)
        self.trans_block8=TransitionBlock(67,16)

        self.conv_refin=nn.Conv2d(19,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(4, 3, kernel_size=3,stride=1,padding=1)

        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.pool1=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)



        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)


        self.res31=BasicBlock_res(512,512)
        self.res32 = BasicBlock_res(512, 512)


        self.res41=BasicBlock_res(387,387)
        self.res42 = BasicBlock_res(387, 387)


        self.res51=BasicBlock_res(259,259)
        self.res52 = BasicBlock_res(259, 259)


        self.res61=BasicBlock_res(67,67)
        self.res62 = BasicBlock_res(67, 67)

        self.res71=BasicBlock_res(35,35)
        self.res72 = BasicBlock_res(35, 35)


        self.resref1 =BasicBlock_res(44,44)
        self.resref2 = BasicBlock_res(44, 44)


    def forward(self, x):
        ## 256x256
        x0=self.pool0(self.relu0(self.norm0(self.conv0(x))))

        ## 64 X 64
        x1=self.dense_block1(x0)
        # print x1.size()
        x1=self.trans_block1(x1)

        ###  32x32
        x2=self.trans_block2(self.dense_block2(x1))
        # print  x2.size()


        ### 16 X 16
        x3=self.trans_block3(self.dense_block3(x2))
        ## Classifier  ##

        ## 8 X 8
        x3 = self.res31(x3)
        x3 = self.res32(x3)
        # print  x3.size()

        ## 8 X 8
        x4 = self.trans_block4(self.dense_block4(x3))
        x43=F.avg_pool2d(x,16)
        # print(x43.size())


        x42 = torch.cat([x4,x2,x43], 1)
        # print(x42.size())

        x42 = self.res41(x42)
        x42 = self.res42(x42)

        ## 16 X 16
        x5 = self.trans_block5(self.dense_block5(x42))
        x53=F.avg_pool2d(x,8)
        x52 = torch.cat([x5,x1,x53], 1)
        # print(x52.size())



        x52 = self.res51(x52)
        x52 = self.res52(x52)

        ##  32 X 32
        x6 = self.trans_block6(self.dense_block6(x52))
        x63=F.avg_pool2d(x,4)

        x62 = torch.cat([x6,x63], 1)
        x62 = self.res61(x62)
        x6 = self.res62(x62)
        # print(x6.size())

        ##  64 X 64
        x7 = self.trans_block7(self.dense_block7(x6))
        x73=F.avg_pool2d(x,2)
        x72 = torch.cat([x7,x73], 1)

        x72 = self.res71(x72)
        x7 = self.res72(x72)

        ##  128 X 128
        x8 = self.trans_block8(self.dense_block8(x7))

        # print x8.size()
        # print x.size()

        x8=torch.cat([x8,x],1)

        # print x8.size()

        # x9=self.relu((self.conv_refin(x8)))
        x9=self.relu(self.batchnorm20(self.conv_refin(x8)))

        shape_out = x9.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(x9, 2)
        x102 = F.avg_pool2d(x9, 4)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 2)

        x1010 = self.upsample(self.relu((self.conv1010(x101))), size=shape_out)
        x1020 = self.upsample(self.relu((self.conv1020(x102))), size=shape_out)
        x1030 = self.upsample(self.relu((self.conv1030(x103))), size=shape_out)
        x1040 = self.upsample(self.relu((self.conv1040(x104))), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        # dehaze = torch.cat((x1010, x1020, x1030, x1040), 1)

        residual = self.tanh(self.refine3(dehaze))
        # dehaze=x - residual

        return residual


class Dense_rain_cvprw5(nn.Module):
    def __init__(self):
        super(Dense_rain_cvprw5, self).__init__()




        ############# 256-256  ##############
        haze_class = models.densenet121(pretrained=True)

        self.conv0=haze_class.features.conv0
        self.norm0=haze_class.features.norm0
        self.relu0=haze_class.features.relu0
        self.pool0=haze_class.features.pool0

        ############# Block1-down 64-64  ##############
        self.dense_block1=haze_class.features.denseblock1
        self.trans_block1=haze_class.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2=haze_class.features.denseblock2
        self.trans_block2=haze_class.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3=haze_class.features.denseblock3
        self.trans_block3=haze_class.features.transition3

        # ############# Block31-down  xx-xx ##############
        self.dense_block31=haze_class.features.denseblock4
        # self.trans_block31=haze_class.features.transition4

        self.dense_norm31=haze_class.features.norm5

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock(512,256)
        self.trans_block4=TransitionBlock(768,128)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock(387,256)
        self.trans_block5=TransitionBlock(643,128)

        ############# Block6-up 32-32   ##############
        self.dense_block6=BottleneckBlock(259,128)
        self.trans_block6=TransitionBlock(387,64)


        ############# Block7-up 64-64   ##############
        self.dense_block7=BottleneckBlock(67,64)
        self.trans_block7=TransitionBlock(131,32)

        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8=BottleneckBlock(35,32)
        self.trans_block8=TransitionBlock(67,16)

        self.conv_refin=nn.Conv2d(19,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(4, 3, kernel_size=3,stride=1,padding=1)

        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.pool1=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)



        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)


        self.res31=BasicBlock_res(512,512)
        self.res32 = BasicBlock_res(512, 512)


        self.res41=BasicBlock_res(387,387)
        self.res42 = BasicBlock_res(387, 387)


        self.res51=BasicBlock_res(259,259)
        self.res52 = BasicBlock_res(259, 259)


        self.res61=BasicBlock_res(67,67)
        self.res62 = BasicBlock_res(67, 67)

        self.res71=BasicBlock_res(35,35)
        self.res72 = BasicBlock_res(35, 35)


        self.resref1 =BasicBlock_res(44,44)
        self.resref2 = BasicBlock_res(44, 44)


    def forward(self, x):
        ## 256x256
        x0=self.pool0(self.relu0(self.norm0(self.conv0(x))))

        ## 64 X 64
        x1=self.dense_block1(x0)
        # print x1.size()
        x1=self.trans_block1(x1)

        ###  32x32
        x2=self.trans_block2(self.dense_block2(x1))
        # print  x2.size()


        ### 16 X 16
        x3=self.trans_block3(self.dense_block3(x2))
        ## Classifier  ##

        ## 8 X 8
        x3 = self.res31(x3)
        x3 = self.res32(x3)
        # print  x3.size()

        ## 8 X 8
        x4 = self.trans_block4(self.dense_block4(x3))
        x43=F.avg_pool2d(x,16)
        # print(x43.size())


        x42 = torch.cat([x4,x2,x43], 1)
        # print(x42.size())

        x42 = self.res41(x42)
        x42 = self.res42(x42)

        ## 16 X 16
        x5 = self.trans_block5(self.dense_block5(x42))
        x53=F.avg_pool2d(x,8)
        x52 = torch.cat([x5,x1,x53], 1)
        # print(x52.size())



        x52 = self.res51(x52)
        x52 = self.res52(x52)

        ##  32 X 32
        x6 = self.trans_block6(self.dense_block6(x52))
        x63=F.avg_pool2d(x,4)

        x62 = torch.cat([x6,x63], 1)
        x62 = self.res61(x62)
        x6 = self.res62(x62)
        # print(x6.size())

        ##  64 X 64
        x7 = self.trans_block7(self.dense_block7(x6))
        x73=F.avg_pool2d(x,2)
        x72 = torch.cat([x7,x73], 1)

        x72 = self.res71(x72)
        x7 = self.res72(x72)

        ##  128 X 128
        x8 = self.trans_block8(self.dense_block8(x7))

        # print x8.size()
        # print x.size()

        x8=torch.cat([x8,x],1)

        # print x8.size()

        x9=self.relu((self.conv_refin(x8)))
        # x9=self.relu(self.batchnorm20(self.conv_refin(x8)))

        shape_out = x9.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(x9, 2)
        x102 = F.avg_pool2d(x9, 4)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 16)

        x1010 = self.upsample(self.relu((self.conv1010(x101))), size=shape_out)
        x1020 = self.upsample(self.relu((self.conv1020(x102))), size=shape_out)
        x1030 = self.upsample(self.relu((self.conv1030(x103))), size=shape_out)
        x1040 = self.upsample(self.relu((self.conv1040(x104))), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        # dehaze = torch.cat((x1010, x1020, x1030, x1040), 1)

        residual = self.tanh(self.refine3(dehaze))
        # dehaze=x - residual

        return residual



class Dense_rain_cvprw6(nn.Module):
    def __init__(self):
        super(Dense_rain_cvprw6, self).__init__()




        ############# 256-256  ##############
        haze_class = models.densenet121(pretrained=True)

        self.conv0=haze_class.features.conv0
        self.norm0=haze_class.features.norm0
        self.relu0=haze_class.features.relu0
        self.pool0=haze_class.features.pool0

        ############# Block1-down 64-64  ##############
        self.dense_block1=haze_class.features.denseblock1
        self.trans_block1=haze_class.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2=haze_class.features.denseblock2
        self.trans_block2=haze_class.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3=haze_class.features.denseblock3
        self.trans_block3=haze_class.features.transition3

        # ############# Block31-down  xx-xx ##############
        self.dense_block31=haze_class.features.denseblock4
        # self.trans_block31=haze_class.features.transition4

        self.dense_norm31=haze_class.features.norm5

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock(512,256)
        self.trans_block4=TransitionBlock(768,128)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock(387,256)
        self.trans_block5=TransitionBlock(643,128)

        ############# Block6-up 32-32   ##############
        self.dense_block6=BottleneckBlock(259,128)
        self.trans_block6=TransitionBlock(387,64)


        ############# Block7-up 64-64   ##############
        self.dense_block7=BottleneckBlock(67,64)
        self.trans_block7=TransitionBlock(131,32)

        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8=BottleneckBlock(35,32)
        self.trans_block8=TransitionBlock(67,16)

        self.conv_refin=nn.Conv2d(19,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(4, 3, kernel_size=3,stride=1,padding=1)

        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.pool1=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)
        self.pool2=nn.AvgPool2d(3, stride=2)



        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)


        self.res31=BasicBlock_res(512,512)
        self.res32 = BasicBlock_res(512, 512)


        self.res41=BasicBlock_res(387,387)
        self.res42 = BasicBlock_res(387, 387)


        self.res51=BasicBlock_res(259,259)
        self.res52 = BasicBlock_res(259, 259)


        self.res61=BasicBlock_res(67,67)
        self.res62 = BasicBlock_res(67, 67)

        self.res71=BasicBlock_res(35,35)
        self.res72 = BasicBlock_res(35, 35)


        self.resref1 =BasicBlock_res(44,44)
        self.resref2 = BasicBlock_res(44, 44)


    def forward(self, x):
        ## 256x256
        x0=self.pool0(self.relu0(self.norm0(self.conv0(x))))

        ## 64 X 64
        x1=self.dense_block1(x0)
        # print x1.size()
        x1=self.trans_block1(x1)

        ###  32x32
        x2=self.trans_block2(self.dense_block2(x1))
        # print  x2.size()


        ### 16 X 16
        x3=self.trans_block3(self.dense_block3(x2))
        ## Classifier  ##

        ## 8 X 8
        x3 = self.res31(x3)
        x3 = self.res32(x3)
        # print  x3.size()

        ## 8 X 8
        x4 = self.trans_block4(self.dense_block4(x3))
        x43=F.avg_pool2d(x,16)
        # print(x43.size())


        x42 = torch.cat([x4,x2,x43], 1)
        # print(x42.size())

        x42 = self.res41(x42)
        x42 = self.res42(x42)

        ## 16 X 16
        x5 = self.trans_block5(self.dense_block5(x42))
        x53=F.avg_pool2d(x,8)
        x52 = torch.cat([x5,x1,x53], 1)
        # print(x52.size())



        x52 = self.res51(x52)
        x52 = self.res52(x52)

        ##  32 X 32
        x6 = self.trans_block6(self.dense_block6(x52))
        x63=F.avg_pool2d(x,4)

        x62 = torch.cat([x6,x63], 1)
        x62 = self.res61(x62)
        x6 = self.res62(x62)
        # print(x6.size())

        ##  64 X 64
        x7 = self.trans_block7(self.dense_block7(x6))
        x73=F.avg_pool2d(x,2)
        x72 = torch.cat([x7,x73], 1)

        x72 = self.res71(x72)
        x7 = self.res72(x72)

        ##  128 X 128
        x8 = self.trans_block8(self.dense_block8(x7))

        # print x8.size()
        # print x.size()

        x8=torch.cat([x8,x],1)

        x9=self.relu(self.batchnorm20(self.conv_refin(x8)))

        shape_out = x9.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(x9, 2)
        x102 = F.avg_pool2d(x9, 4)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 16)

        x1010 = self.upsample(self.relu((self.conv1010(x101))), size=shape_out)
        x1020 = self.upsample(self.relu((self.conv1020(x102))), size=shape_out)
        x1030 = self.upsample(self.relu((self.conv1030(x103))), size=shape_out)
        x1040 = self.upsample(self.relu((self.conv1040(x104))), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        # dehaze = torch.cat((x1010, x1020, x1030, x1040), 1)

        residual = self.tanh(self.refine3(dehaze))
        # dehaze=x - residual

        return residual


class vgg19ca(nn.Module):
    def __init__(self):
        super(vgg19ca, self).__init__()




        ############# 256-256  ##############
        haze_class = models.vgg19_bn(pretrained=True)
        self.feature = nn.Sequential(haze_class.features[0])

        for i in range(1,3):
            self.feature.add_module(str(i),haze_class.features[i])

        self.conv16=nn.Conv2d(64, 24, kernel_size=3,stride=1,padding=1)  # 1mm
        self.dense_classifier=nn.Linear(127896, 512)
        self.dense_classifier1=nn.Linear(512, 4)


    def forward(self, x):

        feature=self.feature(x)
        # feature = Variable(feature.data, requires_grad=True)

        feature=self.conv16(feature)
        # print feature.size()

        # feature=Variable(feature.data,requires_grad=True)



        out = F.relu(feature, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7).view(out.size(0), -1)
        # print out.size()

        # out=Variable(out.data,requires_grad=True)
        out = F.relu(self.dense_classifier(out))
        out = (self.dense_classifier1(out))


        return out


class Dense_rain1(nn.Module):
    def __init__(self):
        super(Dense_rain1, self).__init__()




        ############# 256-256  ##############
        haze_class = models.densenet121(pretrained=True)

        self.conv0=haze_class.features.conv0
        self.norm0=haze_class.features.norm0
        self.relu0=haze_class.features.relu0
        self.pool0=haze_class.features.pool0

        ############# Block1-down 64-64  ##############
        self.dense_block1=haze_class.features.denseblock1
        self.trans_block1=haze_class.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2=haze_class.features.denseblock2
        self.trans_block2=haze_class.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3=haze_class.features.denseblock3
        self.trans_block3=haze_class.features.transition3

        # ############# Block31-down  xx-xx ##############
        self.dense_block31=haze_class.features.denseblock4
        # self.trans_block31=haze_class.features.transition4

        self.dense_norm31=haze_class.features.norm5

        self.dense_classifier=nn.Linear(3072, 512)
        self.dense_classifier1=nn.Linear(512, 4)

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock(512,256)
        self.trans_block4=TransitionBlock(768,128)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock(384,256)
        self.trans_block5=TransitionBlock(640,128)

        ############# Block6-up 32-32   ##############
        self.dense_block6=BottleneckBlock(256,128)
        self.trans_block6=TransitionBlock(384,64)


        ############# Block7-up 64-64   ##############
        self.dense_block7=BottleneckBlock(64,64)
        self.trans_block7=TransitionBlock(128,32)

        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8=BottleneckBlock(32,32)
        self.trans_block8=TransitionBlock(64,16)

        self.conv_refin=nn.Conv2d(19,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)


        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)



    def forward(self, x):
        ## 256x256
        x0=self.pool0(self.relu0(self.norm0(self.conv0(x))))

        ## 64 X 64
        x1=self.dense_block1(x0)
        # print x1.size()
        x1=self.trans_block1(x1)

        ###  32x32
        x2=self.trans_block2(self.dense_block2(x1))
        # print  x2.size()


        ### 16 X 16
        x3=self.trans_block3(self.dense_block3(x2))
        ## Classifier  ##
        x3=Variable(x3.data,requires_grad=True)

        x31=self.dense_block4(x3)


        out = F.relu(x31, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7).view(x31.size(0), -1)

        # out=Variable(out.data,requires_grad=True)
        out = F.relu(self.dense_classifier(out))
        out = (self.dense_classifier1(out))


        return out

class Dense_rain2(nn.Module):
    def __init__(self):
        super(Dense_rain2, self).__init__()




        ############# 256-256  ##############
        haze_class = models.densenet121(pretrained=True)

        self.conv0=haze_class.features.conv0
        self.norm0=haze_class.features.norm0
        self.relu0=haze_class.features.relu0
        self.pool0=haze_class.features.pool0

        ############# Block1-down 64-64  ##############
        self.dense_block1=haze_class.features.denseblock1
        self.trans_block1=haze_class.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2=haze_class.features.denseblock2
        self.trans_block2=haze_class.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3=haze_class.features.denseblock3
        self.trans_block3=haze_class.features.transition3

        # ############# Block31-down  xx-xx ##############
        self.dense_block31=haze_class.features.denseblock4
        # self.trans_block31=haze_class.features.transition4

        self.dense_norm31=haze_class.features.norm5

        self.dense_classifier=nn.Linear(3072, 512)
        self.dense_classifier1=nn.Linear(512, 4)

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock(512,256)
        self.trans_block4=TransitionBlock(768,128)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock(384,256)
        self.trans_block5=TransitionBlock(640,128)

        ############# Block6-up 32-32   ##############
        self.dense_block6=BottleneckBlock(257,128)
        self.trans_block6=TransitionBlock(385,64)


        ############# Block7-up 64-64   ##############
        self.dense_block7=BottleneckBlock(65,64)
        self.trans_block7=TransitionBlock(129,32)

        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8=BottleneckBlock(33,32)
        self.trans_block8=TransitionBlock(65,16)

        self.conv_refin=nn.Conv2d(20,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(4, 3, kernel_size=3,stride=1,padding=1)

        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)


        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)

        self.classifier=Dense_rain1()
        self.classifier.load_state_dict(torch.load('/home/openset/Desktop/derain2018/checkpoints_test_classifica/netG_epoch_2940.pth'))
        self.classifier=self.classifier.train()
        self.classifier=self.classifier.cuda()
        self.softmax=nn.Softmax()



    def forward(self, x, label_d):
        ## 256x256
        x0=self.pool0(self.relu0(self.norm0(self.conv0(x))))

        ## 64 X 64
        x1=self.dense_block1(x0)
        # print x1.size()
        x1=self.trans_block1(x1)

        ###  32x32
        x2=self.trans_block2(self.dense_block2(x1))
        # print  x2.size()


        ### 16 X 16
        x3=self.trans_block3(self.dense_block3(x2))
        ## Classifier  ##
        # x3=Variable(x3.data,requires_grad=True)

        label=self.classifier(x)

        label_result1,label_result=torch.max(label,1)

        label_result = label_result.data.cpu().numpy()
        # print label_result
        label_result=float(label_result[0][0])

        probability=self.softmax(label)

        probability = probability.data.cpu().numpy()

        zz = float(probability[0][0])
        zz1 = float(probability[0][1])
        zz2 = float(probability[0][2])
        zz3 = float(probability[0][3])
        # print(zz,zz1,zz2,zz3)

        batchSize=1
        label_d1 = torch.FloatTensor(batchSize)
        label_d1 = Variable(label_d1.cuda())

        label_d2 = torch.FloatTensor(batchSize)
        label_d2 = Variable(label_d2.cuda())

        label_d3 = torch.FloatTensor(batchSize)
        label_d3 = Variable(label_d3.cuda())

        label_d4 = torch.FloatTensor(batchSize)
        label_d4 = Variable(label_d4.cuda())




        ## 8 X 8
        x4=self.trans_block4(self.dense_block4(x3))

        x42=torch.cat([x4,x2],1)
        ## 16 X 16
        x5=self.trans_block5(self.dense_block5(x42))

        shape_out = x5.data.size()
        # print(shape_out)
        sizePatchGAN = shape_out[3]

        label_result=label_result
        label_result = 1

        label_d1.data.resize_((1, 1, sizePatchGAN, sizePatchGAN)).fill_(label_result)
        label_d2.data.resize_((1, 1, sizePatchGAN, sizePatchGAN)).fill_(zz1)
        label_d3.data.resize_((1, 1, sizePatchGAN, sizePatchGAN)).fill_(zz2)
        label_d4.data.resize_((1, 1, sizePatchGAN, sizePatchGAN)).fill_(zz3)
        label_d=torch.cat([label_d1,label_d2,label_d3,label_d4],1)


        x52=torch.cat([x5,x1,label_d1],1)

        ##  32 X 32
        x6=self.trans_block6(self.dense_block6(x52))
        label_d11 = torch.FloatTensor(batchSize)
        label_d11 = Variable(label_d11.cuda())

        shape_out = x6.data.size()
        # print(shape_out)
        sizePatchGAN = shape_out[3]
        label_d11.data.resize_((1, 1, sizePatchGAN, sizePatchGAN)).fill_(label_result)

        x62=torch.cat([x6,label_d11],1)

        ##  64 X 64
        x7=self.trans_block7(self.dense_block7(x62))

        label_d11 = torch.FloatTensor(batchSize)
        label_d11 = Variable(label_d11.cuda())

        shape_out = x7.data.size()
        # print(shape_out)
        sizePatchGAN = shape_out[3]
        label_d11.data.resize_((1, 1, sizePatchGAN, sizePatchGAN)).fill_(label_result)

        x72=torch.cat([x7,label_d11],1)

        ##  128 X 128
        x8=self.trans_block8(self.dense_block8(x72))

        # print x8.size()
        # print x.size()
        label_d11 = torch.FloatTensor(batchSize)
        label_d11 = Variable(label_d11.cuda())

        shape_out = x8.data.size()
        # print(shape_out)
        sizePatchGAN = shape_out[3]
        label_d11.data.resize_((1, 1, sizePatchGAN, sizePatchGAN)).fill_(label_result)

        x82=torch.cat([x8,label_d11],1)


        x8=torch.cat([x82,x],1)

        # print x8.size()

        # x9=self.relu(self.batchnorm20(self.conv_refin(x8)))
        x9=self.relu((self.conv_refin(x8)))

        shape_out = x9.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)

        x1010 = self.upsample(self.relu((self.conv1010(x101))), size=shape_out)
        x1020 = self.upsample(self.relu((self.conv1020(x102))), size=shape_out)
        x1030 = self.upsample(self.relu((self.conv1030(x103))), size=shape_out)
        x1040 = self.upsample(self.relu((self.conv1040(x104))), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        # dehaze = torch.cat((x1010, x1020, x1030, x1040), 1)

        residual = self.tanh(self.refine3(dehaze))
        # dehaze=x - residual

        return residual


class Dense_rain4(nn.Module):
    def __init__(self):
        super(Dense_rain4, self).__init__()

        # self.conv_refin=nn.Conv2d(9,20,3,1,1)
        self.conv_refin=nn.Conv2d(47,47,3,1,1)

        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(47, 2, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(47, 2, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(47, 2, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(47, 2, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(47+8, 3, kernel_size=3,stride=1,padding=1)

        self.refineclean1= nn.Conv2d(3, 8, kernel_size=7,stride=1,padding=3)
        self.refineclean2= nn.Conv2d(8, 3, kernel_size=3,stride=1,padding=1)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)


        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)

        self.dense0=Dense_base_down0()
        self.dense1=Dense_base_down1()
        self.dense2=Dense_base_down2()

    def forward(self, x, label_d):
        ## 256x256
        label_d11 = torch.FloatTensor(1)
        label_d11 = Variable(label_d11.cuda())
        shape_out = x.data.size()
        sizePatchGAN = shape_out[3]

        # label_result=1
        label_result=float(label_d.data.cpu().float().numpy())

        label_d11.data.resize_((1, 1, sizePatchGAN, sizePatchGAN)).fill_(label_result)

        x1=torch.cat([x,label_d11],1)

        x3=self.dense2(x)
        x2=self.dense1(x)
        x1=self.dense0(x)


        label_d11 = torch.FloatTensor(1)
        label_d11 = Variable(label_d11.cuda())
        shape_out = x3.data.size()
        sizePatchGAN = shape_out[3]

        # label_result=1
        label_result=float(label_d.data.cpu().float().numpy())

        label_d11.data.resize_((1, 8, sizePatchGAN, sizePatchGAN)).fill_(label_result)

        x8=torch.cat([x1,x,x2,x3,label_d11],1)
        # x8=torch.cat([x1,x,x2,x3],1)
        # print(x8.size())

        x9=self.relu((self.conv_refin(x8)))

        shape_out = x9.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)

        x1010 = self.upsample(self.relu((self.conv1010(x101))), size=shape_out)
        x1020 = self.upsample(self.relu((self.conv1020(x102))), size=shape_out)
        x1030 = self.upsample(self.relu((self.conv1030(x103))), size=shape_out)
        x1040 = self.upsample(self.relu((self.conv1040(x104))), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        # dehaze = torch.cat((x1010, x1020, x1030, x1040), 1)
        residual = self.tanh(self.refine3(dehaze))
        clean = x-residual
        clean1=self.relu(self.refineclean1(clean))
        clean2=self.tanh(self.refineclean2(clean1))


        return residual, clean2



class Dense_cvprw(nn.Module):
    def __init__(self):
        super(Dense_cvprw, self).__init__()

        # self.conv_refin=nn.Conv2d(9,20,3,1,1)
        self.conv_refin=nn.Conv2d(39,39,3,1,1)

        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(39, 2, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(39, 2, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(39, 2, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(39, 2, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(39+8, 24, kernel_size=3,stride=1,padding=1)

        self.refineclean1= nn.Conv2d(24, 8, kernel_size=7,stride=1,padding=3)
        self.refineclean2= nn.Conv2d(8, 3, kernel_size=3,stride=1,padding=1)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)


        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)

        self.dense0=Dense_base_down0()
        self.dense1=Dense_base_down1()
        self.dense2=Dense_base_down2()

    def forward(self, x):
        ## 256x256
        label_d11 = torch.FloatTensor(1)
        label_d11 = Variable(label_d11.cuda())
        shape_out = x.data.size()
        sizePatchGAN = shape_out[3]

        # label_result=1
        # label_result=float(label_d.data.cpu().float().numpy())
        #
        # label_d11.data.resize_((1, 1, sizePatchGAN, sizePatchGAN)).fill_(label_result)
        #
        # x1=torch.cat([x,label_d11],1)

        x3=self.dense2(x)
        x2=self.dense1(x)
        x1=self.dense0(x)


        # label_d11 = torch.FloatTensor(1)
        # label_d11 = Variable(label_d11.cuda())
        shape_out = x3.data.size()
        sizePatchGAN = shape_out[3]

        # label_result=1
        # label_result=float(label_d.data.cpu().float().numpy())
        #
        # label_d11.data.resize_((1, 8, sizePatchGAN, sizePatchGAN)).fill_(label_result)

        x8=torch.cat([x1,x,x2,x3],1)
        # x8=torch.cat([x1,x,x2,x3],1)
        # print(x8.size())

        x9=self.relu((self.conv_refin(x8)))

        shape_out = x9.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)

        x1010 = self.upsample(self.relu((self.conv1010(x101))), size=shape_out)
        x1020 = self.upsample(self.relu((self.conv1020(x102))), size=shape_out)
        x1030 = self.upsample(self.relu((self.conv1030(x103))), size=shape_out)
        x1040 = self.upsample(self.relu((self.conv1040(x104))), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        # dehaze = torch.cat((x1010, x1020, x1030, x1040), 1)
        residual = self.relu(self.refine3(dehaze))
        # clean = x-residual
        clean1=self.relu(self.refineclean1(residual))
        clean2=self.tanh(self.refineclean2(clean1))


        return clean2

class Dense_rain5(nn.Module):
    def __init__(self):
        super(Dense_rain5, self).__init__()

        # self.conv_refin=nn.Conv2d(9,20,3,1,1)
        self.conv_refin=nn.Conv2d(66,37,3,1,1)

        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(37, 2, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(37, 2, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(37, 2, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(37, 2, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(37+8, 3, kernel_size=3,stride=1,padding=1)

        self.refineclean1= nn.Conv2d(3, 8, kernel_size=7,stride=1,padding=3)
        self.refineclean2= nn.Conv2d(8, 3, kernel_size=3,stride=1,padding=1)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)


        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)

        self.feature1= nn.Conv2d(3, 20, kernel_size=3,stride=1,padding=1,dilation=1)
        self.feature11= nn.Conv2d(20, 20, kernel_size=3,stride=1,padding=1,dilation=1)


        self.feature2= nn.Conv2d(3, 20, kernel_size=3,stride=1,padding=2,dilation=2)
        self.feature21= nn.Conv2d(20, 20, kernel_size=3,stride=1,padding=2,dilation=2)


        self.feature3= nn.Conv2d(3, 20, kernel_size=3,stride=1,padding=3,dilation=3)
        self.feature31= nn.Conv2d(20, 20, kernel_size=3,stride=1,padding=3,dilation=3)



        self.dense0=Dense_base_down0()
        self.dense1=Dense_base_down1()
        self.dense2=Dense_base_down2()

    def forward(self, x, label_d):
        ## 256x256
        label_d11 = torch.FloatTensor(1)
        label_d11 = Variable(label_d11.cuda())
        shape_out = x.data.size()
        sizePatchGAN = shape_out[3]

        # label_result=1
        label_result=float(label_d.data.cpu().float().numpy())

        label_d11.data.resize_((1, 1, sizePatchGAN, sizePatchGAN)).fill_(label_result)

        x1=torch.cat([x,label_d11],1)

        x11=self.relu(self.feature1(x))
        x12=self.relu(self.feature11(x11))


        x21=self.relu(self.feature2(x))
        x22=self.relu(self.feature21(x21))


        x31=self.relu(self.feature3(x))
        x32=self.relu(self.feature31(x31))





        label_d11 = torch.FloatTensor(1)
        label_d11 = Variable(label_d11.cuda())
        shape_out = x32.data.size()
        sizePatchGAN = shape_out[3]

        # label_result=1
        label_result=float(label_d.data.cpu().float().numpy())

        label_d11.data.resize_((1, 4, sizePatchGAN, sizePatchGAN)).fill_(label_result)

        # x8=torch.cat([x1,x,x2,x3,label_d11],1)
        # print(x12.size())
        # print(x22.size())
        # print(x32.size())

        x8=torch.cat([x12,x22,x32],1)
        # print(x8.size())

        x9=self.relu((self.conv_refin(x8)))

        shape_out = x9.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)

        x1010 = self.upsample(self.relu((self.conv1010(x101))), size=shape_out)
        x1020 = self.upsample(self.relu((self.conv1020(x102))), size=shape_out)
        x1030 = self.upsample(self.relu((self.conv1030(x103))), size=shape_out)
        x1040 = self.upsample(self.relu((self.conv1040(x104))), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        # dehaze = torch.cat((x1010, x1020, x1030, x1040), 1)
        residual = self.tanh(self.refine3(dehaze))
        clean = x-residual
        clean1=self.relu(self.refineclean1(clean))
        clean2=self.tanh(self.refineclean2(clean1))


        return residual, clean2


class Dense_base_down2(nn.Module):
    def __init__(self):
        super(Dense_base_down2, self).__init__()

        self.dense_block1=BottleneckBlock2(3,13)
        self.trans_block1=TransitionBlock1(16,8)

        ############# Block2-down 32-32  ##############
        self.dense_block2=BottleneckBlock2(8,16)
        self.trans_block2=TransitionBlock1(24,16)

        ############# Block3-down  16-16 ##############
        self.dense_block3=BottleneckBlock2(16,16)
        self.trans_block3=TransitionBlock1(32,16)

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock2(16,16)
        self.trans_block4=TransitionBlock(32,16)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock2(32,8)
        self.trans_block5=TransitionBlock(40,8)

        self.dense_block6=BottleneckBlock2(16,8)
        self.trans_block6=TransitionBlock(24,4)


        self.conv_refin=nn.Conv2d(11,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv11 = nn.Conv2d(8, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv21 = nn.Conv2d(16, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv31 = nn.Conv2d(16, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv41 = nn.Conv2d(32, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv51 = nn.Conv2d(16, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv61 = nn.Conv2d(8, 1, kernel_size=3,stride=1,padding=1)  # 1mm

        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)


        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)



    def forward(self, x):
        ## 256x256
        x1=self.dense_block1(x)
        x1=self.trans_block1(x1)

        ###  32x32
        x2=(self.dense_block2(x1))
        x2=self.trans_block2(x2)

        # print x2.size()
        ### 16 X 16
        x3=(self.dense_block3(x2))
        x3=self.trans_block3(x3)

        ## Classifier  ##

        x4=(self.dense_block4(x3))
        x4=self.trans_block4(x4)

        # x4=x4+x2
        x4=torch.cat([x4,x2],1)

        x5=(self.dense_block5(x4))
        x5=self.trans_block5(x5)

        # x5=x5+x1
        x5=torch.cat([x5,x1],1)

        x6=(self.dense_block6(x5))
        x6=(self.trans_block6(x6))

        shape_out = x6.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]


        x11 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x21 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x31 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x41 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x51 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)



        x6=torch.cat([x6,x51,x41,x31,x21,x11,x],1)

        return x6


class Dense_base_down1(nn.Module):
    def __init__(self):
        super(Dense_base_down1, self).__init__()

        self.dense_block1=BottleneckBlock1(3,13)
        self.trans_block1=TransitionBlock1(16,8)

        ############# Block2-down 32-32  ##############
        self.dense_block2=BottleneckBlock1(8,16)
        self.trans_block2=TransitionBlock1(24,16)

        ############# Block3-down  16-16 ##############
        self.dense_block3=BottleneckBlock1(16,16)
        self.trans_block3=TransitionBlock3(32,16)

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock1(16,16)
        self.trans_block4=TransitionBlock3(32,16)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock1(32,8)
        self.trans_block5=TransitionBlock(40,8)

        self.dense_block6=BottleneckBlock1(16,8)
        self.trans_block6=TransitionBlock(24,4)


        self.conv_refin=nn.Conv2d(11,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv11 = nn.Conv2d(8, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv21 = nn.Conv2d(16, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv31 = nn.Conv2d(16, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv41 = nn.Conv2d(32, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv51 = nn.Conv2d(16, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv61 = nn.Conv2d(8, 1, kernel_size=3,stride=1,padding=1)  # 1mm

        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)


        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)



    def forward(self, x):
        ## 256x256
        x1=self.dense_block1(x)
        x1=self.trans_block1(x1)

        ###  32x32
        x2=(self.dense_block2(x1))
        x2=self.trans_block2(x2)

        # print x2.size()
        ### 16 X 16
        x3=(self.dense_block3(x2))
        x3=self.trans_block3(x3)

        ## Classifier  ##

        x4=(self.dense_block4(x3))
        x4=self.trans_block4(x4)

        x4=torch.cat([x4,x2],1)

        x5=(self.dense_block5(x4))
        x5=self.trans_block5(x5)

        # x5=x5+x1
        x5=torch.cat([x5,x1],1)

        x6=(self.dense_block6(x5))
        x6=(self.trans_block6(x6))

        shape_out = x6.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]
        x11 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x21 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x31 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x41 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x51 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)



        x6=torch.cat([x6,x51,x41,x31,x21,x11,x],1)

        return x6

class Dense_base_down0(nn.Module):
    def __init__(self):
        super(Dense_base_down0, self).__init__()

        self.dense_block1=BottleneckBlock(3,5)
        self.trans_block1=TransitionBlock1(8,4)

        ############# Block2-down 32-32  ##############
        self.dense_block2=BottleneckBlock(4,8)
        self.trans_block2=TransitionBlock3(12,12)

        ############# Block3-down  16-16 ##############
        self.dense_block3=BottleneckBlock(12,4)
        self.trans_block3=TransitionBlock3(16,12)

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock(12,4)
        self.trans_block4=TransitionBlock3(16,12)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock(24,8)
        self.trans_block5=TransitionBlock3(32,4)

        self.dense_block6=BottleneckBlock(8,8)
        self.trans_block6=TransitionBlock(16,4)

        self.conv11 = nn.Conv2d(4, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv21 = nn.Conv2d(12, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv31 = nn.Conv2d(12, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv41 = nn.Conv2d(24, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv51 = nn.Conv2d(8, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv61 = nn.Conv2d(8, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.upsample = F.upsample_nearest
        self.relu=nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        ## 256x256
        x1=self.dense_block1(x)
        x1=self.trans_block1(x1)

        ###  32x32
        x2=(self.dense_block2(x1))
        x2=self.trans_block2(x2)

        # print x2.size()
        ### 16 X 16
        x3=(self.dense_block3(x2))
        x3=self.trans_block3(x3)

        ## Classifier  ##

        x4=(self.dense_block4(x3))
        x4=self.trans_block4(x4)

        x4=torch.cat([x4,x2],1)

        x5=(self.dense_block5(x4))
        x5=self.trans_block5(x5)

        # x5=x5+x1
        x5=torch.cat([x5,x1],1)

        x6=(self.dense_block6(x5))
        x6=(self.trans_block6(x6))
        shape_out = x6.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]
        x11 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x21 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x31 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x41 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x51 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)



        x6=torch.cat([x6,x51,x41,x31,x21,x11,x],1)


        return x6


class single(nn.Module):
    def __init__(self):
        super(single, self).__init__()

        self.dense_block1=BottleneckBlock(3,5)
        self.trans_block1=TransitionBlock3(8,4)

        ############# Block2-down 32-32  ##############
        self.dense_block2=BottleneckBlock(4,8)
        self.trans_block2=TransitionBlock3(12,12)

        ############# Block3-down  16-16 ##############
        self.dense_block3=BottleneckBlock(12,4)
        self.trans_block3=TransitionBlock3(16,12)

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock(12,4)
        self.trans_block4=TransitionBlock3(16,12)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock(12,8)
        self.trans_block5=TransitionBlock3(20,4)

        self.dense_block6=BottleneckBlock(4,8)
        self.trans_block6=TransitionBlock3(12,8)

        self.tanh = nn.Tanh()

        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm

        self.refine3 = nn.Conv2d(11, 6, kernel_size=3, stride=1, padding=1)
        self.refine4 = nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1)

        self.upsample = F.upsample_nearest
        self.relu=nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, label):
        ## 256x256
        x1=self.dense_block1(x)
        x1=self.trans_block1(x1)

        ###  32x32
        x2=(self.dense_block2(x1))
        x2=self.trans_block2(x2)

        # print x2.size()
        ### 16 X 16
        x3=(self.dense_block3(x2))
        x3=self.trans_block3(x3)

        ## Classifier  ##

        x4=(self.dense_block4(x3))
        x4=self.trans_block4(x4)

        x4=x4+x2

        x5=(self.dense_block5(x4))
        x5=self.trans_block5(x5)

        x5=x5+x1

        x6=(self.dense_block6(x5))
        x9=(self.trans_block6(x6))

        x9=torch.cat([x9,x],1)
        x9 = self.relu(self.refine3(x9))

        residual = self.tanh(self.refine4(x9))

        return residual

def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function

def _bn_function_factory1(relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(concated_features))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

class _DenseLayer1(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer1, self).__init__()
        #self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        #self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory1(self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(bottleneck_output))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class _Transition1(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition1, self).__init__()
        #self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class _DenseBlock1(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock1, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer1(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            ) 
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)

class DenseNet(nn.Module):
    def __init__(self, growth_rate=16, block_config=(3,6, 12,12), compression=0.5,
                 num_init_features=64, bn_size=4, drop_rate=0,
                 num_classes=10, small_inputs=True, efficient=False):

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = 8 if small_inputs else 7

        # First convolution
        if small_inputs:

            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) :
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Initialization
        # for name, param in self.named_parameters():
        #     if 'conv' in name and 'weight' in name:
        #         n = param.size(0) * param.size(2) * param.size(3)
        #         param.data.normal_().mul_(math.sqrt(2. / n))
        #     elif 'norm' in name and 'weight' in name:
        #         param.data.fill_(1)
        #     elif 'norm' in name and 'bias' in name:
        #         param.data.fill_(0)
        #     elif 'classifier' in name and 'bias' in name:
        #         param.data.fill_(0)

        

        ############# Block5-up  16-16 ##############
        self.dense_block5 = BottleneckBlockdy(163, 93)
        self.trans_block5 = TransitionBlockdy(256, 128)

        ############# Block6-up 32-32   ##############
        self.dense_block6 = BottleneckBlockdy(262, 128)
        self.trans_block6 = TransitionBlockdy(390, 128)

        ############# Block7-up 64-64   ##############

        self.dense_block7 = BottleneckBlockdy(204, 128)
        self.trans_block7 = TransitionBlockdy(332, 128)

        self.dense_block8 = BottleneckBlockdy(128, 64)
        self.trans_block8 = TransitionBlockdy(192, 32)



        self.tanh = nn.Tanh()
        self.refine3 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

        # self= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        # self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        ## 256*256
        x0 = self.features.relu0(self.features.conv0(x))
        #print ('x0.size:',x0.size())

        ## 256*256
        x1 = self.features.denseblock1(x0)
        #print ('x1.size:',x1.size())
        x1 = self.features.transition1(x1)

        #print ('x1.size:', x1.size())
        ###  128*128
        x2 = self.features.transition2(self.features.denseblock2(x1))
        #print ('x2.size:',x2.size())

        ### 64X 64
        x3 = self.features.transition3(self.features.denseblock3(x2))

        #print ('x3.size:', x3.size())
        ## 32 X 32
        x4 = self.features.denseblock4(x3)
        x4 = self.features.transition4(x4)
        #print ('x4.size:', x4.size())


        x5 = self.trans_block5(self.dense_block5(x4))
        x53 = torch.cat([x5, x3], 1)

        #print ('x5.size:', x5.size())
        ## 64 X 64
        x6 = self.trans_block6(self.dense_block6(x53))
        x62 = torch.cat([x6, x2], 1)

        ##  128 X 128
        x7 = self.trans_block7(self.dense_block7(x62))

        x8 = self.trans_block8(self.dense_block8(x7))

        #print ('x8.size:', x8.size())
        ##  256 X 256

        dehaze = self.tanh(self.refine3(x8))

        # dehaze=x - residual

        return dehaze


class DenseNet1(nn.Module):
    def __init__(self, growth_rate=16, block_config=(3, 6, 12, 12), compression=0.5,
                 num_init_features=64, bn_size=4, drop_rate=0,
                 num_classes=10, small_inputs=True, efficient=False):

        super(DenseNet1, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = 8 if small_inputs else 7

        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config):
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Initialization
        # for name, param in self.named_parameters():
        #     if 'conv' in name and 'weight' in name:
        #         n = param.size(0) * param.size(2) * param.size(3)
        #         param.data.normal_().mul_(math.sqrt(2. / n))
        #     elif 'norm' in name and 'weight' in name:
        #         param.data.fill_(1)
        #     elif 'norm' in name and 'bias' in name:
        #         param.data.fill_(0)
        #     elif 'classifier' in name and 'bias' in name:
        #         param.data.fill_(0)

        ############# Block5-up  16-16 ##############
        self.dense_block5 = BottleneckBlockdy(163, 93)
        self.trans_block5 = TransitionBlockdy(256, 128)

        ############# Block6-up 32-32   ##############
        self.dense_block6 = BottleneckBlockdy(262, 128)
        self.trans_block6 = TransitionBlockdy(390, 128)

        ############# Block7-up 64-64   ##############

        self.dense_block7 = BottleneckBlockdy(204, 128)
        self.trans_block7 = TransitionBlockdy(332, 128)

        self.dense_block8 = BottleneckBlockdy(184, 64)
        self.trans_block8 = TransitionBlockdy(248, 32)

        self.tanh = nn.Tanh()
        self.refine3 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

        # self= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        # self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        ## 256*256
        x0 = self.features.relu0(self.features.conv0(x))
        # print ('x0.size:',x0.size())

        ## 256*256
        x1 = self.features.denseblock1(x0)
        # print ('x1.size:',x1.size())
        x1 = self.features.transition1(x1)

        # print ('x1.size:', x1.size())
        ###  128*128
        x2 = self.features.transition2(self.features.denseblock2(x1))
        # print ('x2.size:',x2.size())

        ### 64X 64
        x3 = self.features.transition3(self.features.denseblock3(x2))

        # print ('x3.size:', x3.size())
        ## 32 X 32
        x4 = self.features.denseblock4(x3)
        x4 = self.features.transition4(x4)
        # print ('x4.size:', x4.size())

        x5 = self.trans_block5(self.dense_block5(x4))
        x53 = torch.cat([x5, x3], 1)

        # print ('x5.size:', x5.size())
        ## 64 X 64
        x6 = self.trans_block6(self.dense_block6(x53))
        x62 = torch.cat([x6, x2], 1)

        ##  128 X 128
        x7 = self.trans_block7(self.dense_block7(x62))

        x71 = torch.cat([x7, x1], 1)
        x8 = self.trans_block8(self.dense_block8(x71))

        # print ('x8.size:', x8.size())
        ##  256 X 256

        dehaze = self.tanh(self.refine3(x8))

        # dehaze=x - residual

        return dehaze
        
        
class DenseNet2(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 6, 12, 12), compression=0.5,
                 num_init_features=64, bn_size=4, drop_rate=0,
                 num_classes=10, small_inputs=True, efficient=False):

        super(DenseNet2, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = 8 if small_inputs else 7

        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config):
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Initialization
        # for name, param in self.named_parameters():
        #     if 'conv' in name and 'weight' in name:
        #         n = param.size(0) * param.size(2) * param.size(3)
        #         param.data.normal_().mul_(math.sqrt(2. / n))
        #     elif 'norm' in name and 'weight' in name:
        #         param.data.fill_(1)
        #     elif 'norm' in name and 'bias' in name:
        #         param.data.fill_(0)
        #     elif 'classifier' in name and 'bias' in name:
        #         param.data.fill_(0)

        ############# Block5-up  16-16 ##############
        self.dense_block5 = BottleneckBlockdy(328, 164)
        self.trans_block5 = TransitionBlockdy(492, 246)

        ############# Block6-up 32-32   ##############
        self.dense_block6 = BottleneckBlockdy(518, 259)
        self.trans_block6 = TransitionBlockdy(777, 300)

        ############# Block7-up 64-64   ##############

        self.dense_block7 = BottleneckBlockdy(460, 230 )
        self.trans_block7 = TransitionBlockdy(690, 256)

        self.dense_block8 = BottleneckBlockdy(384, 64)
        self.trans_block8 = TransitionBlockdy(448, 64)

        self.tanh = nn.Tanh()
        self.refine3 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

        # self= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        # self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        ## 256*256
        x0 = self.features.relu0(self.features.conv0(x))
        
        ## 256*256
        x1 = self.features.denseblock1(x0)
        
        x1 = self.features.transition1(x1)

        
        ###  128*128
        x2 = self.features.transition2(self.features.denseblock2(x1))
       

        ### 64X 64
        x3 = self.features.transition3(self.features.denseblock3(x2))

        
        ## 32 X 32
        x4 = self.features.denseblock4(x3)
        x4 = self.features.transition4(x4)
        

        x5 = self.trans_block5(self.dense_block5(x4))
        x53 = torch.cat([x5, x3], 1)

       
        ## 64 X 64
        x6 = self.trans_block6(self.dense_block6(x53))
        x62 = torch.cat([x6, x2], 1)

        ##  128 X 128
        x7 = self.trans_block7(self.dense_block7(x62))

        x71 = torch.cat([x7, x1], 1)
        x8 = self.trans_block8(self.dense_block8(x71))

        # print ('x8.size:', x8.size())
        ##  256 X 256

        dehaze = self.tanh(self.refine3(x8))

        # dehaze=x - residual

        return dehaze


class Dense_rain_cvprw3dy21(nn.Module):
	def __init__(self):
		super(Dense_rain_cvprw3dy21, self).__init__()

		############# 256-256  ##############
		haze_class = models.densenet121(pretrained=True)

		self.conv0 = haze_class.features.conv0
		self.relu0 = haze_class.features.relu0

		############# Block1-down 256-256  ##############
		self.dense_block1 = haze_class.features.denseblock1
		self.trans_block1 = haze_class.features.transition1

		############# Block2-down 128-128  ##############
		self.dense_block2 = haze_class.features.denseblock2
		self.trans_block2 = haze_class.features.transition2

		############# Block3-down  64-64 ##############
		self.dense_block3 = haze_class.features.denseblock3
		self.trans_block3 = haze_class.features.transition3

		# ############# Block31-down  xx-xx ##############
		self.dense_block31 = haze_class.features.denseblock4
		# self.trans_block31=haze_class.features.transition4

		self.dense_norm31 = haze_class.features.norm5

		############# Block4-up  8-8  ##############
		self.dense_block4 = BottleneckBlockdy(512, 256)
		self.trans_block4 = TransitionBlockdy(768, 320)

		############# Block5-up  16-16 ##############
		self.dense_block5 = BottleneckBlockdy(448, 240)
		self.trans_block5 = TransitionBlockdy(688, 256)

		############# Block6-up 32-32   ##############
		self.dense_block6 = BottleneckBlockdy(320, 100)
		self.trans_block6 = TransitionBlockdy(420, 64)

		############# Block7-up 64-64   ##############

		self.conv_refin1 = nn.Conv2d(3, 64, 3, 1, 1)

		self.tanh = nn.Tanh()
		self.refine3 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

		# self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

		self.upsample = F.upsample_nearest

		self.relu = nn.LeakyReLU(0.2, inplace=True)

	def forward(self, x):
		## 256*256
		x0 = self.relu0(self.conv_refin1(x))
		# print ('x0.size:',x0.size())

		## 256*256
		x1 = self.dense_block1(x0)
		# print x1.size()
		x1 = self.trans_block1(x1)
		# print ('x1.size:', x1.size())

		###  128*128
		x2 = self.trans_block2(self.dense_block2(x1))
		# print  x2.size()
		# print ('x2.size:', x2.size())

		### 64X 64
		x3 = self.trans_block3(self.dense_block3(x2))
		# print ('x3.size:', x3.size())

		## 32 X 32
		x4 = self.trans_block4(self.dense_block4(x3))
		# print('x4:',x4.size())

		x11 = F.avg_pool2d (x1,2)
		x41 = torch.cat([x4, x11], 1)
		# print(x42.size())

		## 64 X 64
		x5 = self.trans_block5(self.dense_block5(x41))
		x01 = F.avg_pool2d (x0,2)
		x50 = torch.cat([x5, x01], 1)
		# print('x52.size())

		##  128 X 128
		x6 = self.trans_block6(self.dense_block6(x50))

		##  256 X 256

		dehaze = self.tanh(self.refine3(x6))

		# dehaze=x - residual

		return dehaze


class DenseNet6(nn.Module):
    def __init__(self, growth_rate=16, block_config=(3, 6, 12, 12), compression=0.5,
                 num_init_features=64, bn_size=4, drop_rate=0,
                 num_classes=10, small_inputs=True, efficient=False):

        super(DenseNet6, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = 8 if small_inputs else 7

        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config):
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Initialization
        # for name, param in self.named_parameters():
        #     if 'conv' in name and 'weight' in name:
        #         n = param.size(0) * param.size(2) * param.size(3)
        #         param.data.normal_().mul_(math.sqrt(2. / n))
        #     elif 'norm' in name and 'weight' in name:
        #         param.data.fill_(1)
        #     elif 'norm' in name and 'bias' in name:
        #         param.data.fill_(0)
        #     elif 'classifier' in name and 'bias' in name:
        #         param.data.fill_(0)

        self.conv1 = nn.Conv2d(266, 64, kernel_size=1,stride=1,padding=0)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()
        self.upsample = F.upsample_nearest

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        ## 256*256
        x0 = self.features.relu0(self.features.conv0(x))
        # print ('x0.size:',x0.size())
        shape_out = x0.data.size()
        shape_out1 = shape_out[2:4]

        ## 256*256
        x1 = self.features.denseblock1(x0)
        # print ('x1.size:',x1.size())
        x1 = self.features.transition1(x1)

        print ('x1.size:', x1.size())
        ###  128*128
        x2 = self.features.transition2(self.features.denseblock2(x1))
        print ('x2.size:',x2.size())

        ### 64X 64
        x3 = self.features.transition3(self.features.denseblock3(x2))

        shape_out2  = x1.data.size()
        #print(shape_out)
        shape_out2 = shape_out2[2:4]
        x4 = self.upsample(self.relu(x3),size=shape_out2)
        x5 = self.upsample(self.relu(x2), size=shape_out2)
        
        x541 = torch.cat([x5, x4, x1], 1)
        print ('x541.size:', x541.size())
        x6 = self.relu(self.conv1(x541))
        x6 = self.upsample(self.relu(self.conv2(x6)),size=shape_out1)
        print ('x6.size:', x6.size())

    
        dehaze = self.tanh(x6)

        # dehaze=x - residual

        return dehaze

class Dense_rain_cvprw3dy23(nn.Module):
	def __init__(self):
		super(Dense_rain_cvprw3dy23, self).__init__()

		############# 256-256  ##############
		haze_class = models.densenet121(pretrained=True)

		self.conv0 = haze_class.features.conv0
		self.relu0 = haze_class.features.relu0

		############# Block1-down 256-256  ##############
		self.dense_block1 = haze_class.features.denseblock1
		self.trans_block1 = haze_class.features.transition1

		############# Block2-down 128-128  ##############
		self.dense_block2 = haze_class.features.denseblock2
		self.trans_block2 = haze_class.features.transition2

		############# Block3-down  64-64 ##############
		self.dense_block3 = haze_class.features.denseblock3
		self.trans_block3 = haze_class.features.transition3

		# ############# Block31-down  xx-xx ##############
		self.dense_block31 = haze_class.features.denseblock4
		# self.trans_block31=haze_class.features.transition4

		self.dense_norm31 = haze_class.features.norm5

		############# Block4-up  8-8  ##############
		self.dense_block4 = BottleneckBlockdy1(512, 256)
		self.trans_block4 = TransitionBlockdy1(256, 128)

		############# Block5-up  16-16 ##############
		self.dense_block5 = BottleneckBlockdy1(384, 128)
		self.trans_block5 = TransitionBlockdy(128, 64)

		############# Block6-up 32-32   ##############
		self.dense_block6 = BottleneckBlockdy(64, 32)
		self.trans_block6 = TransitionBlockdy(96, 16)

		############# Block7-up 64-64   ##############

		self.conv_refin1 = nn.Conv2d(3, 64, 3, 1, 1)

		self.tanh = nn.Tanh()
		self.refine3 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

		# self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

		self.upsample = F.upsample_nearest

		self.relu = nn.LeakyReLU(0.2, inplace=True)

	def forward(self, x):
		## 256*256
		x0 = self.relu0(self.conv_refin1(x))
		# print ('x0.size:',x0.size())

		## 256*256
		x1 = self.dense_block1(x0)
		# print x1.size()
		x1 = self.trans_block1(x1)
		# print ('x1.size:', x1.size())

		###  128*128
		x2 = self.trans_block2(self.dense_block2(x1))
		# print  x2.size()
		# print ('x2.size:', x2.size())

		### 64X 64
		x3 = self.trans_block3(self.dense_block3(x2))
		# print ('x3.size:', x3.size())

		## 32 X 32
		x4 = self.trans_block4(self.dense_block4(x3))
		# print('x4:',x4.size())

		x42 = torch.cat([x4, x2], 1)
		# print(x42.size())

		## 64 X 64
		x5 = self.trans_block5(self.dense_block5(x42))
		# x52 = torch.cat([x5, x1], 1)
		# print('x52.size())

		##  128 X 128
		x6 = self.trans_block6(self.dense_block6(x5))

		##  256 X 256

		dehaze = self.tanh(self.refine3(x6))

		# dehaze=x - residual

		return dehaze


class DenseNet7(nn.Module):
    def __init__(self, growth_rate=24, block_config=(6, 6, 12, 12), compression=0.5,
                 num_init_features=64, bn_size=4, drop_rate=0,
                 num_classes=10, small_inputs=True, efficient=False):

        super(DenseNet7, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = 8 if small_inputs else 7

        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config):
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Initialization
        # for name, param in self.named_parameters():
        #     if 'conv' in name and 'weight' in name:
        #         n = param.size(0) * param.size(2) * param.size(3)
        #         param.data.normal_().mul_(math.sqrt(2. / n))
        #     elif 'norm' in name and 'weight' in name:
        #         param.data.fill_(1)
        #     elif 'norm' in name and 'bias' in name:
        #         param.data.fill_(0)
        #     elif 'classifier' in name and 'bias' in name:
        #         param.data.fill_(0)

        ############# Block5-up  16-16 ##############
        self.dense_block4 = BottleneckBlockdy(206, 64)
        self.trans_block4 = TransitionBlockdy(270, 80)

        ############# Block6-up 32-32   ##############
        self.dense_block5 = BottleneckBlockdy(204, 64)
        self.trans_block5 = TransitionBlockdy(268, 64)

        ############# Block7-up 64-64   ##############

        self.dense_block6 = BottleneckBlockdy(64, 32)
        self.trans_block6 = TransitionBlockdy(96, 16)

        self.tanh = nn.Tanh()
        self.refine3 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

        # self= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        # self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        ## 256*256
        x0 = self.features.relu0(self.features.conv0(x))
        # print ('x0.size:',x0.size())

        ## 256*256
        x1 = self.features.denseblock1(x0)
        # print ('x1.size:',x1.size())
        x1 = self.features.transition1(x1)

        x2 = self.features.transition2(self.features.denseblock2(x1))
       
        ### 64X 64
        x3 = self.features.transition3(self.features.denseblock3(x2))

        ## 32 X 32
        x4 = self.trans_block4(self.dense_block4(x3))

        x42 = torch.cat([x4, x2], 1)
        x5 = self.trans_block5(self.dense_block5(x42))

        # print ('x5.size:', x5.size())
        ## 64 X 64
        x6 = self.trans_block6(self.dense_block6(x5))

        ##  256 X 256

        dehaze = self.tanh(self.refine3(x6))

        # dehaze=x - residual

        return dehaze


class DenseNet8(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), compression=0.5,
                 num_init_features=64, bn_size=4, drop_rate=0,
                 num_classes=10, small_inputs=True, efficient=False):

        super(DenseNet8, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = 8 if small_inputs else 7

        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
            #self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock1(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config):
                trans = _Transition1(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        #self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Linear layer
        #self.classifier = nn.Linear(num_features, num_classes)

        # Initialization
        # for name, param in self.named_parameters():
        #     if 'conv' in name and 'weight' in name:
        #         n = param.size(0) * param.size(2) * param.size(3)
        #         param.data.normal_().mul_(math.sqrt(2. / n))
        #     elif 'norm' in name and 'weight' in name:
        #         param.data.fill_(1)
        #     elif 'norm' in name and 'bias' in name:
        #         param.data.fill_(0)
        #     elif 'classifier' in name and 'bias' in name:
        #         param.data.fill_(0)

        self.dense_block4 = BottleneckBlockdy(512, 256)
        self.trans_block4 = TransitionBlockdy(768, 128)

        ############# Block5-up  16-16 ##############
        self.dense_block5 = BottleneckBlockdy(384, 128)
        self.trans_block5 = TransitionBlockdy(512, 64)

        ############# Block6-up 32-32   ##############
        self.dense_block6 = BottleneckBlockdy(64, 32)
        self.trans_block6 = TransitionBlockdy(96, 16)

        ############# Block7-up 64-64   ##############

        self.conv_refin1 = nn.Conv2d(3, 64, 3, 1, 1)

        self.tanh = nn.Tanh()
        self.refine3 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

        # self= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        # self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        ## 256*256
        x0 = self.features.relu0(self.features.conv0(x))
        # print ('x0.size:',x0.size())

        ## 256*256
        x1 = self.features.denseblock1(x0)
        # print ('x1.size:',x1.size())
        x1 = self.features.transition1(x1)

        x2 = self.features.transition2(self.features.denseblock2(x1))

        ### 64X 64
        x3 = self.features.transition3(self.features.denseblock3(x2))

        ## 32 X 32
        x4 = self.trans_block4(self.dense_block4(x3))

        x42 = torch.cat([x4, x2], 1)
        x5 = self.trans_block5(self.dense_block5(x42))

        # print ('x5.size:', x5.size())
        ## 64 X 64
        x6 = self.trans_block6(self.dense_block6(x5))

        ##  256 X 256

        dehaze = self.tanh(self.refine3(x6))

        return dehaze

class DenseNet12(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), compression=0.5,
                 num_init_features=64, bn_size=4, drop_rate=0,
                 num_classes=10, small_inputs=True, efficient=False):

        super(DenseNet12, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = 8 if small_inputs else 7

        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
            #self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock1(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config):
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        #self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Linear layer
        #self.classifier = nn.Linear(num_features, num_classes)

        # Initialization
        # for name, param in self.named_parameters():
        #     if 'conv' in name and 'weight' in name:
        #         n = param.size(0) * param.size(2) * param.size(3)
        #         param.data.normal_().mul_(math.sqrt(2. / n))
        #     elif 'norm' in name and 'weight' in name:
        #         param.data.fill_(1)
        #     elif 'norm' in name and 'bias' in name:
        #         param.data.fill_(0)
        #     elif 'classifier' in name and 'bias' in name:
        #         param.data.fill_(0)

        self.dense_block4 = BottleneckBlockdy(512, 256)
        self.trans_block4 = TransitionBlockdy(768, 128)

        ############# Block5-up  16-16 ##############
        self.dense_block5 = BottleneckBlockdy(384, 128)
        self.trans_block5 = TransitionBlockdy(512, 64)

        ############# Block6-up 32-32   ##############
        self.dense_block6 = BottleneckBlockdy(64, 32)
        self.trans_block6 = TransitionBlockdy(96, 16)

        ############# Block7-up 64-64   ##############

        self.conv_refin1 = nn.Conv2d(3, 64, 3, 1, 1)

        self.tanh = nn.Tanh()
        self.refine3 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

        # self= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        # self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        ## 256*256
        x0 = self.features.relu0(self.features.conv0(x))
        # print ('x0.size:',x0.size())

        ## 256*256
        x1 = self.features.denseblock1(x0)
        # print ('x1.size:',x1.size())
        x1 = self.features.transition1(x1)

        x2 = self.features.transition2(self.features.denseblock2(x1))

        ### 64X 64
        x3 = self.features.transition3(self.features.denseblock3(x2))

        ## 32 X 32
        x4 = self.trans_block4(self.dense_block4(x3))

        x42 = torch.cat([x4, x2], 1)
        x5 = self.trans_block5(self.dense_block5(x42))

        # print ('x5.size:', x5.size())
        ## 64 X 64
        x6 = self.trans_block6(self.dense_block6(x5))

        ##  256 X 256

        dehaze = self.tanh(self.refine3(x6))

        return dehaze

class Dense_rain_cvprw3dy12_B(nn.Module):
	def __init__(self):
		super(Dense_rain_cvprw3dy12_B, self).__init__()

		############# 256-256  ##############
		haze_class = models.densenet169(pretrained=True)

		self.conv0 = haze_class.features.conv0
		self.relu0 = haze_class.features.relu0

		############# Block1-down 256-256  ##############
		self.dense_block1 = haze_class.features.denseblock1
		self.trans_block1 = haze_class.features.transition1

		############# Block2-down 128-128  ##############
		self.dense_block2 = haze_class.features.denseblock2
		self.trans_block2 = haze_class.features.transition2

		############# Block3-down  64-64 ##############
		self.dense_block3 = haze_class.features.denseblock3
		self.trans_block3 = haze_class.features.transition3

		# ############# Block31-down  xx-xx ##############
		self.dense_block31 = haze_class.features.denseblock4
		# self.trans_block31=haze_class.features.transition4

		self.dense_norm31 = haze_class.features.norm5

		############# Block4-up  8-8  ##############
		self.dense_block4 = BottleneckBlockdy(640, 256)
		self.trans_block4 = TransitionBlockdy(896, 320)

		############# Block5-up  16-16 ##############
		self.dense_block5 = BottleneckBlockdy(576, 128)
		self.trans_block5 = TransitionBlockdy(704, 256)

		############# Block6-up 32-32   ##############
		self.dense_block6 = BottleneckBlockdy(256, 64)
		self.trans_block6 = TransitionBlockdy(320, 64)

		############# Block7-up 64-64   ##############

		self.conv_refin1 = nn.Conv2d(3, 64, 3, 1, 1)

		self.tanh = nn.Tanh()
		self.refine3 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

		# self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

		self.upsample = F.upsample_nearest

		self.relu = nn.LeakyReLU(0.2, inplace=True)

	def forward(self, x):
		## 256*256
		x0 = self.relu0(self.conv_refin1(x))
		#print ('x0.size:',x0.size())

		## 256*256
		x1 = self.dense_block1(x0)
		# print x1.size()
		x1 = self.trans_block1(x1)
		#print ('x1.size:', x1.size())

		###  128*128
		x2 = self.trans_block2(self.dense_block2(x1))
		# print  x2.size()
		#print ('x2.size:', x2.size())

		### 64X 64
		x3 = self.trans_block3(self.dense_block3(x2))
		#print ('x3.size:', x3.size())

		## 32 X 32
		x4 = self.trans_block4(self.dense_block4(x3))
		#print('x4:',x4.size())

		x42 = torch.cat([x4, x2], 1)
		#print(x42.size())

		## 64 X 64
		x5 = self.trans_block5(self.dense_block5(x42))
		# x52 = torch.cat([x5, x1], 1)
		#print('x3.size:', x3.size())

		##  128 X 128
		x6 = self.trans_block6(self.dense_block6(x5))

		##  256 X 256

		dehaze = self.tanh(self.refine3(x6))

		# dehaze=x - residual

		return dehaze

class Dense_rain_cvprw3dy12_R_B(nn.Module):
	def __init__(self):
		super(Dense_rain_cvprw3dy12_R_B, self).__init__()

		############# 256-256  ##############
		haze_class = models.densenet169(pretrained=True)

		self.conv0 = haze_class.features.conv0
		self.relu0 = haze_class.features.relu0

		############# Block1-down 256-256  ##############
		self.dense_block1 = haze_class.features.denseblock1
		self.trans_block1 = haze_class.features.transition1

		############# Block2-down 128-128  ##############
		self.dense_block2 = haze_class.features.denseblock2
		self.trans_block2 = haze_class.features.transition2

		############# Block3-down  64-64 ##############
		self.dense_block3 = haze_class.features.denseblock3
		self.trans_block3 = haze_class.features.transition3

		# ############# Block31-down  xx-xx ##############
		self.dense_block31 = haze_class.features.denseblock4
		# self.trans_block31=haze_class.features.transition4

		self.dense_norm31 = haze_class.features.norm5

		############# Block4-up  8-8  ##############
		self.dense_block4 = BottleneckBlockdy(640, 320)
		self.trans_block4 = TransitionBlockdy(960, 320)

		############# Block5-up  16-16 ##############
		self.dense_block5 = BottleneckBlockdy(576, 256)
		self.trans_block5 = TransitionBlockdy(832, 256)

		############# Block6-up 32-32   ##############
		self.dense_block6 = BottleneckBlockdy(256, 128)
		self.trans_block6 = TransitionBlockdy(384, 64)

		############# Block7-up 64-64   ##############

		self.conv_refin1 = nn.Conv2d(3, 64, 3, 1, 1)

		self.conv_refin6 = nn.Conv2d(768,640,3,1,1)
		self.conv_refin5 = nn.Conv2d(256,128,1,1,0)
		self.tanh = nn.Tanh()
		self.conv_refin3 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

		self.conv_refin2= nn.Conv2d(64, 32, kernel_size=1,stride=1,padding=0)

		self.upsample = F.upsample_nearest

		self.conv_refine4 = nn.Conv2d(160, 128, kernel_size=3, stride=1, padding=1)
        #self.relu = nn.LeakyReLU(0.2, inplace=True)

	def forward(self, x):
		## 256*256
		x0 = self.relu0(self.conv_refin1(x))

		#print('x0.size:',x0.size())
		x01 = self.conv_refin2(F.avg_pool2d(x0,2))
        #print ('x0.size:',x0.size())

		## 256*256
		x1 = self.dense_block1(x0)
		#print ('x01.size:',x01.size())
		x1 = self.trans_block1(x1)
		#print ('x1.size:', x1.size())

		###  128*128
		x10 =self.conv_refine4(torch.cat([x01,x1],1))
		x2=self.trans_block2(self.dense_block2(x10))
		#print ('x2.size:', x2.size())

		### 64X 64
		x3 = self.trans_block3(self.dense_block3(x2))
		#print('x3.size:', x3.size())
		x22=self.conv_refin5(F.avg_pool2d(x2,2))

		## 32 X 32
		x4 = self.trans_block4(self.dense_block4(self.conv_refin6(torch.cat([x3, x22], 1))))
		#print('x4:',x4.size())

		x42 = torch.cat([x4, x2], 1)
		#print(x42.size())

		## 64 X 64
		x5 = self.trans_block5(self.dense_block5(x42))
		# x52 = torch.cat([x5, x1], 1)
		#print('x5.size:', x5.size())

		##  128 X 128
		x6 = self.trans_block6(self.dense_block6(x5))

		##  256 X 256

		dehaze = self.tanh(self.conv_refin3(x6))

		# dehaze=x - residual

		return dehaze
   
class Dense_rain_cvprw3dy12_B(nn.Module):
	def __init__(self):
		super(Dense_rain_cvprw3dy12_B, self).__init__()

		############# 256-256  ##############
		haze_class = models.densenet169(pretrained=True)

		self.conv0 = haze_class.features.conv0
		self.relu0 = haze_class.features.relu0

		############# Block1-down 256-256  ##############
		self.dense_block1 = haze_class.features.denseblock1
		self.trans_block1 = haze_class.features.transition1

		############# Block2-down 128-128  ##############
		self.dense_block2 = haze_class.features.denseblock2
		self.trans_block2 = haze_class.features.transition2

		############# Block3-down  64-64 ##############
		self.dense_block3 = haze_class.features.denseblock3
		self.trans_block3 = haze_class.features.transition3

		# ############# Block31-down  xx-xx ##############
		self.dense_block31 = haze_class.features.denseblock4
		# self.trans_block31=haze_class.features.transition4

		self.dense_norm31 = haze_class.features.norm5

		############# Block4-up  8-8  ##############
		self.dense_block4 = BottleneckBlockdy(640, 256)
		self.trans_block4 = TransitionBlockdy(896, 320)

		############# Block5-up  16-16 ##############
		self.dense_block5 = BottleneckBlockdy(576, 128)
		self.trans_block5 = TransitionBlockdy(704, 256)

		############# Block6-up 32-32   ##############
		self.dense_block6 = BottleneckBlockdy(256, 64)
		self.trans_block6 = TransitionBlockdy(320, 64)

		############# Block7-up 64-64   ##############

		self.conv_refin1 = nn.Conv2d(3, 64, 3, 1, 1)

		self.tanh = nn.Tanh()
		self.refine3 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

		# self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

		self.upsample = F.upsample_nearest

		self.relu = nn.LeakyReLU(0.2, inplace=True)

	def forward(self, x):
		## 256*256
		x0 = self.relu0(self.conv_refin1(x))
		#print ('x0.size:',x0.size())

		## 256*256
		x1 = self.dense_block1(x0)
		# print x1.size()
		x1 = self.trans_block1(x1)
		#print ('x1.size:', x1.size())

		###  128*128
		x2 = self.trans_block2(self.dense_block2(x1))
		# print  x2.size()
		#print ('x2.size:', x2.size())

		### 64X 64
		x3 = self.trans_block3(self.dense_block3(x2))
		#print ('x3.size:', x3.size())

		## 32 X 32
		x4 = self.trans_block4(self.dense_block4(x3))
		#print('x4:',x4.size())

		x42 = torch.cat([x4, x2], 1)
		#print(x42.size())

		## 64 X 64
		x5 = self.trans_block5(self.dense_block5(x42))
		# x52 = torch.cat([x5, x1], 1)
		#print('x3.size:', x3.size())

		##  128 X 128
		x6 = self.trans_block6(self.dense_block6(x5))

		##  256 X 256

		dehaze = self.tanh(self.refine3(x6))

		# dehaze=x - residual

		return dehaze

class Dense_rain_cvprw3dy12_R_B(nn.Module):
	def __init__(self):
		super(Dense_rain_cvprw3dy12_R_B, self).__init__()

		############# 256-256  ##############
		haze_class = models.densenet169(pretrained=True)

		self.conv0 = haze_class.features.conv0
		self.relu0 = haze_class.features.relu0

		############# Block1-down 256-256  ##############
		self.dense_block1 = haze_class.features.denseblock1
		self.trans_block1 = haze_class.features.transition1

		############# Block2-down 128-128  ##############
		self.dense_block2 = haze_class.features.denseblock2
		self.trans_block2 = haze_class.features.transition2

		############# Block3-down  64-64 ##############
		self.dense_block3 = haze_class.features.denseblock3
		self.trans_block3 = haze_class.features.transition3

		# ############# Block31-down  xx-xx ##############
		self.dense_block31 = haze_class.features.denseblock4
		# self.trans_block31=haze_class.features.transition4

		self.dense_norm31 = haze_class.features.norm5

		############# Block4-up  8-8  ##############
		self.dense_block4 = BottleneckBlockdy(640, 320)
		self.trans_block4 = TransitionBlockdy(960, 320)

		############# Block5-up  16-16 ##############
		self.dense_block5 = BottleneckBlockdy(576, 256)
		self.trans_block5 = TransitionBlockdy(832, 256)

		############# Block6-up 32-32   ##############
		self.dense_block6 = BottleneckBlockdy(256, 128)
		self.trans_block6 = TransitionBlockdy(384, 64)

		############# Block7-up 64-64   ##############

		self.conv_refin1 = nn.Conv2d(3, 64, 3, 1, 1)

		self.conv_refin6 = nn.Conv2d(768,640,3,1,1)
		self.conv_refin5 = nn.Conv2d(256,128,1,1,0)
		self.tanh = nn.Tanh()
		self.conv_refin3 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

		self.conv_refin2= nn.Conv2d(64, 32, kernel_size=1,stride=1,padding=0)

		self.upsample = F.upsample_nearest

		self.conv_refine4 = nn.Conv2d(160, 128, kernel_size=3, stride=1, padding=1)
        #self.relu = nn.LeakyReLU(0.2, inplace=True)

	def forward(self, x):
		## 256*256
		x0 = self.relu0(self.conv_refin1(x))

		#print('x0.size:',x0.size())
		x01 = self.conv_refin2(F.avg_pool2d(x0,2))
        #print ('x0.size:',x0.size())

		## 256*256
		x1 = self.dense_block1(x0)
		#print ('x01.size:',x01.size())
		x1 = self.trans_block1(x1)
		#print ('x1.size:', x1.size())

		###  128*128
		x10 =self.conv_refine4(torch.cat([x01,x1],1))
		x2=self.trans_block2(self.dense_block2(x10))
		#print ('x2.size:', x2.size())

		### 64X 64
		x3 = self.trans_block3(self.dense_block3(x2))
		#print('x3.size:', x3.size())
		x22=self.conv_refin5(F.avg_pool2d(x2,2))

		## 32 X 32
		x4 = self.trans_block4(self.dense_block4(self.conv_refin6(torch.cat([x3, x22], 1))))
		#print('x4:',x4.size())

		x42 = torch.cat([x4, x2], 1)
		#print(x42.size())

		## 64 X 64
		x5 = self.trans_block5(self.dense_block5(x42))
		# x52 = torch.cat([x5, x1], 1)
		#print('x5.size:', x5.size())

		##  128 X 128
		x6 = self.trans_block6(self.dense_block6(x5))

		##  256 X 256

		dehaze = self.tanh(self.conv_refin3(x6))

		# dehaze=x - residual

		return dehaze
 
 
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=2):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        print('out.shape:',out.shape)
        out = self.classifier(out)
        return out
