# FD-GAN
## FD-GAN: Generative adversarial Networks with Fusion-discriminator for Single Image Dehazing(AAAI'20)
[PAPER](https://arxiv.org/abs/2001.06968)

[Yu Dong](https://github.com/WeilanAnnn).  [Yihao Liu](https://github.com/DoctorYy)

<center >
    <img src= "https://github.com/WeilanAnnn/FD-GAN/blob/master/facades/network.png"/>
</center>


In this paper, we propose a fully end-to-end algorithm FD-GAN for image dehazing. Moreover, we develop a novel Fusion-discriminator which can integrate the frequency information as additional priors and constraints into the dehazing network. Our method can generate more visually pleasing dehazed results with less color distortion. Extensive experimental results have demonstrated that our method performs favorably against several state-of-the-art methods on both synthetic datasets and real-world hazy images.

<center >
    <img src="https://github.com/WeilanAnnn/FD-GAN/blob/master/facades/RealImage.png" width="1200"/>
</center>

## Prerequisites
1. Ubuntu 18.04
2. Python 3
3. NVIDIA GPU + CUDA CuDNN (CUDA 8.0)

## Installation
1. conda install pytorch=0.3.0 torchvision cuda80 -c pytorch
2. Install python package:numpy,scipy,PIL,skimage,h5py

## Demo using pre-trained model
Since the proposed method uses hdf5 file to load the traning samples, the **generate_testsample.py** helps you to creat the testing or training sample yourself.

If your images are real:
```
python demo.py --valDataroot ./facades/'your_folder_name' --netG ./testmodel/netG_epoch_real.pth
```
If your images are synthetic:
```
python demo.py --valDataroot ./facades/'your_folder_name' --netG ./testmodel/netG_epoch_synthetic.pth
```
To obtain the best performance on synthetic and real-world datasets respectively, we provide two models from different  iterations in one  training procedure. In addition, please use netG.train() for testing since the batch for training is 1.

Pre-trained dehazing models can be downloaded at (put it in the folder '**test_model**'):

https://pan.baidu.com/s/10IgnZ0YiGsUxrgxoQQhsOg

or

https://drive.google.com/drive/folders/1Jkf9NgBrGHErQMwFN7wv1QY_vPYUt19r?usp=sharing

## Metric
You can run the **PSNRSSIM.py** for quantitative results
```
python PSNRSSIM.py --gt_dir ./your_folder_name --result_dir ./your_folder_name
```

## Datasets
 You can download our synthetic test-data: RESIDE-SOTS and NTIRE dataset(strored in Hdf5 file and PNG) as following URL： 
https://pan.baidu.com/s/1oZwVX8FWFNzRaY_JyVB1pA

https://pan.baidu.com/s/1U6RjKF-UYXvBIHDt7SU0Ww

## How to read the Hdf5 file
Following are the sample python codes how to read the Hdf5 file:
```
import matplotlib.pyplot as plt
file_name=self.root+'/'+str(index)+'.h5'
f=h5py.File(file_name,'r')

gt=f['gt'][:]
haze=f['haze'][:]
plt.subplot(1,2,1), plt.title('gt')
plt.imshow(gt)
plt.subplot(1,2,2),plt.title('haze')
plt.imshow(haze)
plt.show()
```
## Citation


## Acknowledgments
Thank all co-authors so much!
