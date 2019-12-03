import torch.utils.data as data
from PIL import Image
import os
from skimage.transform import resize
import os.path
import numpy as np
import h5py
import glob
import scipy.ndimage
from PIL import Image
import pdb
from numpy import *
from torchvision import transforms
from skimage import transform
IMG_EXTENSIONS = [
  '.jpg', '.JPG', '.jpeg', '.JPEG',
  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
  return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
  images = []
  if not os.path.isdir(dir):
    raise Exception('Check dataroot')
  for root, _, fnames in sorted(os.walk(dir)):
    for fname in fnames:
      if is_image_file(fname):
        path = os.path.join(dir, fname)
        item = path
        images.append(item)
  return images

def default_loader(path):
  return Image.open(path).convert('RGB')

class pix2pix(data.Dataset):
  def __init__(self, root, transform=None, loader=default_loader, seed=None):
    # imgs = make_dataset(root)
    # if len(imgs) == 0:
    #   raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
    #              "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.root = root
    # self.imgs = imgs
    self.transform = transform
    self.loader = loader

    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index):
    # index = np.random.randint(1,self.__len__())
    # index = np.random.randint(self.__len__(), size=1)[0]

    # path = self.imgs[index]
    # img = self.loader(path)
    #img = img.resize((w, h), Image.BILINEAR)



    file_name=self.root+'/'+str(index)+'.h5'
    f=h5py.File(file_name,'r')

    haze_image=f['haze'][:]
    # trans_map=f['trans'][:]
    # ato_map=f['ato'][:]
    GT=f['gt'][:]
    haze_image = np.swapaxes(haze_image, 0, 2)
    # trans_map=np.swapaxes(trans_map,0,2)
    # ato_map=np.swapaxes(ato_map,0,2)
    GT = np.swapaxes(GT, 0, 2)

    haze_image = np.swapaxes(haze_image, 1, 2)
    # trans_map=np.swapaxes(trans_map,1,2)
    # ato_map=np.swapaxes(ato_map,1,2)
    GT = np.swapaxes(GT, 1, 2)


    # if self.transform is not None:

    #   haze_image = self.transform(haze_image)
	#
	#
    #   GT = self.transform(GT)



      # haze_image = Image.fromarray(haze_image, mode='RGB')
      # GT = Image.fromarray(GT, mode='RGB')
      # # haze_image,GT = tran.ToPILImage(haze_image,GT )
      # haze_image = self.transform(haze_image)
      # GT = self.transform(GT)

	# if self.transform is None:
     #  haze_image=np.swapaxes(haze_image,0,2)
     #  # trans_map=np.swapaxes(trans_map,0,2)
     #  # ato_map=np.swapaxes(ato_map,0,2)
     #  GT=np.swapaxes(GT,0,2)
	#
	# #
	#
    #   haze_image=np.swapaxes(haze_image,1,2)
    #   # trans_map=np.swapaxes(trans_map,1,2)
    #   # ato_map=np.swapaxes(ato_map,1,2)
    #   GT=np.swapaxes(GT,1,2)


    # haze_image=np.swapaxes(haze_image,0,2)
    # # trans_map=np.swapaxes(trans_map,0,2)
    # # ato_map=np.swapaxes(ato_map,0,2)
    # GT=np.swapaxes(GT,0,2)
	#
	#
	#
    # haze_image=np.swapaxes(haze_image,1,2)
    # # trans_map=np.swapaxes(trans_map,1,2)
    # # ato_map=np.swapaxes(ato_map,1,2)
    # GT=np.swapaxes(GT,1,2)
	#



    # if np.random.uniform()>0.5:
    #   haze_image=np.flip(haze_image,2).copy()
    #   GT = np.flip(GT, 2).copy()
    #   trans_map=np.flip(trans_map, 2).copy()
    # if np.random.uniform()>0.5:
    #   angle = np.random.uniform(-10, 10)
    #   haze_image=scipy.ndimage.interpolation.rotate(haze_image, angle)
    #   GT = scipy.ndimage.interpolation.rotate(GT, angle)

    # if np.random.uniform()>0.5:
    #   angle = np.random.uniform(-10, 10)
    #   haze_image=scipy.ndimage.interpolation.rotate(haze_image, angle)
    #   GT = scipy.ndimage.interpolation.rotate(GT, angle)

    # if np.random.uniform()>0.5:
    #   std = np.random.uniform(0.2, 1.2)
    #   haze_image = scipy.ndimage.filters.gaussian_filter(haze_image, std,mode='constant')

    #   haze_image=np.random.uniform(-10/5000,10/5000,size=haze_image.shape)
    #   haze_image = np.maximum(0, haze_image)


      # # NOTE preprocessing for each pair of images
      # # haze_image = self.transform(haze_image)
      # r = Image.fromarray(haze_image[0]).convert('L')
      # g = Image.fromarray(haze_image[1]).convert('L')
      # b = Image.fromarray(haze_image[2]).convert('L')
      # haze_image = Image.merge("RGB",(r,g,b))
      # # haze_image= Image.fromarray(np.uint8(haze_image))
      # haze_image = self.transform(haze_image)
	  #
      # r = Image.fromarray(GT[0]).convert('L')
      # g = Image.fromarray(GT[1]).convert('L')
      # b = Image.fromarray(GT[2]).convert('L')
      # GT = Image.merge("RGB", (r, g, b))
      # # GT = Image.fromarray(GT)
      # GT = self.transform(GT)
    return haze_image, GT
    
    
class new(data.Dataset):
  def __init__(self, root, transform=None, loader=default_loader, seed=None):
    # imgs = make_dataset(root)
    # if len(imgs) == 0:
    #   raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
    #              "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.root = root
    # self.imgs = imgs
    self.transform = transform
    self.loader = loader

    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index):
    # index = np.random.randint(1,self.__len__())
    # index = np.random.randint(self.__len__(), size=1)[0]

    # path = self.imgs[index]
    # img = self.loader(path)
    #img = img.resize((w, h), Image.BILINEAR)



    file_name=self.root+'/'+str(index)+'.h5'
    f=h5py.File(file_name,'r')

    input=f['input'][:]
    input = transform.resize(input, (224,224))
    # trans_map=f['trans'][:]
    # ato_map=f['ato'][:]
    target=f['target'][:]
    input = np.swapaxes(input, 0, 2)
    # trans_map=np.swapaxes(trans_map,0,2)
    # ato_map=np.swapaxes(ato_map,0,2)
    #GT = np.swapaxes(GT, 0, 2)

    input = np.swapaxes(input, 1, 2)
    # trans_map=np.swapaxes(trans_map,1,2)
    # ato_map=np.swapaxes(ato_map,1,2)
    #GT = np.swapaxes(GT, 1, 2)


    # if self.transform is not None:

    #   haze_image = self.transform(haze_image)
	#
	#
    #   GT = self.transform(GT)



      # haze_image = Image.fromarray(haze_image, mode='RGB')
      # GT = Image.fromarray(GT, mode='RGB')
      # # haze_image,GT = tran.ToPILImage(haze_image,GT )
      # haze_image = self.transform(haze_image)
      # GT = self.transform(GT)

	# if self.transform is None:
     #  haze_image=np.swapaxes(haze_image,0,2)
     #  # trans_map=np.swapaxes(trans_map,0,2)
     #  # ato_map=np.swapaxes(ato_map,0,2)
     #  GT=np.swapaxes(GT,0,2)
	#
	# #
	#
    #   haze_image=np.swapaxes(haze_image,1,2)
    #   # trans_map=np.swapaxes(trans_map,1,2)
    #   # ato_map=np.swapaxes(ato_map,1,2)
    #   GT=np.swapaxes(GT,1,2)


    # haze_image=np.swapaxes(haze_image,0,2)
    # # trans_map=np.swapaxes(trans_map,0,2)
    # # ato_map=np.swapaxes(ato_map,0,2)
    # GT=np.swapaxes(GT,0,2)
	#
	#
	#
    # haze_image=np.swapaxes(haze_image,1,2)
    # # trans_map=np.swapaxes(trans_map,1,2)
    # # ato_map=np.swapaxes(ato_map,1,2)
    # GT=np.swapaxes(GT,1,2)
	#



    # if np.random.uniform()>0.5:
    #   haze_image=np.flip(haze_image,2).copy()
    #   GT = np.flip(GT, 2).copy()
    #   trans_map=np.flip(trans_map, 2).copy()
    # if np.random.uniform()>0.5:
    #   angle = np.random.uniform(-10, 10)
    #   haze_image=scipy.ndimage.interpolation.rotate(haze_image, angle)
    #   GT = scipy.ndimage.interpolation.rotate(GT, angle)

    # if np.random.uniform()>0.5:
    #   angle = np.random.uniform(-10, 10)
    #   haze_image=scipy.ndimage.interpolation.rotate(haze_image, angle)
    #   GT = scipy.ndimage.interpolation.rotate(GT, angle)

    # if np.random.uniform()>0.5:
    #   std = np.random.uniform(0.2, 1.2)
    #   haze_image = scipy.ndimage.filters.gaussian_filter(haze_image, std,mode='constant')

    #   haze_image=np.random.uniform(-10/5000,10/5000,size=haze_image.shape)
    #   haze_image = np.maximum(0, haze_image)


      # # NOTE preprocessing for each pair of images
      # # haze_image = self.transform(haze_image)
      # r = Image.fromarray(haze_image[0]).convert('L')
      # g = Image.fromarray(haze_image[1]).convert('L')
      # b = Image.fromarray(haze_image[2]).convert('L')
      # haze_image = Image.merge("RGB",(r,g,b))
      # # haze_image= Image.fromarray(np.uint8(haze_image))
      # haze_image = self.transform(haze_image)
	  #
      # r = Image.fromarray(GT[0]).convert('L')
      # g = Image.fromarray(GT[1]).convert('L')
      # b = Image.fromarray(GT[2]).convert('L')
      # GT = Image.merge("RGB", (r, g, b))
      # # GT = Image.fromarray(GT)
      # GT = self.transform(GT)
    return input, target
    
    
  def __len__(self):
    train_list=glob.glob(self.root+'/*h5')

    # print len(train_list)
    return len(train_list)

    # return len(self.imgs)
