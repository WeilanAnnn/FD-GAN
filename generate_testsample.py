import numpy  as np
import os
import math
np.seterr(divide='ignore', invalid='ignore')
import random
from PIL import  Image
import skimage
from skimage import io
import random
import pickle
import random
import sys
import cv2
import h5py
from PIL import Image
from pylab import *

gt_path = "your image folder" #if you want to get training data, please put clean images here.
gts = os.listdir(gt_path)
gts.sort()

haze_path = "your image folder" #if you want to get training data, please put corresponding hazy images here.
hazes = os.listdir(haze_path)
hazes.sort()

data=zip(gts,hazes)
data=np.array(list(data))
i= 0
for gt,haze in data:
 
    gt_image = np.float32(io.imread(gt_path + gt))/255.0

    haze_image = np.float32(io.imread(haze_path + haze))/255.0
   
    f = h5py.File("your dataroot"+str(i)+'.h5', 'w')
    f.create_dataset('gt', data=gt_image)
   
    f.create_dataset('haze', data=haze_image)

    i=i+1

    print(i)
print('end')
