from absl import logging

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed

logging.set_verbosity(logging.ERROR)

# Some modules to help with reading the UCF101 dataset.
import random
import re
import os
import tempfile
import ssl
import cv2
import glob
import numpy as np
import pickle
from keras.utils import np_utils
import skvideo.io
# Some modules to display an animation using imageio.
import imageio
from IPython import display
_VIDEO_LIST = None
from urllib import request  # requires python3
#for augmentation
from vidaug import augmentors as va
from PIL import Image, ImageSequence
import vidaug.augmentors as va
import random

sometimes = lambda aug: va.Sometimes(0.5, aug) 
seq = va.Sequential([ 
    sometimes(va.RandomCrop(size=(224, 224))),
    sometimes(va.RandomRotate(degrees=10)),
   sometimes(va.VerticalFlip()),
    sometimes(va.HorizontalFlip()),
    sometimes(va.GaussianBlur(1.5))
])

no_of_samples = 30
X_train_aug = []
Y_train_aug = []

def augmentation(X_train,Y_train):
    for i in range(no_of_samples):
        idx = random.randint(0,59)
        print("index is ",idx)
        vid = X_train[idx]
        print("shape of vid",vid.shape)
        #vid = np.expand_dims(vid,3)
        #print("shape of vid after expand",vid.shape)
        video_aug = np.array(seq(vid))
        #video_aug = video_aug.squeeze()
        print("video aug shape",video_aug.shape)
        #X_train_aug = np.append(X_train_aug, np.array(video_aug), axis=0)
        X_train_aug.append(video_aug)
        Y_train_aug.append(Y_train[idx])
        #Y_train_aug = np.append(Y_train_aug, np.array(Y_train[idx]),axis=0)
        print(video_aug[0].shape)
    print("len of X_train_aug",len(X_train_aug))
    
    return np.array(X_train_aug),np.array(Y_train_aug)
    
    


    
    
    