from absl import logging

import tensorflow as tf
import tensorflow_hub as hub


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

from IPython import display

from urllib import request  # requires python3

from lib.preprocessor import loaddata
from lib.to_gif import to_gif

if __name__=="__main__":
    directory = "/tf/mocogan-master/raw_data"
    n_classes = 9 # weizmann dataset
#     Xin, Yin = loaddata(directory,n_classes)
#     print(Xin[45].shape,Yin.shape)
#     weizmann_data = open("data/weizmann_data1","wb")
#     pickle.dump(Xin,weizmann_data)
#     weizmann_label = open("data/weizmann_label1","wb")
#     Yin = np_utils.to_categorical(Yin,9)
#     pickle.dump(Yin,weizmann_label)
    input1 = open("/tf/i3d_finetuning/data/weizmann_data",'rb')
    Xin = pickle.load(input1)
    input2 = open("/tf/i3d_finetuning/data/weizmann_label","rb")
    Yin = pickle.load(input2)
    print(Xin[46].shape,Yin.shape)
    vid = Xin[77]
    print(vid.shape)
    #vid = np.expand_dims(vid,1)
    #print(vid.shape)
    to_gif(vid)
    
    
    