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
from sklearn.model_selection import train_test_split
from IPython import display

from urllib import request  # requires python3

from lib.preprocessor import loaddata
from lib.to_gif import to_gif
from lib.augmentation import augmentation
from lib.pickle_data import pickle_data
from train.train import train_i3d


if __name__=="__main__":
#     directory = "/tf/mocogan-master/raw_data"
#     n_classes = 9 # weizmann dataset
#     Xin, Yin = loaddata(directory,n_classes)
#     print(Xin.shape,Yin.shape)
#     weizmann_data = open("data/weizmann_data1","wb")
#     pickle.dump(Xin,weizmann_data)
#     weizmann_label = open("data/weizmann_label1","wb")
#     Yin = np_utils.to_categorical(Yin,9)
#     pickle.dump(Yin,weizmann_label)
    input1 = open("/tf/i3d_finetuning/data/weizmann_data1",'rb')
    Xin = pickle.load(input1)
    input2 = open("/tf/i3d_finetuning/data/weizmann_label1","rb")
    Yin = pickle.load(input2)
#     print(Xin[46].shape,Yin.shape)
#     vid = Xin[77]
#     print(vid.shape)
    #vid = np.expand_dims(vid,1)
    #print(vid.shape)
    #to_gif(vid)
    X_train, X_test, Y_train, Y_test = train_test_split(Xin, Yin, test_size=0.25, random_state=4)
#     X_train_augmented , Y_train_augmented = augmentation(X_train,Y_train) # data augmentation
    #print("Augmented data shapes: ",type(X_train_augmented),Y_train_augmented.shape)
#     vid = X_train_augmented[22]
#     to_gif(vid) # checking generated augmented video
    #concatenating augmented data
#     train_X = np.concatenate((X_train, X_train_augmented), axis=0)
#     train_y = np.concatenate((Y_train, Y_train_augmented), axis=0)
#     print("Augmented data shape",train_X.shape)
#     print("Augmented label shape",train_y.shape)
    #pickle_data(train_X,train_y)
    input1 = open("/tf/i3d_finetuning/data/wizmann_data_augmented",'rb')
    train_X = pickle.load(input1)
    input2 = open("/tf/i3d_finetuning/data/wizmann_label_augmented","rb")
    train_Y = pickle.load(input2)
    #print(type(train_X),type(train_Y))
    model = train_i3d(train_X,train_Y,X_test,Y_test)
    print(type(model))
    
    
    
    