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
import matplotlib.pyplot as plt
from keras.optimizers import SGD # optimizer
from keras.regularizers import l2
from keras.layers import Dropout
from keras.layers import Conv3D
from keras.layers import Conv3DTranspose

def train_i3d(X_train,Y_train,X_test,Y_test):
#     os.environ["CUDA_VISIBLE_DEVICES"]="1"
    #tf.keras.backend.set_floatx('float16')
    i3d = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"
#     train_data = tf.data.Dataset.from_tensor_slices((X_train,Y_train))
#     test_data = tf.data.Dataset.from_tensor_slices((X_test,Y_test))
#     path = '/tf/i3d_finetuning/i3d-kinetics-400_1'
    hub_layer = hub.KerasLayer(i3d,input_shape=[None,224,224,3],trainable=False)
    #hub_layer(X_train[:2])
    model = tf.keras.Sequential()
    print("Here ------> ",type(model))
    
    model.add(Conv3D(8,kernel_size=(2,2,16),strides=(1, 1, 1),input_shape=(None,224,224,3),padding="same"))
                            # o/p 224*224*3
    model.add(Conv3DTranspose(8,kernel_size=(2,2,16),strides=(1, 1, 1),input_shape=(None,224,224,3),padding="same"))
    model.add(tf.keras.layers.Dense(32,activation='relu'))
    model.add(Conv3D(16,kernel_size=(2,2,16),strides=(1, 1, 1),input_shape=(None,224,224,3),padding="same"))
    model.add(tf.keras.layers.Dense(3,activation='sigmoid'))
    
    
    model.add(hub_layer)
    
    # Adding----------------------------------STA------------------------------
    
#     model.add(Conv3D(8,kernel_size=(2,2,16),strides=(1, 1, 1),input_shape=(224,224,3),padding="same"))
#                             # o/p 224*224*3
#     model.add(Conv3DTranspose(8,kernel_size=(2,2,16),strides=(1, 1, 1),input_shape=(224,224,3),padding="same"))
#     model.add(tf.keras.layers.Dense(512,activation='relu'))
#     model.add(Conv3D(16,kernel_size=(2,2,16),strides=(1, 1, 1),input_shape=(None,224,224,3),padding="same"))
#     model.add(tf.keras.layers.Dense(256,activation='sigmoid'))
    
    
    
    #Added ----------------------------------- STA -------------------------------
    
    #model.add(tf.keras.layers.Dense(256,activation='relu')) #20-05-21
    
    #model.add(Dropout(0.25))
    # added dropout on 05-04-2021
    # model.add(tf.keras.layers.Dense(9, activation='softmax', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
    model.add(tf.keras.layers.Dense(6, activation='softmax')) # change as per no of classes 20-5-21
    # added kernel_regularizer on date 05-04-2021
#     model.summary()
    opt = SGD(learning_rate=(0.002)) # 2*10^-3
    model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy']) #20-5-21
    #print("here1------>",type(X_train))
#     X_train = tf.convert_to_tensor(X_train,dtype=tf.float32)
#     Y_train = tf.convert_to_tensor(Y_train,dtype=tf.float32)
#     X_test = tf.convert_to_tensor(X_test,dtype=tf.float32)
#     Y_test = tf.convert_to_tensor(Y_test,dtype=tf.float32)
    [print(i.shape, i.dtype) for i in model.inputs]
    [print(o.shape, o.dtype) for o in model.outputs]
    [print(l.name, l.input_shape, l.dtype) for l in model.layers]
#     history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=64, epochs=10, verbose=1,shuffle=True)
    
#     history = model.fit(train_data,epochs=10)
    return model
  
    # 07-06-2021


    





    
    
    
   
