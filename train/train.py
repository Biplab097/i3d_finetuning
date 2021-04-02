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

def train_i3d(X_train,Y_train,X_test,Y_test):
    tf.keras.backend.set_floatx('float32')
    i3d = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"
#     train_data = tf.data.Dataset.from_tensor_slices((X_train,Y_train))
#     test_data = tf.data.Dataset.from_tensor_slices((X_test,Y_test))
    hub_layer = hub.KerasLayer(i3d,input_shape=[None,224,224,3],trainable=False)
    #hub_layer(X_train[:2])
    model = tf.keras.Sequential()
    print("Here ------> ",type(model))
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(256,activation='relu'))
    model.add(tf.keras.layers.Dense(9, activation='softmax'))
#     model.summary()
    opt = SGD(learning_rate=(0.002)) # 2*10^-3
    model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])
    print("here1------>",type(X_train))
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
    # summarize history for accuracy
#     plt.plot(history.history['accuracy'])
#     plt.plot(history.history['val_accuracy'])
#     plt.title('model accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.show()
#     # summarize history for loss
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.show()

    





    
    
    
    
