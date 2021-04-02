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
import matplotlib.pyplot as plt
# Some modules to display an animation using imageio.
from sklearn.model_selection import train_test_split
from IPython import display

from urllib import request  # requires python3

from lib.preprocessor import loaddata
from lib.preprocessor import train_array
from lib.to_gif import to_gif
from lib.augmentation import augmentation
from lib.pickle_data import pickle_data
from train.train import train_i3d


import tensorflow_hub as hub
i3d = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1").signatures['default']

def predict(sample_video):
  # Add a batch axis to the to the sample video.
    model_input = tf.constant(sample_video, dtype=tf.float32)[tf.newaxis, ...]

    logits = i3d(model_input)['default'][0]
    probabilities = tf.nn.softmax(logits)

    print("Top 5 actions:")
    for i in np.argsort(probabilities)[::-1][:5]:
        print(f" {probabilities[i] * 100:5.2f}%")
    
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
    print("From Main.py",type(model))
#     model.summary()
    tf.keras.backend.set_floatx('float32')
    print(type(train_X))
    print(train_X[0].shape)
    X_train = train_array(train_X)
    print("again in main",X_train.shape)
    print("from main test shape",train_Y.shape)
    X_test = train_array(X_test)
    print("X_test shape",X_test.shape)
    print("Y_test shape",Y_test.shape)
#     predict(X_train[0])
    model.summary()
    history = model.fit(X_train, train_Y, validation_data=(X_test, Y_test), batch_size=32, epochs=550, verbose=1,shuffle=True)
    model.save_weights("/tf/i3d_finetuning/models/weizmann_i3d_lr_0.002_550_epochs_batch32.h5")
    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', acc)
    
    plt.plot(history.history['accuracy'])
    
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig("/tf/i3d_finetuning/models/accuracy_550.png")
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig("/tf/i3d_finetuning/models/loss_550.png")
    
    
    
    
    
    
