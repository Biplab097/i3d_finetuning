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
#i3d = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1").signatures['default']

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def predict(sample_video):
  # Add a batch axis to the to the sample video.
    model_input = tf.constant(sample_video, dtype=tf.float32)[tf.newaxis, ...]

    logits = i3d(model_input)['default'][0]
    probabilities = tf.nn.softmax(logits)

    print("Top 5 actions:")
    for i in np.argsort(probabilities)[::-1][:5]:
        print(f" {probabilities[i] * 100:5.2f}%")
    
if __name__=="__main__":
    # For Weizmann----------------------------dataset----------start---------------------------
#     directory = "/tf/mocogan-master/raw_data"
#     n_classes = 9 # weizmann dataset
#     Xin, Yin = loaddata(directory,n_classes)
#     print(Xin.shape,Yin.shape)
#     weizmann_data = open("data/weizmann_data1","wb")
#     pickle.dump(Xin,weizmann_data)
#     weizmann_label = open("data/weizmann_label1","wb")
#     Yin = np_utils.to_categorical(Yin,9)
#     pickle.dump(Yin,weizmann_label)    
#     input1 = open("/tf/i3d_finetuning/data/weizmann_data1",'rb')
#     Xin = pickle.load(input1)
#     input2 = open("/tf/i3d_finetuning/data/weizmann_label1","rb")
#     Yin = pickle.load(input2)
   # weizmann data -----------------------------------end -------------------------------------------------------------
   # multi-class ------------------------- start loading-----------------------------------------

#     30-05-2021
    directory = "/tf/i3d_finetuning/multi_action_dataset/train"
    n_classes = 6
    Xin, Yin = loaddata(directory,n_classes)
    print(Xin.shape,Yin.shape)

#     30-05-2021
   


#     multiclass_data = open("data/multiclass_data","wb")
#     pickle.dump(Xin,multiclass_data)
#     multiclass_label = open("data/multiclass_label","wb")
#     pickle.dump(Yin,multiclass_label)

    Yin = np_utils.to_categorical(Yin,n_classes)
    print(Xin[4].shape,Yin[1].shape)
    vid = Xin[4]
    print(vid.shape)
    #vid = np.expand_dims(vid,1)
    to_gif(vid)
    print(Xin.shape,Yin.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(Xin, Yin, test_size=0.25, random_state=4)
    print("coming back to main Split done size is:{0} and {1} ".format(X_train.shape,Y_train.shape))
    
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
    
    
   #----------------------------commented date 19-04-2021 ---------------------
#     input1 = open("/tf/i3d_finetuning/data/wizmann_data_augmented",'rb')
#     train_X = pickle.load(input1)
#     input2 = open("/tf/i3d_finetuning/data/wizmann_label_augmented","rb")
#     train_Y = pickle.load(input2)
#     print(type(train_X),type(train_Y))
#     train_X, X_test, train_Y, Y_test = train_test_split(train_X, train_Y, test_size=0.25, random_state=4)
#     print(train_X.shape,train_Y.shape)
#     print(X_test.shape,Y_test.shape)
  # -------------------------------- 19-04-2021 ----------------------------

# ---------------------------------commented on 19-04-2021 ----------------------------------
#     model = train_i3d(train_X,train_Y,X_test,Y_test)
#     print("From Main.py",type(model))
# #     model.summary()
#     tf.keras.backend.set_floatx('float32')
#     print(type(train_X))
#     print(train_X[0].shape)
#     X_train = train_array(train_X)
#     print("again in main",X_train.shape)
#     print("from main test shape",train_Y.shape)
#     X_test = train_array(X_test)
#     print("X_test shape",X_test.shape)
#     print("Y_test shape",Y_test.shape)
# #     predict(X_train[0])
#     model.summary()
    
#     history = model.fit(X_train, train_Y, validation_data=(X_test, Y_test), batch_size=32, epochs=10, verbose=1,shuffle=True)
#     model.save_weights("/tf/i3d_finetuning/models/weizmann_i3d_lr_0.002_500_epochs_batch32.h5")
    
#     loss, acc = model.evaluate(X_test, Y_test, verbose=0)
#     print('Test loss:', loss)
#     print('Test accuracy:', acc)
    
#     plt.plot(history.history['accuracy'])
    
#     plt.plot(history.history['val_accuracy'])
#     plt.title('model accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.show()
#     plt.savefig("/tf/i3d_finetuning/models/accuracy_500.png")
#     # summarize history for loss
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper right')
#     plt.show()
#     plt.savefig("/tf/i3d_finetuning/models/loss_500.png")
# ------------------------------------------------------------------------19-04-2021 ---------------------------------------



#----------------------------------- written on 19-04-2021 ----------------------------------

#   30-05-2021
    model = train_i3d(X_train,Y_train,X_test,Y_test)
    print("From Main.py",type(model))
    model.summary()
    tf.keras.backend.set_floatx('float16')
    #print(type(train_X))
    print(X_train[0].shape)
    print()
    X_train = train_array(Xin)
    print()
    print("again in main before split",X_train.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Yin, test_size=0.25, random_state=4)
    
#     X_test = train_array(X_test)
    print("X_train shape",X_train.shape)
    print("Y_train shape",Y_train.shape)
    print("X_test shape",X_test.shape)
    print("Y_test shape",Y_test.shape)
#     X_train = X_train.astype('float16')
#     X_test = X_test.astype('float16')
#     Y_train = Y_train.astype('float16')
#     Y_test = Y_test.astype('float16')
    print(type(X_test[0][0][0][0][0]))
# #     predict(X_train[0])
#     model.summary()
#   30-05-2021
    
    
    
#     X_train = tf.convert_to_tensor(X_train,dtype=tf.float16)
#     Y_train = tf.convert_to_tensor(Y_train,dtype=tf.float16)
#     X_test = tf.convert_to_tensor(X_test,dtype=tf.float16)
#     Y_test = tf.convert_to_tensor(Y_test,dtype=tf.float16)
    
#     X_train = np.asarray(X_train).astype(np.ndarray)
#     Y_train = np.asarray(Y_train).astype(np.ndarray)
#     X_test = np.asarray(X_test).astype(np.ndarray)
#     Y_test = np.asarray(Y_test).astype(np.ndarray)                    
    
#     print("type ",type(X_train),X_train.shape)
#     print("type ",type(X_test),X_test.shape)
#     print("type ",type(Y_train),Y_train.shape)
#     print("type ",type(Y_test),Y_test.shape)


# 30-05-2021
    
#     X_train =  np.expand_dims(X_train,axis = 0)    # (570,)
#     #X_train = X_train.expand_dims()
#     print("expanded shape",X_train.shape)
    
#     X_train = tf.convert_to_tensor(X_train,dtype=tf.float16)
#     Y_train = tf.convert_to_tensor(Y_train,dtype=tf.float16)
#     X_test = tf.convert_to_tensor(X_test,dtype=tf.float16)
#     Y_test = tf.convert_to_tensor(Y_test,dtype=tf.float16)
    
    #tf.keras.backend.set_floatx('float16')
    
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=8, epochs=50, verbose=1,shuffle=True)
    
    model.save_weights("/tf/i3d_finetuning/models/multiclass6_STA_npdi_i3d_lr_0.002_50_epochs_batch32.h5")
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
    plt.savefig("/tf/i3d_finetuning/models/multiclass6_STA_accuracy_npdi_50.png")
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    plt.savefig("/tf/i3d_finetuning/models/multiclass6_STA_loss_npdi_50.png")
    
    
# 07-06-2021
    
    
    
    
    
