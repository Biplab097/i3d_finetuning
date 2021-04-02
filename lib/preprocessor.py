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


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

def load_video(path, max_frames=0, resize=(224, 224)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames) / 255.0

def loaddata(Video_dir,n_classes):
    files = os.listdir(Video_dir)
    X = []
    labels = []
    
    for i in range(n_classes):
        path = os.path.join(Video_dir, 'c'+str(i),'*.avi')
        files = glob.glob(path)
        print("load data entry point")
      
        for filename in files:
            labels.append(i)
            #print("in load data count->",count)
            X.append(load_video(filename))

    return np.array(X) , np.array(labels)

T = 16

def trim(video):
    start = np.random.randint(0, video.shape[0] - (T+1)) #changed shape[1] to shape[0]
    end = start + T
    print("Start and end ",start,end)
    return np.array(video[start:end, :, :, :])


def train_array(X_train):
    X = []
    for ele in X_train:
#         print("shape",ele.shape)
        ele = trim(ele)
        X.append(ele)
        print("shape",ele.shape)
    print("from train array",len(X))
    
    return np.array(X)
    


    
    
    
