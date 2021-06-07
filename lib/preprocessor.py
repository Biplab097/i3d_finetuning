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

os.environ["CUDA_VISIBLE_DEVICES"]="1"
T = 16 # depth of frames


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

def load_video(path,i,max_frames=0,resize=(224, 224)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            if i==0 or i==10:
                nframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                #frames = [x * nframe / T for x in range(T)] // 4 frame - 64
                #print("frames ",nframe)  # 3500
                lis = []
#                 diff = random.randint(0,nframe-65)  # 440
#                 for i in range(16):
#                     lis.append(diff) # [ 440,444,448,452,456....]
#                     diff+=4 
                
                
                for i in range(16):
                    idx = random.randint(0,nframe) #4000 ->  (2,6,10,14....AP) 16s frames.
                    lis.append(idx)
                lis.sort() # [ 2,56,789,1004,2003 ]
                #print("lis ",lis)
#                 start = np.random.randint(0, nframe - (T+1))
#                 end = start + T
#                 print(start,end)
#                 #frame = trim1(frame) 

                for i in range(16):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, lis[i])
                    ret, frame = cap.read()
                    #print("frame shape before crop",frame.shape)
                    if frame is not None:
                        frame = crop_center_square(frame)
                        frame = cv2.resize(frame, resize)
                        frame = frame[:, :, [2, 1, 0]]
                        #print("shape of frame ",frame.shape)
                        frames.append(frame)
#                 for i in range(start,end):
#                     cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
#                     ret, frame = cap.read()
#                     frame = crop_center_square(frame)
#                     frame = cv2.resize(frame, resize)
#                     frame = frame[:, :, [2, 1, 0]]
#                     print("shape of frame ",frame.shape)
#                     frames.append(frame)
                    
            else:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = crop_center_square(frame)
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]
            
                #print("appending frame ",frame.shape)
                frames.append(frame)

            if len(frames) == max_frames:
                if(i==0):
                    print("frame size 0") 
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
        print("load data entry point ",i+1)
        no_of_files = 0
        for filename in files:
            labels.append(i)
            #print("in load data count->",count)
            X.append(load_video(filename,i))
            if(i==0 or i==10):
                no_of_files+=1
                print("no of files added ",no_of_files)
    print("loading done")
    return np.array(X) , np.array(labels)

def trim1(video):
    start = np.random.randint(0, video.shape[0] - (T+1)) #changed shape[1] to shape[0]
    end = start + T
    #print("Start and end ",start,end)
    return np.array(video[start:end, :, :])

def trim(video,i):
    #print("video.shape[0] ",video.shape[0])
    try:
        start = np.random.randint(0, video.shape[0] - (T+1)) #changed shape[1] to shape[0]
        end = start + T
#         print("======")
#         print("=============")
#         print("==========================")
        #print("Start and end ",start,end)
        return np.array(video[start:end, :, :, :])
    except:
        print("video.shape[0] ",video.shape[0],i)
        return np.random.rand(16,224,224,3)
    

T = 16
def train_array(X_train):
    X = []
    i = 0
    for ele in X_train:
#         print("shape",ele.shape)
        i+=1
        if ele is not None:
            ele = trim(ele,i)
        else:
            ele =  np.random.rand(16,224,224,3)
        X.append(ele)
        #print("shape",ele.shape)
    print("from train array",len(X))
    
    return np.array(X)
    


    
    
    
