#-*- coding:utf-8 -*-
import os
import numpy as np
from imp import reload
from PIL import Image, ImageOps

from keras.layers import Input
from keras.models import Model
# import keras.backend as K

from . import keys
from train import densenet

DENSENET_WEIGHTS = 'data/checkpoints_densenet/ocr-guji-01-0.9318-0.0749-0.9848.weights' 
GPU_MEMORY_DENSENET = 0.00001
GPU_RUN_DENSENET = False

reload(densenet)

import tensorflow as tf


characters = keys.alphabet[:]
#characters = characters[1:] + u'卍'
characters = characters + u'𠙶'
nclass = len(characters)

# GPU内存控制
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_MEMORY_DENSENET)

# 是否强制使用 CPU
if GPU_RUN_DENSENET:
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
else:
    config = tf.ConfigProto(device_count = {'CPU' : 1, 'GPU' : 0}, gpu_options=gpu_options)

# 建立默认session
graph = tf.Graph()  # 解决多线程不同模型时，keras或tensorflow冲突的问题
session = tf.Session(graph=graph, config=config)
with graph.as_default():
    with session.as_default():

        input = Input(shape=(32, None, 1), name='the_input')
        y_pred= densenet.dense_cnn(input, nclass)
        basemodel = Model(inputs=input, outputs=y_pred)

        modelPath = os.path.join(os.getcwd(), DENSENET_WEIGHTS)
        if os.path.exists(modelPath):
            basemodel.load_weights(modelPath)
            print('densenet load_weights: ', modelPath)

        # https://stackoverflow.com/questions/40850089/is-keras-thread-safe
        basemodel._make_predict_function() # have to initialize before threading


def decode(pred):
    char_list = []
    pred_text = pred.argmax(axis=2)[0]
    for i in range(len(pred_text)):
        if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
            char_list.append(characters[pred_text[i]])
    return u''.join(char_list)

def predict(img):
    width, height = img.size[0], img.size[1]
    scale = height * 1.0 / 32
    width = int(width / scale)
    
    img = img.resize([width, 32], Image.ANTIALIAS)
   
    '''
    img_array = np.array(img.convert('1'))
    boundary_array = np.concatenate((img_array[0, :], img_array[:, width - 1], img_array[31, :], img_array[:, 0]), axis=0)
    if np.median(boundary_array) == 0:  # 将黑底白字转换为白底黑字
        img = ImageOps.invert(img)
    '''

    img = np.array(img).astype(np.float32) / 255.0 - 0.5
    
    X = img.reshape([1, 32, width, 1])
    
    with graph.as_default(): # 解决多线程不同模型时，keras或tensorflow冲突的问题
        with session.as_default():
            y_pred = basemodel.predict(X)
            y_pred = y_pred[:, :, :]

    # out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0][0])[:, :]
    # out = u''.join([characters[x] for x in out[0]])
    out = decode(y_pred)

    return out
