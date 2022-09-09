#-*- coding:utf-8 -*-
import os
import json
import threading
import numpy as np
from PIL import Image

import tensorflow as tf
from keras import losses
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Flatten
from keras.layers.core import Reshape, Masking, Lambda, Permute
from keras.layers.recurrent import GRU, LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard

from imp import reload
import densenet

train_data_num = 40011
val_data_num = 50014 - train_data_num

start_lr = 0.0005 #* 0.4**4
batch_size = 32
epochs = 20

img_h = 32
img_w = 2642 # 最宽的图片 宽度
maxlabellength = 50 # 训练图片最长的字数


def get_session(gpu_fraction=1.0):

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def readfile(filename):
    res = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip())
    dic = {}
    for i in res:
        p = i.split(' ')
        dic[p[0]] = p[1:]
    return dic

class random_uniform_num():
    """
    均匀随机，确保每轮每个只出现一次
    """
    def __init__(self, total):
        self.total = total
        self.range = [i for i in range(total)]
        np.random.shuffle(self.range)
        self.index = 0
    def get(self, batchsize):
        r_n=[]
        if(self.index + batchsize > self.total):
            r_n_1 = self.range[self.index:self.total]
            np.random.shuffle(self.range)
            self.index = (self.index + batchsize) - self.total
            r_n_2 = self.range[0:self.index]
            r_n.extend(r_n_1)
            r_n.extend(r_n_2)
        else:
            r_n = self.range[self.index : self.index + batchsize]
            self.index = self.index + batchsize

        return r_n

def gen(data_file, image_path, nclass, batchsize=128, maxlabellength=10, imagesize=(32, 280)):
    image_label = readfile(data_file)
    _imagefile = [i for i, j in image_label.items()]
    x = np.zeros((batchsize, imagesize[0], imagesize[1], 1), dtype=np.float)
    labels = np.ones([batchsize, maxlabellength]) * (nclass - 1) # 默认最后一个未知的label
    input_length = np.zeros([batchsize, 1])
    label_length = np.zeros([batchsize, 1])

    r_n = random_uniform_num(len(_imagefile))
    _imagefile = np.array(_imagefile)
    while 1:
        max_width = 0
        max_labellength = 0
        shufimagefile = _imagefile[r_n.get(batchsize)]
        for i, j in enumerate(shufimagefile):
            img1 = Image.open(os.path.join(image_path, j)).convert('L')
            img = np.array(img1, 'f')

            max_width = max(max_width, img.shape[1])
            if img.shape[1]<imagesize[1]: # 将图片扩宽到最大，不足的补零
                img = np.hstack([img, np.zeros((imagesize[0], imagesize[1]-img.shape[1]))])
            assert img.shape[1]==imagesize[1]

            img = img / 255.0 - 0.5
            x[i] = np.expand_dims(img, axis=2)
            # print('imag:shape', img.shape)

            str = image_label[j]
            max_labellength = max(max_labellength, len(str))
            if len(str)<maxlabellength: # 将label也扩大到最大，用最后一个无用标签填充
                str.extend([f"{nclass-1}"]*(maxlabellength-len(str)))
            #label_length[i] = len(str)
            assert len(str)==maxlabellength

            #input_length[i] = imagesize[1] // 8
            #labels[i, :len(str)] = [int(k) - 1 for k in str]
            labels[i, :len(str)] = [int(k) for k in str] # 不减1, 码表第一个不是空格

        input_length = np.ones([batchsize, 1]) * max_width // 8  # 固定宽度
        label_length = np.ones([batchsize, 1]) * max_labellength # 固定label数量

        inputs = {'the_input': x[:,:,:max_width],
                'the_labels': labels[:,:max_labellength],
                'input_length': input_length,
                'label_length': label_length,
                }
        outputs = {'ctc': np.zeros([batchsize])}
        yield (inputs, outputs)

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def get_model(img_h, nclass):
    input = Input(shape=(img_h, None, 1), name='the_input')
    y_pred = densenet.dense_cnn(input, nclass)

    basemodel = Model(inputs=input, outputs=y_pred)
    basemodel.summary()

    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam', metrics=['accuracy'])

    return basemodel, model


if __name__ == '__main__':
    if not os.path.exists('./output'):
        os.makedirs('./output')

    char_set = open('../../data/char_code.txt', 'r', encoding='utf-8').readlines()
    #char_set = ''.join([ch.strip('\n') for ch in char_set][1:] + ['卍'])
    char_set = ''.join([ch.strip('\n') for ch in char_set] + ['𠙶']) # 第一个码表不是空格，不跳过；码表中哟有‘卍’，不能用
    nclass = len(char_set)

    K.set_session(get_session())
    reload(densenet)
    basemodel, model = get_model(img_h, nclass)

    modelPath = './output/ocr-densenet-04-35.0070-35.4888-0.0000.weights___'
    if os.path.exists(modelPath):
        print("Loading model weights...", modelPath)
        basemodel.load_weights(modelPath)
        print('done!')

    train_loader = gen('../../data/chardata/train_labels.txt', '../../data/chardata/image', nclass, batchsize=batch_size, maxlabellength=maxlabellength, imagesize=(img_h, img_w))
    test_loader = gen('../../data/chardata/test_labels.txt', '../../data/chardata/image', nclass, batchsize=batch_size, maxlabellength=maxlabellength, imagesize=(img_h, img_w))

    checkpoint = ModelCheckpoint(filepath='./output/ocr-densenet-{epoch:02d}-{loss:.4f}-{val_loss:.4f}-{val_accuracy:.4f}.weights', 
        monitor='val_loss', save_best_only=False, save_weights_only=True)
    lr_schedule = lambda epoch: start_lr * 0.4**epoch
    learning_rate = np.array([lr_schedule(i) for i in range(epochs)])
    changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    earlystop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    #tensorboard = TensorBoard(log_dir='./output/logs', write_graph=True)

    print('-----------Start training-----------')
    model.fit_generator(train_loader,
        steps_per_epoch = train_data_num // batch_size,
        epochs = epochs,
        initial_epoch = 0,
        validation_data = test_loader,
        validation_steps = val_data_num // batch_size,
        callbacks = [checkpoint, earlystop, changelr])
        #callbacks = [checkpoint, earlystop, changelr, tensorboard])


