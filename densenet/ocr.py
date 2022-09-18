#-*- coding:utf-8 -*-
import os
import sys
import time

import cv2
from math import *
import numpy as np
from PIL import Image

#from .model import predict as keras_densenet

from . import model as keras_densenet

def sort_box(box):
    """ 
    对box进行排序
    """
    #box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
    box = sorted(box, key=lambda x: [x[1], x[3], x[5], x[7]])
    return box

def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) // 2
    matRotation[1, 2] += (heightNew - height) // 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    ydim, xdim = imgRotation.shape[:2]
    imgOut = imgRotation[max(1, int(pt1[1])) : min(ydim - 1, int(pt3[1])), max(1, int(pt1[0])) : min(xdim - 1, int(pt3[0]))]

    return imgOut

def charRec(img, text_recs, adjust=False, save_path=None):
    """
    加载OCR模型，进行字符识别
    """
    results = {}
    xDim, yDim = img.shape[1], img.shape[0]
     
    for index, rec in enumerate(text_recs):
        # 调整坐标顺序
        #      左上 (0, 1) 右上 (2, 3) 右下 (4, 5) 左下 (6, 7)
        # -->  左上 (0, 1) 右上 (2, 3) 左下 (4, 5) 右下 (6, 7)

        x4 = rec[6]
        y4 = rec[7]
        rec[6] = rec[4]
        rec[7] = rec[5]
        rec[4] = x4
        rec[5] = y4

        xlength = int((rec[2] - rec[0]) * 0.1)
        ylength = int((rec[5] - rec[1]) * 0.2)

        if adjust:
            pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
            pt2 = (min(rec[2] + xlength, xDim - 2), max(1, rec[3] - ylength))
            pt3 = (min(rec[6] + xlength, xDim - 2), min(yDim - 2, rec[7] + ylength))
            pt4 = (rec[4], rec[5])
        else:
            pt1 = (max(1, rec[0]), max(1, rec[1]))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6], xDim - 2), min(yDim - 2, rec[7]))
            pt4 = (rec[4], rec[5])
         
        degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))  # 图像倾斜角度

        partImg = dumpRotateImage(img, degree, pt1, pt2, pt3, pt4)
    
        if partImg.shape[0] < 1 or partImg.shape[1] < 1 or partImg.shape[0] > partImg.shape[1]:  # 过滤异常图片
            continue

        # 保存输入图片，用于调试
        if save_path:
            cv2.imwrite(os.path.join(save_path, 'part_%d.jpg'%index), partImg)

        # 识别
        image = Image.fromarray(partImg).convert('L')
        text = keras_densenet.predict(image)

        if len(text) > 0:
            results[index] = [rec]
            results[index].append(text)  # 识别文字
    
    return results

def model(img, text_recs, adjust=False, save_path=None):
    """
    @img: 图片
    @adjust: 是否调整文字识别结果
    """
    start = time.time()

    image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    text_recs = sort_box(text_recs)

    result = charRec(image, text_recs, adjust, save_path=save_path)

    #print("ocr ====> {:.2f}s".format(time.time() - start))

    return result

