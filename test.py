# -*- coding: utf-8 -*-

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
#os.environ["PER_PROCESS_GPU_MEMORY_FRACTION"] = "0.4"

# OCR识别测试  ctpn --> CRNN
# python3 -m test.test4
#

import sys, json
import cv2
import numpy as np

#from keras import backend as K
#import tensorflow as tf

from ctpn import detect
from densenet import ocr
#from api.utils import check_rotated

model = None



def identify(filepath):
    global model
    # 读入图片

    #im = cv2.imread(filepath)[:, :, ::-1] # 去色
    im = cv2.imread(filepath)

    print('>>>> ', filepath)

    # 逆时针旋转90度
    im = cv2.rotate(im, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)


    # 定位文字位置 -- ctpn
    img2, boxes = detect.process_text(im, debug=True)
    cv2.imwrite('data/tmp/img.jpg',img2)

    boxes2 = boxes.copy()

    # 打印框坐标
    for i in boxes:
        print(i)

    print(im.shape)

    h,w,_ = im.shape

    # OCR 文字识别
    rr = ocr.model(im, boxes, False, save_path='data/tmp')
    result = []
    for key in sorted(rr.keys()):
        x1 = (rr[key][0].tolist())[:8]
        # 转换坐标，逆旋90度
        x2=[]
        for i in range(4):
            x2.append(h-x1[i*2+1])
            x2.append(x1[i*2])
        x2 = x2[4:6] + x2[:4] + x2[6:]

        result.append({
            'pos'  : x2,
            'text' : rr[key][1]
        })

    return result


if __name__ == '__main__':
    if len(sys.argv)<2:
        print("usage: python3 %s <image_path>" % sys.argv[0])
        sys.exit(2)

    r1 = identify(sys.argv[1])

    print(r1)

    json.dumps(r1)

