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

    im = cv2.imread(filepath)[:, :, ::-1] # 去色
    #im = cv2.imread(filepath)
    max_width = max(im.shape)
    if max_width>1500: # 图片最大宽度为 1500
        ratio = 1500/max_width
        im = cv2.resize(im, (round(im.shape[1]*ratio), round(im.shape[0]*ratio)))

    print('>>>> ', filepath)

    # 是否需要旋转
    #if check_rotated.need_rotated(im):
    #    print("rotating 270")
    #    im = check_rotated.rotate_bound(im, 270)

    # 定位文字位置 -- ctpn
    img2, boxes = detect.process_text(im, debug=True)
    cv2.imwrite('data/outputs/img.jpg',img2)

    boxes2 = boxes.copy()

    # 打印框坐标
    for i in boxes:
        print(i)

    print(im.shape)


    # OCR 文字识别
    rr = ocr.model(im, boxes, True)
    result = []
    for key in sorted(rr.keys()):
        result.append({
            'pos'  : (rr[key][0].tolist())[:8],
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

