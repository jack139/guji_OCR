# -*- coding: utf-8 -*-

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
#os.environ["PER_PROCESS_GPU_MEMORY_FRACTION"] = "0.4"

import sys, json
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob

from ctpn import detect
from densenet import ocr
#from api.utils import check_rotated



image_dir = 'data/test/testa'
result_output = "data/test/results"

if not os.path.exists(result_output):
    os.makedirs(result_output)


model = None

def identify(filepath):
    global model
    # 读入图片

    #im = cv2.imread(filepath)[:, :, ::-1] # 去色
    im = cv2.imread(filepath)

    #print('>>>> ', filepath)

    # 逆时针旋转90度
    im = cv2.rotate(im, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)


    # 定位文字位置 -- ctpn
    img2, boxes = detect.process_text(im, debug=True)
    #cv2.imwrite('data/tmp/img.jpg',img2)

    boxes2 = boxes.copy()

    # 打印框坐标
    #for i in boxes:
    #    print(i)

    #print(im.shape)

    h,w,_ = im.shape

    # OCR 文字识别
    rr = ocr.model(im, boxes, False)
    result = []
    for key in sorted(rr.keys()):
        x1 = (rr[key][0].tolist())[:8]
        # 转换坐标，逆旋90度
        x2=[]
        for i in range(4):
            x2.append(h-x1[i*2+1])
            x2.append(x1[i*2])

        result.append({
            'pos'  : x2,
            'text' : rr[key][1]
        })

    return result


if __name__ == '__main__':

    for f in tqdm(glob(image_dir+'/*.jpg')):
        fn = os.path.split(f)[-1] # 文件名
        bfn, _ = os.path.splitext(fn)

        r1 = identify(f)

        r2 = []
        for x in r1:
            r2.append(','.join([str(i) for i in x['pos']] + [x['text']]))

        with open(os.path.join(result_output, bfn+'.csv'), 'w') as f:
            f.write('\n'.join(r2))

