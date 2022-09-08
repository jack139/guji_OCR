# 生成densenet训练数据：文字条图片

import os
import sys
import time
import json
from glob import glob
from tqdm import tqdm

import cv2
from math import *
import numpy as np
from shapely.geometry import Polygon


label_json_file = '../../data/rotated/label.json'
image_dir = '../../data/rotated/image'
output_dir = '../../data/chardata'

def pickTopLeft(poly):
    idx = np.argsort(poly[:, 0])
    if poly[idx[0], 1] < poly[idx[1], 1]:
        s = idx[0]
    else:
        s = idx[1]

    return poly[(s, (s + 1) % 4, (s + 2) % 4, (s + 3) % 4), :]

def orderConvex2(p):
    points = Polygon(p).minimum_rotated_rectangle
    points = np.array(points.exterior.coords)[:4]
    points = points[::-1]
    points = pickTopLeft(points)
    points = np.array(points).reshape([4, 2])
    p2 = np.zeros([4,2])
    p2[0] = points[0]
    p2[1] = points[3]
    p2[2] = points[2]
    p2[3] = points[1]
    return p2

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



def charRec(img, rec, adjust=False, save_path=None):
    results = {}
    xDim, yDim = img.shape[1], img.shape[0]
     
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
    ylength = int((rec[5] - rec[1]) * 0.1)

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

    if partImg.shape[0] < 1 or partImg.shape[1] < 1:  # 过滤异常图片
        return False

    # 调整高度为 32
    img_size = partImg.shape
    im_scale = 32.0 / img_size[0]
    new_h = 32
    new_w = int(img_size[1] * im_scale)

    re_im = cv2.resize(partImg, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 保存输入图片，用于调试
    if save_path:
        cv2.imwrite(save_path, re_im)
    
    return True


if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(os.path.join(output_dir, "image")):
    os.makedirs(os.path.join(output_dir, "image"))

with open(label_json_file, 'r') as f:
    labels = json.load(f)

poly_labels = []
max_length = 0

for f in tqdm(glob(image_dir+'/*.jpg')):
    fn = os.path.split(f)[-1] # 文件名
    bn, _ = os.path.splitext(fn)

    #print(fn)

    img = cv2.imread(f)

    for index, x in enumerate(labels[fn]):
        # 计算 最小框的面积
        xy = [item for item in map(float, x['points'])]
        poly = np.array(xy).reshape([len(xy)//2, 2])
        poly = orderConvex2(poly)
        poly_rec = poly.astype(np.int32).reshape((8,)).tolist()

        poly_name = f'{bn}_{index}.jpg'

        if charRec(img, poly_rec, adjust=False, save_path=os.path.join(output_dir, "image", poly_name)):
            poly_labels.append( poly_name + ' ' + ' '.join([str(a) for a in x['code']]) )
        else:
            print("error cut:", poly_rec)

            # 画框
            cv2.polylines(img, [poly.astype(np.int32).reshape((-1, 1, 2))], True,color=(0, 255, 0), thickness=2)

            # 保存画框的图片
            cv2.imwrite(os.path.join(output_dir, 'box_'+fn), img)

        max_length = max(max_length, len(x['code']))

    #break # just one round for test


# 保存 labels
with open(os.path.join(output_dir, 'all_labels.txt'), 'w') as output_data:
    for s in poly_labels:
        output_data.write(s.strip() + '\n')

print("\ntotal=", len(poly_labels))
print("max_length=", max_length)