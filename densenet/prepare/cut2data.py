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


#label_json_file = '../../data/rotated1/label.json'
#image_dir = '../../data/rotated1/image'
#output_dir = '../../data/chardata1'

label_json_file = '../../data/rotated/label.json'
image_dir = '../../data/test/1'
output_dir = '../../data/test/2'

def pickTopLeft(poly):
    idx = np.argsort(poly[:, 0])
    if poly[idx[0], 1] < poly[idx[1], 1]:
        s = idx[0]
    else:
        s = idx[1]

    return poly[(s, (s + 1) % 4, (s + 2) % 4, (s + 3) % 4), :]

def orderConvex(box):
    assert len(box)==8
    points = np.array(box, np.int32)
    points = points.reshape((4,2))
    points = points[::-1]
    points = pickTopLeft(points)
    points = np.array(points).reshape([4, 2])
    p2 = np.zeros([4,2])
    p2[0] = points[0]
    p2[1] = points[3]
    p2[2] = points[2]
    p2[3] = points[1]
    return p2


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
        return None

    # 调整高度为 32
    img_size = partImg.shape
    im_scale = 32.0 / img_size[0]
    new_h = 32
    new_w = int(img_size[1] * im_scale)

    if new_w < 1 or new_h < 1:  # 过滤异常图片
        return None

    re_im = cv2.resize(partImg, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 保存输入图片，用于调试
    if save_path:
        cv2.imwrite(save_path, re_im)
    
    return re_im


def get_boxes_32p(x):
    assert len(x['points'])==32
    p = x['points']
    p2 = np.array(p).reshape([16,2])

    # 找出最小的两个，
    min_x1, min_x2 = 1e+6, 1e+6
    min_x1_idx, min_x2_idx = 0, 0
    for idx, pp in enumerate(p2):
        if pp[0]<min_x1:
            min_x2 = min_x1
            min_x2_idx = min_x1_idx
            min_x1 = pp[0]
            min_x1_idx = idx
            continue

        if pp[0]<min_x2:
            min_x2 = pp[0]
            min_x2_idx = idx


    #print(min_x1_idx, min_x2_idx)

    if min_x1_idx==min_x2_idx: # 处理两个最小值相同的情况
        min_x2_idx += 1
        if min_x2_idx>15:
            min_x2_idx = 0
        if p2[min_x1_idx][0]!=p2[min_x2_idx][0]:
            min_x2_idx = min_x1_idx - 1
            if min_x2_idx<0:
                min_x2_idx = 15
            if p2[min_x1_idx][0]!=p2[min_x2_idx][0]:
                print(2, p2)
                assert False


    # 应该是挨着的
    if abs(min_x1_idx - min_x2_idx)==1 or abs(min_x1_idx - min_x2_idx)==15:
        pass
    else:
        print(2, p2)
        return None
        #assert abs(min_x1_idx - min_x2_idx)==1 or abs(min_x1_idx - min_x2_idx)==15

    # 调整到最左位置
    while abs(min_x1_idx - min_x2_idx)!=15:
        p2 = np.roll(p2, 1, axis=0)
        min_x1_idx += 1
        min_x2_idx += 1
        if min_x1_idx==16:
            min_x1_idx = 0
        if min_x2_idx==16:
            min_x2_idx = 0


    if p2[0][1]>p2[-1][1]: # 逆时针
        p2 = p2[::-1]

    # 这里应该是顺时针了
    p2 = p2.tolist()

    # 处理成 两行
    p4 = []
    p5 = []
    width_diff = 5

    idx1 = 0
    idx2 = 15
    while idx1 <= idx2:
        if idx1==0:
            p4.append(p2[idx1])
            p5.append(p2[idx2])
            idx1 += 1
            idx2 -= 1
            continue


        if idx1==idx2: # 相遇，不一定是对半分
            # 找最后一个非零的
            p4_last = -1
            while p4[p4_last][1]==0:
                p4_last -= 1
            p5_last = -1
            while p5[p5_last][1]==0:
                p5_last -= 1

            if abs(p2[idx1][1]-p4[p4_last][1]) < abs(p2[idx1][1]-p5[p5_last][1]):
                p4.append(p2[idx1])
                p5.append([0,0])
            else:
                p4.append([0,0])
                p5.append(p2[idx2])  
            idx1 += 1
            idx2 -= 1
            continue


        if abs(p2[idx1][0]-p2[idx2][0])<=width_diff: # 在同一排
            p4.append(p2[idx1])
            p5.append(p2[idx2])
            idx1 += 1
            idx2 -= 1
        else:
            # 不同排，插入 [0,0]
            if p2[idx1][0]<p2[idx2][0]:
                p4.append(p2[idx1])
                p5.append([0,0])
                idx1 += 1
            else:
                p5.append(p2[idx2])
                p4.append([0,0])
                idx2 -= 1

    #print('len=', len(p4), len(p5))
    # 长度保持相同
    max_l = max(len(p4), len(p5))

    #print(4, p4)
    #print(5, p5)

    assert len(p4)==len(p5)

    boxes = []

    for idx in range(max_l):
        if idx==0:
            boxes.append([p4[idx], [], [], p5[idx]])
            continue

        if p4[idx]==[0,0] or p5[idx]==[0,0]: # 当前有 0,0
            if idx+1<max_l: 
                if p4[idx]==[0,0]:
                    boxes[-1][2] = p5[idx]
                else:
                    boxes[-1][1] = p4[idx]
                continue
            else: # 最后一排
                if boxes[-1][1]==[] and boxes[-1][2]==[]:
                    xxx = boxes.pop() # 废弃最后一个
                    #print('pop:', xxx)
                if p4[idx]==[0,0]:
                    boxes[-1][2] = p5[idx]
                else:
                    boxes[-1][1] = p4[idx]
                continue

        boxes[-1][1] = p4[idx]
        boxes[-1][2] = p5[idx]

        if idx+1<max_l:
            boxes.append([p4[idx], [], [], p5[idx]])

    #print(boxes)

    boxes = [ bb[0]+bb[1]+bb[2]+bb[3] for bb in boxes]

    return boxes


############################################################################3

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

    img2 = img.copy()

    for index, x in enumerate(labels[fn]):
        if len(x['code'])<1: # 忽略0个字的
            continue

        poly_name = f'{bn}_{index}.jpg'

        boxes = []

        if len(x['points'])==8: # 4个点的 直接截取
            boxes.append(x['points'])
        elif len(x['code'])<5: # 四字以下，使用最小面积的方法
            # 计算 最小框的面积
            xy = [item for item in map(float, x['points'])]
            poly = np.array(xy).reshape([len(xy)//2, 2])
            poly = orderConvex2(poly)
            boxes.append(poly.astype(np.int32).reshape((8,)).tolist())
        else:
            if len(x['points'])>32: # 只取32个
                #print(fn, x['points'])
                x['points'] = x['points'][:32]

            boxes = get_boxes_32p(x)
            if boxes is None:
                print(fn, "error!!!")
                continue

        final_img = None

        for box in boxes:
            poly = orderConvex(box)
            poly_rec = poly.astype(np.int32).reshape((8,)).tolist()

            part_im = charRec(img, poly_rec, adjust=False)
            if part_im is not None:
                #poly_labels.append( poly_name + ' ' + ' '.join([str(a) for a in x['code']]) )
                if final_img is None:
                    final_img = part_im
                else:
                    final_img = cv2.hconcat([final_img, part_im])
            else:
                #print("error cut:", box, poly_name)

                # 画框
                poly = np.array(box, np.int32)
                cv2.polylines(img2, [poly.reshape((-1, 1, 2))], True,color=(0, 255, 0), thickness=2)

                # 保存画框的图片
                cv2.imwrite(os.path.join(output_dir, 'box_'+fn), img2)

        if final_img is None:
            continue

        cv2.imwrite(os.path.join(output_dir, "image", poly_name), final_img)
        poly_labels.append( poly_name + ' ' + ' '.join([str(a) for a in x['code']]) )
    
        max_length = max(max_length, len(x['code']))

    #break # just one round for test


# 保存 labels
with open(os.path.join(output_dir, 'all_labels.txt'), 'w') as output_data:
    for s in poly_labels:
        output_data.write(s.strip() + '\n')

print("\ntotal=", len(poly_labels))
print("max_length=", max_length)
