# coding=utf-8
import os
import shutil
import sys
import time

import cv2
from math import *
import numpy as np

import tensorflow as tf
#tf.reset_default_graph()

from ctpn.nets import model_train as model
from ctpn.utils.rpn_msr.proposal_layer import proposal_layer
from ctpn.utils.text_connector.detectors import TextDetector

from config.settings import CTPN_CHECKPOINT, GPU_MEMORY_CTPN

tf.app.flags.DEFINE_string('output_path', '../test_result/', '')
tf.app.flags.DEFINE_string('output_path2', '../test_results/', '')
tf.app.flags.DEFINE_string('gpu', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', CTPN_CHECKPOINT, '')
FLAGS = tf.app.flags.FLAGS


def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im, (float(new_h) / img_size[0], float(new_w) / img_size[1])

def load_tf_model():
    #if os.path.exists(FLAGS.output_path):
    #    shutil.rmtree(FLAGS.output_path)
    #os.makedirs(FLAGS.output_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
    input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    bbox_pred, cls_pred, cls_prob = model.model(input_image)

    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())

    # 控制gpu内存使用
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_MEMORY_CTPN)
    # 建立session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
    model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
    print('Restore from {}'.format(model_path))
    saver.restore(sess, model_path)

    return sess, input_image, input_im_info, bbox_pred, cls_pred, cls_prob

# 初始化tensorflow session
sess, input_image, input_im_info, bbox_pred, cls_pred, cls_prob = load_tf_model()

# 选择图片
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

# 处理 卡号的 图片， 角度，截取
def charRec(img, rec, adjust=False):
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

    xlength = int((rec[6] - rec[0]) * 0.1) # origin is 0.1
    ylength = int((rec[7] - rec[1]) * 0.2)
    if adjust:
        pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
        pt2 = (rec[2], rec[3])
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
        return None

    return partImg


# 处理一个图片，输入是img, 银行卡识别， 返回最宽的文字框（卡号）
# adjust 扩大文本框
def process_one2(im, adjust=False):
    start = time.time()

    img, (rh, rw) = resize_image(im)
    h, w, c = img.shape
    im_info = np.array([h, w, c]).reshape([1, 3])

    bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob], 
            feed_dict={input_image: [img],
            input_im_info: im_info})  ## 耗时 avg 1.2s

    textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)  ## 耗时  avg 0.5s
    scores = textsegs[:, 0]
    textsegs = textsegs[:, 1:5]

    textdetector = TextDetector(DETECT_MODE='O')
    boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
    boxes = np.array(boxes, dtype=np.int)

    img2 = img.copy()

    max_box = [0,0,0]

    # 文字框按纵坐标排序
    #boxes = sorted(boxes, key=lambda b:b[1], reverse=True)

    # draw boxes
    for i, box in enumerate(boxes):
        #cv2.polylines(img, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
        #              thickness=2)
        if box[2]-box[0]>max_box[2]-max_box[0]:
            if box[5]<(h*0.4): # 如果在卡上部，应该不是卡号，可能是标题，卡号应该在图片下半部
                continue
            else:
                max_box = box

    #print(max_box, h, w)

    cv2.polylines(img, [max_box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
                      thickness=2)

    # 画框后的图片  
    img = cv2.resize(img, None, None, fx=1.0 / rh, fy=1.0 / rw, interpolation=cv2.INTER_LINEAR)

    # 裁剪最宽的框（卡号）
    # 左上 (0, 1) 右上 (2, 3) 右下 (4, 5) 左下 (6, 7)
    '''
    if adjust:
        left_x = min(max_box[0], max_box[6])
        left_y = min(max_box[1], max_box[3])
        right_x = max(max_box[2], max_box[4])
        right_y = max(max_box[5], max_box[7])

        xlength = 0 #int((max_box[4] - max_box[0]) * 0.1) # origin is 0.1
        ylength = 0 #int((max_box[5] - max_box[1]) * 0.3)
        cropImg = img2[
            max(1, left_y-ylength):min(h, right_y+ylength),
            max(1, left_x-xlength):min(w, right_x+xlength)
        ]
    else:
        cropImg = img2[max_box[1]:max_box[5],max_box[0]:max_box[4]]
    '''
    cropImg = charRec(img2, max_box, adjust=adjust)

    print("ctpn ====> {:.2f}s".format(time.time() - start))

    # 返回
    return img, cropImg, max_box, (h, w) #boxes


# 处理一个图片，输入是文件路径
def process_one(im_fn):
    print('===============')
    print(im_fn)
    try:
        im = cv2.imread(im_fn)[:, :, ::-1]
    except:
        print("Error reading image {}!".format(im_fn))
        return None

    img, cropImg, boxes = process_one2(im)

    # 保存画框后的图片  
    cv2.imwrite(os.path.join(FLAGS.output_path, os.path.basename(im_fn)), img[:, :, ::-1])

    # 保存坐标文件
    #with open(os.path.join(FLAGS.output_path, os.path.splitext(os.path.basename(im_fn))[0]) + ".txt",
    #          "w") as f:
    #    for i, box in enumerate(boxes):
    #        line = ",".join(str(box[k]) for k in range(8))
    #        line += "," + str(scores[i]) + "\r\n"
    #        f.writelines(line)

    # 裁剪最宽的框（卡号）
    cv2.imwrite(os.path.join(FLAGS.output_path2, 'c_'+os.path.basename(im_fn)), cropImg[:, :, ::-1])

    # 返回画框的坐标
    return boxes



# 处理一个图片，输入是img, 处理身份证，返回所有文本框
def process_one_idcard(im, debug=False):
    start = time.time()

    img, (rh, rw) = resize_image(im)
    h, w, c = img.shape
    im_info = np.array([h, w, c]).reshape([1, 3])

    bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob], 
            feed_dict={input_image: [img],
            input_im_info: im_info})  ## 耗时 avg 1.2s

    textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)  ## 耗时  avg 0.5s
    scores = textsegs[:, 0]
    textsegs = textsegs[:, 1:5]

    textdetector = TextDetector(DETECT_MODE='O')
    boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
    boxes = np.array(boxes, dtype=np.int)

    if debug:
        img2 = img.copy()

        #cropImgs = []

        # 文字框按纵坐标排序
        #boxes = sorted(boxes, key=lambda b:b[1])


        # draw boxes
        for i, box in enumerate(boxes):
            cv2.polylines(img2, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
                          thickness=2)

            # 裁剪框
            #cropImgs.append(img2[box[1]:box[5],box[0]:box[4]])


        # 画框后的图片  
        img2 = cv2.resize(img2, None, None, fx=1.0 / rh, fy=1.0 / rw, interpolation=cv2.INTER_LINEAR)
    else:
        img2 = img

    print("ctpn ====> {:.2f}s".format(time.time() - start))

    # 返回
    return img2, img, boxes  # 画框的， 原始的， 框坐标



# 处理一个图片，输入是img, 通用OCR，返回所有文本框
def process_text(im, debug=False):
    start = time.time()

    img, (rh, rw) = resize_image(im)
    h, w, c = img.shape
    im_info = np.array([h, w, c]).reshape([1, 3])

    bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob], 
            feed_dict={input_image: [img],
            input_im_info: im_info})  ## 耗时 avg 1.2s

    textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)  ## 耗时  avg 0.5s
    scores = textsegs[:, 0]
    textsegs = textsegs[:, 1:5]

    #print(textsegs.shape)
    #img3 = img.copy()
    #for i in np.array(textsegs, dtype=np.int): # 打印所有proposal框
    #    cv2.rectangle(img3, (i[0], i[1]), (i[2], i[3]), (0, 255, 0), 2)
    #cv2.imwrite('../test_data/res/img3.jpg',img3)

    textdetector = TextDetector(DETECT_MODE='O')
    boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
    boxes = np.array(boxes, dtype=np.int)

    # 转换boxes坐标
    for i in boxes:
        i[0] /= rw
        i[1] /= rh
        i[2] /= rw
        i[3] /= rh
        i[4] /= rw
        i[5] /= rh
        i[6] /= rw
        i[7] /= rh

    if debug:
        img2 = im.copy()

        #cropImgs = []

        # 文字框按纵坐标排序
        #boxes = sorted(boxes, key=lambda b:b[1])


        # draw boxes
        for i, box in enumerate(boxes):
            cv2.polylines(img2, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
                          thickness=2)

            # 裁剪框
            #cropImgs.append(img2[box[1]:box[5],box[0]:box[4]])


        # 画框后的图片  
        #img2 = cv2.resize(img2, None, None, fx=1.0 / rh, fy=1.0 / rw, interpolation=cv2.INTER_LINEAR)
    else:
        img2 = img

    print("ctpn ====> {:.2f}s".format(time.time() - start))

    # 返回
    return img2, boxes  # 画框的， 原始的， 框坐标
