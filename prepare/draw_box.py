import os
import json
import cv2
import numpy as np
from glob import glob

label_json_file = '../data/rotated/label.json'
image_dir = '../data/rotated/image'
#image_dir = '../data/test/1'
output_dir = '../data/test/2'

with open(label_json_file, 'r') as f:
    labels = json.load(f)

def draw_a_box(points):
    pts = np.array(points, np.int32)
    pts = pts.reshape((len(points)//2,2))

    pts = pts.reshape((-1,1,2))
    cv2.polylines(img, [pts], False, color=(255, 0, 0), thickness=2)


p4p5_error = 0

for f in glob(image_dir+'/*.jpg'):
    fn = os.path.split(f)[-1] # 文件名

    print('-------->', fn)

    img = cv2.imread(f)

    xn = 0
    for x in labels[fn]:
        #print(x['points'])

        # 可以画16点
        if len(x['points'])==8:
            draw_a_box(x['points'])
        elif len(x['points'])<32: # 小于32个
            print(fn, x['points'])
            assert len(x['points'])==32
        else:
            if len(x['points'])>32: # 只取32个
                print(fn, x['points'])
                x['points'] = x['points'][:32]

            #draw_a_box(x['points'])
            #continue

            p = x['points']
            p2 = np.array(p).reshape([16,2])

            #print(1, p)
            #print(2, p2)

            # 找出最小的两个，
            min_x1, min_x2 = 1e+6, 1e+6
            min_x1_idx, min_x2_idx = 0, 0
            for idx, pp in enumerate(p2):
                if pp[0]<min_x2:
                    min_x2 = pp[0]
                    min_x2_idx = idx

                if pp[0]<min_x1:
                    min_x2 = min_x1
                    min_x2_idx = min_x1_idx
                    min_x1 = pp[0]
                    min_x1_idx = idx
                    continue


            print(min_x1_idx, min_x2_idx)

            if min_x1_idx==min_x2_idx: # 处理两个最小值相同的情况
                min_x2_idx += 1
                if min_x2_idx>15:
                    min_x2_idx = 0
                if p2[min_x1_idx][0]!=p2[min_x2_idx][0]:
                    min_x2_idx = min_x1_idx - 1
                    if min_x2_idx<0:
                        min_x2_idx = 15
                    if p2[min_x1_idx][0]!=p2[min_x2_idx][0]:
                        #p4p5_error += 1 # 出错
                        print(2, p2)
                        assert False
                        continue


            # 应该是挨着的
            if abs(min_x1_idx - min_x2_idx)==1 or abs(min_x1_idx - min_x2_idx)==15:
                pass
            else:
                print(fn)
                print(2, p2)
                p4p5_error += 1
                continue
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

            #print(p2)
            #print(min_x1_idx, min_x2_idx)

            if p2[0][1]>p2[-1][1]: # 逆时针
                p2 = p2[::-1]

            # 这里应该是顺时针了
            #print(3, p2)
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
                        #if abs(p2[idx1][0]-p5[-1][0])>width_diff:
                        p5.append([0,0])
                        #p5.append(p2[idx2])
                        idx1 += 1
                    else:
                        p5.append(p2[idx2])
                        #if abs(p2[idx2][0]-p4[-1][0])>width_diff:
                        p4.append([0,0])
                        #p4.append(p2[idx1])
                        idx2 -= 1

            print('len=', len(p4), len(p5))
            # 长度保持相同
            max_l = max(len(p4), len(p5))
            #p4 = p4 + [[0,0]]*(max_l-len(p4))
            #p5 = p5 + [[0,0]]*(max_l-len(p5))

            print(4, p4)
            print(5, p5)

            assert len(p4)==len(p5)

            #if len(p4)!=len(p5):
            #    p4p5_error += 1
            #    continue


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
                            print('pop:', xxx)
                        if p4[idx]==[0,0]:
                            boxes[-1][2] = p5[idx]
                        else:
                            boxes[-1][1] = p4[idx]
                        continue

                boxes[-1][1] = p4[idx]
                boxes[-1][2] = p5[idx]

                if idx+1<max_l:
                    boxes.append([p4[idx], [], [], p5[idx]])

            print(boxes)

            boxes = [ bb[0]+bb[1]+bb[2]+bb[3] for bb in boxes]

            #print(boxes)

            #draw_a_box(p)
            #continue

            for bb in boxes:
                draw_a_box(bb)

        #if xn==5:
        #    break
        #else:
        #    xn += 1

    cv2.imwrite(os.path.join(output_dir, 'box_'+fn), img)

    print("p4p5_error=", p4p5_error)