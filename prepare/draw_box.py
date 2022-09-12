import os
import json
import cv2
import numpy as np
from glob import glob

label_json_file = '../data/rotated/label.json'
#image_dir = '../data/example/rotated/image'
image_dir = '../data/test/1'
output_dir = '../data'

with open(label_json_file, 'r') as f:
    labels = json.load(f)

def draw_a_box(points):
    pts = np.array(points, np.int32)
    pts = pts.reshape((len(points)//2,2))

    pts = pts.reshape((-1,1,2))
    cv2.polylines(img, [pts], False, color=(255, 0, 0), thickness=2)


for f in glob(image_dir+'/*.jpg'):
    fn = os.path.split(f)[-1] # 文件名

    print(fn)

    img = cv2.imread(f)

    for x in labels[fn]:
        #print(x['points'])

        # 可以画16点
        if len(x['points'])==8:
            draw_a_box(x['points'])
        else:
            assert len(x['points'])==32

            #draw_a_box(x['points'])
            #continue

            p = x['points']

            print(1, p)

            # 检查

            b = []

            if abs(p[0]-p[2]) < abs(p[1]-p[3]) and p[1]>p[3]: # 正常
                # 1  2  3  4  5  6  7  8 
                # 0 15 14 13 12 11 10  9
                b.append(p[2*1:2*3] + p[2*15:] + p[:2*1]) 
                b.append(p[2*2:2*4] + p[2*14:2*15] + p[2*15:])
                b.append(p[2*3:2*5] + p[2*13:2*14] + p[2*14:2*15])
                b.append(p[2*4:2*6] + p[2*12:2*13] + p[2*13:2*14])
                b.append(p[2*5:2*7] + p[2*11:2*12] + p[2*12:2*13])
                b.append(p[2*6:2*8] + p[2*10:2*11] + p[2*11:2*12])
                b.append(p[2*7:2*11])

            elif abs(p[0]-p[2]) > abs(p[1]-p[3]) and p[0]<p[2]:
                # 15   14 13 12 11 10  9  8
                # 0  1  2  3  4  5  6     7
                b.append(p[2*15:] + p[2*14:2*15] + p[2*2:2*3] + p[:2*1]) 
                b.append(p[2*14:2*15] + p[2*13:2*14] + p[2*3:2*4] + p[2*2:2*3])
                b.append(p[2*13:2*14] + p[2*12:2*13] + p[2*4:2*5] + p[2*3:2*4])
                b.append(p[2*12:2*13] + p[2*11:2*12] + p[2*5:2*6] + p[2*4:2*5])
                b.append(p[2*11:2*12] + p[2*10:2*11] + p[2*6:2*7] + p[2*5:2*6])
                b.append(p[2*10:2*11] + p[2*8:2*9] + p[2*7:2*8] + p[2*6:2*7])
            elif abs(p[0]-p[2]) < abs(p[1]-p[3]) and p[1]<p[3]: # img_533.jpg
                # 0 15 14 13 12 11 10  9  8
                # 1     2  3  4  5  6     7
                b.append(p[:2*1] + p[2*14:2*15] + p[2*2:2*3] + p[2*1:2*2]) 
                b.append(p[2*14:2*15] + p[2*13:2*14] + p[2*3:2*4] + p[2*2:2*3])
                b.append(p[2*13:2*14] + p[2*12:2*13] + p[2*4:2*5] + p[2*3:2*4])
                b.append(p[2*12:2*13] + p[2*11:2*12] + p[2*5:2*6] + p[2*4:2*5])
                b.append(p[2*11:2*12] + p[2*10:2*11] + p[2*6:2*7] + p[2*5:2*6])
                b.append(p[2*10:2*11] + p[2*8:2*9] + p[2*7:2*8] + p[2*6:2*7])

            print(2, p)

            #draw_a_box(p)
            #continue

            for bb in b:
                draw_a_box(bb)

    cv2.imwrite(os.path.join(output_dir, 'box_'+fn), img)
