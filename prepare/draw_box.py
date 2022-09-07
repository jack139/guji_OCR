import os
import json
import cv2
import numpy as np
from glob import glob

label_json_file = '../data/example/rotated/label_example.json'
image_dir = '../data/example/rotated/image'
output_dir = '../data'

with open(label_json_file, 'r') as f:
    labels = json.load(f)

for f in glob(image_dir+'/*.jpg'):
    fn = os.path.split(f)[-1] # 文件名

    print(fn)

    img = cv2.imread(f)

    for x in labels[fn]:
        #print(x['points'])

        # 可以画16点
        pts = np.array(x['points'], np.int32)
        pts = pts.reshape((len(x['points'])//2,2))

        # 只画4点
        #if len(x['points'])>8:
        #    x3 = []
        #    x3.extend(x['points'])

        pts = pts.reshape((-1,1,2))
        cv2.polylines(img, [pts], False, color=(255, 0, 0), thickness=2)



    cv2.imwrite(os.path.join(output_dir, 'box_'+fn), img)
