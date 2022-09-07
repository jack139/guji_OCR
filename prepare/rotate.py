import os
import json
import cv2
import numpy as np
from glob import glob

code_file = '../data/char_code.txt'
label_json_file = '../data/example/label_example.json'
image_dir = '../data/example'
output_dir = '../data/example/rotated'
label_json_file2 = output_dir+'/label_example.json'

with open(code_file, 'r') as f:
    char_code = f.read()

char_code = char_code.split('\n')

with open(label_json_file, 'r') as f:
    labels = json.load(f)

for f in glob(image_dir+'/*.jpg'):
    fn = os.path.split(f)[-1] # 文件名

    print(fn)

    img = cv2.imread(f)

    h,w,c = img.shape

    # 逆时针旋转90度
    img2 = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

    for x in labels[fn]:
        # 转换坐标，逆旋90度
        x2=[]
        for i in range(len(x['points'])//2):
            x2.append(x['points'][i*2+1])
            x2.append(w-x['points'][i*2])

        #print(x['points'])
        #print(x2)

        #pts = np.array(x2, np.int32)
        #pts = pts.reshape((len(x2)//2,2))
        #pts = pts.reshape((-1,1,2))
        #cv2.polylines(img2, [pts], True, color=(255, 0, 0), thickness=2)

        # 转换 char code
        code = []
        for c in x['transcription']:
            if c in char_code:
                pos = char_code.index(c)
                code.append(pos)
            else:
                print('code err: %s %x'%(c, ord(c)))
                code.append(char_code.index('#'))  # 未找到的编码用 ‘#’ 代替

        x['code'] = code
        x['points'] = x2

        assert len(x['code'])==len(x['transcription'])

    cv2.imwrite(os.path.join(output_dir, fn), img2)


with open(label_json_file2, 'w') as f:
    json.dump(labels, f, indent=4)
