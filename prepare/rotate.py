import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
from densenet.prepare.cut2data import orderConvex2, get_boxes_32p

code_file = 'data/char_code.txt'

label_json_file = 'data/train/label.json'
image_dir = 'data/train/image1' # image1 为筛选后的，去掉行书、不整齐等, 用于ctc-densenet
output_dir = 'data/rotated1'
#image_dir = 'data/example'
#output_dir = 'data/example/rotated'
label_json_file2 = output_dir+'/label.json'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(os.path.join(output_dir, "image")):
    os.makedirs(os.path.join(output_dir, "image"))
if not os.path.exists(os.path.join(output_dir, "label")):
    os.makedirs(os.path.join(output_dir, "label"))

with open(code_file, 'r') as f:
    char_code = f.read()

char_code = char_code.split('\n')

with open(label_json_file, 'r') as f:
    labels = json.load(f)

for f in tqdm(glob(image_dir+'/*.jpg')):
    fn = os.path.split(f)[-1] # 文件名

    #print(fn)

    img = cv2.imread(f)

    h,w,_ = img.shape

    ctpn_gt = ''

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

        # 生成ctpn标注
        #ctpn_gt += ','.join([str(i) for i in x2]) + '\n'

        if len(x['points'])==8: # 4个点的 直接截取
            ctpn_gt += ','.join([str(i) for i in x2]) + '\n'
        elif len(x['code'])<5: # 四字以下，使用最小面积的方法
            # 计算 最小框的面积
            xy = [item for item in map(float, x['points'])]
            poly = np.array(xy).reshape([len(xy)//2, 2])
            poly = orderConvex2(poly)
            x3 = poly.astype(np.int32).reshape((8,)).tolist()
            ctpn_gt += ','.join([str(i) for i in x3]) + '\n'
        else:
            if len(x['points'])>32: # 只取32个
                #print(fn, x['points'])
                x['points'] = x['points'][:32]

            boxes = get_boxes_32p(x)
            if boxes is None:
                print(fn, "error!!!", x2)
                ctpn_gt += ','.join([str(i) for i in x2]) + '\n'
            else:
                for bb in boxes:
                    ctpn_gt += ','.join([str(i) for i in bb]) + '\n'

    cv2.imwrite(os.path.join(output_dir, "image", fn), img2)

    # 生成 ctpn 文件
    bfn, ext = os.path.splitext(fn)
    with open(os.path.join(output_dir, "label", 'gt_'+bfn+'.txt'), 'w') as f:
        f.write(ctpn_gt)


with open(label_json_file2, 'w') as f:
    json.dump(labels, f, indent=4)
