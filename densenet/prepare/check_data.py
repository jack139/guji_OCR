import os
import cv2
from glob import glob
from tqdm import tqdm

image_dir = '../../data/chardata1/image'
label_file = '../../data/chardata1/f_in_labels.txt'

width_span = [0]*30
label_span = [0]*10

# 读入 labels
max_labellength = 0
res = []
with open(label_file, 'r') as f:
    lines = f.readlines()
    for i in lines:
        res.append(i.strip())

label_dic = {}
for i in res:
    p = i.split(' ')
    label_dic[p[0]] = p[1:]
    max_labellength = max(max_labellength, len(p[1:]))

    label_span[len(p[1:])//10] += 1

print("max_labellength=", max_labellength)


max_h, max_w = 0, 0

# 统计最大的图片宽度
for f in tqdm(label_dic.keys()):
    path = os.path.join(image_dir, f)
    img = cv2.imread(path)

    h, w, _ = img.shape

    max_w = max(max_w, w)
    max_h = max(max_h, h)

    width_span[w//100] += 1

print("max_w=", max_w)
print("max_h=", max_h)


print('\n', '\t'.join([str(i) for i in range(30)]))
print('\t'.join([str(i) for i in width_span]))

print('\n', '\t'.join([str(i) for i in range(10)]))
print('\t'.join([str(i) for i in label_span]))
