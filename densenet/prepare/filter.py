import os
import cv2
from glob import glob
from tqdm import tqdm

image_dir = '../../data/chardata1/image'
label_file = '../../data/chardata1/all_labels.txt'
output_dir = '../../data/chardata1/'

filter_width = 600
filter_label = 30

# 读入 labels
res = []
with open(label_file, 'r') as f:
    lines = f.readlines()
    for i in lines:
        res.append(i.strip())

label_dic = {}
for i in res:
    p = i.split(' ')
    label_dic[p[0]] = p[1:]

train_labels = []
test_labels = []

# 统计最大的图片宽度
for f in tqdm(glob(image_dir+'/*.jpg')):
    fn = os.path.split(f)[-1]
    img = cv2.imread(f)

    h, w, _ = img.shape

    if w >= filter_width or len(label_dic[fn])>=filter_label:
        test_labels.append(fn+' '+' '.join(label_dic[fn]))    
    else:
        train_labels.append(fn+' '+' '.join(label_dic[fn]))


# 保存labels
with open(os.path.join(output_dir, "f_in_labels.txt"), "w") as f:
    for s in train_labels:
        f.write(s.strip() + '\n')

with open(os.path.join(output_dir, "f_out_labels.txt"), "w") as f:
    for s in test_labels:
        f.write(s.strip() + '\n')

print('train: ', len(train_labels))
print('test: ', len(test_labels))
