import os
import cv2
from glob import glob
from tqdm import tqdm

image_dir = '../../data/chardata/image'
output_dir = '../../data/chardata/'
label_file = '../../data/chardata/all_labels.txt'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(os.path.join(output_dir, "c_image")):
    os.makedirs(os.path.join(output_dir, "c_image"))

limit_width = 600

max_h, max_w = 0, 0

images = []

# 统计最大的图片宽度
for f in tqdm(glob(image_dir+'/*.jpg')):
    img = cv2.imread(f)

    h, w, _ = img.shape

    max_w = max(max_w, w)
    max_h = max(max_h, h)

    images.append([f, w])

print("max_w=", max_w)
print("max_h=", max_h)

images = sorted(images, key = lambda x: x[1])

#print(images)

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


# 拼接
c_width = 0
c_img = None
c_bn = []
new_labels = []

for x in tqdm(images):
    img = cv2.imread(x[0])
    fn = os.path.split(x[0])[-1]
    bn, _ = os.path.splitext(fn)

    if c_width + img.shape[1] < limit_width:
        if c_img is None:
            c_img = img
        else:
            c_img = cv2.hconcat([c_img, img])
        c_width += img.shape[1]
        c_bn.append(bn)
    else:
        n_fn = f"{c_bn[0]}_{len(c_bn)}.jpg"
        r = cv2.imwrite(os.path.join(output_dir, "c_image", n_fn), c_img)
        l = [n_fn]
        for i in c_bn:
            l.extend(label_dic[i+".jpg"])
        new_labels.append(' '.join(l))
        c_width = img.shape[1]
        c_img = img
        c_bn = [bn]

if len(bn)>0:
    n_fn = f"{c_bn[0]}_{len(c_bn)}.jpg"
    r = cv2.imwrite(os.path.join(output_dir, "c_image", n_fn), c_img)
    l = [n_fn]
    for i in c_bn:
        l.extend(label_dic[i+".jpg"])
    new_labels.append(' '.join(l))

# 保存labels
with open(os.path.join(output_dir, "c_all_labels.txt"), "w") as f:
    for s in new_labels:
        f.write(s.strip() + '\n')
