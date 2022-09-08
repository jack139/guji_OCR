import cv2
from glob import glob
from tqdm import tqdm

image_dir = '../../data/chardata/image'

max_h, max_w = 0, 0

# 统计最大的图片宽度
for f in tqdm(glob(image_dir+'/*.jpg')):
	img = cv2.imread(f)

	h, w, _ = img.shape

	max_w = max(max_w, w)
	max_h = max(max_h, h)

print("max_w=", max_w)
print("max_h=", max_h)
