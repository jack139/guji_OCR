import os
import random

train_ratio = 0.9
output_dir = '../../data/chardata'

random.seed(1)

with open(os.path.join(output_dir, 'all_labels.txt'), 'r') as f:
    new_labels = f.read().split('\n')

new_labels = [i for i in new_labels if len(i)>0]

# 随机分配 训练集 和 测试集
random.shuffle(new_labels)
n = len(new_labels)
train_n = int(n*train_ratio)


with open(os.path.join(output_dir, 'train_labels.txt'), 'w') as output_data:
    for s in new_labels[:train_n]:
        output_data.write(s.strip() + '\n')

with open(os.path.join(output_dir, 'test_labels.txt'), 'w') as output_data:
    for s in new_labels[train_n:]:
        output_data.write(s.strip() + '\n')

print("total=", len(new_labels))
print("train=", train_n)
