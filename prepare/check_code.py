# 检查数据集与码表对照
import os
import json
from tqdm import tqdm

code_file = '../data/char_code.txt'
label_json_file = '../data/train/label.json'
output_dir = '../data'

with open(label_json_file, 'r') as f:
    labels = json.load(f)

with open(code_file, 'r') as f:
    char_code = f.read()

char_code = char_code.split('\n')

unknown_code = []

for fn in tqdm(labels.keys()):
    for x in labels[fn]:
        #code = []
        for c in x['transcription']:
            if c in char_code:
                pos = char_code.index(c)
                #code.append(pos)
            else:
                #print('code err: %s %x'%(c, ord(c)))
                #code.append(char_code.index('#'))  # 未找到的编码用 ‘#’ 代替
                if c not in unknown_code:
                    unknown_code.append(c)

with open(os.path.join(output_dir, 'unknown_code.txt'), 'w') as f:
    f.write('\n'.join(unknown_code))

print('unknown_code: ', len(unknown_code))