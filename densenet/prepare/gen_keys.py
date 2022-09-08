# 从康熙字典的字符集 keys

with open('../../data/char_code.txt', 'r') as f:
    char_code = f.read().split('\n')

with open('../keys.py', 'w') as output_data:
    output_data.write('alphabet = u"""%s"""'%''.join(char_code) + '\n')
