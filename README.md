# 古籍OCR



## 数据准备

### 旋转图片

```bash
cd prepare
python3 rotate.py
```



### 生成CTPN训练数据

```bash
cd ctpn/utils/prepare
python3 split_label.py
```



### 生成Densenet训练数据

```bash
cd densenet/prepare
python3 cut2data.py
python3 split_labels.py
```



## CTPN训练

### 重新训练

```bash
python3 -m ctpn.main.train
```



### 继续训练

```bash
python3 -m ctpn.main.train --restore true
```



## CTC-Densenet训练

```bash
cd densenet/train
python3 train.py
```
