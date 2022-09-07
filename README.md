# 古籍OCR



## 数据准备

### 旋转图片

```
cd prepare
python3 rotate.py
```



### 生成CTPN训练数据

```
cd ctpn/utils/prepare
python3 split_label.py
```



## CTPN训练

### 重新训练

```
python3 -m ctpn.main.train
```



### 继续训练

```
python3 -m ctpn.main.train --restore true
```
