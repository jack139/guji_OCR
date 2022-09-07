# 古籍OCR



## 数据准备



### 旋转图片

```
cd prepare
python3 rotate.py
```



### 生成CTPN标记

```
cd ctpn/utils/prepare
python3 split_label.py
```



## 训练
```
python3 -m ctpn.main.train
```