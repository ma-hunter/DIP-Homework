# OpenCV 人脸检测

<center><big>软件工程 1804</big></center>

数字图像处理综合实验报告。选题：利用 OpenCV 实现基于照片的人脸检测。

## 前置说明

#### Haar-like

通俗的讲，就是人脸特征。Haar特征值反映了图像的灰度变化情况。例如：脸部的一些特征能由矩形特征简单的描述，如：眼睛要比脸颊颜色要深，鼻梁两侧比鼻梁颜色要深，嘴巴比周围颜色要深等。

#### OpenCV API

本实验中使用到的 API 包含普通的读取图片，灰度转换，显示图像，简单的编辑图像等；说明如下：

##### 读取图片

需要提供图片的目标路径。

```python
import cv2
image = cv2.imread(imagepath)
```

##### 灰度转换

灰度转换的作用就是使得图片转换成灰度从而计算强度降低。

```python
import cv2
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
```

##### 画图

OpenCV 可以对图片进行任意的编辑处理；可以使用下面的函数在图片上绘制矩形：

```python
import cv2
cv2.rectangle(image,(x,y),(x+w,y+w),(0,255,0),2)
```

函数的最后一个参数指定了画笔的大小。

##### 显示图像

被处理后的图像要不被显示出来，要不被保存到物理存储介质。

```python
import cv2
cv2.imshow("Image Title",image)
```

##### 获取训练集

本质是对于人脸特征的一些描述；OpenCV 完成训练后，就可以感知图片上的特征，从而进行人脸检测

```python
import cv2
face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')
```

这个训练数据是开源的，可以直接拿来用。

训练数据参考地址：https://github.com/opencv/opencv/tree/master/data/haarcascades

##### 人脸检测

训练结束后，就可以使用 OpenCV 识别新的图片

```python
import cv2

faces = face_cascade.detectMultiScale(
   gray,
   scaleFactor = 1.15,
   minNeighbors = 5,
   minSize = (5,5),
   flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)
```

可以通过修改该函数的参数值，来实现对识别精度的调整。

通过上述 API 完成实验流程之后，就可以对取得的数据进行后处理，并将结果可视化。

## 代码实现

#### 基于图片

```python
import cv2
import numpy as np
import sys,os,glob,numpy
from skimage import io

img = cv2.imread("test.jpg")
color = (0, 255, 0)

grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

classfier = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")

faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
if len(faceRects) > 0: 
for faceRect in faceRects: 
x, y, w, h = faceRect
cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 3) 

cv2.imwrite('output.jpg',img)
cv2.imshow("Find Faces!",img)
cv2.waitKey(0)
```

#### 基于视频

```python
import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

cascPath = "haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:
if not video_capture.isOpened():
print('Unable to load camera.')
sleep(5)
pass

ret, frame = video_capture.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
gray,
scaleFactor=1.1,
minNeighbors=5,
minSize=(30, 30),
# flags=cv2.cv.CV_HAAR_SCALE_IMAGE
)

for (x, y, w, h) in faces:
cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Video', frame)

if cv2.waitKey(1) & 0xFF == ord('q'):
break

video_capture.release()
cv2.destroyAllWindows()
```

