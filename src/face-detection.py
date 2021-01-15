import cv2
import numpy as np
import sys,os,glob,numpy
from skimage import io

img = cv2.imread("../img/WechatIMG4.jpeg")
color = (0, 255, 0)

grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

classfier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
if len(faceRects) > 0: 
	for faceRect in faceRects:
		x, y, w, h = faceRect
cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 3) 

cv2.imwrite('../img/output.jpg',img)
cv2.imshow("Find Faces!",img)
cv2.waitKey(0)
