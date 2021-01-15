import math
import numpy as np
import cv2

def myOtsu(image):
    n = np.zeros(256,dtype=int)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            n[image[x][y]] += 1
    p = n / sum(n)
    p1 = np.zeros(256)
    m = np.zeros(256)
    for k in range(0, 256):
        p1[k] = sum(p[:k+1])
        if k > 0:
            m[k] = m[k-1] + k * p[k]
    mg = m[255]
    varB = np.zeros(256)
    for k in range(0, 256):
        if p1[k] > 0 and p1[k] < 1:
            varB[k] = ((mg * p1[k] - m[k]) ** 2) / (p1[k] * (1 - p1[k]))
    resultList = []
    for k in range(0, 256):
        if varB[k] == np.amax(varB):
            resultList.append(k)
    result = np.average(resultList).astype(np.uint8)
    print(result)
    newImage = np.zeros(image.shape)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if image[x][y] > result:
                newImage[x][y] = 255
    return newImage

image = cv2.imread('3.jpg',0)
new = myOtsu(image)
cv2.imshow('image', new)
k = cv2.waitKey(0)
if k == 27:       # wait for ESC key to exit     
    cv2.destroyAllWindows()