import math
import numpy as np
import cv2

def myCanny(image, tl, th):

    # Gauss Blur
    image = cv2.GaussianBlur(image, (5,5), 3.0, sigmaY=3.0)
    print('Gauss Blur down')
    # Gradient Calculation
    gx = cv2.Sobel(image,cv2.CV_16S,1,0) 
    gy = cv2.Sobel(image,cv2.CV_16S,0,1) 
    m = (gx*gx + gy*gy) ** 0.5
    theta = np.zeros(m.shape)
    for x in range(0, m.shape[0]):
        for y in range(0, m.shape[1]):
            if(gx[x][y] == 0):
                theta[x][y] = math.pi/2
            else:
                theta[x][y] = math.atan(gy[x][y]/gx[x][y])
    print('Gradient Calculation down')
    # Non-Maximum Suppression
    gn = m.copy()
    for x in range(0, m.shape[0]):
        for y in range(0, m.shape[1]):
            if math.pi * -1/8 <= theta[x][y] and  theta[x][y] <= math.pi * 1/8:
                target = (0,1,0,-1)
            elif math.pi * 1/8 <= theta[x][y] and  theta[x][y] <= math.pi * 3/8:
                target = (-1,-1,1,1)
            elif math.pi * -3/8 <= theta[x][y] and  theta[x][y] <= math.pi * -1/8:
                target = (1,-1,-1,1)
            else:
                target = (1,0,-1,0)
            if 0 <= x + target[0] and x + target[0] < m.shape[0] and 0 <= y + target[1] and y + target[1] < m.shape[1]:
                if m[x][y] < m[x+target[0]][y+target[1]]:
                    gn[x][y] = 0
            if 0 <= x + target[2] and x + target[2] < m.shape[0] and 0 <= y + target[3] and y + target[3] < m.shape[1]:
                if m[x][y] < m[x+target[2]][y+target[3]]:
                    gn[x][y] = 0
    print('Non-Maximum Suppression down')
    # Dual-threshold edge detection
    gnh = gn.copy()
    gnl = gn.copy()
    for x in range(0, gn.shape[0]):
        for y in range(0, gn.shape[1]):
            if gn[x][y] > 255:                                 
                gn[x][y] = 255
            if gn[x][y] < th:
                gnh[x][y] = 0
            if gn[x][y] < tl:
                gnl[x][y] = 0
    gnl = gnl - gnh
    label = np.zeros(gn.shape)
    s = []
    q = []
    connected = False
    for x in range(0, gn.shape[0]):
        for y in range(0, gn.shape[1]):
            if gnl[x][y] > 0 and label[x][y] == 0:
                label[x][y] = 255
                s.append((x,y))
                q.append((x,y))
                while s:
                    xy = s.pop()
                    target = (-1,-1,-1,0,-1,1,0,-1,0,1,1,-1,1,0,1,1)
                    for i in range(0, 8):
                        tempx, tempy = xy[0] + target[i*2], xy[1] + target[i*2+1]
                        if 0 <= tempx and tempx < gn.shape[0] and 0 <= tempy and tempy < gn.shape[1]:
                            if gnl[tempx][tempy] > 0 and label[tempx][tempy] == 0:
                                label[tempx][tempy] = 255
                                s.append((tempx,tempy))
                                q.append((tempx,tempy))
                            if gnh[tempx][tempy] > 0:
                                connected = True
                if connected == False:
                    while q:
                        xy = q.pop()
                        label[xy[0]][xy[1]] = 0
                q = []
                connected = False
            if gnh[x][y] > 0:
                label[x][y] = 255
    print('Dual-threshold edge detection down')
    return label.astype(np.uint8)
    
    

image = cv2.imread("lena.jpg", 0)
mycanny = myCanny(image,20,100)
cvcanny = cv2.Canny(image, 20,100)
cv2.imshow('image', image)
cv2.imshow('myCanny', mycanny)
cv2.imshow('OpencvCanny', cvcanny)
k = cv2.waitKey(0)
if k == 27:       # wait for ESC key to exit     
    cv2.destroyAllWindows()