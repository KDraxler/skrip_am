import cv2
import sys
import numpy as np

# np.set_printoptions(threshold=sys.maxsize)
img = cv2.imread('oli.jpg')

width = 200
height = 200
dim = (width, height)
# resize image
rd = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

img_HSV = cv2.cvtColor(rd, cv2.COLOR_BGR2HSV)
H = cv2.normalize(img_HSV[:,:,0].astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX) #cek dadi range 0 - 1
S = cv2.normalize(img_HSV[:,:,1].astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX) #cek dadi range 0 - 1
V = cv2.normalize(img_HSV[:,:,2].astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX) #cek dadi range 0 - 1
cv2.imshow('HSV image', img_HSV)
print('Hueeee', H)
print('Saturationnn', S)
print('Valueee', V)
#
# cv2.imshow('Hue ', H)
# cv2.imshow('Saturation ', S)
# cv2.imshow('Value ', V)

# cv2.imshow('Huea ', img_HSV[:,:,0])
# cv2.imshow('Satturation ', img_HSV[:,:,1])
# cv2.imshow('Value ', img_HSV[:,:,2])

cv2.waitKey(0)
cv2.destroyAllWindows()