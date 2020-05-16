#!/usr/bin/env python
#-*- coding:utf-8 -*-


import cv2
#import matplotlib
#import pyplot as plt

image = cv2.imread("/home/pi/camera/0331_2.jpg")
image_BGR = image.copy()
p1 = cv2.imwrite("/home/pi/camera/p11.jpg",image_BGR)

# 将图像转换成灰度图像,并执行图像高斯模糊,以及转化成二值图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
ret,image_binary = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY)
p2 = cv2.imwrite("/home/pi/camera/p21.jpg",image_binary)

# 从二值图像中提取轮廓
# contours中包含检测到的所有轮廓,以及每个轮廓的坐标点
#contours = cv2.findContours(image_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

# 遍历检测到的所有轮廓,并将检测到的坐标点画在图像上
# c的类型numpy.ndarray,维度(num, 1, 2), num表示有多少个坐标点
img1, contours, hierarchy = cv2.findContours(image_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (0, 0, 255), 3)

image_contours = image
p3 = cv2.imwrite("/home/pi/camera/p31.jpg",image_contours)

# display BGR image
#plt.subplot(1, 3, 1)
#plt.imshow(image_BGR)
#plt.axis('off')
#plt.title('image_BGR')

# display binary image
#plt.subplot(1, 3, 2)
#plt.imshow(image_binary, cmap='gray')
#plt.axis('off')
#plt.title('image_binary')

# display contours
#plt.subplot(1, 3, 3)
#plt.imshow(image_contours)
#plt.axis('off')
#plt.title('{} contours'.format(len(contours)))

#plt.show()
