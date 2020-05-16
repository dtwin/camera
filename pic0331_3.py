#!/usr/bin/env python
#-*- coding:utf-8 -*-


import cv2
#import matplotlib
#import pyplot as plt

image = cv2.imread("/home/pi/camera/0331_1.jpg")
image_BGR = image.copy()
#p1 = cv2.imwrite("/home/pi/camera/p1.jpg",image_BGR)

# 将图像转换成灰度图像,并执行图像高斯模糊,以及转化成二值图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#blurred = cv2.GaussianBlur(gray, (1,1), 0)
ret,image_binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
p22 = cv2.imwrite("/home/pi/camera/p22.jpg",image_binary)

# 从二值图像中提取轮廓
# contours中包含检测到的所有轮廓,以及每个轮廓的坐标点
#contours = cv2.findContours(image_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

# 遍历检测到的所有轮廓,并将检测到的坐标点画在图像上
# c的类型numpy.ndarray,维度(num, 1, 2), num表示有多少个坐标点
img1, contours, hierarchy = cv2.findContours(image_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (0, 0, 255), 3)

image_contours = image
p3 = cv2.imwrite("/home/pi/camera/p3.jpg",image_contours)


