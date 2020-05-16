#!/usr/bin/env python
#-*- coding:utf-8 -*-
import cv2
import numpy as np

img_path = r"/home/pi/camera/cut/p01.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray", gray)
cv2.imwrite("/home/pi/camera/cut/gray.jpg",gray)

blurred = cv2.GaussianBlur(gray, (9, 9),0)
#cv2.imshow("blurred", blurred)
cv2.imwrite("/home/pi/camera/cut/blurred.jpg",blurred)

gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0)
gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1)

gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)
cv2.imwrite("/home/pi/camera/cut/gradient.jpg",gradient)

blurred = cv2.GaussianBlur(gradient, (9, 9),0)
(_, thresh) = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
cv2.imwrite("/home/pi/camera/cut/thresh.jpg",thresh)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("/home/pi/camera/cut/closed.jpg",closed)

closed = cv2.erode(closed, None, iterations=4)
closed = cv2.dilate(closed, None, iterations=4)
cv2.imwrite("/home/pi/camera/cut/closed2.jpg",closed)

img1, cnts, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect))

# draw a bounding box arounded the detected barcode and display the image
draw_img = cv2.drawContours(closed, [box], -1, (0, 0, 255), 3)
#cv2.imshow("draw_img", draw_img)
cv2.imwrite("/home/pi/camera/cut/draw_img.jpg",draw_img)

Xs = [i[0] for i in box]
Ys = [i[1] for i in box]
x1 = min(Xs)
x2 = max(Xs)
y1 = min(Ys)
y2 = max(Ys)
hight = y2 - y1
width = x2 - x1
crop_img= img[y1:y1+hight, x1:x1+width]
#cv2.imshow('crop_img', crop_img)
cv2.imwrite("/home/pi/camera/cut/crop_img.jpg",crop_img)






cv2.waitKey(0)
