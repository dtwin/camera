#!/usr/bin/env python
#-*- coding:utf-8 -*-
import cv2
import os
import sys
import time

image = cv2.imread("/home/pi/camera/1.jpg")

weight = image.shape[1]
if weight>1600:                         # 正常发票
    cropImg = image[900:2600, 1100:3000]    # 裁剪【y1,y2：x1,x2】
            #cropImg = cv2.resize(cropImg, None, fx=0.5, fy=0.5,
                                 #interpolation=cv2.INTER_CUBIC) #缩小图像
    cv2.imwrite("/home/pi/camera/cut/p01.jpg", cropImg)
else:                                        # 卷帘发票
    cropImg_01 = image[30:150, 50:600]
    cv2.imwrite("/home/pi/camera/cut/p11.jpg", cropImg_01)
        


