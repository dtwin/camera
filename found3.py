#!/usr/bin/env python
#-*- coding:utf-8 -*-
import cv2
import numpy as np
import math
import os
import sys
import time

#rimage_path="/home/pi/test.jpg"
rimage_path="/home/pi/camera/2.jpg"
wimage_path="/home/pi/camera/found/c01.jpg"
rectangle_y1=1200
rectangle_y2=2500
rectangle_x1=1200
rectangle_x2=1800
class Config:
    def __init__(self):
        pass
    src = "photo1.jpg"
    resizeRate = 0.5
    min_area = 1200
    min_contours = 8
    threshold_thresh = 55
    epsilon_start = 10
    epsilon_step = 10

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
   # return the ordered coordinates
    return rect

# 求两点间的距离
def point_distance(a,b):
    return int(np.sqrt(np.sum(np.square(a - b))))

# 找出外接四边形, c是轮廓的坐标数组
def boundingBox(idx,c):
    if len(c) < Config.min_contours: 
        return None
    epsilon = Config.epsilon_start
    while True:
        approxBox = cv2.approxPolyDP(c,epsilon,True)
        #求出拟合得到的多边形的面积
        theArea = math.fabs(cv2.contourArea(approxBox))
        #输出拟合信息
        print("contour idx: %d ,contour_len: %d ,epsilon: %d ,approx_len: %d ,approx_area: %s"%(idx,len(c),epsilon,len(approxBox),theArea))
        if (len(approxBox) < 4):
            return None
        if theArea > Config.min_area:
            if (len(approxBox) > 4):
                # epsilon 增长一个步长值
                epsilon += Config.epsilon_step               
                continue
            else: #approx的长度为4，表明已经拟合成矩形了                
                #转换成4*2的数组
                approxBox = approxBox.reshape((4, 2))                
                return approxBox                
        else:
            print("failed to find boundingBox,idx = %d area=%f"%(idx, theArea))
            return None


cap = cv2.VideoCapture(0)
ret, image = cap.read()
#image = cv2.imread(rimage_path)
#weight = image.shape[1]
#image = image[rectangle_y1:rectangle_y2,rectangle_x1:rectangle_x2]
cv2.imwrite(wimage_path,image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
srcWidth, srcHeight, channels = image.shape
print(srcWidth, srcHeight)

binary = cv2.medianBlur(gray,7)
ret, binary = cv2.threshold(binary, Config.threshold_thresh, 255, cv2.THRESH_BINARY)
binary = cv2.erode (binary, None, iterations = 2)
cv2.imwrite("1-threshold.png", binary, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

# canny提取轮廓
binary = cv2.Canny(binary, 0, 60, apertureSize = 3)
cv2.imwrite("3-canny.png", binary, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

# 提取轮廓后，拟合外接多边形（矩形）
_,contours,_ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("len(contours)=%d"%(len(contours)))

#针对每个轮廓，拟合外接四边形,如果成功，则将该区域切割出来，作透视变换，并保存为图片文件
for idx,c in enumerate(contours):
    approxBox = boundingBox(idx,c)
    if approxBox is None: 
        print("\n")
        continue
    
     # 获取最小矩形包络
    rect = cv2.minAreaRect(approxBox)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    box = box.reshape(4,2)
    box = order_points(box)
    print("boundingBox：\n",box)   

    # 待切割区域的原始位置，
    # approxPolygon 点重排序, [top-left, top-right, bottom-right, bottom-left]
    src_rect = order_points(approxBox)  
    print("src_rect：\n",src_rect)
   
    w,h = point_distance(box[0],box[1]), point_distance(box[1],box[2])
    print("w = %d ,h= %d "%(w,h))
    # 生成透视变换矩阵
    dst_rect = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]],
        dtype="float32")

    # 透视变换
    M = cv2.getPerspectiveTransform(src_rect, dst_rect)

    #得到透视变换后的图像
    warped = cv2.warpPerspective(image, M, (w, h))

    #将变换后的结果图像写入png文件
    #cv2.imwrite("/home/pi/camera/found/warped.jpg",warped)
    cv2.imwrite("/home/pi/camera/found/warped%d.png"%idx, warped, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

    print("\n")

                
print('over')
cv2.waitKey(0)



