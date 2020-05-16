#!/usr/bin/env python
#-*- coding:utf-8 -*-
import cv2


image = cv2.imread("/home/pi/camera/1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
srcWidth, srcHeight, channels = image.shape
print(srcWidth, srcHeight)

binary = cv2.medianBlur(gray,7)
ret, binary = cv2.threshold(binary, 50, 255, cv2.THRESH_BINARY)
cv2.imwrite("1-threshold.png", binary, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

# canny提取轮廓
binary = cv2.Canny(binary, 0, 60, apertureSize = 3)
cv2.imwrite("3-canny.png", binary, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

# 提取轮廓后，拟合外接多边形（矩形）
_,contours,_ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("len(contours)=%d"%(len(contours)))

for idx,c in enumerate(contours):
    #if len(c) < binary.min_contours:
        #continue
    print("1")
    #epsilon = binary.epsilon_start[1]
    while True:
        approx = cv2.approxPolyDP(c,binary.epsilon_start,True)
        print("idx,epsilon,len(approx),len(c)=%d,%d,%d,%d"%(idx,epsilon,len(approx),len(c)))
        if (len(approx) < 4):
            break
        if math.fabs(cv2.contourArea(approx)) > Config.min_area:
            if (len(approx) > 4):
                epsilon += binary.epsilon_step
                print("epsilon=%d, count=%d"%(epsilon,len(approx)))
                continue
            else:
                #for p in approx:
                #    cv2.circle(binary,(p[0][0],p[0][1]),8,(255,255,0),thickness=-1)
                approx = approx.reshape((4, 2))
                # 点重排序, [top-left, top-right, bottom-right, bottom-left]
                src_rect = order_points(approx)

                cv2.drawContours(image, c, -1, (0,255,255),1)
                cv2.line(image, (src_rect[0][0],src_rect[0][1]),(src_rect[1][0],src_rect[1][1]),color=(100,255,100))
                cv2.line(image, (src_rect[2][0],src_rect[2][1]),(src_rect[1][0],src_rect[1][1]),color=(100,255,100))
                cv2.line(image, (src_rect[2][0],src_rect[2][1]),(src_rect[3][0],src_rect[3][1]),color=(100,255,100))
                cv2.line(image, (src_rect[0][0],src_rect[0][1]),(src_rect[3][0],src_rect[3][1]),color=(100,255,100))

                # 获取最小矩形包络
                rect = cv2.minAreaRect(approx)
                # rect = cv2.maxAreaRect(approx)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                box = box.reshape(4,2)
                box = order_points(box)
                print("approx->box")
                print(approx)
                print(src_rect)
                print(box)
                w,h = point_distance(box[0],box[1]), \
                      point_distance(box[1],box[2])
                print("w,h=%d,%d"%(w,h))

                dst_rect = np.array([
                    [0, 0],
                    [w - 1, 0],
                    [w - 1, h - 1],
                    [0, h - 1]],
                    dtype="float32")
                M = cv2.getPerspectiveTransform(src_rect, dst_rect)
                warped = cv2.warpPerspective(image, M, (w, h))
                cv2.imwrite("transfer%d.png"%idx, warped, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
                break


