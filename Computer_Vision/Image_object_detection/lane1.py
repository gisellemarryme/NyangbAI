import numpy as np
import cv2 as cv
import math


img1=cv.imread('lane1.jpeg',cv.IMREAD_COLOR)
img2=cv.imread('lane2.jpeg',cv.IMREAD_COLOR)


if img1 is None:
    print('file load failed')


if img2 is None:
    print('file load failed')


gray1=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
gray2=cv.cvtColor(img2,cv.COLOR_BGR2GRAY)


edge1=cv.Canny(gray1,50,150)
lines1=cv.HoughLines(edge1,1,math.pi/180,140) #겹친 게 140개 이상인 아이만 


edge2=cv.Canny(gray2,50,150)
lines2=cv.HoughLines(edge2,1,math.pi/180,190)


if lines1 is not None:
    for i in range(lines1.shape[0]):
        rho = lines1[i][0][0]
        theta = lines1[i][0][1]


        # 각도 계산
        angle = np.degrees(theta) % 180
        # print('angle',angle) #angle 56.999996185302734, angle 122.0
        
        if 56 <= angle <= 57 or angle == 122.0:
            # print('angle', angle)  # 출력


            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            x0, y0 = rho * cos_t, rho * sin_t
            alpha = 1000
            pt1 = (int(x0 - alpha * sin_t), int(y0 + alpha * cos_t))
            pt2 = (int(x0 + alpha * sin_t), int(y0 - alpha * cos_t))
            cv.line(img1, pt1, pt2, (0, 0, 255), 2, cv.LINE_AA)


if lines2 is not None:
    for i in range(lines2.shape[0]):
        rho = lines2[i][0][0]
        theta = lines2[i][0][1]


        # 각도 계산
        angle = np.degrees(theta) % 180
        # print('angle',angle) #angle 51.999996185302734, angle 120.0
        
        if angle == 53.999996185302734 or angle == 120.0:
            print('angle', angle)  # 출력


            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            x0, y0 = rho * cos_t, rho * sin_t
            alpha = 1000
            pt1 = (int(x0 - alpha * sin_t), int(y0 + alpha * cos_t))
            pt2 = (int(x0 + alpha * sin_t), int(y0 - alpha * cos_t))
            cv.line(img2, pt1, pt2, (0, 0, 255), 2, cv.LINE_AA)


cv.imshow('img1',img1)
cv.imshow('img2',img2)


cv.waitKey()
cv.destroyAllWindows()
