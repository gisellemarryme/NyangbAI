import numpy as np
import cv2 as cv
import math

img1=cv.imread('go1.jpg',cv.IMREAD_GRAYSCALE)
img2=cv.imread('go2.jpg',cv.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print('file load failed')

edge1=cv.Canny(img1,50,150) 
edge2=cv.Canny(img2,50,150) 

lines=cv.HoughLines(edge1,1,math.pi/180,250) #img1에서 직선 검출해서 img2에 복붙할 것

result_img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
result_img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)

if lines is not None:
    for i in range(lines.shape[0]):
        rho=lines[i][0][0]
        theta=lines[i][0][1]
        cos_t=math.cos(theta)
        sin_t=math.sin(theta)
        x0,y0=rho*cos_t,rho*sin_t
        alpha=1000
        pt1=(int(x0-alpha*sin_t),int(y0+alpha*cos_t))
        pt2=(int(x0+alpha*sin_t),int(y0-alpha*cos_t))
        cv.line(result_img1,pt1,pt2,(0,255,0),2,cv.LINE_AA)

if lines is not None:
    for line in lines:
        rho, theta = line[0]
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        x0, y0 = rho * cos_t, rho * sin_t
        alpha = 1000
        pt1 = (int(x0 - alpha * sin_t), int(y0 + alpha * cos_t))
        pt2 = (int(x0 + alpha * sin_t), int(y0 - alpha * cos_t))
        cv.line(result_img2, pt1, pt2, (0, 255, 0), 2, cv.LINE_AA) 

blurred1=cv.blur(img1,(3,3))
circles1=cv.HoughCircles(img1,cv.HOUGH_GRADIENT,1,15,param1=250,param2=70) #블러를 제거하면 더 잘 되려나? >> 아님
#해상도 비율, 최대 반지름, 임계값1(작은 값 입력 >> 더 많은 엣지 검출), 임계값2(작은 값 입력 >> 더 많은 검출)
#20,150,70 >> 검은 돌 4개 일치
#10,150,70 >> 검은 돌 6개 일치
#1,250,70 >> 검은 돌 6개 일치, 지금까지 중에서는 제일 깔끔
min_radius=1
max_radius=15 # 이 방법 쓰면 검은 돌만 완벽하게 나옴 

if circles1 is not None:
    circles1=np.uint16(np.around(circles1)) #circles 배열의 모든 값을 가장 가까운 정수로 반올림(int 대신 쓴 것)
    for i in range(circles1.shape[1]):
        cx,cy,radius=circles1[0][i]
        if min_radius<=radius<=max_radius:
            cv.circle(result_img1,(cx,cy),int(radius),(0,0,255),2,cv.LINE_AA) #검은 돌 파란색으로 검출

# 흰색 돌 검출을 위한 적응적 임계값
thresh1 = cv.adaptiveThreshold(blurred1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)

# 원 검출
circles_1 = cv.HoughCircles(thresh1, cv.HOUGH_GRADIENT, 1, 15, param1=100, param2=20, minRadius=1, maxRadius=15)

# 흰 돌 검출 
if circles_1 is not None:
    circles = np.uint16(np.around(circles_1)) #circles 배열의 모든 값을 가장 가까운 정수로 반올림(int 대신 쓴 것)
    for i in circles[0, :]:
        cv.circle(result_img1,(i[0],i[1]),int(i[2]),(255,0,0),2,cv.LINE_AA) #흰 돌 파란색으로 검출

blurred2=cv.blur(img2,(3,3))
circles2=cv.HoughCircles(img2,cv.HOUGH_GRADIENT,1,15,param1=250,param2=70)

# 흰색 돌 검출을 위한 적응적 임계값
thresh2 = cv.adaptiveThreshold(blurred2, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
circles_2 = cv.HoughCircles(thresh2, cv.HOUGH_GRADIENT, 1, 20, param1=120, param2=20, minRadius=4, maxRadius=15)
#이미 검출된 검은 돌도 같이 표시됨 >> 해결 필요
#minRadius를 1에서 8로 올리니까 됨

# 흰 돌 검출 
if circles_2 is not None:
    circles = np.uint16(np.around(circles_2))
    for i in circles[0, :]:
        cv.circle(result_img2,(i[0],i[1]),int(i[2]),(255,0,0),2,cv.LINE_AA) #흰 돌 파란색으로 검출

# 검은 돌 검출
if circles2 is not None:
    circles2=np.uint16(np.around(circles2)) #circles 배열의 모든 값을 가장 가까운 정수로 반올림(int 대신 쓴 것)
    for i in range(circles2.shape[1]):
        cx,cy,radius=circles2[0][i]
        if min_radius<=radius<=max_radius:
            cv.circle(result_img2,(cx,cy),int(radius),(0,0,255),2,cv.LINE_AA)

cv.imshow('img1',result_img1)
cv.imshow('img2',result_img2)

cv.waitKey()
cv.destroyAllWindows()
