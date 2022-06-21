import cv2
import numpy as np

image = cv2.imread('e.jpg')
image_gray = cv2.imread('e.jpg', cv2.IMREAD_GRAYSCALE)

# cv2.imshow('image', image)
# cv2.imshow('image_gray', image_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

blur = cv2.GaussianBlur(image_gray, ksize=(5,5), sigmaX=0)
ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

"""
cv2.Canny(gray_img, threshold1, threshold2)
- threshold1 : 다른 엣지와의 인접 부분(엣지가 되기 쉬운 부분)에 있어 엣지인지 아닌지를 판단하는 임계값
- threshold2 : 엣지인지 아닌지를 판단하는 임계값

외곽선(엣지) 검출 파라미터 조정을 하는 방법
1. 먼저 threshold1와 threshold2를 같은 값으로 한다.
2. 검출되길 바라는 부분에 엣지가 표시되는지 확인하면서 threshold2 값을 조정한다.
3. 2번의 조정이 끝나면, threshold1를 사용하여 엣지를 연결시킨다.
"""
edged = cv2.Canny(blur, 10, 250)
# cv2.imshow('Edged', edged)
# cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
# cv2.imshow('closed', closed)
# cv2.waitKey(0)

contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
total = 0

contours_xy = np.array(contours)
contours_xy.shape

# x의 min과 max 찾기
x_min, x_max = 0, 0
value = list()
for i in range(len(contours_xy)):
    for j in range(len(contours_xy[i])):
        value.append(contours_xy[i][j][0][0])  # 네번째 괄호가 0일때 x의 값
        x_min = min(value)
        x_max = max(value)

# y의 min과 max 찾기
y_min, y_max = 0, 0
value = list()
for i in range(len(contours_xy)):
    for j in range(len(contours_xy[i])):
        value.append(contours_xy[i][j][0][1])  # 네번째 괄호가 0일때 x의 값
        y_min = min(value)
        y_max = max(value)

# image trim 하기
x = x_min
y = y_min
w = x_max-x_min
h = y_max-y_min

img_trim = image[y:y+h, x:x+w]
cv2.imwrite('e.jpg', img_trim)
org_image = cv2.imread('e.jpg')

cv2.imshow('org_image', org_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#--① 크로마키 배경 영상과 합성할 배경 영상 읽기
img1 = cv2.imread('e.jpg')
img2 = cv2.imread('./bg/bg1.jpg')

#--② ROI 선택을 위한 좌표 계산
height1, width1 = img1.shape[:2]
height2, width2 = img2.shape[:2]
x = (width2 - width1)//2
y = height2 - height1
w = x + width1
h = y + height1
print(x, y, w, h)

#--③ 크로마키 배경 영상에서 크로마키 영역을 10픽셀 정도로 지정
chromakey = img1[:10, :10, :]
offset = 20

#--④ 크로마키 영역과 영상 전체를 HSV로 변경
hsv_chroma = cv2.cvtColor(chromakey, cv2.COLOR_BGR2HSV)
hsv_img = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

#--⑤ 크로마키 영역의 H값에서 offset 만큼 여유를 두어서 범위 지정
# offset 값은 여러차례 시도 후 결정
#chroma_h = hsv_chroma[0]
chroma_h = hsv_chroma[:,:,0]
lower = np.array([chroma_h.min()-offset, 100, 100])
upper = np.array([chroma_h.max()+offset, 255, 255])

#--⑥ 마스크 생성 및 마스킹 후 합성
mask = cv2.inRange(hsv_img, lower, upper)
mask_inv = cv2.bitwise_not(mask)
roi = img2[y:h, x:w]
fg = cv2.bitwise_and(img1, img1, mask=mask_inv)
bg = cv2.bitwise_and(roi, roi, mask=mask)
img2[y:h, x:w] = fg + bg

#--⑦ 결과 출력
cv2.imshow('added', img2)
cv2.waitKey()
cv2.destroyAllWindows()