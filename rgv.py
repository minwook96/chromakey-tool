import cv2
import numpy as np
from matplotlib import pyplot as plt

## 조도 값 추출
origin_image = cv2.imread('output/image1.jpg')
hsv_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2HSV)

image1 = cv2.imread('output/image2.jpg')
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)

# image2 = cv2.imread('output/image3.jpg')
# image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

# val = 100
# array1 = np.full(hsv_image.shape, (0, 0, val), dtype=np.uint8)
# array2 = np.full(hsv_image.shape, (0, 50, 0), dtype=np.uint8)
#
# val_add_image = cv2.add(hsv_image, array1)
# saturation_add_image = cv2.add(hsv_image, array2)

print('BGR : \t', origin_image[55, 116, :])
print('hsv : \t', hsv_image[55, 116, :])
print('hsv 밝기(v) 증가 : \t', image1[55, 116, :])
# print('hsv 채도(s) 증가 : \t', image2[55, 116, :])

val_add_image = cv2.cvtColor(image1, cv2.COLOR_HSV2BGR)
# saturation_add_image = cv2.cvtColor(image2, cv2.COLOR_HSV2BGR)

image = np.concatenate((origin_image, val_add_image), axis=1)
# image = np.concatenate((ima도ge, saturation_add_image), axis=1)
image = cv2.resize(image, (1500, 720))
cv2.imshow('add, subtract', image)
cv2.waitKey(0)





## 특정 부분 이미지 값 변경
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# origin_image = cv2.imread('puppy.jpg')
# hsv_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2HSV)
#
# val = 255
#
# mask = hsv_image[:50,:100,:]
#
# array = np.full(mask.shape, (0,0,val), dtype=np.uint8)
#
# sub_mask_image = cv2.subtract(mask, array)
#
# hsv_image[:50,:100,:] = sub_mask_image
#
# val_sub_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
#
# print("원본 bgr \t:",origin_image[0,0,:])
# print("변환 bgr \t:",val_sub_image[0,0,:])
#
# image = np.concatenate((origin_image,val_sub_image), axis=1)
#
# cv2.imshow('add, subtract', image)
# cv2.waitKey(0)