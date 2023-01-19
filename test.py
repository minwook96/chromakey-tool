import cv2

for i in range(1, 51):
    img1 = cv2.imread(f"tc3/bg/bg_{i}.jpg")

    dst = cv2.resize(img1, (1920, 1080))

    cv2.imwrite(f"bg/bg_{i}.jpg", dst)