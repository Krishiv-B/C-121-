import cv2, time
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*"XVID")
output_file = cv2.VideoWriter("output.avi", fourcc, 20.0, (640, 480))

cap = cv2.VideoCapture(0)

time.sleep(2)
bg = 0

for i in range(60):
    ret, bg = cap.read()

bg = np.flip(bg, axis=1)

while(cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break
    img = np.flip(img, axis = 1)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    lower_gray = np.array([0, 5, 50], np.uint8)
    upper_gray = np.array([179, 50, 255], np.uint8)
    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
    img_res = cv2.bitwise_and(img, img, mask = mask_gray)


    # mask_1 = mask_1 + mask_2

    # mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    # mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_DILATE, np.ones((3,3), np.uint8))

    # mask_2 = cv2.bitwise_not(mask_1)

    # res_1 = cv2.bitwise_and(img, img, mask=mask_2)

    # res_2 = cv2.bitwise_and(bg, bg, mask=mask_1)

    # final_output = cv2.addWeighted(res_1, 1, res_2, 1, 0)
    output_file.write(img_res)
    cv2.imshow("magic", img_res)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()