import cv2
import numpy as np

cap = cv2.VideoCapture(0)
background = cv2.imread('./image.jpg')

while cap.isOpened():
    ret, current_frame = cap.read()
    if ret:
        hsv_frame = cv2.cvtColor(current_frame,cv2.COLOR_BGR2HSV)

        l_red = np.array([45,0,45])
        u_red = np.array([101,255,255])
        mask1 = cv2.inRange(hsv_frame, l_red,u_red)

        # l_red = np.array([200,200,000])
        # u_red = np.array([255,255,255])
        # mask2 = cv2.inRange(hsv_frame, l_red,u_red)

        red_mask = mask1 #+ mask2

        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations = 10) 
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, np.ones((3,3), np.uint8), iterations = 1)  

        part1 = cv2.bitwise_and(background,background,mask=red_mask)

        red_mask_inv = cv2.bitwise_not(red_mask)
        part2 = cv2.bitwise_and(current_frame,current_frame,mask=red_mask_inv)

        final = part1 + part2

        cv2.imshow("red mask", final )
        if cv2.waitKey(5) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()