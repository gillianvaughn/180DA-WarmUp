import numpy as np
import cv2

cap = cv2.VideoCapture(0)
while(1):

    # Take each frame
    _, frame = cap.read()

    # define range of blue color in RGB
    lower_blue = np.array([100,0,0])  
    upper_blue = np.array([255,100,100])

    # Threshold the RGB image to get only blue colors
    mask = cv2.inRange(frame, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    bluecnts = cv2.findContours(mask.copy(),
                              cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(bluecnts)>0:
        blue_area = max(bluecnts, key=cv2.contourArea)
        (xg,yg,wg,hg) = cv2.boundingRect(blue_area)
        cv2.rectangle(frame,(xg,yg),(xg+wg, yg+hg),(0,255,0),2)

    cv2.imshow('frame',frame)
    #cv2.imshow('mask',mask)
    #cv2.imshow('res',res)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

"""
Reference: 
    BoundingBox: https://answers.opencv.org/question/200861/drawing-a-rectangle-around-a-color-as-shown/?fbclid=IwAR3_P7pmzYS7Mw8sSBtK5d7szhg-RNjCkZzlrln65rmqrbxqHm10JTxzG1I
    Tracking Blue Object: https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Changing_ColorSpaces_RGB_HSV_HLS.php
"""