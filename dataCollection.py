import time

import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector #Detect Hands
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20 #Dabba jo bana aa raha uska size adjust krne k waaste
imgSize = 300
folder = "Data/D"
counter = 0
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img) #Detect Hands
    # To Crop Hands
    if hands:
        hand = hands[0] #0 means number of hands  = 1
        x, y, w, h = hand['bbox']

        #To adjust width of box jo bana aa raha hands k upr
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset] # To Crop Hands

        imgCropShape = imgCrop.shape

        aspectRatio = h/w #Adjust range of white space height/weidth
        #Adjust Height
        if aspectRatio>1:
            k = imgSize/h
            wCal = math.ceil(k*w) #ceil number ko roundoff krne k waaste
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
        #Adjust Width
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)  # ceil number ko roundoff krne k waaste
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop) # To Crop Hands
        cv2.imshow("ImageWhite", imgWhite) #To adjust size of box
    cv2.imshow("image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter +=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
