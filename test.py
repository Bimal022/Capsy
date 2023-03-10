import time
import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector #Detect Hands
from cvzone.ClassificationModule import Classifier
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20 #Dabba jo bana aa raha uska size adjust krne k waaste
imgSize = 300
folder = "Data/C"
counter = 0
labels = ["A", "B", "C"]
while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img= detector.findHands(img) #Detect Hands
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
            prediction, index = classifier.getPrediction(imgWhite, draw = False)
            print(prediction, index)

        #Adjust Width
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)  # ceil number ko roundoff krne k waaste
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw = False)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x-offset+90,y - offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y-27), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 0, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (255, 0, 255), 4)
        cv2.imshow("ImageCrop", imgCrop) # To Crop Hands
        cv2.imshow("ImageWhite", imgWhite) #To adjust size of box
    cv2.imshow("image", imgOutput)
    key = cv2.waitKey(1)
