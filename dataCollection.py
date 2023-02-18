import cv2
import math
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import time
cap=cv2.VideoCapture(0)#0 parameter is for accessing the webcamera
detector=HandDetector(maxHands=1)#1 hand can only be detected at once
offset=20#extra space for the imgCrop
imgSize=300#pixel size of our custom image
counter=1
folder="images/r"
while True:
    success,img=cap.read()
    hands,img=detector.findHands(img)
    if hands:
        hand=hands[0]#hands variable can store any no of hand
        x,y,w,h =hand['bbox']#boundary box
        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255#np.ones creates a numpy array 
        #of size imgSize*imgSize and stores value 1 in it 

        imgCrop=img[y-offset:y+h+offset,x-offset:x+h+offset]#offset value is used to get
        # more space around hands in the image
        
        #putting the crop image into the white image
        imgCropShape=imgCrop.shape
        aspectRatio=h/w
        if aspectRatio>1:
            k=imgSize/h
            wCal=math.ceil(k*w)
            imgResize=cv2.resize(imgCrop,(wCal,imgSize ))
            imgResizeShape=imgResize.shape
            wGap=math.ceil((imgSize-wCal)/2)
            imgWhite[:,wGap:wCal+wGap]=imgResize
        else:
            k=imgSize/w
            hCal=math.ceil(k*h)
            imgResize=cv2.resize(imgCrop,(imgSize,hCal ))
            imgResizeShape=imgResize.shape
            hGap=math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap,:]=imgResize


        cv2.imshow("ImageCrop",imgCrop)#display the cropped image
        cv2.imshow("ImageWhite",imgWhite)
    cv2.imshow("Image",img)
    key=cv2.waitKey(1) #wait for 1 ms
    if key==ord("s"):
        counter+=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)

