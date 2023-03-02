import cv2
import math
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import time
from cvzone.ClassificationModule import Classifier
cap=cv2.VideoCapture(0)#0 parameter is for accessing the webcamera
detector=HandDetector(maxHands=1)#1 hand can only be detected at once
offset=20#extra space for the imgCrop
imgSize=300#pixel size of our custom image
classifier =Classifier("model/keras_model.h5","model/labels.txt")#model  trained using google teachable machine tools and the lables 
labels=["A","B","C"]#lables for displaying the ans 
while True:
    success,img=cap.read()
    imgOutput=img.copy()#taking copy of image because we do not want to show the lines in the hands in the output image 
    hands,img=detector.findHands(img)#function finds the hands
    if hands:
        hand=hands[0]#hands variable can store any no of hand
        x,y,w,h =hand['bbox']#boundary box gives the dimentions and position of the image created
        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255#np.ones creates a numpy array 
        #of size imgSize*imgSize and stores value 1 in it 

        imgCrop=img[y-offset:y+h+offset,x-offset:x+w+offset]#offset value is used to get
        # more space around hands in the image
        
        #putting the crop image into the white image
        imgCropShape=imgCrop.shape
        #below code is used to get the image in  the between of the frame and to make the 
        #so that we can make the correct prediction 
        #
        aspectRatio=h/w

        if aspectRatio>1:#if the hieght is greater than the width
            k=imgSize/h#the times image is compressed to get the height of 300px
            wCal=math.ceil(k*w)#compressing the width the same as hieght so that the aspect ratio is maintained
            imgResize=cv2.resize(imgCrop,(wCal,imgSize ))#resizing the image according to the given pixels 
            imgResizeShape=imgResize.shape
            wGap=math.ceil((imgSize-wCal)/2)#calculating the gap in the both side of the image to  make the image in the center
            imgWhite[:,wGap:wCal+wGap]=imgResize#putting the image in the to of imagewhite to create a square size image
            prediction ,index =classifier.getPrediction(imgWhite,draw="False")#getting the prediction vector and the  predicted index value  
            print(prediction,index)
        else:
            k=imgSize/w
            hCal=math.ceil(k*h)
            imgResize=cv2.resize(imgCrop,(imgSize,hCal ))
            imgResizeShape=imgResize.shape
            hGap=math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap,:]=imgResize
            prediction ,index =classifier.getPrediction(imgWhite,draw="False")
        cv2.rectangle(imgOutput,(x-offset,y-offset-50),(x-offset+90,y-offset),(255,0,0),cv2.FILLED)#putting the lables inside the 
        cv2.putText(imgOutput,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(25,25,25),2)
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,0),4)
        cv2.imshow("ImageCrop",imgCrop)#display the cropped image
        cv2.imshow("ImageWhite",imgWhite)
    cv2.imshow("Image",imgOutput)
    cv2.waitKey(1) #wait for 1 ms

