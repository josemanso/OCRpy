#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 17:13:25 2022

@author: josemo
"""

import cv2
import pytesseract
#import numpy as np
import imutils  # basic function image processing

# 

image = cv2.imread("/home/josemo/Documentos/imagenes/testcar.jpg")

# resize
#image = cv2.resize(image, (620,480) )
image = imutils.resize(image, width=300 )
cv2.imshow("original image", image)
cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grey scale
cv2.imshow("greyed image", gray)
cv2.waitKey(0)

#gray = cv2.medianBlur(gray, 5) #Blur to reduce noise
gray = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("greyed image", gray)
cv2.waitKey(0)
# detecting the edges of the smoothened image
edged = cv2.Canny(gray, 30, 200) #Perform Edge detection
cv2.imshow("edged image", edged)
cv2.waitKey(0)
# Finding the contours from the edged image
cnts,new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
image1 = image.copy()
#cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cnts = imutils.grab_contours(cnts)
#cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
#screenCnt = None
#img1=image.copy()
cv2.drawContours(image1,cnts,-1,(0,255,0),3)
cv2.imshow("image1",image1)
cv2.waitKey(0)

# Sorting the identified comtours
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
screenCnt = None #will store the number plate contour
image2 = image.copy()

cv2.drawContours(image2,cnts,-1,(0,255,0),3) 
cv2.imshow("Top 30 contours",image2) #top 30 contours
cv2.waitKey(0)

# Finding the contours with four sides
#count=0
idx=7
# loop over contours
for c in cnts:
  # approximate the contour
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
        if len(approx) == 4: #chooses contours with 4 corners
                screenCnt = approx
                # Cropping the rectangular part identifie as license plate
                x,y,w,h = cv2.boundingRect(c) #finds co-ordinates of the plate
                new_img=image[y:y+h,x:x+w]
                cv2.imwrite('./'+str(idx)+'.png',new_img) #stores the new image
                idx+=1
                break
#draws the selected contour on original image        
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
cv2.imshow("Final image with plate detected",image)
cv2.waitKey(0)

# Extracting text from the image of the cropped license plate
Cropped_loc='./7.png' #the filename of cropped image
cv2.imshow("cropped",cv2.imread(Cropped_loc))
#plate = pytesseract.image_to_string(Cropped_loc, lang='eng')
#pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe" #exe file for using ocr 

text=pytesseract.image_to_string(Cropped_loc,lang='eng') #converts image characters to string
print("Number is:" ,text)
cv2.waitKey(0)
cv2.destroyAllWindows() 
