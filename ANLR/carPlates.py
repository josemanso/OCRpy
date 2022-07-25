#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 16:04:32 2022

@author: josemo
"""

import cv2
#import time
#import numpy as np

# Create our body classifier; ('haarcascade_russian_plate_number.xml')
#car_classifier = cv2.CascadeClassifier('/home/josemo/opencv/data_trained/data/haarcascades/haarcascade_car.xml')
plate_classifier = cv2.CascadeClassifier('/home/josemo/opencv/data_trained/data/haarcascades/haarcascade_russian_plate_number.xml')
# Initiate video capture for video file
#cap = cv2.VideoCapture('/home/josemo/Documentos/imagenes/image_examples_cars')#'/cars.avi')
cap = cv2.VideoCapture('/home/josemo/Documentos/imagenes/vid.mp4')#

if(cap.isOpened()==False):
    print('Error reading video')
    


# Loop once video is successfully loaded
#while cap.isOpened():
while True:
    #time.sleep(.05)
    # Read first frame
    ret, frame= cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Pass frame to our car classifier
    #cars = car_classifier.detectMultiScale(gray, 1.4, 2)
    plate = plate_classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5,
                                              minSize=(25,25))
    
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in plate: #cars:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        #frame[y:y+h,x:x+w] = cv2.blur(frame[y:y+h, x:x+w], ksize=(10,10))
        frame[y:y+h,x:x+w] = cv2.blur(frame[y:y+h, x:x+w], ksize=(20,20))      
     
        cv2.putText(frame, text='License Plate', org=(x-3, y-3), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    color=(0.0,255), thickness=1,fontScale=0.8)
        #cv2.imshow('Cars', frame)
    if ret == True:
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) == 13:  # 13 is the Enter Key
            break
    else:
        #cap.release()
        break
cap.release()
cv2.destroyAllWindows()