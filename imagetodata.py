#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 16:58:11 2022

@author: josemo
"""

import cv2
import pytesseract
from pytesseract import Output

image = cv2.imread("/home/josemo/Documentos/imagenes/invoice-sample.jpg")

d = pytesseract.image_to_data(image,output_type=Output.DICT)
n_boxes = len(d['level'])
for i in range(n_boxes):
    (x,y,w,h) = (d['left'][i], d['top'][i], d['width'][i],d['height'][i])
    cv2.rectangle(image, (x,y),(x+w,y+h), (0,255,0),2)
    
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
    