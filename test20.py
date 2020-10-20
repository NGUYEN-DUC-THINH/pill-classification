import numpy as np
from sklearn.svm import SVC
import pickle
import cv2
from skimage.feature import hog
import os
from img_proc import *
import snap7.client as c
from snap7.util import *
from snap7.snap7types import *

plc = c.Client()
plc.connect('192.168.0.1',0,1)

model = pickle.load(open('data_folder\SVM_new_version44.mod','rb'))
kernel = np.ones((5,5),np.uint8)
cap = cv2.VideoCapture(0) 
while True:
    
    ret,frame = cap.read()
    if ret is not True:
        break
    
    frame = frame[50:420,:]
    gaussian = cv2.GaussianBlur(frame, (21, 21), 0)
    gray = cv2.cvtColor(gaussian,cv2.COLOR_BGR2BGRA)
    edges = auto_canny(gaussian)
    dilation = cv2.dilate(edges,kernel,iterations = 1)
    lines = cv2.HoughLinesP(dilation , 1, np.pi/180, 0, maxLineGap=8 )
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(edges, (x1, y1), (x2, y2), (255, 255, 255), 2)
    contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL ,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if (cv2.contourArea(cnt) > 2000):
            x,y = center(cnt)
            rect = cv2.minAreaRect(cnt)
            if x > 230 and x < 370:
                img = crop_minAreaRect(frame, rect)
                if img.shape[0]<img.shape[1]:
                    img =  cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                img = cv2.resize(img,(60,120))

                fd = hog(img, orientations=8,pixels_per_cell=(16,16),
                         cells_per_block=(2, 2), visualize= False,
                         transform_sqrt=False,feature_vector=True,
                         multichannel=True)
                test = model.predict(np.array([fd]))
                
                
                if test[0] == 1:
                    color = (255,0,0)
                else:
                    color = (0,0,255)
                draw_ret(cnt,color,frame)
                if test[0] == 0 and ReadMemory(plc,300,0,S7WLWord)== 1:
                    WriteMemory(plc,100,0,S7WLWord,1)
                else:
                    WriteMemory(plc,100,0,S7WLWord,0)
    cv2.imshow('frame',frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
