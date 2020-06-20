# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 09:57:41 2020

@author: dell
"""
# -*- coding: utf-8 -*-
from sklearn.ensemble import IsolationForest
from imutils import paths
import numpy as np
import cv2
from os import listdir
import os

def quantify_image(img, bins=(4,6,3)):
    hist = cv2.calcHist([img], [0,1,2], None, bins,[0,180, 0,256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    return hist

def load_dataset(path, bins):
    images=[]
    for img in listdir(path):
        images.append(img)
    data = []
    for img in images:
        image = cv2.imread(path+'/'+img)
        print(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = quantify_image(image, bins)
        data.append(features)
    return np.array(data)

### TRAIN MODEL

os.chdir(r'C:\Users\dell\Desktop\Anomaly Detection\ML\intro-anomaly-detection')
path='./forest'
data = load_dataset(path, bins=(3,3,3))

model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
model.fit(data)

### TEST MODEL
test_img = './examples/coast_osun52.jpg'
test_img = cv2.imread(test_img)
image = cv2.cvtColor(test_img, cv2.COLOR_BGR2HSV)
features = quantify_image(image, bins=(3,3,3))

preds = model.predict([features])[0]

label = "anomaly" if preds == -1 else "normal"
color = (0, 0, 255) if preds == -1 else (0, 255, 0)

# draw the predicted label text on the original image
cv2.putText(test_img, label, (10,  25), cv2.FONT_HERSHEY_SIMPLEX,
	0.7, color, 2)

# display the image

cv2.imshow("Output", test_img)
cv2.waitKey(0)