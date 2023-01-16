import numpy as np
import cv2
import pickle
from keras.models import load_model

threshold = 0.65 
cap1 = cv2.imread('5.png')
cap2 = cv2.imread('6.png')
cap3 = cv2.imread('3.png')

#### LOAD THE TRAINNED MODEL
'''pickle_in = open("model_trained_10.p","rb")
model = pickle.load(pickle_in)'''

model = load_model('OCR_Digits.h5')
 
#### PREPORCESSING FUNCTION
def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img
 
def img(cap1):    
    img = np.asarray(cap1)
    img = cv2.resize(img,(32,32))
    img = preProcessing(img)
    cv2.imshow("Processsed Image",img)
    img = img.reshape(1,32,32,1)
        #### PREDICT
    classIndex = int(model.predict_classes(img))
        #print(classIndex)
    predictions = model.predict(img)
        #print(predictions)
    probVal= np.amax(predictions)
    print(classIndex,probVal)
     
    if probVal> threshold:
        cv2.putText(cap1,str(classIndex) + "   "+str(probVal), (50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
     
    cv2.imshow("Original Image",cap1)
    cv2.waitKey(0)

img(cap1)
img(cap2)
img(cap3)