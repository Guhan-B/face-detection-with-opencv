import cv2 as cv
import numpy as np

faceCascade = cv.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eyeCascade =  cv.CascadeClassifier("haarcascades/haarcascade_eye.xml")

webcamInput = cv.VideoCapture(0)

webcamInput.set(3,550)
webcamInput.set(4,550)
webcamInput.set(10,100)

while True:
    Success, img = webcamInput.read()
    imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(imgGray,1.1,2)

    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
        cv.putText(img,"face",(x,y-20),cv.QT_FONT_NORMAL,0.5,(0,255,0),1)

    cv.imshow("gray scale",imgGray)
    cv.imshow("Video",img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# img = cv.imread("res/nasa.jpg")
# imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#
# faces = faceCascade.detectMultiScale(imgGray,1.1,2)
#
# print("{0} faces detected".format(len(faces)))
# print(faces)
#
# for (x,y,w,h) in faces:
#     cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
#
# cv.imshow("faces",img)

cv.waitKey(0)
