import cv2 
import numpy as np
face_cascade = cv2.CascadeClassifier(r'C:\Users\lui33\Downloads\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture (r'C:\Users\lui33\Desktop\video.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray) 
    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0) , 5) # face detection frame size and colour
        roi_gray = gray[y:y+h, x:x+w]
        roi_color =frame[y:y+h, x:x+w]
    cv2.imshow('original', frame)
    cv2.imshow('fg', fgmask)
    
    k= cv2.waitKey(30)
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()
