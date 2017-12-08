import cv2
import numpy as np 

face_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_mcs_nose.xml')

if face_cascade.empty():
    raise IOError('Unable to load haar xml')

if eye_cascade.empty():
    raise IOError('Unable to load haar eye xml')

if nose_cascade.empty():
    raise IOError('Unable to load haar nose xml')


cap = cv2.VideoCapture(0)

scaling_factor = 0.5

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in face_rects:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eye_rects = eye_cascade.detectMultiScale(roi_gray)
        nose_rects = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)

        for (x_eye, y_eye, w_eye, h_eye) in eye_rects:
            center =(int(x_eye + 0.5 * w_eye), int(y_eye + 0.5 * h_eye))
            radius = int(0.3 * (w_eye + h_eye))
            color = (0, 255, 0)
            thickness = 3
            cv2.circle(roi_color, center, radius, color, thickness)

        for (x_nose, y_nose, w_nose, h_nose) in nose_rects:
            cv2.rectangle(roi_color, (x_nose, y_nose), (x_nose + w_nose, y_nose + h_nose), (0, 255, 0 ), 3)
            break

#    cv2.imshow('WebCam', frame)
    cv2.imshow('Face Eye nose Detector',frame)
 
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()