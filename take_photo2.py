import cv2


cap = cv2.VideoCapture(0)

scaling_factor = 1

i = 0
take_name = "john"
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Video Record',frame)

    c = cv2.waitKey(1)
    if c == 27:
        break
    if c == 112:
        filename = '{}{:0>6d}.jpg'.format(take_name,i)
        cv2.imwrite(filename, frame)
        print('Frame {} is captured.'.format(i))
        i = i + 1    

cap.release()
cv2.destroyAllWindows()