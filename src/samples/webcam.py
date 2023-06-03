from ultralytics import YOLO
import numpy as np
import cv2


cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
