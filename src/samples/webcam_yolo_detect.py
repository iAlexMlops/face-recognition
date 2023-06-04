import torch
from ultralytics import YOLO
import numpy as np
import cv2

video_url = "Video.mov"

# Создание объекта VideoCapture с ссылкой на видео
cap = cv2.VideoCapture(video_url)

mod_p = '../../crowdhuman_yolov5m.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=mod_p, device='mps')

while True:

    ret, frame = cap.read()

    results = model(frame)
    result = results[0]

    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")

    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
        cv2.putText(frame, str(cls), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
