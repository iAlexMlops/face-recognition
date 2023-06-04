import face_recognition
import cv2
import numpy as np
from ultralytics import YOLO
import torch

video_url = "Video.mov"

# Создание объекта VideoCapture с ссылкой на видео
video_capture = cv2.VideoCapture(0)

putin_image = face_recognition.load_image_file("img/known_images/Putin.jpg")
putin_face_encoding = face_recognition.face_encodings(putin_image)[0]

egorov_image = face_recognition.load_image_file("img/known_images/Egorov.jpg")
egorov_face_encoding = face_recognition.face_encodings(egorov_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    putin_face_encoding,
    egorov_face_encoding
]
known_face_names = [
    "Vladimir Putin",
    "Alex Egorov"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

mod_p = '../models/crowdhuman_yolov5m.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=mod_p, device='mps')

frame_skipper = 0

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # frame = cv2.imread("img/Egorov.jpg")
    # Only process every other frame of video to save time
    if frame_skipper == 0:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        results = model(rgb_small_frame)

        df = results.pandas().xyxy[0]
        result = df[df["class"] == 1]

        # print(df[df["class"] == 1])
        # print(result[["xmin", "ymin", "xmax", "ymax"]])

        bboxes = np.array(result[["xmin", "ymin", "xmax", "ymax"]], dtype="int")
        # print(bboxes)

        if len(bboxes) == 0:
            continue
        else:
            face_locations = []
            for bbox in bboxes:

                (x1, y1, x2, y2) = bbox

                top = y1
                right = x2
                bottom = y2
                left = x1

                face_locations.append(tuple([top, right, bottom, left]))

            # Find all the faces and face encodings in the current frame of video
            # face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

    if frame_skipper >= 2:
        frame_skipper = -1

    frame_skipper +=1

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
