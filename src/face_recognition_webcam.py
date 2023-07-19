import face_recognition
import cv2
import numpy as np
from feast import FeatureStore
from ultralytics import YOLO
import torch

video_url = "../some_girl.mp4"

# Создание объекта VideoCapture с ссылкой на видео
# video_capture = cv2.VideoCapture(0)
video_capture = cv2.VideoCapture(video_url)

store = FeatureStore(repo_path="../feast")
df = store.get_saved_dataset(name="faces_dataset").to_df()

fei_indexes = ['face_id', 'event_timestamp', 'image_name']
features_indexes = [f'feature_{i}' for i in range(1, 129)]
new_order = fei_indexes + features_indexes

df = df.reindex(columns=new_order)

known_face_names = df['face_id'].tolist()

columns_to_drop = ['face_id', 'event_timestamp', 'image_name']
df = df.drop(columns=columns_to_drop)

known_face_encodings = df.values.tolist()


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
face_ids_set = set()
face_ids_dict = dict()

process_this_frame = True

mod_p = '../models/crowdhuman_yolov5m.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=mod_p, device='mps')

frame_skipper = 0

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    if not ret:
        continue

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
                    face_ids_set.add(name)
                    if face_ids_dict.get(name) is None:
                        face_ids_dict[name] = 0
                    else:
                        face_ids_dict[name] = face_ids_dict[name] + 1

                face_names.append(name)

    if frame_skipper >= 2:
        frame_skipper = -1

    frame_skipper += 1

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

        result_image = cv2.imread(f"../update_images/{name}.jpg")
        cv2.imshow(f"{name}.jpg", result_image)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
