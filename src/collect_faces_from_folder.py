from PIL import Image
import glob

import face_recognition

face_names = []
known_faces = []

for filename in glob.glob('img/*.jpg'):
    new_img = face_recognition.load_image_file(filename)
    new_face = face_recognition.face_encodings(new_img)[0]

    known_faces.append(new_face)
    face_names.append(filename.split("/")[1])

unknown_image = face_recognition.load_image_file("img/unknown_images/Putin2.jpg")
unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces(known_faces, unknown_face_encoding)
print(face_names)
print(results)
