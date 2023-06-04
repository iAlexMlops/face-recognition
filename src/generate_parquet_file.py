import os
import face_recognition
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime


def process_images():
    image_folder = "/Users/alexegorov/PycharmProjects/faceRecognition/images"
    faces_data = []

    face_id = 0
    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)

        # Загрузка изображения и поиск лиц
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)

        if len(face_locations) == 1:
            # Если на изображении найдено ровно одно лицо, создаем вектор
            face_encodings = face_recognition.face_encodings(image, face_locations)
            face_vector = face_encodings[0].tolist()

            # Получаем имя изображения без расширения
            image_name = os.path.splitext(filename)[0]

            # Добавляем имя и вектор в список

            faces_data.append([face_id, datetime.now(), image_name] + face_vector)
            face_id += 1
            print(faces_data[-1])

    column_names = ['face_id', 'event_timestamp', 'image_name'] + \
                   [f'feature_{i}' for i in range(1, 129)]

    faces_df = pd.DataFrame(faces_data, columns=column_names)

    print(faces_df.head())
    # Сохраняем DataFrame в файл faces.parquet
    faces_df.to_parquet(
        '/Users/alexegorov/PycharmProjects/faceRecognition/feast/data/faces.parquet')  # Укажите путь для сохранения файла


process_images()
