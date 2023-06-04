from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from dotenv import load_dotenv
import face_recognition
import os
import pandas as pd

load_dotenv('.env')


def process_images():
    image_folder = os.getenv('IMAGES_FOLDER')
    faces_data = []

    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)

        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)

        if len(face_locations) == 1:

            face_encodings = face_recognition.face_encodings(image, face_locations)
            face_vector = face_encodings[0].tolist()

            image_name = os.path.splitext(filename)[0]
            faces_data.append([image_name, datetime.now(), image_name] + face_vector)

        column_names = [
                       'face_id',
                       'event_timestamp',
                       'image_name'
                   ] + \
                   [
                       f'feature_{i}' for i in range(1, 129)
                   ]
    faces_df = pd.DataFrame(faces_data, columns=column_names)

    faces_df.to_parquet(f"{os.getenv('FEAST_REPO')}/data/faces.parquet")


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 6, 1),
}

# Создание DAG
dag = DAG(
    'face_recognition_dag',
    default_args=default_args,
    schedule_interval=None
)

# Определение задачи для обработки изображений
process_images_task = PythonOperator(
    task_id='process_images',
    python_callable=process_images,
    dag=dag
)
