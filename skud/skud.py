from skud.feast_client.feast_client import FeastClient
import argparse
import face_recognition
import os
import pandas as pd
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Skud:
    def __init__(self, feast_repo_path,):
        """
        Инициализация объекта проекта Skud.

        Args:
            :feast_repo_path (str): Определяет репозиторий Feast Feature Store.

        Returns:
            None
        """
        self.feast_repo_path = feast_repo_path
        self.feast_client = FeastClient(feast_repo_path=feast_repo_path)

    def generate_parquet(self, images_folder: str, faces_parquet_path: str):
        """
        Генерация parquet файла из исходных картинок в указанной папке.

        Args:
            images_folder (str): Путь до каталога с изображениями лиц (/path/to/image_folder/).
            faces_parquet_path (str): Путь куда будет сохранен итоговый parquet файл (/path/to/file.parquet).

        Returns:
            None
        """
        faces_data = []
        face_id = 0

        for filename in os.listdir(images_folder):
            image_path = os.path.join(images_folder, filename)
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)

            if len(face_locations) == 1:
                face_encodings = face_recognition.face_encodings(image, face_locations)
                face_vector = face_encodings[0].tolist()

                image_name = os.path.splitext(filename)[0]

                faces_data.append([face_id, datetime.now(), image_name] + face_vector)
                face_id += 1

            # TODO: Remove breakpoint
            if face_id == 3:
                break

        column_names = ['face_id', 'event_timestamp', 'image_name'] + \
                       [f'feature_{i}' for i in range(1, 129)]

        faces_df = pd.DataFrame(faces_data, columns=column_names)
        faces_df.to_parquet(faces_parquet_path)

    def create_feast_dataset(self, feature_view_name: str, faces_parquet_path: str, faces_dataset_name: str):
        """
        Генерация Dataset в Feast из FeatureView и исходного parquet файла.

        Args:
            feature_view_name (str): Название существующего FeatureView в Feast.
            faces_parquet_path (str): Путь до исходного parquet файла (/path/to/file.parquet).
            faces_dataset_name (str): Название будущего Dataset в Feast
        Returns:
            None
        """
        self.feast_client.create_dataset(feature_view_name=feature_view_name,
                                         faces_parquet_path=faces_parquet_path,
                                         faces_dataset_name=faces_dataset_name,
                                         )

    def get_saved_dataset_as_df(self, faces_dataset_name: str) -> pd.DataFrame:
        """
        Получение существующего Dataset в Feast по имени.

        Args:
            faces_dataset_name (str): Название существующего Dataset в Feast
        Returns:
            pd.DataFrame
        """
        return self.feast_client.get_dataset(faces_dataset_name)


def run_cli():
    parser = argparse.ArgumentParser(description='Skud - Face Detection and Recognition System')
    parser.add_argument('create_dataset', type=bool, help='')
    parser.add_argument('feast_repo_path', type=str, help='')
    parser.add_argument('feature_view_name', type=str, help='')

    args = parser.parse_args()


if __name__ == "__main__":
    run_cli()
