import face_recognition
import pandas as pd

from skud.feast_client.feast_client import FeastClient


class FaceRecognizer:
    def __init__(self, feast_repo_path, face_dataset_name):
        self.feast_client = FeastClient(feast_repo_path=feast_repo_path)
        self.dataset = self.load_face_database(face_dataset_name)

    def load_face_database(self, face_database_name) -> pd.DataFrame:
        df = self.feast_client.get_dataset(faces_dataset_name=face_database_name)

        fei_indexes = ['face_id', 'event_timestamp', 'image_name']
        features_indexes = [f'feature_{i}' for i in range(1, 129)]

        new_order = fei_indexes + features_indexes

        df = df.reindex(columns=new_order)

        columns_to_drop = ['face_id', 'event_timestamp']
        df = df.drop(columns=columns_to_drop)

        index_column = 'image_name'
        df = df.set_index(index_column)

        return df

    def encode(self, face_image) -> list:
        face_encodings = face_recognition.face_encodings(face_image)
        return face_encodings[0]


    def recognize(self, face_encoding):
        names = self.dataset.index.tolist()
        features = self.dataset.values.tolist()

        for name, feature_list in zip(names, features):
            matches = face_recognition.compare_faces([feature_list], face_encoding)
            if matches[0]:
                return name
        return "Unknown"
