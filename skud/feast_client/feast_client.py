import pandas as pd
from feast import FeatureStore
from feast.infra.offline_stores.file_source import SavedDatasetFileStorage
from feast.saved_dataset import SavedDataset


class FeastClient:
    def __init__(self,
                 feast_repo_path,
                 feature_view_name
                 ):
        self.feast_repo_path = feast_repo_path
        self.feature_view_name = feature_view_name
        self.store = FeatureStore(repo_path=feast_repo_path)
        # self.face_database = self.load_face_database(face_database_path)

    def create_dataset(self) -> SavedDataset:
        # Reading our targets as an entity DataFrame
        entity_df = pd.read_parquet(path=f"../{self.feast_repo_path}/data/faces.parquet")

        # Getting the indicated historical features
        # and joining them with our entity DataFrame
        training_data = self.store.get_historical_features(
            entity_df=entity_df,
            features=[
                f"{self.feature_view_name}:feature{i}" for i in range(1, 129)
            ]
        )

        # Storing the dataset as a local file
        dataset = self.store.create_saved_dataset(
            from_=training_data,
            name="faces_dataset",
            storage=SavedDatasetFileStorage("../{self.feast_repo_path}/data/faces_dataset.parquet")
        )

        return dataset

    # def load_face_database(self, face_database_path):
    #     # Загрузка базы данных лиц из файла CSV или другого источника данных
    #     # и создание DataFrame с информацией о лицах
    #     face_database = pd.read_csv(face_database_path)
    #     return face_database
    #
    # def query_face(self, face_encoding):
    #     # Запрос информации о лице с заданным вектором признаков лица из базы данных Feast
    #     face_info = self.face_database.loc[self.face_database['encoding'] == face_encoding]
    #     if not face_info.empty:
    #         face_name = face_info['name'].values[0]
    #         return face_name
    #     else:
    #         return "Unknown"
