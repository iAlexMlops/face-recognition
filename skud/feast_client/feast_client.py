import pandas as pd
from feast import FeatureStore
from feast.infra.offline_stores.file_source import SavedDatasetFileStorage
from feast.saved_dataset import SavedDataset

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class FeastClient:
    def __init__(self,
                 feast_repo_path,
                 ):
        self.feast_repo_path = feast_repo_path
        self.store = FeatureStore(repo_path=feast_repo_path)

        # self.faces_parquet_path = f"{feast_repo_path}/data/faces.parquet"
        # self.face_database = self.load_face_database(face_database_path)

    def create_dataset(self, feature_view_name, faces_parquet_path, faces_dataset_name) -> SavedDataset:
        entity_df = pd.read_parquet(path=faces_parquet_path)

        features = [f"{feature_view_name}:image_name"] + [
            f"{feature_view_name}:feature_{i}" for i in range(1, 129)
        ]
        faces_dataset_from_data = self.store.get_historical_features(
            entity_df=entity_df,
            features=features
        )

        # Storing the dataset as a local file
        dataset = self.store.create_saved_dataset(
            from_=faces_dataset_from_data,
            name=faces_dataset_name,
            storage=SavedDatasetFileStorage(f"{self.feast_repo_path}/data/{faces_dataset_name}.parquet")
        )

        return dataset

    def get_dataset(self, faces_dataset_name) -> pd.DataFrame:
        store = FeatureStore(repo_path=self.feast_repo_path)
        return store.get_saved_dataset(name=faces_dataset_name).to_df()
