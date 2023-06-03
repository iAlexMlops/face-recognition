import argparse

from feast.saved_dataset import SavedDataset

from skud.feast_client.feast_client import FeastClient


class Skud:
    def __init__(self,
                 feast_repo_path,
                 feature_view_name
                 ):
        self.feast_repo_path = feast_repo_path
        self.feature_view_name = feature_view_name

        self.feast_client = FeastClient(feast_repo_path=feast_repo_path, feature_view_name=feature_view_name)

    def generate_parquet(self):
        pass

    def create_feast_dataset(self) -> SavedDataset:
        return self.feast_client.create_dataset()

    def run_cli(self):
        parser = argparse.ArgumentParser(description='Skud - Face Detection and Recognition System')
        parser.add_argument('create_dataset', type=bool, help='')
        parser.add_argument('feast_repo_path', type=str, help='')
        parser.add_argument('feature_view_name', type=str, help='')

        args = parser.parse_args()