# from skud.face_detection.face_detection import FaceDetector
# from skud.face_recognition.face_recognition import FaceRecognizer
# from skud import FeastClient
import argparse

from skud.feast_client.feast_client import FeastClient


class Skud:
    def __init__(self):
        # self.face_detector = FaceDetector()
        # self.face_recognizer = FaceRecognizer()
        self.feast_client = FeastClient(feast_repo_path='../../feast', feature_view_name='faces_feature_view')

    def generate_parquet(self):
        pass

    def create_feast_dataset(self):
        self.feast_client.create_dataset()

    def run_cli(self):
        parser = argparse.ArgumentParser(description='Skud - Face Detection and Recognition System')
        parser.add_argument('create_dataset', type=bool, help='')
        parser.add_argument('feast_repo_path', type=str, help='')
        parser.add_argument('feature_view_name', type=str, help='')

        args = parser.parse_args()