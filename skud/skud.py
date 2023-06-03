from PIL import Image
from skud.face_detection import FaceDetector
from skud.face_recognition import FaceRecognizer
from skud.feast_client import FeastClient
import numpy as np
import argparse


class Skud:
    def __init__(self):
        self.face_detector = FaceDetector()  # Инициализация класса для детекции лиц
        self.face_recognizer = FaceRecognizer()  # Инициализация класса для распознавания лиц
        self.feast_client = FeastClient()  # Инициализация клиента Feast для доступа к базе данных Feast

    def detect_faces(self, image):
        # Метод для детекции лиц на изображении
        detected_faces = self.face_detector.detect(image)
        return detected_faces

    def recognize_faces(self, image, detected_faces):
        # Метод для распознавания лиц на изображении
        recognized_faces = []
        for face in detected_faces:
            face_encoding = self.face_recognizer.encode(face)
            face_info = self.feast_client.query_face(face_encoding)
            recognized_faces.append(face_info)
        return recognized_faces

    def run_cli(self):
        parser = argparse.ArgumentParser(description='Skud - Face Detection and Recognition System')
        parser.add_argument('image_path', type=str, help='Path to the image file')

        args = parser.parse_args()

        image = Image.open(args.image_path)
        image = image.convert("RGB")
        image_array = np.array(image)

        detected_faces = self.face_detector.detect(image_array)
        recognized_faces = []
        for face in detected_faces:
            face_encoding = self.face_recognizer.encode(face)
            face_info = self.feast_client.query_face(face_encoding)
            recognized_faces.append(face_info)

        print(recognized_faces)
