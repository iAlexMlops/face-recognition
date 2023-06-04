from unittest import TestCase
import cv2
from skud.face_recognition.face_recognition import FaceRecognizer


class TestFaceRecognizer(TestCase):
    def setUp(self):
        self.feast_repo_path = "./feast"
        self.face_dataset_name = "faces_dataset"
        self.image_for_encoding = "./images/1601731018330.jpg"

    def test_load_face_database(self):
        fr = FaceRecognizer(feast_repo_path=self.feast_repo_path, face_dataset_name=self.face_dataset_name)

        image = cv2.imread(self.image_for_encoding)

        encoded_face = fr.encode(image)
        print(encoded_face)

        face_id = fr.recognize(encoded_face)
        print(face_id)
        self.assertNotEqual(face_id, None)

