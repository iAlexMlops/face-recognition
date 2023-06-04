from unittest import TestCase
import cv2
from skud.face_recognition.face_recognition import FaceRecognizer


class TestFaceRecognizer(TestCase):
    def setUp(self):
        self.feast_repo_path = "../feast"
        self.face_dataset_name = "faces_dataset"
        self.image_for_encoding = "../update_images/5.jpg"
        self.result_image_encoding = "../update_images/"

    def test_load_face_database(self):
        fr = FaceRecognizer(feast_repo_path=self.feast_repo_path, face_dataset_name=self.face_dataset_name)

        image = cv2.imread(self.image_for_encoding)
        encoded_face = fr.encode(image)
        face_id = fr.recognize(encoded_face)

        result_image = cv2.imread(f"{self.result_image_encoding}/{face_id}")

        cv2.imshow(self.image_for_encoding.split("/")[-1], image)
        cv2.imshow(f"Out {str(face_id)}.jpg", image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        self.assertNotEqual(face_id, None)

