from unittest import TestCase
from skud.face_detection.face_detection import FaceDetector
import cv2


class TestFaceDetector(TestCase):
    def setUp(self):
        self.model_path = "./models/crowdhuman_yolov5m.pt"
        self.image_for_detect = "./src/img/unknown_images/Putin2.jpg"

    def test_detect_face(self):
        fd = FaceDetector(self.model_path)
        faces = fd.detect_face(self.image_for_detect)
        print("Координаты лица: ", faces)

        self.assertNotEqual(faces, None)

        image = cv2.imread(self.image_for_detect)
        for face in faces:
            ymin, xmax, ymax, xmin = face
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        cv2.imshow('Image with Square', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
