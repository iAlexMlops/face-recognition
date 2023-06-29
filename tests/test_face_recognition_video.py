from unittest import TestCase
import cv2
from skud.face_recognition.face_recognition import FaceRecognizer


class TestFaceRecognizer(TestCase):
    def setUp(self):
        self.feast_repo_path = "../feast"
        self.face_dataset_name = "faces_dataset"
        self.video_for_encoding = "../video_for_encoding.mp4"
        self.result_image_encoding = "../update_images/"

    def test_load_face_database(self):
        video_capture = cv2.VideoCapture(self.video_for_encoding)
        fr = FaceRecognizer(feast_repo_path=self.feast_repo_path, face_dataset_name=self.face_dataset_name)

        face_ids = set()

        while True:
            ret, frame = video_capture.read()
            # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # rgb_small_frame = small_frame[:, :, ::-1]

            try:
                encoded_face = fr.encode(frame)
                face_ids.add(fr.recognize(encoded_face))
            except Exception as e:
                print("Out of range")

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print(face_ids)

        # result_image = cv2.imread(f"{self.result_image_encoding}/{face_id}.jpg")
        #
        # cv2.imshow(self.image_for_encoding.split("/")[-1], image)
        # cv2.imshow(f"Out {str(face_id)}.jpg", result_image)

        cv2.waitKey(0)
        video_capture.release()
        cv2.destroyAllWindows()

        # Release handle to the webcam
        self.assertNotEqual(face_ids, None)




