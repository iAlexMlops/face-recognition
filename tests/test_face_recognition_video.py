from unittest import TestCase
import cv2
from skud.face_recognition.face_recognition import FaceRecognizer


def count_duplicates(my_list):
    counts = {}  # Создаем пустой словарь для подсчета повторений
    for item in my_list:
        if item in counts:
            counts[item] += 1  # Увеличиваем значение для существующего элемента
        else:
            counts[item] = 1  # Добавляем новый элемент в словарь с начальным значением 1

    return counts


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
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            if not ret:
                continue

            try:
                encoded_face = fr.encode(frame)
                face = fr.recognize(encoded_face)
                face_ids.add(face)
                print(face_ids)

                result_image = cv2.imread(f"{self.result_image_encoding}/{face}.jpg")
                cv2.imshow(f"Out {str(face)}.jpg", result_image)

            except Exception as e:
                print(e)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.waitKey(0)
        video_capture.release()
        cv2.destroyAllWindows()

        # Release handle to the webcam
        self.assertNotEqual(face_ids, None)




