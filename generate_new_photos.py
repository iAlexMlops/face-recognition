from skud.face_detection.face_detection import FaceDetector
import os
import glob
import cv2


folder_path = "./images"
output_folder = "./update_images"
os.makedirs(output_folder, exist_ok=True)

fd = FaceDetector(model_path="models/crowdhuman_yolov5m.pt")

jpg_files = glob.glob(folder_path + "/*.jpg")

indexer = 0
for jpg_file in jpg_files:
    image = cv2.imread(jpg_file)
    faces = fd.detect_face(image)

    if len(faces) == 0:
        continue

    ymin, xmax, ymax, xmin = faces[0]
    cropped_image = image[ymin:ymax, xmin:xmax]

    output_path = os.path.join(output_folder, str(indexer)+".jpg")
    indexer+=1
    print(output_path)
    cv2.imwrite(output_path, cropped_image)



