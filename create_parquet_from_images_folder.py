import os
import pandas as pd
import face_recognition


def process_images(input_folder, output_file):
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    face_encodings = []
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)
        face_encodings.append(face_encoding[0])

    df = pd.DataFrame(face_encodings, index=image_files)
    print(df.head())
    df.to_parquet(output_file)


if __name__ == "__main__":
    input_folder = "images/some_group"
    output_file = "datasets/some_group.parquet"

    process_images(input_folder, output_file)
