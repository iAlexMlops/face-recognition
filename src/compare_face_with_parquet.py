import argparse
from builtins import print

import face_recognition
import numpy as np
import pandas as pd

INPUT_PARQUET_PATH = str
INPUT_IMAGE_PATH = str


def get_names_of_matching_faces(indexes, compared_output):
    """Function return names of matching faces"""
    return [name_of_human for name_of_human, state_value in zip(indexes, compared_output) if state_value]


def main():
    """Main function"""
    print(f"Input parquet path: {INPUT_PARQUET_PATH}")
    print(f"Input image path: {INPUT_IMAGE_PATH}")

    df = pd.read_parquet(INPUT_PARQUET_PATH)
    print(df.head())

    image = face_recognition.load_image_file(INPUT_IMAGE_PATH)
    encoding = face_recognition.face_encodings(image)

    print(f"Len of encoding: {len(encoding)}")
    if len(encoding) != 0:
        result = face_recognition.compare_faces(known_face_encodings=df,
                                                face_encoding_to_check=encoding[0]
                                                )
        face_distances = face_recognition.face_distance(face_encodings=df,
                                                        face_to_compare=encoding[0]
                                                        )
        best_match_index = np.argmin(face_distances)

        name = df.index.values[best_match_index] if result[best_match_index] else "Unknown"
        names = get_names_of_matching_faces(df.index.values, result)

        print(f"Result: {result}")
        print(f"Face distances: {face_distances}")
        print(f"Names: {names}")
        print(f"Name: {name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get names of matching faces')

    parser.add_argument('--input_parquet_path',
                        default='../datasets/some_group.parquet',
                        help='Path to parquet file of face encodings')
    parser.add_argument('--input_image_path',
                        default='../images/test_group/Yana.jpeg',
                        help='Path to image to compare')

    args = parser.parse_args()

    INPUT_PARQUET_PATH = args.input_parquet_path
    INPUT_IMAGE_PATH = args.input_image_path

    main()
