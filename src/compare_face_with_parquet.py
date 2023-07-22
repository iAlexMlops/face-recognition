import face_recognition
import pandas as pd
import numpy as np

INPUT_PARQUET_PATH = "../datasets/group_encoding.parquet"
INPUT_IMAGE_PATH = "../images/biden.jpg"


def get_names_of_matching_faces(indexes, compared_output):
    return [name_of_human for name_of_human, state_value in zip(indexes, compared_output) if state_value]


df = pd.read_parquet(INPUT_PARQUET_PATH)
print(df.head())

biden_image = face_recognition.load_image_file(INPUT_IMAGE_PATH)
biden_encoding = face_recognition.face_encodings(biden_image)
if len(biden_encoding) != 0:
    result = face_recognition.compare_faces(known_face_encodings=df,
                                            face_encoding_to_check=biden_encoding[0]
                                            )
    face_distances = face_recognition.face_distance(face_encodings=df,
                                                    face_to_compare=biden_encoding[0]
                                                    )
    best_match_index = np.argmin(face_distances)

    name = "Unknown"
    if result[best_match_index]:
        name = df.index.values[best_match_index]

    name = df.index.values[best_match_index] if result[best_match_index] else "Unknown"

    names = get_names_of_matching_faces(df.index.values, result)
    print(f"Result: {result}")
    print(f"Face distances: {face_distances}")
    print(f"Names: {names}")
    print(f"Name: {name}")
