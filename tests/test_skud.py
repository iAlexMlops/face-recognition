import unittest
import pandas as pd
import os
from skud import skud

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestProcessImages(unittest.TestCase):

    def test_create_constructor(self):
        pj_skud = skud.Skud(feast_repo_path="./feast")
        self.assertNotEqual(pj_skud, None)

    def test_create_parquet(self):
        pj_skud = skud.Skud(feast_repo_path="./feast")
        pj_skud.generate_parquet(images_folder="./images", faces_parquet_path="./feast/data/test.parquet")

        self.assertTrue(os.path.exists('./feast/data/test.parquet'))

        faces_df = pd.read_parquet('./feast/data/test.parquet')
        self.assertEqual(len(faces_df), 3)

        expected_columns = ['face_id', 'event_timestamp', 'image_name'] + \
                           [f'feature_{i}' for i in range(1, 129)]
        self.assertListEqual(list(faces_df.columns), expected_columns)

    def test_create_dataset(self):
        pj_skud = skud.Skud(feast_repo_path='./feast')
        try:
            os.system("rm ./feast/data/test_dataset.parquet")
        except:
            print("Test_dataset has already deleted")

        pj_skud.create_feast_dataset(feature_view_name='faces_feature_view',
                                     faces_parquet_path='./feast/data/test.parquet',
                                     faces_dataset_name='test_dataset')

        # Retrieving the saved dataset and converting it to a DataFrame
        training_df = pj_skud.get_saved_dataset_as_df(faces_dataset_name="test_dataset")

        self.assertEqual(len(training_df), 3)
        expected_columns = ['face_id', 'event_timestamp', 'image_name'] + \
                           [f'feature_{i}' for i in range(1, 129)]

        # TODO: fix order of dataset join_keys
        print(training_df.head())
        os.system("rm ./feast/data/test_dataset.parquet")



if __name__ == '__main__':
    unittest.main()
