from skud import skud

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

pj_skud = skud.Skud(feast_repo_path="../feast")

pj_skud.create_feast_dataset(feature_view_name="faces_feature_view",
                             faces_parquet_path="../feast/data/faces.parquet",
                             faces_dataset_name="faces_dataset"
                             )