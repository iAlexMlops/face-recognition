# Importing dependencies
import pandas as pd
from feast import FeatureStore
from feast.infra.offline_stores.file_source import SavedDatasetFileStorage

# Getting our FeatureStore
store = FeatureStore(repo_path="../feast/")

# Reading our targets as an entity DataFrame
entity_df = pd.read_parquet(path="../feast/data/faces.parquet")

features = ["faces_feature_view:image_name"] + [
    f"faces_feature_view:feature_{i}" for i in range(1, 129)
]

# Getting the indicated historical features
# and joining them with our entity DataFrame
training_data = store.get_historical_features(
    entity_df=entity_df,
    features=features
)

# Storing the dataset as a local file
dataset = store.create_saved_dataset(
    from_=training_data,
    name="faces_dataset",
    storage=SavedDatasetFileStorage("../feast/data/faces_dataset.parquet")
)
