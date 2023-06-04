# Importing dependencies
from datetime import timedelta
from feast import Entity, FeatureView, FileSource, ValueType, Field
from feast.types import Float32, String

# Declaring an entity for the dataset
face = Entity(
    name="face_id",
    value_type=ValueType.INT64,
    description="The ID of the Face")

# Declaring the source of the first set of features
f_source1 = FileSource(
    name="f_source1",
    path=r"/Users/alexegorov/PycharmProjects/faceRecognition/feast/data/faces.parquet",
    timestamp_field="event_timestamp"
)

# Defining the first set of features
df1_fv = FeatureView(
    name="faces_feature_view",
    ttl=timedelta(days=1),
    entities=[face],
    schema=
    [Field(name="image_name", dtype=String)]+[
        Field(name=f"feature_{i}", dtype=Float32) for i in range(1, 129)
    ],
    source=f_source1
)
