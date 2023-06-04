from skud import skud

pj_skud = skud.Skud("./feast")
df = pj_skud.get_saved_dataset_as_df(faces_dataset_name="faces_dataset")

fei_indexes = ['face_id', 'event_timestamp', 'image_name']
features_indexes = [f'feature_{i}' for i in range(1, 129)]

new_order = fei_indexes + features_indexes

df = df.reindex(columns=new_order)
print(df.columns.tolist())
