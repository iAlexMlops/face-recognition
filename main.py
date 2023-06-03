from skud.cli.skud import Skud

skud = Skud(feast_repo_path="feast",
            feature_view_name="faces_feature_view"
            )
dataset = skud.create_feast_dataset()

df = dataset.to_df()
print(df.head())
