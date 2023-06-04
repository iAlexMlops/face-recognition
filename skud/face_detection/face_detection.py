import torch


class FaceDetector:
    def __init__(self, model_path: str):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, device='mps')

        # TODO: add maintain of classes
        # self.model_classes = model_classes

    def detect_face(self, image) -> list:
        results = self.model(image)
        df = results.pandas().xyxy[0]

        if df.empty:
            return []

        faces_df = df[df["class"] == 1]
        faces = [tuple([int(bbox[key]) for key in ['ymin', 'xmax', 'ymax', 'xmin']]) for bbox in
                 faces_df.to_dict(orient='records')]

        return faces
