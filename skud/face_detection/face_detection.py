import cv2
import numpy as np

class FaceDetector:
    def __init__(self, config_path, weights_path, class_names_path):
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.class_names = self.load_class_names(class_names_path)

    def load_class_names(self, class_names_path):
        with open(class_names_path, 'r') as f:
            class_names = f.read().splitlines()
        return class_names

    def detect(self, image):
        height, width, _ = image.shape

        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        outputs = self.net.forward(output_layers)
        faces = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if class_id == 0 and confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    face_width = int(detection[2] * width)
                    face_height = int(detection[3] * height)
                    x = int(center_x - face_width / 2)
                    y = int(center_y - face_height / 2)

                    face = image[y:y+face_height, x:x+face_width]
                    faces.append(face)

        return faces
