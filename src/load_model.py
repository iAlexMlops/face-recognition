import torch
import ultralytics

# Model
mod_p = '../crowdhuman_yolov5m.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=mod_p)


# Image
im = 'https://ultralytics.com/images/zidane.jpg'

# Inference
results = model(im)

print(results.pandas().xyxy[0])
