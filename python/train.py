from ultralytics import YOLO
import torch
device = torch.device("cuda")

# 加载预训练的YOLO模型（推荐用于训练）
model = YOLO('./yolo11n.pt')
model.to(device)
results = model.train(data='data.yaml', epochs=50, workers=0)



