from ultralytics import YOLO
import torch

# опціонально, трохи ліберальніший матмул для нових RTX
torch.set_float32_matmul_precision("high")

model = YOLO("yolov8n.pt")  # вперше ваги самі скачаються
img = r"C:\dev\pcb\yolo_proto\data\face_0_4-THT.jpg"

# GPU: device=0. Якщо захочеш CPU, просто прибери параметр device
results = model(img, save=True, imgsz=640, device=0)

print(results[0].boxes)      # короткий підсумок боксов
