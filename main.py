from ultralytics import YOLO

model = YOLO("best.pt")
model.predict(source="fruit.jpg", imgsz=640, conf=0.6, save=True, project="runs", name="detect")
