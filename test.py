from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

results = model("dataset/agri_0_3.jpeg")

for box in results[0].boxes:
    cls = int(box.cls)
    conf = float(box.conf)

    print("Class:", cls, "Confidence:", round(conf, 2))