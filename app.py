from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import shutil, uuid, os

app = FastAPI()

# ✅ Use pretrained model
model = YOLO("yolov8n.pt")

@app.get("/")
def home():
    return {"message": "Model API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    filename = f"{uuid.uuid4()}.jpg"

    with open(filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(filename)

    output = []

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = box

            output.append({
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "class": int(cls),
                "confidence": float(conf)
            })

    os.remove(filename)

    return {"objects": output}
