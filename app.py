from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import shutil
import uuid
import os

app = FastAPI()

# Load model once
model = YOLO("best.pt")

@app.get("/")
def home():
    return {"message": "Model API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file
    filename = f"{uuid.uuid4()}.jpg"
    
    with open(filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run model
    results = model(filename)

    output = []

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
        output = boxes

    # Remove temp file
    os.remove(filename)

    return {
        "detections": output
    }
