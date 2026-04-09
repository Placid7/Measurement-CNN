from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil, uuid, os

app = FastAPI()

# ✅ Enable CORS (for Vercel frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # You can restrict to your Vercel URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load YOLO model
model = YOLO("yolov8n.pt")

@app.get("/")
def home():
    return {"message": "Model API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    filename = f"{uuid.uuid4()}.jpg"

    # Save uploaded file
    with open(filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run model with confidence threshold
    results = model(filename, conf=0.5)

    output = []
    names = model.names  # class names

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confs):

            # Extra filtering (optional but good)
            if conf < 0.5:
                continue

            x1, y1, x2, y2 = box

            output.append({
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "class": names[int(cls)],   # 🔥 returns name like 'car'
                "confidence": round(float(conf), 2)
            })

    # Delete temp file
    os.remove(filename)

    return {"objects": output}
