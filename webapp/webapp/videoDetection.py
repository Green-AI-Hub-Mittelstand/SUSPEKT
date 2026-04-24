import json
import shutil
import time
import uuid
from pathlib import Path

import cv2
from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import (
    HTMLResponse,
    FileResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO

router = APIRouter(prefix="/video")

templates = Jinja2Templates(directory="templates")

# Directory paths
UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Load YOLO model
MODEL = "model/system180custommodel_v1.pt"
model = YOLO(MODEL)


@router.get("", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Render the image upload page"""
    return templates.TemplateResponse("videoDetection.html", {"request": request})


@router.post("/upload/")
def upload_video(file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())
    file_extension = file.filename.rsplit(".", 1)[-1].lower()
    print(f"Uploading file: {file.filename} (Detected extension: {file_extension})")

    if file_extension not in ["mp4", "mov"]:
        return JSONResponse(content={"error": "Unsupported file format"}, status_code=400)

    video_path = UPLOAD_DIR / f"{video_id}.{file_extension}"

    with video_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return JSONResponse(content={
        "video_id": video_id,
        "uploaded_video_url": f"/video/uploaded/{video_id}.{file_extension}"
    })


@router.get("/stream/{video_id}/{file_extension}")
def stream_video(video_id: str, file_extension: str):
    video_path = UPLOAD_DIR / f"{video_id}.{file_extension}"

    if not video_path.exists():
        return JSONResponse(content={"error": "File not found"}, status_code=404)

    def generate_frames():
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            elapsed_time = time.time() - start_time
            current_fps = frame_number / elapsed_time if elapsed_time > 0 else 0

            results = model(frame)
            result = results[0]
            predictions = []

            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0].item())
                    cls = int(box.cls[0].item())
                    class_name = result.names[cls]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    predictions.append({
                        "class": class_name,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2]
                    })

            # FPS Anzeige auf das Video zeichnen
            cv2.putText(frame, f"FPS: {current_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            _, buffer = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            yield json.dumps({"predictions": predictions}) + "\n"

        cap.release()

    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.get("/uploaded/{filename}")
def get_uploaded_file(filename: str):
    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        return FileResponse(file_path)
    return JSONResponse(content={"error": "File not found"}, status_code=404)


"""@router.post("/analyze/")
def analyze_video(video_id: str, file_extension: str):
    video_path = UPLOAD_DIR / f"{video_id}.{file_extension}"
    output_video_path = PROCESSED_DIR / f"{video_id}_processed.mp4"
    output_json_path = PROCESSED_DIR / f"{video_id}.json"

    if not video_path.exists():
        return JSONResponse(content={"error": "Video not found"}, status_code=404)

    process_video(video_path, output_video_path, output_json_path)

    return JSONResponse(content={
        "processed_video_url": f"/video/processed/{video_id}_processed.mp4",
        "json_url": f"/video/processed/{video_id}.json"
    })


def process_video(input_path: Path, output_path: Path, json_path: Path):
    cap = cv2.VideoCapture(str(input_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        result = results[0]  # Extrahiere das erste Ergebnis

        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0].item())
                cls = int(box.cls[0].item())

                class_name = result.names[cls]  # Klassennamen abrufen

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                predictions.append({
                    "class": class_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })

        out.write(frame)

    cap.release()
    out.release()

    with open(json_path, "w") as json_file:
        json.dump(predictions, json_file)

"""


@router.get("/processed/{filename}")
def get_processed_file(filename: str):
    file_path = PROCESSED_DIR / filename
    if file_path.exists():
        return FileResponse(file_path)
    return JSONResponse(content={"error": "File not found"}, status_code=404)
