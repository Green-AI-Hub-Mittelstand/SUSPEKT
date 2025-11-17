import base64
import os
from typing import List

import cv2
from datetime import datetime
import json
import numpy as np
import pandas as pd
import requests
import torch
import uvicorn

from PIL import Image
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

# Router import
from .auth import router as auth_router
from .conditionDetection import ZustandModel, Single_Transformer
from .config import model, HERE_API_KEY
from .decorDetection import UnidekorDetector


from .decorDetection import router as color_detection_router
from .inventory_routes import router as inventory_router
from .modelTraining import router as model_training_router
from .transport_emission import router as resource_router
from .videoDetection import router as video_router


from .processImage import process_images
from .user_db_models import init_db
from .neo4jIntegration import Neo4jDatabase


from dotenv import load_dotenv

# HERE API Key laden
load_dotenv()

# Load YOLO model
# MODEL = "model/system180custommodel_v1.pt"
# MODEL = "model/system180CustomModelCaniaYolo11200Epochs.pt"
# model = YOLO(MODEL)

"""# List of classes that need color detection
color_detection_classes = [
    "Auszug", "Verkleidung", "Systemboden",
    "Einzeltuer", "Doppeltuerblatt", "Doppeltuer", "Seitenverkleidung-0-IN",
    "Seitenverkleidung-0-0",
    "Seitenverkleidung-IN-IN"

]"""

detector = UnidekorDetector()

# FastAPI App Setup
app = FastAPI(redirect_slashes=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify domains or allow all with "*"
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# app.add_middleware(HTTPSRedirectMiddleware)
app.add_middleware(SessionMiddleware, secret_key="supergeheim123")


# Setup static and template directories
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Ensure detected_images directory exists
os.makedirs("static/detected_images", exist_ok=True)

# Datenbank initialisieren
init_db()

# Router einbinden
app.include_router(auth_router)
app.include_router(color_detection_router)
app.include_router(inventory_router)
app.include_router(video_router)
app.include_router(model_training_router)
app.include_router(resource_router)
#(app.routes)


db = Neo4jDatabase()

@app.get("/", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Render the image upload page"""
    return templates.TemplateResponse("upload.html", {"request": request})


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")

from fastapi import Query

# 🟢 Zwischenspeicher für erkannte Objekte
image_results_cache = {}

from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Form, Query, Depends


@app.get("/detect/", response_class=HTMLResponse)
@app.post("/detect/", response_class=HTMLResponse)
async def detect_objects(
        request: Request,
        files: Optional[List[UploadFile]] = None,  # Changed to Optional
        image_views: Optional[List[str]] = Form(None),  # Changed to List[str]
        capture_type: Optional[str] = Form("single"),  # Added default value
        updated_filename: Optional[str] = Query(None)
):
    """Erkennung & Laden der aktualisierten Objekte"""
    print(f"Empfangene image_views: {image_views}")  # Debug-Log
    print(f"Empfangener capture_type: {capture_type}")  # Debug-Log

    # 🟢 Wenn nur ein Update gemacht wurde, zeigen wir die aktualisierten Daten an
    if request.method == "GET" and updated_filename:
        if updated_filename in image_results_cache:
            updated_results_df = image_results_cache[updated_filename]  # DataFrame abrufen
            updated_image_results_dict = {updated_filename: updated_results_df.to_dict(orient="records")}

            return templates.TemplateResponse("results.html", {
                "request": request,
                "detected_images": [updated_filename],
                "image_results": updated_image_results_dict,  # KEIN JSON-String!
                "capture_type": capture_type  # Aufnahmetyp an das Template weitergeben
            })
        else:
            return HTMLResponse("Fehler: Bild nicht in Cache gefunden", status_code=404)

    # 🟢 Wenn neue Bilder hochgeladen werden, führen wir eine komplette Erkennung durch
    if request.method == "POST":
        # Handle form manually if needed
        if files is None:
            # Try to get files from form directly
            form = await request.form()
            files = form.getlist("files")
            if not files:
                return HTMLResponse("Fehler: Keine Bilder hochgeladen", status_code=400)

        # Parsen der Bild-Ansichten, die im Format "Dateiname:Ansicht" kommen
        view_dict = {}
        if image_views:
            for view_info in image_views:
                try:
                    if isinstance(view_info, str) and ":" in view_info:
                        filename, view = view_info.split(":", 1)
                        view_dict[filename] = view
                        print(f"Parsed view: {filename} -> {view}")  # Detailliertes Debug-Log
                except Exception as e:
                    print(f"Fehler beim Parsen der Ansicht '{view_info}': {str(e)}")

        # Make sure files is a list
        if not isinstance(files, list):
            files = [files]

        # Filter out invalid files
        valid_files = [f for f in files if hasattr(f, 'file') and f.file]

        if not valid_files:
            return HTMLResponse("Fehler: Keine gültigen Bilder gefunden", status_code=400)

        # Füge die Ansichtsinformationen und Aufnahmetyp an die Verarbeitungsfunktion weiter
        detected_images, image_results = process_images(
            valid_files,
            views=view_dict,
            capture_type=capture_type
        )

        # 🚀 Zwischenspeicher aktualisieren
        for key, df in image_results.items():
            image_results_cache[key] = df

        # Wandelt DataFrames in ein JSON-kompatibles Dictionary um
        image_results_dict = {key: df.to_dict(orient="records") for key, df in image_results.items()}

        return templates.TemplateResponse("results.html", {
            "request": request,
            "detected_images": detected_images,
            "image_results": image_results_dict,  # KEIN JSON-String!
            "capture_type": capture_type  # Aufnahmetyp an das Template weitergeben
        })

    return templates.TemplateResponse("results.html", {"request": request})


@app.post("/update_condition/")
async def update_condition(
    request: Request,
    filename: str = Form(...),
    bbox_id: int = Form(...),
    zustand: str = Form(...)
):
    try:
        #print(f"🔍 Datei: {filename}, 🆔 bbox_id: {bbox_id}, 📌 Neuer Zustand: {zustand}")

        if filename not in image_results_cache:
            return JSONResponse(status_code=404, content={"success": False, "message": f"Datei {filename} nicht gefunden."})

        df = image_results_cache[filename]

        if bbox_id not in df["bbox_id"].values:
            return JSONResponse(status_code=404, content={"success": False, "message": f"Objekt-ID {bbox_id} nicht gefunden."})

        df.loc[df["bbox_id"] == bbox_id, "zustand"] = zustand
        is_damaged = zustand in ["MDF-Platzer", "Rohr_Kratzer", "Delle"]
        df.loc[df["bbox_id"] == bbox_id, "reusable"] = not is_damaged
        image_results_cache[filename] = df

        print(f"✅ Aktualisiertes DataFrame für {filename}:\n{df}")

        return JSONResponse(status_code=200, content={"success": True, "message": "Zustand erfolgreich aktualisiert!"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": f"Fehler: {str(e)}"})


@app.post("/review_results/")
async def review_results(
        request: Request,
        image_results: str = Form(...),
        location: str = Form(None),
        latitude: str = Form(None),
        longitude: str = Form(None),
        formatted_address: str = Form(None),
        deleted_rows: str = Form("[]")  # Feld für gelöschte Zeilen
):
    """Ergebnisse zur Überprüfung anzeigen"""

    # JSON-String aus dem Formular in Dictionary umwandeln
    image_results_dict = json.loads(image_results)

    # Gelöschte Zeilen als Liste von Indizes parsen
    deleted_rows_list = json.loads(deleted_rows)

    print(f"Empfangene gelöschte Zeilen: {deleted_rows_list}")
    print(f"Image Results vor Verarbeitung: {type(image_results_dict)}")

    # Geodaten erfassen
    location_data = {
        "location": location,
        "latitude": latitude,
        "longitude": longitude,
        "formatted_address": formatted_address
    }

    # Hinweis: Die eigentliche Löschung sollte bereits im Frontend im JavaScript erfolgt sein,
    # aber wir überprüfen hier zur Sicherheit nochmals

    try:
        # Überprüfen, ob image_results_dict ein Array oder ein Dict von Arrays ist
        if isinstance(image_results_dict, list):
            # Direkt als DataFrame konvertieren
            df = pd.DataFrame(image_results_dict)

            # Reusable-Flag setzen
            df["reusable"] = df["zustand"].apply(
                lambda z: z not in ["MDF-Platzer", "Rohr_Kratzer", "Delle", "Unbekannt"])

            # Geodaten zu DataFrame hinzufügen
            df["location"] = location
            df["latitude"] = latitude
            df["longitude"] = longitude
            df["formatted_address"] = formatted_address

            # DataFrame zurück in dict-Format umwandeln
            image_results_json = df.to_dict(orient="records")

        else:
            # JSON-Dictionary in Pandas DataFrames umwandeln
            image_results_df = {key: pd.DataFrame(value) for key, value in image_results_dict.items() if value}

            # Für jedes DataFrame die Geodaten hinzufügen
            for df in image_results_df.values():
                # Reusable-Flag setzen
                df["reusable"] = df["zustand"].apply(
                    lambda z: z not in ["MDF-Platzer", "Rohr_Kratzer", "Delle", "Unbekannt"])

                # Geodaten zu jedem DataFrame hinzufügen
                df["location"] = location
                df["latitude"] = latitude
                df["longitude"] = longitude
                df["formatted_address"] = formatted_address

            # DataFrames in eine JSON-serialisierbare Struktur umwandeln
            image_results_json = {key: df.to_dict(orient="records") for key, df in image_results_df.items()}
    except Exception as e:
        print(f"Fehler bei der Verarbeitung: {str(e)}")
        # Fallback - Image Results unverändert verwenden
        image_results_json = image_results_dict

    today = datetime.now().strftime('%Y%m%d')  # Heute als String

    if isinstance(image_results_json, dict) and image_results_json:
        first_image = next(iter(image_results_json.keys()), "unbekannt")
    else:
        first_image = "unbekannt"

    generated_order_id = f"{first_image}_{today}"  # Vorgangsnummer erstellen

    return templates.TemplateResponse("review_results.html", {
        "request": request,
        "generated_order_id": generated_order_id,
        "image_results": image_results_json,
        "location_data": location_data
    })


@app.post("/confirm_results/")
async def confirm_results(
    request: Request,
    order_id: str = Form(...),
    system180_order: str = Form(...),
    contact_email: str = Form(...),
    contact_phone: str = Form(...),
    location: str = Form(...),
    latitude: str = Form(None),
    longitude: str = Form(None),
    formatted_address: str = Form(None),
    additional_info: str = Form(...),
    order_type: str = Form(...),  # Online oder Vor-Ort
    image_results: str = Form(...)
):
    """Speichert bestätigte Ergebnisse in Neo4j"""

    # 🔹 Debugging: Prüfen, ob image_results wirklich ankommt
    if not image_results:
        return {"error": "image_results wurde nicht übermittelt"}

    # JSON-String in Dictionary umwandeln
    try:
        image_results_dict = json.loads(image_results)
    except json.JSONDecodeError as e:
        return {"error": f"Fehler beim Parsen von image_results: {str(e)}"}

    # JSON-Dictionary in DataFrames konvertieren
    image_results_df = {key: pd.DataFrame(value) for key, value in image_results_dict.items()}

    # Ergebnisse in Neo4j speichern
    try:
        db.store_image_results(
            image_results=image_results_df,
            order_id=order_id,
            system180_order=system180_order,
            contact_email=contact_email,
            contact_phone=contact_phone,
            location=location,
            latitude=latitude,
            longitude=longitude,
            formatted_address=formatted_address,
            order_type=order_type,
            additional_info=additional_info
        )
        message = "Daten erfolgreich in Neo4j gespeichert!"
        success = True
    except Exception as e:
        message = f"Fehler beim Speichern in Neo4j: {str(e)}"
        success = False

    # Bestätigungsseite rendern
    return templates.TemplateResponse("confirmation.html", {
        "request": request,
        "success": success,
        "message": message
    })


detector = UnidekorDetector()


@app.post("/detect_decor/")
async def detect_decor(request: Request, files: List[UploadFile] = File(...)):
    """Handle decor detection for multiple images"""
    results = []
    for file in files:
        try:
            # Lese den Dateiinhalt
            contents = await file.read()

            # Konvertiere zu OpenCV Format
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError(f"Konnte Bild nicht laden: {file.filename}")

            # Erstelle Base64 für Vorschau
            base64_image = base64.b64encode(contents).decode('utf-8')

            # Führe die Analyse durch
            result = detector.analyze_uploaded_image(image)
            # Erstelle das Ergebnis-Dictionary
            result_dict = {
                'filename': file.filename,
                'original': base64_image,
                'erkannte_farbe': result['erkannte_farbe'],
                'hex_code': result['hex_code'],
                'ncs_code': result['ncs_code'],
                'confidence': result['confidence'],
                'erkannte_rgb': result['erkannte_rgb']
            }
            results.append(result_dict)

        except Exception as e:
            print(f"Fehler bei {file.filename}: {str(e)}")
            continue

    return templates.TemplateResponse("results.html", {
        "request": request,
        "detected_images": results
    })






zustand_erkennung_model = ZustandModel(input_shape=(3, 300, 300), num_features=30, num_labels=3, feat_active='relu')


@app.post("/api/detect_condition")
async def detect_condition(file: UploadFile = File(...), part_type: str = Form(...)):
    try:
        # Parse the JSON body from the request
        contents = await file.read()  # Read the uploaded image
        filename = file.filename
        full_path = os.path.join("static", "detected_images", filename)

        # Save the file
        with open(full_path, "wb") as f:
            f.write(contents)

        # print(f"Processing image: {filename}")
        # print(f"Full path: {full_path}")

        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail=f"Image not found: {full_path}")

        if not os.path.exists("model/faultDetection.pt"):
            raise HTTPException(status_code=500, detail="Model file not found")

        # Load the model and transformer
        zustand_erkennung_model.load_state_dict(torch.load("model/faultDetection.pt"))
        zustand_erkennung_model.eval()  # Set to evaluation mode
        transform = Single_Transformer(300, 300)

        # Load and process image
        image = Image.open(full_path).convert("RGB")
        image_tensor = transform(image)

        # Move tensor to the same device as the model
        device = next(zustand_erkennung_model.parameters()).device
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            outputs = zustand_erkennung_model(image_tensor)
            _, predicted_class = torch.max(outputs, 1)

        class_map = {
            0: "Okay",
            1: "MDF_Platzer",
            2: "Rohr_Kratzer"
        }

        # status = class_map.get(predicted_class.item(), "Unknown")
        status = condition_sanity(predicted_class, part_type)
        print(f"Classification result: {status}")
        return {"status": "success", "classification": status}

    except Exception as e:
        print(f"Error in detect_condition: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def condition_sanity(prediction, type):
    scraches = ["Diagonale", "Gerade", "Griff", "Mutternstab", "Noppenscheiben", "Schraube"]
    if (prediction == 0):
        return "Okay"
    elif (type in scraches):
        return "Rohr_Kratzer"
    else:
        return "MDF_Platzer"


@app.post("/api/zustandImage_upload/")
async def zustandImage_upload(request: Request):
    data = await request.json()
    image_path = data.get("cropImage", "")
    partId = data.get("partId", "")
    partType = data.get("partType", "")

    print(f"Received Image Path: {image_path}, Part ID: {partId}")

    # Redirect to GET route with query parameters
    return RedirectResponse(url=f"/zustandImage_upload?img={image_path}&id={partId}&type={partType}", status_code=303)


@app.get("/zustandImage_upload", response_class=HTMLResponse)
async def show_upload_page(request: Request, img: str = "", id: str = "", type: str = ""):
    """Render the page and pass the image path if available."""
    return templates.TemplateResponse("statusUpload.html", {
        "request": request,
        "img_path": img,
        "part_id": id,
        "part_type": type
    })


# Main entry point
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
