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
from .approval_routes import router as approval_router
from .bill_of_materials import (
    beschaedigte_je_bauteil,
    erstelle_stueckliste,
    markiere_duplikate,
)
from .auth import router as auth_router
from .conditionDetection import ZustandModel, Single_Transformer
from .config import HERE_API_KEY
from .decorDetection import UnidekorDetector


from .decorDetection import router as color_detection_router
from .inventory_routes import router as inventory_router
from .labelStudioAdmin import router as label_studio_admin_router
from .damageDetection import router as damage_detection_router
from .modelRegistry import router as model_registry_router
from .modelTraining import router as model_training_router
from .transport_emission import router as resource_router
from .videoDetection import router as video_router


from .class_properties import CLASS_PROPERTIES
from .measurement import (
    ALL_STRAIGHT_LENGTHS,
    DIAGONAL_LENGTHS,
    SYSTEM_DEPTHS,
    TUBE_DIAMETER_MM,
    FLAECHEN_OPTIONEN,
    reference_values,
)
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
app.add_middleware(
    SessionMiddleware, secret_key=os.getenv("SECRET_KEY", "supergeheim123"))


# Setup static and template directories
app.mount("/static", StaticFiles(directory="static"), name="static")
from .templating import templates


# Ensure detected_images directory exists
os.makedirs("static/detected_images", exist_ok=True)

# Datenbank initialisieren
init_db()

# Router einbinden
app.include_router(auth_router)
app.include_router(approval_router)
app.include_router(color_detection_router)
app.include_router(inventory_router)
app.include_router(video_router)
app.include_router(model_training_router)
app.include_router(model_registry_router)
app.include_router(damage_detection_router)
app.include_router(label_studio_admin_router)
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

# Zwischenspeicher für erkannte Objekte
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

    # Wenn nur ein Update gemacht wurde, zeigen wir die aktualisierten Daten an
    if request.method == "GET" and updated_filename:
        if updated_filename in image_results_cache:
            updated_results_df = image_results_cache[updated_filename]  # DataFrame abrufen
            updated_image_results_dict = {updated_filename: updated_results_df.to_dict(orient="records")}

            return templates.TemplateResponse("results.html", {
                **_korrektur_optionen(),
                "request": request,
                "detected_images": [updated_filename],
                "image_results": updated_image_results_dict,  # KEIN JSON-String!
                "capture_type": capture_type  # Aufnahmetyp an das Template weitergeben
            })
        else:
            return HTMLResponse("Fehler: Bild nicht in Cache gefunden", status_code=404)

    # Wenn neue Bilder hochgeladen werden, führen wir eine komplette Erkennung durch
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

        # Zwischenspeicher aktualisieren
        for key, df in image_results.items():
            image_results_cache[key] = df

        # Wandelt DataFrames in ein JSON-kompatibles Dictionary um
        image_results_dict = {key: df.to_dict(orient="records") for key, df in image_results.items()}

        # Gesamtübersicht über alle Ansichten: Strukturteile aus dem Raster,
        # Einbauten je Ebene gezählt - damit nichts doppelt gezählt wird.
        try:
            stueckliste = erstelle_stueckliste(image_results, capture_type)
        except Exception as e:
            print(f"Stückliste konnte nicht erstellt werden: {e}")
            stueckliste = None

        return templates.TemplateResponse("results.html", {
            **_korrektur_optionen(),
            "request": request,
            "stueckliste": stueckliste,
            "detected_images": detected_images,
            "image_results": image_results_dict,  # KEIN JSON-String!
            "capture_type": capture_type  # Aufnahmetyp an das Template weitergeben
        })

    return templates.TemplateResponse("results.html", {**_korrektur_optionen(), "request": request})


# ---------------------------------------------------------------------------
# Manuelle Korrektur der KI-Ergebnisse
#
# Jedes Feld wird gegen eine feste Auswahlliste geprüft. Dadurch kann nur ein
# bekannter Wert in den Datensatz gelangen - freier Text (und damit HTML/JS oder
# ein Wert, der das Neo4j-Schema sprengt) wird abgewiesen, nicht bereinigt.
# ---------------------------------------------------------------------------
ZUSTAND_OPTIONEN = ["unbeschädigt", "MDF-Platzer", "Rohr_Kratzer", "Delle"]
BESCHAEDIGTE_ZUSTAENDE = ["MDF-Platzer", "Rohr_Kratzer", "Delle"]
# "Rolle" wird erkannt, steht aber nicht in CLASS_PROPERTIES.
BAUTEIL_OPTIONEN = sorted(set(CLASS_PROPERTIES) | {"Rolle"})


def _bestimme_reusable(row):
    """Wiederverwendbarkeit bestimmen, ohne eine manuelle Entscheidung zu überschreiben.

    Unbeschädigte Bauteile sind immer wiederverwendbar. Nur bei einem Schaden
    zählt, was die Person in der Korrektur angehakt hat.
    """
    if row.get("zustand") in BESCHAEDIGTE_ZUSTAENDE:
        return bool(row.get("reusable", False))
    return True

# Längen, die je Bauteil im Systemraster überhaupt vorkommen können.
LAENGEN_OPTIONEN = {
    "Gerade": ALL_STRAIGHT_LENGTHS,
    "Mutternstab": SYSTEM_DEPTHS,
    "Diagonale": DIAGONAL_LENGTHS,
}


def _lade_farboptionen():
    """Unidekore und Beschichtungen zusammen anbieten.

    Die Farberkennung nutzt colors.json, die Beschichtungserkennung
    beschichtung.json (Edelstahl, Schwarz Pulverbeschichtet). In der Korrektur
    muss beides zur Auswahl stehen, sonst lässt sich ein als "Edelstahl"
    erkanntes Rohr nicht bestätigen.
    """
    optionen = {}
    for datei in ("colors.json", "beschichtung.json"):
        try:
            with open(datei, encoding="utf-8") as f:
                optionen.update(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    return dict(sorted(optionen.items()))


def _korrektur_optionen():
    """Auswahllisten für die manuelle Korrektur an das Template geben."""
    return {
        "bauteil_optionen": BAUTEIL_OPTIONEN,
        "laengen_optionen": LAENGEN_OPTIONEN,
        "zustand_optionen": ZUSTAND_OPTIONEN,
        "beschaedigte_zustaende": BESCHAEDIGTE_ZUSTAENDE,
        "farb_optionen": _lade_farboptionen(),
        "flaechen_optionen": FLAECHEN_OPTIONEN,
        # Gewicht und Typ hängen am Bauteil und werden nicht eingegeben.
        "gewicht_map": {
            name: eigenschaften.get("gewicht", "Nicht verfügbar")
            for name, eigenschaften in CLASS_PROPERTIES.items()
        },
        "typ_map": {
            name: eigenschaften.get("typ", "Unbekannt")
            for name, eigenschaften in CLASS_PROPERTIES.items()
        },
    }


@app.post("/update_component/")
async def update_component(
    request: Request,
    filename: str = Form(...),
    bbox_id: int = Form(...),
    bauteil: str = Form(...),
    zustand: str = Form(...),
    reusable: str = Form("false"),
    laenge: str = Form(""),
    breite: str = Form(""),
    farbe: str = Form(""),
):
    """Übernimmt eine manuelle Korrektur eines erkannten Bauteils."""
    try:
        if filename not in image_results_cache:
            return JSONResponse(status_code=404, content={
                "success": False, "message": f"Datei {filename} nicht gefunden."})

        df = image_results_cache[filename]
        if bbox_id not in df["bbox_id"].values:
            return JSONResponse(status_code=404, content={
                "success": False, "message": f"Objekt-ID {bbox_id} nicht gefunden."})

        # --- Validierung gegen die Auswahllisten ---
        if bauteil not in BAUTEIL_OPTIONEN:
            return JSONResponse(status_code=400, content={
                "success": False, "message": f"Unbekanntes Bauteil: {bauteil}"})

        if zustand not in ZUSTAND_OPTIONEN:
            return JSONResponse(status_code=400, content={
                "success": False, "message": f"Unbekannter Zustand: {zustand}"})

        # Streben haben eine Länge, Flächen zwei Systemmaße.
        flaeche = FLAECHEN_OPTIONEN.get(bauteil)

        def _pruefe_mass(rohwert, erlaubt, bezeichnung):
            """Gibt (wert, fehlermeldung) zurück."""
            if not rohwert:
                return None, None
            try:
                kandidat = int(rohwert)
            except ValueError:
                return None, f"{bezeichnung} muss eine Zahl sein."
            if kandidat not in erlaubt:
                return None, f"{bezeichnung} {kandidat} ist für {bauteil} nicht im Systemraster."
            return kandidat, None

        if flaeche:
            laenge_wert, fehler = _pruefe_mass(laenge, flaeche["laenge"], flaeche["laenge_label"])
            if fehler:
                return JSONResponse(status_code=400, content={"success": False, "message": fehler})
            breite_wert, fehler = _pruefe_mass(breite, flaeche["breite"], flaeche["breite_label"])
            if fehler:
                return JSONResponse(status_code=400, content={"success": False, "message": fehler})
        else:
            laenge_wert, fehler = _pruefe_mass(laenge, LAENGEN_OPTIONEN.get(bauteil, []), "Länge")
            if fehler:
                return JSONResponse(status_code=400, content={"success": False, "message": fehler})
            # Die Breite einer Strebe ist immer der Rohrdurchmesser.
            breite_wert = TUBE_DIAMETER_MM if laenge_wert else None

        farb_daten = None
        if farbe:
            farboptionen = _lade_farboptionen()
            if farbe not in farboptionen:
                return JSONResponse(status_code=400, content={
                    "success": False, "message": f"Unbekannte Farbe: {farbe}"})
            eintrag = farboptionen[farbe]
            farb_daten = {
                "erkannte_farbe": farbe,
                "hex_code": eintrag.get("hex_code", "#CCCCCC"),
                "ncs_code": eintrag.get("ncs_code", "N/A"),
                # Manuell gesetzt: volle Sicherheit statt KI-Konfidenz.
                "confidence": 1.0,
            }

        ist_beschaedigt = zustand in BESCHAEDIGTE_ZUSTAENDE
        # Unbeschädigte Bauteile sind immer wiederverwendbar; nur bei einem
        # Schaden entscheidet die Person, ob das Teil noch nutzbar ist.
        ist_wiederverwendbar = (reusable == "true") if ist_beschaedigt else True

        # --- Übernehmen ---
        maske = df["bbox_id"] == bbox_id
        eigenschaften = CLASS_PROPERTIES.get(bauteil, {})

        # Nur Streben tragen ihre Länge im Namen ("Gerade 540"), Flächen nicht.
        anzeigename = f"{bauteil} {laenge_wert}" if (laenge_wert and not flaeche) else bauteil
        df.loc[maske, "class"] = anzeigename
        df.loc[maske, "zustand"] = zustand
        df.loc[maske, "reusable"] = ist_wiederverwendbar
        # Gewicht und Typ folgen dem gewählten Bauteil, nicht der Eingabe.
        df.loc[maske, "gewicht"] = eigenschaften.get("gewicht", "Nicht verfügbar")
        df.loc[maske, "typ"] = eigenschaften.get("typ", "Unbekannt")

        if laenge_wert or breite_wert:
            df.loc[maske, "laenge"] = laenge_wert
            df.loc[maske, "breite"] = breite_wert
            df.loc[maske, "maße"] = (
                f"{breite_wert} x {laenge_wert} mm" if breite_wert and laenge_wert else None
            )
        else:
            # Bauteil ohne Systemlänge: Maß des alten Bauteils darf nicht
            # stehen bleiben. Feste Referenzmaße (z.B. Noppenscheibe 30x30)
            # übernehmen, sonst leeren.
            referenz = reference_values.get(bauteil, {})
            ref_breite = referenz.get("width_mm")
            ref_laenge = referenz.get("height_mm")
            df.loc[maske, "breite"] = ref_breite
            df.loc[maske, "laenge"] = ref_laenge
            df.loc[maske, "maße"] = (
                f"{ref_breite} x {ref_laenge} mm" if ref_breite and ref_laenge else None
            )
        if farb_daten:
            df.loc[maske, "farbe"] = [farb_daten] * int(maske.sum())

        # Manuelle Korrektur schlägt jede Messung: als Quelle festhalten.
        if "mass_quelle" in df.columns:
            df.loc[maske, "mass_quelle"] = "manuell korrigiert"

        image_results_cache[filename] = df

        return JSONResponse(status_code=200, content={
            "success": True,
            "message": "Korrektur übernommen.",
            "bauteil": anzeigename,
            "gewicht": eigenschaften.get("gewicht", "Nicht verfügbar"),
            "typ": eigenschaften.get("typ", "Unbekannt"),
            "reusable": ist_wiederverwendbar,
            "farbe": farb_daten,
            "breite": breite_wert,
            "laenge": laenge_wert,
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False, "message": f"Fehler: {str(e)}"})


@app.post("/update_condition/")
async def update_condition(
    request: Request,
    filename: str = Form(...),
    bbox_id: int = Form(...),
    zustand: str = Form(...)
):
    try:
        #print(f"Datei: {filename},  bbox_id: {bbox_id},  Neuer Zustand: {zustand}")

        if filename not in image_results_cache:
            return JSONResponse(status_code=404, content={"success": False, "message": f"Datei {filename} nicht gefunden."})

        df = image_results_cache[filename]

        if bbox_id not in df["bbox_id"].values:
            return JSONResponse(status_code=404, content={"success": False, "message": f"Objekt-ID {bbox_id} nicht gefunden."})

        df.loc[df["bbox_id"] == bbox_id, "zustand"] = zustand
        is_damaged = zustand in ["MDF-Platzer", "Rohr_Kratzer", "Delle"]
        df.loc[df["bbox_id"] == bbox_id, "reusable"] = not is_damaged
        image_results_cache[filename] = df

        print(f"Aktualisiertes DataFrame für {filename}:\n{df}")

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
            df["reusable"] = df.apply(_bestimme_reusable, axis=1)

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
                df["reusable"] = df.apply(_bestimme_reusable, axis=1)

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
    """Reicht die erfassten Ergebnisse zur Prüfung ein."""

    # Debugging: Prüfen, ob image_results wirklich ankommt
    if not image_results:
        return {"error": "image_results wurde nicht übermittelt"}

    # JSON-String in Dictionary umwandeln
    try:
        image_results_dict = json.loads(image_results)
    except json.JSONDecodeError as e:
        return {"error": f"Fehler beim Parsen von image_results: {str(e)}"}

    # JSON-Dictionary in DataFrames konvertieren
    image_results_df = {key: pd.DataFrame(value) for key, value in image_results_dict.items()}

    # Ergebnisse speichern - zunächst unbestätigt, bis System 180 freigibt.
    #
    # Gespeichert wird ein Mischmodell: Sammelpositionen der Stückliste tragen
    # die Anzahl (und damit den Bestand), einzelne Erkennungen bleiben mit
    # Ausschnitt und Zustand als Beleg erhalten und sind als Dublette markiert,
    # wenn sie dasselbe Bauteil aus einer zweiten Ansicht zeigen.
    capture_type = "single"
    for datensatz in image_results_dict.values():
        if isinstance(datensatz, list) and datensatz:
            capture_type = datensatz[0].get("capture_type") or "single"
            break

    stueckliste = None
    beschaedigt = {}
    try:
        image_results_df = markiere_duplikate(image_results_df)
        stueckliste = erstelle_stueckliste(image_results_df, capture_type)
        beschaedigt = beschaedigte_je_bauteil(image_results_df)
    except Exception as e:
        print(f"Stückliste konnte nicht erstellt werden: {e}")

    try:
        if stueckliste:
            db.store_stueckliste(
                stueckliste=stueckliste,
                image_results=image_results_df,
                beschaedigt_je_bauteil=beschaedigt,
                order_id=order_id,
                system180_order=system180_order,
                contact_email=contact_email,
                contact_phone=contact_phone,
                location=location,
                latitude=latitude,
                longitude=longitude,
                formatted_address=formatted_address,
                order_type=order_type,
                additional_info=additional_info,
            )
            message = ("Auftrag erfolgreich eingereicht. Nach der Prüfung durch System 180 "
                       "erscheinen die Bauteile im digitalen Lager.")
            success = True
            return templates.TemplateResponse("confirmation.html", {
                "request": request, "success": success, "message": message})

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
        message = ("Auftrag erfolgreich eingereicht. Nach der Prüfung durch System 180 "
                   "erscheinen die Bauteile im digitalen Lager.")
        success = True
    except Exception as e:
        message = f"Fehler beim Einreichen: {str(e)}"
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
        **_korrektur_optionen(),
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
