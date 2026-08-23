# processImage.py

import io
import os
import uuid
import cv2

from datetime import datetime
from fastapi import UploadFile
import numpy as np
import pandas as pd
from PIL import Image


from .class_properties import CLASS_PROPERTIES
from . import config
from .config import color_detection_classes, beschichtung_detection_classes
from .damageDetection import inspect_crop
from .decorDetection import UnidekorDetector
from .measurement import calculate_pixel_to_mm_ratio, calculate_straight_lengths, reference_values
from .trainingDataCollector import TrainingDataCollector
from .visualize import get_color_for_class, draw_bounding_boxes
from .combineYOLOModels import ensemble_predictions
from .beschichtungDetection import BeschichtungDetector


import base64
import cv2


def encode_image_to_base64(image_np):
    """Konvertiert ein NumPy-Array in einen Base64-String für die UI."""
    _, buffer = cv2.imencode(".jpg", image_np)
    return base64.b64encode(buffer).decode("utf-8")



def process_images(files: list[UploadFile], class_confidence=0.4, save_images: bool=False, views=None, capture_type="single"):
    """Verarbeitet hochgeladene Bilder mit YOLO und speichert Ergebnisse in separaten DataFrames pro Bild."""

    beschichtung_detector = BeschichtungDetector()
    color_detector = UnidekorDetector()
    training_collector = TrainingDataCollector()

    detected_images = []  # Liste der Originalbilder
    image_results = {}  # Dictionary für DataFrames pro Bild
    views = views or {}
    print(f"Verarbeite Bilder mit views: {views}")  # Debug-Log
    print(f"Ist symmetrisch: {capture_type}")  # Debug-Log

    original_images_dir = os.path.join("static", "detected_images", "original_images")
    processed_images_dir = os.path.join("static", "detected_images", "processed_images")
    crops_base_dir = os.path.join("static", "detected_images", "crops")


    os.makedirs(original_images_dir, exist_ok=True)
    os.makedirs(processed_images_dir, exist_ok=True)



    for file in files:
        try:
            # 📌 **Bild laden**
            image_bytes = file.file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            original_filename = file.filename.replace(' ', '_')
            original_filename_base, original_extension = os.path.splitext(original_filename)
            original_path = os.path.join(original_images_dir, original_filename)
            image.save(original_path)  # Speichert das Originalbild
            detected_images.append(original_filename)

            # Ansicht aus dem Views-Dictionary abrufen
            view = views.get(original_filename, "any")  # Default auf "any" statt "unbekannt"
            print(f"Verwende View für {original_filename}: {view}")  # Debug-Log zur Verfolgung des View-Werts


            crop_dir = os.path.join(crops_base_dir, original_filename_base)
            os.makedirs(crop_dir, exist_ok=True)


            image_np = np.array(image)
            #image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            # 📌 **YOLO Vorhersage mit zwei Modellen**
            results1 = config.model1.predict(image, save=False, conf=class_confidence)
            results2 = config.model2.predict(image, save=False, conf=class_confidence)

            # 📌 **Ensemble-Methode kombiniert beide YOLO Ergebnisse**
            final_boxes, final_scores, final_classes = ensemble_predictions(results1, results2)

            detected_data = []  # Neu initialisieren für jedes Bild!

            # 📌 **Bounding Box Daten erfassen**
            for i in range(len(final_boxes)):
                x1, y1, x2, y2 = map(int, final_boxes[i])
                conf = float(final_scores[i])
                class_id = int(final_classes[i])
                class_name = results1[0].names[class_id]

                # Create crop of the detected object
                crop = image_np[y1:y2, x1:x2]
                crop_image = Image.fromarray(crop)

                if crop.size == 0:
                    print(f"⚠️ Fehler: Crop für {class_name} ist leer! BBox: ({x1}, {y1}, {x2}, {y2})")

                # Generate unique filename for crop
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                crop_filename = f"{original_filename_base}_{timestamp}_{i}_crop.jpg"
                crop_path = os.path.join(crop_dir, crop_filename)
                crop_path_rel = os.path.relpath(crop_path, "static")

                try:
                    crop_image.save(crop_path)
                except Exception as e:
                    print(f"❌ Fehler beim Speichern des Crops: {e}")

                # Standardwerte setzen
                properties = CLASS_PROPERTIES.get(class_name, {
                    "gewicht": "Nicht verfügbar",
                    "typ": "Unbekannt",
                    "zustand": "Unbekannt",
                })

                # 📌 Schadensprüfung auf dem Crop: Schadensmodell (falls
                # hochgeladen) + Verdachts-Ampel (falls Gutteil-Banks trainiert)
                zustand = properties.get("zustand", "Unbekannt")
                reusable = True
                inspection = inspect_crop(class_name, crop)
                if inspection["zustand"]:
                    zustand = inspection["zustand"]
                    reusable = False

                farbe = None


                if class_name in color_detection_classes:
                    try:
                        absolute_crop_path = os.path.join("static", crop_path)
                        if os.path.exists(absolute_crop_path):
                            color_result = color_detector.analyze_image(absolute_crop_path)
                            farbe = color_result.get('erkannte_farbe', "Nicht verfügbar")
                    except Exception as e:
                        print(f"❌ Fehler bei Farberkennung für {class_name}: {e}")

                if class_name in beschichtung_detection_classes:
                    try:
                        absolute_crop_path = os.path.join("static", crop_path)
                        if os.path.exists(absolute_crop_path):
                            beschichtung_result = beschichtung_detector.analyze_image(absolute_crop_path)
                            farbe = beschichtung_result.get('erkannte_farbe', "Nicht verfügbar")
                    except Exception as e:
                        print(f"❌ Fehler bei Beschichtungserkennung für {class_name}: {e}")

                detected_data.append({
                    "bbox_id": i,
                    "class": class_name,
                    "confidence": conf,
                    "x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2,
                    "crop_path": crop_path_rel,
                    "maße": None,
                    "farbe": farbe,
                    "typ": properties.get("typ", "Unbekannt"),
                    "zustand": zustand,
                    "zustand_conf": inspection["zustand_conf"],
                    "verdacht": inspection["verdacht"],
                    "anomalie_score": inspection["anomalie_score"],
                    "gewicht": properties.get("gewicht", "Nicht verfügbar"),
                    "reusable": reusable,
                    "ansicht": view,  # Neue Spalte für die Bildansicht
                    "capture_type": capture_type
                })



            # 📌 **Erstelle DataFrame für das aktuelle Bild**
            df_boxes = pd.DataFrame(detected_data)

            # 📌 **Maßstab berechnen & zuweisen**
            scaling_factors = calculate_pixel_to_mm_ratio(df_boxes)
            df_boxes = calculate_straight_lengths(df_boxes, pixel_to_mm_ratio=scaling_factors.get("Gerade", 1))

            # Falls die Spalte nicht existiert oder leer ist, initialisiere sie mit None
            if "breite" not in df_boxes.columns:
                df_boxes["breite"] = None
            if "laenge" not in df_boxes.columns:
                df_boxes["laenge"] = None

            # 📌 **Maße für Referenzobjekte zuweisen**
            for index, row in df_boxes.iterrows():
                class_name = row["class"]
                if class_name in reference_values:
                    width_mm = reference_values[class_name].get("width_mm")
                    height_mm = reference_values[class_name].get("height_mm")

                    # Setze Werte nur, wenn sie nicht bereits vorhanden sind
                    df_boxes.at[index, "breite"] = width_mm if pd.isna(row["breite"]) else row["breite"]
                    df_boxes.at[index, "laenge"] = height_mm if pd.isna(row["laenge"]) else row["laenge"]


            # 📌 **Maße als formatierten String hinzufügen**
            df_boxes["maße"] = df_boxes.apply(lambda row: f"{row['breite']} x {row['laenge']} mm" if pd.notna(row["breite"]) and pd.notna(row["laenge"]) else None, axis=1)





            # 📌 **Farberkennung hinzufügen**
            for index, row in df_boxes.iterrows():
                if row["class"] in color_detection_classes:
                    try:
                        crop_path = row["crop_path"]
                        absolute_crop_path = os.path.join("static", crop_path)  # Absoluten Pfad erstellen

                        if not os.path.exists(absolute_crop_path):
                            print(f"⚠️ Datei existiert nicht: {absolute_crop_path}")
                            df_boxes.at[index, "farbe"] = "Fehler: Datei fehlt"
                            continue  # Springe zur nächsten Zeile, wenn die Datei fehlt

                        # Farbanalyse mit absolutem Pfad
                        color_result = color_detector.analyze_image(absolute_crop_path)
                        df_boxes.at[index, "farbe"] = {
                            "erkannte_farbe": color_result.get('erkannte_farbe', "Nicht verfügbar"),
                            "hex_code": color_result.get('hex_code', "#CCCCCC"),
                            "ncs_code": color_result.get('ncs_code', "N/A"),
                            "confidence": color_result.get('confidence', 0.0)
                        }



                    except Exception as e:
                        print(f"❌ Fehler bei Farberkennung für {row['class']}: {e}")
                        df_boxes.at[index, "farbe"] = "Fehler bei Farberkennung"

                if row["class"] in beschichtung_detection_classes:
                    try:
                        crop_path = row["crop_path"]
                        absolute_crop_path = os.path.join("static", crop_path)  # Absoluten Pfad erstellen

                        if not os.path.exists(absolute_crop_path):
                            print(f"⚠️ Datei existiert nicht: {absolute_crop_path}")
                            df_boxes.at[index, "farbe"] = "Fehler: Datei fehlt"
                            continue  # Springe zur nächsten Zeile, wenn die Datei fehlt

                        # Farbanalyse mit absolutem Pfad
                        color_result = beschichtung_detector.analyze_image(absolute_crop_path)
                        df_boxes.at[index, "farbe"] = {
                            "erkannte_farbe": color_result.get('erkannte_farbe', "Nicht verfügbar"),
                            "hex_code": color_result.get('hex_code', "#CCCCCC"),
                            "ncs_code": color_result.get('ncs_code', "N/A"),
                            "confidence": color_result.get('confidence', 0.0)
                        }
                    except Exception as e:
                        print(f"❌ Fehler bei Beschichtungsrkennung für {row['class']}: {e}")
                        df_boxes.at[index, "farbe"] = "Fehler bei Beschichtungsrkennung"



                if row["class"] == "Gerade":
                    try:
                        df_boxes.at[index, "class"] = f"{row['class']} {row['laenge']}"

                    except Exception as e:
                        print(f"❌ Fehler bei Geradenbenennung für {row['class']}: {e}")




            # 📌 **Speichere DataFrame für das aktuelle Bild**
            image_results[original_filename] = df_boxes
            print(f"File {original_filename}: {df_boxes}")

            image_np = draw_bounding_boxes(image_np, final_boxes, final_scores, final_classes, results1[0].names, df_boxes)


            # 📌 **Speichere das Bild mit Bounding Boxen**
            processed_image_path = os.path.join(processed_images_dir, original_filename)
            cv2.imwrite(processed_image_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"Fehler beim Verarbeiten von {file.filename}: {e}")
            continue

    #print(f"processImage - Image Results: {image_results.keys()}")  # Debug-Print

    return detected_images, image_results  # Dictionary mit DataFrames pro Bild zurückgeben

