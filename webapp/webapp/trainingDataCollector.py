"""Sammelt Kundenaufnahmen samt Labels als Trainingsmaterial.

Jede Erfassung liefert ein Bild und die dazugehörigen Erkennungen. Beides wird
abgelegt, damit der Datenbestand mitwächst und das Modell später darauf
nachtrainiert werden kann:

* ``images/`` das Originalbild,
* ``labels/`` die Erkennungen im YOLO-Format (direkt zum Nachtrainieren),
* ``meta/``   dieselben Angaben als JSON mit Klasse, Zustand, Farbe und Maßen.

Das Bild wird bewusst nur hier abgelegt. Label Studio liest denselben Ordner
über einen eigenen Mount (siehe compose.yaml), sodass keine zweite Kopie
entsteht.
"""
import json
import os
import shutil
from datetime import datetime

# Kundenaufnahmen liegen neben den übrigen Label-Studio-Daten und überstehen
# damit einen Rebuild des Containers (siehe Mount in compose.yaml).
BASE_DIR_UPLOADED_TRAININGSDATA = os.getenv(
    "TRAINING_DATA_DIR", "./labelstudio-data/saved_customer_files")


class TrainingDataCollector:
    def __init__(self, base_dir=BASE_DIR_UPLOADED_TRAININGSDATA):
        self.base_dir = base_dir
        self.images_dir = os.path.join(base_dir, "images")
        self.labels_dir = os.path.join(base_dir, "labels")
        self.meta_dir = os.path.join(base_dir, "meta")
        self._create_directories()

    def _create_directories(self):
        for ordner in (self.images_dir, self.labels_dir, self.meta_dir):
            os.makedirs(ordner, exist_ok=True)

    def save_training_data(self, original_image_path, df_boxes, original_filename,
                           bild_breite, bild_hoehe, class_ids=None):
        """Legt Bild, YOLO-Labels und Metadaten einer Erfassung ab.

        ``class_ids`` bildet Klassennamen auf die IDs des aktiven Modells ab.
        Fehlt eine Klasse dort, wird sie im YOLO-Label übersprungen - im JSON
        bleibt sie aber vollständig erhalten.
        """
        if not bild_breite or not bild_hoehe:
            return None

        class_ids = class_ids or {}
        zeitstempel = datetime.now().strftime("%Y%m%d_%H%M%S")
        stamm = f"{zeitstempel}_{os.path.splitext(original_filename)[0]}"

        bild_ziel = os.path.join(self.images_dir, f"{stamm}{os.path.splitext(original_filename)[1]}")
        shutil.copy2(original_image_path, bild_ziel)

        yolo_zeilen, objekte = [], []
        for _, zeile in df_boxes.iterrows():
            # Der Klassenname trägt bei Streben die Länge ("Gerade 540") - für
            # das Label zählt die Grundklasse, die das Modell kennt.
            klasse = str(zeile.get("class") or "")
            grundklasse = klasse.split(" ")[0] if klasse.split(" ")[0] in class_ids else klasse

            x_min, y_min = float(zeile.get("x_min", 0)), float(zeile.get("y_min", 0))
            x_max, y_max = float(zeile.get("x_max", 0)), float(zeile.get("y_max", 0))
            breite, hoehe = x_max - x_min, y_max - y_min
            if breite <= 0 or hoehe <= 0:
                continue

            if grundklasse in class_ids:
                yolo_zeilen.append(
                    f"{class_ids[grundklasse]} "
                    f"{((x_min + x_max) / 2) / bild_breite:.6f} "
                    f"{((y_min + y_max) / 2) / bild_hoehe:.6f} "
                    f"{breite / bild_breite:.6f} "
                    f"{hoehe / bild_hoehe:.6f}"
                )

            objekte.append({
                "class": klasse,
                "grundklasse": grundklasse,
                "bbox": [x_min, y_min, x_max, y_max],
                "confidence": float(zeile.get("confidence") or 0),
                "zustand": zeile.get("zustand"),
                "reusable": bool(zeile.get("reusable")),
                "farbe": zeile.get("farbe"),
                "breite_mm": zeile.get("breite"),
                "laenge_mm": zeile.get("laenge"),
                "ansicht": zeile.get("ansicht"),
                "crop_path": zeile.get("crop_path"),
            })

        with open(os.path.join(self.labels_dir, f"{stamm}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_zeilen))

        with open(os.path.join(self.meta_dir, f"{stamm}.json"), "w", encoding="utf-8") as f:
            json.dump({
                "bild": os.path.relpath(bild_ziel, self.base_dir),
                "original_filename": original_filename,
                "zeitstempel": zeitstempel,
                "bild_breite": bild_breite,
                "bild_hoehe": bild_hoehe,
                "objekte": objekte,
            }, f, indent=2, ensure_ascii=False, default=str)

        return {"bild": bild_ziel, "objekte": len(objekte), "yolo_labels": len(yolo_zeilen)}
