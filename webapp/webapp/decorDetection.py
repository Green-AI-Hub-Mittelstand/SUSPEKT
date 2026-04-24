#decorDetection.py
import base64
from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import os
import numpy as np
from fastapi import UploadFile, APIRouter, HTTPException, Form, Request
from pydantic import BaseModel
from sklearn.cluster import KMeans
from fastapi.templating import Jinja2Templates
import json

from fastapi.responses import HTMLResponse

@dataclass
class UnidekorFarbe:
    """Klasse für die Eigenschaften eines Unidekors"""
    name: str
    hex_code: str
    ncs_code: str
    rgb_values: Tuple[int, int, int]




class DetectionResult(BaseModel):
    erkannte_farbe: str
    hex_code: str
    ncs_code: str
    confidence: float
    erkannte_rgb: Tuple[int, int, int]
    original_image: str
    filename: str


class ColorInfo(BaseModel):
    hex_code: str
    ncs_code: str
    rgb_values: Tuple[int, int, int]


class UnidekorDetector:
    COLORS_FILE = "colors.json"  # Datei zur Speicherung der Farben

    def __init__(self):
        self.unidekore = self.load_colors()

    def load_colors(self) -> Dict[str, UnidekorFarbe]:
        """Lädt die Farben aus der JSON-Datei"""
        if os.path.exists(self.COLORS_FILE):
            with open(self.COLORS_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)
                return {name: UnidekorFarbe(**details) for name, details in data.items()}
        return {}

    def save_colors(self):
        """Speichert die Farben in die JSON-Datei"""
        with open(self.COLORS_FILE, "w", encoding="utf-8") as file:
            json.dump({name: vars(color) for name, color in self.unidekore.items()}, file, indent=4)

    def add_color(self, name: str, hex_code: str, ncs_code: str):
        """Fügt eine neue Farbe hinzu und speichert sie dauerhaft"""
        self.unidekore[name] = UnidekorFarbe(
            name=name,
            hex_code=hex_code,
            ncs_code=ncs_code,
            rgb_values=self._hex_to_rgb(hex_code)
        )
        self.save_colors()  # Direkt in JSON speichern


    @staticmethod
    def _hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
        """Konvertiert einen Hex-Farbcode in RGB-Werte"""
        hex_code = hex_code.lstrip("#")
        return tuple(int(hex_code[i:i + 2], 16) for i in (0, 2, 4))


    @staticmethod
    def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
        """Konvertiert RGB-Werte in einen Hex-Farbcode"""
        return '#{:02x}{:02x}{:02x}'.format(*rgb)

    def _calculate_color_distance(self, color1: Tuple[int, int, int],
                                  color2: Tuple[int, int, int]) -> float:
        """Berechnet den euklidischen Abstand zwischen zwei Farben im RGB-Raum"""
        return sum((a - b) ** 2 for a, b in zip(color1, color2)) ** 0.5

    async def analyze_uploaded_image(self, file: UploadFile) -> DetectionResult:
        # Lese das Bild
        contents = await file.read()

        # Erstelle Base64 für Vorschau
        base64_image = base64.b64encode(contents).decode('utf-8')

        # Konvertiere zu OpenCV Format
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Führe Analyse durch
        result = self.analyze_image(image)

        # print(result)

        # Erstelle DetectionResult
        return DetectionResult(
            erkannte_farbe=result['erkannte_farbe'],
            hex_code=result['hex_code'],
            ncs_code=result['ncs_code'],
            confidence=result['confidence'],
            erkannte_rgb=result['erkannte_rgb'],
            original_image=base64_image,
            filename=file.filename
        )

    # TODO: C901 'UnidekorDetector.analyze_image' is too complex
    def analyze_image(self, image_path: str) -> Dict:
        """
        Verbesserte Bildanalyse mit optimierter Erkennung heller Farben
        """

        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Bild konnte nicht geladen werden")

        # Konvertiere zu RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        """
        if not os.path.exists(image_path):
            raise ValueError(f"⚠ Fehler: Datei existiert nicht: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"⚠ Fehler: Bild konnte nicht geladen werden ({image_path})")

        # Konvertiere zu RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Forme das Bild in eine Liste von Pixeln um
        pixels = image_rgb.reshape(-1, 3)

        # Führe K-Means Clustering durch
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(pixels)

        # Hole die Cluster-Zentren und ihre Häufigkeiten
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        color_counts = np.bincount(labels)

        # Berechne den prozentualen Anteil jeder Farbe
        total_pixels = len(labels)
        color_percentages = color_counts / total_pixels

        # Sortiere die Farben nach Häufigkeit
        sorted_indices = np.argsort(color_percentages)[::-1]
        dominant_colors = colors[sorted_indices]
        dominant_percentages = color_percentages[sorted_indices]

        # Optimierte Farbfilterung
        filtered_colors = []
        filtered_percentages = []

        for color, percentage in zip(dominant_colors, dominant_percentages):
            brightness = np.mean(color)
            saturation = np.std(color)  # Standardabweichung als Maß für Sättigung

            # Spezielle Behandlung für helle Farben
            if brightness > 220 and percentage > 0.25:  # Für sehr helle Farben
                # Prüfe, ob es sich um Weiß handelt
                if saturation < 10:  # Geringe Farbsättigung deutet auf Weiß hin
                    filtered_colors.insert(0, color)  # Priorität für Weiß
                    filtered_percentages.insert(0, percentage)
                else:
                    filtered_colors.append(color)
                    filtered_percentages.append(percentage)
            # Normale Farbfilterung für andere Farben
            elif 30 < brightness < 240 and saturation > 3:  # Mindest-Sättigung für Farben
                filtered_colors.append(color)
                filtered_percentages.append(percentage)

        if not filtered_colors:
            # Fallback: Nimm die häufigste Farbe
            filtered_colors = [dominant_colors[0]]
            filtered_percentages = [dominant_percentages[0]]

        # Wähle die dominante Farbe basierend auf Häufigkeit und Sättigung
        dominant_color = None
        max_score = -1

        for color, percentage in zip(filtered_colors, filtered_percentages):
            saturation = np.std(color)
            # Score basierend auf Häufigkeit und Sättigung
            score = percentage * (1 + saturation / 255)
            if score > max_score:
                max_score = score
                dominant_color = color

        dominant_color = tuple(map(int, dominant_color))

        # Verbesserte Farbzuordnung mit Gewichtung für Helligkeit
        best_match = None
        min_distance = float('inf')

        for unidekor in self.unidekore.values():
            # Berechne gewichtete Distanz
            base_distance = self._calculate_color_distance(dominant_color,
                                                           unidekor.rgb_values)
            # Zusätzliche Gewichtung für Helligkeitsunterschiede
            brightness_diff = abs(
                np.mean(dominant_color) - np.mean(unidekor.rgb_values))
            # Gesamtdistanz mit stärkerer Berücksichtigung der Helligkeit
            weighted_distance = base_distance + brightness_diff * 0.5

            if weighted_distance < min_distance:
                min_distance = weighted_distance
                best_match = unidekor

        # Berechne Konfidenz
        max_distance = 441.67
        confidence = 1 - (min_distance / max_distance)

        return {
            'erkannte_farbe': best_match.name,
            'hex_code': best_match.hex_code,
            'ncs_code': best_match.ncs_code,
            'confidence': round(confidence, 2),
            'erkannte_rgb': dominant_color,
            'debug': {
                'dominant_color': dominant_color,
                'filtered_colors': [tuple(map(int, c)) for c in filtered_colors[:3]],
                'matched_color': best_match.rgb_values,
                'color_percentages': [round(p, 3) for p in filtered_percentages[:3]]
            }
        }


    def get_all_colors(self) -> Dict[str, ColorInfo]:
        """Get all available Unidekor colors"""
        return {name: ColorInfo(
            hex_code=unidekor.hex_code,
            ncs_code=unidekor.ncs_code,
            rgb_values=unidekor.rgb_values
        ) for name, unidekor in self.unidekore.items()}


# Create router
router = APIRouter(
    prefix="/decordetection",
    tags=["decordetection"],
    responses={404: {"description": "Not found"}},
)

templates = Jinja2Templates(directory="templates")


detector = UnidekorDetector()


@router.post("/", response_model=DetectionResult)
async def analyze_image(file: UploadFile):
    """
    Analyze an uploaded image and detect its color
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        result = await detector.analyze_image(file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/colors", response_model=Dict[str, ColorInfo])
async def get_available_colors():
    """
    Get all available Unidekor colors
    """
    return detector.get_all_colors()

@router.get("/manage_colors", response_class=HTMLResponse)
async def manage_colors(request: Request):
    colors = detector.get_all_colors()
    return templates.TemplateResponse("manage_colors.html", {"request": request, "colors": colors})


@router.post("/add_color/")
async def add_color(
    name: str = Form(...),
    hex_code: str = Form(...),
    ncs_code: str = Form(...)
):
    detector.add_color(name, hex_code, ncs_code)  # Speichert in JSON
    return {"message": f"Farbe {name} erfolgreich hinzugefügt"}
