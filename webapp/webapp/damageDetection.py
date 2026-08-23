"""Schadensprüfung auf Bauteil-Crops: Verdachts-Ampel + Schadensmodell.

Zwei sich ergänzende Prüfungen, beide laufen nach der Komponentenerkennung
auf dem Ausschnitt (Crop) jedes erkannten Bauteils:

1. **Anomalie-Erkennung (Verdachts-Ampel)** — PatchCore-light in reinem
   PyTorch: Patch-Features eines vortrainierten ResNet18 (layer2+layer3)
   werden pro Bauteilklasse in einer "Memory Bank" aus GUTTEILEN gespeichert.
   Ein neuer Crop gilt als auffällig, wenn seine Patches weit von allen
   bekannten Gutteil-Patches entfernt liegen. Es werden also NUR Bilder
   unbeschädigter Teile benötigt — ideal, solange kaum Fehlerbilder
   existieren. Training per Knopfdruck auf der /training-Seite; die Banks
   liegen in model/anomaly/ (im Docker-Volume persistiert).

2. **Schadensmodell** — ein YOLO-Modell mit den Klassen MDF-Platzer,
   Rohr_Kratzer und Delle (Projekt "Schäden" im Label-Studio-Kreislauf).
   Solange keines hochgeladen wurde, wird dieser Teil einfach übersprungen.
   Treffer werden gegen die pro Bauteiltyp physikalisch möglichen
   Schadensarten gefiltert (Rohre kriegen Kratzer, Platten MDF-Platzer).
"""

import os
import re
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

# Torchvision-Gewichte im gemounteten model/-Volume cachen, damit der
# einmalige ResNet18-Download Container-Neustarts überlebt.
os.environ.setdefault("TORCH_HOME", os.path.join("model", "torch_cache"))

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .auth import is_admin
from .modelRegistry import get_active_model, get_active_model_path

router = APIRouter(prefix="/training/anomaly")

ANOMALY_DIR = os.path.join("model", "anomaly")

# Wo die Gutbilder herkommen: derselbe Ordner, aus dem auch das
# Komponenten-Labeling gespeist wird (docker: /label-studio/files/komponenten,
# lokal: ./training_captures/komponenten).
_FILES_ROOT = os.getenv("LABEL_STUDIO_FILES_ROOT", "/label-studio/files")
GOOD_IMAGE_DIRS = [
    os.path.join(_FILES_ROOT, "komponenten"),
    os.path.join("training_captures", "komponenten"),
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MIN_CROPS_PER_CLASS = 15   # darunter ist keine verlässliche Bank möglich
MAX_BANK_PATCHES = 4000    # Patches pro Klasse (Speicher/Geschwindigkeit)
PATCHES_PER_CROP = 256     # Zufallsauswahl pro Trainings-Crop
CROP_CONF = 0.4            # wie class_confidence in processImage
MIN_CROP_PX = 40           # zu kleine Crops sind featurelos

# Bauteiltyp → physikalisch mögliche Schäden (gleiche Logik wie
# condition_sanity in model.py: Rohr-Klassen kratzen, Platten platzen).
TUBE_CLASSES = {"Gerade", "Diagonale", "Griff", "Mutternstab",
                "Noppenscheiben", "Schraube", "Sockelfuss"}


def allowed_damages(class_name: str) -> set[str]:
    base = class_name.split()[0] if class_name else ""
    if base in TUBE_CLASSES:
        return {"Rohr_Kratzer", "Delle"}
    return {"MDF-Platzer", "Delle"}


# ------------------------------------------------------------ progress ----

_progress_lock = threading.Lock()
_progress = {"state": "idle", "message": "", "done": 0, "total": 0}


def _set_progress(**kwargs) -> None:
    with _progress_lock:
        _progress.update(kwargs)


def _start_job(target) -> bool:
    with _progress_lock:
        if _progress["state"] == "running":
            return False
        _progress.update(state="running", message="Starte …", done=0, total=0)
    threading.Thread(target=target, daemon=True).start()
    return True


# ------------------------------------------------- feature extraction ----

_extractor_lock = threading.Lock()
_extractor = None


def _get_extractor():
    """Lazy singleton: vortrainiertes ResNet18 im eval-Modus (CPU)."""
    global _extractor
    with _extractor_lock:
        if _extractor is None:
            from torchvision.models import ResNet18_Weights, resnet18
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
            model.eval()
            _extractor = model
        return _extractor


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _patch_features(crop_rgb: np.ndarray):
    """RGB-Crop → Patch-Feature-Matrix (1024×384) aus layer2+layer3."""
    import cv2
    import torch
    import torch.nn.functional as F

    img = cv2.resize(crop_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    img = (img.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    m = _get_extractor()
    with torch.no_grad():
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        x = m.layer1(x)
        f2 = m.layer2(x)                                # 1×128×32×32
        f3 = m.layer3(f2)                               # 1×256×16×16
        f3 = F.interpolate(f3, size=f2.shape[-2:],
                           mode="bilinear", align_corners=False)
        f = torch.cat([f2, f3], dim=1)                  # 1×384×32×32
        f = F.avg_pool2d(f, 3, stride=1, padding=1)     # lokale Glättung
    return f.squeeze(0).permute(1, 2, 0).reshape(-1, f.shape[1])


def _crop_score(patches, bank, ignore_self: bool = False) -> float:
    """Anomalie-Score eines Crops: 99%-Quantil der Nächste-Nachbar-Distanzen
    seiner Patches zur Gutteil-Bank (robuster als das reine Maximum)."""
    import torch
    dists = torch.cdist(patches, bank)
    if ignore_self:
        # Beim Threshold-Bestimmen liegen eigene Patches mit in der Bank —
        # exakte Treffer (Distanz ≈ 0) auf den zweitnächsten ausweichen.
        vals, _ = dists.topk(2, dim=1, largest=False)
        mins = torch.where(vals[:, 0] < 1e-4, vals[:, 1], vals[:, 0])
    else:
        mins = dists.min(dim=1).values
    return float(torch.quantile(mins, 0.99))


# ------------------------------------------------------------- banks ------

def _safe_class_filename(class_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", class_name) + ".pt"


_banks_lock = threading.Lock()
_banks_cache: dict = {}      # class_name -> {"bank", "threshold", "mtime"}


def _load_bank(class_name: str):
    """Bank für eine Klasse laden (mit mtime-Cache); None wenn nicht trainiert."""
    import torch
    path = os.path.join(ANOMALY_DIR, _safe_class_filename(class_name))
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    with _banks_lock:
        cached = _banks_cache.get(class_name)
        if cached and cached["mtime"] == mtime:
            return cached
        data = torch.load(path, map_location="cpu")
        entry = {"bank": data["bank"].float(),
                 "threshold": float(data["threshold"]),
                 "mtime": mtime}
        _banks_cache[class_name] = entry
        return entry


def list_banks() -> dict:
    """Metadaten aller trainierten Banks (für die Status-Anzeige)."""
    import torch
    result = {}
    if not os.path.isdir(ANOMALY_DIR):
        return result
    for name in sorted(os.listdir(ANOMALY_DIR)):
        if not name.endswith(".pt"):
            continue
        try:
            data = torch.load(os.path.join(ANOMALY_DIR, name),
                              map_location="cpu")
            result[data.get("class", name[:-3])] = {
                "crops": data.get("crops", 0),
                "trained": data.get("trained"),
                "threshold": round(float(data["threshold"]), 3),
            }
        except Exception:
            continue
    return result


def anomaly_check(class_name: str, crop_rgb: np.ndarray):
    """Crop gegen die Gutteil-Bank seiner Klasse prüfen.
    Returns None (keine Bank/zu klein) oder
    {"score": float, "verdacht": bool} — score ist auf den Schwellwert
    normiert (1.0 = Schwelle, 1.3 = 30 % darüber)."""
    if crop_rgb is None or crop_rgb.size == 0:
        return None
    if min(crop_rgb.shape[0], crop_rgb.shape[1]) < MIN_CROP_PX:
        return None
    entry = _load_bank(class_name.split()[0] if class_name else class_name)
    if entry is None:
        return None
    score = _crop_score(_patch_features(crop_rgb), entry["bank"])
    ratio = score / entry["threshold"] if entry["threshold"] > 0 else 0.0
    return {"score": round(ratio, 2), "verdacht": ratio > 1.0}


# -------------------------------------------------------- training job ----

def _list_good_images() -> list[str]:
    images = []
    for root in GOOD_IMAGE_DIRS:
        if not os.path.isdir(root):
            continue
        for path in sorted(Path(root).rglob("*")):
            if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
                images.append(str(path))
    return images


def _job_train_banks() -> None:
    """Gutbilder mit dem Komponentenmodell croppen, pro Klasse Features
    sammeln und Memory Banks + Schwellwerte schreiben."""
    import cv2
    import torch
    try:
        images = _list_good_images()
        if len(images) < 5:
            _set_progress(
                state="error",
                message="Zu wenige Gutbilder gefunden — bitte Bilder "
                        "unbeschädigter Teile in den Ordner "
                        "training_captures/komponenten/ legen (mindestens 5, "
                        "besser 50+).")
            return

        # Exakt dieselbe Crop-Quelle wie die Erkennungs-Pipeline
        # (Ensemble aus Komponenten- + Synthetik-Modell, conf 0.4), sonst
        # passen die Schwellwerte nicht zu den Crops der Inferenz.
        _set_progress(message="Lade Komponentenmodelle …", total=len(images))
        from PIL import Image
        from ultralytics import YOLO
        from .combineYOLOModels import ensemble_predictions
        from .config import MODEL2
        detector1 = YOLO(get_active_model_path("komponenten"))
        detector2 = YOLO(MODEL2)

        per_class_feats: dict[str, list] = {}
        per_class_crops: dict[str, int] = {}
        rng = np.random.default_rng(42)

        for done, image_path in enumerate(images, start=1):
            _set_progress(done=done,
                          message=f"Analysiere Gutbilder … ({done}/{len(images)})")
            try:
                bgr = cv2.imread(image_path)
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb)
                results1 = detector1.predict(pil_image, conf=CROP_CONF,
                                             verbose=False)
                results2 = detector2.predict(pil_image, conf=CROP_CONF,
                                             verbose=False)
                boxes, _, classes = ensemble_predictions(results1, results2)
                for box, class_id in zip(boxes, classes):
                    x1, y1, x2, y2 = map(int, box)
                    if min(x2 - x1, y2 - y1) < MIN_CROP_PX:
                        continue
                    crop = rgb[max(y1, 0):y2, max(x1, 0):x2]
                    class_name = results1[0].names[int(class_id)]
                    feats = _patch_features(crop)
                    idx = rng.choice(feats.shape[0],
                                     size=min(PATCHES_PER_CROP, feats.shape[0]),
                                     replace=False)
                    per_class_feats.setdefault(class_name, []).append(
                        feats[torch.from_numpy(idx)])
                    per_class_crops[class_name] = per_class_crops.get(class_name, 0) + 1
            except Exception as exc:
                print(f"Anomalie-Training: {image_path} übersprungen ({exc})")

        trained, skipped = [], []
        os.makedirs(ANOMALY_DIR, exist_ok=True)
        for class_name, feat_list in sorted(per_class_feats.items()):
            n_crops = per_class_crops[class_name]
            if n_crops < MIN_CROPS_PER_CLASS:
                skipped.append(f"{class_name} ({n_crops})")
                continue
            _set_progress(message=f"Baue Gutteil-Bank für „{class_name}“ …")
            all_feats = torch.cat(feat_list, dim=0)
            if all_feats.shape[0] > MAX_BANK_PATCHES:
                idx = rng.choice(all_feats.shape[0], size=MAX_BANK_PATCHES,
                                 replace=False)
                bank = all_feats[torch.from_numpy(idx)]
            else:
                bank = all_feats
            # Schwellwert aus den Trainings-Crops selbst: Mittel + 3σ, aber
            # mindestens 10 % über dem schlechtesten Gutteil — kein einziges
            # Trainingsbild darf einen Fehlalarm auslösen (Vertrauen!).
            scores = [_crop_score(feats, bank, ignore_self=True)
                      for feats in feat_list]
            threshold = float(max(np.mean(scores) + 3 * np.std(scores),
                                  max(scores) * 1.1))
            torch.save({"class": class_name, "bank": bank.half(),
                        "threshold": threshold, "crops": n_crops,
                        "trained": datetime.now().isoformat(timespec="seconds")},
                       os.path.join(ANOMALY_DIR,
                                    _safe_class_filename(class_name)))
            trained.append(f"{class_name} ({n_crops} Gutteile)")

        with _banks_lock:
            _banks_cache.clear()

        if not trained:
            _set_progress(
                state="error",
                message="Keine Klasse hat genug Gutteil-Crops (mindestens "
                        f"{MIN_CROPS_PER_CLASS} nötig). Mehr Bilder in "
                        "training_captures/komponenten/ legen.")
            return
        message = f"Fertig: Verdachts-Ampel aktiv für {len(trained)} Klassen."
        if skipped:
            message += (f" Übersprungen (zu wenige Beispiele): "
                        f"{', '.join(skipped[:8])}"
                        + (" …" if len(skipped) > 8 else "") + ".")
        _set_progress(state="done", message=message)
    except Exception as exc:
        _set_progress(state="error", message=f"Fehler: {exc}")


# ---------------------------------------------------- damage detection ----

def inspect_crop(class_name: str, crop_rgb: np.ndarray) -> dict:
    """Beide Prüfungen für einen Bauteil-Crop.
    Returns {"zustand": str|None, "zustand_conf": float|None,
             "verdacht": bool|None, "anomalie_score": float|None} —
    None-Werte bedeuten: Prüfung nicht möglich/nichts gefunden."""
    out = {"zustand": None, "zustand_conf": None,
           "verdacht": None, "anomalie_score": None}
    if crop_rgb is None or crop_rgb.size == 0:
        return out

    # 1) Schadensmodell (falls eines hochgeladen wurde)
    from . import config
    if config.model_schaeden is not None:
        try:
            from PIL import Image
            # Als PIL übergeben — NumPy-Arrays würde ultralytics als BGR lesen.
            result = config.model_schaeden.predict(
                Image.fromarray(crop_rgb), conf=0.5, verbose=False)[0]
            allowed = allowed_damages(class_name)
            best_conf = 0.0
            for box in result.boxes:
                damage = result.names[int(box.cls[0])]
                conf = float(box.conf[0])
                if damage in allowed and conf > best_conf:
                    out["zustand"], out["zustand_conf"] = damage, round(conf, 2)
                    best_conf = conf
        except Exception as exc:
            print(f"Schadensmodell-Fehler für {class_name}: {exc}")

    # 2) Verdachts-Ampel (falls eine Gutteil-Bank existiert)
    try:
        anomaly = anomaly_check(class_name, crop_rgb)
        if anomaly is not None:
            out["verdacht"] = anomaly["verdacht"]
            out["anomalie_score"] = anomaly["score"]
    except Exception as exc:
        print(f"Anomalie-Check-Fehler für {class_name}: {exc}")
    return out


# ---------------------------------------------------------- endpoints -----

def _forbidden():
    return JSONResponse({"error": "Nur für eingeloggte Admins."},
                        status_code=403)


@router.get("/status")
async def anomaly_status(request: Request):
    if not is_admin(request):
        return _forbidden()
    schaden_path = get_active_model_path("schaeden")
    return {
        "banks": list_banks(),
        "available_images": len(_list_good_images()),
        "source_dir": "training_captures/komponenten/",
        "damage_model": (get_active_model("schaeden")
                         if os.path.exists(schaden_path) else None),
        "progress": dict(_progress),
    }


@router.post("/train")
async def train_banks(request: Request):
    """Gutteil-Banks (neu) aufbauen — Hintergrund-Job mit Fortschritt."""
    if not is_admin(request):
        return _forbidden()
    if not _start_job(_job_train_banks):
        return JSONResponse({"error": "Es läuft bereits ein Vorgang."},
                            status_code=409)
    return {"status": "Gestartet."}


@router.get("/progress")
async def anomaly_progress(request: Request):
    if not is_admin(request):
        return _forbidden()
    return dict(_progress)
