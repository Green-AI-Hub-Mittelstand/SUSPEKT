"""Model version registry + upload API for the /training page.

Non-technical admins drop the ``.pt``/``.onnx`` files produced by the Colab
retraining notebook onto the training page. Every upload is stored as a new
file in ``model/`` (existing versions are never overwritten or deleted) and
recorded in ``config/models.json``. Activating a ``.pt`` switches the live
detection/prelabeling model without a restart; older versions stay listed and
can be re-activated at any time (rollback).

The registry file lives in CONFIG_DIR next to labelstudio.json, so in docker
compose it is persisted via the mounted ./config volume.
"""

import json
import os
import re
import shutil
import threading
from datetime import datetime

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .auth import is_admin

router = APIRouter(prefix="/training/models")

MODEL_DIR = os.getenv("MODEL_DIR", "model")
CONFIG_DIR = os.getenv("CONFIG_DIR", "config")
REGISTRY_FILE = os.path.join(CONFIG_DIR, "models.json")

# Fallback when nothing has been activated on the /training page yet: the
# models the webapp shipped with (same defaults as docker compose).
DEFAULT_MODELS = {
    "komponenten": os.getenv("MODEL_NAME") or "system180custommodel_v1.pt",
    "nubs": os.getenv("NUBS_MODEL_NAME") or "NubsUpDown.pt",
    # There is no shipped damage model yet — the first one is trained via the
    # "Schäden" labeling project (or the augmentation/synthetic scripts in
    # training/) and uploaded here. Until then the damage check is skipped.
    "schaeden": os.getenv("SCHADEN_MODEL_NAME") or "schaeden.pt",
}
KIND_TITLES = {
    "komponenten": "Komponenten (Draufsicht)",
    "nubs": "Nubs (Seitenkameras)",
    "schaeden": "Schäden (Kratzer, Dellen, MDF-Platzer)",
}

ALLOWED_EXTENSIONS = {".pt", ".onnx"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024

_registry_lock = threading.Lock()


# --------------------------------------------------------------- registry --

def _load_registry() -> dict:
    try:
        with open(REGISTRY_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_registry(registry: dict) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def get_active_model(kind: str) -> str:
    """File name (inside model/) of the active .pt model for this kind."""
    name = _load_registry().get("active", {}).get(kind)
    if name and os.path.exists(os.path.join(MODEL_DIR, name)):
        return name
    return DEFAULT_MODELS[kind]


def get_active_model_path(kind: str) -> str:
    return os.path.join(MODEL_DIR, get_active_model(kind))


def _set_active_model(kind: str, filename: str) -> None:
    with _registry_lock:
        registry = _load_registry()
        registry.setdefault("active", {})[kind] = filename
        _save_registry(registry)


def _register_upload(entry: dict) -> None:
    with _registry_lock:
        registry = _load_registry()
        registry.setdefault("history", []).append(entry)
        _save_registry(registry)


# ---------------------------------------------------------------- helpers --

def _safe_filename(name: str) -> str:
    name = os.path.basename(name.replace("\\", "/"))
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).lstrip(".")


def _unique_filename(name: str) -> str:
    base, ext = os.path.splitext(name)
    candidate, counter = name, 1
    while os.path.exists(os.path.join(MODEL_DIR, candidate)):
        candidate = f"{base}_{counter}{ext}"
        counter += 1
    return candidate


def _detect_kind(filename: str) -> str:
    lowered = filename.lower()
    if "nub" in lowered:
        return "nubs"
    if "schaden" in lowered or "schaeden" in lowered or "schäden" in lowered:
        return "schaeden"
    return "komponenten"


def _validate_pt(path: str) -> dict:
    """Load the .pt with ultralytics to make sure it is a usable YOLO model.
    Returns {"classes": [...]} or raises ValueError with a German message."""
    try:
        from ultralytics import YOLO
        names = YOLO(path).names
    except Exception as exc:
        raise ValueError(
            "Die Datei konnte nicht als YOLO-Modell geladen werden — ist es "
            f"wirklich die .pt-Datei aus dem Colab-Notebook? ({exc})") from exc
    return {"classes": [names[i] for i in sorted(names)]}


def _reload_live_models() -> None:
    """Swap the models used by the running app (upload page, video, prelabel)."""
    from . import config, videoDetection
    config.reload_models()
    videoDetection.reload_model()


def _class_warning(kind: str, new_classes: list[str], old_active: str) -> str | None:
    """Warn (without blocking) when the class list of the new model differs
    from the previously active one — labeling projects then need attention."""
    old_path = os.path.join(MODEL_DIR, old_active)
    if not os.path.exists(old_path):
        return None
    try:
        old_classes = _validate_pt(old_path)["classes"]
    except ValueError:
        return None
    if old_classes == new_classes:
        return None
    return (f"Hinweis: Das neue Modell hat {len(new_classes)} Klassen, das "
            f"bisherige {len(old_classes)}. Falls das keine Absicht ist "
            "(z. B. falsche Datei), einfach unten die alte Version wieder "
            "aktivieren.")


def adopt_trained_model(kind: str, source_path: str, name_hint: str) -> str:
    """Store a freshly trained model as a new version and activate it (used by
    the legacy local training). Never overwrites; returns the new file name."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    filename = _unique_filename(_safe_filename(name_hint))
    target = os.path.join(MODEL_DIR, filename)
    shutil.copy2(source_path, target)
    info = _validate_pt(target)
    _register_upload({"file": filename, "kind": kind, "type": "pt",
                      "uploaded": datetime.now().isoformat(timespec="seconds"),
                      "size": os.path.getsize(target),
                      "classes": info["classes"]})
    _set_active_model(kind, filename)
    _reload_live_models()
    return filename


def list_models() -> list[dict]:
    """All known model versions, newest first, incl. the shipped defaults."""
    registry = _load_registry()
    active = {kind: get_active_model(kind) for kind in DEFAULT_MODELS}
    rows = []
    seen = set()
    for entry in reversed(registry.get("history", [])):
        path = os.path.join(MODEL_DIR, entry["file"])
        if entry["file"] in seen or not os.path.exists(path):
            continue
        seen.add(entry["file"])
        rows.append({
            "file": entry["file"],
            "kind": entry.get("kind", "komponenten"),
            "type": entry.get("type") or os.path.splitext(entry["file"])[1].lstrip("."),
            "uploaded": entry.get("uploaded"),
            "size_mb": round(os.path.getsize(path) / 1024 / 1024, 1),
            "classes": len(entry.get("classes") or []) or None,
            "active": entry["file"] == active.get(entry.get("kind")),
        })
    # The shipped default / currently active models may predate the registry.
    for kind, filename in active.items():
        path = os.path.join(MODEL_DIR, filename)
        if filename in seen or not os.path.exists(path):
            continue
        seen.add(filename)
        rows.append({
            "file": filename, "kind": kind, "type": "pt",
            "uploaded": None, "original": True,
            "size_mb": round(os.path.getsize(path) / 1024 / 1024, 1),
            "classes": None, "active": True,
        })
    return rows


# -------------------------------------------------------------- endpoints --

def _forbidden():
    return JSONResponse({"error": "Nur für eingeloggte Admins."},
                        status_code=403)


@router.get("")
async def get_models(request: Request):
    if not is_admin(request):
        return _forbidden()
    return {"models": list_models(),
            "active": {kind: get_active_model(kind) for kind in DEFAULT_MODELS},
            "kinds": KIND_TITLES}


@router.post("/upload")
async def upload_model(request: Request,
                       file: UploadFile = File(...),
                       kind: str = Form("auto"),
                       activate: bool = Form(True)):
    """Store an uploaded .pt/.onnx as a NEW version (never overwrites) and,
    for .pt with activate=true, switch the live model to it."""
    if not is_admin(request):
        return _forbidden()

    original_name = _safe_filename(file.filename or "")
    ext = os.path.splitext(original_name)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            {"error": f"„{file.filename}“ wird nicht unterstützt — bitte nur "
                      "die .pt- und .onnx-Dateien aus dem Colab-Notebook "
                      "hochladen."}, status_code=400)
    if kind not in KIND_TITLES:
        kind = _detect_kind(original_name)

    os.makedirs(MODEL_DIR, exist_ok=True)
    filename = _unique_filename(original_name)
    target = os.path.join(MODEL_DIR, filename)

    size = 0
    try:
        with open(target, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise ValueError(
                        "Datei ist größer als 500 MB — das ist keine "
                        "normale Modelldatei.")
                out.write(chunk)
        if size == 0:
            raise ValueError("Die Datei ist leer.")

        entry = {"file": filename, "kind": kind, "type": ext.lstrip("."),
                 "uploaded": datetime.now().isoformat(timespec="seconds"),
                 "size": size}
        warning = None
        activated = False

        if ext == ".pt":
            previous_active = get_active_model(kind)
            entry["classes"] = _validate_pt(target)["classes"]
            warning = _class_warning(kind, entry["classes"], previous_active)
            _register_upload(entry)
            if activate:
                _set_active_model(kind, filename)
                try:
                    _reload_live_models()
                except Exception as exc:
                    _set_active_model(kind, previous_active)
                    _reload_live_models()
                    return JSONResponse(
                        {"error": "Das neue Modell konnte nicht übernommen "
                                  f"werden ({exc}). Die bisherige Version "
                                  "bleibt aktiv."}, status_code=500)
                activated = True
                status = (f"Neue Version „{filename}“ ist jetzt das aktive "
                          f"Modell für {KIND_TITLES[kind]}. Die alte Version "
                          f"„{previous_active}“ bleibt erhalten.")
            else:
                status = f"„{filename}“ gespeichert (nicht aktiviert)."
        else:
            _register_upload(entry)
            status = (f"„{filename}“ gespeichert. Die .onnx-Datei ist für den "
                      "Jetson-Demonstrator — dort herunterladen und TensorRT "
                      "neu bauen.")

        return {"status": status, "file": filename, "kind": kind,
                "type": ext.lstrip("."), "activated": activated,
                "warning": warning}

    except ValueError as exc:
        if os.path.exists(target):
            os.remove(target)
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        if os.path.exists(target):
            os.remove(target)
        return JSONResponse({"error": f"Upload fehlgeschlagen: {exc}"},
                            status_code=500)


@router.post("/activate")
async def activate_model(request: Request):
    """Switch the active .pt model (also used for rollback to old versions)."""
    if not is_admin(request):
        return _forbidden()
    body = await request.json()
    filename = _safe_filename(body.get("file") or "")
    kind = body.get("kind")
    if kind not in KIND_TITLES:
        return JSONResponse({"error": "Unbekannter Modell-Typ."},
                            status_code=400)
    path = os.path.join(MODEL_DIR, filename)
    if not filename.endswith(".pt") or not os.path.exists(path):
        return JSONResponse({"error": f"Modelldatei „{filename}“ nicht "
                                      "gefunden."}, status_code=404)
    try:
        _validate_pt(path)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    previous_active = get_active_model(kind)
    _set_active_model(kind, filename)
    try:
        _reload_live_models()
    except Exception as exc:
        _set_active_model(kind, previous_active)
        _reload_live_models()
        return JSONResponse(
            {"error": f"Aktivierung fehlgeschlagen ({exc}). Die bisherige "
                      "Version bleibt aktiv."}, status_code=500)
    return {"status": f"„{filename}“ ist jetzt das aktive Modell für "
                      f"{KIND_TITLES[kind]}."}


@router.get("/download/{filename}")
async def download_model(filename: str, request: Request):
    """Download a stored model file (e.g. the .onnx for the Jetson)."""
    if not is_admin(request):
        return _forbidden()
    filename = _safe_filename(filename)
    path = os.path.join(MODEL_DIR, filename)
    if os.path.splitext(filename)[1].lower() not in ALLOWED_EXTENSIONS \
            or not os.path.exists(path):
        return JSONResponse({"error": "Datei nicht gefunden."},
                            status_code=404)
    return FileResponse(path, filename=filename,
                        media_type="application/octet-stream")
