"""Admin API for the Label Studio integration on the /training page.

Everything a non-technical admin needs is exposed as buttons in the UI:

- status check (is Label Studio up, is the token valid, do projects exist)
- one-click project setup (labeling config generated from the model classes,
  local-files source storage in a per-project subfolder, initial sync)
- "sync & prelabel": import new images and write the current model's
  detections as Label Studio predictions (pre-annotations)
- YOLO dataset export as a zip download for the Colab notebook

The Label Studio API token is stored in config/labelstudio.json (mounted as a
volume in docker compose), so it survives container restarts and never has to
be typed into a terminal.
"""

import colorsys
import json
import os
import shutil
import tempfile
import threading
import urllib.parse
from pathlib import Path
from xml.sax.saxutils import quoteattr

import requests
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

from .auth import is_admin
from .modelTraining import (LABEL_STUDIO_API_KEY, LABEL_STUDIO_API_URL,
                            LABEL_STUDIO_MEDIA_DIR, MODEL_NAME)

router = APIRouter(prefix="/training/labelstudio")

# Internal base URL (container-to-container); scheme+host of the API URL.
LS_INTERNAL_URL = urllib.parse.urlsplit(LABEL_STUDIO_API_URL)._replace(
    path="", query="", fragment="").geturl()

CONFIG_DIR = os.getenv("CONFIG_DIR", "config")
CONFIG_FILE = os.path.join(CONFIG_DIR, "labelstudio.json")

# Root of the local-files storage as seen by Label Studio *and* this app
# (both containers mount ./training_captures on this path).
FILES_ROOT = os.getenv("LABEL_STUDIO_FILES_ROOT", "/label-studio/files")

# The two labeling projects of the pilot. "model" is resolved relative to
# the model/ directory of the webapp.
PROJECT_KINDS = {
    "komponenten": {
        "title": "Komponenten (Draufsicht)",
        "model": MODEL_NAME,
        "subdir": "komponenten",
    },
    "nubs": {
        "title": "Nubs (Seitenkameras)",
        "model": os.getenv("NUBS_MODEL_NAME", "NubsUpDown.pt"),
        "subdir": "nubs",
    },
}

_progress_lock = threading.Lock()
_progress = {"state": "idle", "message": "", "done": 0, "total": 0}


# --------------------------------------------------------------- config ----

def _load_config() -> dict:
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_config(cfg: dict) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def _api_key() -> str | None:
    return _load_config().get("api_key") or LABEL_STUDIO_API_KEY or None


def _session() -> requests.Session:
    s = requests.Session()
    key = _api_key()
    if key:
        scheme = _load_config().get("auth_scheme", "Token")
        s.headers["Authorization"] = f"{scheme} {key}"
    return s


def _api(method: str, path: str, **kwargs):
    response = _session().request(
        method, f"{LS_INTERNAL_URL}{path}", timeout=60, **kwargs)
    response.raise_for_status()
    return response.json() if response.content else None


# ------------------------------------------------------------ model bits ---

def _model_path(kind: str) -> str:
    return os.path.join("model", PROJECT_KINDS[kind]["model"])


def _class_names(kind: str) -> list[str]:
    from ultralytics import YOLO
    names = YOLO(_model_path(kind)).names
    return [names[i] for i in sorted(names)]


def _label_config_xml(class_names: list[str]) -> str:
    lines = [
        "<View>",
        '  <Image name="image" value="$image" zoom="true" zoomControl="true"/>',
        '  <RectangleLabels name="label" toName="image">',
    ]
    for i, name in enumerate(class_names):
        hue = i / max(len(class_names), 1)
        r, g, b = (int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.65, 0.85))
        lines.append(f'    <Label value={quoteattr(name)} '
                     f'background="#{r:02x}{g:02x}{b:02x}"/>')
    lines += ["  </RectangleLabels>", "</View>"]
    return "\n".join(lines)


# -------------------------------------------------------------- progress ---

def _set_progress(**kwargs) -> None:
    with _progress_lock:
        _progress.update(kwargs)


def _start_job(target, *args) -> bool:
    with _progress_lock:
        if _progress["state"] == "running":
            return False
        _progress.update(state="running", message="Starte …", done=0, total=0)
    threading.Thread(target=target, args=args, daemon=True).start()
    return True


# ------------------------------------------------------------- LS helpers --

def _find_project(kind: str) -> dict | None:
    cfg = _load_config()
    project_id = cfg.get("projects", {}).get(kind)
    if not project_id:
        return None
    try:
        return _api("GET", f"/api/projects/{project_id}")
    except requests.RequestException:
        return None


def _iter_tasks(project_id: int):
    page = 1
    while True:
        try:
            data = _api("GET", "/api/tasks", params={
                "project": project_id, "page": page,
                "page_size": 100, "fields": "all"})
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return  # past the last page
            raise
        tasks = data if isinstance(data, list) else data.get("tasks", [])
        if not tasks:
            return
        yield from tasks
        page += 1


def _resolve_image(image_ref: str, tmp_dir: str) -> str:
    """Return a readable local path for a task image, downloading if needed."""
    if image_ref.startswith("/data/local-files/"):
        query = urllib.parse.urlparse(image_ref).query
        rel = urllib.parse.parse_qs(query).get("d", [""])[0]
        local = Path(FILES_ROOT) / rel
        if local.exists():
            return str(local)
    elif image_ref.startswith("/data/upload/"):
        rel = urllib.parse.unquote(image_ref)[len("/data/upload/"):]
        local = Path(LABEL_STUDIO_MEDIA_DIR) / rel
        if local.exists():
            return str(local)
    # Fall back to downloading through the API.
    url = image_ref if image_ref.startswith(("http://", "https://")) \
        else f"{LS_INTERNAL_URL}{image_ref}"
    response = _session().get(url, timeout=120)
    response.raise_for_status()
    suffix = Path(urllib.parse.urlparse(url).path).suffix or ".jpg"
    fd, path = tempfile.mkstemp(suffix=suffix, dir=tmp_dir)
    with os.fdopen(fd, "wb") as f:
        f.write(response.content)
    return path


# ------------------------------------------------------------------ jobs ---

def _job_sync_and_prelabel(kind: str) -> None:
    try:
        project = _find_project(kind)
        if not project:
            _set_progress(state="error",
                          message="Projekt nicht gefunden — erst anlegen.")
            return
        project_id = project["id"]

        # 1) Sync all local-files storages so new images become tasks.
        _set_progress(message="Synchronisiere neue Bilder …")
        storages = _api("GET", "/api/storages/localfiles",
                        params={"project": project_id})
        for storage in storages or []:
            _api("POST", f"/api/storages/localfiles/{storage['id']}/sync")

        # 2) Collect tasks that have no prediction yet.
        _set_progress(message="Suche Bilder ohne Vorschläge …")
        todo = [t for t in _iter_tasks(project_id)
                if not t.get("predictions") and t.get("data", {}).get("image")]
        if not todo:
            _set_progress(state="done",
                          message="Alles aktuell — keine neuen Bilder ohne "
                                  "Vorschläge.", done=0, total=0)
            return

        # 3) Run the model and write predictions.
        from ultralytics import YOLO
        model = YOLO(_model_path(kind))
        model_version = Path(_model_path(kind)).stem
        done = failed = 0
        _set_progress(total=len(todo),
                      message=f"Erzeuge Vorschläge für {len(todo)} Bilder …")
        with tempfile.TemporaryDirectory() as tmp_dir:
            for task in todo:
                try:
                    image = _resolve_image(task["data"]["image"], tmp_dir)
                    result = model.predict(image, conf=0.4, verbose=False)[0]
                    img_h, img_w = result.orig_shape
                    items = []
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        items.append({
                            "from_name": "label", "to_name": "image",
                            "type": "rectanglelabels",
                            "original_width": img_w, "original_height": img_h,
                            "score": float(box.conf[0]),
                            "value": {
                                "x": x1 / img_w * 100, "y": y1 / img_h * 100,
                                "width": (x2 - x1) / img_w * 100,
                                "height": (y2 - y1) / img_h * 100,
                                "rectanglelabels":
                                    [model.names[int(box.cls[0])]],
                            },
                        })
                    scores = [i["score"] for i in items]
                    _api("POST", "/api/predictions", json={
                        "task": task["id"], "model_version": model_version,
                        "score": sum(scores) / len(scores) if scores else 0.0,
                        "result": items,
                    })
                    done += 1
                except Exception:
                    failed += 1
                _set_progress(done=done + failed)
        message = f"Fertig: {done} Bilder vorgelabelt."
        if failed:
            message += f" {failed} fehlgeschlagen."
        _set_progress(state="done", message=message)
    except Exception as exc:
        _set_progress(state="error", message=f"Fehler: {exc}")


# ------------------------------------------------------------- endpoints ---

def _forbidden():
    return JSONResponse({"error": "Nur für eingeloggte Admins."},
                        status_code=403)


@router.get("/status")
async def status(request: Request):
    """Traffic-light status for the setup assistant on /training."""
    if not is_admin(request):
        return _forbidden()
    result = {
        "reachable": False, "token_ok": False,
        "projects": {}, "progress": dict(_progress),
    }
    try:
        requests.get(f"{LS_INTERNAL_URL}/", timeout=5)
        result["reachable"] = True
    except requests.RequestException:
        return result
    if _api_key():
        try:
            _api("GET", "/api/current-user/whoami")
            result["token_ok"] = True
        except requests.RequestException:
            pass
    if result["token_ok"]:
        for kind, spec in PROJECT_KINDS.items():
            project = _find_project(kind)
            entry = {"title": spec["title"], "exists": project is not None,
                     "model": spec["model"],
                     "model_available": os.path.exists(_model_path(kind))}
            if project:
                entry.update(
                    id=project["id"],
                    tasks=project.get("task_number", 0),
                    annotated=project.get("num_tasks_with_annotations", 0),
                )
            result["projects"][kind] = entry
    return result


@router.post("/token")
async def save_token(request: Request):
    """Validate and store the Label Studio access token."""
    if not is_admin(request):
        return _forbidden()
    body = await request.json()
    token = (body.get("token") or "").strip()
    if not token:
        return JSONResponse({"error": "Kein Token übergeben."}, status_code=400)
    # Legacy tokens use "Token …", personal access tokens use "Bearer …".
    for scheme in ("Token", "Bearer"):
        try:
            response = requests.get(
                f"{LS_INTERNAL_URL}/api/current-user/whoami",
                headers={"Authorization": f"{scheme} {token}"}, timeout=10)
            if response.status_code == 200:
                cfg = _load_config()
                cfg["api_key"] = token
                cfg["auth_scheme"] = scheme
                _save_config(cfg)
                return {"status": "Token gespeichert und geprüft."}
        except requests.RequestException:
            pass
    return JSONResponse(
        {"error": "Token ungültig — bitte in Label Studio unter "
                  "Account & Settings neu kopieren."}, status_code=400)


@router.post("/setup/{kind}")
async def setup_project(kind: str, request: Request):
    """Create project + labeling config + source storage in one click."""
    if not is_admin(request):
        return _forbidden()
    if kind not in PROJECT_KINDS:
        return JSONResponse({"error": "Unbekanntes Projekt."}, status_code=404)
    spec = PROJECT_KINDS[kind]
    if _find_project(kind):
        return {"status": "Projekt existiert bereits."}
    if not os.path.exists(_model_path(kind)):
        return JSONResponse(
            {"error": f"Modell {spec['model']} fehlt in webapp/model/."},
            status_code=400)
    try:
        xml = _label_config_xml(_class_names(kind))
        project = _api("POST", "/api/projects", json={
            "title": spec["title"], "label_config": xml})
        # Storage root itself is rejected by Label Studio; use a subfolder.
        subdir = Path(FILES_ROOT) / spec["subdir"]
        subdir.mkdir(parents=True, exist_ok=True)
        storage = _api("POST", "/api/storages/localfiles", json={
            "project": project["id"], "title": "Trainingsbilder",
            "path": str(subdir), "use_blob_urls": True,
            "regex_filter": r".*\.(jpe?g|png|bmp|webp)$",
        })
        _api("POST", f"/api/storages/localfiles/{storage['id']}/sync")
        cfg = _load_config()
        cfg.setdefault("projects", {})[kind] = project["id"]
        _save_config(cfg)
        return {"status": f"Projekt „{spec['title']}“ angelegt "
                          f"(Nr. {project['id']})."}
    except requests.RequestException as exc:
        return JSONResponse({"error": f"Label-Studio-API-Fehler: {exc}"},
                            status_code=502)


@router.post("/prelabel/{kind}")
async def sync_and_prelabel(kind: str, request: Request):
    """Sync new images and write model predictions (background job)."""
    if not is_admin(request):
        return _forbidden()
    if kind not in PROJECT_KINDS:
        return JSONResponse({"error": "Unbekanntes Projekt."}, status_code=404)
    if not _start_job(_job_sync_and_prelabel, kind):
        return JSONResponse({"error": "Es läuft bereits ein Vorgang."},
                            status_code=409)
    return {"status": "Gestartet."}


@router.get("/progress")
async def progress(request: Request):
    if not is_admin(request):
        return _forbidden()
    return dict(_progress)


@router.get("/export/{kind}")
async def export_dataset(kind: str, request: Request):
    """Download the annotated tasks as a ready-to-train YOLO dataset zip."""
    if not is_admin(request):
        return _forbidden()
    if kind not in PROJECT_KINDS:
        return JSONResponse({"error": "Unbekanntes Projekt."}, status_code=404)
    project = _find_project(kind)
    if not project:
        return JSONResponse({"error": "Projekt nicht gefunden."},
                            status_code=404)

    class_names = _class_names(kind)
    class_index = {name: i for i, name in enumerate(class_names)}
    tasks = _api("GET", f"/api/projects/{project['id']}/export",
                 params={"exportType": "JSON"})

    out = Path(tempfile.mkdtemp()) / "yolo_dataset"
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    import random
    tasks = sorted(tasks, key=lambda t: t["id"])
    random.Random(42).shuffle(tasks)
    train_count = int(len(tasks) * 0.8)

    unknown: set[str] = set()
    exported = 0
    for i, task in enumerate(tasks):
        annotations = [a for a in task.get("annotations", [])
                       if not a.get("was_cancelled")]
        image_ref = task.get("data", {}).get("image")
        if not annotations or not image_ref:
            continue
        annotation = max(annotations, key=lambda a: a.get("updated_at") or "")
        lines = []
        for item in annotation.get("result", []):
            value = item.get("value", {})
            labels = value.get("rectanglelabels")
            if not labels:
                continue
            if labels[0] not in class_index:
                unknown.add(labels[0])
                continue
            x_c = (value["x"] + value["width"] / 2) / 100
            y_c = (value["y"] + value["height"] / 2) / 100
            lines.append(f"{class_index[labels[0]]} {x_c:.6f} {y_c:.6f} "
                         f"{value['width'] / 100:.6f} "
                         f"{value['height'] / 100:.6f}")
        split = "train" if i < train_count else "val"
        try:
            local = _resolve_image(image_ref, str(out))
        except Exception:
            continue
        stem = f"task{task['id']}_{Path(local).stem}"
        target = out / "images" / split / f"{stem}{Path(local).suffix}"
        if Path(local).is_relative_to(out):
            shutil.move(local, target)
        else:
            shutil.copy2(local, target)
        (out / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        exported += 1

    if unknown:
        shutil.rmtree(out.parent, ignore_errors=True)
        return JSONResponse(
            {"error": "Annotationen enthalten Klassen, die das Modell nicht "
                      f"kennt: {sorted(unknown)}. Bitte in Label Studio "
                      "korrigieren."}, status_code=409)
    if not exported:
        shutil.rmtree(out.parent, ignore_errors=True)
        return JSONResponse(
            {"error": "Noch keine fertig gelabelten Bilder zum Exportieren."},
            status_code=409)

    names_yaml = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    (out / "data.yaml").write_text(
        f"path: .\ntrain: images/train\nval: images/val\n\n"
        f"nc: {len(class_names)}\nnames:\n{names_yaml}\n", encoding="utf-8")
    archive = shutil.make_archive(str(out), "zip", root_dir=out)
    return FileResponse(archive, filename=f"yolo_dataset_{kind}.zip",
                        media_type="application/zip")
