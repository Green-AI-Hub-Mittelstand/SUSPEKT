#!/usr/bin/env python3
"""Pre-label Label Studio tasks with predictions from an existing YOLO model.

For every task in the given Label Studio project that has no prediction yet,
this script runs the YOLO model on the task's image and uploads the detected
boxes as Label Studio *predictions* (pre-annotations). Annotators then only
correct the suggestions instead of labeling from scratch.

This is the batch alternative to running a Label Studio ML backend: no extra
service needed, just run it manually or via cron after new images were synced.

Usage:
    export LABEL_STUDIO_API_KEY=...   # Account & Settings -> Access Token
    python prelabel_predictions.py \
        --ls-url http://localhost:8082 \
        --project 4 \
        --model ../webapp/model/system180custommodel_v1.pt \
        --conf 0.4

If Label Studio serves the images from a local-files storage that is also
mounted on this machine, pass --media-root to read the files directly instead
of downloading them through the API.
"""

import argparse
import os
import sys
import tempfile
import urllib.parse
from pathlib import Path

import requests


def api(session: requests.Session, ls_url: str, path: str, **kwargs):
    response = session.get(f"{ls_url.rstrip('/')}{path}", timeout=60, **kwargs)
    response.raise_for_status()
    return response.json()


def iter_tasks(session: requests.Session, ls_url: str, project_id: int):
    page = 1
    while True:
        params = {"project": project_id, "page": page, "page_size": 100,
                  "fields": "all"}
        try:
            data = api(session, ls_url, "/api/tasks", params=params)
        except requests.HTTPError as exc:
            # Label Studio returns 404 for pages past the end.
            if exc.response is not None and exc.response.status_code == 404:
                return
            raise
        # Depending on the Label Studio version the endpoint returns either
        # {"tasks": [...]} or a plain list.
        tasks = data if isinstance(data, list) else data.get("tasks", [])
        if not tasks:
            return
        yield from tasks
        page += 1


def resolve_image(session: requests.Session, ls_url: str, image_ref: str,
                  media_root: str | None, tmp_dir: str) -> str:
    """Return a local file path for the task image, downloading if needed."""
    if image_ref.startswith(("http://", "https://")):
        url = image_ref
    elif image_ref.startswith("/data/local-files/") and media_root:
        query = urllib.parse.urlparse(image_ref).query
        rel = urllib.parse.parse_qs(query).get("d", [""])[0]
        local = Path(media_root) / rel
        if local.exists():
            return str(local)
        url = f"{ls_url.rstrip('/')}{image_ref}"
    else:
        url = f"{ls_url.rstrip('/')}{image_ref}"

    response = session.get(url, timeout=120)
    response.raise_for_status()
    suffix = Path(urllib.parse.urlparse(url).path).suffix or ".jpg"
    fd, path = tempfile.mkstemp(suffix=suffix, dir=tmp_dir)
    with os.fdopen(fd, "wb") as f:
        f.write(response.content)
    return path


def predict_boxes(model, image_path: str, conf: float) -> list[dict]:
    """Run YOLO and convert detections to Label Studio result items."""
    results = model.predict(image_path, conf=conf, verbose=False)
    result = results[0]
    img_h, img_w = result.orig_shape
    items = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_name = model.names[int(box.cls[0])]
        items.append({
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "original_width": img_w,
            "original_height": img_h,
            "score": float(box.conf[0]),
            "value": {
                # Label Studio expects percentages of the image size.
                "x": x1 / img_w * 100,
                "y": y1 / img_h * 100,
                "width": (x2 - x1) / img_w * 100,
                "height": (y2 - y1) / img_h * 100,
                "rectanglelabels": [class_name],
            },
        })
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ls-url", default=os.getenv("LABEL_STUDIO_URL",
                                                      "http://localhost:8082"))
    parser.add_argument("--api-key", default=os.getenv("LABEL_STUDIO_API_KEY"))
    parser.add_argument("--project", type=int, required=True,
                        help="Label Studio project id")
    parser.add_argument("--model", required=True, help="Path to YOLO .pt model")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Confidence threshold (default 0.4)")
    parser.add_argument("--media-root", default=None,
                        help="Local path of the local-files storage root")
    parser.add_argument("--overwrite", action="store_true",
                        help="Also predict for tasks that already have predictions")
    args = parser.parse_args()

    if not args.api_key:
        sys.exit("Missing API key: pass --api-key or set LABEL_STUDIO_API_KEY.")

    from ultralytics import YOLO
    model = YOLO(args.model)
    model_version = Path(args.model).stem

    session = requests.Session()
    session.headers["Authorization"] = f"Token {args.api_key}"

    done = skipped = failed = 0
    with tempfile.TemporaryDirectory() as tmp_dir:
        for task in iter_tasks(session, args.ls_url, args.project):
            if task.get("predictions") and not args.overwrite:
                skipped += 1
                continue
            image_ref = task.get("data", {}).get("image")
            if not image_ref:
                skipped += 1
                continue
            try:
                image_path = resolve_image(session, args.ls_url, image_ref,
                                           args.media_root, tmp_dir)
                items = predict_boxes(model, image_path, args.conf)
                scores = [item["score"] for item in items]
                payload = {
                    "task": task["id"],
                    "model_version": model_version,
                    "score": sum(scores) / len(scores) if scores else 0.0,
                    "result": items,
                }
                response = session.post(
                    f"{args.ls_url.rstrip('/')}/api/predictions",
                    json=payload, timeout=60)
                response.raise_for_status()
                done += 1
                print(f"task {task['id']}: {len(items)} boxes")
            except Exception as exc:
                failed += 1
                print(f"task {task['id']}: FAILED ({exc})", file=sys.stderr)

    print(f"\nPredictions created: {done}, skipped: {skipped}, failed: {failed}")


if __name__ == "__main__":
    main()
