#!/usr/bin/env python3
"""Export a Label Studio project as a ready-to-train YOLO dataset.

Downloads all annotated tasks from Label Studio, converts the annotations to
YOLO format and writes a dataset directory with a train/val split and a
data.yaml. The class order is taken from the *base model* (or an explicit
class file), never from the annotations — this keeps class indices stable
across retraining runs, which is essential when fine-tuning an existing model.

Usage:
    export LABEL_STUDIO_API_KEY=...
    python export_yolo_dataset.py \
        --ls-url http://localhost:8082 \
        --project 4 \
        --model ../webapp/model/system180custommodel_v1.pt \
        --out yolo_dataset \
        --zip

The resulting zip can be uploaded to Google Drive and trained with
notebooks/SUSPEKT_YOLO_Weitertraining_Colab.ipynb.
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import requests

from prelabel_predictions import resolve_image


def load_class_names(model_path: str | None, classes_file: str | None) -> list[str]:
    if classes_file:
        text = Path(classes_file).read_text(encoding="utf-8")
        return [line.strip() for line in text.splitlines() if line.strip()]
    if model_path:
        from ultralytics import YOLO
        names = YOLO(model_path).names
        return [names[i] for i in sorted(names)]
    sys.exit("Provide --model or --classes to fix the class order.")


def fetch_annotated_tasks(session: requests.Session, ls_url: str,
                          project_id: int) -> list[dict]:
    url = f"{ls_url.rstrip('/')}/api/projects/{project_id}/export"
    response = session.get(url, params={"exportType": "JSON"}, timeout=300)
    response.raise_for_status()
    return response.json()


def latest_annotation(task: dict) -> dict | None:
    annotations = [a for a in task.get("annotations", [])
                   if not a.get("was_cancelled")]
    if not annotations:
        return None
    return max(annotations, key=lambda a: a.get("updated_at") or "")


def annotation_to_yolo_lines(annotation: dict, class_index: dict[str, int],
                             unknown: set[str]) -> list[str]:
    lines = []
    for item in annotation.get("result", []):
        value = item.get("value", {})
        labels = value.get("rectanglelabels")
        if not labels:
            continue
        label = labels[0]
        if label not in class_index:
            unknown.add(label)
            continue
        # Label Studio stores percentages of the image size.
        x_center = (value["x"] + value["width"] / 2) / 100
        y_center = (value["y"] + value["height"] / 2) / 100
        width = value["width"] / 100
        height = value["height"] / 100
        lines.append(f"{class_index[label]} {x_center:.6f} {y_center:.6f} "
                     f"{width:.6f} {height:.6f}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ls-url", default=os.getenv("LABEL_STUDIO_URL",
                                                      "http://localhost:8082"))
    parser.add_argument("--api-key", default=os.getenv("LABEL_STUDIO_API_KEY"))
    parser.add_argument("--project", type=int, required=True)
    parser.add_argument("--model", help="YOLO .pt model defining the class order")
    parser.add_argument("--classes", help="Text file with one class name per line")
    parser.add_argument("--out", default="yolo_dataset", help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--media-root", default=None,
                        help="Local path of the local-files storage root")
    parser.add_argument("--zip", action="store_true",
                        help="Also create <out>.zip for the Colab upload")
    args = parser.parse_args()

    if not args.api_key:
        sys.exit("Missing API key: pass --api-key or set LABEL_STUDIO_API_KEY.")

    class_names = load_class_names(args.model, args.classes)
    class_index = {name: i for i, name in enumerate(class_names)}

    session = requests.Session()
    session.headers["Authorization"] = f"Token {args.api_key}"

    tasks = fetch_annotated_tasks(session, args.ls_url, args.project)
    print(f"Fetched {len(tasks)} annotated tasks from project {args.project}.")

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    # Deterministic split so repeated exports keep images in the same split.
    tasks = sorted(tasks, key=lambda t: t["id"])
    random.Random(args.seed).shuffle(tasks)
    train_count = int(len(tasks) * args.train_ratio)

    unknown: set[str] = set()
    counts = {"train": 0, "val": 0}
    for i, task in enumerate(tasks):
        annotation = latest_annotation(task)
        image_ref = task.get("data", {}).get("image")
        if not annotation or not image_ref:
            continue
        lines = annotation_to_yolo_lines(annotation, class_index, unknown)
        split = "train" if i < train_count else "val"
        try:
            local = resolve_image(session, args.ls_url, image_ref,
                                  args.media_root, str(out))
        except Exception as exc:
            print(f"task {task['id']}: image download failed ({exc})",
                  file=sys.stderr)
            continue
        stem = f"task{task['id']}_{Path(local).stem}"
        suffix = Path(local).suffix
        target = out / "images" / split / f"{stem}{suffix}"
        if Path(local).is_relative_to(out):
            # Temp download inside the output dir — safe to move.
            shutil.move(local, target)
        else:
            # File lives in the Label Studio media storage — never move it.
            shutil.copy2(local, target)
        (out / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts[split] += 1

    if unknown:
        shutil.rmtree(out)
        sys.exit(f"Aborted: annotations contain labels unknown to the model: "
                 f"{sorted(unknown)}.\nFix the labels in Label Studio or "
                 f"retrain with an extended class list on purpose.")

    names_yaml = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
    (out / "data.yaml").write_text(
        f"path: .\ntrain: images/train\nval: images/val\n\n"
        f"nc: {len(class_names)}\nnames:\n{names_yaml}\n", encoding="utf-8")

    print(f"Wrote {counts['train'] + counts['val']} images "
          f"({counts['train']} train / {counts['val']} val) to {out}/")

    if args.zip:
        archive = shutil.make_archive(str(out), "zip", root_dir=out)
        print(f"Created {archive} — upload this to Google Drive for Colab.")


if __name__ == "__main__":
    main()
