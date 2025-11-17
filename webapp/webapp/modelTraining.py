import json
import os
import random
import shutil
import urllib.parse

import requests
from dotenv import load_dotenv
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO

router = APIRouter(prefix="/training")
templates = Jinja2Templates(directory="templates")

load_dotenv()
LABEL_STUDIO_API_URL = os.getenv("LABEL_STUDIO_API_URL").format(id=os.getenv("PROJECT_ID"))
LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY")

YOLO_DATASET_DIR = "yolo_dataset"
IMAGES_DIR = os.path.join(YOLO_DATASET_DIR, "images")
LABELS_DIR = os.path.join(YOLO_DATASET_DIR, "labels")
DATA_YAML = os.path.join(YOLO_DATASET_DIR, "data.yaml")
MODEL_NAME = os.getenv("MODEL_NAME", "my_custom_model.pt")

MODEL_PATH = os.path.join("model", MODEL_NAME)
RUNS_DIR = "C:/runs/detect"
TRAIN_RATIO = 0.8
ANNOTATIONS_FILE = "annotations.json"
TRAINED_IMAGES_FILE = "trained_images.json"
user_home = os.path.expanduser("~")
LABEL_STUDIO_MEDIA_DIR = os.path.join(user_home, "AppData", "Local", "label-studio", "label-studio", "media", "upload")
training_in_progress = False


def remove_folder_safely(folder):
    if os.path.exists(folder):
        print(f"Removing old dataset: {folder}")
        try:
            shutil.rmtree(folder, ignore_errors=True)
        except Exception as e:
            print(f"Error removing {folder}: {e}")


def get_latest_training_folder(runs_dir):
    training_folders = [f for f in os.listdir(runs_dir) if f.startswith("train")]

    if not training_folders:
        raise Exception("No training directories found.")

    valid_folders = []
    for folder in training_folders:
        try:

            folder_number = int(folder.replace('train', ''))
            valid_folders.append(folder)
        except ValueError:
            print(f"Skipping invalid folder: {folder}")

    if not valid_folders:
        raise Exception("No valid training directories found.")

    latest_folder = sorted(valid_folders, key=lambda x: int(x.replace('train', '')))[-1]
    return os.path.join(runs_dir, latest_folder)


def start_training():
    """Trigger the full pipeline to fetch annotations and train the YOLO model."""
    global training_in_progress
    if training_in_progress:
        print("Training already in progress, skipping the new request.")
        return {"status": "Training already in progress."}

    # Set the flag to indicate training is in progress
    training_in_progress = True
    try:
        remove_folder_safely(YOLO_DATASET_DIR)

        os.makedirs(os.path.join(IMAGES_DIR, "train"), exist_ok=True)
        os.makedirs(os.path.join(IMAGES_DIR, "val"), exist_ok=True)
        os.makedirs(os.path.join(LABELS_DIR, "train"), exist_ok=True)
        os.makedirs(os.path.join(LABELS_DIR, "val"), exist_ok=True)

        # Download annotations from Label Studio*
        headers = {"Authorization": f"Token {LABEL_STUDIO_API_KEY}"}
        response = requests.get(LABEL_STUDIO_API_URL, headers=headers)

        if response.status_code == 200:
            with open(ANNOTATIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(response.json(), f, indent=4)
            print("Annotations downloaded.")
        else:
            return {"error": f"Failed to fetch annotations (Status: {response.status_code})"}

        # Process Annotations*
        with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        label_map = {}
        label_index = 0
        all_images = []
        trained_images = []

        # Try to load previously trained images if any
        if os.path.exists(TRAINED_IMAGES_FILE):
            with open(TRAINED_IMAGES_FILE, "r") as f:
                trained_images = json.load(f)
        else:
            print("No previous trained images found. Starting fresh.")

        # Copy images & Convert annotations
        for task in annotations:
            image_path = urllib.parse.unquote(task["data"]["image"]).replace("/data/upload/", "")
            image_filename = os.path.basename(image_path)
            full_image_path = os.path.join(LABEL_STUDIO_MEDIA_DIR, image_path)
            target_path = os.path.join(IMAGES_DIR, image_filename)

            # Only process new images
            if os.path.exists(full_image_path) and image_filename not in trained_images:
                shutil.copy(full_image_path, target_path)
                all_images.append(image_filename)
            else:
                print(f"Image not found or already trained: {full_image_path}")

            # Convert annotation to YOLO format
            yolo_label_path = os.path.join(LABELS_DIR, f"{os.path.splitext(image_filename)[0]}.txt")
            with open(yolo_label_path, "w") as yolo_label_file:
                for annotation in task["annotations"]:
                    for result in annotation["result"]:
                        value = result["value"]
                        if "rectanglelabels" in value:
                            label_name = value["rectanglelabels"][0]
                            if label_name not in label_map:
                                label_map[label_name] = label_index
                                label_index += 1
                            x_center = (value["x"] + value["width"] / 2) / 100
                            y_center = (value["y"] + value["height"] / 2) / 100
                            width = value["width"] / 100
                            height = value["height"] / 100
                            yolo_label_file.write(f"{label_map[label_name]} {x_center} {y_center} {width} {height}\n")

        if len(all_images) == 0:
            print("No new image detected, skipping training and update.")
            return {"status": "No new image detected, skipping training and update."}
        if len(all_images) == 1:
            print("Only one new image detected, skipping training and update.")
            return {"status": "Only one new image detected, skipping training and update."}

        # Append new images to the existing trained_images list
        trained_images.extend([img for img in all_images if img not in trained_images])

        # Save the updated list of trained images
        with open(TRAINED_IMAGES_FILE, "w") as f:
            json.dump(trained_images, f, indent=4)

        # Split Dataset into Train/Val*
        if all_images:
            random.shuffle(all_images)
            train_count = int(len(all_images) * TRAIN_RATIO)

            train_images = all_images[:train_count]
            val_images = all_images[train_count:]

            for img in train_images:
                shutil.move(os.path.join(IMAGES_DIR, img), os.path.join(IMAGES_DIR, "train", img))
                shutil.move(os.path.join(LABELS_DIR, f"{os.path.splitext(img)[0]}.txt"), os.path.join(LABELS_DIR, "train", f"{os.path.splitext(img)[0]}.txt"))

            for img in val_images:
                shutil.move(os.path.join(IMAGES_DIR, img), os.path.join(IMAGES_DIR, "val", img))
                shutil.move(os.path.join(LABELS_DIR, f"{os.path.splitext(img)[0]}.txt"), os.path.join(LABELS_DIR, "val", f"{os.path.splitext(img)[0]}.txt"))

            print(f"Dataset split: {len(train_images)} train, {len(val_images)} val.")
        else:
            print("No new images found for training.")

        with open(DATA_YAML, "w") as yaml_file:
            yaml_content = f"""
train: {os.path.abspath(os.path.join(IMAGES_DIR, 'train'))}
val: {os.path.abspath(os.path.join(IMAGES_DIR, 'val'))}

nc: {len(label_map)}
names: {list(label_map.keys())}
    """
            yaml_file.write(yaml_content.strip())

        model = YOLO(MODEL_PATH if os.path.exists(MODEL_PATH) else "yolov8n.pt")
        model.train(data=DATA_YAML, epochs=10, imgsz=640)

        latest_training_folder = get_latest_training_folder(RUNS_DIR)
        print(f"Using training folder: {latest_training_folder}")

        # Define paths for the best and last model files in the latest training folder
        best_model_path = os.path.join(latest_training_folder, "weights", "best.pt")
        last_model_path = os.path.join(latest_training_folder, "weights", "last.pt")

        # Check if the best model exists, otherwise use last model
        if os.path.exists(best_model_path):
            print(f"Best model found at {best_model_path}, copying to {MODEL_PATH}")
            shutil.copy(best_model_path, MODEL_PATH)
        elif os.path.exists(last_model_path):
            print(f"Best model not found, using last model at {last_model_path}, copying to {MODEL_PATH}")
            shutil.copy(last_model_path, MODEL_PATH)
        else:
            print("Neither best nor last model found in the training folder.")
            return {"error": "No model found for saving, training might have failed."}

        return {"status": "Training complete!", "model_saved_at": MODEL_PATH}

    except Exception as e:
        print(f"Error in training: {e}")
        raise Exception(f"Error in training: {e}")
    finally:
        # Reset the training flag once training is complete
        training_in_progress = False


@router.get("", response_class=HTMLResponse)
async def training_index(request: Request):
    """Render the image upload page"""
    return templates.TemplateResponse("training.html", {"request": request})


@router.post("/train")
async def train_model():
    """Trigger the training process"""
    status = start_training()
    return status
