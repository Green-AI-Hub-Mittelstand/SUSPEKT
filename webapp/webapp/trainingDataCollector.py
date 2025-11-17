import json
import os
import shutil
from datetime import datetime

BASE_DIR_UPLOADED_TRAININGSDATA = "./data/uploaded_original_images"


class TrainingDataCollector:
    def __init__(self, base_dir=BASE_DIR_UPLOADED_TRAININGSDATA):
        self.base_dir = base_dir
        self.images_dir = os.path.join(base_dir, "images")
        self.labels_dir = os.path.join(base_dir, "labels")
        self.crops_dir = os.path.join(base_dir, "crops")
        self._create_directories()

    def _create_directories(self):
        """Create necessary directory structure"""
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        os.makedirs(self.crops_dir, exist_ok=True)

    def _create_class_directory(self, class_name):
        """Create directory for specific class crops"""
        class_dir = os.path.join(self.crops_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        return class_dir

    def save_training_data(self, original_image_path, detected_objects,
                           original_filename):
        """Save image, crops, and annotations for training"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{timestamp}_{original_filename}"

        # Save original image
        image_path = os.path.join(self.images_dir, base_filename)
        shutil.copy2(original_image_path, image_path)

        annotations = []

        for obj in detected_objects:
            class_name = obj['name']
            bbox = obj['bbox']
            confidence = float(obj['confidence'])
            crop_filename = obj['crop_path']

            # Save crop in class-specific directory
            class_dir = self._create_class_directory(class_name)
            crop_dest = os.path.join(class_dir, f"{timestamp}_{crop_filename}")
            shutil.copy2(os.path.join("static/detected_images", crop_filename),
                         crop_dest)

            # Collect annotation data
            annotation = {
                'class': class_name,
                'bbox': bbox,
                'confidence': confidence,
                'crop_path': os.path.relpath(crop_dest, self.base_dir),
                'properties': obj.get('properties', {}),
                'color_info': obj.get('color_info', {})
            }
            annotations.append(annotation)

        # Save annotations
        annotation_path = os.path.join(self.labels_dir,
                                       f"{os.path.splitext(base_filename)[0]}.json")
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump({
                'image_path': os.path.relpath(image_path, self.base_dir),
                'original_filename': original_filename,
                'timestamp': timestamp,
                'objects': annotations
            }, f, indent=2, ensure_ascii=False)

        return {
            'image_path': image_path,
            'annotation_path': annotation_path,
            'num_objects': len(annotations)
        }
