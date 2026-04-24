import random
import cv2

def get_color_for_class(class_id):
    """Generiert eine zufällige, aber konsistente Farbe für eine gegebene Klasse."""
    class_id = int(class_id)  # Konvertiere class_id in int, um Fehler zu vermeiden
    random.seed(class_id)  # Gleiche Klasse → gleiche Farbe
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def draw_bounding_boxes(image_np, boxes, scores, classes, class_names, df_boxes):
    """
    Zeichnet Bounding Boxen mit Klassennamen, Konfidenz und Länge für 'Gerade'.
    """
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        conf = scores[i]
        class_id = classes[i]
        class_name = class_names[class_id] if class_id in class_names else "Unbekannt"

        # Falls die Klasse "Gerade" ist, Länge hinzufügen
        if class_name == "Gerade":
            laenge = df_boxes.iloc[i]["laenge"] if "laenge" in df_boxes.columns else "?"
            label = f"{class_name} {int(laenge)} ({conf:.2f})"
        else:
            label = f"{class_name} ({conf:.2f})"

        # Generiere eine Farbe basierend auf der Klasse
        color = get_color_for_class(class_id)

        # Bounding Box zeichnen
        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)

        # Textgröße berechnen
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x, text_y = x1, max(y1 - 10, 10)

        # Hintergrund für den Text zeichnen
        cv2.rectangle(image_np, (text_x, text_y - text_size[1] - 5),
                      (text_x + text_size[0], text_y + 5), color, -1)

        # Text einfügen (schwarz, damit er lesbar ist)
        cv2.putText(image_np, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 2, cv2.LINE_AA)

    return image_np


def update_model_names_with_lengths(result, boxes_with_lengths):
    """
    Update the model's class names dictionary to include length information
    This affects how the built-in YOLO visualizer will display the labels
    """
    # Copy the original names dictionary
    original_names = result.names.copy()
    updated_names = original_names.copy()

    # Update straight line class names with their lengths
    for item in boxes_with_lengths:
        bbox_id = item["bbox_id"]
        length_mm = item["length_mm"]

        # Find the class ID for this bbox_id
        for i, box in enumerate(result.boxes):
            if i == bbox_id:
                class_id = int(box.cls[0])
                original_name = original_names[class_id]

                if original_name.startswith("Gerade"):
                    # Update the name to include length
                    updated_names[class_id] = f"Gerade {length_mm}"
                    break

    # Set the updated names
    result.names = updated_names
    return result
