# measurement.py
import numpy as np
import pandas as pd

# 📌 **Referenzwerte für Skalierung**
reference_values = {
    "Gerade": {"thickness_mm": 20, "known_lengths_mm": [180, 270, 360, 450, 540, 720, 900]},
    "Griff": {"width_mm": 100, "height_mm": 30},
    "Sockelfuss": {"width_mm": 30, "height_mm": 50},
    "Rolle": {"width_mm": 75, "height_mm": 120},
    "Noppenscheiben": {"width_mm": 30, "height_mm": 30}
}





def calculate_pixel_to_mm_ratio(df_boxes):
    """Berechnet den Skalierungsfaktor basierend auf bekannten Objektgrößen."""
    scaling_factors = {}

    for class_name in reference_values:
        if "width_mm" in reference_values[class_name] and "height_mm" in reference_values[class_name]:
            class_boxes = df_boxes[df_boxes["class"] == class_name]
            if not class_boxes.empty:
                width_pixels = class_boxes["x_max"] - class_boxes["x_min"]
                height_pixels = class_boxes["y_max"] - class_boxes["y_min"]

                if width_pixels.mean() > 0 and height_pixels.mean() > 0:
                    scale_width = reference_values[class_name]["width_mm"] / width_pixels.mean()
                    scale_height = reference_values[class_name]["height_mm"] / height_pixels.mean()
                    scaling_factors[class_name] = np.mean([scale_width, scale_height])
    print(f"scaling_factos: {scaling_factors}")
    return scaling_factors


def calculate_diagonale_lengths():
    pass

def calculate_straight_lengths(df_boxes, pixel_to_mm_ratio=1):
    """Berechnet die Längen von 'Gerade'-Objekten basierend auf Referenzwerten."""
    df_boxes["breite"] = None
    df_boxes["laenge"] = None

    for index, row in df_boxes.iterrows():
        if row["class"] != "Gerade":
            continue

        width_pixels = row["x_max"] - row["x_min"]
        height_pixels = row["y_max"] - row["y_min"]
        length_pixels = max(width_pixels, height_pixels)

        width_mm = reference_values["Gerade"]["thickness_mm"]
        length_mm = round(length_pixels * pixel_to_mm_ratio)

        # 🔥 Wähle die nächstgelegene bekannte Länge
        matched_length_mm = min(reference_values["Gerade"]["known_lengths_mm"], key=lambda x: abs(x - length_mm))

        df_boxes.at[index, "breite"] = width_mm
        df_boxes.at[index, "laenge"] = matched_length_mm

    return df_boxes

