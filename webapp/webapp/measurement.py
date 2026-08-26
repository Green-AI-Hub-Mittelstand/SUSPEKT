# measurement.py
import math

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Systemraster laut "System 180 Facts & Details"
#
# Die Systemmaße sind Knoten-zu-Knoten-Maße und damit gleichzeitig die
# Nennlängen der Streben, die genau ein Rasterfeld überspannen.
# ---------------------------------------------------------------------------
SYSTEM_WIDTHS = [360, 450, 540, 630, 720, 810, 900]     # Systembreiten
SYSTEM_HEIGHTS = [180, 270, 360, 450, 540, 720]         # Systemhöhen
SYSTEM_DEPTHS = [210, 340, 430, 600, 690, 780, 870]     # Systemtiefen

# Horizontale Streben spannen entweder eine Breite (Front/Rückseite) oder eine
# Tiefe (Seitenansicht) auf, vertikale Streben immer eine Höhe.
HORIZONTAL_LENGTHS = sorted(set(SYSTEM_WIDTHS + SYSTEM_DEPTHS))
VERTICAL_LENGTHS = sorted(set(SYSTEM_HEIGHTS))

# Eine Gerade spannt eine Systembreite oder eine Systemhöhe auf - die Tiefe
# trägt der Mutternstab. Ohne diese Trennung fällt eine Gerade von etwa 440 mm
# auf die Tiefe 430 statt auf die Breite 450: 430 kommt ausschließlich im
# Tiefenraster vor.
GERADE_LENGTHS = sorted(set(SYSTEM_WIDTHS + SYSTEM_HEIGHTS))
ALL_STRAIGHT_LENGTHS = GERADE_LENGTHS

# Edelstahl-Präzisionsrohr 20x1: jede Strebe ist 20 mm dick. Dadurch ist jede
# erkannte Strebe zugleich ein Referenzobjekt für den Maßstab.
TUBE_DIAMETER_MM = 20

STRAIGHT_CLASSES = ("Gerade", "Mutternstab")
DIAGONAL_CLASSES = ("Diagonale",)
STRUT_CLASSES = STRAIGHT_CLASSES + DIAGONAL_CLASSES


def _diagonal_candidates():
    """Mögliche Diagonalenlängen: eine Diagonale verspannt genau ein Rasterfeld,
    ihre Länge ist deshalb die Hypotenuse aus Feldbreite/-tiefe und Feldhöhe."""
    candidates = set()
    for horizontal in HORIZONTAL_LENGTHS:
        for vertical in VERTICAL_LENGTHS:
            candidates.add(int(round(math.hypot(horizontal, vertical))))
    return sorted(candidates)


DIAGONAL_LENGTHS = _diagonal_candidates()

# Referenzobjekte mit bekannten Abmessungen. Die Fußhöhen sind über
# "Gesamthöhe = Systemhöhe + 15 mm + Höhe des Fußes" gegengeprüft.
reference_values = {
    "Gerade": {"thickness_mm": TUBE_DIAMETER_MM, "known_lengths_mm": ALL_STRAIGHT_LENGTHS},
    "Mutternstab": {"thickness_mm": TUBE_DIAMETER_MM, "known_lengths_mm": SYSTEM_DEPTHS},
    "Diagonale": {"thickness_mm": TUBE_DIAMETER_MM, "known_lengths_mm": DIAGONAL_LENGTHS},
    "Griff": {"width_mm": 100, "height_mm": 30},
    "Sockelfuss": {"width_mm": 30, "height_mm": 50},
    "Rolle": {"width_mm": 75, "height_mm": 120},
    "Noppenscheiben": {"width_mm": 30, "height_mm": 30},
}

# Verlässlichkeit der Maßstabsquellen: kompakte, rotationsunabhängige Objekte
# wiegen am schwersten. Die Rolle ist mehrdeutig (50/75/100 mm mit 90/120/160 mm
# Gesamthöhe), der Griff kann hochkant verbaut sein - beide zählen weniger.
_SCALE_WEIGHTS = {
    "Noppenscheiben": 1.0,
    "Sockelfuss": 0.8,
    "Griff": 0.5,
    "Rolle": 0.4,
    "_tube": 0.6,
}

# Ab diesem Seitenverhältnis gilt eine Box als eindeutig gestreckt, sodass ihre
# kurze Kante dem Rohrdurchmesser entspricht.
_ELONGATED_RATIO = 3.0

_MEASURE_COLUMNS = ("breite", "laenge", "laenge_roh", "mass_quelle")

# Ansichten, in denen die jeweilige Achse in der Bildebene liegt: in der
# Seitenansicht die Systemtiefe, in der Frontansicht die Systembreite.
_SIDE_VIEWS = ("left", "right")
_FRONT_VIEWS = ("front", "back")


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
def _bbox(row):
    return (
        float(row["x_min"]), float(row["y_min"]),
        float(row["x_max"]), float(row["y_max"]),
    )


def _dims(row):
    x_min, y_min, x_max, y_max = _bbox(row)
    return abs(x_max - x_min), abs(y_max - y_min)


def _center(row):
    x_min, y_min, x_max, y_max = _bbox(row)
    return (x_min + x_max) / 2.0, (y_min + y_max) / 2.0


def _snap(value, candidates):
    """Rundet auf das nächstgelegene Systemmaß."""
    if value is None or not candidates:
        return None
    return min(candidates, key=lambda candidate: abs(candidate - value))


def _weighted_median(values, weights):
    """Gegenüber dem Mittelwert robust gegen einzelne Ausreißer-Boxen."""
    pairs = [(v, w) for v, w in zip(values, weights) if v and v > 0 and w > 0]
    if not pairs:
        return None
    pairs.sort(key=lambda pair: pair[0])
    half = sum(weight for _, weight in pairs) / 2.0
    running = 0.0
    for value, weight in pairs:
        running += weight
        if running >= half:
            return value
    return pairs[-1][0]


def _most_common(values):
    """Häufigster Wert, bei Gleichstand der mittlere."""
    if not values:
        return None
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    best = max(counts.values())
    winners = sorted(value for value, count in counts.items() if count == best)
    return winners[len(winners) // 2]


def _robust_max(values):
    """Größter Messwert, ohne auf einen einzelnen Ausreißer hereinzufallen."""
    if not values:
        return None
    return float(np.percentile(sorted(values), 85))


def _snap_foreshortened(value, candidates, tolerance=0.05):
    """Rundet einen möglicherweise perspektivisch verkürzten Messwert.

    Verkürzung wirkt nur in eine Richtung - gemessen wird nie zu lang. Liegt der
    Messwert nahe an einem Rastermaß, wird darauf gerundet; andernfalls wird
    Verkürzung angenommen und auf das nächstgrößere Rastermaß aufgerundet.
    """
    if value is None or not candidates:
        return None
    nearest = _snap(value, candidates)
    if nearest and abs(nearest - value) / max(value, 1.0) <= tolerance:
        return nearest
    larger = [candidate for candidate in candidates if candidate >= value]
    return min(larger) if larger else max(candidates)


def _ensure_measure_columns(df_boxes):
    if df_boxes is None:
        return pd.DataFrame()
    for column in _MEASURE_COLUMNS:
        if column not in df_boxes.columns:
            df_boxes[column] = None
    return df_boxes


def _has_boxes(df_boxes):
    if df_boxes is None or df_boxes.empty:
        return False
    required = {"class", "x_min", "y_min", "x_max", "y_max"}
    return required.issubset(set(df_boxes.columns))


# ---------------------------------------------------------------------------
# Maßstab
# ---------------------------------------------------------------------------
class ScaleContext:
    """Sammelt alle Maßstabsschätzungen (mm pro Pixel) eines Bildes.

    Perspektive führt dazu, dass der Maßstab über das Bild variiert: näher an
    der Kamera liegende Bauteile erscheinen größer. Deshalb gibt es neben dem
    globalen Wert einen ortsabhängigen Maßstab, der Referenzen in der Nähe des
    gesuchten Bauteils stärker gewichtet.
    """

    def __init__(self, samples=None, span=0.0):
        self.samples = samples or []
        self.span = span or 0.0
        self.global_scale = _weighted_median(
            [sample["mm_per_px"] for sample in self.samples],
            [sample["weight"] for sample in self.samples],
        )

    @property
    def is_valid(self):
        return bool(self.global_scale and self.global_scale > 0)

    def at(self, x, y):
        """Ortsabhängiger Maßstab am Punkt (x, y)."""
        if not self.is_valid:
            return None
        falloff = max(self.span * 0.25, 1.0)
        values, weights = [], []
        for sample in self.samples:
            distance = math.hypot(sample["x"] - x, sample["y"] - y)
            values.append(sample["mm_per_px"])
            weights.append(sample["weight"] / (1.0 + distance / falloff))
        return _weighted_median(values, weights) or self.global_scale


def _image_span(df_boxes):
    try:
        width = float(df_boxes["x_max"].max()) - float(df_boxes["x_min"].min())
        height = float(df_boxes["y_max"].max()) - float(df_boxes["y_min"].min())
    except (KeyError, TypeError, ValueError):
        return 0.0
    return math.hypot(max(width, 0.0), max(height, 0.0))


def estimate_scale_context(df_boxes):
    """Leitet den Pixel-zu-Millimeter-Maßstab aus allen verfügbaren Referenzen ab.

    Neben den klassischen Referenzobjekten (Noppenscheibe, Sockelfuß, Rolle,
    Griff) dient die kurze Kante jeder gestreckten Strebe als Referenz, weil
    jedes Systemrohr 20 mm dick ist. Dadurch steht praktisch immer ein Maßstab
    zur Verfügung, sobald überhaupt Streben erkannt wurden.
    """
    if not _has_boxes(df_boxes):
        return ScaleContext()

    samples = []
    for _, row in df_boxes.iterrows():
        width_px, height_px = _dims(row)
        if width_px <= 0 or height_px <= 0:
            continue

        center_x, center_y = _center(row)
        class_name = row.get("class")
        reference = reference_values.get(class_name, {})
        known_width = reference.get("width_mm")
        known_height = reference.get("height_mm")

        if known_width and known_height:
            weight = _SCALE_WEIGHTS.get(class_name, 0.4)
            samples.append({
                "x": center_x, "y": center_y,
                "mm_per_px": known_width / width_px, "weight": weight,
            })
            samples.append({
                "x": center_x, "y": center_y,
                "mm_per_px": known_height / height_px, "weight": weight,
            })
            continue

        if class_name in STRUT_CLASSES:
            long_px = max(width_px, height_px)
            short_px = min(width_px, height_px)
            # Nur bei eindeutig gestreckter Box entspricht die kurze Kante dem
            # Rohrdurchmesser - die Box einer Diagonale ist annähernd quadratisch.
            if short_px > 0 and long_px / short_px >= _ELONGATED_RATIO:
                samples.append({
                    "x": center_x, "y": center_y,
                    "mm_per_px": TUBE_DIAMETER_MM / short_px,
                    "weight": _SCALE_WEIGHTS["_tube"],
                })

    context = ScaleContext(samples, _image_span(df_boxes))
    print(f"Maßstab: {context.global_scale} mm/px aus {len(samples)} Referenzen")
    return context


def calculate_pixel_to_mm_ratio(df_boxes):
    """Maßstab je Referenzklasse (Rückwärtskompatibilität).

    Der Schlüssel "global" enthält zusätzlich den robust kombinierten Maßstab
    über alle Referenzen hinweg.
    """
    scaling_factors = {}
    if not _has_boxes(df_boxes):
        return scaling_factors

    for class_name, reference in reference_values.items():
        if not (reference.get("width_mm") and reference.get("height_mm")):
            continue
        class_boxes = df_boxes[df_boxes["class"] == class_name]
        if class_boxes.empty:
            continue

        width_pixels = (class_boxes["x_max"] - class_boxes["x_min"]).mean()
        height_pixels = (class_boxes["y_max"] - class_boxes["y_min"]).mean()
        if width_pixels > 0 and height_pixels > 0:
            scale_width = reference["width_mm"] / width_pixels
            scale_height = reference["height_mm"] / height_pixels
            scaling_factors[class_name] = float(np.mean([scale_width, scale_height]))

    context = estimate_scale_context(df_boxes)
    if context.is_valid:
        scaling_factors["global"] = context.global_scale
    return scaling_factors


def _coerce_scale(scale, pixel_to_mm_ratio, df_boxes):
    """Nimmt einen ScaleContext, einen festen Faktor oder gar nichts entgegen."""
    if isinstance(scale, ScaleContext):
        return scale
    if pixel_to_mm_ratio and pixel_to_mm_ratio > 0:
        return ScaleContext(
            [{"x": 0.0, "y": 0.0, "mm_per_px": float(pixel_to_mm_ratio), "weight": 1.0}],
            _image_span(df_boxes),
        )
    return estimate_scale_context(df_boxes)


# ---------------------------------------------------------------------------
# Gerade Streben
# ---------------------------------------------------------------------------
def _straight_candidates(class_name, is_vertical, view=None):
    if class_name == "Mutternstab":
        # Der Mutternstab läuft in der Seitenebene und spannt die Systemtiefe auf.
        return SYSTEM_DEPTHS
    if is_vertical:
        return VERTICAL_LENGTHS
    # Waagerechte Gerade: in der Front-/Rückansicht eine Systembreite. Ist die
    # Ansicht unbekannt, bleiben Breiten und Höhen möglich - Tiefen nicht, die
    # gehören zum Mutternstab.
    if view in _FRONT_VIEWS:
        return SYSTEM_WIDTHS
    return GERADE_LENGTHS


def calculate_straight_lengths(df_boxes, pixel_to_mm_ratio=None, scale=None):
    """Bestimmt Dicke und Länge aller geraden Streben.

    Gemessen wird die lange Bounding-Box-Kante mit dem lokalen Maßstab,
    anschließend wird orientierungsabhängig auf das nächstgelegene Systemmaß
    gerundet: vertikale Streben spannen eine Systemhöhe auf, horizontale eine
    Systembreite oder -tiefe.
    """
    df_boxes = _ensure_measure_columns(df_boxes)
    if not _has_boxes(df_boxes):
        return df_boxes

    scale = _coerce_scale(scale, pixel_to_mm_ratio, df_boxes)

    for index, row in df_boxes.iterrows():
        if row.get("class") not in STRAIGHT_CLASSES:
            continue

        width_px, height_px = _dims(row)
        length_px = max(width_px, height_px)
        if length_px <= 0:
            continue

        mm_per_px = scale.at(*_center(row))
        if not mm_per_px:
            continue

        view = str(row.get("ansicht") or "").lower()
        candidates = _straight_candidates(row.get("class"), height_px > width_px, view)
        matched = _snap(length_px * mm_per_px, candidates)

        df_boxes.at[index, "breite"] = TUBE_DIAMETER_MM
        df_boxes.at[index, "laenge"] = matched
        # Rohwert aufheben: der Abgleich über mehrere Ansichten braucht die
        # ungerundete Messung, um Verkürzungen zu erkennen.
        df_boxes.at[index, "laenge_roh"] = round(length_px * mm_per_px, 1)
        df_boxes.at[index, "mass_quelle"] = "gemessen"

    return df_boxes


# ---------------------------------------------------------------------------
# Diagonalen
# ---------------------------------------------------------------------------
def _find_module_edges(diagonal_row, df_boxes):
    """Sucht die geraden Streben, die dasselbe Rasterfeld begrenzen.

    Eine Diagonale läuft von Knoten zu Knoten quer durch ein Feld, ihre
    Bounding-Box deckt sich deshalb näherungsweise mit dem Feld selbst. Streben,
    die an dessen Ober-/Unterkante bzw. linker/rechter Kante liegen und es
    vollständig überspannen, liefern die beiden Feldkanten in Millimetern.
    """
    dx_min, dy_min, dx_max, dy_max = _bbox(diagonal_row)
    field_width = dx_max - dx_min
    field_height = dy_max - dy_min
    if field_width <= 0 or field_height <= 0:
        return None, None

    tolerance_x = max(0.20 * field_width, 12.0)
    tolerance_y = max(0.20 * field_height, 12.0)

    horizontal_edges, vertical_edges = [], []
    for index, row in df_boxes.iterrows():
        if index == diagonal_row.name:
            continue
        if row.get("class") not in STRAIGHT_CLASSES:
            continue

        length_mm = row.get("laenge")
        if not length_mm or pd.isna(length_mm):
            continue

        x_min, y_min, x_max, y_max = _bbox(row)
        width_px, height_px = abs(x_max - x_min), abs(y_max - y_min)
        center_x, center_y = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0

        if width_px >= height_px:
            overlap = min(x_max, dx_max) - max(x_min, dx_min)
            spans_field = overlap / field_width >= 0.6
            same_length = abs(width_px - field_width) <= 0.35 * field_width
            on_edge = min(abs(center_y - dy_min), abs(center_y - dy_max)) <= tolerance_y
            if overlap > 0 and spans_field and same_length and on_edge:
                horizontal_edges.append(int(length_mm))
        else:
            overlap = min(y_max, dy_max) - max(y_min, dy_min)
            spans_field = overlap / field_height >= 0.6
            same_length = abs(height_px - field_height) <= 0.35 * field_height
            on_edge = min(abs(center_x - dx_min), abs(center_x - dx_max)) <= tolerance_x
            if overlap > 0 and spans_field and same_length and on_edge:
                vertical_edges.append(int(length_mm))

    return _most_common(horizontal_edges), _most_common(vertical_edges)


def calculate_diagonale_lengths(df_boxes, scale=None):
    """Bestimmt die Länge von Diagonalen.

    Direktes Messen scheitert hier regelmäßig: Liegt eine Diagonale in einer
    Seitenebene, ist sie perspektivisch stark verkürzt, und ihre Bounding-Box
    zeigt nur die Projektion. Zuverlässiger ist die Geometrie - eine Diagonale
    verspannt genau ein Rasterfeld, ihre Länge ist also die Hypotenuse aus den
    beiden Feldkanten. Diese Kanten liefern die umliegenden geraden Streben,
    deren Längen bereits auf das Systemraster gerundet wurden.

    Reihenfolge der Verfahren:
    1. beide Feldkanten bekannt  -> Hypotenuse direkt aus dem Raster
    2. eine Feldkante bekannt    -> zweite Kante über das Seitenverhältnis der Box
    3. keine Kante bekannt       -> gemessen und auf gültige Hypotenusen gerundet
    """
    df_boxes = _ensure_measure_columns(df_boxes)
    if not _has_boxes(df_boxes):
        return df_boxes

    scale = _coerce_scale(scale, None, df_boxes)

    for index, row in df_boxes.iterrows():
        if row.get("class") not in DIAGONAL_CLASSES:
            continue

        width_px, height_px = _dims(row)
        if width_px <= 0 or height_px <= 0:
            continue

        horizontal_mm, vertical_mm = _find_module_edges(row, df_boxes)
        aspect_ratio = height_px / width_px

        if horizontal_mm and vertical_mm:
            length_mm = math.hypot(horizontal_mm, vertical_mm)
            source = "aus Nachbarstreben"
        elif horizontal_mm:
            # Zweite Kante über das Seitenverhältnis der Box schätzen und auf
            # das Höhenraster runden.
            vertical_mm = _snap(horizontal_mm * aspect_ratio, VERTICAL_LENGTHS)
            length_mm = math.hypot(horizontal_mm, vertical_mm)
            source = "aus Nachbarstrebe und Seitenverhältnis"
        elif vertical_mm:
            horizontal_mm = _snap(vertical_mm / aspect_ratio, HORIZONTAL_LENGTHS) if aspect_ratio else None
            if not horizontal_mm:
                continue
            length_mm = math.hypot(horizontal_mm, vertical_mm)
            source = "aus Nachbarstrebe und Seitenverhältnis"
        else:
            mm_per_px = scale.at(*_center(row))
            if not mm_per_px:
                continue
            # Die Diagonale läuft von Ecke zu Ecke ihrer Box, ihre Bildlänge ist
            # deshalb die Boxdiagonale und nicht die längere Boxkante.
            length_mm = math.hypot(width_px, height_px) * mm_per_px
            source = "gemessen"

        matched = _snap(length_mm, DIAGONAL_LENGTHS)
        if not matched:
            continue

        df_boxes.at[index, "breite"] = TUBE_DIAMETER_MM
        df_boxes.at[index, "laenge"] = matched
        df_boxes.at[index, "mass_quelle"] = source

    return df_boxes


# ---------------------------------------------------------------------------
# Abgleich über mehrere Ansichten
# ---------------------------------------------------------------------------
def reconcile_measurements(image_results):
    """Gleicht die Maße über alle Ansichten eines Möbels hinweg ab.

    Aus einer einzelnen Aufnahme ist die Tiefenachse nicht bestimmbar: eine
    perspektivisch verkürzte Strebe von 430 mm erzeugt dasselbe Bild wie eine
    kürzere, weniger stark verkürzte. Zwei Eigenschaften lösen das auf:

    * Verkürzung wirkt nur in eine Richtung - gemessen wird nie zu lang. Der
      größte Messwert einer Strebe liegt deshalb ihrer wahren Länge am nächsten.
    * Laut Systembeschreibung lassen sich die Tiefen innerhalb eines Möbels
      nicht kombinieren. Es gibt also genau eine Systemtiefe für alle Ansichten,
      und in der Seitenansicht liegt sie unverkürzt in der Bildebene.

    Breiten und Höhen bleiben unangetastet: sie dürfen innerhalb eines Möbels
    variieren und werden ohnehin in der Bildebene gemessen.
    """
    if not image_results:
        return image_results

    depth_observations = []
    for df_boxes in image_results.values():
        if not _has_boxes(df_boxes) or "laenge_roh" not in df_boxes.columns:
            continue
        for _, row in df_boxes.iterrows():
            raw_length = row.get("laenge_roh")
            if not raw_length or pd.isna(raw_length):
                continue
            width_px, height_px = _dims(row)
            view = str(row.get("ansicht") or "").lower()
            if row.get("class") == "Mutternstab":
                depth_observations.append(float(raw_length))
            elif (row.get("class") == "Gerade"
                  and width_px >= height_px
                  and view in _SIDE_VIEWS):
                depth_observations.append(float(raw_length))

    depth_mm = _snap_foreshortened(_robust_max(depth_observations), SYSTEM_DEPTHS)
    if depth_mm:
        print(f"Systemtiefe aus {len(depth_observations)} Messungen: {depth_mm} mm")
        for df_boxes in image_results.values():
            if not _has_boxes(df_boxes):
                continue
            for index, row in df_boxes.iterrows():
                if row.get("class") != "Mutternstab":
                    continue
                df_boxes.at[index, "breite"] = TUBE_DIAMETER_MM
                df_boxes.at[index, "laenge"] = depth_mm
                df_boxes.at[index, "mass_quelle"] = "Systemtiefe über alle Ansichten"

    # Systembreite abgleichen.
    #
    # Dieselbe Überlegung wie bei der Tiefe: gemessen wird nie zu lang. Liegt der
    # größte Messwert der waagerechten Geraden zwischen zwei Rastermaßen, ist das
    # größere richtig - eine 450er Strebe kann auf 400 mm schrumpfen, eine 360er
    # aber nicht auf 400 mm wachsen.
    breiten_messungen = []
    for df_boxes in image_results.values():
        if not _has_boxes(df_boxes) or "laenge_roh" not in df_boxes.columns:
            continue
        for _, row in df_boxes.iterrows():
            if str(row.get("class") or "").split(" ")[0] != "Gerade":
                continue
            roh = row.get("laenge_roh")
            if not roh or pd.isna(roh):
                continue
            width_px, height_px = _dims(row)
            view = str(row.get("ansicht") or "").lower()
            if width_px >= height_px and view in _FRONT_VIEWS:
                breiten_messungen.append(float(roh))

    breite_mm = _snap_foreshortened(_robust_max(breiten_messungen), SYSTEM_WIDTHS)
    if breite_mm:
        print(f"Systembreite aus {len(breiten_messungen)} Messungen: {breite_mm} mm")
        for df_boxes in image_results.values():
            if not _has_boxes(df_boxes):
                continue
            for index, row in df_boxes.iterrows():
                if str(row.get("class") or "").split(" ")[0] != "Gerade":
                    continue
                width_px, height_px = _dims(row)
                view = str(row.get("ansicht") or "").lower()
                if width_px >= height_px and view in _FRONT_VIEWS:
                    df_boxes.at[index, "laenge"] = breite_mm
                    df_boxes.at[index, "breite"] = TUBE_DIAMETER_MM
                    df_boxes.at[index, "mass_quelle"] = "Systembreite über alle Ansichten"

    # Rohroberfläche vereinheitlichen.
    #
    # Ein Möbel ist in einem Finish gebaut - SteelLine oder eine der
    # pulverbeschichteten Linien, nicht gemischt. Einzelne Streben im Schatten
    # werden aber gerne als "Schwarz Pulverbeschichtet" gelesen. Die Mehrheit
    # über alle Ansichten entscheidet deshalb für alle Profile.
    _vereinheitliche_profilfarbe(image_results)

    # Diagonalen und Flächen mit den abgeglichenen Feldkanten neu ableiten.
    for image_name, df_boxes in image_results.items():
        df_boxes = calculate_diagonale_lengths(df_boxes)
        image_results[image_name] = calculate_panel_dimensions(df_boxes, depth_mm=depth_mm)

    return image_results


# ---------------------------------------------------------------------------
# Flächenbauteile (Türen, Verkleidungen, Böden)
# ---------------------------------------------------------------------------
# Flächen in der Front-/Rückebene: Systembreite x Systemhöhe.
FRONT_PANEL_CLASSES = (
    "Einzeltuer", "Doppeltuer", "Doppeltuerblatt", "Kombifront", "Auszug",
    "Verkleidung", "Magazin",
)

# Flächen in der Seitenebene: Systemtiefe x Systemhöhe.
SIDE_PANEL_CLASSES = (
    "Seitenverkleidung", "Seitenverkleidung-0-0", "Seitenverkleidung-0-IN",
    "Seitenverkleidung-IN-IN",
    "Seitenverkleidung - 0/0", "Seitenverkleidung - 0/IN", "Seitenverkleidung - IN/IN",
)

# Flächen, die ein Rasterfeld senkrecht ausfüllen.
PANEL_CLASSES = FRONT_PANEL_CLASSES + SIDE_PANEL_CLASSES

# Waagerechte Flächen: Systembreite x Systemtiefe.
BOARD_CLASSES = ("Systemboden", "Fachboden", "Fachboden mit Verstaerkung")


def _flaechen_optionen():
    """Auswahllisten für die beiden Maße flächiger Bauteile.

    Anders als eine Strebe hat eine Fläche zwei Systemmaße. Welche Raster
    infrage kommen, hängt von der Ebene ab, in der die Fläche liegt.
    """
    optionen = {}
    for klasse in FRONT_PANEL_CLASSES:
        optionen[klasse] = {
            "breite": SYSTEM_WIDTHS, "breite_label": "Breite",
            "laenge": SYSTEM_HEIGHTS, "laenge_label": "Höhe",
        }
    for klasse in SIDE_PANEL_CLASSES:
        optionen[klasse] = {
            "breite": SYSTEM_DEPTHS, "breite_label": "Tiefe",
            "laenge": SYSTEM_HEIGHTS, "laenge_label": "Höhe",
        }
    for klasse in BOARD_CLASSES:
        optionen[klasse] = {
            "breite": SYSTEM_WIDTHS, "breite_label": "Breite",
            "laenge": SYSTEM_DEPTHS, "laenge_label": "Tiefe",
        }
    return optionen


FLAECHEN_OPTIONEN = _flaechen_optionen()


def calculate_panel_dimensions(df_boxes, scale=None, depth_mm=None):
    """Leitet die Maße flächiger Bauteile aus dem Rasterfeld ab.

    Eine Tür oder Verkleidung füllt genau ein Rasterfeld aus, ihre Bounding-Box
    deckt sich deshalb näherungsweise mit dem Feld. Die umliegenden geraden
    Streben liefern die beiden Feldkanten - dieselbe Überlegung wie bei den
    Diagonalen, nur dass hier beide Kanten das Ergebnis sind.

    Waagerechte Böden zeigen im Bild ihre Breite, aber nicht ihre Tiefe: die
    kommt aus der abgeglichenen Systemtiefe.
    """
    df_boxes = _ensure_measure_columns(df_boxes)
    if not _has_boxes(df_boxes):
        return df_boxes

    scale = _coerce_scale(scale, None, df_boxes)

    for index, row in df_boxes.iterrows():
        class_name = row.get("class")
        ist_panel = class_name in PANEL_CLASSES
        ist_board = class_name in BOARD_CLASSES
        if not (ist_panel or ist_board):
            continue

        width_px, height_px = _dims(row)
        if width_px <= 0 or height_px <= 0:
            continue

        horizontal_mm, vertical_mm = _find_module_edges(row, df_boxes)

        # Fehlende Kanten aus der Messung ergänzen.
        mm_per_px = scale.at(*_center(row))
        if not horizontal_mm and mm_per_px:
            horizontal_mm = _snap(width_px * mm_per_px, HORIZONTAL_LENGTHS)
        if not vertical_mm and mm_per_px:
            vertical_mm = _snap(height_px * mm_per_px, VERTICAL_LENGTHS)

        if ist_board:
            # Die Tiefe eines Bodens ist im Bild nur perspektivisch verkürzt zu
            # sehen. Wenn der Ansichtenabgleich eine Systemtiefe ergeben hat,
            # zählt die; sonst bleibt nur die verkürzte Messung.
            if depth_mm:
                tiefe = depth_mm
            elif vertical_mm:
                tiefe = _snap(vertical_mm, SYSTEM_DEPTHS)
            else:
                tiefe = None
            if not (horizontal_mm and tiefe):
                continue
            df_boxes.at[index, "breite"] = horizontal_mm
            df_boxes.at[index, "laenge"] = tiefe
            df_boxes.at[index, "mass_quelle"] = (
                "Breite aus Nachbarstreben, Tiefe aus Systemtiefe"
                if depth_mm else "aus Nachbarstreben"
            )
            continue

        if not (horizontal_mm and vertical_mm):
            continue
        df_boxes.at[index, "breite"] = horizontal_mm
        df_boxes.at[index, "laenge"] = vertical_mm
        df_boxes.at[index, "mass_quelle"] = "aus Rasterfeld"

    return df_boxes


# Profile aus Systemrohr - ihre Oberfläche ist innerhalb eines Möbels einheitlich.
PROFIL_KLASSEN = ("Gerade", "Diagonale", "Mutternstab", "Noppenscheiben",
                  "Schraube", "Sockelfuss", "Winkelfuss", "Griff")


def _vereinheitliche_profilfarbe(image_results):
    """Setzt für alle Profile die mehrheitlich erkannte Oberfläche.

    Einzelne Streben liegen im Schatten und werden dann als dunkle Beschichtung
    gelesen, obwohl das Möbel durchgehend aus gebürstetem Edelstahl besteht. Die
    Mehrheit über alle Ansichten ist verlässlicher als die Einzelmessung.
    """
    stimmen = {}
    for df in image_results.values():
        if df is None or df.empty or "class" not in df.columns:
            continue
        for _, zeile in df.iterrows():
            basis = str(zeile.get("class") or "").split(" ")[0]
            if basis not in PROFIL_KLASSEN:
                continue
            farbe = zeile.get("farbe")
            name = farbe.get("erkannte_farbe") if isinstance(farbe, dict) else farbe
            if not name:
                continue
            stimmen[name] = stimmen.get(name, 0) + 1

    if len(stimmen) < 2:
        return image_results

    mehrheit = max(stimmen, key=stimmen.get)

    # Den vollständigen Farbeintrag als Vorlage übernehmen (mit Hex und NCS).
    vorlage = None
    for df in image_results.values():
        if df is None or df.empty:
            continue
        for _, zeile in df.iterrows():
            farbe = zeile.get("farbe")
            if isinstance(farbe, dict) and farbe.get("erkannte_farbe") == mehrheit:
                vorlage = dict(farbe)
                break
        if vorlage:
            break
    if not vorlage:
        vorlage = {"erkannte_farbe": mehrheit}

    geaendert = 0
    for df in image_results.values():
        if df is None or df.empty or "class" not in df.columns:
            continue
        for index, zeile in df.iterrows():
            basis = str(zeile.get("class") or "").split(" ")[0]
            if basis not in PROFIL_KLASSEN:
                continue
            farbe = zeile.get("farbe")
            name = farbe.get("erkannte_farbe") if isinstance(farbe, dict) else farbe
            if name and name != mehrheit:
                df.at[index, "farbe"] = dict(vorlage)
                geaendert += 1

    if geaendert:
        print(f"Profiloberfläche vereinheitlicht auf {mehrheit} "
              f"({geaendert} Abweichungen korrigiert, Stimmen: {stimmen})")
    return image_results
