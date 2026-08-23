"""Cut-&-Paste-Augmentation: Aus wenigen echten Schadensfotos einen ganzen
YOLO-Trainingsdatensatz für das Schadensmodell erzeugen.

Idee: Die wenigen vorhandenen Schadensstellen werden als kleine Bildausschnitte
("Patches") auf viele Bilder unbeschädigter Teile kopiert — zufällig skaliert,
gedreht, gespiegelt und farblich angepasst, mit weich auslaufendem Rand. Aus
z. B. 10 echten Kratzern und 100 Gutbildern entstehen so hunderte gelabelte
Trainingsbilder.

Vorbereitung (einmalig, ~30 Minuten Handarbeit):
    1. Gutbilder sammeln: der Ordner training_captures/komponenten/ eignet
       sich direkt (dieselben Bilder wie fürs Labeln).
    2. Schadens-Patches ausschneiden: aus den echten Schadensfotos je Schaden
       einen engen Ausschnitt speichern (jedes Bildbearbeitungsprogramm oder
       Windows-Foto-Zuschnitt reicht) und in Unterordner nach Klasse legen:

           schadens_patches/
           ├── MDF-Platzer/   platzer1.jpg, platzer2.jpg, …
           ├── Rohr_Kratzer/  kratzer1.jpg, …
           └── Delle/         delle1.jpg, …

Aufruf (lokal oder in Colab, braucht nur Pillow + NumPy):

    python training/augment_defects.py \
        --gut training_captures/komponenten \
        --patches schadens_patches \
        --out yolo_dataset_schaeden \
        --anzahl 500

    Danach den Ordner zippen und im Colab-Notebook mit
    MODELL_ZWECK = 'schaeden' trainieren:
        (cd yolo_dataset_schaeden && zip -r ../yolo_dataset_schaeden.zip .)

Grenzen: Die Schäden landen an zufälligen Positionen, nicht nur auf
Bauteilen — für einen ersten Start ist das in Ordnung, weil das Modell die
Schadens-OPTIK lernt. Mit echten gelabelten Bildern aus dem Projekt "Schäden"
wird es danach Schritt für Schritt besser (Replay-Mix im Notebook nutzen).
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*")
                  if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())


def load_patches(patch_root: Path) -> dict[str, list[Image.Image]]:
    """Patches je Klassen-Unterordner laden."""
    patches: dict[str, list[Image.Image]] = {}
    for class_dir in sorted(p for p in patch_root.iterdir() if p.is_dir()):
        images = [Image.open(p).convert("RGB") for p in list_images(class_dir)]
        if images:
            patches[class_dir.name] = images
    return patches


def feathered_alpha(size: tuple[int, int], rng: random.Random) -> Image.Image:
    """Elliptische Alphamaske mit weichem Rand, damit die eingefügten
    Schäden keine harten (leicht lernbaren) Schnittkanten haben."""
    w, h = size
    mask = Image.new("L", (w, h), 0)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    inset_w, inset_h = int(w * 0.08) + 1, int(h * 0.08) + 1
    draw.ellipse([inset_w, inset_h, w - inset_w, h - inset_h], fill=255)
    blur = max(2, int(min(w, h) * rng.uniform(0.06, 0.12)))
    return mask.filter(ImageFilter.GaussianBlur(blur))


def transform_patch(patch: Image.Image, target_px: int,
                    rng: random.Random) -> Image.Image:
    """Zufällige Größe, Drehung, Spiegelung und Farb-/Helligkeitsanpassung."""
    scale = target_px / max(patch.size)
    scale *= rng.uniform(0.6, 1.4)
    new_size = (max(8, int(patch.width * scale)),
                max(8, int(patch.height * scale)))
    out = patch.resize(new_size, Image.LANCZOS)
    if rng.random() < 0.5:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.8, 1.2))
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.85, 1.15))
    out = ImageEnhance.Color(out).enhance(rng.uniform(0.85, 1.15))
    angle = rng.uniform(-35, 35)
    out = out.rotate(angle, expand=True, resample=Image.BICUBIC)
    return out


def paste_defect(base: Image.Image, patch: Image.Image,
                 rng: random.Random) -> tuple[float, float, float, float]:
    """Patch einfügen; YOLO-Box (x_center, y_center, w, h, normiert) zurück."""
    # Zielgröße relativ zur Bildgröße: Schäden sind klein (4–14 %).
    target_px = max(16, int(min(base.size) * rng.uniform(0.04, 0.14)))
    patch = transform_patch(patch, target_px, rng)
    if patch.width >= base.width or patch.height >= base.height:
        patch = patch.resize((min(patch.width, base.width // 3),
                              min(patch.height, base.height // 3)),
                             Image.LANCZOS)
    x = rng.randint(0, base.width - patch.width)
    y = rng.randint(0, base.height - patch.height)
    base.paste(patch, (x, y), feathered_alpha(patch.size, rng))
    return ((x + patch.width / 2) / base.width,
            (y + patch.height / 2) / base.height,
            patch.width / base.width,
            patch.height / base.height)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Erzeugt einen synthetischen YOLO-Schadensdatensatz "
                    "per Cut-&-Paste.")
    parser.add_argument("--gut", required=True, type=Path,
                        help="Ordner mit Bildern unbeschädigter Teile")
    parser.add_argument("--patches", required=True, type=Path,
                        help="Ordner mit Schadens-Patches (Unterordner je Klasse)")
    parser.add_argument("--out", required=True, type=Path,
                        help="Ausgabeordner (YOLO-Datensatz)")
    parser.add_argument("--anzahl", type=int, default=500,
                        help="Anzahl zu erzeugender Bilder (Default: 500)")
    parser.add_argument("--val-anteil", type=float, default=0.2,
                        help="Anteil Validierungs-Split (Default: 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    good_images = list_images(args.gut)
    patches = load_patches(args.patches)
    if not good_images:
        raise SystemExit(f"Keine Gutbilder in {args.gut} gefunden.")
    if not patches:
        raise SystemExit(
            f"Keine Patches in {args.patches} gefunden — Unterordner je "
            "Schadensklasse anlegen (z. B. MDF-Platzer/, Rohr_Kratzer/, Delle/).")

    class_names = sorted(patches)
    class_index = {name: i for i, name in enumerate(class_names)}
    print(f"{len(good_images)} Gutbilder, Patches: "
          + ", ".join(f"{k} ({len(v)})" for k, v in patches.items()))

    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_val = int(args.anzahl * args.val_anteil)
    for i in range(args.anzahl):
        split = "val" if i < n_val else "train"
        base = Image.open(rng.choice(good_images)).convert("RGB")
        # Bilder > 1600 px verkleinern (Trainings-Zeit/Speicher).
        if max(base.size) > 1600:
            factor = 1600 / max(base.size)
            base = base.resize((int(base.width * factor),
                                int(base.height * factor)), Image.LANCZOS)
        lines = []
        for _ in range(rng.randint(1, 3)):
            class_name = rng.choice(class_names)
            patch = rng.choice(patches[class_name])
            xc, yc, w, h = paste_defect(base, patch, rng)
            lines.append(f"{class_index[class_name]} "
                         f"{xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        stem = f"aug_{i:05d}"
        base.save(args.out / "images" / split / f"{stem}.jpg", quality=92)
        (args.out / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{args.anzahl} Bilder erzeugt …")

    names_yaml = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    (args.out / "data.yaml").write_text(
        f"path: .\ntrain: images/train\nval: images/val\n\n"
        f"nc: {len(class_names)}\nnames:\n{names_yaml}\n", encoding="utf-8")
    print(f"\nFertig: {args.anzahl} Bilder in {args.out} "
          f"({args.anzahl - n_val} train / {n_val} val).")
    print("Ordner zippen und im Colab-Notebook mit MODELL_ZWECK='schaeden' "
          "trainieren.")


if __name__ == "__main__":
    main()
