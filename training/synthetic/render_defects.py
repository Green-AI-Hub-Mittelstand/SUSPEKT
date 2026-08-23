"""Synthetische Schadensbilder aus (CAD-)Geometrie rendern — BlenderProc-Gerüst.

Erzeugt Szenen mit System180-ähnlichen Bauteilen (Rohre = Zylinder,
Platten = Quader — oder echte CAD-Dateien via --cad) und fügt prozedural
Schäden hinzu:

    Rohr_Kratzer  helle, dünne Ritzlinie auf der Rohroberfläche
    Delle         dunkler, flacher Fleck (Schattierung einer Eindellung)
    MDF-Platzer   heller, unregelmäßiger Fleck an einer Plattenkante

Beleuchtung, Kamerawinkel und Materialien werden zufällig variiert
(Domain Randomization). Ausgabe ist ein fertiger YOLO-Datensatz
(images/ + labels/ + data.yaml), der direkt ins Colab-Notebook passt
(MODELL_ZWECK = 'schaeden').

Installation & Aufruf (eigene Python-Umgebung, NICHT in der Webapp):

    pip install blenderproc
    blenderproc run training/synthetic/render_defects.py -- \
        --out yolo_dataset_synthetisch --anzahl 200

Optional echte CAD-Modelle (.obj/.ply/.stl) statt der Grundkörper:

    blenderproc run training/synthetic/render_defects.py -- \
        --out ... --anzahl 200 --cad pfad/zu/cad_ordner

WICHTIG — Gerüst-Status: Dieses Skript ist ein getestbarer Startpunkt, kein
fertiges Produkt. Für fotorealistische Ergebnisse Materialien (PBR-Texturen
der echten Beschichtungen), Schadensgeometrie und Hintergründe verfeinern —
siehe README.md in diesem Ordner. Pro Szene wird maximal EIN Schaden je
Klasse erzeugt, damit die Boxen aus der Segmentierung eindeutig sind.
"""

import blenderproc as bproc  # muss der erste Import sein (BlenderProc-Regel)

import argparse
import random
from pathlib import Path

import numpy as np

CLASSES = ["MDF-Platzer", "Rohr_Kratzer", "Delle"]
# category_id 0 ist "kein Schaden"; Schäden beginnen bei 1.
CATEGORY = {name: i + 1 for i, name in enumerate(CLASSES)}


def random_metal_material(name: str):
    mat = bproc.material.create(name)
    grey = random.uniform(0.55, 0.85)
    mat.set_principled_shader_value("Base Color", [grey, grey, grey, 1.0])
    mat.set_principled_shader_value("Metallic", random.uniform(0.7, 1.0))
    mat.set_principled_shader_value("Roughness", random.uniform(0.2, 0.5))
    return mat


def random_mdf_material(name: str):
    mat = bproc.material.create(name)
    base = random.uniform(0.75, 0.95)
    mat.set_principled_shader_value("Base Color", [base, base, base * 0.98, 1.0])
    mat.set_principled_shader_value("Roughness", random.uniform(0.5, 0.9))
    return mat


def make_tube():
    """Rohr (Gerade) als Zylinder."""
    length = random.uniform(0.4, 1.2)
    tube = bproc.object.create_primitive(
        "CYLINDER", radius=0.015, depth=length)
    tube.set_rotation_euler([0, np.pi / 2, random.uniform(0, np.pi)])
    tube.set_location([random.uniform(-0.3, 0.3),
                       random.uniform(-0.3, 0.3), 0.02])
    tube.replace_materials(random_metal_material("metall"))
    tube.set_cp("category_id", 0)
    return tube


def make_plate():
    """Platte (Verkleidung/Systemboden) als flacher Quader."""
    plate = bproc.object.create_primitive("CUBE")
    plate.set_scale([random.uniform(0.15, 0.35),
                     random.uniform(0.15, 0.35), 0.008])
    plate.set_location([random.uniform(-0.3, 0.3),
                        random.uniform(-0.3, 0.3), 0.01])
    plate.set_rotation_euler([0, 0, random.uniform(0, np.pi)])
    plate.replace_materials(random_mdf_material("mdf"))
    plate.set_cp("category_id", 0)
    return plate


def add_scratch(tube):
    """Heller, dünner Ritz auf der Rohroberfläche."""
    scratch = bproc.object.create_primitive("CUBE")
    scratch.set_scale([random.uniform(0.01, 0.05), 0.0006, 0.0006])
    loc = np.array(tube.get_location())
    loc[2] += 0.0155  # knapp über der Rohroberfläche
    loc[0] += random.uniform(-0.1, 0.1)
    scratch.set_location(loc)
    scratch.set_rotation_euler([0, 0, random.uniform(0, np.pi)])
    mat = bproc.material.create("kratzer")
    mat.set_principled_shader_value("Base Color", [0.95, 0.95, 0.95, 1.0])
    mat.set_principled_shader_value("Metallic", 1.0)
    mat.set_principled_shader_value("Roughness", 0.15)
    scratch.replace_materials(mat)
    scratch.set_cp("category_id", CATEGORY["Rohr_Kratzer"])
    return scratch


def add_dent(tube):
    """Delle: dunkler, flacher Fleck (angenäherte Eindell-Schattierung)."""
    dent = bproc.object.create_primitive("SPHERE", radius=1.0)
    dent.set_scale([random.uniform(0.004, 0.012),
                    random.uniform(0.003, 0.008), 0.0008])
    loc = np.array(tube.get_location())
    loc[2] += 0.0152
    loc[0] += random.uniform(-0.1, 0.1)
    dent.set_location(loc)
    mat = bproc.material.create("delle")
    mat.set_principled_shader_value("Base Color", [0.25, 0.25, 0.27, 1.0])
    mat.set_principled_shader_value("Metallic", 0.9)
    mat.set_principled_shader_value("Roughness", 0.7)
    dent.replace_materials(mat)
    dent.set_cp("category_id", CATEGORY["Delle"])
    return dent


def add_chip(plate):
    """MDF-Platzer: heller, unregelmäßiger Fleck an einer Plattenkante."""
    chip = bproc.object.create_primitive("SPHERE", radius=1.0)
    chip.set_scale([random.uniform(0.006, 0.02),
                    random.uniform(0.004, 0.012), 0.0012])
    loc = np.array(plate.get_location())
    scale = plate.get_scale()
    edge = random.choice([-1, 1])
    loc[0] += edge * scale[0] * random.uniform(0.75, 0.98)
    loc[1] += random.uniform(-scale[1] * 0.8, scale[1] * 0.8)
    loc[2] += 0.012
    chip.set_location(loc)
    mat = bproc.material.create("platzer")
    mat.set_principled_shader_value("Base Color", [0.93, 0.85, 0.7, 1.0])
    mat.set_principled_shader_value("Roughness", 0.95)
    chip.replace_materials(mat)
    chip.set_cp("category_id", CATEGORY["MDF-Platzer"])
    return chip


def load_cad_parts(cad_dir: Path):
    """Echte CAD-Meshes (.obj/.ply/.stl) laden; Rückgabe pro Szene neu platziert."""
    files = [p for p in sorted(cad_dir.rglob("*"))
             if p.suffix.lower() in {".obj", ".ply", ".stl"}]
    return files


def yolo_lines_from_segmap(segmap: np.ndarray) -> list[str]:
    """Bounding Boxes je Schadensklasse aus der Kategorie-Segmentierung.
    (Ein Schaden je Klasse und Szene → Gesamt-Maske = korrekte Box.)"""
    h, w = segmap.shape
    lines = []
    for name, cat in CATEGORY.items():
        ys, xs = np.where(segmap == cat)
        if len(xs) < 6:  # Schaden verdeckt oder außerhalb des Bildes
            continue
        x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
        lines.append(f"{CLASSES.index(name)} "
                     f"{(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                     f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--anzahl", type=int, default=100)
    parser.add_argument("--val-anteil", type=float, default=0.2)
    parser.add_argument("--cad", type=Path, default=None,
                        help="Ordner mit CAD-Meshes statt Grundkörpern")
    parser.add_argument("--aufloesung", type=int, default=1280)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    bproc.init()
    bproc.camera.set_resolution(args.aufloesung, args.aufloesung)
    bproc.renderer.enable_segmentation_output(
        map_by=["category_id"], default_values={"category_id": 0})
    bproc.renderer.set_max_amount_of_samples(64)

    cad_files = load_cad_parts(args.cad) if args.cad else []

    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
    n_val = int(args.anzahl * args.val_anteil)

    for i in range(args.anzahl):
        bproc.utility.reset_keyframes()
        scene_objects = []

        # Boden/Hintergrund
        ground = bproc.object.create_primitive("PLANE")
        ground.set_scale([2, 2, 1])
        ground_mat = bproc.material.create("boden")
        g = random.uniform(0.3, 0.8)
        ground_mat.set_principled_shader_value("Base Color", [g, g, g, 1.0])
        ground.replace_materials(ground_mat)
        ground.set_cp("category_id", 0)
        scene_objects.append(ground)

        # Bauteile + je maximal ein Schaden pro Klasse
        if cad_files:
            part = bproc.loader.load_obj(str(random.choice(cad_files)))[0]
            part.set_location([0, 0, 0.02])
            part.set_rotation_euler([0, 0, random.uniform(0, np.pi)])
            part.set_cp("category_id", 0)
            scene_objects.append(part)
            tube, plate = part, part  # Schäden landen auf dem CAD-Teil
        else:
            tube = make_tube()
            plate = make_plate()
            scene_objects += [tube, plate]

        if random.random() < 0.8:
            scene_objects.append(add_scratch(tube))
        if random.random() < 0.6:
            scene_objects.append(add_dent(tube))
        if random.random() < 0.8:
            scene_objects.append(add_chip(plate))

        # Licht: Hauptlicht + flaches Streiflicht (macht Dellen sichtbar)
        lights = []
        main_light = bproc.types.Light()
        main_light.set_type("AREA")
        main_light.set_location([random.uniform(-1, 1),
                                 random.uniform(-1, 1),
                                 random.uniform(1.0, 2.0)])
        main_light.set_energy(random.uniform(30, 120))
        lights.append(main_light)
        raking = bproc.types.Light()
        raking.set_type("POINT")
        raking.set_location([random.uniform(-1.5, 1.5),
                             random.uniform(-1.5, 1.5),
                             random.uniform(0.05, 0.25)])
        raking.set_energy(random.uniform(10, 60))
        lights.append(raking)

        # Kamera: Draufsicht mit zufälliger Neigung (wie der Demonstrator)
        cam_location = np.array([random.uniform(-0.25, 0.25),
                                 random.uniform(-0.25, 0.25),
                                 random.uniform(0.7, 1.3)])
        look_at = np.array([random.uniform(-0.1, 0.1),
                            random.uniform(-0.1, 0.1), 0.0])
        rotation = bproc.camera.rotation_from_forward_vec(
            look_at - cam_location,
            inplane_rot=random.uniform(-0.3, 0.3))
        bproc.camera.add_camera_pose(
            bproc.math.build_transformation_mat(cam_location, rotation))

        data = bproc.renderer.render()
        rgb = data["colors"][0]
        segmap = data["category_id_segmaps"][0]

        split = "val" if i < n_val else "train"
        stem = f"syn_{i:05d}"
        from PIL import Image
        Image.fromarray(rgb).save(
            args.out / "images" / split / f"{stem}.jpg", quality=92)
        lines = yolo_lines_from_segmap(np.asarray(segmap))
        (args.out / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        # Szene für die nächste Iteration aufräumen
        for obj in scene_objects:
            obj.delete()
        for light in lights:
            light.delete()
        print(f"[{i + 1}/{args.anzahl}] {stem}: {len(lines)} Schäden")

    names_yaml = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    (args.out / "data.yaml").write_text(
        f"path: .\ntrain: images/train\nval: images/val\n\n"
        f"nc: {len(CLASSES)}\nnames:\n{names_yaml}\n", encoding="utf-8")
    print(f"\nFertig: Datensatz in {args.out}. Zippen und im Colab-Notebook "
          "mit MODELL_ZWECK='schaeden' trainieren (am besten gemischt mit "
          "echten Bildern per Replay-Mix).")


if __name__ == "__main__":
    main()
