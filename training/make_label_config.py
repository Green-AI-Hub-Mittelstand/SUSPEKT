#!/usr/bin/env python3
"""Generate a Label Studio labeling config from the classes of a YOLO model.

Using the model's own class list guarantees that the labels in Label Studio
match the model classes exactly (names and spelling), which the pre-labeling
and export scripts rely on.

Usage:
    python make_label_config.py --model ../webapp/model/system180custommodel_v1.pt
    python make_label_config.py --model model.pt --out label_config.xml
"""

import argparse
import colorsys
from xml.sax.saxutils import quoteattr


def class_names_from_model(model_path: str) -> list[str]:
    from ultralytics import YOLO

    names = YOLO(model_path).names
    return [names[i] for i in sorted(names)]


def build_config(class_names: list[str]) -> str:
    lines = [
        "<View>",
        '  <Image name="image" value="$image" zoom="true" zoomControl="true"/>',
        '  <RectangleLabels name="label" toName="image">',
    ]
    for i, name in enumerate(class_names):
        # Evenly spaced hues so every class gets a distinct color.
        hue = i / max(len(class_names), 1)
        r, g, b = (int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.65, 0.85))
        lines.append(
            f'    <Label value={quoteattr(name)} '
            f'background="#{r:02x}{g:02x}{b:02x}"/>'
        )
    lines += ["  </RectangleLabels>", "</View>"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to the YOLO .pt model")
    parser.add_argument("--out", help="Write XML to this file (default: stdout)")
    args = parser.parse_args()

    xml = build_config(class_names_from_model(args.model))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(xml + "\n")
        print(f"Label config written to {args.out}")
    else:
        print(xml)


if __name__ == "__main__":
    main()
