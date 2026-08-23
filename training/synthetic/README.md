# Synthetische Schadensbilder (BlenderProc)

Weg 3 der Schadenserkennung (siehe `docs/retraining/schaeden.md`): Wenn es zu
wenige echte Fehlerbilder gibt, werden Trainingsbilder **gerendert** — aus
Grundkörpern (Rohre, Platten) oder echten CAD-Modellen, mit prozedural
erzeugten Kratzern, Dellen und MDF-Abplatzern. Die Labels entstehen dabei
automatisch und sind pixelgenau.

> **Status: Gerüst.** Das Skript erzeugt lauffähige Datensätze, aber die
> Schadens-Optik ist bewusst einfach gehalten (Decal-Flächen statt echter
> Geometrie-Verformung). Für gute Ergebnisse an der Realitätslücke arbeiten —
> siehe unten.

## Benutzung

Eigene Python-Umgebung (nicht die Webapp — BlenderProc bringt ein komplettes
Blender mit, ~1 GB):

```bash
pip install blenderproc

# 200 Bilder mit Grundkörper-Geometrie:
blenderproc run training/synthetic/render_defects.py -- \
    --out yolo_dataset_synthetisch --anzahl 200

# Mit echten CAD-Modellen (.obj / .ply / .stl):
blenderproc run training/synthetic/render_defects.py -- \
    --out yolo_dataset_synthetisch --anzahl 200 --cad pfad/zu/cad_ordner
```

Ergebnis ist ein YOLO-Datensatz (`images/`, `labels/`, `data.yaml`). Ordner
zippen und im Colab-Notebook mit `MODELL_ZWECK = 'schaeden'` trainieren —
**immer gemischt mit echten Bildern** (Replay-Mix im Notebook), sonst lernt
das Modell die Render-Optik statt der Schäden.

## Die Realitätslücke verkleinern (Reihenfolge nach Wirkung)

1. **Echte Texturen:** Fotos der echten Pulverbeschichtung/MDF-Oberflächen
   als PBR-Material einbinden (`bproc.material` + Image-Textur) statt der
   einfachen Farben in `random_metal_material` / `random_mdf_material`.
2. **Echte Hintergründe:** Fotos des Demonstrator-Tisches als Boden-Textur.
3. **Schadensgeometrie:** Dellen als echte Vertex-Verformung
   (Displacement/Sculpt) statt dunkler Flecken; Abplatzer als Boolean-Schnitt
   an der Plattenkante.
4. **CAD-Modelle:** echte System180-Teile via `--cad` laden (Export aus dem
   CAD-System als .obj/.stl).
5. **Mehr Variation:** Kamerahöhen, Brennweite, Lichtfarbe, leichte
   Bewegungsunschärfe.

Faustregel aus der Praxis: synthetisch **vortrainieren**, mit den (wenigen)
echten gelabelten Bildern **nachschärfen** — genau dieses Muster wurde im
Projekt schon mit `best_synthetic_v2.pt` für die Komponentenerkennung genutzt.
