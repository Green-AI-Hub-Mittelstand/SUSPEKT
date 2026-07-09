# Trainings-Pipeline: Skripte

Werkzeuge für den Weitertrainings-Loop **Label Studio → YOLO-Datensatz → Colab**.
Gesamtkonzept: [`docs/retraining/konzept.md`](../docs/retraining/konzept.md) ·
Schulung: [`docs/retraining/schulung.md`](../docs/retraining/schulung.md) ·
Colab-Notebook: [`notebooks/SUSPEKT_YOLO_Weitertraining_Colab.ipynb`](../notebooks/SUSPEKT_YOLO_Weitertraining_Colab.ipynb)

## Installation

```bash
pip install -r requirements.txt
export LABEL_STUDIO_API_KEY=<Token aus Label Studio: Account & Settings>
export LABEL_STUDIO_URL=http://localhost:8082
```

## Skripte

| Skript | Zweck |
| --- | --- |
| `make_label_config.py` | Erzeugt die Label-Studio-Labeling-Config (XML) aus den Klassen eines `.pt`-Modells — UI-Labels und Modellklassen bleiben garantiert identisch. |
| `prelabel_predictions.py` | **Vorlabeling:** schreibt YOLO-Vorhersagen als Predictions in alle Tasks ohne Vorhersage (Batch-Alternative zum ML-Backend; cron-tauglich). |
| `export_yolo_dataset.py` | **Export:** lädt alle annotierten Tasks, konvertiert sie in einen YOLO-Datensatz mit train/val-Split und `data.yaml`. Klassenreihenfolge kommt aus dem Basismodell (stabile Indizes!). `--zip` erzeugt das Upload-Archiv für Colab. |

## Typischer Ablauf

```bash
# 1. Einmalig: Labeling-Config für das Label-Studio-Projekt erzeugen
python make_label_config.py --model ../webapp/model/system180custommodel_v1.pt

# 2. Nach jedem Bild-Sync: Vorlabeling
python prelabel_predictions.py --project 4 \
    --model ../webapp/model/system180custommodel_v1.pt

# 3. Wenn genug gelabelt ist: Datensatz exportieren und zippen
python export_yolo_dataset.py --project 4 \
    --model ../webapp/model/system180custommodel_v1.pt \
    --out yolo_dataset --zip

# 4. yolo_dataset.zip nach Google Drive laden und das Colab-Notebook ausführen.
```

## Hinweise

- `--media-root` nutzen, wenn die Bilder als Local-Files-Storage auf derselben
  Maschine liegen — spart die HTTP-Downloads.
- Für das Nubs-Modell (Seitenkameras) ein eigenes Label-Studio-Projekt und das
  entsprechende `.pt` verwenden; die Skripte sind modell-agnostisch.
- Bricht `export_yolo_dataset.py` mit „labels unknown to the model" ab, wurden
  in Label Studio Klassen verwendet, die das Basismodell nicht kennt — entweder
  Tippfehler korrigieren oder bewusst mit erweiterter Klassenliste neu trainieren
  (dann `--classes` mit der neuen Liste verwenden).
