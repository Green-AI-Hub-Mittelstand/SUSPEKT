# Konzept: Kontinuierliches Weitertraining der YOLO-Modelle (SUSPEKT / System 180)

Dieses Dokument beschreibt, wie Demonstrator und Webseite so erweitert werden, dass
System 180 die bestehenden YOLO-Modelle (PT/ONNX/TensorRT) eigenständig mit neuen
Bildern weiter trainieren und verbessern kann — inklusive Werkzeugvergleich
(Label Studio vs. Roboflow vs. CVAT), Zielarchitektur und Deployment.

**Kurzfassung der Empfehlung:**

> **Label Studio (self-hosted) für Erfassung + Labeling, Google Colab für das
> GPU-Training, Deployment wie bisher (PT für die Webapp, ONNX→TensorRT auf dem
> Jetson).** Roboflow ist eine valide Komfort-Alternative, bringt aber laufende
> Kosten, US-Cloud-Datenhaltung und Lock-in-Risiken mit — für ein
> Green-AI-Hub-Pilotprojekt mit proprietären Produktdaten ist die
> Open-Source-Kette vorzuziehen. Ein Teil der Label-Studio-Anbindung existiert
> bereits im Code (`webapp/webapp/modelTraining.py`) und wird hier auf ein
> robustes, plattformunabhängiges Fundament gestellt.

---

## 1. Ausgangslage

| Baustein | Stand |
| --- | --- |
| **Webseite** (`webapp/`) | FastAPI-App: Upload → YOLO-Erkennung → Review → Neo4j. Login/Session vorhanden (`webapp/auth.py`). Trainingsseite `/training` mit rudimentärer Label-Studio-Anbindung (`webapp/modelTraining.py`). Upload-Daten werden bereits gesammelt (`webapp/trainingDataCollector.py`). |
| **Demonstrator** (`suspekt-demonstrator`) | Jetson Orin Nano, 3 Kameras: 1× USB-Webcam (Draufsicht, Mitte), 2× OAK-1 Max (Seitenansichten links/rechts). FastAPI-UI (`src/demonstrator/apps/normal.py`), TensorRT-Engines, Erfassung per `/api/erfassung`, Etikettendruck. |
| **Modelle** | Ultralytics YOLO (`.pt`), z. B. `251104_real_Y12m_detect_29cls.pt` (29 Klassen, Draufsicht), `251105_nubs_Y12m_detect_2cls.pt` (NubsUp/NubsDown, Seitenkameras), `251101_components_Y11n_detect_6cls.pt`. Für den Jetson exportiert nach ONNX → TensorRT (`tools/convert_models.sh`). |

### Schwächen des bestehenden Trainings-Codes (`webapp/webapp/modelTraining.py`)

Diese Punkte müssen behoben werden, bevor die Pipeline produktiv genutzt wird —
die Skripte unter [`training/`](../../training/) tun genau das:

1. **Instabile Klassen-IDs:** Die Klassenliste (`label_map`) wird dynamisch aus den
   heruntergeladenen Annotationen aufgebaut. Die Klassenindizes hängen damit von
   der zufälligen Reihenfolge der Tasks ab. Beim Weitertrainieren eines
   bestehenden Modells verschieben sich so die Bedeutungen der Klassen —
   das Modell „verlernt". **Fix:** feste Klassenliste, abgeleitet aus dem
   Basis-Modell (`model.names`).
2. **Windows-gebundene Pfade** (`C:/runs/detect`, `AppData/.../label-studio/media`) —
   funktioniert weder auf dem Linux-Server noch auf dem Jetson.
3. **Training auf dem Webserver** blockiert die App und braucht eine GPU, die dort
   nicht vorhanden ist. **Fix:** Training nach Google Colab auslagern.
4. **Kein Replay-Mix:** Es werden nur *neue* Bilder trainiert (10 Epochen auf
   Mini-Datensätzen) → katastrophales Vergessen. **Fix:** neue Bilder immer mit
   einem Anteil des Alt-Datensatzes mischen (siehe Abschnitt 6).
5. **Kein Qualitäts-Gate:** Das neue Modell überschreibt das alte ohne Vergleich
   der Validierungsmetriken. **Fix:** mAP-Vergleich im Colab-Notebook, bewusstes
   Deployment mit Versionsschema (`YYMMDD_<zweck>_<basis>_<task>_<n>cls.pt`).

---

## 2. Werkzeugvergleich: Label Studio vs. Roboflow vs. CVAT

| Kriterium | **Label Studio** (self-hosted) | **Roboflow** (Cloud) | **CVAT** (self-hosted) |
| --- | --- | --- | --- |
| Kosten | kostenlos (Open Source) | Free-Tier stark limitiert (u. a. öffentliche Datasets); private Nutzung → Bezahlplan | kostenlos (Open Source) |
| Datenhoheit / DSGVO | Daten bleiben im Haus | Daten in US-Cloud | Daten bleiben im Haus |
| Vorlabeling mit dem **eigenen** YOLO-Modell | ✅ ML-Backend (offizielles YOLO-Beispiel) oder Predictions per API | ✅ eigene Gewichte hochladbar („Label Assist") | ⚠️ möglich, aber Deployment über Nuclio-Functions deutlich aufwendiger |
| Training | ❌ extern nötig (→ Colab) | ✅ gehostetes Training inklusive | ❌ extern nötig |
| Export trainierter Gewichte | — (Training extern, volle Kontrolle) | ⚠️ plan-/modellabhängig, Lock-in-Risiko | — |
| YOLO-Datensatz-Export | ✅ nativ | ✅ nativ | ✅ nativ |
| API/Automatisierung (Demonstrator-Anbindung) | ✅ REST-API + Python-SDK | ✅ API + „Workflows" | ✅ REST-API |
| Bedienbarkeit für Nicht-Experten | gut | sehr gut | mittel |
| Betriebsaufwand | 1 Docker-Container | keiner | mehrere Container |
| Bereits im Projekt angelegt | ✅ (`modelTraining.py`, `.env`) | ❌ | ❌ |

### Empfehlung

**Label Studio + Google Colab.** Gründe:

- **Datenhoheit:** Produktions- und Produktbilder von System 180 verlassen das
  Haus nicht (bzw. nur kontrolliert als Trainings-Zip Richtung Google Drive —
  auch das ließe sich bei Bedarf durch eine lokale GPU ersetzen, ohne die
  restliche Pipeline zu ändern).
- **Kostenfrei und ohne Lock-in:** Die gesamte Kette (Label Studio, Ultralytics,
  Colab Free/Pro) ist ohne Abo nutzbar; das Wissen bleibt übertragbar.
- **Anschlussfähig:** Die `.env` der Webapp und `modelTraining.py` referenzieren
  Label Studio bereits — der Umbau ist eine Härtung, kein Neuanfang.
- **Vorlabeling:** Das offizielle Label-Studio-ML-Backend hat ein fertiges
  YOLO-Beispiel, in das die bestehenden `.pt`-Gewichte direkt eingehängt werden.

**Wann wäre Roboflow trotzdem die richtige Wahl?** Wenn (a) niemand den
Label-Studio-Container betreiben will, (b) Budget für einen Bezahlplan vorhanden
ist und (c) die Datenschutzfrage (Auftragsverarbeitung, US-Cloud) geklärt wurde.
Dann ersetzt Roboflow Label Studio *und* Colab in einem Werkzeug: Bilder
hochladen → Label Assist mit eigenen Gewichten → gehostetes Training →
Gewichte-Export. Ein „Roboflow-Workflow/Agent" kann zusätzlich Inferenz- und
Automationsketten abbilden, ist aber für den Kern-Loop (labeln → trainieren →
deployen) nicht erforderlich. Der Colab-Pfad in
[`notebooks/SUSPEKT_YOLO_Weitertraining_Colab.ipynb`](../../notebooks/SUSPEKT_YOLO_Weitertraining_Colab.ipynb)
enthält eine optionale Roboflow-Zelle, falls der Datensatz dort gepflegt wird.

---

## 3. Zielarchitektur: der Trainings-Loop

```
   Demonstrator (Jetson, 3 Kameras)                Webseite (Server)
   ┌──────────────────────────────┐        ┌────────────────────────────┐
   │ [1] Knopf „Trainingsbild     │        │ [1b] Admin: „Für Training  │
   │     speichern“               │        │      speichern“ (Upload)   │
   │  → 3 Frames + Metadaten      │        │  → Original + Detections   │
   └──────────────┬───────────────┘        └──────────────┬─────────────┘
                  │  Sync (rsync/Share)                   │
                  ▼                                       ▼
          ┌─────────────────────────────────────────────────────┐
          │ [2] Label Studio (Docker, self-hosted)              │
          │     • Local Storage auf Capture-Ordner              │
          │     • Vorlabeling: ML-Backend mit bestehendem .pt   │
          │       (alternativ: training/prelabel_predictions.py)│
          │     • Mensch korrigiert nur noch die Vorschläge     │
          └──────────────────────────┬──────────────────────────┘
                                     │ [3] Export
                                     ▼
          ┌─────────────────────────────────────────────────────┐
          │ training/export_yolo_dataset.py                     │
          │  → YOLO-Datensatz (train/val, feste Klassenliste)   │
          │  → Zip nach Google Drive                            │
          └──────────────────────────┬──────────────────────────┘
                                     │ [4] Training (GPU)
                                     ▼
          ┌─────────────────────────────────────────────────────┐
          │ Google Colab: notebooks/..._Colab.ipynb             │
          │  • Fine-Tuning ab bestehendem .pt (+ Replay-Mix)    │
          │  • Validierung: mAP-Vergleich alt vs. neu           │
          │  • Export: best.pt + best.onnx (versioniert)        │
          └──────────────────────────┬──────────────────────────┘
                                     │ [5] Deployment
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
          Webseite: webapp/model/<neu>.pt   Jetson: ONNX → TensorRT
          (MODEL_NAME in .env umstellen)    (tools/convert_models.sh)
```

---

## 4. Erweiterung des Demonstrators (Knopfdruck-Capture)

Änderungen im Repo `suspekt-demonstrator` (hier beschrieben, da dieses Konzept im
SUSPEKT-Repo liegt):

1. **Neuer Endpoint `POST /api/trainingsbild`** in
   `src/demonstrator/apps/normal.py`:
   - holt von allen drei Streams das aktuelle **Voll-Frame**
     (Center: `get_latest_full_frame()` — wichtig: *nicht* das ROI-beschnittene
     Frame, damit die Trainingsbilder die volle Szene zeigen; die OAK-Handler
     analog über ihre letzten Frames),
   - speichert die drei Bilder als
     `data/training_captures/<YYYYmmdd_HHMMSS>_{center|left|right}.jpg`,
   - schreibt eine Metadaten-JSON daneben (Zeitstempel, Kamera-Rolle,
     ROI-Konfiguration, aktuelle Detections inkl. Klasse/Box/Confidence als
     spätere Vorlabel-Hilfe),
   - gibt Anzahl und Pfade zurück, damit die UI Feedback zeigen kann.
2. **Button in `templates/normal_index.html`** neben dem bestehenden
   „Erfassen"-Button: „📷 Trainingsbild speichern" → ruft den Endpoint auf und
   zeigt eine Bestätigung („3 Bilder gespeichert, Nr. 47").
3. **Sync zum Label-Studio-Server:** Der Ordner `data/training_captures/` wird
   per `rsync`-Cronjob (oder Netzwerk-Share/Syncthing) auf den Server gespiegelt,
   auf dem Label Studio läuft. Alternativ kann
   [`training/prelabel_predictions.py`](../../training/prelabel_predictions.py)
   direkt auf dem Jetson laufen und die Bilder inkl. Vorhersagen per API
   hochladen — dann entfällt der Datei-Sync.

> **Hinweis Seitenkameras:** Links/rechts dienen der Nubs-Erkennung
> (2-Klassen-Modell). Sinnvoll ist **ein Label-Studio-Projekt pro Modell**:
> Projekt „Komponenten" (Draufsicht, 29 Klassen) und Projekt „Nubs"
> (Seitenansichten, 2 Klassen). Die Capture-Metadaten enthalten die Kamera-Rolle,
> sodass die Bilder automatisch dem richtigen Projekt zugeordnet werden können.

## 5. Erweiterung der Webseite (Admin-Bereich)

1. **„Für Training speichern"** im Ergebnis-Review (`templates/results.html`):
   Der Hook existiert faktisch schon — `TrainingDataCollector.save_training_data()`
   legt Original, Crops und Annotationen ab. Neu: zusätzlich in den
   Label-Studio-Sync-Ordner schreiben (bzw. per API hochladen), damit die Bilder
   im Labeling-Backlog landen.
2. **Admin-Navigation:** Link „Labeling" (öffnet Label Studio, gleicher Server,
   eigener Port/Reverse-Proxy-Pfad in `nginx.conf`) und Link „Training"
   (bestehende Seite `/training`, umgebaut zu einer Status-/Anleitung-Seite:
   Anzahl ungelabelter Bilder, letzter Export, Link zum Colab-Notebook).
   Zugriff über das bestehende Login (`webapp/auth.py`) absichern.
3. **`/training/train` entschärfen:** Das serverseitige Training entfällt
   zugunsten des Colab-Wegs. Die Route kann stattdessen den Datensatz-Export
   (`training/export_yolo_dataset.py`) anstoßen und das Zip zum Download anbieten.

## 6. Label Studio einrichten

```bash
# docker-compose.labelstudio.yml (auf dem Webseiten-Server)
services:
  labelstudio:
    image: heartexlabs/label-studio:latest
    ports: ["8082:8080"]
    volumes:
      - ./labelstudio-data:/label-studio/data
      - ./training_captures:/label-studio/files:ro   # Sync-Ordner vom Demonstrator
    environment:
      - LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
      - LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/files
```

- **Projekt anlegen** (je Modell eins), Labeling-Config mit
  [`training/make_label_config.py`](../../training/make_label_config.py) direkt
  aus den Modellklassen generieren — so sind UI-Labels und Modellklassen
  garantiert identisch.
- **Local Storage** auf `/label-studio/files` zeigen lassen und „Sync" nutzen —
  neue Demonstrator-Bilder erscheinen automatisch als Tasks.
- **Vorlabeling**, zwei gleichwertige Wege:
  - **ML-Backend (empfohlen, „live"):** offizielles
    [label-studio-ml-backend](https://github.com/HumanSignal/label-studio-ml-backend)
    mit dem YOLO-Beispiel, eigenes `.pt` einhängen. Label Studio fragt das
    Backend beim Öffnen eines Tasks an und zeigt die Boxen sofort.
  - **Batch-Skript (einfacher Betrieb):**
    [`training/prelabel_predictions.py`](../../training/prelabel_predictions.py)
    läuft per Cron und schreibt Predictions für alle Tasks ohne Vorhersage.
    Kein zusätzlicher Dienst nötig.
- Die Annotator:innen **korrigieren nur noch**: falsche Klasse umschalten, Boxen
  nachziehen, Fehldetektionen löschen, Übersehenes ergänzen. Das ist der
  eigentliche Effizienzgewinn des Vorlabelings (erfahrungsgemäß 3–10× schneller
  als Labeln von Null).

## 7. Training in Google Colab

Vollständig vorbereitet in
[`notebooks/SUSPEKT_YOLO_Weitertraining_Colab.ipynb`](../../notebooks/SUSPEKT_YOLO_Weitertraining_Colab.ipynb):
Drive einbinden → Datensatz-Zip entpacken → Klassenabgleich mit dem Basismodell
→ Fine-Tuning → Validierung mit mAP-Vergleich alt/neu → Export `best.pt` +
`best.onnx` (versioniert) zurück nach Drive.

Fachliche Leitplanken (im Notebook umgesetzt bzw. dokumentiert):

- **Feste Klassenliste:** Reihenfolge kommt immer aus dem Basismodell
  (`model.names`); das Notebook remappt abweichende Datensatz-Indizes automatisch
  und bricht bei unbekannten Klassen ab.
- **Replay-Mix gegen Vergessen:** Neue Bilder mit ~30–50 % Alt-Daten mischen
  (der historische Datensatz liegt einmalig als Zip in Drive). Nie nur auf den
  neuen 20 Bildern trainieren.
- **Konservative Hyperparameter fürs Fine-Tuning:** niedrige Lernrate
  (`lr0=0.001`), `epochs=50` mit `patience=20`, `imgsz=640` (wie die
  Originaltrainings; die 320er-Auflösung ist nur eine Jetson-Deploy-Optimierung).
- **Qualitäts-Gate:** Deployment nur, wenn mAP50-95 auf dem festen
  Validierungs-Set ≥ Vorgängermodell.
- **Ressourcen:** Colab Free (T4) reicht für Fine-Tuning-Läufe dieser Größe
  (Richtwert: 29-Klassen-Modell, einige hundert Bilder, 50 Epochen ≈ 1–2 h).
  Bei häufigem Training lohnt Colab Pro; Sitzungslimits beachten (Laufzeit-Abbrüche
  → `resume=True` wird im Notebook erklärt).

## 8. Deployment des neuen Modells

| Ziel | Schritte |
| --- | --- |
| **Webseite** | `best.pt` versioniert nach `webapp/model/` kopieren (Schema `YYMMDD_<zweck>_<basis>_<task>_<n>cls.pt`), `MODEL_NAME` in `.env` umstellen, Container neu starten. Altes Modell liegen lassen → Rollback = `.env` zurückstellen. |
| **Demonstrator (Jetson)** | `best.pt` nach `models/` kopieren, `YOLO_SOURCE_*` in `src/demonstrator/config/settings.py` anpassen, dann `tools/convert_models.sh` **auf dem Jetson** ausführen (TensorRT-Engines sind gerätespezifisch und müssen auf der Zielhardware gebaut werden, FP16/imgsz 320 wie gehabt). |
| **ONNX** | erzeugt das Notebook mit (`opset`-kompatibel zur Jetson-TensorRT-Version); wird nur als Zwischenformat für TensorRT gebraucht. |

## 9. Umsetzungsreihenfolge (Vorschlag)

1. Label Studio per Docker auf dem Webseiten-Server aufsetzen, Projekte +
   Labeling-Configs anlegen (`make_label_config.py`), Accounts für System 180.
2. Vorlabeling in Betrieb nehmen (`prelabel_predictions.py` per Cron, später
   optional ML-Backend).
3. Demonstrator: Capture-Endpoint + Button + Sync (Abschnitt 4).
4. Webseite: Admin-Links + „Für Training speichern" an Label Studio anbinden
   (Abschnitt 5); alte `modelTraining.py`-Route stilllegen/umbauen.
5. Ersten Colab-Durchlauf gemeinsam durchführen (= Teil der Schulung,
   siehe [`schulung.md`](schulung.md)), Qualitäts-Gate dokumentieren.
6. Übergabe an System 180 mit Checklisten aus der Schulung.
