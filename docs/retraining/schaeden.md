# Schäden erkennen: Kratzer, Dellen, MDF-Platzer

Diese Anleitung beschreibt, wie SUSPEKT Schäden auf Bauteilen erkennt und wie
ihr die Erkennung selbst aufbaut und verbessert — auch **ohne viele
Fehlerbilder**. Alles Wichtige läuft über die Seite **`/training`** der Webapp.

Die Schadensklassen entsprechen dem Zustands-Feld der Ergebnisseite:

| Klasse | betrifft | Beispiele |
|---|---|---|
| `Rohr_Kratzer` | Rohre (Gerade, Diagonale, Mutternstab, Griff, …) | Kratzer in der Beschichtung |
| `Delle` | vor allem Rohre | Eindellungen, Stauchungen |
| `MDF-Platzer` | Platten (Verkleidung, Systemboden, Türen, …) | ausgebrochene Kanten, Abplatzer |

Die Prüfung läuft zweistufig: Das Komponentenmodell findet die Bauteile, dann
wird **jeder Bauteil-Ausschnitt (Crop)** auf Schäden geprüft. Pro Bauteiltyp
sind nur die physikalisch möglichen Schäden zugelassen (Rohre → Kratzer/Delle,
Platten → MDF-Platzer/Delle).

## Weg 1 — Verdachts-Ampel (sofort einsatzbereit, keine Fehlerbilder nötig)

Die Verdachts-Ampel ist eine **Anomalie-Erkennung**: Sie lernt aus Bildern
*unbeschädigter* Teile, wie „normal" aussieht. Weicht ein erkanntes Bauteil
davon ab (Kratzer, Delle, Schmutz, …), erscheint im Erkennungsergebnis der
Hinweis **„⚠ KI-Verdacht: bitte prüfen"** — die Entscheidung trifft der Mensch.

**Einrichten (einmalig, ~10 Minuten):**

1. Bilder unbeschädigter Teile in den Ordner `training_captures/komponenten/`
   legen (dieselben Bilder, die auch fürs Labeln gesammelt werden; ab ~50
   Bildern wird es brauchbar).
2. Auf `/training` im Abschnitt **„Verdachts-Ampel"** auf
   **„Aus Gutbildern lernen"** klicken und warten.
3. Fertig — ab sofort werden alle Erkennungen mitgeprüft. Nach neuen
   Gutbildern einfach **„Neu lernen"** klicken.

Technik: PatchCore-light — ResNet18-Patch-Features pro Bauteilklasse in einer
Gutteil-Referenz („Memory Bank", `webapp/model/anomaly/`). Kein GPU nötig.

**Grenzen:** Die Ampel sagt nur *dass* etwas auffällig ist, nicht *was*. Sie
reagiert auch auf Schmutz oder Aufkleber. Deshalb ist sie eine
Vorsortierung — die genaue Klassifizierung übernimmt Weg 2.

## Weg 2 — Eigenes Schadensmodell (der normale Kreislauf)

Genau derselbe Ablauf wie beim Komponenten-Retraining, nur mit dem Projekt
**„Schäden"**:

1. **Sammeln:** Fotos beschädigter Teile in `training_captures/schaeden/`
   legen. Wichtig: Streiflicht (Licht flach von der Seite) macht Kratzer und
   Dellen sichtbar.
2. **Projekt anlegen:** `/training` → Einrichtungs-Kasten → „Schäden …" →
   **„Projekt anlegen"**. Das Label-Set (MDF-Platzer, Rohr_Kratzer, Delle)
   ist fest eingebaut — es braucht noch kein Modell.
3. **Labeln:** Rechteck eng um jeden Schaden ziehen, Klasse wählen.
   Beim allerersten Datensatz gibt es keine Vorschläge — ab dem zweiten
   Training labelt das Modell vor.
4. **Trainieren:** „Datensatz für Colab (zip)" herunterladen, Colab-Notebook
   öffnen, `MODELL_ZWECK = 'schaeden'` setzen, Zellen durchlaufen lassen.
   Das erste Training startet automatisch vom generischen `yolo11n.pt` und
   nutzt eine höhere Auflösung (1280), weil Schäden klein sind.
5. **Einspielen:** Die fertige `.pt` per Drag & Drop auf `/training`
   hochladen — sie wird am Dateinamen (`…_schaeden_…`) erkannt und aktiviert.
   Ab sofort setzt jede Erkennung den Zustand beschädigter Teile automatisch
   (mit KI-Konfidenz-Badge); die manuelle Korrektur bleibt möglich.

Richtwert: 150–300 gelabelte Schadensstellen pro Klasse für einen brauchbaren
Start. MDF-Platzer lernt das Modell schnell; feine Kratzer und flache Dellen
brauchen gutes Licht bei der Aufnahme.

## Weg 3 — Zu wenige Fehlerbilder? Daten künstlich erzeugen

### 3a. Cut-&-Paste-Augmentation (`training/augment_defects.py`)

Aus **wenigen** echten Schadensfotos werden hunderte Trainingsbilder: Die
Schadensstellen werden als kleine Ausschnitte auf viele Gutbilder kopiert
(zufällig skaliert/gedreht/angepasst, mit weichem Rand) — Labels entstehen
automatisch.

```bash
# Einmalig: Schadens-Ausschnitte in Ordner je Klasse speichern:
#   schadens_patches/MDF-Platzer/…  /Rohr_Kratzer/…  /Delle/…
python training/augment_defects.py \
    --gut training_captures/komponenten \
    --patches schadens_patches \
    --out yolo_dataset_schaeden --anzahl 500
# Ordner zippen und in Colab mit MODELL_ZWECK='schaeden' trainieren.
```

### 3b. Rendern aus CAD (`training/synthetic/`)

Für Fortgeschrittene: BlenderProc rendert System180-Geometrie mit prozedural
erzeugten Schäden und pixelgenauen Labels — unbegrenzt skalierbar. Siehe
`training/synthetic/README.md`. Regel für beide Wege: synthetisch
**vortrainieren**, mit echten Bildern **nachschärfen** (Replay-Mix im
Notebook) — und jedes real auftauchende Schadensteil weiter fotografieren.

## Empfohlene Reihenfolge

1. **Heute:** Verdachts-Ampel aktivieren (Weg 1) — kostet nichts, hilft sofort.
2. **Diese Woche:** Projekt „Schäden" anlegen, erste Schadensfotos sammeln,
   mit Cut-&-Paste (Weg 3a) den ersten Trainingsdatensatz aufblasen, erstes
   Schadensmodell trainieren und hochladen.
3. **Laufend:** Jeden echten Schaden fotografieren und labeln — mit jedem
   Training wird das Modell besser, die Verdachts-Ampel bleibt als
   Sicherheitsnetz aktiv.
4. **Bei Bedarf:** CAD-Rendering (Weg 3b), wenn bestimmte Schadensbilder in
   der Realität zu selten sind.
