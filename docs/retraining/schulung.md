# Schulung: „Unser KI-Modell selbst verbessern" (System 180)

Schulungskonzept für die Übergabe der Weitertrainings-Pipeline an System 180.
Grundlage ist die in [`konzept.md`](konzept.md) beschriebene Kette:
**Demonstrator/Webseite → Label Studio → Google Colab → Deployment**.

## Rahmen

| | |
| --- | --- |
| **Zielgruppen** | (A) Anwender:innen in Werkstatt/Lager: Bilder erfassen, labeln. (B) Technik-Verantwortliche:r („KI-Pate/Patin", 1–2 Personen): Training, Deployment, Betrieb. |
| **Vorkenntnisse** | keine ML-Kenntnisse nötig; Gruppe B: grundlegende PC-/Terminal-Sicherheit |
| **Format** | 1 Präsenztag am Demonstrator (6–7 h netto) — alternativ 2 halbe Tage: Tag 1 = Module 1–4 (alle), Tag 2 = Module 5–7 (nur Gruppe B) |
| **Teilnehmerzahl** | 3–8 Personen |
| **Ort/Technik** | Demonstrator vor Ort, echte System-180-Bauteile, 1 Laptop pro 2 Personen (Browser genügt), Google-Konto für Colab, Beamer |
| **Vorbereitung durch DFKI** | Label Studio läuft mit angelegten Projekten + Accounts; Vorlabeling aktiv; Drive-Ordner mit Basis-Datensatz und Notebook eingerichtet; ~20 ungelabelte Beispielbilder im Backlog |

## Agenda (Tagesformat)

| Zeit | Modul | Zielgruppe |
| --- | --- | --- |
| 09:00–09:45 | 1 — Wie „sieht" unsere KI? | alle |
| 09:45–10:15 | 2 — Der SUSPEKT-Kreislauf | alle |
| 10:30–11:15 | 3 — Trainingsbilder erfassen (Hands-on am Demonstrator) | alle |
| 11:15–12:30 | 4 — Labeln mit Label Studio (Hands-on) | alle |
| 13:15–14:45 | 5 — Training in Google Colab (Hands-on) | B (A optional) |
| 15:00–15:45 | 6 — Das neue Modell in Betrieb nehmen | B |
| 15:45–16:30 | 7 — Betrieb, Qualität, Spielregeln + Abschlussquiz | alle |

---

## Modul 1 — Wie „sieht" unsere KI? (45 min)

**Lernziele:** verstehen, was Objekterkennung ist, was ein Modell gelernt hat und
warum es Fehler macht.

Inhalte:
- Objekterkennung anschaulich: Klassen, Bounding Box, Confidence — live am
  Demonstrator mit echten Bauteilen zeigen (Diagonale, Gerade, Mutternstab …).
- „Das Modell kennt nur, was es gesehen hat": Trainingsdaten = Erfahrung.
  Beispiele provozieren: ungewöhnliche Lage, Verdeckung, fremdes Objekt,
  anderes Licht → Fehldetektionen zeigen und erklären.
- Die drei Modelle des Projekts (Draufsicht-Komponenten, Nubs links/rechts,
  Zustands-CNN) und was sie jeweils tun.
- Kernbotschaft: **Mehr gute, vielfältige Bilder → besseres Modell. Genau dafür
  ist die heutige Schulung da.**

## Modul 2 — Der SUSPEKT-Kreislauf (30 min)

**Lernziele:** den Gesamtprozess und die eigene Rolle darin kennen.

- Architekturbild aus `konzept.md` Schritt für Schritt: Erfassen → Vorlabeln →
  Korrigieren → Exportieren → Trainieren → Prüfen → Ausrollen.
- Wer macht was: Gruppe A erfasst und labelt „nebenbei" im Alltag; Gruppe B
  trainiert z. B. monatlich oder nach ~100 neuen Bildern.
- Was ist Vorlabeling: Das *aktuelle* Modell schlägt Boxen vor, Menschen
  korrigieren nur. Aufwand pro Bild: Sekunden statt Minuten.

## Modul 3 — Trainingsbilder erfassen (45 min, Hands-on)

**Lernziele:** am Demonstrator und in der Webseite gezielt gute Trainingsbilder
speichern.

- Demonstrator: Bauteil auflegen → Knopf „Trainingsbild speichern" → 3 Bilder
  (Draufsicht + Seiten) landen automatisch im Labeling-Backlog.
- Webseite: Upload → Ergebnis prüfen → „Für Training speichern".
- **Was sind gute Trainingsbilder?** (Checkliste als Aushang für den
  Demonstrator):
  - Vielfalt schlägt Menge: verschiedene Lagen, Drehungen, Kombinationen,
    Lichtverhältnisse, auch teilweise Verdeckung.
  - Gerade die **Fehlerfälle** speichern! Wenn das Modell etwas falsch erkennt,
    ist genau dieses Bild Gold wert.
  - Bauteil vollständig im Bild, scharf, keine Hände im Bild (außer gewollt).
  - Keine Duplikate: 20× dasselbe unbewegte Bauteil bringt nichts.
- **Übung:** Jede:r erfasst 3 Situationen: (1) Standardlage, (2) ungewöhnliche
  Lage, (3) eine aktuelle Fehldetektion.

## Modul 4 — Labeln mit Label Studio (75 min, Hands-on)

**Lernziele:** Vorlabels sicher korrigieren, Labeling-Richtlinien anwenden.

- Login, Projektübersicht (Projekt „Komponenten" vs. „Nubs"), Task öffnen.
- Werkzeuge: Box verschieben/nachziehen, Klasse ändern, Box löschen, neue Box,
  Tastaturkürzel, „Submit" vs. „Skip".
- **Labeling-Richtlinien** (als eigenes Handout drucken):
  1. Box eng ans Bauteil (kein Luftrand > wenige Pixel), das ganze Bauteil
     einschließen — auch verdeckte Teile so weit sichtbar.
  2. Jedes sichtbare Bauteil labeln, auch angeschnittene am Bildrand
     (ab ~20 % Sichtbarkeit).
  3. Klassenzweifel? → Klassenkatalog (Fotos aller 29 Klassen) prüfen; bleibt
     es unklar: „Skip" + Kommentar, niemals raten.
  4. Vorgeschlagene Boxen nie ungeprüft übernehmen — der häufigste Fehler ist
     blindes Bestätigen.
  5. Konsistenz vor Perfektion: lieber einheitlich „gut" als uneinheitlich
     „perfekt".
- **Übung:** Die in Modul 3 erzeugten Bilder labeln (mit Vorlabels).
  Anschließend Peer-Review: Nachbar:in prüft 3 Tasks, Abweichungen diskutieren.
- Qualitätsblick für Gruppe B: Annotator-Übereinstimmung stichprobenartig
  prüfen, Kommentar-Funktion, Filter „ohne Annotation".

## Modul 5 — Training in Google Colab (90 min, Hands-on, Gruppe B)

**Lernziele:** einen kompletten Trainingslauf selbstständig durchführen und die
Ergebnisse beurteilen.

- Export aus Label Studio (`training/export_yolo_dataset.py` bzw. Button in der
  Webseite) → Zip nach Google Drive.
- Notebook `SUSPEKT_YOLO_Weitertraining_Colab.ipynb` Zelle für Zelle:
  GPU-Laufzeit wählen, Drive einbinden, Konfiguration (Basismodell, Datensatz),
  Klassenabgleich, Training starten.
- **Während das Training läuft** (~Kaffeepause-tauglich) Theorie kompakt:
  - Train/Val-Split: warum das Modell auf ungesehenen Bildern geprüft wird.
  - Metriken lesen: Precision („Wie viele Treffer waren richtig?"), Recall
    („Wie viel wurde gefunden?"), mAP50 / mAP50-95 als Gesamtnote,
    Confusion Matrix („Welche Klassen werden verwechselt?").
  - Replay-Mix: warum immer Alt-Daten beigemischt werden (Vergessen vermeiden).
  - Epochen, Early Stopping, was `best.pt` bedeutet.
- Ergebnis beurteilen: **Qualitäts-Gate** — neues Modell nur ausrollen, wenn
  mAP50-95 ≥ Vorgänger (Vergleichszelle im Notebook).
- **Übung:** kompletter Lauf mit dem vorbereiteten Übungsdatensatz; Metriken im
  Team interpretieren; ONNX-Export ausführen.
- Praxisfragen: Colab-Sitzungslimits, `resume=True` nach Abbruch, Kosten
  (Free vs. Pro), wie lange dauert was.

## Modul 6 — Das neue Modell in Betrieb nehmen (45 min, Gruppe B)

**Lernziele:** Deployment und Rollback beherrschen.

- Versionsschema `YYMMDD_<zweck>_<basis>_<task>_<n>cls.pt` und Modell-Logbuch
  (einfache Tabelle: Datum, Datensatzgröße, mAP, wer, Bemerkung).
- Webseite: Gewichte nach `webapp/model/`, `MODEL_NAME` in `.env`, Neustart,
  Smoke-Test mit 3 Referenzbildern.
- Jetson: Gewichte nach `models/`, `settings.py` anpassen,
  `tools/convert_models.sh` **auf dem Gerät**, Demonstrator-Smoke-Test.
- **Rollback-Übung:** absichtlich „schlechtes" Modell eintragen und in < 5 min
  zurückwechseln.

## Modul 7 — Betrieb, Qualität, Spielregeln (45 min, alle)

- **Retraining-Rhythmus:** nach ~100 neuen gelabelten Bildern oder wenn sich
  Fehldetektionen im Alltag häufen; mindestens quartalsweise prüfen.
- **Datenhygiene:** Backlog nicht wachsen lassen (Ziel: < 50 ungelabelte Bilder),
  Duplikate löschen, „Skip"-Fälle wöchentlich klären.
- **Verantwortlichkeiten** festhalten: Wer labelt, wer trainiert, wer deployt,
  wer führt das Modell-Logbuch?
- Typische Fehler-Top-5 (blind bestätigte Vorlabels, Klassenliste verändert,
  nur neue Bilder trainiert, Deployment ohne Qualitäts-Gate, Engine nicht auf
  dem Jetson gebaut).
- **Abschlussquiz** (10 Fragen, gemeinsam): festigt Kernbotschaften.
- Feedbackrunde + Vereinbarung des ersten eigenständigen Trainingslaufs
  (Termin!) mit DFKI-Begleitung auf Abruf.

---

## Materialliste

- Handout 1: Labeling-Richtlinien (Modul 4) — 1 Seite, laminiert an den
  Demonstrator.
- Handout 2: Checkliste „Gute Trainingsbilder" (Modul 3) — 1 Seite, Aushang.
- Handout 3: Runbook „Training & Deployment" für Gruppe B — Schrittliste mit
  Screenshots aus Colab/Label Studio (bei Einrichtung erstellen).
- Klassenkatalog: 1 Foto + Name je Klasse (aus `docs/models.md` des
  Demonstrator-Repos ableiten).
- Modell-Logbuch-Vorlage (Tabelle).
- Übungsdatensatz (~50 gelabelte + 20 ungelabelte Bilder) in Drive.

## Erfolgskriterien der Schulung

- Jede Person aus Gruppe A hat ≥ 5 Bilder erfasst und ≥ 10 Tasks gelabelt.
- Gruppe B hat einen kompletten Trainingslauf inkl. ONNX-Export und
  (Test-)Deployment mit Rollback durchgeführt.
- Der erste eigenständige Trainingslauf ist terminiert.
