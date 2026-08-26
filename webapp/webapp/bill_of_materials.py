"""Stückliste eines Möbels aus mehreren Ansichten.

Ein Möbel wird aus mehreren Richtungen fotografiert. Die Bauteile einfach über
alle Ansichten zu summieren, zählt doppelt: eine Strebe, die vorne rechts sitzt,
erscheint in der rechten Seitenansicht erneut.

Die Lösung nutzt die Systemlogik statt einer Bild-zu-Bild-Zuordnung, die ohne
3D-Rekonstruktion ohnehin nicht verlässlich wäre:

* **Tragende Teile** (Streben, Mutternstäbe, Knoten, Böden, Füße) sind durch das
  Raster festgelegt. Steht die Zahl der Spalten und Reihen fest, ergibt sich
  ihre Anzahl aus der Geometrie - unabhängig davon, wie viele davon die
  Erkennung tatsächlich gefunden hat.
* **Sichtbare Einbauten** (Diagonalen, Verkleidungen, Türen, Auszüge) sitzen
  jeweils in genau einer Ebene. Sie werden pro Ansicht gezählt, und die
  Ansichten zeigen unterschiedliche physische Teile - hier wird summiert.

Das Raster selbst wird über die Ausdehnung bestimmt, nicht über Stückzahlen:
Gesamthöhe geteilt durch Modulhöhe ergibt die Reihen. Das bleibt auch dann
richtig, wenn die Erkennung einzelne Streben übersehen hat.
"""
import statistics

FRONT_VIEWS = ("front", "back")
SIDE_VIEWS = ("left", "right")

# Tragende Teile - Anzahl folgt aus dem Raster.
STRUKTUR_KLASSEN = ("Gerade", "Mutternstab", "Noppenscheiben", "Schraube",
                    "Systemboden", "Rolle", "Sockelfuss", "Winkelfuss")

# Einbauten sitzen in einer bestimmten Ebene und dürfen nur aus den Ansichten
# gezählt werden, die diese Ebene zeigen. Eine Seitendiagonale ist zwar auch
# durch ein offenes Regal hindurch von vorne sichtbar - gezählt wird sie
# trotzdem nur in den Seitenansichten, sonst zählt man sie doppelt.
EINBAU_EBENE = {
    "Diagonale": SIDE_VIEWS,
    "Seitenverkleidung": SIDE_VIEWS,
    "Seitenverkleidung-0-0": SIDE_VIEWS,
    "Seitenverkleidung-0-IN": SIDE_VIEWS,
    "Seitenverkleidung-IN-IN": SIDE_VIEWS,
    "Seitenverkleidung - 0/0": SIDE_VIEWS,
    "Seitenverkleidung - 0/IN": SIDE_VIEWS,
    "Seitenverkleidung - IN/IN": SIDE_VIEWS,
    "Verkleidung": FRONT_VIEWS,
    "Einzeltuer": FRONT_VIEWS,
    "Doppeltuer": FRONT_VIEWS,
    "Doppeltuerblatt": FRONT_VIEWS,
    "Kombifront": FRONT_VIEWS,
    "Auszug": FRONT_VIEWS,
    "Magazin": FRONT_VIEWS,
    "Griff": FRONT_VIEWS,
}
EINBAU_KLASSEN = tuple(EINBAU_EBENE)

# Beim Spiegeln einer symmetrischen Aufnahme: Was vorne eine Tür oder ein Auszug
# ist, sitzt auf der Rückseite derselben Ebene in aller Regel als geschlossene
# Verkleidung ohne Griff. Nur die Front ist bebildert, die Rückseite wird
# angenommen - deshalb ist die Position in der Oberfläche als Annahme markiert
# und lässt sich bei der Freigabe korrigieren.
RUECKSEITE_STATT = {
    "Einzeltuer": "Verkleidung",
    "Doppeltuer": "Verkleidung",
    "Doppeltuerblatt": "Verkleidung",
    "Kombifront": "Verkleidung",
    "Auszug": "Verkleidung",
    "Griff": None,  # Ein Griff sitzt nur auf der Front.
}

# Ganze Möbel sind kein Bauteil.
IGNORIERTE_KLASSEN = ("Regal", "Sideboard", "Tisch", "Tresen", "Dekor")


def _basisklasse(name):
    """"Gerade 540" -> "Gerade"."""
    if not name:
        return ""
    kopf = str(name).split(" ")[0]
    return kopf if kopf else str(name)


def _masse(zeile):
    breite = abs(float(zeile["x_max"]) - float(zeile["x_min"]))
    hoehe = abs(float(zeile["y_max"]) - float(zeile["y_min"]))
    return breite, hoehe


def _ansicht(df):
    if "ansicht" not in df.columns or df.empty:
        return ""
    werte = [str(v).lower() for v in df["ansicht"].dropna().tolist()]
    return werte[0] if werte else ""


def erkenne_raster(df):
    """Bestimmt Spalten und Reihen aus der Ausdehnung des Möbels im Bild.

    Bewusst über die Ausdehnung und nicht über Stückzahlen: fehlt der Erkennung
    eine von sechs waagerechten Streben, wäre die Zählung um eine Ebene daneben,
    die Gesamthöhe aber kaum verändert.
    """
    if df is None or df.empty or "class" not in df.columns:
        return None

    streben = df[df["class"].apply(lambda c: _basisklasse(c) in
                                   ("Gerade", "Mutternstab", "Diagonale"))]
    if streben.empty:
        return None

    waagerecht, senkrecht = [], []
    for _, zeile in streben.iterrows():
        if _basisklasse(zeile["class"]) != "Gerade":
            continue
        breite, hoehe = _masse(zeile)
        if breite <= 0 or hoehe <= 0:
            continue
        (waagerecht if breite > hoehe else senkrecht).append((breite, hoehe))

    if not waagerecht or not senkrecht:
        return None

    modul_breite = statistics.median(b for b, _ in waagerecht)
    modul_hoehe = statistics.median(h for _, h in senkrecht)
    if modul_breite <= 0 or modul_hoehe <= 0:
        return None

    gesamt_breite = float(streben["x_max"].max()) - float(streben["x_min"].min())
    gesamt_hoehe = float(streben["y_max"].max()) - float(streben["y_min"].min())

    spalten = max(1, round(gesamt_breite / modul_breite))
    reihen = max(1, round(gesamt_hoehe / modul_hoehe))

    # Nennmaße aus den bereits auf das Raster gerundeten Längen.
    def _laenge(auswahl):
        werte = [z["laenge"] for _, z in auswahl.iterrows()
                 if z.get("laenge") and str(z.get("laenge")) != "nan"]
        return statistics.median(werte) if werte else None

    geraden = streben[streben["class"].apply(lambda c: _basisklasse(c) == "Gerade")]
    quer = geraden[geraden.apply(lambda z: _masse(z)[0] > _masse(z)[1], axis=1)]
    hoch = geraden[geraden.apply(lambda z: _masse(z)[0] <= _masse(z)[1], axis=1)]
    mutternstaebe = streben[streben["class"].apply(lambda c: _basisklasse(c) == "Mutternstab")]

    return {
        "spalten": spalten,
        "reihen": reihen,
        "systembreite": _laenge(quer),
        "systemhoehe": _laenge(hoch),
        "systemtiefe": _laenge(mutternstaebe),
    }


def _fusstyp(image_results):
    """Welcher Fußtyp verbaut ist, sagt die Erkennung - die Anzahl das Raster."""
    for df in image_results.values():
        if df is None or df.empty or "class" not in df.columns:
            continue
        for name in ("Rolle", "Sockelfuss", "Winkelfuss"):
            if (df["class"].apply(_basisklasse) == name).any():
                return name
    return "Fuß oder Rolle"


def _struktur_stueckliste(spalten, reihen, breite, hoehe, tiefe, fusstyp="Fuß oder Rolle"):
    """Tragende Teile aus dem Raster ableiten.

    Bei S Spalten und R Reihen gibt es R+1 Ebenen und S+1 Pfostenachsen, jede
    Achse vorne und hinten - daraus folgt alles Weitere.
    """
    ebenen = reihen + 1
    achsen = spalten + 1
    knoten = 2 * achsen * ebenen

    def _name(basis, mass):
        return f"{basis} {int(mass)}" if mass else basis

    positionen = [
        # Beide Gruppen heißen "Gerade" - ohne die Rolle wären sie in der Liste
        # nicht zu unterscheiden, erst recht wenn beide auf dasselbe Maß fallen.
        {"bauteil": _name("Gerade", breite), "rolle": "Breite",
         "anzahl": 2 * spalten * ebenen,
         "herkunft": "Raster", "hinweis": "waagerecht, Front- und Rückseite je Ebene"},
        {"bauteil": _name("Gerade", hoehe), "rolle": "Höhe",
         "anzahl": 2 * achsen * reihen,
         "herkunft": "Raster", "hinweis": "senkrechte Pfosten zwischen den Ebenen"},
        {"bauteil": _name("Mutternstab", tiefe), "anzahl": achsen * ebenen,
         "herkunft": "Raster", "hinweis": "Tiefenrichtung je Achse und Ebene"},
        {"bauteil": "Noppenscheiben", "anzahl": knoten,
         "herkunft": "Raster", "hinweis": "eine je Systemknoten"},
        {"bauteil": "Schraube", "anzahl": knoten,
         "herkunft": "Raster", "hinweis": "mindestens eine je Systemknoten"},
        {"bauteil": "Systemboden", "anzahl": spalten * ebenen,
         "herkunft": "Raster", "hinweis": "je Fach und Ebene, auch verdeckte"},
        {"bauteil": fusstyp, "anzahl": 2 * achsen,
         "herkunft": "Raster", "hinweis": "je Pfostenachse vorne und hinten"},
    ]
    return [p for p in positionen if p["anzahl"] > 0]


def _einbau_stueckliste(image_results, capture_type):
    """Einbauten je Ansicht zählen und über die Ansichten zusammenführen.

    Bei einer symmetrischen Aufnahme liegen nur Front und eine Seite vor. Dann
    gilt: die Rückseite trägt gleich viele Teile wie die Front, die zweite Seite
    gleich viele wie die erste.
    """
    je_ansicht = {}
    for df in image_results.values():
        if df is None or df.empty or "class" not in df.columns:
            continue
        ansicht = _ansicht(df)
        for _, zeile in df.iterrows():
            basis = _basisklasse(zeile["class"])
            zustaendig = EINBAU_EBENE.get(basis)
            if not zustaendig:
                continue
            # Nur aus der Ansicht zählen, die diese Ebene frontal zeigt.
            if ansicht in FRONT_VIEWS + SIDE_VIEWS and ansicht not in zustaendig:
                continue
            je_ansicht.setdefault(ansicht, {}).setdefault(basis, 0)
            je_ansicht[ansicht][basis] += 1

    gesehene_front = [a for a in je_ansicht if a in FRONT_VIEWS]
    gesehene_seiten = [a for a in je_ansicht if a in SIDE_VIEWS]

    zusammen, hinweise, annahmen = {}, {}, set()
    for ansicht, teile in je_ansicht.items():
        # Die fotografierte Seite zählt unverändert.
        for bauteil, anzahl in teile.items():
            zusammen[bauteil] = zusammen.get(bauteil, 0) + anzahl

        if capture_type != "symmetric":
            continue

        # Bei symmetrischer Aufnahme fehlt je eine Gegenseite - sie wird ergänzt.
        spiegeln = ((ansicht in FRONT_VIEWS and len(gesehene_front) == 1)
                    or (ansicht in SIDE_VIEWS and len(gesehene_seiten) == 1))
        if not spiegeln:
            continue

        for bauteil, anzahl in teile.items():
            if ansicht in FRONT_VIEWS:
                # Front -> Rückseite: Türen und Auszüge werden zu Verkleidungen.
                gegenstueck = RUECKSEITE_STATT.get(bauteil, bauteil)
                if gegenstueck is None:
                    continue
                if gegenstueck != bauteil:
                    annahmen.add(gegenstueck)
                    hinweise[gegenstueck] = (
                        "Rückseite angenommen: Front zeigt "
                        f"{anzahl}x {bauteil} – dort vermutlich Verkleidung ohne Griff"
                    )
            else:
                gegenstueck = bauteil
            zusammen[gegenstueck] = zusammen.get(gegenstueck, 0) + anzahl
            hinweise.setdefault(gegenstueck, "Gegenseite gespiegelt angenommen")

    positionen = []
    for bauteil, anzahl in sorted(zusammen.items()):
        positionen.append({
            "bauteil": bauteil,
            "anzahl": anzahl,
            "herkunft": "angenommen" if bauteil in annahmen else "gezählt",
            "hinweis": hinweise.get(bauteil, "je Ansicht gezählt, Ansichten summiert"),
        })
    return positionen


def erstelle_stueckliste(image_results, capture_type="single"):
    """Stückliste des gesamten Möbels über alle Ansichten.

    Gibt ``None`` zurück, wenn sich das Raster nicht bestimmen lässt.
    """
    if not image_results:
        return None

    # Raster aus der Front-/Rückansicht ableiten, sonst aus der ergiebigsten.
    kandidat, kandidat_raster = None, None
    for df in image_results.values():
        if df is None or df.empty:
            continue
        raster = erkenne_raster(df)
        if not raster:
            continue
        if _ansicht(df) in FRONT_VIEWS:
            kandidat, kandidat_raster = df, raster
            break
        if kandidat is None or len(df) > len(kandidat):
            kandidat, kandidat_raster = df, raster

    if not kandidat_raster:
        return None

    positionen = _struktur_stueckliste(
        kandidat_raster["spalten"], kandidat_raster["reihen"],
        kandidat_raster["systembreite"], kandidat_raster["systemhoehe"],
        kandidat_raster["systemtiefe"], _fusstyp(image_results),
    )
    einbauten = _einbau_stueckliste(image_results, capture_type)

    # Jedes Seitenfeld ist entweder ausgekreuzt oder verkleidet. Aus der Zahl
    # der Felder und der gezählten Diagonalen folgt die Zahl der Seitenteile -
    # auch die, die die Erkennung nicht gefunden hat.
    seitenfelder = 2 * kandidat_raster["reihen"]
    diagonalen = sum(p["anzahl"] for p in einbauten if p["bauteil"] == "Diagonale")
    erkannte_seitenteile = sum(p["anzahl"] for p in einbauten
                               if p["bauteil"].startswith("Seitenverkleidung"))
    offene_felder = seitenfelder - diagonalen - erkannte_seitenteile
    if diagonalen and offene_felder > 0:
        einbauten.append({
            "bauteil": "Seitenverkleidung",
            "anzahl": offene_felder,
            "herkunft": "abgeleitet",
            "hinweis": f"{seitenfelder} Seitenfelder minus {diagonalen} Diagonalen",
        })

    positionen += einbauten

    vollstaendig = capture_type == "asymmetric" and len(image_results) >= 4
    return {
        "raster": kandidat_raster,
        "positionen": positionen,
        "gesamt": sum(p["anzahl"] for p in positionen),
        "ansichten": sorted({_ansicht(df) for df in image_results.values() if df is not None}),
        "capture_type": capture_type,
        "vollstaendig": vollstaendig,
    }


def markiere_duplikate(image_results):
    """Markiert Erkennungen, die dasselbe Bauteil aus einer zweiten Ansicht zeigen.

    Beweisbar ist das für Einbauten: eine Seitendiagonale, die in der
    Frontaufnahme durch das offene Regal hindurch erscheint, ist dort eine
    Dublette. Für tragende Teile lässt sich nicht sagen, welche der Erkennungen
    dieselbe Strebe meint - deshalb zählt für den Bestand ohnehin die
    Sammelposition aus der Stückliste, nicht die einzelne Erkennung.
    """
    for df in image_results.values():
        if df is None or df.empty or "class" not in df.columns:
            continue
        ansicht = _ansicht(df)
        df["duplikat"] = False
        for index, zeile in df.iterrows():
            basis = _basisklasse(zeile["class"])
            zustaendig = EINBAU_EBENE.get(basis)
            if not zustaendig:
                continue
            if ansicht in FRONT_VIEWS + SIDE_VIEWS and ansicht not in zustaendig:
                df.at[index, "duplikat"] = True
    return image_results


def beschaedigte_je_bauteil(image_results):
    """Zählt beschädigte Bauteile je Klasse - Dubletten ausgenommen."""
    beschaedigt = {}
    for df in image_results.values():
        if df is None or df.empty or "class" not in df.columns:
            continue
        for _, zeile in df.iterrows():
            if zeile.get("duplikat"):
                continue
            zustand = zeile.get("zustand")
            if not zustand or zustand == "unbeschädigt":
                continue
            basis = _basisklasse(zeile["class"])
            beschaedigt[basis] = beschaedigt.get(basis, 0) + 1
    return beschaedigt
