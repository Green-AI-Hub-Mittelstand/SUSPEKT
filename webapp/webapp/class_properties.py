# Define class properties
CLASS_PROPERTIES = {
    "Gerade": {
        "gewicht": "150-300 g",
        "farbe": "edelstahl",
        "maße": "35x2 cm",
        "typ": "Strebe",
        "zustand": "unbeschädigt"
    },
    "Diagonale": {
        "gewicht": "160-320 g",
        "farbe": "edelstahl",
        "maße": "40x2 cm",
        "typ": "Strebe",
        "zustand": "unbeschädigt"
    },
    "Einzeltuer": {
        "gewicht": "500-650 g",
        "farbe": "undefined",
        "maße": "to be calculated",
        "typ": "Möbeltür",
        "zustand": "unbeschädigt"
    },
    "Doppeltuer": {
        "gewicht": "500-650 g",
        "farbe": "undefined",
        "maße": "to be calculated",
        "typ": "Möbeltür",
        "zustand": "unbeschädigt"
    },
    "Doppeltuerblatt": {
        "gewicht": "250 g",
        "farbe": "undefined",
        "maße": "to be calculated",
        "typ": "Möbeltür",
        "zustand": "unbeschädigt"
    },
    "Mutternstab": {
        "gewicht": "150-300 g",
        "farbe": "edelstahl",
        "maße": "35x2 cm",
        "typ": "Strebe",
        "zustand": "unbeschädigt"
    },
    "Noppenscheiben": {
        "gewicht": "10 g",
        "farbe": "edelstahl",
        "maße": "3x3 cm",
        "typ": "Möbelteil",
        "zustand": "unbeschädigt"
    },
    "Systemboden": {
        "gewicht": "500-650 g",
        "farbe": "weiss",
        "maße": "to be calculated",
        "typ": "Systemboden",
        "zustand": "unbeschädigt"
    },
    "Sockelfuss": {
        "gewicht": "250 g",
        "farbe": "edelstahl/schwarz",
        "maße": "10 cm",
        "typ": "Möbelfuß",
        "zustand": "unbeschädigt"
    },
    "Griff": {
        "gewicht": "170 g",
        "farbe": "edelstahl/schwarz",
        "maße": "10x5 cm",
        "typ": "Möbelteil",
        "zustand": "unbeschädigt"
    },
    "Verkleidung": {
        "gewicht": "450-650 g",
        "farbe": "to be detected",
        "maße": "35x35 cm",
        "typ": "Möbelteil",
        "zustand": "unbeschädigt"
    },
    "Seitenverkleidung-0-IN": {
        "gewicht": "450-650 g",
        "farbe": "to be detected",
        "maße": "35x35 cm",
        "typ": "Verkleidung",
        "zustand": "unbeschädigt"
    },
    "Seitenverkleidung-0-0": {
        "gewicht": "450-650 g",
        "farbe": "to be detected",
        "maße": "35x35 cm",
        "typ": "Verkleidung",
        "zustand": "unbeschädigt"
    },
    "Seitenverkleidung-IN-IN": {
        "gewicht": "450-650 g",
        "farbe": "to be detected",
        "maße": "35x35 cm",
        "typ": "Verkleidung",
        "zustand": "unbeschädigt"
    },
    "Schraube": {
        "gewicht": "10 g",
        "farbe": "to be detected",
        "maße": "5x30 mm",
        "typ": "Schraube",
        "zustand": "unbeschädigt"
    },
    "Auszug": {
        "gewicht": "1000 g",
        "farbe": "to be detected",
        "maße": "455x170 mm",
        "typ": "Auszug",
        "zustand": "unbeschädigt"
    },

    # --- Flächen und Möbelteile ohne festes Einzelmaß ---
    # Ihre Maße ergeben sich aus dem Rasterfeld, das sie ausfüllen, und werden
    # in measurement.py aus den umliegenden Streben abgeleitet.
    "Fachboden": {
        "gewicht": "Nicht verfügbar",
        "farbe": "to be detected",
        "maße": "to be calculated",
        "typ": "Fachboden",
        "zustand": "unbeschädigt"
    },
    "Fachboden mit Verstaerkung": {
        "gewicht": "Nicht verfügbar",
        "farbe": "to be detected",
        "maße": "to be calculated",
        "typ": "Fachboden",
        "zustand": "unbeschädigt"
    },
    "Kombifront": {
        "gewicht": "Nicht verfügbar",
        "farbe": "to be detected",
        "maße": "to be calculated",
        "typ": "Front",
        "zustand": "unbeschädigt"
    },
    "Magazin": {
        "gewicht": "Nicht verfügbar",
        "farbe": "to be detected",
        "maße": "to be calculated",
        "typ": "Magazin",
        "zustand": "unbeschädigt"
    },
    "Seitenverkleidung": {
        "gewicht": "450-650 g",
        "farbe": "to be detected",
        "maße": "to be calculated",
        "typ": "Verkleidung",
        "zustand": "unbeschädigt"
    },
    # Schreibweisen des zweiten Modells - gleiche Bauteile, andere Benennung.
    "Seitenverkleidung - 0/0": {
        "gewicht": "450-650 g",
        "farbe": "to be detected",
        "maße": "to be calculated",
        "typ": "Verkleidung",
        "zustand": "unbeschädigt"
    },
    "Seitenverkleidung - 0/IN": {
        "gewicht": "450-650 g",
        "farbe": "to be detected",
        "maße": "to be calculated",
        "typ": "Verkleidung",
        "zustand": "unbeschädigt"
    },
    "Seitenverkleidung - IN/IN": {
        "gewicht": "450-650 g",
        "farbe": "to be detected",
        "maße": "to be calculated",
        "typ": "Verkleidung",
        "zustand": "unbeschädigt"
    },
    "Dekor": {
        "gewicht": "Nicht verfügbar",
        "farbe": "to be detected",
        "maße": "to be calculated",
        "typ": "Oberfläche",
        "zustand": "unbeschädigt"
    },

    # --- Füße und Rollen (Maße aus dem Katalog, Seite 16/17) ---
    "Rolle": {
        "gewicht": "Nicht verfügbar",
        "farbe": "vernickelt",
        "maße": "75x120 mm",
        "typ": "Lenkrolle",
        "zustand": "unbeschädigt"
    },
    "Winkelfuss": {
        "gewicht": "Nicht verfügbar",
        "farbe": "vernickelt",
        "maße": "20 mm",
        "typ": "Möbelfuß",
        "zustand": "unbeschädigt"
    },
    "Verbindungswinkel": {
        "gewicht": "Nicht verfügbar",
        "farbe": "edelstahl",
        "maße": "Nicht verfügbar",
        "typ": "Verbinder",
        "zustand": "unbeschädigt"
    },

    # --- Ganze Möbel: kein Einzelbauteil, deshalb ohne Maß ---
    "Regal": {
        "gewicht": "Nicht verfügbar",
        "farbe": "to be detected",
        "maße": "Nicht verfügbar",
        "typ": "Möbel",
        "zustand": "unbeschädigt"
    },
    "Sideboard": {
        "gewicht": "Nicht verfügbar",
        "farbe": "to be detected",
        "maße": "Nicht verfügbar",
        "typ": "Möbel",
        "zustand": "unbeschädigt"
    },
    "Tisch": {
        "gewicht": "Nicht verfügbar",
        "farbe": "to be detected",
        "maße": "Nicht verfügbar",
        "typ": "Möbel",
        "zustand": "unbeschädigt"
    },
    "Tresen": {
        "gewicht": "Nicht verfügbar",
        "farbe": "to be detected",
        "maße": "Nicht verfügbar",
        "typ": "Möbel",
        "zustand": "unbeschädigt"
    },
    # Weitere Klassen können hier hinzugefügt werden
}
