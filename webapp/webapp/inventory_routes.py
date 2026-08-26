import ast
import json
import re

from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

#from .auth import get_current_user
from .neo4j_database import db
from .resource_efficiency import fetch_reusable_components, analyze_reusable_components
from .transport_emission import get_transport_emissions,get_total_distance

router = APIRouter(prefix="/inventory")
from .templating import templates

# Simulierte Nutzer-Datenbank
users_db = {
    "admin": {"username": "admin", "role": "admin"},
    "user": {"username": "user", "role": "user"}
}


def _parse_farbe(raw):
    """Parst das gespeicherte farbe-Feld, das je nach Schreibpfad als JSON-String,
    als Python-dict-Repr-String oder als reiner Farbname vorliegen kann."""
    if not raw:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("{"):
            try:
                return json.loads(s)
            except (json.JSONDecodeError, ValueError):
                try:
                    return ast.literal_eval(s)
                except (ValueError, SyntaxError):
                    return None
        return {"erkannte_farbe": s}
    return None


# Manche Klassennamen sind aus den YOLO-Trainingsdaten ASCII-safe ohne Umlaute
# benannt (z.B. "Doppeltuer"). Für die Anzeige im Lager auf die korrekte
# deutsche Schreibweise abbilden, ohne den zugrunde liegenden Klassennamen
# (der z.B. beim Erkennungs-Matching verwendet wird) zu verändern.
CLASS_DISPLAY_NAMES = {
    "Einzeltuer": "Einzeltür",
    "Doppeltuer": "Doppeltür",
    "Doppeltuerblatt": "Doppeltürblatt",
    "Sockelfuss": "Sockelfuß",
}

_TRAILING_NUMBER_RE = re.compile(r"^(.*?)(?:\s+(\d+))?$")


def _display_class_name(class_name):
    return CLASS_DISPLAY_NAMES.get(class_name, class_name)


def _class_sort_key(display_name):
    """Sortiert Klassen alphabetisch, gruppiert dabei aber Größenvarianten
    (z.B. "Gerade", "Gerade 180", "Gerade 270", "Gerade 540") numerisch statt
    lexikografisch hinter ihrem gemeinsamen Namen."""
    match = _TRAILING_NUMBER_RE.match(display_name.strip())
    prefix = match.group(1).strip().lower() if match else display_name.lower()
    number = int(match.group(2)) if match and match.group(2) else -1
    return (prefix, number, display_name.lower())


def _is_reusable(value):
    """reusable liegt im Regelfall als natives Neo4j-Boolean vor, kann bei
    älteren/manuell eingespielten Daten aber auch als String vorliegen."""
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "")
    return bool(value)




# Referenzmöbel für den Produktpass: das Regal aus den Beispielaufnahmen.
# Die Stückliste ist die von System 180 gezählte Soll-Liste - sie zeigt, was ein
# vollständiger Pass enthält, ohne von der Erkennungsqualität abzuhängen.
REFERENZ_MOEBEL = {
    "bezeichnung": "Regal, 1 Feld x 5 Ebenen",
    "bild": "/static/samples/Regal Front.jpg",
    "systembreite": 450,
    "systemhoehe": 360,
    "ebenen": 6,
    "oberflaeche": "Piniengrün / Holzoptik",
    "nutzungszyklus": 2,
    # Je Position: wie viele Teile direkt weiterverwendet wurden und was mit den
    # übrigen geschah. "verbleib" folgt den R-Strategien der Kreislaufwirtschaft -
    # nicht jedes Teil geht direkt zurück ins Lager, aber keines fällt hinten
    # einfach heraus.
    "stueckliste": [
        {"bauteil": "Gerade 450", "anzahl": 12, "hinweis": "waagerecht, Front und Rückseite",
         "wiederverwendet": 12},
        {"bauteil": "Gerade 360", "anzahl": 20, "hinweis": "senkrechte Pfosten",
         "wiederverwendet": 20},
        {"bauteil": "Mutternstab", "anzahl": 12, "hinweis": "Tiefenrichtung",
         "wiederverwendet": 12},
        {"bauteil": "Diagonale", "anzahl": 8, "hinweis": "Seitenaussteifung",
         "wiederverwendet": 8},
        {"bauteil": "Noppenscheiben", "anzahl": 24, "hinweis": "eine je Systemknoten",
         "wiederverwendet": 22,
         "verbleib": {
             "anzahl": 2,
             "strategie": "R8",
             "strategie_name": "Recycling",
             "text": "Zwei Scheiben waren durch wiederholtes Anziehen verformt und "
                     "hielten die Klemmkraft nicht mehr. Sie gingen als sortenreiner "
                     "Edelstahl in die Wiederverwertung - ohne Qualitätsverlust, weil "
                     "Edelstahl beliebig oft eingeschmolzen werden kann.",
         }},
        {"bauteil": "Schraube M8x50", "anzahl": 24, "hinweis": "mindestens eine je Knoten",
         "wiederverwendet": 24},
        {"bauteil": "Systemboden (Piniengrün)", "anzahl": 6,
         "hinweis": "einer davon hinter der Tür verdeckt",
         "wiederverwendet": 5,
         "verbleib": {
             "anzahl": 1,
             "strategie": "R8",
             "strategie_name": "Recycling",
             "text": "Ein Boden hatte einen MDF-Platzer an der Kante. Eine Reparatur "
                     "hätte die Melaminbeschichtung nicht wiederhergestellt, deshalb "
                     "wurde die Platte zerlegt: Trägermaterial in die MDF-Verwertung, "
                     "die Edelstahl-Verstärkungsschiene zurück ins Lager.",
         }},
        {"bauteil": "Rückverkleidung (Piniengrün)", "anzahl": 5, "hinweis": "je Ebene",
         "wiederverwendet": 5},
        {"bauteil": "Seitenverkleidung (Holzoptik)", "anzahl": 2, "hinweis": "links und rechts",
         "wiederverwendet": 2},
        {"bauteil": "Tür (Holzoptik)", "anzahl": 1, "hinweis": "Front",
         "wiederverwendet": 1,
         "verbleib": {
             "anzahl": 1,
             "strategie": "R5",
             "strategie_name": "Aufarbeitung",
             "aufgearbeitet": True,
             "text": "Die Tür zeigte Gebrauchsspuren am Griffbereich. Statt sie zu "
                     "ersetzen, wurde die Oberfläche nachbehandelt und das Scharnier "
                     "neu justiert - sie ist wieder im Einsatz.",
         }},
        {"bauteil": "Rolle 75 mm", "anzahl": 4, "hinweis": "je Pfostenachse",
         "wiederverwendet": 4},
    ],
}


def _beispiel_produktpass():
    """Digitaler Produktpass des Referenzmöbels.

    Die Stückliste ist fest hinterlegt (gezählte Soll-Liste), die Herkunfts-
    angaben kommen soweit vorhanden aus der Datenbank. So zeigt der Pass ein
    vollständiges Möbel, auch wenn die Erkennung einzelne Teile übersieht.
    """
    pass_daten = dict(REFERENZ_MOEBEL)
    positionen = REFERENZ_MOEBEL["stueckliste"]
    gesamt = sum(p["anzahl"] for p in positionen)
    wiederverwendet = sum(p.get("wiederverwendet", p["anzahl"]) for p in positionen)
    aufgearbeitet = sum(p["verbleib"]["anzahl"] for p in positionen
                        if p.get("verbleib", {}).get("aufgearbeitet"))
    verwertet = sum(p["verbleib"]["anzahl"] for p in positionen
                    if p.get("verbleib") and not p["verbleib"].get("aufgearbeitet"))

    pass_daten["teile_gesamt"] = gesamt
    pass_daten["positionen"] = len(positionen)
    pass_daten["wiederverwendet"] = wiederverwendet
    pass_daten["aufgearbeitet"] = aufgearbeitet
    pass_daten["verwertet"] = verwertet
    pass_daten["quote"] = round(wiederverwendet / gesamt * 100, 1) if gesamt else 0

    # Herkunft aus dem jüngsten erfassten Auftrag ergänzen, falls vorhanden.
    query = """
    MATCH (o:Order)
    OPTIONAL MATCH (o)-[:LOCATED_AT]->(loc:Location)
    RETURN o.order_id AS auftrag, o.order_type AS auftragsart,
           o.process_id AS vorgang,
           coalesce(loc.formatted_address, o.location) AS standort
    ORDER BY o.process_id DESC LIMIT 1
    """
    try:
        treffer = db.run_query(query)
    except Exception:
        treffer = []

    if treffer:
        pass_daten.update(treffer[0])
        vorgang = str(pass_daten.get("vorgang") or "")
        if len(vorgang) >= 8 and vorgang[-8:].isdigit():
            d = vorgang[-8:]
            pass_daten["erfasst_am"] = f"{d[6:8]}.{d[4:6]}.{d[0:4]}"

    return pass_daten


@router.get("", response_class=HTMLResponse)
async def get_inventory(request: Request):
    """Stellt das digitale Lager als gruppierten Bauteil-Katalog dar: pro Bauteil-Klasse
    eine Karte mit Beispielbild, verfügbaren Farben und Bestand (statt einer rohen
    Auflistung jeder einzelnen erfassten Komponente)."""
    try:
        # Nur freigegebene Bauteile: erfasste Aufträge durchlaufen erst die
        # Prüfung durch System 180 (siehe /freigabe) und erscheinen danach hier.
        #
        # Gezählt werden ausschließlich Sammelpositionen mit ihrer Menge. Die
        # einzelnen Erkennungen (:Erkennung) sind Belege und würden dasselbe
        # Bauteil aus mehreren Ansichten mehrfach zählen.
        query = """
        MATCH (c:Component)
        WHERE c.confirmed = true
        RETURN c.class AS class, c.typ AS typ, c.farbe AS farbe,
               c.zustand AS zustand, c.reusable AS reusable,
               coalesce(c.anzahl, 1) AS anzahl,
               coalesce(c.anzahl_beschaedigt, 0) AS anzahl_beschaedigt,
               c.anzahl_wiederverwendbar AS anzahl_wiederverwendbar
        """
        components = db.run_query(query)

        groups = {}
        for comp in components:
            class_name = comp.get("class") or "Unbekannt"
            group = groups.setdefault(class_name, {
                "class": class_name,
                "display_class": _display_class_name(class_name),
                "typ": comp.get("typ") or "Sonstige",
                "total_count": 0,
                "reusable_count": 0,
                "colors": {},
            })

            menge = int(comp.get("anzahl") or 1)
            group["total_count"] += menge

            # Sammelpositionen führen den wiederverwendbaren Anteil selbst mit.
            # Ältere Einzelknoten haben das Feld nicht - dort entscheidet der
            # Zustand über das eine Bauteil.
            verwendbar = comp.get("anzahl_wiederverwendbar")
            if verwendbar is None:
                verwendbar = menge if _is_reusable(comp.get("reusable")) else 0
            group["reusable_count"] += int(verwendbar)

            farbe = _parse_farbe(comp.get("farbe"))
            if farbe and farbe.get("hex_code"):
                group["colors"].setdefault(farbe["hex_code"], farbe.get("erkannte_farbe") or "")


        catalog = sorted(
            groups.values(),
            key=lambda g: _class_sort_key(g["display_class"])
        )
        for group in catalog:
            group["colors"] = [
                {"hex": hex_code, "label": label}
                for hex_code, label in group["colors"].items()
            ]

        return templates.TemplateResponse("inventory.html", {
            "request": request,
            "catalog": catalog,
            "total_reusable": sum(g["reusable_count"] for g in catalog),
            "total_all": sum(g["total_count"] for g in catalog),
            "reusable_class_count": sum(1 for g in catalog if g["reusable_count"] > 0),
            "total_class_count": len(catalog),
            "produktpass": _beispiel_produktpass(),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

"""
@router.put("/components/update/{component_id}")
async def update_component(
    component_id: str, data: dict, user: dict = Depends(get_current_user)
):
    #Nur Admins können Komponenten bearbeiten
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Keine Berechtigung")

    # Hier sollte die Neo4j-Query stehen
    return JSONResponse(content={"message": "Komponente aktualisiert"})


@router.delete("/components/delete/{component_id}")
async def delete_component(component_id: str, user: dict = Depends(get_current_user)):
    #Nur Admins können Komponenten löschen
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Keine Berechtigung")

    # Hier sollte die Neo4j-Query stehen
    # await broadcast_update('{"action": "delete", "id": "' + component_id + '"}')
    return JSONResponse(content={"message": "Komponente gelöscht"})
"""
