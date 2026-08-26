"""Freigabe erfasster Aufträge durch System-180-Mitarbeitende.

Erfasste Bauteile werden zunächst mit ``confirmed = false`` gespeichert und sind
damit noch nicht im digitalen Lager sichtbar. Auf dieser Seite prüft eine
berechtigte Person den Auftrag, korrigiert oder entfernt einzelne Bauteile und
gibt ihn anschließend frei.
"""
import ast
import json
from datetime import datetime

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .auth import is_admin
from .class_properties import CLASS_PROPERTIES
from .measurement import (
    ALL_STRAIGHT_LENGTHS,
    DIAGONAL_LENGTHS,
    FLAECHEN_OPTIONEN,
    SYSTEM_DEPTHS,
    TUBE_DIAMETER_MM,
)
from .neo4jIntegration import Neo4jDatabase

router = APIRouter(prefix="/freigabe")
from .templating import templates

db = Neo4jDatabase()

ZUSTAND_OPTIONEN = ["unbeschädigt", "MDF-Platzer", "Rohr_Kratzer", "Delle"]
BESCHAEDIGTE_ZUSTAENDE = ["MDF-Platzer", "Rohr_Kratzer", "Delle"]
BAUTEIL_OPTIONEN = sorted(set(CLASS_PROPERTIES) | {"Rolle"})
LAENGEN_OPTIONEN = {
    "Gerade": ALL_STRAIGHT_LENGTHS,
    "Mutternstab": SYSTEM_DEPTHS,
    "Diagonale": DIAGONAL_LENGTHS,
}


def _lade_farboptionen():
    """Unidekore und Beschichtungen - dieselbe Auswahl wie bei der Erfassung."""
    optionen = {}
    for datei in ("colors.json", "beschichtung.json"):
        try:
            with open(datei, encoding="utf-8") as f:
                optionen.update(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    return dict(sorted(optionen.items()))


def _parse_farbe(raw):
    """Das Farbfeld liegt je nach Schreibpfad als JSON, als dict-Repr oder als
    reiner Name vor."""
    if not raw:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        wert = raw.strip()
        if wert.startswith("{"):
            try:
                return json.loads(wert)
            except (json.JSONDecodeError, ValueError):
                try:
                    return ast.literal_eval(wert)
                except (ValueError, SyntaxError):
                    return None
        return {"erkannte_farbe": wert}
    return None


def _zeitpunkt(ms):
    """Neo4js ``timestamp()`` liefert Millisekunden - hier als lesbares Datum."""
    if not ms:
        return None
    try:
        return datetime.fromtimestamp(int(ms) / 1000).strftime("%d.%m.%Y %H:%M")
    except (TypeError, ValueError, OSError):
        return None


def _bauteile_aufbereiten(bauteile):
    """Farbfeld entpacken und die Basisklasse fuer die Auswahlliste ableiten."""
    for bauteil in bauteile:
        bauteil["farbe"] = _parse_farbe(bauteil.get("farbe"))
        bauteil["basisklasse"] = _basisklasse(bauteil.get("class"))
    return bauteile


def _basisklasse(klassenname):
    """"Gerade 540" -> "Gerade" - Flächen tragen kein Maß im Namen."""
    if not klassenname:
        return ""
    kopf = klassenname.split(" ")[0]
    return kopf if kopf in BAUTEIL_OPTIONEN else klassenname


@router.get("", response_class=HTMLResponse)
async def freigabe_uebersicht(request: Request):
    """Offene Aufträge zur Prüfung, darunter das Archiv aller Aufträge.

    Im Archiv lassen sich bereits freigegebene Aufträge nachträglich
    korrigieren oder ganz löschen.
    """
    if not is_admin(request):
        return RedirectResponse("/login", status_code=302)

    try:
        auftraege = db.get_pending_orders()
        alle_auftraege = db.get_all_orders()
    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "message": f"Aufträge konnten nicht geladen werden: {e}",
        }, status_code=500)

    for auftrag in auftraege:
        _bauteile_aufbereiten(auftrag.get("komponenten", []))

    # Das Archiv zeigt jeden Auftrag - bearbeitbar sind dort aber nur die
    # bereits freigegebenen Bauteile. Noch offene stehen oben in der Pruefung
    # und wuerden hier sonst doppelt zum Bearbeiten angeboten.
    for auftrag in alle_auftraege:
        bauteile = _bauteile_aufbereiten(auftrag.get("komponenten", []))
        auftrag["freigegebene_bauteile"] = [b for b in bauteile if b.get("confirmed")]
        auftrag["freigegeben_am_text"] = _zeitpunkt(auftrag.get("freigegeben_am"))

    return templates.TemplateResponse("approval.html", {
        "request": request,
        "auftraege": auftraege,
        "alle_auftraege": alle_auftraege,
        "bauteil_optionen": BAUTEIL_OPTIONEN,
        "zustand_optionen": ZUSTAND_OPTIONEN,
        "beschaedigte_zustaende": BESCHAEDIGTE_ZUSTAENDE,
        "laengen_optionen": LAENGEN_OPTIONEN,
        "flaechen_optionen": FLAECHEN_OPTIONEN,
        "farb_optionen": _lade_farboptionen(),
    })


@router.post("/bauteil/aktualisieren")
async def bauteil_aktualisieren(
    request: Request,
    comp_id: str = Form(...),
    bauteil: str = Form(...),
    zustand: str = Form(...),
    reusable: str = Form("false"),
    laenge: str = Form(""),
    breite: str = Form(""),
    farbe: str = Form(""),
):
    """Ein einzelnes Bauteil vor der Freigabe korrigieren.

    Wie in der Erfassung gilt: jedes Feld wird gegen eine feste Auswahlliste
    geprüft, damit kein freier Text in die Datenbank gelangt.
    """
    if not is_admin(request):
        return JSONResponse(status_code=403, content={
            "success": False, "message": "Keine Berechtigung."})

    if bauteil not in BAUTEIL_OPTIONEN:
        return JSONResponse(status_code=400, content={
            "success": False, "message": f"Unbekanntes Bauteil: {bauteil}"})
    if zustand not in ZUSTAND_OPTIONEN:
        return JSONResponse(status_code=400, content={
            "success": False, "message": f"Unbekannter Zustand: {zustand}"})

    flaeche = FLAECHEN_OPTIONEN.get(bauteil)

    def _pruefe(rohwert, erlaubt, bezeichnung):
        if not rohwert:
            return None, None
        try:
            wert = int(rohwert)
        except ValueError:
            return None, f"{bezeichnung} muss eine Zahl sein."
        if wert not in erlaubt:
            return None, f"{bezeichnung} {wert} ist für {bauteil} nicht im Systemraster."
        return wert, None

    if flaeche:
        laenge_wert, fehler = _pruefe(laenge, flaeche["laenge"], flaeche["laenge_label"])
        if fehler:
            return JSONResponse(status_code=400, content={"success": False, "message": fehler})
        breite_wert, fehler = _pruefe(breite, flaeche["breite"], flaeche["breite_label"])
        if fehler:
            return JSONResponse(status_code=400, content={"success": False, "message": fehler})
    else:
        laenge_wert, fehler = _pruefe(laenge, LAENGEN_OPTIONEN.get(bauteil, []), "Länge")
        if fehler:
            return JSONResponse(status_code=400, content={"success": False, "message": fehler})
        breite_wert = TUBE_DIAMETER_MM if laenge_wert else None

    farb_daten = None
    if farbe:
        farboptionen = _lade_farboptionen()
        if farbe not in farboptionen:
            return JSONResponse(status_code=400, content={
                "success": False, "message": f"Unbekannte Farbe: {farbe}"})
        eintrag = farboptionen[farbe]
        farb_daten = {
            "erkannte_farbe": farbe,
            "hex_code": eintrag.get("hex_code", "#CCCCCC"),
            "ncs_code": eintrag.get("ncs_code", "N/A"),
            "confidence": 1.0,
        }

    ist_beschaedigt = zustand in BESCHAEDIGTE_ZUSTAENDE
    # Unbeschädigte Bauteile sind immer wiederverwendbar.
    ist_wiederverwendbar = (reusable == "true") if ist_beschaedigt else True

    eigenschaften_klasse = CLASS_PROPERTIES.get(bauteil, {})
    anzeigename = f"{bauteil} {laenge_wert}" if (laenge_wert and not flaeche) else bauteil

    eigenschaften = {
        "class": anzeigename,
        "zustand": zustand,
        "reusable": ist_wiederverwendbar,
        "gewicht": eigenschaften_klasse.get("gewicht", "Nicht verfügbar"),
        "typ": eigenschaften_klasse.get("typ", "Unbekannt"),
        "breite": breite_wert,
        "laenge": laenge_wert,
        "geprueft_von": "system180",
    }
    if farb_daten:
        # Gleiche Serialisierung wie beim Speichern der Erfassung.
        eigenschaften["farbe"] = json.dumps(farb_daten)

    try:
        if not db.update_component(comp_id, eigenschaften):
            return JSONResponse(status_code=404, content={
                "success": False, "message": "Bauteil nicht gefunden."})
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False, "message": f"Fehler: {e}"})

    return JSONResponse(status_code=200, content={
        "success": True,
        "message": "Bauteil aktualisiert.",
        "bauteil": anzeigename,
        "gewicht": eigenschaften["gewicht"],
        "typ": eigenschaften["typ"],
        "reusable": ist_wiederverwendbar,
        "breite": breite_wert,
        "laenge": laenge_wert,
        "farbe": farb_daten,
    })


@router.post("/bauteil/entfernen")
async def bauteil_entfernen(request: Request, comp_id: str = Form(...)):
    """Ein Bauteil aus dem Auftrag entfernen (z.B. Fehlerkennung)."""
    if not is_admin(request):
        return JSONResponse(status_code=403, content={
            "success": False, "message": "Keine Berechtigung."})
    try:
        db.delete_component(comp_id)
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False, "message": f"Fehler: {e}"})
    return JSONResponse(status_code=200, content={
        "success": True, "message": "Bauteil entfernt."})


@router.post("/auftrag/freigeben")
async def auftrag_freigeben(request: Request, order_id: str = Form(...)):
    """Auftrag freigeben - seine Bauteile erscheinen danach im digitalen Lager."""
    if not is_admin(request):
        return JSONResponse(status_code=403, content={
            "success": False, "message": "Keine Berechtigung."})
    try:
        db.confirm_order(order_id)
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False, "message": f"Fehler: {e}"})
    return JSONResponse(status_code=200, content={
        "success": True, "message": "Auftrag freigegeben."})


@router.post("/auftrag/verwerfen")
async def auftrag_verwerfen(request: Request, order_id: str = Form(...)):
    """Auftrag samt Bauteilen verwerfen."""
    if not is_admin(request):
        return JSONResponse(status_code=403, content={
            "success": False, "message": "Keine Berechtigung."})
    try:
        db.delete_order(order_id)
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False, "message": f"Fehler: {e}"})
    return JSONResponse(status_code=200, content={
        "success": True, "message": "Auftrag verworfen."})
