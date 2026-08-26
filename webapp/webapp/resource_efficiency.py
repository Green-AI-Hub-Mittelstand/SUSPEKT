from fastapi import APIRouter
from neo4j import GraphDatabase
from pydantic import BaseModel
from typing import List
from .config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

from .neo4j_database import db

import math

router = APIRouter()

# Neo4j-Verbindung
#driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Emissionsfaktoren (kg CO₂ pro kg Material)
EMISSIONS_FACTOR_STEEL = 7.0  # Edelstahl
EMISSIONS_FACTOR_MDF = 0.6    # MDF-Platten
TRANSPORT_EMISSIONS_FACTOR = 0.14  # kg CO₂ pro km (PKW)
DENSITY_STEEL = 7850  # kg/m³
TUBE_DIAMETER_M = 0.02  # Systemrohr 20 mm


class OrderAnalysis(BaseModel):
    order_id: str
    distance_km: float  # Entfernung zum Kunden


def fetch_reusable_components():
    """Holt die freigegebenen Bauteile mit ihrer Menge aus Neo4j.

    Zwei Datenformate liegen nebeneinander vor:

    * Sammelpositionen der Stückliste tragen ``anzahl`` und
      ``anzahl_wiederverwendbar`` - eine Zeile steht für mehrere Bauteile.
    * Ältere Einzelknoten haben diese Felder nicht; dort entscheidet
      ``reusable`` über das eine Bauteil.

    Nur freigegebene Aufträge zählen, damit die Auswertung zum digitalen Lager
    passt. ``:Erkennung``-Knoten sind ein eigenes Label und damit ohnehin außen
    vor - sie würden dasselbe Bauteil aus mehreren Ansichten doppelt zählen.
    """
    query = """
    MATCH (c:Component)
    WHERE c.confirmed = true
    RETURN c.class AS component_class, c.laenge AS length,
           c.typ AS typ, c.zustand AS status, c.farbe AS color,
           coalesce(c.anzahl, 1) AS anzahl,
           c.anzahl_wiederverwendbar AS anzahl_wiederverwendbar,
           c.reusable AS reusable
    """
    return db.run_query(query)



def fetch_order_components(order_id: str):
    """ Holt alle Bauteile eines Auftrags aus Neo4j """
    query = """
    MATCH (o:Order {order_id: $order_id})<-[:PART_OF]-(c:Component)
    RETURN c.class AS component_class, c.gewicht AS weight, 
           c.typ AS typ, c.zustand AS status
    """
    with db.session() as session:
        result = session.run(query, order_id=order_id)
        return [record.data() for record in result]


def fetch_relevant_components():
    """ Holt alle Komponenten mit Farbe Edelstahl oder Schwarz Pulverbeschichtet aus Neo4j """
    query = """
    MATCH (c:Component)
    WHERE c.farbe CONTAINS 'Edelstahl' OR c.farbe CONTAINS 'Schwarz Pulverbeschichtet'
    RETURN c.class AS component_class, c.laenge AS length, 
           c.typ AS typ, c.zustand AS status
    """
    with db.session() as session:
        result = session.run(query)
        return [record for record in result]


def _zahl(wert):
    """Robust in eine Zahl wandeln - die Felder kommen teils als Text."""
    try:
        if wert is None or str(wert).strip().lower() in ("", "none", "nan"):
            return 0.0
        return float(wert)
    except (TypeError, ValueError):
        return 0.0


def analyze_reusable_components():
    """Wertet den freigegebenen Bestand aus.

    Die CO2-Ersparnis ist die Menge, die durch Wiederverwendung nicht neu
    produziert werden muss: Masse der wiederverwendbaren Teile mal
    Emissionsfaktor. Eine Rechnung über Längen mal Emissionsfaktor wäre
    dimensional falsch - der Faktor bezieht sich auf Kilogramm, nicht auf
    Millimeter.
    """
    components = fetch_reusable_components()

    total_length = reusable_length = 0.0
    total_weight = total_weight_saved = 0.0
    stueck_gesamt = stueck_verwendbar = 0

    for component in components:
        menge = int(_zahl(component.get("anzahl")) or 1)

        # Sammelpositionen fuehren den verwendbaren Anteil selbst mit,
        # Einzelknoten entscheiden ueber reusable.
        verwendbar = component.get("anzahl_wiederverwendbar")
        if verwendbar is None:
            verwendbar = menge if component.get("reusable") else 0
        verwendbar = max(0, min(int(_zahl(verwendbar)), menge))

        stueck_gesamt += menge
        stueck_verwendbar += verwendbar

        laenge = _zahl(component.get("length"))
        if laenge <= 0:
            continue
        total_length += laenge * menge
        reusable_length += laenge * verwendbar

        farbe = str(component.get("color") or "").lower()
        if "edelstahl" not in farbe:
            continue

        # Systemrohr 20x1: Masse ueber den Zylinder aus Aussendurchmesser.
        radius = TUBE_DIAMETER_M / 2
        masse_je_stueck = math.pi * (radius ** 2) * (laenge / 1000) * DENSITY_STEEL
        total_weight += masse_je_stueck * menge
        total_weight_saved += masse_je_stueck * verwendbar

    co2_savings_production = total_weight_saved * EMISSIONS_FACTOR_STEEL
    material_savings = (reusable_length / total_length * 100) if total_length else 0

    return {
        "total_length_m": round(total_length / 1000, 2),
        "reusable_length_m": round(reusable_length / 1000, 2),
        "material_savings": round(material_savings, 1),
        "total_weight_kg": round(total_weight, 1),
        "total_weight_saved_kg": round(total_weight_saved, 1),
        "co2_savings_production": round(co2_savings_production, 1),
        "stueck_gesamt": stueck_gesamt,
        "stueck_verwendbar": stueck_verwendbar,
        # Rueckwaertskompatibel fuer aeltere Aufrufer/Templates.
        "total_length": round(total_length, 1),
        "reusable_length": round(reusable_length, 1),
        "components": components,
    }


@router.post("/re_order")
def analyze_order(order: OrderAnalysis):
    components = fetch_order_components(order.order_id)

    total_weight, reusable_weight = 0, 0
    co2_new, co2_reuse = 0, 0

    for component in components:
        weight = float(component["weight"].replace(" g", "")) / 1000  # g → kg
        total_weight += weight

        if component["status"] == "unbeschädigt":
            reusable_weight += weight
        else:
            material_factor = EMISSIONS_FACTOR_STEEL if "stahl" in component["material"].lower() else EMISSIONS_FACTOR_MDF
            co2_new += weight * material_factor

    # CO₂ Einsparung durch Wiederverwendung
    co2_reuse = (total_weight - reusable_weight) * (EMISSIONS_FACTOR_STEEL if "stahl" in component["material"].lower() else EMISSIONS_FACTOR_MDF)
    co2_savings_production = co2_new - co2_reuse

    # Transport-Emissionen (wenn physischer Besuch entfällt)
    co2_savings_transport = order.distance_km * 2 * TRANSPORT_EMISSIONS_FACTOR

    return {
        "total_weight": total_weight,
        "reusable_weight": reusable_weight,
        "material_savings": (reusable_weight / total_weight) * 100,
        "co2_savings_production": co2_savings_production,
        "co2_savings_transport": co2_savings_transport
    }
