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


class OrderAnalysis(BaseModel):
    order_id: str
    distance_km: float  # Entfernung zum Kunden


def fetch_reusable_components():
    """ Holt alle Komponenten mit reuseable=true aus Neo4j """
    query = """
    MATCH (c:Component)
    WHERE c.reusable = true
    RETURN c.class AS component_class, c.laenge AS length, 
           c.typ AS typ, c.zustand AS status, c.farbe AS color
    """
    result = db.run_query(query)


    return result



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


def analyze_reusable_components():
    components = fetch_reusable_components()
    total_length, reusable_length = 0, 0
    total_weight_saved = 0  # Gesamte Gewichtseinsparung in kg
    co2_new, co2_reuse = 0, 0

    for component in components:
        # Sicherstellen, dass length existiert und numerisch ist
        try:
            length = float(component["length"]) if component["length"] and str(component["length"]).lower() not in ["none", "nan", ""] else 0
        except ValueError:
            length = 0

        if length > 0:
            total_length += length
        #else:
        #    print(f"⚠️ Warnung: Ungültige Länge für Komponente {component['component_class']}: {component['length']}")

        # Prüfen, ob `color` vorhanden ist
        if "color" in component and component["color"]:
            color = component["color"].lower()
        else:
            color = ""

        # Prüfen, ob die Komponente Edelstahl ist
        if "edelstahl" in color:
            diameter = 0.02  # Meter (20 mm)
            radius = diameter / 2

            # Volumen des Edelstahl-Zylinders berechnen
            volume = math.pi * (radius ** 2) * (length / 1000)  # Länge in Meter umwandeln
            weight = volume * DENSITY_STEEL  # Masse in kg

            # Prüfen, ob der `status` unbeschädigt ist
            if "status" in component and component["status"] == "unbeschädigt":
                reusable_length += length
                total_weight_saved += weight  # Gewicht der wiederverwendbaren Teile
            else:
                material_factor = EMISSIONS_FACTOR_STEEL
                co2_new += weight * material_factor  # Neue CO2-Emission
        #else:
        #    print(f"⚠️ Warnung: Keine Edelstahl-Farbe erkannt für {component['component_class']}")

    # CO₂ Einsparung durch Wiederverwendung
    if total_length > 0:
        co2_reuse = (total_length - reusable_length) * EMISSIONS_FACTOR_STEEL
        co2_savings_production = co2_new - co2_reuse
        material_savings = (reusable_length / total_length) * 100
    else:
        co2_savings_production = 0
        material_savings = 0

    #print(f"📊 Gesamtlänge: {total_length} mm | Wiederverwendbare Länge: {reusable_length} mm")
    #print(f"📊 Gewichtseinsparung: {total_weight_saved} kg | CO₂ Einsparung: {co2_savings_production} kg")

    return {
        "total_length": total_length,
        "reusable_length": reusable_length,
        "material_savings": round(material_savings, 2),
        "total_weight_saved_kg": round(total_weight_saved, 2),
        "co2_savings_production": round(co2_savings_production, 2),
        "components": components
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
