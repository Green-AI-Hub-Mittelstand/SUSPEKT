"""
Transportemissionen-Modul für System 180 Ressourceneffizienz-Dashboard.
Berechnet und analysiert CO2-Emissionen und Einsparungen für verschiedene Transportmittel.
"""
from fastapi import FastAPI, Request, APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

import json
import requests
from typing import Dict, List, Any, Optional, Tuple

#from .auth import get_current_user
from .neo4j_database import db
from .resource_efficiency import fetch_reusable_components, analyze_reusable_components
from .config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, HERE_API_KEY

# Router-Setup
router = APIRouter(prefix="/resource")
templates = Jinja2Templates(directory="templates")

# Konstanten für Emissionsberechnungen
EMISSIONS_FACTORS = {
    "diesel": 2.7,  # kg CO2 pro Liter
    "benzin": 2.3,  # kg CO2 pro Liter
    "schiff": 0.015,  # kg CO2 pro km pro Tonne
    "flugzeug": 0.25  # kg CO2 pro km
}

FUEL_CONSUMPTION = {
    "lkw": 30.0,  # l/100km
    "pkw": 7.5  # l/100km
}


TRANSPORT_VERTEILUNG = {
    "lkw": 60,  # 60% der Transportstrecke mit LKW
    "pkw": 30,  # 30% mit PKW
    "schiff": 8,  # 8% mit Schiff
    "flugzeug": 2  # 2% mit Flugzeug
}


# Angenommene Einsparung durch optimierte Logistik
EINSPARUNG_PROZENT = 20

# Standardwerte für Berlin als Ausgangspunkt
BERLIN_KOORDINATEN = {
    "lat": 52.43294690368163,
    "lon": 13.540966368802218
}


def get_route_distance(end_lat: float, end_lon: float,
                       start_lat: float = BERLIN_KOORDINATEN["lat"],
                       start_lon: float = BERLIN_KOORDINATEN["lon"]) -> float:
    """
    Berechnet die Fahrstrecke von Berlin zu einer Zielkoordinate mit HERE API.

    Args:
        end_lat: Zielbreite
        end_lon: Ziellänge
        start_lat: Startbreite (Standard: Berlin)
        start_lon: Startlänge (Standard: Berlin)

    Returns:
        float: Distanz in Kilometern
    """
    url = f"https://router.hereapi.com/v8/routes?transportMode=car&origin={start_lat},{start_lon}&destination={end_lat},{end_lon}&return=summary&apiKey={HERE_API_KEY}"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Prüfen auf HTTP-Fehler
        data = response.json()

        if "routes" in data:
            distanz_m = data["routes"][0]["sections"][0]["summary"]["length"]
            return distanz_m / 1000  # Umwandlung in km
        return 0
    except Exception as e:
        # Logging oder Fehlerbehandlung hinzufügen
        print(f"Fehler bei der HERE API-Anfrage: {e}")
        return 0


def get_total_distance() -> Dict[str, float]:
    """
    Berechnet die gesamte Fahrstrecke aller Bestellungen von Berlin.

    Returns:
        Dict: Enthält total_distance, online_distance und vor_ort_distance in km
    """
    query = """
    MATCH (o:Order)-[:LOCATED_AT]->(loc:Location)
    RETURN loc.latitude AS lat, loc.longitude AS lon, o.order_type AS order_type, loc AS location
    """

    total_distance = 0.0
    online_distance = 0.0
    vor_ort_distance = 0.0

    results = db.run_query(query)
    for record in results:
        lat, lon, order_type = record["lat"], record["lon"], record["order_type"]

        if lat and lon:
            distance = get_route_distance(lat, lon)
            total_distance += distance

            if order_type == "online":
                online_distance += distance
            if order_type == "vor_ort":
                vor_ort_distance += distance

    return {
        "total_distance": total_distance,
        "online_distance": online_distance,
        "vor_ort_distance": vor_ort_distance
    }


def calculate_emissions(distance_km: float, fuel_consumption: float, co2_per_liter: float) -> float:
    """
    Berechnet CO2-Emissionen basierend auf Strecke, Verbrauch und Emissionsfaktor.

    Args:
        distance_km: Zurückgelegte Strecke in km
        fuel_consumption: Kraftstoffverbrauch in l/100km
        co2_per_liter: CO2-Emissionen pro Liter Kraftstoff in kg

    Returns:
        float: CO2-Emissionen in kg
    """
    fuel_used = (distance_km / 100) * fuel_consumption
    return fuel_used * co2_per_liter


def get_transport_emissions() -> Dict[str, Any]:
    """
    Liefert detaillierte Transportemissionen, aufgeteilt nach Transportmitteln.

    Returns:
        Dict: Enthält Emissionswerte, Strecken und Einsparungen
    """
    # Streckendaten holen
    distances = get_total_distance()
    total_distance = distances["total_distance"]

    # Distanzen nach Transportmittel aufteilen
    distanzen = {
        transport: total_distance * (prozent / 100)
        for transport, prozent in TRANSPORT_VERTEILUNG.items()
    }

    # Emissionen berechnen
    emissions = {
        "lkw": calculate_emissions(distanzen["lkw"], FUEL_CONSUMPTION["lkw"], EMISSIONS_FACTORS["diesel"]),
        "pkw": calculate_emissions(distanzen["pkw"], FUEL_CONSUMPTION["pkw"], EMISSIONS_FACTORS["benzin"]),
        "schiff": distanzen["schiff"] * EMISSIONS_FACTORS["schiff"] * 10,  # Annahme: 10 Tonnen durchschnittliche Ladung
        "flugzeug": distanzen["flugzeug"] * EMISSIONS_FACTORS["flugzeug"]
    }

    # Einsparungen berechnen
    einsparungen = {
        transport: emission * (EINSPARUNG_PROZENT / 100)
        for transport, emission in emissions.items()
    }
    gesamt_einsparung = sum(einsparungen.values())

    # Originale Werte beibehalten
    total_emissions = calculate_emissions(
        distances["total_distance"],
        FUEL_CONSUMPTION["lkw"],
        EMISSIONS_FACTORS["diesel"]
    )

    online_emissions = calculate_emissions(
        distances["online_distance"],
        FUEL_CONSUMPTION["pkw"],
        EMISSIONS_FACTORS["benzin"]
    )

    vor_ort_emissions = calculate_emissions(
        distances["vor_ort_distance"],
        FUEL_CONSUMPTION["pkw"],
        EMISSIONS_FACTORS["benzin"]
    )

    # Ergebnisse zusammenstellen
    return {
        # Originale Werte
        "total_distance_km": distances["total_distance"],
        "online_distance_km": distances["online_distance"],
        "vor_ort_distance_km": distances["vor_ort_distance"],
        "total_emissions_kg": total_emissions,
        "online_emissions_kg": online_emissions,
        "vor_ort_emissions_kg": vor_ort_emissions,

        # Neue detaillierte Werte
        "transportmittel": {
            transport: {
                "distanz_km": distanzen[transport],
                "emissionen_kg": emissions[transport],
                "einsparung_kg": einsparungen[transport]
            } for transport in TRANSPORT_VERTEILUNG.keys()
        },
        "gesamt_einsparung_kg": gesamt_einsparung
    }


async def get_orders() -> List[Dict[str, Any]]:
    """
    Liefert alle Bestellungen mit Koordinaten für die Karte.

    Returns:
        List[Dict]: Liste von Bestellungen mit ID, Koordinaten und Typ
    """
    query = """
    MATCH (o:Order)-[:LOCATED_AT]->(loc:Location)
    RETURN o.order_id AS id, loc.latitude AS lat, loc.longitude AS lon, o.order_type AS type
    """
    orders = []

    results = db.run_query(query)
    for record in results:
        orders.append({
            "id": record["id"],
            "lat": record["lat"],
            "lon": record["lon"],
            "type": record["type"]
        })

    return orders


@router.get("", response_class=HTMLResponse)
async def resource_efficiency_dashboard(request: Request):
    """
    Dashboard-Endpunkt, der Inventardaten, Emissionsdaten und Bestellungen lädt.

    Args:
        request: FastAPI Request-Objekt

    Returns:
        TemplateResponse: Gerenderte HTML-Seite mit allen Daten
    """
    try:
        # Daten laden
        re_data = analyze_reusable_components()
        emissions_data = get_transport_emissions()
        orders = await get_orders()

        # Template mit Daten rendern
        return templates.TemplateResponse("resource_efficiency.html", {
            "request": request,
            "username": "admin",
            "role": "admin",
            "re_data": re_data,
            "emissions_data": emissions_data,
            "orders": orders if orders else []
        })
    except Exception as e:
        # Strukturierte Fehlerbehandlung
        import traceback
        error_details = traceback.format_exc()
        print(f"Fehler im Dashboard: {error_details}")
        raise HTTPException(status_code=500, detail=str(e))