from neo4j import GraphDatabase
import json
import pandas as pd
from datetime import datetime
from .config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
import uuid

class Neo4jDatabase:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def _initialize_constraints(self):
        """ Erstellt Constraints in der Datenbank, falls sie noch nicht existieren. """
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS ON (o:Order) ASSERT o.order_id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS ON (i:Image) ASSERT i.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS ON (c:Component) ASSERT c.bbox_id IS UNIQUE")

    def store_image_results(self, image_results, order_id, system180_order, contact_email, contact_phone, location,
                            latitude, longitude, formatted_address, order_type, additional_info):
        """Speichert Erkennungsergebnisse und Metadaten in Neo4j"""

        with self.driver.session() as session:
            for image_name, df in image_results.items():
                for _, row in df.iterrows():
                    # Eindeutige ID für jede Komponente erstellen
                    unique_id = f"{row['bbox_id']}_{order_id}_{uuid.uuid4().hex[:8]}"  # bbox_id + Order-ID + zufällige UUID

                    properties = {
                        "comp_id": unique_id,  # Eindeutige ID für Neo4j
                        "bbox_id": row["bbox_id"],
                        "class": row["class"],
                        "confidence": row["confidence"],
                        "x_min": row["x_min"], "y_min": row["y_min"], "x_max": row["x_max"], "y_max": row["y_max"],
                        "farbe": json.dumps(row["farbe"]) if isinstance(row["farbe"], dict) else row["farbe"],
                        "typ": row["typ"],
                        "zustand": row["zustand"],
                        "reusable": bool(row["reusable"]),
                        "gewicht": row["gewicht"],
                        "breite": row["breite"],
                        "laenge": row["laenge"],
                        "image_name": image_name,
                        "crop_path": row["crop_path"] if "crop_path" in row else None,
                        "confirmed": False,
                        "system180_order": system180_order,
                        "contact_email": contact_email,
                        "contact_phone": contact_phone,
                        "location": location,
                        "additional_info": additional_info,
                        "order_type": order_type, #online oder vor Ort
                        "process_id": f"{order_id}_{datetime.now().strftime('%Y%m%d')}"
                    }

                    # Erstelle einen separaten Location-Knoten, falls Geodaten vorhanden sind
                    location_query_part = ""
                    if latitude and longitude:
                        location_query_part = """
                        MERGE (loc:Location {latitude: $latitude, longitude: $longitude}) 
                        ON CREATE SET loc.address = $location, 
                                      loc.formatted_address = $formatted_address
                        MERGE (order)-[:LOCATED_AT]->(loc)
                        """

                    query = (
                            "MERGE (img:Image {name: $image_name}) "
                            "ON CREATE SET img.name = $image_name "

                            "MERGE (order:Order {order_id: $order_id}) "
                            "ON CREATE SET order.name = 'Order ' + $order_id, "
                            "order.system180_order = $system180_order, order.contact_email = $contact_email, "
                            "order.contact_phone = $contact_phone, order.location = $location, "
                            "order.latitude = $latitude, order.longitude = $longitude, "
                            "order.formatted_address = $formatted_address, "
                            "order.order_type = $order_type, "
                            "order.additional_info = $additional_info, "
                            "order.process_id = $process_id "

                            # Nutze `CREATE`, damit immer eine neue Komponente erstellt wird!
                            "CREATE (comp:Component {comp_id: $comp_id}) "
                            "SET comp += $properties "

                            "MERGE (comp)-[:BELONGS_TO]->(img) "
                            "MERGE (comp)-[:PART_OF]->(order) "
                            "MERGE (img)-[:PART_OF]->(order) "
                            + location_query_part
                    )

                    # Konvertiere Geodaten zu Float, falls vorhanden
                    lat_value = float(latitude) if latitude and latitude.strip() else None
                    long_value = float(longitude) if longitude and longitude.strip() else None

                    session.run(query,
                                image_name=image_name,
                                order_id=order_id,
                                system180_order=system180_order,
                                contact_email=contact_email,
                                contact_phone=contact_phone,
                                location=location,
                                latitude=lat_value,
                                longitude=long_value,
                                formatted_address=formatted_address,
                                order_type=order_type,
                                additional_info=additional_info,
                                process_id=properties["process_id"],
                                comp_id=unique_id,
                                properties=properties)


    # ------------------------------------------------------------------
    # Freigabe durch System-180-Mitarbeitende
    #
    # Erfasste Bauteile landen zunaechst mit confirmed = false in der Datenbank.
    # Erst nach der Freigabe erscheinen sie im digitalen Lager.
    # ------------------------------------------------------------------
    def store_stueckliste(self, stueckliste, image_results, order_id, system180_order,
                          contact_email, contact_phone, location, latitude, longitude,
                          formatted_address, order_type, additional_info,
                          beschaedigt_je_bauteil=None):
        """Speichert das Mischmodell aus Sammelpositionen und Einzelerkennungen.

        Zwei Knotenarten mit klarer Aufgabenteilung:

        * ``:Component`` ist eine Sammelposition der Stueckliste. Sie traegt die
          Anzahl und ist die einzige Quelle fuer den Lagerbestand - dadurch kann
          dieselbe Strebe nicht ueber mehrere Ansichten doppelt gezaehlt werden.
        * ``:Erkennung`` ist eine einzelne Detektion mit Ausschnittsbild,
          Zustand und Farbe. Sie dient der Nachvollziehbarkeit, zaehlt aber nie
          fuer den Bestand, und Dubletten sind als solche markiert.
        """
        beschaedigt_je_bauteil = beschaedigt_je_bauteil or {}
        prozess_id = f"{order_id}_{datetime.now().strftime('%Y%m%d')}"
        lat_wert = float(latitude) if latitude and str(latitude).strip() else None
        lon_wert = float(longitude) if longitude and str(longitude).strip() else None

        auftrag_query = (
            "MERGE (order:Order {order_id: $order_id}) "
            "ON CREATE SET order.name = 'Order ' + $order_id, "
            "order.system180_order = $system180_order, order.contact_email = $contact_email, "
            "order.contact_phone = $contact_phone, order.location = $location, "
            "order.latitude = $latitude, order.longitude = $longitude, "
            "order.formatted_address = $formatted_address, "
            "order.order_type = $order_type, "
            "order.additional_info = $additional_info, "
            "order.process_id = $process_id "
        )

        with self.driver.session() as session:
            session.run(auftrag_query,
                        order_id=order_id, system180_order=system180_order,
                        contact_email=contact_email, contact_phone=contact_phone,
                        location=location, latitude=lat_wert, longitude=lon_wert,
                        formatted_address=formatted_address, order_type=order_type,
                        additional_info=additional_info, process_id=prozess_id)

            if latitude and longitude:
                session.run(
                    "MATCH (order:Order {order_id: $order_id}) "
                    "MERGE (loc:Location {latitude: $latitude, longitude: $longitude}) "
                    "ON CREATE SET loc.address = $location, loc.formatted_address = $formatted_address "
                    "MERGE (order)-[:LOCATED_AT]->(loc)",
                    order_id=order_id, latitude=lat_wert, longitude=lon_wert,
                    location=location, formatted_address=formatted_address)

            # --- Sammelpositionen aus der Stueckliste ---
            for position in stueckliste.get("positionen", []):
                basis = str(position["bauteil"]).split(" ")[0]
                anzahl = int(position["anzahl"])
                anzahl_beschaedigt = min(int(beschaedigt_je_bauteil.get(basis, 0)), anzahl)
                comp_id = f"{order_id}_{position['bauteil']}_{uuid.uuid4().hex[:8]}"

                eigenschaften = {
                    "comp_id": comp_id,
                    "art": "sammelposition",
                    "class": position["bauteil"],
                    "rolle": position.get("rolle"),
                    "anzahl": anzahl,
                    "anzahl_beschaedigt": anzahl_beschaedigt,
                    "anzahl_wiederverwendbar": anzahl - anzahl_beschaedigt,
                    "herkunft": position.get("herkunft"),
                    "hinweis": position.get("hinweis"),
                    "confirmed": False,
                    "system180_order": system180_order,
                    "contact_email": contact_email,
                    "contact_phone": contact_phone,
                    "location": location,
                    "additional_info": additional_info,
                    "order_type": order_type,
                    "process_id": prozess_id,
                }
                session.run(
                    "MATCH (order:Order {order_id: $order_id}) "
                    "CREATE (comp:Component {comp_id: $comp_id}) "
                    "SET comp += $eigenschaften "
                    "MERGE (comp)-[:PART_OF]->(order)",
                    order_id=order_id, comp_id=comp_id, eigenschaften=eigenschaften)

            # --- Einzelerkennungen als Beleg, nie fuer den Bestand ---
            for image_name, df in image_results.items():
                session.run(
                    "MATCH (order:Order {order_id: $order_id}) "
                    "MERGE (img:Image {name: $image_name}) "
                    "MERGE (img)-[:PART_OF]->(order)",
                    order_id=order_id, image_name=image_name)

                for _, row in df.iterrows():
                    erk_id = f"{row.get('bbox_id')}_{order_id}_{uuid.uuid4().hex[:8]}"
                    eigenschaften = {
                        "erkennung_id": erk_id,
                        "art": "erkennung",
                        "class": row.get("class"),
                        "confidence": row.get("confidence"),
                        "x_min": row.get("x_min"), "y_min": row.get("y_min"),
                        "x_max": row.get("x_max"), "y_max": row.get("y_max"),
                        "farbe": json.dumps(row["farbe"]) if isinstance(row.get("farbe"), dict) else row.get("farbe"),
                        "typ": row.get("typ"),
                        "zustand": row.get("zustand"),
                        "reusable": bool(row.get("reusable")),
                        "gewicht": row.get("gewicht"),
                        "breite": row.get("breite"),
                        "laenge": row.get("laenge"),
                        "crop_path": row.get("crop_path"),
                        "image_name": image_name,
                        "ansicht": row.get("ansicht"),
                        # Dublette aus einer zweiten Ansicht - siehe bill_of_materials.
                        "duplikat": bool(row.get("duplikat", False)),
                        # Der Bestand kommt ausschliesslich aus den Sammelpositionen.
                        "zaehlt_fuer_bestand": False,
                        "confirmed": False,
                        "process_id": prozess_id,
                    }
                    session.run(
                        "MATCH (order:Order {order_id: $order_id}) "
                        "MATCH (img:Image {name: $image_name}) "
                        "CREATE (erk:Erkennung {erkennung_id: $erk_id}) "
                        "SET erk += $eigenschaften "
                        "MERGE (erk)-[:BELONGS_TO]->(img) "
                        "MERGE (erk)-[:PART_OF]->(order)",
                        order_id=order_id, image_name=image_name,
                        erk_id=erk_id, eigenschaften=eigenschaften)

    def get_pending_orders(self):
        """Alle noch nicht freigegebenen Auftraege samt ihrer Bauteile."""
        query = """
        MATCH (o:Order)<-[:PART_OF]-(c:Component)
        WHERE c.confirmed IS NULL OR c.confirmed = false
        RETURN o.order_id AS order_id,
               o.system180_order AS system180_order,
               o.contact_email AS contact_email,
               o.contact_phone AS contact_phone,
               o.location AS location,
               o.formatted_address AS formatted_address,
               o.order_type AS order_type,
               o.additional_info AS additional_info,
               o.process_id AS process_id,
               collect({
                   comp_id: c.comp_id,
                   class: c.class,
                   anzahl: coalesce(c.anzahl, 1),
                   anzahl_beschaedigt: coalesce(c.anzahl_beschaedigt, 0),
                   herkunft: c.herkunft,
                   typ: c.typ,
                   farbe: c.farbe,
                   zustand: c.zustand,
                   reusable: c.reusable,
                   breite: c.breite,
                   laenge: c.laenge,
                   gewicht: c.gewicht,
                   crop_path: c.crop_path,
                   image_name: c.image_name
               }) AS komponenten
        ORDER BY o.order_id
        """
        with self.driver.session() as session:
            return [record.data() for record in session.run(query)]

    def get_all_orders(self):
        """Alle Auftraege - offene wie freigegebene - samt ihrer Bauteile.

        Grundlage der nachtraeglichen Bearbeitung auf der Freigabeseite. Anders
        als ``get_pending_orders`` werden hier auch bereits freigegebene
        Bauteile geliefert; jedes traegt sein eigenes ``confirmed``-Kennzeichen.
        """
        query = """
        MATCH (o:Order)
        OPTIONAL MATCH (o)<-[:PART_OF]-(c:Component)
        WITH o, collect(c) AS bauteile
        RETURN o.order_id AS order_id,
               o.system180_order AS system180_order,
               o.contact_email AS contact_email,
               o.contact_phone AS contact_phone,
               o.location AS location,
               o.formatted_address AS formatted_address,
               o.order_type AS order_type,
               o.additional_info AS additional_info,
               o.process_id AS process_id,
               [c IN bauteile | {
                   comp_id: c.comp_id,
                   class: c.class,
                   anzahl: coalesce(c.anzahl, 1),
                   anzahl_beschaedigt: coalesce(c.anzahl_beschaedigt, 0),
                   herkunft: c.herkunft,
                   typ: c.typ,
                   farbe: c.farbe,
                   zustand: c.zustand,
                   reusable: c.reusable,
                   breite: c.breite,
                   laenge: c.laenge,
                   gewicht: c.gewicht,
                   crop_path: c.crop_path,
                   image_name: c.image_name,
                   confirmed: coalesce(c.confirmed, false)
               }] AS komponenten,
               size([c IN bauteile WHERE coalesce(c.confirmed, false) = false]) AS offene_bauteile,
               reduce(letzte = 0, c IN bauteile |
                      CASE WHEN coalesce(c.confirmed_at, 0) > letzte
                           THEN c.confirmed_at ELSE letzte END) AS freigegeben_am
        ORDER BY freigegeben_am DESC, order_id DESC
        """
        with self.driver.session() as session:
            return [record.data() for record in session.run(query)]

    def update_component(self, comp_id, eigenschaften):
        """Einzelnes Bauteil aendern. Die Aufrufseite prueft die Werte."""
        query = """
        MATCH (c:Component {comp_id: $comp_id})
        SET c += $eigenschaften
        RETURN c.comp_id AS comp_id
        """
        with self.driver.session() as session:
            ergebnis = session.run(query, comp_id=comp_id, eigenschaften=eigenschaften)
            return ergebnis.single() is not None

    def delete_component(self, comp_id):
        """Bauteil endgueltig entfernen."""
        query = "MATCH (c:Component {comp_id: $comp_id}) DETACH DELETE c RETURN count(*) AS n"
        with self.driver.session() as session:
            return session.run(query, comp_id=comp_id).single() is not None

    def delete_order(self, order_id):
        """Auftrag mit allen Bauteilen verwerfen."""
        query = """
        MATCH (o:Order {order_id: $order_id})
        OPTIONAL MATCH (o)<-[:PART_OF]-(c:Component)
        DETACH DELETE c, o
        """
        with self.driver.session() as session:
            session.run(query, order_id=order_id)

    def get_unconfirmed_orders(self):
        with self.driver.session() as session:
            query = "MATCH (o:Order)<-[:PART_OF]-(c:Component) WHERE c.confirmed = false RETURN DISTINCT o.order_id"
            result = session.run(query)
            return [record["o.order_id"] for record in result]

    def confirm_order(self, order_id):
        with self.driver.session() as session:
            query = "MATCH (o:Order)<-[:PART_OF]-(c:Component) WHERE o.order_id = $order_id SET c.confirmed = true, c.confirmed_at = timestamp()"
            session.run(query, order_id=order_id)
