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
                    # 🚀 Eindeutige ID für jede Komponente erstellen
                    unique_id = f"{row['bbox_id']}_{order_id}_{uuid.uuid4().hex[:8]}"  # bbox_id + Order-ID + zufällige UUID

                    properties = {
                        "comp_id": unique_id,  # 🔥 Eindeutige ID für Neo4j
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


    def get_unconfirmed_orders(self):
        with self.driver.session() as session:
            query = "MATCH (o:Order)<-[:PART_OF]-(c:Component) WHERE c.confirmed = false RETURN DISTINCT o.order_id"
            result = session.run(query)
            return [record["o.order_id"] for record in result]

    def confirm_order(self, order_id):
        with self.driver.session() as session:
            query = "MATCH (o:Order)<-[:PART_OF]-(c:Component) WHERE o.order_id = $order_id SET c.confirmed = true, c.confirmed_at = timestamp()"
            session.run(query, order_id=order_id)
