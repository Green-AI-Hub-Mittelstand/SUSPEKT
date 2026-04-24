from neo4j import GraphDatabase

from .config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


class Neo4jDatabase:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def save_detection(self, image_id, class_name, x_min, y_min, x_max, y_max, confidence):
        """Speichert eine Erkennung mit Standardzustand 'In Ordnung' in Neo4j"""
        query = """
        MERGE (img:Image {id: $image_id})
        CREATE (obj:Detection {
            class_name: $class_name,
            x_min: $x_min, y_min: $y_min, x_max: $x_max, y_max: $y_max,
            confidence: $confidence,
            is_approved: false,
            zustand: "In Ordnung"
        })
        MERGE (obj)-[:DETECTED_IN]->(img)
        RETURN obj
        """
        self.run_query(query, {
            "image_id": image_id,
            "class_name": class_name,
            "x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max,
            "confidence": confidence
        })

    def get_pending_detections(self):
        """Abrufen der unbestätigten Erkennungen aus Neo4j"""
        query = """
        MATCH (obj:Detection)-[:DETECTED_IN]->(img:Image)
        WHERE obj.is_approved = false
        RETURN obj.class_name AS class_name, obj.confidence AS confidence,
               obj.x_min AS x_min, obj.y_min AS y_min, obj.x_max AS x_max, obj.y_max AS y_max,
               img.id AS image_id, ID(obj) AS obj_id
        """
        raw_results = self.run_query(query)

        # Ergebnisse nach Bild gruppieren
        grouped_results = {}
        for record in raw_results:
            image_id = record["image_id"]
            if image_id not in grouped_results:
                grouped_results[image_id] = {
                    "filename": image_id,
                    "detected": f"{image_id}_detected.jpg",  # Annahme: Das erkannte Bild hat "_detected.jpg"
                    "classes": []
                }

            grouped_results[image_id]["classes"].append({
                "name": record["class_name"],
                "confidence": f"{record['confidence']:.2f}",
                "bbox": [record["x_min"], record["y_min"], record["x_max"], record["y_max"]],
                "crop_path": f"{image_id}_crop.jpg",  # Annahme: Cropped Image existiert mit diesem Namen
                "properties": {
                    "gewicht": "Unbekannt",
                    "farbe": "Unbekannt",
                    "maße": "Unbekannt",
                    "typ": "Unbekannt",
                    "zustand": "Unbekannt"
                },
                "color_info": {}
            })

        return list(grouped_results.values())

    def get_approved_detections(self):
        query = """
        MATCH (obj:Detection)-[:DETECTED_IN]->(img:Image)
        WHERE obj.is_approved = true
        RETURN obj.class_name AS class_name, obj.confidence AS confidence,
               obj.x_min AS x_min, obj.y_min AS y_min, obj.x_max AS x_max, obj.y_max AS y_max,
               obj.is_approved AS is_approved, img.id AS image_id, ID(obj) AS obj_id
        """
        return [record.data() for record in self.run_query(query)]

    def approve_detection(self, detection_id):
        query = """
        MATCH (obj:Detection)
        WHERE ID(obj) = $detection_id
        SET obj.is_approved = true
        RETURN obj
        """
        result = self.run_query(query, {"detection_id": detection_id})
        return result.single() is not None

    def update_detection(self, detection_id, x_min, y_min, x_max, y_max):
        query = """
        MATCH (obj:Detection)
        WHERE ID(obj) = $detection_id
        SET obj.x_min = $x_min, obj.y_min = $y_min, obj.x_max = $x_max, obj.y_max = $y_max
        RETURN obj
        """
        result = self.run_query(query, {
            "detection_id": detection_id,
            "x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max
        })
        return result.single() is not None

    def delete_detection(self, detection_id):
        query = """
        MATCH (obj:Detection)
        WHERE ID(obj) = $detection_id
        DETACH DELETE obj
        """
        self.run_query(query, {"detection_id": detection_id})


db = Neo4jDatabase(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
