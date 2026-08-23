from ultralytics import YOLO

from .modelRegistry import get_active_model_path

# Static helper model for synthetic data (combined with the komponenten model)
MODEL2 = "./model/best_synthetic_v2.pt"
model2 = YOLO(MODEL2)

# Active models come from the registry (config/models.json) so new versions
# uploaded on the /training page replace them without a redeploy. Access these
# via the module (config.model1), not `from .config import model1`, otherwise
# the reference goes stale after reload_models().
model = None
model1 = None
model_nubs_detecttion = None
# Damage model (Kratzer/Dellen/MDF-Platzer on crops); stays None until the
# first one is uploaded on the /training page — the check is skipped then.
model_schaeden = None


def reload_models():
    """(Re)load the active models — at startup and after a new model version
    is activated on the /training page."""
    global model, model1, model_nubs_detecttion, model_schaeden
    model1 = model = YOLO(get_active_model_path("komponenten"))
    model_nubs_detecttion = YOLO(get_active_model_path("nubs"))
    import os
    schaeden_path = get_active_model_path("schaeden")
    model_schaeden = YOLO(schaeden_path) if os.path.exists(schaeden_path) else None


reload_models()


# List of classes that need color detection
color_detection_classes = [
    "Auszug", "Verkleidung", "Systemboden",
    "Einzeltuer", "Doppeltuerblatt", "Doppeltuer", "Seitenverkleidung-0-IN",
    "Seitenverkleidung-0-0",
    "Seitenverkleidung-IN-IN"

]


beschichtung_detection_classes = ["Gerade", "Diagonale", "Sockelfuss", "Mutternstab","Noppenscheiben", "Griff", "Schraube"]

# ⚠️ Ersetze mit deinen Neo4j Aura Zugangsdaten
NEO4J_URI = "neo4j+s://XXXXXX.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "XXXXXX"

HERE_API_KEY = "XXXXXX"