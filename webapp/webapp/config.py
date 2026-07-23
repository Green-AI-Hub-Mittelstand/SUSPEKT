import os

from ultralytics import YOLO

# Load YOLO model
MODEL = "model/system180custommodel_v1.pt"
# MODEL = "model/system180CustomModelCaniaYolo11200Epochs.pt"
model = YOLO(MODEL)


#combine models
#MODEL1 = "./model/system180CustomModelCaniaYolo11200Epochs.pt"
MODEL1 = "./model/system180custommodel_v1.pt"
MODEL2 = "./model/best_synthetic_v2.pt"

model1 = YOLO(MODEL1)
model2 = YOLO(MODEL2)

MODEL_NUB = "./model/NubsUpDown.pt"

model_nubs_detecttion = YOLO(MODEL_NUB)


# List of classes that need color detection
color_detection_classes = [
    "Auszug", "Verkleidung", "Systemboden",
    "Einzeltuer", "Doppeltuerblatt", "Doppeltuer", "Seitenverkleidung-0-IN",
    "Seitenverkleidung-0-0",
    "Seitenverkleidung-IN-IN"

]


beschichtung_detection_classes = ["Gerade", "Diagonale", "Sockelfuss", "Mutternstab","Noppenscheiben", "Griff", "Schraube"]

# Neo4j Aura Zugangsdaten, siehe webapp/.env.example
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

HERE_API_KEY = os.getenv("HERE_API_KEY")