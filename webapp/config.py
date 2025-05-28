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

# ⚠️ Ersetze mit deinen Neo4j Aura Zugangsdaten
NEO4J_URI = "neo4j+s://589b02ea.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "system180"

HERE_API_KEY = "oYsUg1tZds2ED0hpl0ICUM58Dib-TyGZs4AIFQJXDm4"