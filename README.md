<div align="center">
    <img src="docs/images/emblem.png" alt="shellsmith">
</div>

<div align="center">
    <b>SUSPEKT</b>
</div>

Digital Product Passports for System180’s modular furniture ecosystem, developed at DFKI within the Green-AI Hub Mittelstand. SUSPEKT unifies a FastAPI web application, an edge demonstrator, and a Neo4j knowledge graph to track every reusable part without physical IDs by enforcing the **Collection Principle** (each component always belongs to exactly one digital collection).

## Project Snapshot
- **Use cases:** Automated bill-of-material capture, condition logging, reuse planning, and remote/onsite pre-commissioning for System180 shelves, workbenches, and accessories.
- **AI stack:** Dual YOLOv11 detectors (real vs. synthetic weights), custom CNN for surface damage, K-Means based décor/coating classifiers, and geometry extraction (pixel→mm, ArUco markers).
- **Data sources:** ~10k Isaac-Sim renders + ~1k annotated photographs, incremental uploads from the web UI, and 3-camera demonstrator streams.
- **Persistence:** Neo4j knowledge graph storing Orders ↔ Images ↔ Components ↔ Locations, enabling DPP exports and circular-economy analytics.

## Key Capabilities
- **Collection-based DPPs:** Components move between digital collections instead of carrying QR/RFID tags, ensuring ESPR-ready traceability for ID-free stainless-steel tubes, panels, doors, and fasteners.
- **Dual Experience:** A browser workflow for batch uploads plus a mobile Jetson Orin Nano demonstrator with two OAK-1 Max (nub orientation) and one OBSBOT 4K camera (defect detection + ArUco measurement).
- **Automatic Enrichment:** Each detection inherits class templates (`class_properties.py`), calculated lengths/weights, décor or coating codes (`colors.json`, `beschichtung.json`), and reuse flags derived from condition recognition.
- **Circular Analytics:** The `resource_efficiency` and `transport_emission` modules quantify material savings, CO₂ avoidance, and logistics emissions per order and per transport mode.

## Architecture Overview
1. **Acquisition:** Images or video streams enter `/detect` or `/video`, optionally tagged with view metadata (front/side/top) and capture type (single vs. multi-angle).
2. **Inference & Enrichment:** `processImage.py` ensembles YOLO outputs, generates crops, measures geometry, infers décor/coating, and triggers condition checks (manual overrides or `/api/detect_condition` CNN).
3. **Review:** Results are cached for interactive editing in `templates/results.html`, then confirmed via `/review_results` and `/confirm_results`, which geo-code orders (HERE API) before writing to Neo4j.
4. **Knowledge Graph:** `neo4jIntegration.py` stores each component as a dedicated node with bounding boxes, materials, reuse flag, and order metadata; collections can be queried for DPP generation.
5. **Dashboards:** `/inventory` and `/resource` render System180-branded UIs for inventory status, reusable stock, transport footprints, and order geographies (Leaflet + Chart.js).

## Component Map
| Area | Description | Key Files |
| --- | --- | --- |
| Detection pipeline | Uploads, YOLO ensemble, color/beschichtung inference, measurement, caching | `webapp/model.py`, `webapp/processImage.py`, `webapp/measurement.py` |
| Condition handling | CNN inference + manual overrides for OK / MDF-Platzer / Rohr_Kratzer / Delle | `webapp/conditionDetection.py`, `templates/results.html` |
| Data persistence | SQLite users + Neo4j component/order graph with Collection relationships | `webapp/user_db_models.py`, `webapp/neo4jIntegration.py`, `webapp/neo4j_database.py` |
| Resource dashboards | Material reuse, CO₂ analytics, HERE-powered transport distances, Leaflet map | `webapp/resource_efficiency.py`, `webapp/transport_emission.py`, `templates/resource_efficiency.html` |
| Video + training | Streaming YOLO detections for MP4/MOV uploads, Label Studio → YOLO retraining | `webapp/videoDetection.py`, `webapp/modelTraining.py`, `webapp/trainingDataCollector.py` |
| Edge demonstrator | Jetson-based triple-camera setup with TensorRT engines and ArUco measuring | `demonstrator/` (see `main.py`, scripts, docs) |

## Models & Data Highlights
- **YOLOv11 (Ultralytics):** `model/system180custommodel_v1.pt` (real-photo training) + `model/best_synthetic_v2.pt`; fused via `combineYOLOModels.py` with IoU-based NMS.
- **Nub orientation:** `model/NubsUpDown.pt` deployed on OAK-1 Max cameras for on-device segmentation.
- **Surface condition CNN:** `model/faultDetection.pt` (custom conv net, `ZustandModel`) fed by 300×300 crops and class-aware sanity checks.
- **Color/Beschichtung libraries:** JSON palettes with HEX + NCS codes; K-Means clustering promotes dominant décor shades while safeguarding whites and powder coats.
- **Data workflow:** Every upload persists original images, processed overlays, and per-class crops (`static/detected_images/...`) plus Label Studio-compatible annotations for continual learning.

## Edge Demonstrator
- **Hardware:** Jetson Orin Nano, dual Luxonis OAK-1 Max (left/right nub inspection), one OBSBOT Meet 2 4K (top-down defect + ArUco measurement), mobile workshop trolley.
- **Inference stack:** TensorRT `.engine` files (`custom_320_FP16_detect`, `synthetic_320_FP16_detect`, `NubsUpDown_320_FP16_segment`) streamed via FastAPI/DepthAI (`demonstrator/main.py`).
- **Purpose:** Rapid onsite digitization of disassembled System180 parts, feeding the same Collection graph as the webapp while providing real-time operator feedback and FPS overlays.

## Background & Partners
- **System 180** (Berlin): Manufactures modular stainless-steel furniture (tubes, cross braces, doors, drawers) designed for reconfiguration and long service life; pilot customer for ID-free DPP tracking.
- **DFKI – German Research Center for Artificial Intelligence:** Europe’s leading applied AI institute; owns the SUSPEKT research, hardware integration, and scientific dissemination (ISM 2025 paper).
- **Green-AI Hub Mittelstand:** German Federal Ministry for the Environment initiative fostering resource-efficient AI pilots for SMEs; provides coordination, dissemination, and funding context.

## Documentation Map
- `docs/system180_project_reference.md` — deep dive into partners, use cases, data flow, modules, models, demonstrator mechanics, and deployment playbooks.
- `paper/System_180_GAIH_Pilot__ISM_2025_.pdf` + `System180.tex` — full scientific manuscript accepted for ISM 2025.
- `presentations/` — current slide decks (`perplexity/`, `system180/`) slated for consolidation into the final conference presentation.

> Use this README as the high-level briefing. All technical nuances, training parameters, and presentation prep material are captured in the doc referenced above for quick onboarding ahead of the next iteration.
