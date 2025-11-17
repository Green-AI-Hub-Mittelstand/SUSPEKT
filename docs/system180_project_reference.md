# System180 × DFKI — SUSPEKT Project Reference

> Status: March 2025 — internal briefing for presentation work on the ISM 2025 paper

## 1. Organizations & Stakeholders
- **System 180 (Berlin)**: Manufactures modular stainless-steel furniture systems (tubes, diagonals, connectors, cladding, drawers) that can be reconfigured for offices, studios, workshops, and event spaces. Components are designed for repeated reuse; unused stock and returned parts accumulate in Berlin warehouses.
- **DFKI – German Research Center for Artificial Intelligence**: Europe’s leading applied AI research institute. The Smart Factory / Innovative Factory Systems group leads SUSPEKT, covering computer vision, semantic modeling, deployment, and dissemination.
- **Green-AI Hub Mittelstand**: A German Federal Ministry for the Environment initiative that finances and coordinates SME pilots linking AI to resource efficiency. Provides outreach formats (“Green AI im Dialog”) and documentation requirements.
- **System context**: The work directly supports the Ecodesign for Sustainable Products Regulation (ESPR, COM/2022/142), which requires furniture makers to deliver Digital Product Passports (DPPs) that span composition, condition, and lifecycle transitions.

## 2. Problem Statement & Requirements
1. **Lack of identifiers**: System180 parts (20 mm stainless tubes, diagonals, MDF panels, drawers, feet, etc.) do not carry QR/RFID tags; adhesive or engraving is infeasible because of aesthetics, small surfaces, and sheer part counts.
2. **Manual inventory**: Disassembling a workbench creates tens of loose pieces; technicians currently measure, sort, and document each part in spreadsheets—slow, error-prone, and not ESPR-compliant.
3. **Reuse workflows**: Reusable parts must be sorted, quality-checked, optionally recoated, and reassigned to new customer orders or internal collections.
4. **Transparency**: Stakeholders (System180 logistics, clients, regulators) demand machine-readable DPP exports, provenance of reused parts, and audit trails for CO₂ savings.

### Functional requirements
- Detect classes + condition from RGB images and short videos captured with commodity cameras.
- Estimate geometric properties (length, thickness) well enough to distinguish the nine straight tube lengths System180 sells.
- Determine décor or coating finishes in HEX/NCS codes for MDF panels, doors, drawers, and powder-coated steel.
- Collect detections into orders/collections, allow manual corrections, and persist to a graph DB for traceability.
- Provide dashboards for inventory status, material savings, and transport emissions (online vs. onsite jobs).

### Non-functional requirements
- Edge-capable for field work (Jetson Orin Nano + DepthAI cameras).
- Incrementally trainable (new images enter Label Studio, YOLO is retrained directly from repo scripts).
- Open-source stack (FastAPI, Ultralytics, Neo4j Aura, SQLite) with Docker-based deployment and reproducible setup docs.

## 3. Solution Overview — Collection-based DPPs
- **Collection Principle**: Every component belongs to exactly one digital collection (e.g., “Order_4711_Workbench” or “Berlin_Stockpile_A”). Moving parts between collections mirrors physical reorganizations and replaces physical IDs.
- **Two complementary interfaces**:
  1. **Web application** for batch image uploads, review, and data export.
  2. **Edge demonstrator** (3 cameras) for shop-floor digitization and measurement.
- **Knowledge graph**: Orders, Images, Components, and Locations become Neo4j nodes; Collections appear implicitly as `Order` nodes or custom groups. This enables queries such as “list all reusable stainless tubes stored in Berlin depot X”.

## 4. Web Application Modules (FastAPI, `webapp/`)
1. **Authentication (`webapp/auth.py`)**: Session middleware with minimal password gate for admin-only routes.
2. **Detection endpoint (`/detect`, `webapp/model.py`)**:
   - Accepts multipart uploads (supporting up to 4 images + optional labels describing view (front/side/top) and capture type).
   - Calls `process_images` for inference and caches per-image pandas DataFrames for later editing.
3. **Ensemble detection (`webapp/processImage.py`)**:
   - Runs YOLOv11 twice (`model1` for real-trained weights, `model2` for synthetic) and fuses predictions through IoU-based Non-Maximum Suppression (`combineYOLOModels.py`).
   - Saves originals, overlays, and per-object crops under `static/detected_images/...` for auditability and training.
4. **Class property enrichment (`webapp/class_properties.py`)**: Provides default weight, dimensions, type, and nominal condition for every System180 class (Gerade, Diagonale, Auszug, Türvarianten, etc.).
5. **Measurement (`webapp/measurement.py`)**: Calculates pixel-to-mm ratios using reference classes (Griff, Sockelfuß, Rollen, Noppenscheiben) and assigns the nearest standard length to straight tubes (known lengths: 180–900 mm). This is crucial for automatically labeling “Gerade 540” and similar SKUs.
6. **Décor detection (`webapp/decorDetection.py`)**:
   - Maintains a JSON palette with HEX + NCS codes from System180 décor catalog (Azurblau, Waldgrün, Sonnengelb, etc.).
   - Performs K-Means clustering on RGB images, prioritizing high-saturation dominant colors while safeguarding bright whites; returns structured metadata (`erkannte_farbe`, `hex_code`, `confidence`).
7. **Beschichtung detection (`webapp/beschichtungDetection.py`)**: Same approach for stainless vs. black powder-coated finishes.
8. **Condition CNN (`webapp/conditionDetection.py`)**:
   - `ZustandModel` processes 300×300 crops, outputting `Okay`, `MDF_Platzer`, or `Rohr_Kratzer`. Post-processing enforces class-aware sanity (e.g., screws cannot have MDF damage).
   - FastAPI endpoint `/api/detect_condition` writes predictions back into the cached DataFrame and updates the `reusable` flag.
9. **Review + confirmation**:
   - `/update_condition` for manual overrides.
   - `/review_results` collects user inputs, geocoded addresses (HERE API), and deleted rows before rendering `review_results.html`.
   - `/confirm_results` persists confirmed components into Neo4j via `neo4jIntegration.py`.
10. **Neo4j integration**:
    - Each component node contains bounding boxes, décor/finish JSON, condition, reuse flag, mass/length, `comp_id` (bbox + order + UUID suffix), and `process_id` (order + date).
    - Orders connect to Locations (lat/lon) and Images; components link to both order and originating image.
11. **Inventory UI (`/inventory`)**: Simple System180-branded table with filters for Auftrag (order), Komponente, Material, Zustand.
12. **Resource efficiency dashboard (`/resource`)**:
    - `resource_efficiency.py` aggregates lengths, weights, and material savings from `reusable=true` components; calculates CO₂ avoidance via stainless steel factors (7 kg CO₂/kg) and MDF (0.6 kg CO₂/kg).
    - `transport_emission.py` sums total/online/on-site distances using HERE Routing, splits travel into LKW/PKW/ship/air percentages, and displays emissions plus projected optimizations.
    - `templates/resource_efficiency.html` renders KPI cards, Chart.js visualizations, and a Leaflet map (Berlin HQ + order markers, colored by capture mode).
13. **Video pipeline (`webapp/videoDetection.py`)**: Stores uploads, streams processed frames via multipart MJPEG, overlays YOLO predictions + FPS, and exposes processed downloads.
14. **Training utilities**:
    - `trainingDataCollector.py` archives uploads/crops for manual annotation.
    - `modelTraining.py` pulls Label Studio annotations via API, builds YOLO datasets, splits train/val (80/20), and retrains using Ultralytics CLI; best checkpoints replace `model/system180custommodel_v1.pt`.
15. **R&D modules**:
    - `object_tracking/structure.py` demonstrates BERT-based semantic matching between detected components and expected assembly relationships, flagging missing parts in a bill of materials.

## 5. Models & Training Details
- **YOLOv11 detectors**:
  - `system180custommodel_v1.pt`: trained on real photographs (approx. 1 000 annotated images) provided by System180 and DFKI labeling sessions (Label Studio, Roboflow).
  - `best_synthetic_v2.pt`: trained on 10 000 Isaac-Sim renders covering controlled lighting, occlusions, and rare classes.
  - Ensemble logic ensures each bounding box carries the highest available confidence and class ID from either model.
- **Nubs segmentation**: `NubsUpDown.pt` identifies whether assembly knobs (Nubs) face up or down; deployed directly on DepthAI hardware for faster throughput and polygon ROI support.
- **Condition CNN**: Custom 6-layer conv net with dropout + MLP head (relu) for classification; training data derived from cropped detections labeled as `Okay`, `Rohr_Kratzer`, or `MDF_Platzer`.
- **Décor/coating detectors**: K-Means (k=5) per crop, heuristics for brightness/saturation to avoid misclassifying white backgrounds.
- **ArUco-based measurement**: In the demonstrator, OBSBOT frames include known-size ArUco markers; `aruco_utils.py` tracks markers and feeds corrected dimensions into detection overlays.

## 6. Edge Demonstrator ( `demonstrator/` )
- **Hardware layout**:
  - Mobile trolley with Jetson Orin Nano (8 GB) and three camera mounts.
  - Left/right **OAK-1 Max** cameras perform Nub segmentation locally using TensorRT `.engine` blobs.
  - Center **OBSBOT Meet 2 (4K)** streams into OpenCV; two detection models run simultaneously for ensemble predictions.
  - Optional OBSBOT zoom + lighting to inspect scratches/dents.
- **Software**:
  - `main.py` defines `FrameGrabber` subclasses for DepthAI and USB cameras, spawns threads, and serves MJPEG streams via FastAPI.
  - `convert_models_to_engine.sh`: Converts `.pt` YOLO models into TensorRT `.engine` files optimized for Jetson.
  - `setup.sh` installs Python deps, DepthAI SDK, and system packages.
  - `run.sh` launches the demonstrator (auto port negotiation) and opens http://localhost:8000.
  - `docs/GETTING_STARTED.md` and `install_desktop_icon.sh` provide non-programmer guidance plus desktop shortcuts.
- **Purpose**: Capture high-quality, multi-angle footage onsite (factory, client locations) where network access may be limited. Operators can perform quick triage: identify class, length, nub orientation, and visible damage in real time.

## 7. Data & Knowledge Graph
- **Node types**:
  - `Order`: Represents a collection or customer job; fields include `order_id`, `system180_order`, `contact_email`, `contact_phone`, `location`, `order_type` (online/vor_ort), `additional_info`, `process_id`.
  - `Image`: Original upload identifier; linked to stored files.
  - `Component`: Each detection/crop. Attributes include bounding boxes, class, décor/finish JSON, `gewicht`, `breite`, `laenge`, `zustand`, `reusable`, `image_name`, timestamps, and `confirmed` flags.
  - `Location`: Latitude/longitude + formatted address; optional but created when geocoding succeeds.
- **Relationships**:
  - `(Component)-[:BELONGS_TO]->(Image)`
  - `(Component)-[:PART_OF]->(Order)`
  - `(Order)-[:LOCATED_AT]->(Location)`
  - `(Image)-[:PART_OF]->(Order)` ensures traceability from raw input to final DPP.
- **Collections**: Implemented via `Order` or derived nodes; sub-collections can be created by cloning subsets of components with new `order_id`s, preserving provenance in the graph.
- **Reuse logic**: `reusable` boolean toggled automatically based on condition classes; dashboards filter by this attribute to report saved mass and CO₂.

### Current State (Aura export)
- Proof artifacts live in `webapp/docs/neo4j/`:
  - `node-export.csv` lists actual `Order`, `Image`, `Component`, and `Location` nodes, e.g. “Gerade 48” components with measured `laenge`, décor JSON, and `reusable=true`.
  - `relationship-export.csv` shows `PART_OF`, `BELONGS_TO`, and `LOCATED_AT` edges, confirming components are connected to both their orders and images.
  - `neo4j.png` / `bloom-visualisation.png` are screenshots from the live Neo4j Aura instance (`neo4j+s://589b02ea.databases.neo4j.io`).
- Each confirmation creates a fresh `Component` node with a unique `comp_id`; there is no deduplication or explicit “Collection” node beyond the order semantics yet.
- Some orders currently have multiple `LOCATED_AT` relationships when different geocode candidates were submitted—harmless but worth cleaning later.
- `farbe` is stored as a JSON string (e.g. `{'erkannte_farbe':'Sandgrau', ...}`); works for dashboards but could be normalized into separate properties if we need fine-grained graph queries.

## 8. Deployment & Operations
- **Web application**:
  - `webapp/Dockerfile` builds a Python 3.10-slim image with OpenCV dependencies; copies models, static assets, templates, and .env.
  - `webapp/compose.yaml` exposes FastAPI on 8000 plus optional Nginx (production profile) on 80/443. SSL certs from Let’s Encrypt reside in `/etc/letsencrypt` and are mounted read-only.
  - `webapp/docs/deployment.md` documents the Ubuntu 24.04 VM baseline, users (`stein`, `deployer`), Docker engine installation, certbot flow, and Compose commands for dev vs. production.
- **Edge demonstrator**:
  - Shell helpers (`setup.sh`, `run.sh`, `convert_models_to_engine.sh`) keep Jetson deployments reproducible for non-experts.
  - Desktop shortcuts and Python scripts (`snippets/find_cameras.py`, `swap_cameras.py`) help technicians identify and reassign OAK cameras.
- **Data management**:
  - Uploaded media lives under `static/detected_images`; `uploads/` and `processed/` hold video files, while `_training` directories capture additional training corpora.
  - Neo4j Aura cloud instance (credentials in `.env`/`webapp/config.py`) stores the knowledge graph.

## 9. Presentation & Paper Assets
- **Scientific paper**: `paper/System_180_GAIH_Pilot__ISM_2025_.pdf` and accompanying LaTeX (`System180.tex`, `backup.tex`) describe methodology, datasets, and results; anchor citations for the ISM 2025 talk.
- **Existing presentations**:
  - `presentations/perplexity/index.html`: Auto-generated slide deck (needs reinforcement) summarizing challenge, workflow, AI stack, demonstrator, benefits, and roadmap.
  - `presentations/system180/*.pdf|pptx`: Earlier talks (“Green AI im Dialog”, “GAIH Kurzvorstellung”) providing stakeholder context, mission statements, and event branding.
- **Next actions**: Consolidate the above into a polished ISM deck (12–15 slides) that emphasizes the Collection Principle, AI pipeline, demonstrator, and quantified resource benefits. The documentation in this file serves as the factual backbone.

## 10. Quick Reference — Repository Highlights
- `README.md`: Executive summary + navigation.
- `webapp/`: FastAPI backend, templates, static assets, models.
- `demonstrator/`: Jetson/DepthAI application scripts.
- `object_tracking/`: Experimental tracking + NLP reasoning.
- `paper/`: Manuscript, figures, bibliography.
- `presentations/`: Slide decks to be merged/expanded.

---
Prepared by: Codex CLI — consolidate updates here whenever new insights emerge during presentation work.
