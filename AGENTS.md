# AGENTS.md — SUSPEKT Knowledge Pack for LLM Operators

 _Last updated: 2025-11-14. Keep this file synchronized with major repo or presentation changes._

## 1. Mission & Context
- **Project name:** SUSPEKT (System180 Sustainable Product Passport Toolkit).
- **Goal:** Deliver Digital Product Passports (DPPs) for System180’s modular furniture without physical identifiers by combining computer vision, a Neo4j knowledge graph, and reusable collection logic.
- **Partners:**
  - **System 180 (Berlin):** Manufactures stainless-steel tube systems, panels, drawers, doors; all parts are modular and intended for reuse.
  - **DFKI – Smart Factory / Innovative Factory Systems:** Leads AI research, software, demonstrator work, and scientific dissemination.
  - **Green-AI Hub Mittelstand:** Federal initiative financing the pilot and expecting reproducible documentation and conference outputs.
- **Why it matters:** Upcoming Ecodesign for Sustainable Products Regulation (ESPR) requires granular DPPs for furniture. System180 needs fully auditable traceability of every component’s history, condition, and reuse cycle.

## 2. Current Focus (ISM 2025)
- Paper: `paper/System180.tex` (+ PDF) describes SUSPEKT for ISM 2025 in Malta.
- Presentations to maintain/build:
  - `presentations/ism/index.html` — 15‑min general talk.
  - `presentations/ism-paper/index.html` — paper-aligned talk with intro → motivation → methods → results → limitations.
  - Legacy decks live under `presentations/perplexity/` and `presentations/system180/` for reference.
- **Requirement:** Slides must only claim features that exist. When referencing roadmap ideas (QR export, catalogue classifier), describe them as future or potential work.
- **Helper docs:**
  - `docs/system180_project_reference.md` — exhaustive briefing (architecture, modules, partners, datasets, demonstrator mechanics).
  - `docs/paper_claims_vs_implementation.md` — table comparing every ISM paper claim with real code; cite this whenever validating statements.

## 3. Repository Layout (top level)
| Path | Purpose |
| --- | --- |
| `README.md` | Executive summary already rewritten with high-level context, AI stack, and component map. Keep consistent with AGENTS.md. |
| `AGENTS.md` | This file; onboarding + truth-source for future LLM agents. Update whenever architecture, datasets, or claims change. |
| `webapp/` | FastAPI backend + frontend assets + ML weights for browser workflow. Contains Docker setup and docs/screenshots. |
| `demonstrator/` | Jetson/DepthAI triple-camera edge rig with TensorRT engines and ArUco measurement scripts. |
| `object_tracking/` | Experimental tracking + semantic reasoning scripts (DeepSORT, SORT, BERT experiments). Not in production but useful for roadmap.
| `paper/` | LaTeX + PDF for ISM submission. Keep aligned with actual implementation. |
| `presentations/` | All slide decks (current ISM versions plus historical talk material and Perplexity draft). |
| `docs/` | Living documentation (system reference + paper claims verification). |
| `documents/` | External PDFs/Word docs (stakeholder briefs, class descriptions, ISM program). |
| `demonstrator/docs`, `webapp/docs` | Sub-documentation for respective subsystems (e.g., Neo4j exports, architecture diagrams). |

## 4. Web Application (`webapp/`)
### Stack & Entry Points
- Python 3.11+, FastAPI, Uvicorn, Jinja2 templates, Tailwind/HTMX front-end touches.
- `webapp/webapp/model.py` exposes main routes: `/`, `/detect`, `/review_results`, `/confirm_results`, `/inventory`, `/resource`, `/video`, `/api/detect_condition`, etc.
- Local run: `uvicorn webapp.model:app --reload`. Docker: `docker-compose up -d` using `webapp/compose.yaml`.

### Modules (all in `webapp/webapp/`)
- `auth.py` — minimal session auth (`gaih` credentials). No roles.
- `model.py` — FastAPI app factory, routing, caching, background jobs.
- `processImage.py` — orchestrates YOLO ensemble inference, saves overlays/crops under `webapp/static/detected_images/`.
- `combineYOLOModels.py` — IoU-based fusion of real-photo and synthetic YOLO outputs.
- `class_properties.py` — canonical metadata per component class (material, mass, typical condition, allowed reuse).
- `measurement.py` — pixel-to-mm scaling via known references; assigns nearest canonical tube length/diameter.
- `decorDetection.py` / `beschichtungDetection.py` — colour/decor inference via K-Means and JSON palettes (`colors.json`, `beschichtung.json`).
- `conditionDetection.py` — custom CNN (`ZustandModel`) handling OK/MDF_Platzer/Rohr_Kratzer/Delle classification, plus manual overrides.
- `inventory_routes.py`, `resource_efficiency.py`, `transport_emission.py` — dashboards quantifying inventory, resource savings, and CO₂ emissions (uses HERE geocoding + heuristics per transport mode).
- `neo4jIntegration.py` / `neo4j_database.py` — connectors to Neo4j Aura (credentials pulled from `.env`/`config.py`). Write nodes: `Order`, `Image`, `Component`, `Location`; relationships maintain Collection Principle (each component belongs to one order/collection).
- `modelTraining.py`, `trainingDataCollector.py`, `videoDetection.py`, `visualize.py` — data acquisition, YOLO retraining scripts, and MP4/MOV processing.

### Assets & Evidence
- `webapp/screenshots/` — UI captures (upload, review, dashboards) for presentations.
- `webapp/docs/neo4j/*` — CSV exports (`node-export.csv`, `relationship-export.csv`, `graph-export.csv`) + screenshots (`neo4j.png`, `bloom-visualisation.png`) proving the graph is populated.
- `webapp/docs/images/Architektur.png` — architecture diagram referenced in README/presentations.
- `webapp/model/` — ML weights (`system180custommodel_v1.pt`, `best_synthetic_v2.pt`, `faultDetection.pt`, `NubsUpDown.pt`, etc.). Keep sizes small; do not commit new weights unless necessary.

### Known Limitations (important for presentations)
- Web photo workflow **does not** enforce ArUco markers; measurement relies on reference-object heuristics.
- No DPP/QR export endpoint exists yet. Data stays inside Neo4j until implemented.
- Furniture-level classification (e.g., “Workbench type A”) is missing. Only component-level detections are stored.
- Authentication is static and admin-only.

## 5. Edge Demonstrator (`demonstrator/`)
- Purpose: Field-grade digitization cart with Jetson Orin Nano + two OAK-1 Max + OBSBOT 4K for onsite measurement.
- Entry: `demonstrator/main.py` (FastAPI + DepthAI loop). `run.sh`/`setup.sh` manage deployment on Jetson.
- Models:
  - `models/custom_320_FP16_detect.engine` (YOLOv11 detection on real data).
  - `models/synthetic_320_FP16_detect.engine` (synthetic complement).
  - `models/NubsUpDown_320_FP16_segment.engine` (nub orientation segmentation on OAK cameras).
- Utilities: `aruco_utils.py` for marker detection, `swap_cameras.py` for device reassignment, `convert_models_to_engine.sh` for TensorRT conversion.
- Templates/static under `demonstrator/templates/` & `demonstrator/static/` drive operator UI (FPS overlays, measurement readouts).
- README summarises camera assignments and workflow; keep it updated when hardware changes.

## 6. Experimental Area (`object_tracking/`)
- Contains attempts at DeepSORT + YOLO tracking (`deepsort.py`, `sort/`, `tracking/`).
- `structure.py` and related JSON files (`detected_objects.json`, `structured_relationships.json`) experiment with combining detections and NLP (BERT embeddings) to infer adjacency/assembly structure. Not integrated yet but can inspire future “hidden component inference” features.

## 7. Data, Models & Measurements
- **YOLO ensemble:** Two Ultralytics YOLOv11 checkpoints (real + synthetic). Combined with IoU-based NMS to improve recall on low-data classes.
- **Nub segmentation:** `NubsUpDown.pt` (PyTorch) + `.engine` variant for DepthAI.
- **Condition CNN:** Custom convolutional net (`ZustandModel`) with `faultDetection.pt`. Not VGG-16 (contrary to paper text); highlight this in Q&A.
- **Colour libraries:** `colors.json` for décor, `beschichtung.json` for powder coatings; both include HEX/NCS codes.
- **Measurement heuristics:** Known reference lengths (handles, feet, rollers) stored in code; measurement module maps pixel ratios to canonical tube lengths (180–900 mm). ArUco-based measurement exists only in demonstrator.
- **Datasets:** ~10k Isaac-Sim renders + ~1k annotated photos (not checked into repo). Retraining scripts expect Label Studio exports placed in `webapp/data/`.

## 8. Graph Persistence & Evidence
- Neo4j Aura instance credentials defined in `.env` (not tracked); connector code expects `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASS`.
- Graph schema:
  - `(:Order {order_id, order_name, status, geo})` linked to `(:Location)`.
  - `(:Image {path, capture_type, camera})` linked to orders and components.
  - `(:Component {class_name, length_mm, color_json, reuse_flag, condition, bbox})` linked to exactly one order (Collection Principle enforcement).
- Evidence stored in `webapp/docs/neo4j/` plus screenshot `neo4j.png` and Bloom capture `bloom-visualisation.png` prove nodes/relationships exist today.

## 9. Documentation & External References
- `docs/system180_project_reference.md` — canonical long-form description (stakeholders, requirements, solution, module map, deployment notes). Use this as first read for new collaborators.
- `docs/paper_claims_vs_implementation.md` — truth table for ISM paper. Always consult before making commitments. Update if code or paper changes.
- `documents/` — stakeholder material:
  - `Klassenbeschreibung_System180.*` lists component taxonomy.
  - `ISM_2025_Final_Program.pdf` — event schedule/branding cues.
  - `2407xx_*.pdf` — grant paperwork and Green-AI Hub presentation templates.
- `paper/` — LaTeX (`System180.tex`, `backup.tex`) + final PDF. Images live alongside; use for referencing formal statements in presentations.

## 10. Presentation Assets & Guidance
- All slides built with HTML/CSS (no external build step). Open directly in browser.
- `presentations/ism/index.html`: general-purpose talk (DFKI intro, motivation, architecture, demonstrator, results, outlook). Contains TODO spots for images from `webapp/screenshots/`.
- `presentations/ism-paper/index.html`: paper-faithful narrative with speaker notes and placeholders for figures (graph screenshot, demonstrator photo, dashboards).
- `presentations/perplexity/index.html`: auto-generated deck; only use for reference.
- `presentations/system180/*.pptx|pdf`: historical System180 decks; reusable visuals/logos.
- Upcoming work: incorporate verified metrics (mAP, throughput) once measurement data is centralized; add roadmap slide referencing missing features (DPP export, furniture classifier, QR tagging).

## 11. Known Discrepancies / Future Work (straight from verification memo)
- No furniture-level classifier or dataset.
- No DPP/QR export service or ESPR/AAS mapping pipeline.
- Colour storage is JSON, not CIELab histograms.
- Condition model is custom CNN, not VGG-16.
- Web workflow measurement lacks ArUco markers; only demonstrator enforces them.
- No automated mapper that shares graph fragments via REST.
Refer to `docs/paper_claims_vs_implementation.md` for the authoritative table and update it whenever features move across columns.

## 12. Workflow Expectations for Future Agents
1. **Accuracy first:** Never claim unimplemented features. When describing roadmap items, label them clearly as future work or “can be extended to…”.
2. **Evidence:** Back every statement with actual files (code, CSV exports, screenshots). Cite `webapp/docs/neo4j/*` or `webapp/screenshots/*` when proving functionality.
3. **Planning:** For multi-step tasks (docs, code, presentation), draft a plan and keep `docs/system180_project_reference.md` + this file aligned.
4. **Testing:** When touching code, run targeted checks (e.g., FastAPI unit tests, lint, or demo scripts). Avoid destructive git ops; repo may contain user changes.
5. **Security:** Do not expose real credentials. Reference `.env` placeholders or instructions instead.
6. **Presentation prep:** Start every ISM talk with who we are (DFKI Smart Factory + System180 context), then motivation → methodology → results → limitations/outlook, per user request.
7. **Neo4j:** Treat graph as source of truth. When debugging, use the CSV exports or connect to Aura (credentials outside repo). Document any schema or data changes immediately.
8. **Handovers:** Update `README.md`, this `AGENTS.md`, and relevant docs whenever new insights or assets are produced.

## 13. Quick Start Checklist for New Sessions
- Read `README.md` → `AGENTS.md` → `docs/system180_project_reference.md`.
- Review `docs/paper_claims_vs_implementation.md` to align messaging.
- Scan `presentations/ism*/index.html` to understand current storytelling and required assets.
- For Neo4j or demonstrator questions, consult `webapp/docs/neo4j/*` and `demonstrator/docs`.
- Use `webapp/screenshots/` and `documents/` for figures in decks.
- Confirm any new claims by tracing code paths (`rg` is pre-installed; prefer it for search).

Keeping AGENTS.md current saves hours when switching operators—update it as soon as you add features, data, or presentation material.
