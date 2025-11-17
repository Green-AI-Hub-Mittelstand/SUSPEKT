# ISM Paper vs. Current Implementation — Verification Notes

_Date: $(date +%Y-%m-%d)_

This memo cross-checks the claims in `paper/System180.tex` / `System_180_GAIH_Pilot__ISM_2025_.pdf` against the code that currently lives in the repository (webapp, demonstrator, Neo4j exports). It is meant as speaker prep for the ISM presentation.

## 1. Claims that Match Reality

| Paper Claim | Status in Repo |
| --- | --- |
| “Collection principle + Neo4j graph keeps every component linked to one collection/order.” (Sections 3.4, 5) | ✅ `webapp/webapp/neo4jIntegration.py` writes `Component` → `Image`/`Order` edges and optional `Order` → `Location`. CSV exports in `webapp/docs/neo4j/` confirm live data with these relationships. |
| “Dual demonstrators: browser upload + 3-camera edge rig.” (Intro, Sec. 4) | ✅ Web app exists under `webapp/`; Jetson-based demonstrator with two OAK-1 Max + OBSBOT 4K is implemented in `demonstrator/` (three camera grabbers, YOLO ensemble, ArUco measurement). |
| “YOLOv11 ensemble (real + synthetic) + nub segmentation.” (Sec. 3.3, 4) | ✅ Web app uses `model1` + `model2` ensemble (`processImage.py` and `combineYOLOModels.py`). Demonstrator also runs two YOLO engines (`demonstrator/main.py`). Nub segmentation weight `NubsUpDown.pt` present in `model/`. |
| “Graph-backed dashboards for inventory/material savings.” (Sec. 3.5, 4.2) | ✅ `/inventory` and `/resource` routes query Neo4j for orders/components, compute reuse metrics, and render Tailwind/Chart.js dashboards. |
| “Edge demonstrator runs fully on Jetson Orin Nano, triple cameras, FP16.” (Sec. 4.1) | ✅ `demonstrator/main.py` performs FP16 TensorRT inference, uses DepthAI for OAK cameras, OBSBOT for center, and overlays detections. |

## 2. Claims Partially Implemented

| Paper Claim | Reality & Gaps |
| --- | --- |
| **Furniture classification:** “YOLOv11 predicts overall product category (‘Table’, ‘Shelf’) and catalogue model name.” (Sec. 3.2) | ⚠️ Only component-level detection exists. There is no code that infers or stores an overall furniture type/model. Related functionality would need an additional classifier and storage schema. |
| **Aruco-based length measurement for every image.** (Sec. 3.4) | ⚠️ Edge demonstrator uses ArUco markers (`demonstrator/aruco_utils.py`), but the FastAPI pipeline uses heuristic scaling in `webapp/webapp/measurement.py` (reference object sizes + pixel ratios). Uploaded web photos do not require ArUco markers, so the claim is only true for the physical rig. |
| **Colour/decor stored as CIELab histograms.** (Sec. 3.4) | ⚠️ Current implementation uses K-Means clustering + predefined palettes to assign German colour names + HEX + NCS codes (`webapp/webapp/decorDetection.py`). No histogram vector is persisted; instead, a JSON blob (name/HEX/NCS/confidence) is stored in Neo4j. |
| **Condition classifier = pruned VGG-16 (half precision).** (Sec. 3.3/3.4) | ⚠️ Repo ships a custom PyTorch CNN (`ZustandModel` in `webapp/webapp/conditionDetection.py`) plus `faultDetection.pt`. No VGG-16 is referenced. Train script for damage classifier is absent. |
| **Vision-Language Model infers hidden parts.** (Sec. 3.4) | ⚠️ There is an experimental script (`object_tracking/structure.py`) that combines YOLO detections with BERT embeddings to match textual relations. It is not integrated into the production pipeline or UI. |

## 3. Claims Currently Missing in Code

| Paper Claim | Evidence of Absence |
| --- | --- |
| **DPP export service:** “Mapper converts JSON graph fragment into a DPP record that can be shared via QR code/REST.” (Sec. 3.2/3.5) | ❌ No mapper or export endpoint exists. After `/confirm_results` everything lives in Neo4j; there is no DPP serialization, QR generation, or REST sharing layer. |
| **Catalogue-level “Furniture classification” dataset/model.** | ❌ Only component classes are defined (`webapp/webapp/class_properties.py`). No multi-class classifier for entire furniture is present. |
| **Damage detection labels ‘Scratch’ / ‘Dent’ shown in UI.** (Sec. 4.1) | ❌ Condition labels available today are `Okay`, `MDF_Platzer`, `Rohr_Kratzer`, `Delle` (depending on manual update). Screenshot `webapp/screenshots/Bearbeitung.png` confirms naming mismatch vs. paper text. |
| **CIELab histogram storage in Neo4j.** | ❌ `farbe` property is stored as a JSON string describing recognized colour; no histogram vector/CIELab coordinates exist in the component nodes. |
| **Mapper to formal ESPR/IEC schema (Section 5 outlook).** | ❌ No code touches IEC/AAS/ODP mappings yet. All data remains in custom property graph form. |

## 4. Additional Observations

1. **Metrics (mAP₅₀ = 0.75 / mAP₅₀:95 = 0.596)** — Claimed values stem from training but the repo does not include evaluation notebooks or logs. We rely on the stated numbers; cannot rerun without datasets.
2. **Training pipeline** — `webapp/webapp/modelTraining.py` automates dataset pulls from Label Studio and YOLO retraining, aligning with the “hybrid dataset” claim, but actual image counts (10k synthetic, 1k real) are not versioned here.
3. **Neo4j data reality** — `webapp/docs/neo4j/node-export.csv` and `relationship-export.csv` provide up-to-date evidence (stored under version control) that components/orders/locations exist today. The graph is not a hypothetical construct.
4. **Reference/length estimation** — Because web uploads do not enforce ArUco markers, measurements rely on assumptions about known parts (e.g., handles 100 mm wide). This limitation is not described in the paper and may come up during Q&A.
5. **Security & roles** — Paper notes lack of fine-grained roles; in code `auth.py` is a placeholder with hard-coded credentials (`gaih` / `gaih`). This is consistent but worth mentioning as “future work” if asked.

## 5. Talking Points for Presentation

- Emphasize the parts that are truly implemented (graph, dual detectors, dashboards, demonstrator). Point to the Neo4j exports if challenged.
- Be transparent about features that are roadmap items: catalogue-level classification, DPP export service, richer colour vectors. Frame them as ongoing work.
- Clarify that ArUco-based measurement currently exists only in the physical rig; the FastAPI workflow uses reference-based scaling until we control the capture environment.
- Note the condition-model discrepancy (internal custom CNN vs. VGG-16). Either update slides or be ready to explain that we switched to a leaner architecture for deployment reasons.

