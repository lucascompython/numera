# Numera Robust Batch Sorter Plan

Last updated: 2026-07-03

This document maps the original robust motorcycle event sorter plan to the current codebase state. Status labels:

- Done: implemented in code and wired into the relevant processing module.
- Partial: implemented as a scaffold or backend module, but not fully wired into the user-facing workflow or not complete enough for production use.
- Not started: not implemented yet.

## Product State

Numera is currently a Rust/GPUI desktop app with:

- Batch image processing for exports, previews, resizing, overlays, watermarking, and PDF/image output.
- Manual numbering mode for opening a folder and assigning rider numbers by hand.
- Autonomous numbering window for background numbering while the main numbering page stays usable.
- Shared numbering session state between the main numbering page and the autonomous window.
- Review images from autonomous mode can be opened back in the main/manual numbering page.
- Main/manual numbering no longer runs OCR automatically, which keeps manual numbering faster.
- Autonomous processing starts from the end of the folder so the user can manually number from the beginning at the same time.

Important current limitation: the user-facing autonomous window still uses the simpler OCR-based autonomous processor in `src/processing/autonomous_numbering.rs`. The more robust OpenCV/SQLite/event-sticker pipeline exists under `src/processing/event_sorter/`, but it is not yet wired into the autonomous UI as the main user-facing flow.

## Original Architecture Checklist

### 1. Event Setup

Original requirements:

- User provides the event sticker template image.
- User can mark/select the number region on the sticker template.
- Store the selected number region in normalized coordinates relative to the sticker template.
- Store event configuration in SQLite using `rusqlite`.

Current status: Partial.

Implemented:

- `src/processing/event_sorter/event_config.rs` defines event configuration and normalized number-region data.
- `src/processing/event_sorter/db.rs` creates the `events` table.
- `ProcessingDb::create_event` stores:
  - event name
  - sticker template path
  - template width/height
  - normalized number-region x/y/w/h
- `ProcessingDb::get_event` loads and validates the event config.

Still to do:

- Add user-facing setup UI for selecting the event sticker template.
- Add UI for drawing/selecting the number region on the sticker.
- Persist and reuse event configs from the app UI.
- Let the user choose an existing event config instead of creating a new one each run.
- Add optional event-edition/secondary sticker matching configuration.

### 2. Per-Image Processing

Original requirements:

- Load each image.
- Find the provided sticker template in the photo.
- Use OpenCV feature matching, preferably ORB or AKAZE.
- Estimate homography when enough good matches exist.
- Warp the detected sticker to template coordinates.
- Crop the configured number region.
- Preprocess the crop with grayscale, contrast normalization, denoising, sharpening, adaptive thresholding, morphology, and optional HSV/color filtering.
- Run existing OCR on the crop.
- Restrict OCR output to digits only.
- Store OCR candidate, confidence, sticker confidence, homography quality, and processing status.

Current status: Mostly done as backend modules.

Implemented:

- `src/processing/event_sorter/image_loader.rs` discovers input images.
- `src/processing/event_sorter/sticker_matcher.rs` performs OpenCV sticker/template matching.
- The matcher uses OpenCV feature matching and stores quality values such as match confidence, good match count, and homography validity.
- Homography/perspective warp is implemented for normalized sticker output.
- `src/processing/event_sorter/number_cropper.rs` crops the configured normalized number region from the warped sticker.
- `src/processing/event_sorter/preprocessing.rs` preprocesses the number crop for OCR.
- `src/processing/event_sorter/pipeline.rs` runs the existing OCR on the preprocessed crop.
- OCR records store raw text, digits-only value, confidence, preprocessing variant, and whether the result is high confidence.
- Sticker match records store found/not found, match confidence, good match count, homography validity, warped sticker debug path, and number crop debug path.
- Processing thresholds are explicit in `ProcessingThresholds`.

Still to do:

- Verify whether the current matcher is ORB or AKAZE in practice and tune it on real motorcycle event photos.
- Improve detection for small/distant stickers.
- Add multiple sticker candidates per image instead of only the best/current detection path.
- Add optional HSV/color filtering for known solid number-region backgrounds.
- Add more OCR preprocessing variants and pick the best result by confidence/plausibility.
- Reject wrong event stickers or old event-edition stickers more explicitly.
- Calibrate confidence scoring on real event folders.

### 3. Visual Rider/Motorcycle Matching Fallback

Original requirements:

- For every image, detect/crop the rider + motorcycle region.
- Initially use pretrained YOLO through ONNX Runtime when suitable.
- Crop a region containing motorcycle and rider together.
- Generate a visual embedding using CLIP, DINOv2, MobileNet, or another ONNX embedding model.
- Store embeddings in SQLite.
- Do not classify rider identity directly.
- Use embeddings for similarity search and grouping.

Current status: Partial.

Implemented:

- `src/processing/event_sorter/embedding.rs` defines `EmbeddingProvider`.
- `OpenCvEmbeddingProvider` keeps the handcrafted OpenCV color/texture embedding as the default fallback.
- `OnnxEmbeddingProvider` has been added using `ort` / ONNX Runtime.
- Visual embedding backend is configurable:
  - OpenCV or ONNX backend
  - ONNX model path
  - input size
  - normalization on/off
  - mean/std
  - model name
- ONNX provider loads the image crop, resizes it, converts to RGB tensor data, normalizes it, runs ONNX inference, extracts an f32 embedding, and L2-normalizes it.
- Embeddings are saved to `visual_embeddings` with `model_name`, `embedding_blob`, and `embedding_dim`.
- Debug mode can save the current visual crop path.
- The code is structured so the current crop can be replaced later.

Still to do:

- YOLO/person/motorcycle detection is not implemented yet.
- Current rider/motorcycle crop is only a central/lower proxy crop, not an actual detected rider-bike crop.
- Need a detector crop provider abstraction before adding YOLO.
- Need to select, ship/configure, and validate a real ONNX embedding model.
- Need model-specific preprocessing presets for the chosen embedding model.
- Need support for multiple rider/motorcycle crops when several motorcycles appear in one image.
- Need similarity quality evaluation on real event datasets.

### 4. Anchor-Based Assignment

Original requirements:

- High-confidence OCR results become anchors.
- Unknown/low-confidence images compare embeddings against anchor embeddings.
- Use nearest-neighbor/top-k voting.
- Assign only when nearest anchors strongly agree on one number.
- Mark weak or conflicting matches as review.
- Avoid assigning automatically when confidence is low or multiple numbers are plausible.

Current status: Mostly done as backend modules.

Implemented:

- `src/processing/event_sorter/visual_matcher.rs` loads high-confidence OCR assignments as anchors.
- Visual candidates are images still marked for review and not already assigned by high-confidence OCR.
- Matching uses cosine similarity on normalized embeddings.
- Top-k voting is implemented.
- Configurable thresholds exist:
  - `min_visual_similarity`
  - `min_topk_agreement`
  - `max_conflicting_anchor_similarity`
  - `min_anchor_count`
  - `min_confidence`
- Visual assignment preserves method `assigned_by_visual_match`.
- High-confidence OCR results are not overwritten by visual matching.
- Conflicting visual evidence can be stored as `ambiguous`.
- Top-k visual matches are stored in `visual_matches`.
- Debug logging can report anchor count, top-k matches, scores, and assignment/rejection reasons.

Still to do:

- Tune thresholds using real event data.
- Add better confidence calibration based on embedding model, crop quality, and anchor reliability.
- Add safeguards for visually similar motorcycles with different riders/numbers.
- Add review feedback so manually corrected images become anchors.
- Migrate embedding top-k lookup from brute-force Rust cosine search to local libSQL native vector search.

### 5. Clustering And Review

Original requirements:

- Optionally cluster visually similar images.
- Propagate a dominant high-confidence OCR number within a cluster.
- Mark conflicting clusters as ambiguous.
- Create review states:
  - `assigned_by_ocr`
  - `assigned_by_visual_match`
  - `needs_review`
  - `ambiguous`
  - `no_sticker_found`
  - `ocr_failed`

Current status: Partial.

Implemented:

- Review/assignment states exist in backend assignment flow:
  - `assigned_by_ocr`
  - `assigned_by_visual_match`
  - `needs_review`
  - `ambiguous`
  - `no_sticker_found`
  - `ocr_failed`
- `sorter.rs` routes assigned numbers, `_review`, `_ambiguous`, and `_no_sticker`.
- Visual conflict handling can mark images as `ambiguous`.
- The autonomous UI has a review queue for the simpler autonomous processor.
- Review images can be clicked/opened in the main numbering window.

Still to do:

- Clustering is not implemented.
- Cluster-level propagation is not implemented.
- Cluster conflict detection is not implemented.
- Backend event-sorter review states are not fully wired into the autonomous UI.
- Manual corrections do not yet update the event-sorter SQLite assignment/anchor state.
- Need a review UI that shows why each image needs review, including sticker match/OCR/visual evidence.

### 6. Output Sorting

Original requirements:

- Create output folders such as:
  - `/output/49/`
  - `/output/72/`
  - `/output/118/`
  - `/output/_review/`
  - `/output/_ambiguous/`
  - `/output/_no_sticker/`
- Copy or move based on user setting.
- Never silently overwrite files.
- Preserve original filenames or add conflict-safe suffixes.
- Store final assignment decisions in SQLite.

Current status: Mostly done as backend modules.

Implemented:

- `src/processing/event_sorter/sorter.rs` routes files by final status.
- Numbered assignments go to number folders.
- Review states go to `_review`, `_ambiguous`, or `_no_sticker`.
- Copy and move modes exist.
- Destination filename conflicts are handled with suffixes.
- Final assignment decisions are stored in SQLite `assignments`.
- `pipeline.rs` loads final sort decisions after OCR and visual matching.

Still to do:

- Wire these sorting settings into the autonomous/event setup UI.
- Add clearer user-facing controls for copy vs move.
- Add summary reporting after sorting.
- Decide how to handle already-sorted images when reprocessing.
- Keep database paths consistent if files are moved rather than copied.

## Database Schema Checklist

Current status: Mostly done for SQLite/rusqlite processing state. Native vector search migration is planned.

Implemented tables in `src/processing/event_sorter/db.rs`:

- `events`: done.
- `images`: done.
- `sticker_matches`: done.
- `ocr_results`: done.
- `visual_embeddings`: done.
- `assignments`: done.
- `visual_matches`: done.

Implemented behavior:

- Uses `rusqlite`.
- Creates schema automatically.
- Stores event config.
- Stores image metadata.
- Stores OCR/sticker/embedding/assignment/visual match results.
- Stores embedding blobs as little-endian f32 bytes.
- Uses `model_name` and `embedding_dim`.
- Uses WAL and busy timeout.

Current embedding search behavior:

- Embeddings are currently stored as regular blobs in SQLite.
- Anchor-based matching currently loads anchor and candidate embeddings into Rust memory.
- `visual_matcher.rs` computes cosine similarity in Rust for every candidate-anchor pair.
- This is simple and correct for the first implementation, but it scales as `candidate_count * anchor_count * embedding_dim`.

Local vector database experiment:

- A local experiment exists at `experiments/db_vector_test/`.
- Tested local file databases only, no cloud/server.
- Tested `F32_BLOB`, `vector32(...)`, `vector_distance_cos(...)`, `libsql_vector_idx(...)`, and `vector_top_k(...)`.
- Local `libsql = 0.9.30` worked end-to-end for indexed vector top-k search.
- Local `turso = 0.7.0-pre.17` did not work end-to-end for indexed vector search in this test because it rejected `libsql_vector_idx(...)` in `CREATE INDEX`.
- Decision for Numera: use local libSQL for native vector search when we migrate embedding lookup. Keep Turso Database on the radar, but do not target it for this app until local indexed vector search is verified.

Still to do:

- Add migrations/versioning if schema changes continue.
- Add event config UI persistence and selection.
- Add indexes if large event folders make lookup slow.
- Add manual review/correction history if needed.
- Add a database/vector search abstraction so the processing metadata can stay stable while embedding search moves to libSQL.
- Add a libSQL-backed vector index for embeddings, likely using `F32_BLOB(dim)` and `libsql_vector_idx(...)`.
- Replace brute-force top-k anchor lookup with `vector_top_k(...)` while keeping voting/confidence logic in Rust.
- Keep the current brute-force matcher as a fallback for portability and debugging.

## Resumability And Debug Checklist

Current status: Partial to mostly done.

Implemented:

- `images.processed_at` is stored.
- Pipeline skips already processed images unless `reprocess` is set.
- Processing results are stored incrementally in SQLite.
- Debug paths exist for:
  - warped sticker
  - cropped number region
  - thresholded OCR input
  - visual crop
- Confidence values and notes are stored.

Still to do:

- Wire reprocess/skip behavior into the user-facing UI.
- Add better pause/cancel/resume controls in the robust event-sorter path.
- Make debug output easy to inspect from the UI.
- Ensure resume works correctly when files have already been moved.

## Specific First Implementation Goal Checklist

Original first implementation goal:

1. `rusqlite` schema and processing state.
2. Event configuration with sticker template path and normalized number region.
3. OpenCV sticker template matching using ORB or AKAZE.
4. Homography + `warpPerspective`.
5. Crop configured number region.
6. Preprocess crop.
7. Pass crop to existing OCR.
8. Save OCR/sticker results to SQLite.
9. Sort high-confidence OCR results into number folders and send low-confidence images to `_review`.

Current status:

1. Done.
2. Done as backend, UI still pending.
3. Done as backend, needs real-data tuning.
4. Done.
5. Done.
6. Done, but more preprocessing variants/color filtering are still needed.
7. Done.
8. Done.
9. Done as backend sorting flow, UI integration still pending.

Conclusion: the first implementation goal is substantially implemented in backend modules. The main missing piece is product integration: the autonomous UI still needs to call this robust event-sorter pipeline instead of only the older simple OCR autonomous processor.

## After-First-Goal Checklist

Original next items:

10. Rider/motorcycle detection.
11. Visual embeddings.
12. Anchor-based visual matching.
13. Cluster-based propagation.

Current status:

10. Not started. No YOLO/person/motorcycle detector yet. Only a proxy crop exists.
11. Partial. OpenCV fallback embeddings and ONNX embedding provider exist, but no validated production model is selected/wired through UI.
12. Mostly done as backend. Anchor loading, top-k voting, cosine similarity, thresholds, and ambiguous handling exist.
13. Not started. No clustering or cluster propagation yet.

## Current Critical Gaps

The largest gaps are:

1. The robust event sorter is not yet the autonomous window's main processing path.
2. Event setup UI is missing.
3. YOLO/person/motorcycle crop detection is missing.
4. Multiple motorcycles/stickers per image are not handled robustly.
5. Visual embeddings need a real tested ONNX model.
6. Clustering and cluster propagation are not implemented.
7. Distant/small sticker OCR needs a focused accuracy pass.
8. Event-edition/wrong-sticker rejection needs to be added.
9. Performance needs measurement and thread budgeting on large folders.
10. Manual review corrections need to feed back into SQLite and anchor state.
11. Embedding search should migrate from brute-force Rust cosine search to local libSQL vector search before relying heavily on visual matching at scale.

## Recommended Next Implementation Order

### Phase 1: Wire The Robust Pipeline Into The App

Goal: make the existing backend useful from the UI.

Tasks:

- Add event setup UI in the numbering/autonomous flow.
- Let the user choose sticker template.
- Let the user mark number region on the sticker template.
- Create/load event config in SQLite.
- Start `event_sorter::pipeline::run_first_stage` from the autonomous window.
- Show event-sorter progress in both autonomous and main numbering windows.
- Keep the old simple autonomous OCR path only as fallback or legacy mode.
- Add UI controls for copy/move, debug mode, reprocess, and thresholds.

### Phase 2: Add libSQL Vector Search For Embeddings

Goal: make embedding lookup scalable before visual matching becomes a core production path.

Why this belongs early:

- The current brute-force Rust search is fine for the scaffold, but production events can have thousands of images and high-dimensional ONNX embeddings.
- libSQL local vector search was verified in `experiments/db_vector_test/`.
- Turso local vector indexing was not verified successfully, so local libSQL is the practical vector backend.
- This should happen before clustering and before depending heavily on visual propagation.

Tasks:

- Add a `VectorSearchProvider` abstraction.
- Keep the current brute-force Rust matcher as `BruteForceVectorSearchProvider`.
- Add `LibSqlVectorSearchProvider` for local indexed vector search.
- Decide whether libSQL becomes the whole processing DB or only the vector-search store.
- Prefer a low-risk first step: keep existing rusqlite metadata as source of truth and add libSQL-backed vector lookup for embeddings.
- Store vectors in libSQL as `F32_BLOB(dim)` using `vector32(...)` or binary f32 blobs in the expected format.
- Create a vector index with `libsql_vector_idx(embedding)`.
- Query anchor candidates with `vector_top_k(...)`.
- Keep assignment voting, conflict checks, and confidence scoring in Rust.
- Add benchmark coverage with realistic embedding counts and dimensions.
- Only switch the main processing DB from `rusqlite` to libSQL after the vector path is stable and measured.

### Phase 3: Accuracy Pass On Sticker/OCR

Goal: make high-confidence OCR anchors trustworthy.

Tasks:

- Tune ORB/AKAZE matcher thresholds on real event images.
- Improve small/distant sticker handling.
- Add multiple preprocessing variants for the number crop.
- Add optional background-color/HSV filtering.
- Add event-edition/sticker-version validation.
- Support multiple sticker candidates in one image.
- Store and display rejection reasons.

### Phase 4: Detector-Based Visual Crop

Goal: replace proxy crop with real rider/motorcycle crop.

Tasks:

- Add a crop provider trait/module.
- Keep current proxy crop as fallback.
- Add YOLO ONNX provider using `ort`.
- Detect person and motorcycle classes.
- Build combined rider-bike crop from detections.
- Handle multiple motorcycles by generating multiple crops/candidates.
- Store crop bounding boxes and crop quality.

### Phase 5: Production Visual Embeddings

Goal: make visual fallback useful for real re-identification.

Tasks:

- Choose a practical embedding model, for example CLIP, DINOv2, MobileNet embedding, or a re-id model.
- Add model-specific preprocessing presets.
- Validate cosine similarity on real known rider folders.
- Tune `min_visual_similarity`, `min_topk_agreement`, `max_conflicting_anchor_similarity`, `min_anchor_count`, and `min_confidence`.
- Add stronger conflict handling for similar bikes/riders.

### Phase 6: Clustering And Review Feedback

Goal: reduce manual review without increasing wrong assignments.

Tasks:

- Add visual clustering over embeddings.
- Propagate a dominant high-confidence OCR number within a cluster only when confidence is strong.
- Mark conflicting clusters as ambiguous.
- When the user manually fixes an image, update SQLite assignment state.
- Let manually fixed images become anchors.
- Show top visual matches and reasons in the review UI.

### Phase 7: Performance And Reliability

Goal: process thousands of images quickly without hurting UI responsiveness.

Tasks:

- Measure per-stage timings: decode, sticker match, warp/crop, OCR, embedding, visual match, sort.
- Budget threads across Rayon, OCR, OpenCV, and ONNX.
- Avoid oversubscribing CPU cores.
- Cache previous and next manual previews.
- Avoid reducing OCR input quality for small number regions.
- Add batch summaries and failure reports.
- Add test fixtures/regression folders for common failure cases.

## Immediate Next Step

The next concrete coding step should be Phase 1: connect the robust `src/processing/event_sorter/` pipeline to the autonomous window.

Minimum useful slice:

1. Add an event setup panel/button in numbering mode.
2. Let the user select a sticker template.
3. Let the user define the number region.
4. Create an `events` row in SQLite.
5. Start the robust event sorter from the autonomous window.
6. Show progress and final counts in the existing autonomous progress UI.
7. Route uncertain images to review instead of assigning them.
