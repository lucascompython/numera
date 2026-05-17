use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use rayon::prelude::*;
use rusqlite::Connection;

use super::db::{
    self, AssignmentRecord, ImageProcessingRecord, OcrRecord, ProcessingDb, StickerMatchRecord,
    VisualEmbeddingRecord,
};
use super::embedding::{
    EmbeddingProvider, VisualEmbeddingConfig, build_embedding_provider,
    generate_visual_embedding_with_provider,
};
use super::event_config::EventConfig;
use super::image_loader::discover_images;
use super::number_cropper::crop_number_region;
use super::preprocessing::{mat_to_dynamic_image, preprocess_number_crop, write_debug_image};
use super::sorter::{self, SortMode};
use super::sticker_matcher::{StickerDetection, StickerMatcher};
use super::visual_matcher::{self, VisualMatcherConfig};

#[derive(Debug, Clone)]
pub struct FirstStageConfig {
    pub db_path: PathBuf,
    pub event_id: i64,
    pub source_dir: PathBuf,
    pub output_dir: PathBuf,
    pub mode: SortMode,
    pub debug_mode: bool,
    pub reprocess: bool,
    pub thresholds: ProcessingThresholds,
    pub visual_embedding: VisualEmbeddingConfig,
    pub visual_matching: VisualMatcherConfig,
}

#[derive(Debug, Clone, Copy)]
pub struct ProcessingThresholds {
    pub min_sticker_confidence: f32,
    pub min_ocr_confidence: f32,
    pub min_good_matches: usize,
}

impl Default for ProcessingThresholds {
    fn default() -> Self {
        Self {
            min_sticker_confidence: 0.55,
            min_ocr_confidence: 0.65,
            min_good_matches: 12,
        }
    }
}

#[derive(Debug, Default)]
pub struct FirstStageSummary {
    pub discovered: usize,
    pub processed: usize,
    pub skipped: usize,
    pub assigned_by_ocr: usize,
    pub assigned_by_visual_match: usize,
    pub needs_review: usize,
    pub no_sticker_found: usize,
    pub ocr_failed: usize,
    pub failed: usize,
}

#[derive(Debug, Clone)]
struct ImageRunResult {
    image_id: Option<i64>,
    status: String,
    skipped: bool,
    error: Option<String>,
}

pub fn run_first_stage(
    config: &FirstStageConfig,
    progress_callback: impl Fn(usize, usize) + Send + Sync,
) -> Result<FirstStageSummary> {
    let db = ProcessingDb::open(&config.db_path)?;
    let event = db.get_event(config.event_id)?;
    let _template_probe = StickerMatcher::new(event.clone())?;
    let _embedding_probe = build_embedding_provider(&config.visual_embedding)?;
    let paths = discover_images(&config.source_dir)?;
    let total = paths.len();
    let completed = Arc::new(AtomicUsize::new(0));
    let progress_callback = &progress_callback;

    let results: Vec<ImageRunResult> = paths
        .par_iter()
        .map_init(
            || Worker::new(config, event.clone()).map_err(|err| err.to_string()),
            |worker, path| {
                let result = match worker {
                    Ok(worker) => worker.process(path),
                    Err(err) => Err(anyhow::anyhow!(err.clone())),
                };
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                progress_callback(done, total);
                match result {
                    Ok(result) => result,
                    Err(err) => ImageRunResult {
                        image_id: None,
                        status: "failed".to_string(),
                        skipped: false,
                        error: Some(err.to_string()),
                    },
                }
            },
        )
        .collect();

    let processed_ids: Vec<i64> = results
        .iter()
        .filter(|result| !result.skipped && result.error.is_none())
        .filter_map(|result| result.image_id)
        .collect();

    let mut final_results: Vec<ImageRunResult> = results
        .iter()
        .filter(|result| result.skipped || result.error.is_some())
        .cloned()
        .collect();

    if !processed_ids.is_empty() {
        let mut conn = db.connect()?;
        let mut visual_matching = config.visual_matching.clone();
        visual_matching.model_name = config.visual_embedding.resolved_model_name();
        visual_matching.debug_logging |= config.debug_mode;
        let _visual_summary = visual_matcher::run_anchor_visual_matching(
            &mut conn,
            config.event_id,
            &visual_matching,
        )?;
        final_results.extend(sort_processed_images(&db, config, &processed_ids)?);
    }

    Ok(summarize(total, &final_results))
}

struct Worker {
    conn: Connection,
    event: EventConfig,
    matcher: StickerMatcher,
    output_dir: PathBuf,
    debug_mode: bool,
    reprocess: bool,
    thresholds: ProcessingThresholds,
    embedding_provider: Box<dyn EmbeddingProvider>,
}

impl Worker {
    fn new(config: &FirstStageConfig, event: EventConfig) -> Result<Self> {
        let conn = db::connect(&config.db_path)?;
        let matcher = StickerMatcher::new(event.clone())?;
        let embedding_provider = build_embedding_provider(&config.visual_embedding)?;
        Ok(Self {
            conn,
            event,
            matcher,
            output_dir: config.output_dir.clone(),
            debug_mode: config.debug_mode,
            reprocess: config.reprocess,
            thresholds: config.thresholds,
            embedding_provider,
        })
    }

    fn process(&mut self, image_path: &Path) -> Result<ImageRunResult> {
        if !self.reprocess && db::is_image_processed(&self.conn, self.event.id, image_path)? {
            return Ok(ImageRunResult {
                image_id: None,
                status: "skipped".to_string(),
                skipped: true,
                error: None,
            });
        }

        let detection = match self.matcher.detect(image_path) {
            Ok(detection) => detection,
            Err(err) => {
                return self.persist_failure_review(image_path, err);
            }
        };

        let image_identity = db::upsert_image(
            &self.conn,
            self.event.id,
            image_path,
            Some(detection.source_width),
            Some(detection.source_height),
        )?;
        if image_identity.already_processed && !self.reprocess {
            return Ok(ImageRunResult {
                image_id: Some(image_identity.id),
                status: "skipped".to_string(),
                skipped: true,
                error: None,
            });
        }

        let debug_paths = DebugPaths::new(
            self.debug_mode,
            &self.output_dir,
            self.event.id,
            image_identity.id,
            image_path,
        );

        let visual_embedding =
            self.visual_embedding_record(image_identity.id, image_path, &debug_paths);
        let record =
            self.process_detection(image_identity.id, detection, &debug_paths, visual_embedding)?;
        db::save_processing_result(&mut self.conn, &record)?;

        Ok(ImageRunResult {
            image_id: Some(record.image_id),
            status: record.status,
            skipped: false,
            error: None,
        })
    }

    fn process_detection(
        &self,
        image_id: i64,
        detection: StickerDetection,
        debug_paths: &DebugPaths,
        visual_embedding: Option<VisualEmbeddingRecord>,
    ) -> Result<ImageProcessingRecord> {
        if !detection.found {
            let note = detection
                .note
                .clone()
                .unwrap_or_else(|| "sticker not found".to_string());
            let record = ImageProcessingRecord {
                image_id,
                status: "no_sticker_found".to_string(),
                sticker_match: sticker_record(&detection, None, None),
                ocr_result: None,
                visual_embedding,
                assignment: AssignmentRecord {
                    final_number: None,
                    assignment_method: "no_sticker_found".to_string(),
                    confidence: 0.0,
                    needs_review: true,
                    notes: Some(note),
                },
            };
            return Ok(record);
        }

        let warped = detection
            .warped_sticker
            .as_ref()
            .context("sticker detection was found but no warped sticker was produced")?;
        if let Some(path) = debug_paths.warped_sticker.as_ref() {
            write_debug_image(path, warped)?;
        }

        let number_crop = crop_number_region(warped, self.event.number_region)?;
        if let Some(path) = debug_paths.number_crop.as_ref() {
            write_debug_image(path, &number_crop)?;
        }

        let preprocessed = preprocess_number_crop(&number_crop)?;
        if let Some(path) = debug_paths.thresholded_crop.as_ref() {
            write_debug_image(path, &preprocessed.mat)?;
        }

        let ocr_image = mat_to_dynamic_image(&preprocessed.mat)?;
        let ocr = crate::ocr::recognize_number(&ocr_image);
        let ocr_record = ocr.as_ref().map(|ocr| {
            let raw_text = if ocr.all_detections.is_empty() {
                ocr.text.clone()
            } else {
                ocr.all_detections
                    .iter()
                    .map(|detection| detection.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            let is_high_confidence = !ocr.text.is_empty()
                && ocr.confidence >= self.thresholds.min_ocr_confidence
                && detection.match_confidence >= self.thresholds.min_sticker_confidence
                && detection.good_match_count >= self.thresholds.min_good_matches;

            OcrRecord {
                raw_text,
                digits_only: ocr.text.clone(),
                confidence: ocr.confidence,
                preprocessing_variant: preprocessed.variant.clone(),
                is_high_confidence,
            }
        });

        let (status, assignment) = match ocr_record.as_ref() {
            Some(ocr) if ocr.is_high_confidence => (
                "assigned_by_ocr".to_string(),
                AssignmentRecord {
                    final_number: Some(ocr.digits_only.clone()),
                    assignment_method: "assigned_by_ocr".to_string(),
                    confidence: combined_confidence(detection.match_confidence, ocr.confidence),
                    needs_review: false,
                    notes: None,
                },
            ),
            Some(ocr) => (
                "needs_review".to_string(),
                AssignmentRecord {
                    final_number: None,
                    assignment_method: "needs_review".to_string(),
                    confidence: combined_confidence(detection.match_confidence, ocr.confidence),
                    needs_review: true,
                    notes: Some(format!(
                        "low confidence OCR digits='{}' ocr={:.2} sticker={:.2} good_matches={}",
                        ocr.digits_only,
                        ocr.confidence,
                        detection.match_confidence,
                        detection.good_match_count
                    )),
                },
            ),
            None => (
                "ocr_failed".to_string(),
                AssignmentRecord {
                    final_number: None,
                    assignment_method: "ocr_failed".to_string(),
                    confidence: detection.match_confidence,
                    needs_review: true,
                    notes: Some("OCR returned no digit candidate".to_string()),
                },
            ),
        };

        let record = ImageProcessingRecord {
            image_id,
            status: status.clone(),
            sticker_match: sticker_record(
                &detection,
                debug_paths.warped_sticker.clone(),
                debug_paths.number_crop.clone(),
            ),
            ocr_result: ocr_record,
            visual_embedding,
            assignment,
        };

        Ok(record)
    }

    fn persist_failure_review(
        &mut self,
        image_path: &Path,
        err: anyhow::Error,
    ) -> Result<ImageRunResult> {
        let image_identity = db::upsert_image(&self.conn, self.event.id, image_path, None, None)?;
        let debug_paths = DebugPaths::new(
            self.debug_mode,
            &self.output_dir,
            self.event.id,
            image_identity.id,
            image_path,
        );
        let visual_embedding =
            self.visual_embedding_record(image_identity.id, image_path, &debug_paths);
        let record = ImageProcessingRecord {
            image_id: image_identity.id,
            status: "needs_review".to_string(),
            sticker_match: StickerMatchRecord {
                found: false,
                match_confidence: 0.0,
                good_match_count: 0,
                homography_valid: false,
                warped_sticker_path: None,
                number_crop_path: None,
            },
            ocr_result: None,
            visual_embedding,
            assignment: AssignmentRecord {
                final_number: None,
                assignment_method: "needs_review".to_string(),
                confidence: 0.0,
                needs_review: true,
                notes: Some(format!("processing failed: {err}")),
            },
        };
        db::save_processing_result(&mut self.conn, &record)?;

        Ok(ImageRunResult {
            image_id: Some(image_identity.id),
            status: "needs_review".to_string(),
            skipped: false,
            error: None,
        })
    }

    fn visual_embedding_record(
        &self,
        image_id: i64,
        image_path: &Path,
        debug_paths: &DebugPaths,
    ) -> Option<VisualEmbeddingRecord> {
        let generated = generate_visual_embedding_with_provider(
            self.embedding_provider.as_ref(),
            image_path,
            debug_paths.visual_crop.as_deref(),
        )
        .ok()?;
        Some(VisualEmbeddingRecord {
            image_id,
            model_name: self.embedding_provider.model_name().to_string(),
            crop_path: generated.crop_path,
            embedding: generated.embedding,
        })
    }
}

#[derive(Debug)]
struct DebugPaths {
    warped_sticker: Option<PathBuf>,
    number_crop: Option<PathBuf>,
    thresholded_crop: Option<PathBuf>,
    visual_crop: Option<PathBuf>,
}

impl DebugPaths {
    fn new(
        enabled: bool,
        output_dir: &Path,
        event_id: i64,
        image_id: i64,
        image_path: &Path,
    ) -> Self {
        if !enabled {
            return Self {
                warped_sticker: None,
                number_crop: None,
                thresholded_crop: None,
                visual_crop: None,
            };
        }

        let stem = image_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("image");
        let base = output_dir
            .join("_debug")
            .join(format!("event_{event_id}"))
            .join(format!("{image_id}_{stem}"));

        Self {
            warped_sticker: Some(base.with_extension("warped_sticker.png")),
            number_crop: Some(base.with_extension("number_crop.png")),
            thresholded_crop: Some(base.with_extension("thresholded_number.png")),
            visual_crop: Some(base.with_extension("visual_crop.png")),
        }
    }
}

fn sort_processed_images(
    db: &ProcessingDb,
    config: &FirstStageConfig,
    image_ids: &[i64],
) -> Result<Vec<ImageRunResult>> {
    let conn = db.connect()?;
    let decisions = db::load_sort_decisions(&conn, image_ids)?;
    let output_lock = Arc::new(Mutex::new(()));

    Ok(decisions
        .into_iter()
        .map(|decision| {
            let result = {
                let _guard = output_lock.lock().expect("output lock poisoned");
                sorter::place_file(
                    &decision.file_path,
                    &config.output_dir,
                    &decision.status,
                    decision.final_number.as_deref(),
                    config.mode,
                )
            };

            match result {
                Ok(_) => ImageRunResult {
                    image_id: Some(decision.image_id),
                    status: decision.status,
                    skipped: false,
                    error: None,
                },
                Err(err) => ImageRunResult {
                    image_id: Some(decision.image_id),
                    status: "failed".to_string(),
                    skipped: false,
                    error: Some(err.to_string()),
                },
            }
        })
        .collect())
}

fn sticker_record(
    detection: &StickerDetection,
    warped_sticker_path: Option<PathBuf>,
    number_crop_path: Option<PathBuf>,
) -> StickerMatchRecord {
    StickerMatchRecord {
        found: detection.found,
        match_confidence: detection.match_confidence,
        good_match_count: detection.good_match_count as i32,
        homography_valid: detection.homography_valid,
        warped_sticker_path,
        number_crop_path,
    }
}

fn combined_confidence(sticker_confidence: f32, ocr_confidence: f32) -> f32 {
    (sticker_confidence * 0.45 + ocr_confidence * 0.55).clamp(0.0, 1.0)
}

fn summarize(discovered: usize, results: &[ImageRunResult]) -> FirstStageSummary {
    let mut summary = FirstStageSummary {
        discovered,
        ..FirstStageSummary::default()
    };

    for result in results {
        if result.skipped {
            summary.skipped += 1;
            continue;
        }
        if result.error.is_some() {
            summary.failed += 1;
            continue;
        }

        summary.processed += 1;
        match result.status.as_str() {
            "assigned_by_ocr" => summary.assigned_by_ocr += 1,
            "assigned_by_visual_match" => summary.assigned_by_visual_match += 1,
            "no_sticker_found" => summary.no_sticker_found += 1,
            "ocr_failed" => summary.ocr_failed += 1,
            "needs_review" => summary.needs_review += 1,
            "ambiguous" => summary.needs_review += 1,
            _ => {}
        }
    }

    summary
}
