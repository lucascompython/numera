use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use rayon::prelude::*;
use rusqlite::Connection;

use super::db::{
    self, AssignmentRecord, ImageProcessingRecord, OcrRecord, ProcessingDb, StickerMatchRecord,
};
use super::event_config::EventConfig;
use super::image_loader::discover_images;
use super::number_cropper::crop_number_region;
use super::preprocessing::{mat_to_dynamic_image, preprocess_number_crop, write_debug_image};
use super::sorter::{self, SortMode};
use super::sticker_matcher::{StickerDetection, StickerMatcher};

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
    pub needs_review: usize,
    pub no_sticker_found: usize,
    pub ocr_failed: usize,
    pub failed: usize,
}

#[derive(Debug)]
struct ImageRunResult {
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
    let paths = discover_images(&config.source_dir)?;
    let total = paths.len();
    let completed = Arc::new(AtomicUsize::new(0));
    let output_lock = Arc::new(Mutex::new(()));
    let progress_callback = &progress_callback;

    let results: Vec<ImageRunResult> = paths
        .par_iter()
        .map_init(
            || Worker::new(config, event.clone()).map_err(|err| err.to_string()),
            |worker, path| {
                let result = match worker {
                    Ok(worker) => worker.process(path, &output_lock),
                    Err(err) => Err(anyhow::anyhow!(err.clone())),
                };
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                progress_callback(done, total);
                match result {
                    Ok(result) => result,
                    Err(err) => ImageRunResult {
                        status: "failed".to_string(),
                        skipped: false,
                        error: Some(err.to_string()),
                    },
                }
            },
        )
        .collect();

    Ok(summarize(total, &results))
}

struct Worker {
    conn: Connection,
    event: EventConfig,
    matcher: StickerMatcher,
    output_dir: PathBuf,
    mode: SortMode,
    debug_mode: bool,
    reprocess: bool,
    thresholds: ProcessingThresholds,
}

impl Worker {
    fn new(config: &FirstStageConfig, event: EventConfig) -> Result<Self> {
        let conn = db::connect(&config.db_path)?;
        let matcher = StickerMatcher::new(event.clone())?;
        Ok(Self {
            conn,
            event,
            matcher,
            output_dir: config.output_dir.clone(),
            mode: config.mode,
            debug_mode: config.debug_mode,
            reprocess: config.reprocess,
            thresholds: config.thresholds,
        })
    }

    fn process(
        &mut self,
        image_path: &Path,
        output_lock: &Arc<Mutex<()>>,
    ) -> Result<ImageRunResult> {
        if !self.reprocess && db::is_image_processed(&self.conn, self.event.id, image_path)? {
            return Ok(ImageRunResult {
                status: "skipped".to_string(),
                skipped: true,
                error: None,
            });
        }

        let detection = match self.matcher.detect(image_path) {
            Ok(detection) => detection,
            Err(err) => {
                return self.persist_failure_review(image_path, err, output_lock);
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

        let (record, sort_status, sort_number) =
            self.process_detection(image_identity.id, detection, &debug_paths)?;
        db::save_processing_result(&mut self.conn, &record)?;

        {
            let _guard = output_lock.lock().expect("output lock poisoned");
            sorter::place_file(
                image_path,
                &self.output_dir,
                &sort_status,
                sort_number.as_deref(),
                self.mode,
            )?;
        }

        Ok(ImageRunResult {
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
    ) -> Result<(ImageProcessingRecord, String, Option<String>)> {
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
                assignment: AssignmentRecord {
                    final_number: None,
                    assignment_method: "no_sticker_found".to_string(),
                    confidence: 0.0,
                    needs_review: true,
                    notes: Some(note),
                },
            };
            return Ok((record, "no_sticker_found".to_string(), None));
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

        let (status, assignment, sort_number) = match ocr_record.as_ref() {
            Some(ocr) if ocr.is_high_confidence => (
                "assigned_by_ocr".to_string(),
                AssignmentRecord {
                    final_number: Some(ocr.digits_only.clone()),
                    assignment_method: "assigned_by_ocr".to_string(),
                    confidence: combined_confidence(detection.match_confidence, ocr.confidence),
                    needs_review: false,
                    notes: None,
                },
                Some(ocr.digits_only.clone()),
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
                None,
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
                None,
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
            assignment,
        };

        Ok((record, status, sort_number))
    }

    fn persist_failure_review(
        &mut self,
        image_path: &Path,
        err: anyhow::Error,
        output_lock: &Arc<Mutex<()>>,
    ) -> Result<ImageRunResult> {
        let image_identity = db::upsert_image(&self.conn, self.event.id, image_path, None, None)?;
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
            assignment: AssignmentRecord {
                final_number: None,
                assignment_method: "needs_review".to_string(),
                confidence: 0.0,
                needs_review: true,
                notes: Some(format!("processing failed: {err}")),
            },
        };
        db::save_processing_result(&mut self.conn, &record)?;
        {
            let _guard = output_lock.lock().expect("output lock poisoned");
            sorter::place_file(
                image_path,
                &self.output_dir,
                "needs_review",
                None,
                self.mode,
            )?;
        }

        Ok(ImageRunResult {
            status: "needs_review".to_string(),
            skipped: false,
            error: None,
        })
    }
}

#[derive(Debug)]
struct DebugPaths {
    warped_sticker: Option<PathBuf>,
    number_crop: Option<PathBuf>,
    thresholded_crop: Option<PathBuf>,
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
        }
    }
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
            "no_sticker_found" => summary.no_sticker_found += 1,
            "ocr_failed" => summary.ocr_failed += 1,
            "needs_review" => summary.needs_review += 1,
            _ => {}
        }
    }

    summary
}
