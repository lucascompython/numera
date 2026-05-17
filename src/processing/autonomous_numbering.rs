use std::ffi::{OsStr, OsString};
use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::numbering_session::NumberingSession;

#[derive(Clone)]
pub struct AutonomousNumberingConfig {
    pub source_dir: PathBuf,
    pub min_confidence: f32,
    pub max_workers: usize,
    pub cancel_flag: Arc<AtomicBool>,
    pub session: NumberingSession,
}

#[derive(Clone, Debug, Default)]
pub struct AutonomousNumberingSummary {
    pub discovered: usize,
    pub processed: usize,
    pub assigned: usize,
    pub needs_review: usize,
    pub already_completed: usize,
    pub skipped_missing: usize,
    pub failed: usize,
    pub cancelled: bool,
}

#[derive(Clone, Debug, Default)]
pub struct AutonomousNumberingProgress {
    pub completed: usize,
    pub total: usize,
    pub assigned: usize,
    pub needs_review: usize,
    pub already_completed: usize,
    pub skipped_missing: usize,
    pub failed: usize,
    pub active_files: Vec<String>,
    pub last_message: String,
    pub last_result: Option<AutonomousNumberingItemResult>,
}

#[derive(Clone, Debug)]
pub struct AutonomousNumberingItemResult {
    pub file_name: String,
    pub status: AutonomousNumberingItemStatus,
    pub number: Option<String>,
    pub confidence: Option<f32>,
    pub destination: Option<PathBuf>,
    pub error: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AutonomousNumberingItemStatus {
    Assigned,
    NeedsReview,
    AlreadyCompleted,
    SkippedMissing,
    Failed,
}

#[derive(Default)]
struct Counters {
    completed: AtomicUsize,
    assigned: AtomicUsize,
    needs_review: AtomicUsize,
    already_completed: AtomicUsize,
    skipped_missing: AtomicUsize,
    failed: AtomicUsize,
}

pub fn discover_numbering_images(source_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let entries = fs::read_dir(source_dir)
        .map_err(|error| format!("Failed to read folder {}: {error}", source_dir.display()))?;

    let mut images = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| format!("Failed to read folder entry: {error}"))?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|error| format!("Failed to inspect {}: {error}", path.display()))?;

        if file_type.is_file() && is_supported_image(&path) {
            images.push(path);
        }
    }

    images.sort();
    Ok(images)
}

pub fn run_autonomous_numbering(
    config: AutonomousNumberingConfig,
    progress_callback: impl Fn(AutonomousNumberingProgress) + Send + Sync,
) -> AutonomousNumberingSummary {
    let mut paths = match discover_numbering_images(&config.source_dir) {
        Ok(paths) => paths,
        Err(error) => {
            let result = AutonomousNumberingItemResult {
                file_name: config.source_dir.display().to_string(),
                status: AutonomousNumberingItemStatus::Failed,
                number: None,
                confidence: None,
                destination: None,
                error: Some(error.clone()),
            };
            progress_callback(AutonomousNumberingProgress {
                failed: 1,
                last_message: error,
                last_result: Some(result),
                ..AutonomousNumberingProgress::default()
            });
            return AutonomousNumberingSummary {
                failed: 1,
                ..AutonomousNumberingSummary::default()
            };
        }
    };
    paths.reverse();

    let total = paths.len();
    let worker_count = worker_count(config.max_workers, total);
    let counters = Arc::new(Counters::default());
    let next_index = Arc::new(AtomicUsize::new(0));
    let active_files = Arc::new(Mutex::new(vec![None::<String>; worker_count]));
    let paths = Arc::new(paths);
    let progress_callback = &progress_callback;

    progress_callback(snapshot(
        total,
        &counters,
        &active_files,
        format!("Ready - {total} images discovered"),
        None,
    ));

    if total == 0 {
        return AutonomousNumberingSummary::default();
    }

    if !crate::ocr::is_ocr_available() && crate::ocr::init_ocr().is_err() {
        let result = AutonomousNumberingItemResult {
            file_name: "OCR".to_string(),
            status: AutonomousNumberingItemStatus::Failed,
            number: None,
            confidence: None,
            destination: None,
            error: Some("OCR engine is unavailable".to_string()),
        };
        counters.failed.fetch_add(total, Ordering::Relaxed);
        counters.completed.fetch_add(total, Ordering::Relaxed);
        progress_callback(snapshot(
            total,
            &counters,
            &active_files,
            "OCR engine is unavailable".to_string(),
            Some(result),
        ));
        return summary(total, &counters, false);
    }

    std::thread::scope(|scope| {
        for worker_slot in 0..worker_count {
            let paths = paths.clone();
            let source_dir = config.source_dir.clone();
            let cancel_flag = config.cancel_flag.clone();
            let session = config.session.clone();
            let counters = counters.clone();
            let next_index = next_index.clone();
            let active_files = active_files.clone();

            scope.spawn(move || {
                loop {
                    if cancel_flag.load(Ordering::Relaxed) {
                        break;
                    }

                    let index = next_index.fetch_add(1, Ordering::Relaxed);
                    let Some(path) = paths.get(index).cloned() else {
                        break;
                    };

                    set_active(&active_files, worker_slot, path_file_name(&path));
                    progress_callback(snapshot(
                        total,
                        &counters,
                        &active_files,
                        format!("Processing {}", path_file_name(&path)),
                        None,
                    ));

                    let item = process_one(&path, &source_dir, config.min_confidence, &session);
                    apply_item_result(&counters, &item);
                    counters.completed.fetch_add(1, Ordering::Relaxed);
                    clear_active(&active_files, worker_slot);

                    let message = item_message(&item);
                    progress_callback(snapshot(
                        total,
                        &counters,
                        &active_files,
                        message,
                        Some(item),
                    ));
                }
            });
        }
    });

    summary(total, &counters, config.cancel_flag.load(Ordering::Relaxed))
}

fn process_one(
    path: &Path,
    source_dir: &Path,
    min_confidence: f32,
    session: &NumberingSession,
) -> AutonomousNumberingItemResult {
    let file_name = path_file_name(path);

    if session.is_completed(path) {
        return AutonomousNumberingItemResult {
            file_name,
            status: AutonomousNumberingItemStatus::AlreadyCompleted,
            number: None,
            confidence: None,
            destination: None,
            error: None,
        };
    }

    if !path.exists() {
        return AutonomousNumberingItemResult {
            file_name,
            status: AutonomousNumberingItemStatus::SkippedMissing,
            number: None,
            confidence: None,
            destination: None,
            error: None,
        };
    }

    let Some(ocr) = crate::ocr::recognize_number_from_path(path) else {
        return move_to_review(path, source_dir, file_name, None, None, session);
    };

    let digits = digits_only(&ocr.text);
    if digits.is_empty() || ocr.confidence < min_confidence {
        let number = (!digits.is_empty()).then_some(digits);
        return move_to_review(
            path,
            source_dir,
            file_name,
            number,
            Some(ocr.confidence),
            session,
        );
    }

    match move_into_folder(path, &source_dir.join(&digits)) {
        Ok(destination) => {
            session.mark_completed(path.to_path_buf());
            AutonomousNumberingItemResult {
                file_name,
                status: AutonomousNumberingItemStatus::Assigned,
                number: Some(digits),
                confidence: Some(ocr.confidence),
                destination: Some(destination),
                error: None,
            }
        }
        Err(error) if error.kind() == ErrorKind::NotFound => AutonomousNumberingItemResult {
            file_name,
            status: AutonomousNumberingItemStatus::SkippedMissing,
            number: Some(digits),
            confidence: Some(ocr.confidence),
            destination: None,
            error: None,
        },
        Err(error) => AutonomousNumberingItemResult {
            file_name,
            status: AutonomousNumberingItemStatus::Failed,
            number: Some(digits),
            confidence: Some(ocr.confidence),
            destination: None,
            error: Some(error.to_string()),
        },
    }
}

fn move_to_review(
    source: &Path,
    source_dir: &Path,
    file_name: String,
    number: Option<String>,
    confidence: Option<f32>,
    session: &NumberingSession,
) -> AutonomousNumberingItemResult {
    match move_into_folder(source, &source_dir.join("_review")) {
        Ok(destination) => {
            session.mark_completed(source.to_path_buf());
            AutonomousNumberingItemResult {
                file_name,
                status: AutonomousNumberingItemStatus::NeedsReview,
                number,
                confidence,
                destination: Some(destination),
                error: None,
            }
        }
        Err(error) if error.kind() == ErrorKind::NotFound => AutonomousNumberingItemResult {
            file_name,
            status: AutonomousNumberingItemStatus::SkippedMissing,
            number,
            confidence,
            destination: None,
            error: None,
        },
        Err(error) => AutonomousNumberingItemResult {
            file_name,
            status: AutonomousNumberingItemStatus::Failed,
            number,
            confidence,
            destination: None,
            error: Some(error.to_string()),
        },
    }
}

fn move_into_folder(source: &Path, folder: &Path) -> std::io::Result<PathBuf> {
    fs::create_dir_all(folder)?;

    let Some(file_name) = source.file_name() else {
        return Err(std::io::Error::new(
            ErrorKind::InvalidInput,
            format!("source path has no filename: {}", source.display()),
        ));
    };

    for attempt in 0..10_000 {
        let destination = folder.join(conflict_safe_name(file_name, attempt));
        if copy_create_new(source, &destination)? {
            match fs::remove_file(source) {
                Ok(()) => {}
                Err(error) if error.kind() == ErrorKind::NotFound => {}
                Err(error) => return Err(error),
            }
            return Ok(destination);
        }
    }

    Err(std::io::Error::new(
        ErrorKind::AlreadyExists,
        format!("no free destination filename for {}", source.display()),
    ))
}

fn copy_create_new(source: &Path, destination: &Path) -> std::io::Result<bool> {
    let mut input = File::open(source)?;
    let mut output = match OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)
    {
        Ok(file) => file,
        Err(error) if error.kind() == ErrorKind::AlreadyExists => return Ok(false),
        Err(error) => return Err(error),
    };

    let copy_result = (|| {
        let mut buffer = [0u8; 128 * 1024];
        loop {
            let read = input.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            output.write_all(&buffer[..read])?;
        }
        output.sync_all()
    })();

    if let Err(error) = copy_result {
        let _ = fs::remove_file(destination);
        return Err(error);
    }

    Ok(true)
}

fn conflict_safe_name(file_name: &OsStr, attempt: usize) -> OsString {
    if attempt == 0 {
        return file_name.to_os_string();
    }

    let path = Path::new(file_name);
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("image");
    let extension = path.extension().and_then(|extension| extension.to_str());

    match extension {
        Some(extension) => format!("{stem}_{attempt}.{extension}").into(),
        None => format!("{stem}_{attempt}").into(),
    }
}

fn worker_count(max_workers: usize, total: usize) -> usize {
    if total == 0 {
        return 0;
    }

    let available = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    let capped = max_workers.max(1).min(available).min(4);
    capped.min(total)
}

fn set_active(active_files: &Mutex<Vec<Option<String>>>, slot: usize, file_name: String) {
    if let Ok(mut active_files) = active_files.lock()
        && let Some(active) = active_files.get_mut(slot)
    {
        *active = Some(file_name);
    }
}

fn clear_active(active_files: &Mutex<Vec<Option<String>>>, slot: usize) {
    if let Ok(mut active_files) = active_files.lock()
        && let Some(active) = active_files.get_mut(slot)
    {
        *active = None;
    }
}

fn snapshot(
    total: usize,
    counters: &Counters,
    active_files: &Mutex<Vec<Option<String>>>,
    last_message: String,
    last_result: Option<AutonomousNumberingItemResult>,
) -> AutonomousNumberingProgress {
    let active_files = active_files
        .lock()
        .map(|files| files.iter().filter_map(Clone::clone).collect())
        .unwrap_or_default();

    AutonomousNumberingProgress {
        completed: counters.completed.load(Ordering::Relaxed),
        total,
        assigned: counters.assigned.load(Ordering::Relaxed),
        needs_review: counters.needs_review.load(Ordering::Relaxed),
        already_completed: counters.already_completed.load(Ordering::Relaxed),
        skipped_missing: counters.skipped_missing.load(Ordering::Relaxed),
        failed: counters.failed.load(Ordering::Relaxed),
        active_files,
        last_message,
        last_result,
    }
}

fn apply_item_result(counters: &Counters, item: &AutonomousNumberingItemResult) {
    match item.status {
        AutonomousNumberingItemStatus::Assigned => {
            counters.assigned.fetch_add(1, Ordering::Relaxed);
        }
        AutonomousNumberingItemStatus::NeedsReview => {
            counters.needs_review.fetch_add(1, Ordering::Relaxed);
        }
        AutonomousNumberingItemStatus::AlreadyCompleted => {
            counters.already_completed.fetch_add(1, Ordering::Relaxed);
        }
        AutonomousNumberingItemStatus::SkippedMissing => {
            counters.skipped_missing.fetch_add(1, Ordering::Relaxed);
        }
        AutonomousNumberingItemStatus::Failed => {
            counters.failed.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn summary(total: usize, counters: &Counters, cancelled: bool) -> AutonomousNumberingSummary {
    let assigned = counters.assigned.load(Ordering::Relaxed);
    let needs_review = counters.needs_review.load(Ordering::Relaxed);
    let already_completed = counters.already_completed.load(Ordering::Relaxed);
    let skipped_missing = counters.skipped_missing.load(Ordering::Relaxed);
    let failed = counters.failed.load(Ordering::Relaxed);
    let processed = assigned + needs_review + already_completed + skipped_missing + failed;

    AutonomousNumberingSummary {
        discovered: total,
        processed,
        assigned,
        needs_review,
        already_completed,
        skipped_missing,
        failed,
        cancelled,
    }
}

fn item_message(item: &AutonomousNumberingItemResult) -> String {
    match item.status {
        AutonomousNumberingItemStatus::Assigned => {
            let number = item.number.as_deref().unwrap_or("?");
            format!("Assigned {} to {number}", item.file_name)
        }
        AutonomousNumberingItemStatus::NeedsReview => {
            format!("Left {} for review", item.file_name)
        }
        AutonomousNumberingItemStatus::AlreadyCompleted => {
            format!("Skipped already-numbered file {}", item.file_name)
        }
        AutonomousNumberingItemStatus::SkippedMissing => {
            format!("Skipped missing file {}", item.file_name)
        }
        AutonomousNumberingItemStatus::Failed => {
            let error = item.error.as_deref().unwrap_or("unknown error");
            format!("Failed {}: {error}", item.file_name)
        }
    }
}

fn path_file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("?")
        .to_string()
}

fn digits_only(text: &str) -> String {
    text.chars().filter(|ch| ch.is_ascii_digit()).collect()
}

fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            let extension = extension.to_ascii_lowercase();
            matches!(extension.as_str(), "jpg" | "jpeg" | "png")
        })
        .unwrap_or(false)
}
