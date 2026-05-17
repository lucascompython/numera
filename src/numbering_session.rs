use rapidhash::fast::RapidHashSet as HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, Default)]
pub struct NumberingSession {
    inner: Arc<Mutex<NumberingSessionInner>>,
}

#[derive(Debug, Default)]
struct NumberingSessionInner {
    source_dir: Option<PathBuf>,
    source_revision: u64,
    completed_revision: u64,
    completed_files: Vec<CompletedFile>,
    completed_lookup: HashSet<PathBuf>,
    autonomous_progress_revision: u64,
    autonomous_progress: AutonomousProgressSnapshot,
    manual_open_revision: u64,
    manual_open_path: Option<PathBuf>,
}

#[derive(Clone, Debug)]
struct CompletedFile {
    revision: u64,
    path: PathBuf,
}

#[derive(Clone, Debug, Default)]
pub struct NumberingSessionChanges {
    pub source_dir: Option<PathBuf>,
    pub source_revision: u64,
    pub completed_revision: u64,
    pub autonomous_progress_revision: u64,
    pub manual_open_revision: u64,
    pub source_changed: bool,
    pub completed_paths: Vec<PathBuf>,
    pub autonomous_progress_changed: bool,
    pub autonomous_progress: AutonomousProgressSnapshot,
    pub manual_open_path: Option<PathBuf>,
}

#[derive(Clone, Debug, Default)]
pub struct AutonomousProgressSnapshot {
    pub running: bool,
    pub completed: usize,
    pub total: usize,
    pub assigned: usize,
    pub needs_review: usize,
    pub already_completed: usize,
    pub skipped_missing: usize,
    pub failed: usize,
    pub active_files: Vec<String>,
    pub last_message: String,
}

impl NumberingSession {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn source_dir(&self) -> Option<PathBuf> {
        self.inner
            .lock()
            .ok()
            .and_then(|inner| inner.source_dir.clone())
    }

    pub fn set_source_dir(&self, source_dir: PathBuf) -> NumberingSessionChanges {
        let Ok(mut inner) = self.inner.lock() else {
            return NumberingSessionChanges::default();
        };

        if inner.source_dir.as_ref() != Some(&source_dir) {
            inner.source_dir = Some(source_dir);
            inner.source_revision = inner.source_revision.wrapping_add(1);
            inner.completed_revision = inner.completed_revision.wrapping_add(1);
            inner.completed_files.clear();
            inner.completed_lookup.clear();
        }

        NumberingSessionChanges {
            source_dir: inner.source_dir.clone(),
            source_revision: inner.source_revision,
            completed_revision: inner.completed_revision,
            autonomous_progress_revision: inner.autonomous_progress_revision,
            manual_open_revision: inner.manual_open_revision,
            source_changed: true,
            completed_paths: Vec::new(),
            autonomous_progress_changed: true,
            autonomous_progress: inner.autonomous_progress.clone(),
            manual_open_path: None,
        }
    }

    pub fn mark_completed(&self, path: PathBuf) {
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };

        if !inner.completed_lookup.insert(path.clone()) {
            return;
        }

        inner.completed_revision = inner.completed_revision.wrapping_add(1);
        let revision = inner.completed_revision;
        inner.completed_files.push(CompletedFile { revision, path });
    }

    pub fn is_completed(&self, path: &Path) -> bool {
        self.inner
            .lock()
            .map(|inner| inner.completed_lookup.contains(path))
            .unwrap_or(false)
    }

    pub fn changes_since(
        &self,
        source_revision: u64,
        completed_revision: u64,
        autonomous_progress_revision: u64,
        manual_open_revision: u64,
    ) -> NumberingSessionChanges {
        let Ok(inner) = self.inner.lock() else {
            return NumberingSessionChanges::default();
        };

        let source_changed = inner.source_revision != source_revision;
        let completed_paths = if source_changed {
            Vec::new()
        } else {
            inner
                .completed_files
                .iter()
                .filter(|completed| completed.revision > completed_revision)
                .map(|completed| completed.path.clone())
                .collect()
        };
        let autonomous_progress_changed =
            inner.autonomous_progress_revision != autonomous_progress_revision;
        let manual_open_path = (inner.manual_open_revision != manual_open_revision)
            .then(|| inner.manual_open_path.clone())
            .flatten();

        NumberingSessionChanges {
            source_dir: inner.source_dir.clone(),
            source_revision: inner.source_revision,
            completed_revision: inner.completed_revision,
            autonomous_progress_revision: inner.autonomous_progress_revision,
            manual_open_revision: inner.manual_open_revision,
            source_changed,
            completed_paths,
            autonomous_progress_changed,
            autonomous_progress: inner.autonomous_progress.clone(),
            manual_open_path,
        }
    }

    pub fn publish_autonomous_progress(&self, progress: AutonomousProgressSnapshot) {
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };

        inner.autonomous_progress = progress;
        inner.autonomous_progress_revision = inner.autonomous_progress_revision.wrapping_add(1);
    }

    pub fn request_manual_open(&self, path: PathBuf) {
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };

        inner.manual_open_path = Some(path);
        inner.manual_open_revision = inner.manual_open_revision.wrapping_add(1);
    }

    pub fn manual_open_revision(&self) -> u64 {
        self.inner
            .lock()
            .map(|inner| inner.manual_open_revision)
            .unwrap_or(0)
    }
}
