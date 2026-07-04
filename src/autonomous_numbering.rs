use gpui::prelude::FluentBuilder;
use gpui::*;
use gpui_component::button::{Button, ButtonVariants as _};
use gpui_component::progress::Progress;
use gpui_component::scroll::ScrollableElement;
use gpui_component::{ActiveTheme, Disableable, Sizable, h_flex, v_flex};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::numbering_session::{
    AutonomousProgressSnapshot, NumberingSession, NumberingSessionChanges,
};
use crate::processing::autonomous_numbering::{
    AutonomousNumberingConfig, AutonomousNumberingItemResult, AutonomousNumberingItemStatus,
    AutonomousNumberingProgress, AutonomousNumberingSummary,
};

const MIN_AUTO_CONFIDENCE: f32 = 0.85;
const MAX_AUTO_WORKERS: usize = 8;
const MAX_RECENT_RESULTS: usize = 12;

#[derive(Clone, Debug, Default)]
struct AutonomousUiProgress {
    completed: usize,
    total: usize,
    assigned: usize,
    needs_review: usize,
    already_completed: usize,
    skipped_missing: usize,
    failed: usize,
    active_files: Vec<String>,
    last_message: String,
    recent_results: Vec<AutonomousNumberingItemResult>,
    review_results: Vec<AutonomousNumberingItemResult>,
}

pub struct AutonomousNumberingWindow {
    source_dir: Option<PathBuf>,
    is_running: bool,
    progress: AutonomousUiProgress,
    started_at: Option<Instant>,
    finished_at: Option<Instant>,
    summary: Option<AutonomousNumberingSummary>,
    status_message: String,
    shared_progress: Option<Arc<Mutex<AutonomousUiProgress>>>,
    cancel_flag: Option<Arc<AtomicBool>>,
    session: NumberingSession,
    session_source_revision: u64,
    session_completed_revision: u64,
}

impl AutonomousNumberingWindow {
    pub fn new(_window: &mut Window, cx: &mut Context<Self>, session: NumberingSession) -> Self {
        let initial_source_dir = session.source_dir();
        let status_message = if let Some(source_dir) = initial_source_dir.as_ref() {
            format!("Source: {}", source_dir.display())
        } else {
            "Select a source folder".to_string()
        };

        let mut this = Self {
            source_dir: initial_source_dir,
            is_running: false,
            progress: AutonomousUiProgress::default(),
            started_at: None,
            finished_at: None,
            summary: None,
            status_message,
            shared_progress: None,
            cancel_flag: None,
            session,
            session_source_revision: 0,
            session_completed_revision: 0,
        };
        this.start_session_sync(cx);
        this
    }

    fn choose_folder(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.is_running {
            return;
        }

        let entity = cx.entity().clone();
        cx.spawn_in(window, async move |_this, cx| {
            let handle = rfd::AsyncFileDialog::new()
                .set_title("Select image folder for autonomous numbering")
                .pick_folder()
                .await;

            if let Some(folder) = handle {
                let dir = folder.path().to_path_buf();
                entity.update(cx, |this, cx| {
                    let changes = this.session.set_source_dir(dir);
                    this.apply_session_changes(changes);
                    cx.notify();
                });
            }
        })
        .detach();
    }

    fn start(&mut self, cx: &mut Context<Self>) {
        if self.is_running {
            return;
        }

        let Some(source_dir) = self.source_dir.clone() else {
            self.status_message = "Select a source folder first".to_string();
            cx.notify();
            return;
        };

        self.is_running = true;
        self.started_at = Some(Instant::now());
        self.finished_at = None;
        self.summary = None;
        self.progress = AutonomousUiProgress {
            last_message: "Scanning source folder...".to_string(),
            ..AutonomousUiProgress::default()
        };
        self.status_message = "Running autonomous numbering...".to_string();

        let shared_progress = Arc::new(Mutex::new(self.progress.clone()));
        let done_flag = Arc::new(AtomicBool::new(false));
        let cancel_flag = Arc::new(AtomicBool::new(false));
        self.session
            .publish_autonomous_progress(progress_snapshot(&self.progress, true));

        self.shared_progress = Some(shared_progress.clone());
        self.cancel_flag = Some(cancel_flag.clone());
        cx.notify();

        let executor = cx.background_executor().clone();
        let timer_executor = executor.clone();
        let shared_for_timer = shared_progress.clone();
        let done_for_timer = done_flag.clone();

        cx.spawn(async move |this, cx| {
            loop {
                timer_executor.timer(Duration::from_millis(100)).await;
                let progress = clone_shared_progress(&shared_for_timer);
                let done = done_for_timer.load(Ordering::Relaxed);

                _ = this.update(cx, |this, cx| {
                    if this.is_running || done {
                        this.progress = progress;
                        cx.notify();
                    }
                });

                if done {
                    break;
                }
            }
        })
        .detach();

        let shared_for_worker = shared_progress.clone();
        let session_for_worker_progress = self.session.clone();
        let done_for_worker = done_flag.clone();
        let worker_cancel_flag = cancel_flag.clone();
        let config = AutonomousNumberingConfig {
            source_dir,
            min_confidence: MIN_AUTO_CONFIDENCE,
            max_workers: MAX_AUTO_WORKERS,
            cancel_flag,
            session: self.session.clone(),
        };

        cx.spawn(async move |this, cx| {
            let summary = executor
                .spawn(async move {
                    crate::processing::autonomous_numbering::run_autonomous_numbering(
                        config,
                        move |progress| {
                            merge_progress(&shared_for_worker, progress.clone());
                            let ui_progress = clone_shared_progress(&shared_for_worker);
                            session_for_worker_progress
                                .publish_autonomous_progress(progress_snapshot(&ui_progress, true));
                        },
                    )
                })
                .await;

            done_for_worker.store(true, Ordering::Relaxed);

            _ = this.update(cx, |this, cx| {
                this.is_running = false;
                this.finished_at = Some(Instant::now());
                if let Some(shared_progress) = this.shared_progress.as_ref() {
                    this.progress = clone_shared_progress(shared_progress);
                }
                this.session
                    .publish_autonomous_progress(progress_snapshot(&this.progress, false));
                this.source_dir = this.session.source_dir();
                this.summary = Some(summary.clone());
                this.cancel_flag = None;
                this.status_message = final_status_message(&summary, worker_cancel_flag);
                cx.notify();
            });
        })
        .detach();
    }

    fn stop(&mut self, cx: &mut Context<Self>) {
        if let Some(cancel_flag) = self.cancel_flag.as_ref() {
            cancel_flag.store(true, Ordering::Relaxed);
            self.status_message = "Stopping after current OCR jobs finish...".to_string();
            cx.notify();
        }
    }

    fn start_session_sync(&mut self, cx: &mut Context<Self>) {
        let session = self.session.clone();
        let timer = cx.background_executor().clone();

        cx.spawn(async move |this, cx| {
            loop {
                timer.timer(Duration::from_millis(125)).await;
                let changes = this
                    .update(cx, |this, _| {
                        session.changes_since(
                            this.session_source_revision,
                            this.session_completed_revision,
                            0,
                            0,
                        )
                    })
                    .unwrap_or_default();

                if !changes.source_changed && changes.completed_paths.is_empty() {
                    continue;
                }

                _ = this.update(cx, |this, cx| {
                    this.apply_session_changes(changes);
                    cx.notify();
                });
            }
        })
        .detach();
    }

    fn apply_session_changes(&mut self, changes: NumberingSessionChanges) {
        self.session_source_revision = changes.source_revision;
        self.session_completed_revision = changes.completed_revision;

        if !changes.source_changed {
            return;
        }

        if self.is_running {
            if let Some(source_dir) = changes.source_dir {
                self.status_message = format!(
                    "Source changed to {}; current run will finish first",
                    source_dir.display()
                );
            }
            return;
        }

        self.source_dir = changes.source_dir;
        self.progress = AutonomousUiProgress::default();
        self.summary = None;
        self.started_at = None;
        self.finished_at = None;
        self.status_message = if let Some(source_dir) = self.source_dir.as_ref() {
            format!("Source: {}", source_dir.display())
        } else {
            "Select a source folder".to_string()
        };
    }

    fn progress_fraction(&self) -> f32 {
        if self.progress.total == 0 {
            0.0
        } else {
            (self.progress.completed as f32 / self.progress.total as f32).clamp(0.0, 1.0)
        }
    }

    fn elapsed(&self) -> Option<Duration> {
        let started_at = self.started_at?;
        Some(
            self.finished_at
                .unwrap_or_else(Instant::now)
                .saturating_duration_since(started_at),
        )
    }

    fn eta(&self) -> Option<Duration> {
        if !self.is_running || self.progress.completed == 0 {
            return None;
        }

        let elapsed = self.elapsed()?;
        let remaining = self.progress.total.saturating_sub(self.progress.completed);
        if remaining == 0 {
            return Some(Duration::ZERO);
        }

        let seconds_per_image = elapsed.as_secs_f64() / self.progress.completed as f64;
        Some(Duration::from_secs_f64(
            seconds_per_image * remaining as f64,
        ))
    }

    fn render_toolbar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let entity = cx.entity().clone();
        let source_label = self
            .source_dir
            .as_ref()
            .and_then(|path| path.file_name())
            .and_then(|name| name.to_str())
            .unwrap_or("Not set")
            .to_string();

        let start_button = if self.is_running {
            Button::new("auto-start")
                .label("Running")
                .primary()
                .small()
                .disabled(true)
        } else {
            Button::new("auto-start")
                .label("Start")
                .primary()
                .small()
                .disabled(self.source_dir.is_none())
                .on_click({
                    let entity = entity.clone();
                    move |_, _, cx| {
                        entity.update(cx, |this, cx| this.start(cx));
                    }
                })
        };

        h_flex()
            .px_3()
            .py_2()
            .gap_3()
            .items_center()
            .bg(cx.theme().background)
            .border_b_1()
            .border_color(cx.theme().border)
            .child(
                div()
                    .text_base()
                    .font_weight(FontWeight::SEMIBOLD)
                    .child("Autonomous Numbering"),
            )
            .child(div().w(px(1.)).h(px(20.)).bg(cx.theme().border))
            .child(
                div()
                    .text_sm()
                    .text_color(cx.theme().muted_foreground)
                    .child(format!("Source: {source_label}")),
            )
            .child(div().flex_1())
            .child(
                Button::new("auto-folder")
                    .label("Browse")
                    .small()
                    .disabled(self.is_running)
                    .on_click({
                        let entity = entity.clone();
                        move |_, window, cx| {
                            entity.update(cx, |this, cx| this.choose_folder(window, cx));
                        }
                    }),
            )
            .child(start_button)
            .child(
                Button::new("auto-stop")
                    .label("Stop")
                    .small()
                    .disabled(!self.is_running)
                    .on_click({
                        let entity = entity.clone();
                        move |_, _, cx| {
                            entity.update(cx, |this, cx| this.stop(cx));
                        }
                    }),
            )
    }

    fn render_stats(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let stat = |label: &str, value: String| -> AnyElement {
            v_flex()
                .gap_1()
                .child(
                    div()
                        .text_xs()
                        .text_color(cx.theme().muted_foreground)
                        .child(label.to_string()),
                )
                .child(
                    div()
                        .text_lg()
                        .font_weight(FontWeight::SEMIBOLD)
                        .child(value),
                )
                .into_any_element()
        };

        let elapsed = self
            .elapsed()
            .map(format_duration)
            .unwrap_or_else(|| "--".to_string());
        let eta = self
            .eta()
            .map(format_duration)
            .unwrap_or_else(|| "--".to_string());

        v_flex().gap_3().child(
            h_flex()
                .gap_8()
                .child(stat(
                    "Progress",
                    format!("{}/{}", self.progress.completed, self.progress.total),
                ))
                .child(stat("Assigned", self.progress.assigned.to_string()))
                .child(stat("Needs Review", self.progress.needs_review.to_string()))
                .child(stat(
                    "Already Done",
                    self.progress.already_completed.to_string(),
                ))
                .child(stat("Missing", self.progress.skipped_missing.to_string()))
                .child(stat("Failed", self.progress.failed.to_string()))
                .child(stat("Elapsed", elapsed))
                .child(stat("ETA", eta)),
        )
    }

    fn render_active_files(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let active_content = if self.progress.active_files.is_empty() {
            div()
                .text_sm()
                .text_color(cx.theme().muted_foreground)
                .child("Idle")
                .into_any_element()
        } else {
            v_flex()
                .gap_1()
                .children(self.progress.active_files.iter().map(|name| {
                    div()
                        .text_sm()
                        .text_color(cx.theme().foreground)
                        .child(name.clone())
                }))
                .into_any_element()
        };

        v_flex()
            .gap_2()
            .child(
                div()
                    .text_sm()
                    .font_weight(FontWeight::MEDIUM)
                    .child("Current Images"),
            )
            .child(active_content)
    }

    fn render_recent_results(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let entity = cx.entity().clone();
        let rows =
            if self.progress.recent_results.is_empty() {
                div()
                    .text_sm()
                    .text_color(cx.theme().muted_foreground)
                    .child("No results yet")
                    .into_any_element()
            } else {
                v_flex()
                    .gap_1()
                    .children(self.progress.recent_results.iter().enumerate().map(
                        |(index, result)| {
                            let entity = entity.clone();
                            let (label, color) = result_label(result.status);
                            let confidence = result
                                .confidence
                                .map(|value| format!(" {:.0}%", value * 100.0))
                                .unwrap_or_default();
                            let number = result
                                .number
                                .as_ref()
                                .map(|number| format!(" {number}"))
                                .unwrap_or_default();
                            let suffix = result
                                .error
                                .as_ref()
                                .map(|error| format!(" - {error}"))
                                .unwrap_or_default();
                            let destination = result
                                .destination
                                .as_ref()
                                .and_then(|path| path.parent())
                                .and_then(|path| path.file_name())
                                .and_then(|name| name.to_str())
                                .map(|folder| format!(" -> {folder}"))
                                .unwrap_or_default();
                            let review_path = (result.status
                                == AutonomousNumberingItemStatus::NeedsReview)
                                .then(|| result.destination.clone())
                                .flatten();

                            h_flex()
                                .id(("auto-result", index))
                                .gap_2()
                                .items_center()
                                .text_sm()
                                .when(review_path.is_some(), |el| el.cursor_pointer())
                                .on_click(move |_, _, cx| {
                                    if let Some(path) = review_path.clone() {
                                        entity.update(cx, |this, cx| {
                                            this.session.request_manual_open(path);
                                            this.status_message =
                                                "Review image opened in the main window"
                                                    .to_string();
                                            cx.notify();
                                        });
                                    }
                                })
                                .child(div().w(px(96.)).text_color(color).child(label.to_string()))
                                .child(
                                    div()
                                        .flex_1()
                                        .min_w(px(0.))
                                        .truncate()
                                        .text_color(cx.theme().foreground)
                                        .child(result.file_name.clone()),
                                )
                                .child(
                                    div()
                                        .max_w(px(280.))
                                        .truncate()
                                        .text_color(cx.theme().muted_foreground)
                                        .child(format!(
                                            "{number}{confidence}{destination}{suffix}"
                                        )),
                                )
                        },
                    ))
                    .into_any_element()
            };

        v_flex()
            .gap_2()
            .flex_1()
            .min_h(px(0.))
            .child(
                div()
                    .text_sm()
                    .font_weight(FontWeight::MEDIUM)
                    .child("Recent Results"),
            )
            .child(
                div()
                    .flex_1()
                    .min_h(px(0.))
                    .overflow_y_scrollbar()
                    .child(rows),
            )
    }

    fn render_review_queue(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let entity = cx.entity().clone();
        let rows =
            if self.progress.review_results.is_empty() {
                div()
                    .text_sm()
                    .text_color(cx.theme().muted_foreground)
                    .child("No review images")
                    .into_any_element()
            } else {
                v_flex()
                    .gap_1()
                    .children(self.progress.review_results.iter().enumerate().map(
                        |(index, result)| {
                            let entity = entity.clone();
                            let path = result.destination.clone();
                            let confidence = result
                                .confidence
                                .map(|value| format!(" {:.0}%", value * 100.0))
                                .unwrap_or_default();
                            let number = result
                                .number
                                .as_ref()
                                .map(|number| format!(" {number}"))
                                .unwrap_or_default();

                            h_flex()
                                .id(("review-result", index))
                                .gap_2()
                                .items_center()
                                .text_sm()
                                .cursor_pointer()
                                .on_click(move |_, _, cx| {
                                    if let Some(path) = path.clone() {
                                        entity.update(cx, |this, cx| {
                                            this.session.request_manual_open(path);
                                            this.status_message =
                                                "Review image opened in the main window"
                                                    .to_string();
                                            cx.notify();
                                        });
                                    }
                                })
                                .child(
                                    div()
                                        .flex_1()
                                        .min_w(px(0.))
                                        .truncate()
                                        .text_color(cx.theme().foreground)
                                        .child(result.file_name.clone()),
                                )
                                .child(
                                    div()
                                        .max_w(px(180.))
                                        .truncate()
                                        .text_color(cx.theme().muted_foreground)
                                        .child(format!("{number}{confidence}")),
                                )
                        },
                    ))
                    .into_any_element()
            };

        v_flex()
            .gap_2()
            .h(px(140.))
            .min_h(px(100.))
            .child(
                div()
                    .text_sm()
                    .font_weight(FontWeight::MEDIUM)
                    .child("Review Images"),
            )
            .child(
                div()
                    .flex_1()
                    .min_h(px(0.))
                    .overflow_y_scrollbar()
                    .child(rows),
            )
    }

    fn render_status(&self, cx: &mut Context<Self>) -> impl IntoElement {
        h_flex()
            .px_3()
            .py_1()
            .gap_3()
            .items_center()
            .bg(cx.theme().muted)
            .border_t_1()
            .border_color(cx.theme().border)
            .child(
                div()
                    .text_xs()
                    .text_color(cx.theme().muted_foreground)
                    .child(self.status_message.clone()),
            )
            .child(div().flex_1())
            .child(
                div()
                    .text_xs()
                    .text_color(cx.theme().muted_foreground)
                    .child(format!(
                        "High-confidence threshold: {:.0}%",
                        MIN_AUTO_CONFIDENCE * 100.0
                    )),
            )
    }
}

impl Render for AutonomousNumberingWindow {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let progress_value = self.progress_fraction() * 100.0;

        v_flex()
            .size_full()
            .bg(cx.theme().background)
            .text_color(cx.theme().foreground)
            .child(self.render_toolbar(cx))
            .child(
                v_flex()
                    .flex_1()
                    .min_h(px(0.))
                    .gap_5()
                    .p_5()
                    .child(Progress::new("auto-progress").value(progress_value))
                    .child(
                        div()
                            .text_sm()
                            .text_color(cx.theme().muted_foreground)
                            .child(self.progress.last_message.clone()),
                    )
                    .child(self.render_stats(cx))
                    .child(div().h(px(1.)).bg(cx.theme().border))
                    .child(self.render_active_files(cx))
                    .child(div().h(px(1.)).bg(cx.theme().border))
                    .child(self.render_review_queue(cx))
                    .child(div().h(px(1.)).bg(cx.theme().border))
                    .child(self.render_recent_results(cx)),
            )
            .child(self.render_status(cx))
    }
}

fn merge_progress(
    shared: &Arc<Mutex<AutonomousUiProgress>>,
    progress: AutonomousNumberingProgress,
) {
    let Ok(mut shared) = shared.lock() else {
        return;
    };

    shared.completed = progress.completed;
    shared.total = progress.total;
    shared.assigned = progress.assigned;
    shared.needs_review = progress.needs_review;
    shared.already_completed = progress.already_completed;
    shared.skipped_missing = progress.skipped_missing;
    shared.failed = progress.failed;
    shared.active_files = progress.active_files;
    shared.last_message = progress.last_message;

    if let Some(result) = progress.last_result {
        if result.status == AutonomousNumberingItemStatus::NeedsReview
            && result.destination.is_some()
        {
            shared.review_results.insert(0, result.clone());
        }
        shared.recent_results.insert(0, result);
        shared.recent_results.truncate(MAX_RECENT_RESULTS);
    }
}

fn clone_shared_progress(shared: &Arc<Mutex<AutonomousUiProgress>>) -> AutonomousUiProgress {
    shared
        .lock()
        .map(|progress| progress.clone())
        .unwrap_or_default()
}

fn progress_snapshot(progress: &AutonomousUiProgress, running: bool) -> AutonomousProgressSnapshot {
    AutonomousProgressSnapshot {
        running,
        completed: progress.completed,
        total: progress.total,
        assigned: progress.assigned,
        needs_review: progress.needs_review,
        already_completed: progress.already_completed,
        skipped_missing: progress.skipped_missing,
        failed: progress.failed,
        active_files: progress.active_files.clone(),
        last_message: progress.last_message.clone(),
    }
}

fn final_status_message(
    summary: &AutonomousNumberingSummary,
    cancel_flag: Arc<AtomicBool>,
) -> String {
    if summary.cancelled || cancel_flag.load(Ordering::Relaxed) {
        return format!(
            "Stopped - {} of {} processed, {} assigned, {} left for review, {} already done, {} failed",
            summary.processed,
            summary.discovered,
            summary.assigned,
            summary.needs_review,
            summary.already_completed,
            summary.failed
        );
    }

    format!(
        "Done - {} of {} processed, {} assigned, {} left for review, {} already done, {} missing, {} failed",
        summary.processed,
        summary.discovered,
        summary.assigned,
        summary.needs_review,
        summary.already_completed,
        summary.skipped_missing,
        summary.failed
    )
}

fn result_label(status: AutonomousNumberingItemStatus) -> (&'static str, Hsla) {
    match status {
        AutonomousNumberingItemStatus::Assigned => ("Assigned", hsla(0.33, 0.75, 0.45, 1.0)),
        AutonomousNumberingItemStatus::NeedsReview => ("Review", hsla(0.12, 0.85, 0.5, 1.0)),
        AutonomousNumberingItemStatus::AlreadyCompleted => ("Done", hsla(0.58, 0.45, 0.62, 1.0)),
        AutonomousNumberingItemStatus::SkippedMissing => ("Missing", hsla(0.58, 0.55, 0.62, 1.0)),
        AutonomousNumberingItemStatus::Failed => ("Failed", hsla(0.0, 0.8, 0.58, 1.0)),
    }
}

fn format_duration(duration: Duration) -> String {
    let total_seconds = duration.as_secs();
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;

    if hours > 0 {
        format!("{hours}h {minutes}m")
    } else if minutes > 0 {
        format!("{minutes}m {seconds}s")
    } else {
        format!("{seconds}s")
    }
}
