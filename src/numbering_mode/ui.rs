//! Numbering mode UI component.

use gpui::*;
use gpui_component::button::{Button, ButtonVariants as _};
use gpui_component::input::{Input, InputEvent, InputState};
use gpui_component::{ActiveTheme, ElementExt, Sizable, h_flex, v_flex};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::state::NumberingState;
use crate::numbering_session::{
    AutonomousProgressSnapshot, NumberingSession, NumberingSessionChanges,
};
use crate::processing::image_cache::ImageCache;

/// NumberingMode component that handles the image numbering workflow.
pub struct NumberingMode {
    state: NumberingState,
    input_state: Entity<InputState>,
    image_cache: Arc<ImageCache>,
    preview_image: Option<Arc<RenderImage>>,
    preview_dimensions: Option<(u32, u32)>,
    image_view_size: Option<(f32, f32)>,
    preview_version: usize,
    session: NumberingSession,
    session_source_revision: u64,
    session_completed_revision: u64,
    session_autonomous_progress_revision: u64,
    session_manual_open_revision: u64,
    autonomous_progress: AutonomousProgressSnapshot,
    _subscriptions: Vec<Subscription>,
}

impl NumberingMode {
    pub fn new(
        window: &mut Window,
        cx: &mut Context<Self>,
        image_cache: Arc<ImageCache>,
        session: NumberingSession,
    ) -> Self {
        let input_state =
            cx.new(|cx| InputState::new(window, cx).placeholder("Type motorcycle number..."));

        let mut subs = Vec::new();

        subs.push(cx.subscribe_in(
            &input_state,
            window,
            |this, _state, ev: &InputEvent, window, cx| {
                if let InputEvent::PressEnter { .. } = ev {
                    this.confirm_and_advance(window, cx);
                }
            },
        ));

        let mut this = Self {
            state: NumberingState::new(),
            input_state,
            image_cache,
            preview_image: None,
            preview_dimensions: None,
            image_view_size: None,
            preview_version: 0,
            session,
            session_source_revision: 0,
            session_completed_revision: 0,
            session_autonomous_progress_revision: 0,
            session_manual_open_revision: 0,
            autonomous_progress: AutonomousProgressSnapshot::default(),
            _subscriptions: subs,
        };
        this.start_session_sync(cx);
        this
    }

    pub fn open_folder(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let entity = cx.entity().clone();
        cx.spawn_in(window, async move |_this, cx| {
            let handle = rfd::AsyncFileDialog::new()
                .set_title("Select image folder for numbering")
                .pick_folder()
                .await;

            if let Some(folder) = handle {
                let dir = folder.path().to_path_buf();

                entity.update(cx, |this, cx| {
                    let changes = this.session.set_source_dir(dir.clone());
                    this.session_source_revision = changes.source_revision;
                    this.session_completed_revision = changes.completed_revision;
                    this.load_folder(dir, cx);
                    cx.notify();
                });
            }
        })
        .detach();
    }

    fn open_sticker_template(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let entity = cx.entity().clone();
        cx.spawn_in(window, async move |_this, cx| {
            let handle = rfd::AsyncFileDialog::new()
                .set_title("Select event sticker template")
                .add_filter("Images", &["png", "jpg", "jpeg"])
                .pick_file()
                .await;

            if let Some(file) = handle {
                let path = file.path().to_path_buf();
                let load_result = crate::ocr::set_sticker_template(&path);

                entity.update(cx, |this, cx| {
                    match load_result {
                        Ok(()) => {
                            let label = path
                                .file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or("template");
                            this.state.status_message = format!("Sticker template loaded: {label}");
                            this.load_current_image(cx);
                        }
                        Err(err) => {
                            this.state.status_message = format!("Template load failed: {err}");
                        }
                    }
                    cx.notify();
                });
            }
        })
        .detach();
    }

    fn clear_sticker_template(&mut self, cx: &mut Context<Self>) {
        crate::ocr::clear_sticker_template();
        self.state.status_message = "Sticker template cleared".into();
        self.load_current_image(cx);
        cx.notify();
    }

    fn open_autonomous_window(&mut self, cx: &mut Context<Self>) {
        let session = self.session.clone();
        let opts = gpui::WindowOptions {
            titlebar: Some(gpui::TitlebarOptions {
                title: Some("Autonomous Numbering".into()),
                ..Default::default()
            }),
            ..Default::default()
        };

        match cx.open_window(opts, move |window, cx| {
            let view = cx.new(|cx| {
                crate::autonomous_numbering::AutonomousNumberingWindow::new(
                    window,
                    cx,
                    session.clone(),
                )
            });
            cx.new(|cx| gpui_component::Root::new(view, window, cx))
        }) {
            Ok(_) => {
                self.state.status_message = "Autonomous numbering window opened".to_string();
            }
            Err(error) => {
                self.state.status_message =
                    format!("Failed to open autonomous numbering window: {error}");
            }
        }
        cx.notify();
    }

    fn start_session_sync(&mut self, cx: &mut Context<Self>) {
        let session = self.session.clone();
        let timer = cx.background_executor().clone();

        cx.spawn(async move |this, cx| {
            loop {
                timer.timer(std::time::Duration::from_millis(125)).await;
                let changes = this
                    .update(cx, |this, _| {
                        session.changes_since(
                            this.session_source_revision,
                            this.session_completed_revision,
                            this.session_autonomous_progress_revision,
                            this.session_manual_open_revision,
                        )
                    })
                    .unwrap_or_default();

                if !changes.source_changed
                    && changes.completed_paths.is_empty()
                    && !changes.autonomous_progress_changed
                    && changes.manual_open_path.is_none()
                {
                    continue;
                }

                _ = this.update(cx, |this, cx| {
                    this.apply_session_changes(changes, cx);
                    cx.notify();
                });
            }
        })
        .detach();
    }

    fn apply_session_changes(&mut self, changes: NumberingSessionChanges, cx: &mut Context<Self>) {
        if changes.source_changed {
            self.session_source_revision = changes.source_revision;
            self.session_completed_revision = changes.completed_revision;
            self.session_autonomous_progress_revision = changes.autonomous_progress_revision;
            self.session_manual_open_revision = changes.manual_open_revision;
            self.autonomous_progress = changes.autonomous_progress;
            if let Some(source_dir) = changes.source_dir {
                self.load_folder(source_dir, cx);
            }
            return;
        }

        if changes.autonomous_progress_changed {
            self.session_autonomous_progress_revision = changes.autonomous_progress_revision;
            self.autonomous_progress = changes.autonomous_progress;
        }
        if !changes.completed_paths.is_empty() {
            self.remove_completed_paths(&changes.completed_paths, cx);
        }
        if let Some(path) = changes.manual_open_path {
            self.open_review_image(path, cx);
        }
        self.session_completed_revision = changes.completed_revision;
        self.session_manual_open_revision = changes.manual_open_revision;
    }

    fn load_folder(&mut self, dir: PathBuf, cx: &mut Context<Self>) {
        match crate::processing::autonomous_numbering::discover_numbering_images(&dir) {
            Ok(images) => {
                self.state.source_folder = Some(dir);
                self.state.image_paths = images;
                self.state.current_index = 0;
                self.state.undo_stack.clear();
                self.state.input_buffer.clear();
                self.preview_image = None;
                self.preview_dimensions = None;
                self.state.status_message =
                    format!("Loaded {} images", self.state.image_paths.len());

                self.load_current_image(cx);
            }
            Err(error) => {
                self.state.status_message = error;
                self.state.image_paths.clear();
                self.preview_image = None;
                self.preview_dimensions = None;
            }
        }
    }

    fn load_current_image(&mut self, cx: &mut Context<Self>) {
        if let Some(path) = self.state.current_image().cloned() {
            self.preview_version = self.preview_version.wrapping_add(1);
            let version = self.preview_version;
            let cache = self.image_cache.clone();
            let preview_executor = cx.background_executor().clone();
            let preview_path = path.clone();
            let missing_path = path.clone();
            self.state.pan_x = 0.0;
            self.state.pan_y = 0.0;
            self.state.is_dragging = false;

            // Load and present preview as soon as possible, but keep decode/resize work off the UI executor.
            cx.spawn(async move |this, cx| {
                let still_current = this
                    .update(cx, |this, _| version == this.preview_version)
                    .unwrap_or(false);
                if !still_current {
                    return;
                }

                let (preview, dimensions) = preview_executor
                    .spawn(async move {
                        let cached = cache.get_or_decode(
                            &preview_path,
                            crate::processing::image_ops::Rotation::None,
                            None,
                        );

                        cached
                            .as_ref()
                            .map(|c| (Some(c.preview_image.clone()), Some((c.width, c.height))))
                            .unwrap_or((None, None))
                    })
                    .await;

                _ = this.update(cx, |this, cx| {
                    if version == this.preview_version {
                        if let Some(preview) = preview {
                            this.preview_image = Some(preview);
                            this.preview_dimensions = dimensions;
                        } else {
                            this.remove_unavailable_image(&missing_path, cx);
                        }
                    }
                    cx.notify();
                });
            })
            .detach();

            // Preload adjacent images so manual navigation stays responsive.
            self.preload_adjacent(cx);
        } else {
            self.preview_image = None;
            self.preview_dimensions = None;
            cx.notify();
        }
    }

    fn preload_adjacent(&self, cx: &mut Context<Self>) {
        let (priority_preload, secondary_preload): (Vec<PathBuf>, Vec<PathBuf>) = {
            let idx = self.state.current_index;
            let paths = &self.state.image_paths;
            let mut priority = Vec::with_capacity(2);
            let mut secondary = Vec::with_capacity(4);

            // Keep one previous image hot in cache for instant "Prev".
            if idx >= 1 {
                priority.push(paths[idx - 1].clone());
            }

            // Keep immediate next image hot as well.
            if idx + 1 < paths.len() {
                priority.push(paths[idx + 1].clone());
            }

            // Additional preload budget in the background.
            for offset in 2..=3 {
                if idx + offset < paths.len() {
                    secondary.push(paths[idx + offset].clone());
                }
            }
            for offset in 2..=2 {
                if idx >= offset {
                    secondary.push(paths[idx - offset].clone());
                }
            }

            (priority, secondary)
        };

        if !priority_preload.is_empty() {
            let cache = self.image_cache.clone();
            cx.background_executor()
                .spawn(async move {
                    cache.preload(
                        &priority_preload,
                        crate::processing::image_ops::Rotation::None,
                        None,
                    );
                })
                .detach();
        }

        if !secondary_preload.is_empty() {
            let cache = self.image_cache.clone();
            cx.background_executor()
                .spawn(async move {
                    cache.preload(
                        &secondary_preload,
                        crate::processing::image_ops::Rotation::None,
                        None,
                    );
                })
                .detach();
        }
    }

    fn remove_unavailable_image(&mut self, path: &Path, cx: &mut Context<Self>) {
        self.remove_completed_paths(&[path.to_path_buf()], cx);
    }

    fn remove_completed_paths(&mut self, paths: &[PathBuf], cx: &mut Context<Self>) {
        if paths.is_empty() || self.state.image_paths.is_empty() {
            return;
        }

        let current_path = self.state.current_image().cloned();
        let mut removed_current = false;
        let mut removed_count = 0usize;

        for path in paths {
            if let Some(position) = self
                .state
                .image_paths
                .iter()
                .position(|candidate| candidate == path)
            {
                if current_path.as_ref() == Some(path) {
                    removed_current = true;
                }

                self.state.image_paths.remove(position);
                removed_count += 1;

                if position < self.state.current_index {
                    self.state.current_index = self.state.current_index.saturating_sub(1);
                } else if self.state.current_index >= self.state.image_paths.len()
                    && self.state.current_index > 0
                {
                    self.state.current_index -= 1;
                }
            }
        }

        if removed_count == 0 {
            return;
        }

        self.state.input_buffer.clear();
        self.state.status_message = if removed_count == 1 {
            "Removed 1 already-numbered image from the manual queue".to_string()
        } else {
            format!("Removed {removed_count} already-numbered images from the manual queue")
        };

        if removed_current {
            self.preview_image = None;
            self.preview_dimensions = None;
            if !self.state.image_paths.is_empty() {
                self.load_current_image(cx);
            }
        }
    }

    fn open_review_image(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        if !path.exists() {
            self.state.status_message =
                format!("Review image is no longer available: {}", path.display());
            return;
        }

        let index = if let Some(index) = self
            .state
            .image_paths
            .iter()
            .position(|candidate| candidate == &path)
        {
            index
        } else {
            self.state.image_paths.insert(0, path.clone());
            0
        };

        self.state.current_index = index;
        self.state.input_buffer.clear();
        self.state.status_message = format!(
            "Opened review image: {}",
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("image")
        );
        self.load_current_image(cx);
    }

    fn confirm_and_advance(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        // Get value from input state
        let value = self.input_state.read(cx).value().to_string();
        self.state.input_buffer = value;
        let completed_path = self.state.current_image().cloned();

        match self.state.confirm_number() {
            Ok(()) => {
                if let Some(completed_path) = completed_path {
                    self.session.mark_completed(completed_path);
                }
                // Clear input and load next image
                self.input_state.update(cx, |state, cx| {
                    state.set_value("", window, cx);
                });
                self.load_current_image(cx);
            }
            Err(msg) => {
                self.state.status_message = format!("Error: {msg}");
            }
        }
        cx.notify();
    }

    fn skip_image(&mut self, cx: &mut Context<Self>) {
        self.state.next_image();
        self.load_current_image(cx);
        cx.notify();
    }

    fn prev_image(&mut self, cx: &mut Context<Self>) {
        self.state.prev_image();
        self.load_current_image(cx);
        cx.notify();
    }

    fn undo(&mut self, cx: &mut Context<Self>) {
        match self.state.undo() {
            Ok(()) => {
                self.load_current_image(cx);
            }
            Err(msg) => {
                self.state.status_message = format!("Undo failed: {msg}");
            }
        }
        cx.notify();
    }

    fn handle_scroll(&mut self, delta: f32, cx: &mut Context<Self>) {
        if delta > 0.0 {
            self.state.zoom_in();
        } else {
            self.state.zoom_out();
        }

        if let Some((base_w, base_h)) = self.preview_dimensions {
            let (max_x, max_y) = self.pan_limits(base_w as f32, base_h as f32);
            self.state.pan_x = self.state.pan_x.clamp(-max_x, max_x);
            self.state.pan_y = self.state.pan_y.clamp(-max_y, max_y);
        }

        if self.state.zoom_level <= 1.01 {
            self.state.pan_x = 0.0;
            self.state.pan_y = 0.0;
            self.state.is_dragging = false;
        }
        cx.notify();
    }

    fn fitted_size(&self, base_w: f32, base_h: f32) -> (f32, f32) {
        let (view_w, view_h) = self.image_view_size.unwrap_or((base_w, base_h));
        if base_w <= 0.0 || base_h <= 0.0 || view_w <= 0.0 || view_h <= 0.0 {
            return (base_w.max(1.0), base_h.max(1.0));
        }

        let fit_scale = (view_w / base_w).min(view_h / base_h);
        (base_w * fit_scale, base_h * fit_scale)
    }

    fn pan_limits(&self, base_w: f32, base_h: f32) -> (f32, f32) {
        if self.state.zoom_level <= 1.01 {
            return (0.0, 0.0);
        }

        let (fit_w, fit_h) = self.fitted_size(base_w, base_h);
        let scaled_w = fit_w * self.state.zoom_level;
        let scaled_h = fit_h * self.state.zoom_level;
        (
            ((scaled_w - fit_w) * 0.5).max(0.0),
            ((scaled_h - fit_h) * 0.5).max(0.0),
        )
    }

    fn start_pan(&mut self, event: &MouseDownEvent, cx: &mut Context<Self>) {
        if self.state.zoom_level <= 1.01 {
            return;
        }
        self.state.is_dragging = true;
        self.state.drag_start_x = event.position.x.as_f32();
        self.state.drag_start_y = event.position.y.as_f32();
        cx.notify();
    }

    fn handle_pan_move(&mut self, event: &MouseMoveEvent, cx: &mut Context<Self>) {
        if !self.state.is_dragging {
            return;
        }

        let x = event.position.x.as_f32();
        let y = event.position.y.as_f32();
        let dx = x - self.state.drag_start_x;
        let dy = y - self.state.drag_start_y;
        self.state.drag_start_x = x;
        self.state.drag_start_y = y;

        self.state.pan_x += dx;
        self.state.pan_y += dy;

        if let Some((base_w, base_h)) = self.preview_dimensions {
            let (max_x, max_y) = self.pan_limits(base_w as f32, base_h as f32);
            self.state.pan_x = self.state.pan_x.clamp(-max_x, max_x);
            self.state.pan_y = self.state.pan_y.clamp(-max_y, max_y);
        }

        cx.notify();
    }

    fn end_pan(&mut self, cx: &mut Context<Self>) {
        if self.state.is_dragging {
            self.state.is_dragging = false;
            cx.notify();
        }
    }

    fn render_toolbar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let entity = cx.entity().clone();
        let (done, total) = self.state.progress();
        let remaining = self.state.remaining();
        let sticker_name = crate::ocr::sticker_template_name();
        let has_sticker = crate::ocr::has_sticker_template();

        let mut row = h_flex()
            .px_3()
            .py_2()
            .gap_3()
            .items_center()
            .bg(cx.theme().background)
            .border_b_1()
            .border_color(cx.theme().border)
            .child(
                Button::new("open-folder")
                    .label("📂 Open Folder")
                    .small()
                    .on_click({
                        let entity = entity.clone();
                        move |_, window, cx| {
                            entity.update(cx, |this, cx| this.open_folder(window, cx));
                        }
                    }),
            )
            .child(
                Button::new("open-sticker-template")
                    .label("Sticker Template")
                    .small()
                    .on_click({
                        let entity = entity.clone();
                        move |_, window, cx| {
                            entity.update(cx, |this, cx| this.open_sticker_template(window, cx));
                        }
                    }),
            )
            .child(
                Button::new("open-autonomous-numbering")
                    .label("Auto Window")
                    .small()
                    .on_click({
                        let entity = entity.clone();
                        move |_, _, cx| {
                            entity.update(cx, |this, cx| this.open_autonomous_window(cx));
                        }
                    }),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(cx.theme().muted_foreground)
                    .child(format!("Progress: {done}/{total} ({remaining} remaining)")),
            )
            .child(div().flex_1())
            .child(Button::new("prev").label("← Prev").small().on_click({
                let entity = entity.clone();
                move |_, _, cx| {
                    entity.update(cx, |this, cx| this.prev_image(cx));
                }
            }))
            .child(Button::new("skip").label("Skip →").small().on_click({
                let entity = entity.clone();
                move |_, _, cx| {
                    entity.update(cx, |this, cx| this.skip_image(cx));
                }
            }))
            .child(Button::new("undo").label("↩ Undo").small().on_click({
                let entity = entity.clone();
                move |_, _, cx| {
                    entity.update(cx, |this, cx| this.undo(cx));
                }
            }));

        if has_sticker {
            row = row.child(
                Button::new("clear-sticker-template")
                    .label("Clear Sticker")
                    .small()
                    .on_click({
                        let entity = entity.clone();
                        move |_, _, cx| {
                            entity.update(cx, |this, cx| this.clear_sticker_template(cx));
                        }
                    }),
            );
        }

        if let Some(name) = sticker_name {
            row = row.child(
                div()
                    .text_sm()
                    .text_color(cx.theme().muted_foreground)
                    .child(format!("Template: {name}")),
            );
        }

        row
    }

    fn render_image_view(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let entity = cx.entity().clone();

        let content = if let Some(ref preview) = self.preview_image {
            let source = gpui::ImageSource::Render(preview.clone());
            let (base_w, base_h) = self.preview_dimensions.unwrap_or((1200, 900));
            let (fit_w, fit_h) = self.fitted_size(base_w as f32, base_h as f32);
            let scaled_w = (fit_w * self.state.zoom_level).max(1.0);
            let scaled_h = (fit_h * self.state.zoom_level).max(1.0);
            let (max_pan_x, max_pan_y) = self.pan_limits(base_w as f32, base_h as f32);
            let pan_x = self.state.pan_x.clamp(-max_pan_x, max_pan_x);
            let pan_y = self.state.pan_y.clamp(-max_pan_y, max_pan_y);
            let (view_w, view_h) = self.image_view_size.unwrap_or((fit_w, fit_h));
            let left = ((view_w - scaled_w) * 0.5 + pan_x).round();
            let top = ((view_h - scaled_h) * 0.5 + pan_y).round();

            div().size_full().relative().overflow_hidden().child(
                img(source)
                    .absolute()
                    .left(px(left))
                    .top(px(top))
                    .w(px(scaled_w))
                    .h(px(scaled_h))
                    .object_fit(ObjectFit::Fill),
            )
        } else {
            div()
                .size_full()
                .flex()
                .items_center()
                .justify_center()
                .child(
                    div()
                        .text_base()
                        .text_color(hsla(0., 0., 0.5, 1.0))
                        .child("Open a folder to begin"),
                )
        };

        div()
            .id("image-view")
            .flex_1()
            .min_w(px(0.))
            .min_h(px(0.))
            .w_full()
            .bg(hsla(0., 0., 0.1, 1.0))
            .overflow_hidden()
            .on_scroll_wheel({
                let entity = entity.clone();
                move |ev, window, cx| {
                    window.prevent_default();
                    cx.stop_propagation();
                    let delta = ev.delta.pixel_delta(px(1.0)).y.as_f32();
                    entity.update(cx, |this, cx| this.handle_scroll(delta, cx));
                }
            })
            .on_mouse_down(MouseButton::Left, {
                let entity = entity.clone();
                move |ev, _, cx| {
                    entity.update(cx, |this, cx| this.start_pan(ev, cx));
                }
            })
            .on_mouse_move({
                let entity = entity.clone();
                move |ev, _, cx| {
                    entity.update(cx, |this, cx| this.handle_pan_move(ev, cx));
                }
            })
            .on_mouse_up(MouseButton::Left, {
                let entity = entity.clone();
                move |_, _, cx| {
                    entity.update(cx, |this, cx| this.end_pan(cx));
                }
            })
            .on_mouse_up_out(MouseButton::Left, {
                let entity = entity.clone();
                move |_, _, cx| {
                    entity.update(cx, |this, cx| this.end_pan(cx));
                }
            })
            .on_prepaint({
                let entity = entity.clone();
                move |bounds, _, cx| {
                    entity.update(cx, |this, _| {
                        this.image_view_size =
                            Some((bounds.size.width.as_f32(), bounds.size.height.as_f32()));
                    });
                }
            })
            .child(content)
    }

    fn render_input_bar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let entity = cx.entity().clone();

        h_flex()
            .px_3()
            .py_2()
            .gap_4()
            .items_center()
            .bg(cx.theme().background)
            .border_t_1()
            .border_color(cx.theme().border)
            .child(
                div()
                    .text_sm()
                    .font_weight(FontWeight::MEDIUM)
                    .child("Number:"),
            )
            .child(div().child(Input::new(&self.input_state).w(px(200.0)).tab_index(-1)))
            .child(
                Button::new("confirm")
                    .label("Enter ↵")
                    .small()
                    .primary()
                    .tab_index(-1)
                    .on_click({
                        let entity = entity.clone();
                        move |_, window, cx| {
                            entity.update(cx, |this, cx| this.confirm_and_advance(window, cx));
                        }
                    }),
            )
            .child(div().flex_1())
            .child(
                div()
                    .text_sm()
                    .text_color(cx.theme().muted_foreground)
                    .child(format!("Zoom: {:.0}%", self.state.zoom_level * 100.0)),
            )
    }

    fn render_autonomous_progress(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let progress = &self.autonomous_progress;
        let pct = if progress.total == 0 {
            0.0
        } else {
            (progress.completed as f32 / progress.total as f32 * 100.0).clamp(0.0, 100.0)
        };
        let active = if progress.active_files.is_empty() {
            "Idle".to_string()
        } else {
            progress.active_files.join(", ")
        };

        h_flex()
            .px_3()
            .py_1()
            .gap_3()
            .items_center()
            .bg(cx.theme().background)
            .border_t_1()
            .border_color(cx.theme().border)
            .child(
                div()
                    .text_xs()
                    .text_color(cx.theme().muted_foreground)
                    .child(if progress.running {
                        "Autonomous running"
                    } else {
                        "Autonomous idle"
                    }),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(cx.theme().muted_foreground)
                    .child(format!(
                        "{}/{} assigned {} review {} done {} missing {} failed {}",
                        progress.completed,
                        progress.total,
                        progress.assigned,
                        progress.needs_review,
                        progress.already_completed,
                        progress.skipped_missing,
                        progress.failed
                    )),
            )
            .child(
                div().w(px(160.)).child(
                    gpui_component::progress::Progress::new("manual-auto-progress").value(pct),
                ),
            )
            .child(
                div()
                    .flex_1()
                    .min_w(px(0.))
                    .truncate()
                    .text_xs()
                    .text_color(cx.theme().muted_foreground)
                    .child(if progress.last_message.is_empty() {
                        active
                    } else {
                        format!("{} - {}", progress.last_message, active)
                    }),
            )
    }

    fn render_status_bar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        h_flex()
            .px_3()
            .py_1()
            .bg(cx.theme().muted)
            .child(
                div()
                    .text_xs()
                    .text_color(cx.theme().muted_foreground)
                    .child(self.state.status_message.clone()),
            )
            .child(div().flex_1())
            .child(
                div()
                    .text_xs()
                    .text_color(cx.theme().muted_foreground)
                    .child("Scroll over image: zoom | Ctrl+Z: undo"),
            )
    }
}

impl Render for NumberingMode {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let entity = cx.entity().clone();

        v_flex()
            .size_full()
            .capture_key_down({
                let entity = entity.clone();
                move |ev, window, cx| {
                    let key = ev.keystroke.key.as_str();
                    let mods = ev.keystroke.modifiers;

                    if (mods.control || mods.platform) && key.eq_ignore_ascii_case("z") {
                        window.prevent_default();
                        cx.stop_propagation();
                        entity.update(cx, |this, cx| this.undo(cx));
                    }
                }
            })
            .child(self.render_toolbar(cx))
            .child(self.render_image_view(cx))
            .child(self.render_autonomous_progress(cx))
            .child(self.render_input_bar(cx))
            .child(self.render_status_bar(cx))
    }
}
