use gpui::prelude::FluentBuilder;
use gpui::*;
use gpui_component::IndexPath;
use gpui_component::button::{Button, ButtonVariants as _};
use gpui_component::checkbox::Checkbox;
use gpui_component::color_picker::{ColorPicker, ColorPickerEvent, ColorPickerState};
use gpui_component::input::{Input, InputEvent, InputState};
use gpui_component::progress::Progress;
use gpui_component::scroll::ScrollableElement;
use gpui_component::scroll::{Scrollbar, ScrollbarShow};
use gpui_component::select::{SelectEvent, SelectItem, SelectState};
use gpui_component::slider::{Slider, SliderEvent, SliderState};
use gpui_component::{ActiveTheme, Disableable, Sizable, h_flex, v_flex};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use crate::numbering_mode::NumberingMode;
use crate::processing::batch::{
    BatchConfig, OutputFormat, PosterOptions, ProcessResult, WatermarkConfig, WatermarkRotation,
    apply_preview_effects,
};
use crate::processing::image_cache::ImageCache;
use crate::processing::image_ops::{Rotation, TextColor, TextOverlayConfig, TextPosition};
use crate::processing::sort::{
    NumberedSortConfig, NumberedSortMode, NumberedSortResult, NumberedSortSummary,
};

#[derive(Clone, Debug)]
struct SelectOption<V: Clone> {
    label: SharedString,
    value: V,
}

impl<V: Clone + 'static> SelectItem for SelectOption<V> {
    type Value = V;
    fn title(&self) -> SharedString {
        self.label.clone()
    }
    fn value(&self) -> &V {
        &self.value
    }
}

type FormatSelectState = SelectState<Vec<SelectOption<OutputFormat>>>;
type RotationSelectState = SelectState<Vec<SelectOption<Rotation>>>;
type WatermarkRotationSelectState = SelectState<Vec<SelectOption<WatermarkRotation>>>;
type PositionSelectState = SelectState<Vec<SelectOption<TextPosition>>>;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AppMode {
    BatchProcessing,
    Numbering,
    Sorting,
}

pub struct App {
    // Current mode
    mode: AppMode,
    numbering_mode: Entity<NumberingMode>,

    // Shared image cache
    image_cache: Arc<ImageCache>,

    // Image list (batch mode)
    image_paths: Vec<PathBuf>,
    image_names: Vec<SharedString>,
    selected_index: Option<usize>,
    /// In-memory preview image ready for GPUI rendering.
    preview_image: Option<Arc<RenderImage>>,
    preview_version: usize,

    // Settings (batch mode)
    quality: u8,
    rotation: Rotation,
    output_format: OutputFormat,
    output_dir: Option<PathBuf>,

    // Text overlay
    text_enabled: bool,
    text_position: TextPosition,
    text_font_size: f32,
    text_edge_margin: u32,
    text_color_r: u8,
    text_color_g: u8,
    text_color_b: u8,
    poster_resize_33x66: bool,
    poster_margin_enabled: bool,
    poster_margin_px: u32,
    watermark_enabled: bool,
    watermark_image_path: Option<PathBuf>,
    watermark_position: TextPosition,
    watermark_edge_margin: u32,
    watermark_scale_percent: f32,
    watermark_opacity: f32,
    watermark_rotation: WatermarkRotation,

    // Processing state
    is_processing: bool,
    progress: f32,
    status_message: SharedString,
    batch_results: Vec<ProcessResult>,

    // Sorting state
    sorting_source_dir: Option<PathBuf>,
    sorting_copy_files: bool,
    is_sorting: bool,
    sorting_progress: f32,
    sorting_status_message: SharedString,
    sorting_results: Vec<NumberedSortResult>,
    sorting_summary: Option<NumberedSortSummary>,

    // Entity handles
    quality_slider: Entity<SliderState>,
    font_size_slider: Entity<SliderState>,
    text_margin_slider: Entity<SliderState>,
    poster_margin_slider: Entity<SliderState>,
    watermark_margin_slider: Entity<SliderState>,
    watermark_scale_slider: Entity<SliderState>,
    watermark_opacity_slider: Entity<SliderState>,
    format_select: Entity<FormatSelectState>,
    rotation_select: Entity<RotationSelectState>,
    watermark_rotation_select: Entity<WatermarkRotationSelectState>,
    settings_scroll_handle: ScrollHandle,
    position_select: Entity<PositionSelectState>,
    watermark_position_select: Entity<PositionSelectState>,
    text_input: Entity<InputState>,
    color_picker: Entity<ColorPickerState>,
    text_template_value: String,

    _subscriptions: Vec<Subscription>,
}

impl App {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let image_cache = Arc::new(ImageCache::new(15));

        let numbering_mode = {
            let cache = image_cache.clone();
            cx.new(|cx| NumberingMode::new(window, cx, cache))
        };

        let quality_slider = cx.new(|_| {
            SliderState::new()
                .min(1.0)
                .max(100.0)
                .step(1.0)
                .default_value(70.0)
        });

        let font_size_slider = cx.new(|_| {
            SliderState::new()
                .min(8.0)
                .max(200.0)
                .step(1.0)
                .default_value(24.0)
        });
        let text_margin_slider = cx.new(|_| {
            SliderState::new()
                .min(0.0)
                .max(200.0)
                .step(1.0)
                .default_value(10.0)
        });
        let poster_margin_slider = cx.new(|_| {
            SliderState::new()
                .min(0.0)
                .max(400.0)
                .step(1.0)
                .default_value(40.0)
        });
        let watermark_margin_slider = cx.new(|_| {
            SliderState::new()
                .min(0.0)
                .max(200.0)
                .step(1.0)
                .default_value(10.0)
        });
        let watermark_scale_slider = cx.new(|_| {
            SliderState::new()
                .min(5.0)
                .max(300.0)
                .step(1.0)
                .default_value(100.0)
        });
        let watermark_opacity_slider = cx.new(|_| {
            SliderState::new()
                .min(0.05)
                .max(1.0)
                .step(0.05)
                .default_value(0.25)
        });

        let format_items = vec![
            SelectOption {
                label: "JPEG".into(),
                value: OutputFormat::Jpeg,
            },
            SelectOption {
                label: "PDF".into(),
                value: OutputFormat::Pdf,
            },
        ];
        let format_select = cx.new(|cx| {
            SelectState::new(format_items, Some(IndexPath::default().row(1)), window, cx)
        });

        let rotation_items = vec![
            SelectOption {
                label: "None".into(),
                value: Rotation::None,
            },
            SelectOption {
                label: "90° CW".into(),
                value: Rotation::Cw90,
            },
            SelectOption {
                label: "180°".into(),
                value: Rotation::Cw180,
            },
            SelectOption {
                label: "270° CW".into(),
                value: Rotation::Cw270,
            },
        ];
        let rotation_select = cx.new(|cx| {
            SelectState::new(
                rotation_items,
                Some(IndexPath::default().row(0)),
                window,
                cx,
            )
        });
        let watermark_rotation_items = vec![
            SelectOption {
                label: "Match Image Rotation".into(),
                value: WatermarkRotation::MatchImage,
            },
            SelectOption {
                label: "None".into(),
                value: WatermarkRotation::Fixed(Rotation::None),
            },
            SelectOption {
                label: "90° CW".into(),
                value: WatermarkRotation::Fixed(Rotation::Cw90),
            },
            SelectOption {
                label: "180°".into(),
                value: WatermarkRotation::Fixed(Rotation::Cw180),
            },
            SelectOption {
                label: "270° CW".into(),
                value: WatermarkRotation::Fixed(Rotation::Cw270),
            },
        ];
        let watermark_rotation_select = cx.new(|cx| {
            SelectState::new(
                watermark_rotation_items,
                Some(IndexPath::default().row(0)),
                window,
                cx,
            )
        });

        let position_items = vec![
            SelectOption {
                label: "Top Left".into(),
                value: TextPosition::TopLeft,
            },
            SelectOption {
                label: "Top Right".into(),
                value: TextPosition::TopRight,
            },
            SelectOption {
                label: "Bottom Left".into(),
                value: TextPosition::BottomLeft,
            },
            SelectOption {
                label: "Bottom Right".into(),
                value: TextPosition::BottomRight,
            },
            SelectOption {
                label: "Center".into(),
                value: TextPosition::Center,
            },
        ];
        let position_select = cx.new(|cx| {
            SelectState::new(
                position_items.clone(),
                Some(IndexPath::default().row(3)),
                window,
                cx,
            )
        });
        let watermark_position_select = cx.new(|cx| {
            SelectState::new(
                position_items,
                Some(IndexPath::default().row(4)),
                window,
                cx,
            )
        });

        let text_input = cx.new(|cx| InputState::new(window, cx).placeholder("e.g. {filename}"));
        text_input.update(cx, |state, cx| {
            state.set_value("{filename}", window, cx);
        });
        let color_picker =
            cx.new(|cx| ColorPickerState::new(window, cx).default_value(hsla(0.0, 0.0, 1.0, 1.0)));

        let mut subs = Vec::new();

        subs.push(cx.subscribe_in(
            &quality_slider,
            window,
            |this, _, ev: &SliderEvent, _window, cx| {
                let SliderEvent::Change(val) = ev;
                this.quality = val.start() as u8;
                cx.notify();
            },
        ));

        subs.push(cx.subscribe_in(
            &font_size_slider,
            window,
            |this, _, ev: &SliderEvent, _window, cx| {
                let SliderEvent::Change(val) = ev;
                this.text_font_size = val.start();
                this.schedule_preview_update(cx);
                cx.notify();
            },
        ));
        subs.push(cx.subscribe_in(
            &text_margin_slider,
            window,
            |this, _, ev: &SliderEvent, _window, cx| {
                let SliderEvent::Change(val) = ev;
                this.text_edge_margin = val.start() as u32;
                this.schedule_preview_update(cx);
                cx.notify();
            },
        ));
        subs.push(cx.subscribe_in(
            &poster_margin_slider,
            window,
            |this, _, ev: &SliderEvent, _window, cx| {
                let SliderEvent::Change(val) = ev;
                this.poster_margin_px = val.start() as u32;
                this.schedule_preview_update(cx);
                cx.notify();
            },
        ));
        subs.push(cx.subscribe_in(
            &watermark_margin_slider,
            window,
            |this, _, ev: &SliderEvent, _window, cx| {
                let SliderEvent::Change(val) = ev;
                this.watermark_edge_margin = val.start() as u32;
                this.schedule_preview_update(cx);
                cx.notify();
            },
        ));
        subs.push(cx.subscribe_in(
            &watermark_scale_slider,
            window,
            |this, _, ev: &SliderEvent, _window, cx| {
                let SliderEvent::Change(val) = ev;
                this.watermark_scale_percent = val.start();
                this.schedule_preview_update(cx);
                cx.notify();
            },
        ));
        subs.push(cx.subscribe_in(
            &watermark_opacity_slider,
            window,
            |this, _, ev: &SliderEvent, _window, cx| {
                let SliderEvent::Change(val) = ev;
                this.watermark_opacity = val.start();
                this.schedule_preview_update(cx);
                cx.notify();
            },
        ));

        subs.push(cx.subscribe_in(
            &format_select,
            window,
            |this, _, ev: &SelectEvent<Vec<SelectOption<OutputFormat>>>, _window, cx| {
                if let SelectEvent::Confirm(Some(value)) = ev {
                    this.output_format = *value;
                    cx.notify();
                }
            },
        ));

        subs.push(cx.subscribe_in(
            &rotation_select,
            window,
            |this, _, ev: &SelectEvent<Vec<SelectOption<Rotation>>>, _window, cx| {
                if let SelectEvent::Confirm(Some(value)) = ev {
                    this.rotation = *value;
                    this.schedule_preview_update(cx);
                    cx.notify();
                }
            },
        ));
        subs.push(cx.subscribe_in(
            &watermark_rotation_select,
            window,
            |this, _, ev: &SelectEvent<Vec<SelectOption<WatermarkRotation>>>, _window, cx| {
                if let SelectEvent::Confirm(Some(value)) = ev {
                    this.watermark_rotation = *value;
                    this.schedule_preview_update(cx);
                    cx.notify();
                }
            },
        ));

        subs.push(cx.subscribe_in(
            &position_select,
            window,
            |this, _, ev: &SelectEvent<Vec<SelectOption<TextPosition>>>, _window, cx| {
                if let SelectEvent::Confirm(Some(value)) = ev {
                    this.text_position = *value;
                    this.schedule_preview_update(cx);
                    cx.notify();
                }
            },
        ));
        subs.push(cx.subscribe_in(
            &watermark_position_select,
            window,
            |this, _, ev: &SelectEvent<Vec<SelectOption<TextPosition>>>, _window, cx| {
                if let SelectEvent::Confirm(Some(value)) = ev {
                    this.watermark_position = *value;
                    this.schedule_preview_update(cx);
                    cx.notify();
                }
            },
        ));

        subs.push(cx.subscribe_in(
            &text_input,
            window,
            |this, state, ev: &InputEvent, _window, cx| {
                if let InputEvent::Change = ev {
                    let val = state.read(cx).value();
                    this.text_template_value = val.to_string();
                    this.schedule_preview_update(cx);
                    cx.notify();
                }
            },
        ));
        subs.push(cx.subscribe_in(
            &color_picker,
            window,
            |this, _, ev: &ColorPickerEvent, _window, cx| {
                if let ColorPickerEvent::Change(Some(hsla)) = ev {
                    let (r, g, b) = hsla_to_rgb(hsla.h, hsla.s, hsla.l);
                    this.text_color_r = r;
                    this.text_color_g = g;
                    this.text_color_b = b;
                    this.schedule_preview_update(cx);
                    cx.notify();
                }
            },
        ));

        Self {
            mode: AppMode::BatchProcessing,
            numbering_mode,
            image_cache,

            image_paths: Vec::new(),
            image_names: Vec::new(),
            selected_index: None,
            preview_image: None,
            preview_version: 0,

            quality: 70,
            rotation: Rotation::None,
            output_format: OutputFormat::Pdf,
            output_dir: None,

            text_enabled: false,
            text_position: TextPosition::BottomRight,
            text_font_size: 24.0,
            text_edge_margin: 10,
            text_color_r: 255,
            text_color_g: 255,
            text_color_b: 255,
            poster_resize_33x66: false,
            poster_margin_enabled: false,
            poster_margin_px: 40,
            watermark_enabled: false,
            watermark_image_path: None,
            watermark_position: TextPosition::Center,
            watermark_edge_margin: 10,
            watermark_scale_percent: 100.0,
            watermark_opacity: 0.25,
            watermark_rotation: WatermarkRotation::MatchImage,

            is_processing: false,
            progress: 0.0,
            status_message: "Ready - Open a folder to begin".into(),
            batch_results: Vec::new(),

            sorting_source_dir: None,
            sorting_copy_files: false,
            is_sorting: false,
            sorting_progress: 0.0,
            sorting_status_message: "Select a folder of numbered files to sort".into(),
            sorting_results: Vec::new(),
            sorting_summary: None,

            quality_slider,
            font_size_slider,
            text_margin_slider,
            poster_margin_slider,
            watermark_margin_slider,
            watermark_scale_slider,
            watermark_opacity_slider,
            format_select,
            rotation_select,
            watermark_rotation_select,
            settings_scroll_handle: ScrollHandle::new(),
            position_select,
            watermark_position_select,
            text_input,
            color_picker,
            text_template_value: "{filename}".to_string(),

            _subscriptions: subs,
        }
    }

    fn set_mode(&mut self, mode: AppMode, cx: &mut Context<Self>) {
        self.mode = mode;
        cx.notify();
    }

    fn open_folder(&mut self, cx: &mut Context<Self>) {
        cx.spawn(async move |this, cx| {
            let handle = rfd::AsyncFileDialog::new()
                .set_title("Select image folder")
                .pick_folder()
                .await;

            if let Some(folder) = handle {
                let dir = folder.path().to_path_buf();
                let mut images: Vec<PathBuf> = std::fs::read_dir(&dir)
                    .ok()
                    .into_iter()
                    .flat_map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()))
                    .filter(|p| {
                        matches!(
                            p.extension()
                                .and_then(|e| e.to_str())
                                .map(|e| e.to_ascii_lowercase()),
                            Some(ref ext) if ext == "jpg" || ext == "jpeg" || ext == "png"
                        )
                    })
                    .collect();
                images.sort();
                let image_names: Vec<SharedString> = images
                    .iter()
                    .map(|path| {
                        path.file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("???")
                            .to_string()
                            .into()
                    })
                    .collect();

                _ = this.update(cx, |this, cx| {
                    this.status_message = format!("Loaded {} images", images.len()).into();
                    this.image_paths = images;
                    this.image_names = image_names;
                    this.selected_index = (!this.image_paths.is_empty()).then_some(0);
                    this.batch_results.clear();
                    if this.selected_index.is_some() {
                        this.schedule_preview_update(cx);
                    } else {
                        this.preview_image = None;
                    }
                    cx.notify();
                });
            }
        })
        .detach();
    }

    fn select_image(&mut self, idx: usize, cx: &mut Context<Self>) {
        self.selected_index = Some(idx);
        self.schedule_preview_update(cx);
        cx.notify();
    }

    fn choose_output_dir(&mut self, cx: &mut Context<Self>) {
        cx.spawn(async move |this, cx| {
            let handle = rfd::AsyncFileDialog::new()
                .set_title("Select output folder")
                .pick_folder()
                .await;
            if let Some(folder) = handle {
                let dir = folder.path().to_path_buf();
                _ = this.update(cx, |this, cx| {
                    this.status_message = format!("Output: {}", dir.display()).into();
                    this.output_dir = Some(dir);
                    cx.notify();
                });
            }
        })
        .detach();
    }

    fn choose_watermark_image(&mut self, cx: &mut Context<Self>) {
        cx.spawn(async move |this, cx| {
            let handle = rfd::AsyncFileDialog::new()
                .set_title("Select watermark image")
                .add_filter("Images", &["png", "jpg", "jpeg"])
                .pick_file()
                .await;
            if let Some(file) = handle {
                let path = file.path().to_path_buf();
                _ = this.update(cx, |this, cx| {
                    this.watermark_image_path = Some(path);
                    this.schedule_preview_update(cx);
                    cx.notify();
                });
            }
        })
        .detach();
    }

    fn start_batch(&mut self, cx: &mut Context<Self>) {
        if self.image_paths.is_empty() {
            self.status_message = "No images loaded".into();
            cx.notify();
            return;
        }
        let Some(ref output_dir) = self.output_dir else {
            self.status_message = "Select an output folder first".into();
            cx.notify();
            return;
        };
        if self.watermark_enabled && self.watermark_image_path.is_none() {
            self.status_message = "Select a watermark image first".into();
            cx.notify();
            return;
        }

        self.is_processing = true;
        self.progress = 0.0;
        self.batch_results.clear();
        self.status_message = "Processing...".into();
        cx.notify();

        let paths = self.image_paths.clone();
        let config = BatchConfig {
            quality: self.quality,
            rotation: self.rotation,
            text_overlay: if self.text_enabled {
                Some(self.build_text_config())
            } else {
                None
            },
            poster: self.build_poster_options(),
            watermark: self.build_watermark_config(),
            output_format: self.output_format,
            output_dir: output_dir.clone(),
        };

        let executor = cx.background_executor().clone();

        let _entity = cx.entity().downgrade();
        cx.spawn(async move |this, cx| {
            let results = executor
                .spawn(async move {
                    crate::processing::batch::process_batch(&paths, &config, |_, _| {})
                })
                .await;

            _ = this.update(cx, |this, cx| {
                this.is_processing = false;
                this.progress = 1.0;
                let success = results.iter().filter(|r| r.success).count();
                let fail = results.len() - success;
                this.status_message = format!(
                    "Done - {success} succeeded, {fail} failed out of {} total",
                    results.len()
                )
                .into();
                this.batch_results = results;
                cx.notify();
            });
        })
        .detach();
    }

    fn choose_sorting_source_dir(&mut self, cx: &mut Context<Self>) {
        cx.spawn(async move |this, cx| {
            let handle = rfd::AsyncFileDialog::new()
                .set_title("Select folder with numbered files")
                .pick_folder()
                .await;

            if let Some(folder) = handle {
                let dir = folder.path().to_path_buf();
                _ = this.update(cx, |this, cx| {
                    this.sorting_source_dir = Some(dir.clone());
                    this.sorting_status_message =
                        format!("Sorting source: {}", dir.display()).into();
                    this.sorting_results.clear();
                    this.sorting_summary = None;
                    this.sorting_progress = 0.0;
                    cx.notify();
                });
            }
        })
        .detach();
    }

    fn start_sorting(&mut self, cx: &mut Context<Self>) {
        if self.is_sorting {
            return;
        }

        let Some(source_dir) = self.sorting_source_dir.clone() else {
            self.sorting_status_message = "Select a sorting folder first".into();
            cx.notify();
            return;
        };

        self.is_sorting = true;
        self.sorting_progress = 0.0;
        self.sorting_results.clear();
        self.sorting_summary = None;
        self.sorting_status_message = "Sorting numbered files...".into();
        cx.notify();

        let config = NumberedSortConfig {
            source_dir,
            mode: if self.sorting_copy_files {
                NumberedSortMode::Copy
            } else {
                NumberedSortMode::Move
            },
        };
        let executor = cx.background_executor().clone();

        let progress_completed = Arc::new(AtomicUsize::new(0));
        let progress_total = Arc::new(AtomicUsize::new(0));
        let progress_done = Arc::new(AtomicBool::new(false));

        let progress_completed_for_ui = progress_completed.clone();
        let progress_total_for_ui = progress_total.clone();
        let progress_done_for_ui = progress_done.clone();
        let timer_executor = executor.clone();

        cx.spawn(async move |this, cx| {
            loop {
                timer_executor.timer(Duration::from_millis(75)).await;

                let total = progress_total_for_ui.load(Ordering::Relaxed);
                let completed = progress_completed_for_ui.load(Ordering::Relaxed);
                let done = progress_done_for_ui.load(Ordering::Relaxed);

                _ = this.update(cx, |this, cx| {
                    if this.is_sorting {
                        this.sorting_progress = if total == 0 {
                            0.0
                        } else {
                            (completed as f32 / total as f32).clamp(0.0, 1.0)
                        };
                        cx.notify();
                    }
                });

                if done {
                    break;
                }
            }
        })
        .detach();

        let progress_completed_for_worker = progress_completed.clone();
        let progress_total_for_worker = progress_total.clone();
        let progress_done_for_worker = progress_done.clone();

        cx.spawn(async move |this, cx| {
            let (summary, results) = executor
                .spawn(async move {
                    crate::processing::sort::sort_numbered_files(&config, |completed, total| {
                        progress_total_for_worker.store(total, Ordering::Relaxed);
                        progress_completed_for_worker.store(completed, Ordering::Relaxed);
                    })
                })
                .await;

            progress_done_for_worker.store(true, Ordering::Relaxed);

            _ = this.update(cx, |this, cx| {
                this.is_sorting = false;
                this.sorting_progress = 1.0;
                this.sorting_status_message = format!(
                    "Sorting complete - {} sorted, {} failed, {} skipped, {} folders created",
                    summary.sorted, summary.failed, summary.skipped, summary.folders_created
                )
                .into();
                this.sorting_summary = Some(summary);
                this.sorting_results = results;
                cx.notify();
            });
        })
        .detach();
    }

    fn schedule_preview_update(&mut self, cx: &mut Context<Self>) {
        let Some(idx) = self.selected_index else {
            return;
        };
        self.preview_version = self.preview_version.wrapping_add(1);
        let version = self.preview_version;

        let preview_path = self.image_paths[idx].clone();
        let status_path = preview_path.clone();
        let rotation = self.rotation;
        let preview_config = BatchConfig {
            quality: self.quality,
            rotation,
            text_overlay: if self.text_enabled {
                Some(self.build_text_config())
            } else {
                None
            },
            poster: self.build_poster_options(),
            watermark: self.build_watermark_config(),
            output_format: self.output_format,
            output_dir: self.output_dir.clone().unwrap_or_default(),
        };

        let cache = self.image_cache.clone();
        let executor = cx.background_executor().clone();

        cx.spawn(async move |this, cx| {
            // Check if still current before doing work
            let still_current = this
                .update(cx, |this, _| version == this.preview_version)
                .unwrap_or(false);
            if !still_current {
                return;
            }

            let result: Option<Arc<RenderImage>> = executor
                .spawn(async move {
                    (|| {
                        let cached = cache.get_or_decode(&preview_path, rotation, None)?;
                        if preview_config.text_overlay.is_none()
                            && !preview_config.poster.resize_to_33x66cm
                            && preview_config.poster.margin_px == 0
                            && preview_config.watermark.is_none()
                        {
                            return Some(cached.preview_image.clone());
                        }

                        let preview = image::DynamicImage::ImageRgba8((*cached.rgba).clone());
                        let filename = preview_path
                            .file_stem()
                            .and_then(|n| n.to_str())
                            .unwrap_or("image");
                        let rgba =
                            apply_preview_effects(preview, filename, &preview_config, None, None)
                                .into_rgba8();

                        Some(Arc::new(
                            crate::processing::image_cache::render_preview_image(&rgba),
                        ))
                    })()
                })
                .await;

            _ = this.update(cx, |this, cx| {
                if version == this.preview_version {
                    if result.is_some() {
                        this.preview_image = result;
                    } else {
                        this.preview_image = None;
                        let name = status_path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("image");
                        this.status_message = format!("Skipping unreadable image: {name}").into();
                        if this.selected_index == Some(idx) && idx + 1 < this.image_paths.len() {
                            this.selected_index = Some(idx + 1);
                            this.schedule_preview_update(cx);
                        }
                    }
                }
                cx.notify();
            });
        })
        .detach();

        self.preload_adjacent(cx);
    }

    fn preload_adjacent(&self, cx: &mut Context<Self>) {
        let Some(idx) = self.selected_index else {
            return;
        };

        let paths = &self.image_paths;
        let mut adjacent = Vec::new();
        for offset in [1usize, 2usize] {
            if idx >= offset {
                adjacent.push(paths[idx - offset].clone());
            }
            let next = idx + offset;
            if next < paths.len() {
                adjacent.push(paths[next].clone());
            }
        }

        if adjacent.is_empty() {
            return;
        }

        let cache = self.image_cache.clone();
        let rotation = self.rotation;

        // Preload base images (no text overlay - that's applied on the fly)
        cx.background_executor()
            .spawn(async move {
                cache.preload(&adjacent, rotation, None);
            })
            .detach();
    }

    fn build_text_config(&self) -> TextOverlayConfig {
        TextOverlayConfig {
            text_template: self.text_template_value.clone(),
            position: self.text_position,
            font_size: self.text_font_size,
            color: TextColor {
                r: self.text_color_r,
                g: self.text_color_g,
                b: self.text_color_b,
                a: 255,
            },
            margin: self.text_edge_margin,
        }
    }

    fn build_poster_options(&self) -> PosterOptions {
        PosterOptions {
            resize_to_33x66cm: self.poster_resize_33x66,
            margin_px: if self.poster_margin_enabled {
                self.poster_margin_px
            } else {
                0
            },
        }
    }

    fn build_watermark_config(&self) -> Option<WatermarkConfig> {
        if !self.watermark_enabled {
            return None;
        }
        Some(WatermarkConfig {
            image_path: self.watermark_image_path.clone()?,
            position: self.watermark_position,
            margin: self.watermark_edge_margin,
            scale_percent: self.watermark_scale_percent,
            opacity: self.watermark_opacity,
            rotation: self.watermark_rotation,
        })
    }

    fn render_sidebar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let entity = cx.entity().clone();

        let header = h_flex()
            .px_3()
            .py_2()
            .items_center()
            .justify_between()
            .child(
                div()
                    .text_base()
                    .font_weight(FontWeight::SEMIBOLD)
                    .child("Images"),
            )
            .child(Button::new("open-btn").label("📂 Open").small().on_click({
                let entity = entity.clone();
                move |_, _window, cx| {
                    entity.update(cx, |this, cx| this.open_folder(cx));
                }
            }));

        let file_list = if self.image_paths.is_empty() {
            div()
                .p_4()
                .flex_1()
                .child(
                    div()
                        .text_sm()
                        .text_color(cx.theme().muted_foreground)
                        .child("No images loaded.\nClick Open to select a folder."),
                )
                .into_any_element()
        } else {
            div()
                .flex_1()
                .overflow_y_scrollbar()
                .children(self.image_names.iter().enumerate().map(|(i, name)| {
                    let is_selected = self.selected_index == Some(i);
                    let entity = entity.clone();

                    div()
                        .id(("file-item", i))
                        .px_2()
                        .py_1()
                        .w_full()
                        .text_sm()
                        .cursor_pointer()
                        .when(is_selected, |el| {
                            el.bg(cx.theme().accent)
                                .text_color(cx.theme().accent_foreground)
                        })
                        .when(!is_selected, |el| el.hover(|el| el.bg(cx.theme().muted)))
                        .on_click(move |_, _window, cx| {
                            entity.update(cx, |this, cx| this.select_image(i, cx));
                        })
                        .child(name.clone())
                }))
                .into_any_element()
        };

        v_flex()
            .w(px(250.))
            .h_full()
            .border_r_1()
            .border_color(cx.theme().border)
            .bg(cx.theme().background)
            .child(header)
            .child(div().h(px(1.)).bg(cx.theme().border))
            .child(file_list)
    }

    fn render_preview(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let content = if let Some(ref preview) = self.preview_image {
            div()
                .size_full()
                .flex()
                .items_center()
                .justify_center()
                .child(
                    img(gpui::ImageSource::Render(preview.clone()))
                        .max_w_full()
                        .max_h_full()
                        .object_fit(ObjectFit::Contain),
                )
                .into_any_element()
        } else {
            div()
                .size_full()
                .flex()
                .items_center()
                .justify_center()
                .child(
                    div()
                        .text_base()
                        .text_color(cx.theme().muted_foreground)
                        .child("Select an image to preview"),
                )
                .into_any_element()
        };

        div()
            .flex_1()
            .min_w(px(0.))
            .min_h(px(0.))
            .h_full()
            .overflow_hidden()
            .p_1()
            .child(content)
    }

    fn render_sorting(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let entity = cx.entity().clone();

        let section_title = |label: &str| -> AnyElement {
            div()
                .text_sm()
                .font_weight(FontWeight::MEDIUM)
                .text_color(cx.theme().foreground)
                .child(label.to_string())
                .into_any_element()
        };

        let folder_label: SharedString = if let Some(ref dir) = self.sorting_source_dir {
            dir.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("Selected")
                .to_string()
                .into()
        } else {
            "Not set".into()
        };

        let folder_section = v_flex()
            .gap_2()
            .child(section_title("Source Folder"))
            .child(
                h_flex()
                    .gap_2()
                    .items_center()
                    .child(div().text_sm().child(folder_label))
                    .child(div().flex_1())
                    .child(
                        Button::new("sorting-browse-btn")
                            .label("Browse")
                            .small()
                            .on_click({
                                let entity = entity.clone();
                                move |_, _, cx| {
                                    entity
                                        .update(cx, |this, cx| this.choose_sorting_source_dir(cx));
                                }
                            }),
                    ),
            );

        let mode_section = v_flex().gap_2().child(section_title("Mode")).child(
            Checkbox::new("sorting-copy-files")
                .checked(self.sorting_copy_files)
                .label("Copy files instead of moving them")
                .on_click({
                    let entity = entity.clone();
                    move |checked, _, cx| {
                        entity.update(cx, |this, cx| {
                            this.sorting_copy_files = *checked;
                            this.sorting_results.clear();
                            this.sorting_summary = None;
                            this.sorting_status_message = if *checked {
                                "Copy mode enabled - originals will stay in place".into()
                            } else {
                                "Move mode enabled - files will be moved into number folders".into()
                            };
                            cx.notify();
                        });
                    }
                }),
        );

        let examples_section = v_flex()
            .gap_1()
            .child(section_title("Matching Examples"))
            .child(
                div()
                    .text_sm()
                    .text_color(cx.theme().muted_foreground)
                    .child("82.jpg, 82 (2).jpg, 82-001.jpg, 82-002 (2).jpg → folder 82"),
            );

        let start_button = if self.is_sorting {
            Button::new("sorting-start-btn")
                .label("Sorting...")
                .primary()
                .w_full()
                .disabled(true)
        } else {
            Button::new("sorting-start-btn")
                .label(if self.sorting_copy_files {
                    "Copy Into Number Folders"
                } else {
                    "Move Into Number Folders"
                })
                .primary()
                .w_full()
                .on_click({
                    let entity = entity.clone();
                    move |_, _, cx| {
                        entity.update(cx, |this, cx| this.start_sorting(cx));
                    }
                })
        };

        let summary_section = if let Some(ref summary) = self.sorting_summary {
            v_flex()
                .gap_1()
                .child(section_title("Last Run"))
                .child(
                    div()
                        .text_sm()
                        .text_color(cx.theme().muted_foreground)
                        .child(format!(
                            "{} discovered, {} sorted, {} failed, {} skipped, {} folders created",
                            summary.discovered,
                            summary.sorted,
                            summary.failed,
                            summary.skipped,
                            summary.folders_created
                        )),
                )
                .into_any_element()
        } else {
            div().into_any_element()
        };

        let failures_section = {
            let failed: Vec<_> = self
                .sorting_results
                .iter()
                .filter(|result| !result.success)
                .take(8)
                .collect();

            if failed.is_empty() {
                div().into_any_element()
            } else {
                v_flex()
                    .gap_1()
                    .child(section_title("Failures"))
                    .children(failed.into_iter().map(|result| {
                        let name = result
                            .source
                            .file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or("?");
                        let destination = result
                            .destination
                            .as_ref()
                            .map(|path| path.display().to_string())
                            .unwrap_or_else(|| "no destination".to_string());
                        let number = result.number.as_deref().unwrap_or("?");
                        let error = result.error.as_deref().unwrap_or("unknown error");
                        div()
                            .text_xs()
                            .text_color(gpui::hsla(0.0, 0.8, 0.5, 1.0))
                            .child(format!(
                                "❌ {name} → {destination} (number {number}): {error}"
                            ))
                    }))
                    .into_any_element()
            }
        };

        let divider = || div().h(px(1.)).bg(cx.theme().border);

        let panel = v_flex()
            .gap_4()
            .p_6()
            .max_w(px(720.))
            .child(
                div()
                    .text_2xl()
                    .font_weight(FontWeight::SEMIBOLD)
                    .child("Sort Numbered Files"),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(cx.theme().muted_foreground)
                    .child("Select a folder and sort files into directories based on the leading number in each filename."),
            )
            .child(divider())
            .child(folder_section)
            .child(divider())
            .child(mode_section)
            .child(divider())
            .child(examples_section)
            .child(divider())
            .child(start_button)
            .when(self.is_sorting, |el| {
                el.child(Progress::new("sorting-progress").value(self.sorting_progress * 100.0))
            })
            .child(
                div()
                    .text_sm()
                    .text_color(cx.theme().muted_foreground)
                    .child(self.sorting_status_message.clone()),
            )
            .child(summary_section)
            .child(failures_section);

        div()
            .size_full()
            .overflow_y_scrollbar()
            .flex()
            .justify_center()
            .child(panel)
    }

    fn render_settings(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let entity = cx.entity().clone();
        let section_title = |label: &str| -> AnyElement {
            div()
                .text_sm()
                .font_weight(FontWeight::MEDIUM)
                .text_color(cx.theme().foreground)
                .child(label.to_string())
                .into_any_element()
        };

        let format_section = v_flex()
            .gap_1()
            .child(section_title("Output Format"))
            .child(self.format_select.clone());

        let quality_section = v_flex()
            .gap_1()
            .child(section_title(&format!("JPEG Quality: {}%", self.quality)))
            .child(Slider::new(&self.quality_slider).w_full());

        let rotation_section = v_flex()
            .gap_1()
            .child(section_title("Rotation"))
            .child(self.rotation_select.clone());

        let mut text_section = v_flex().gap_2().child(
            Checkbox::new("text-enabled")
                .checked(self.text_enabled)
                .label("Add Text Overlay")
                .on_click({
                    let entity = entity.clone();
                    move |checked, _, cx| {
                        entity.update(cx, |this, cx| {
                            this.text_enabled = *checked;
                            this.schedule_preview_update(cx);
                            cx.notify();
                        });
                    }
                }),
        );

        if self.text_enabled {
            text_section = text_section
                .child(
                    v_flex()
                        .gap_1()
                        .child(section_title("Text ({filename} = file name)"))
                        .child(Input::new(&self.text_input).small()),
                )
                .child(
                    v_flex()
                        .gap_1()
                        .child(section_title("Position"))
                        .child(self.position_select.clone()),
                )
                .child(
                    v_flex()
                        .gap_1()
                        .child(section_title(&format!(
                            "Font Size: {:.0}px",
                            self.text_font_size
                        )))
                        .child(Slider::new(&self.font_size_slider).w_full()),
                )
                .child(
                    v_flex()
                        .gap_1()
                        .child(section_title(&format!(
                            "Edge Margin: {} px",
                            self.text_edge_margin
                        )))
                        .child(Slider::new(&self.text_margin_slider).w_full()),
                )
                .child(
                    v_flex().gap_1().child(section_title("Text Color")).child(
                        h_flex()
                            .items_center()
                            .gap_2()
                            .child(
                                div()
                                    .p_1()
                                    .border_1()
                                    .rounded_md()
                                    .border_color(cx.theme().border)
                                    .bg(cx.theme().background)
                                    .child(ColorPicker::new(&self.color_picker)),
                            )
                            .child(div().flex_1()),
                    ),
                );
        }

        let mut poster_section = v_flex()
            .gap_2()
            .child(section_title("Poster Options"))
            .child(
                Checkbox::new("poster-33x66")
                    .checked(self.poster_resize_33x66)
                    .label("Resize/Crop to 33 x 66 cm (300 DPI)")
                    .on_click({
                        let entity = entity.clone();
                        move |checked, _, cx| {
                            entity.update(cx, |this, cx| {
                                this.poster_resize_33x66 = *checked;
                                this.schedule_preview_update(cx);
                                cx.notify();
                            });
                        }
                    }),
            )
            .child(
                Checkbox::new("poster-margin")
                    .checked(self.poster_margin_enabled)
                    .label("Add white border margin")
                    .on_click({
                        let entity = entity.clone();
                        move |checked, _, cx| {
                            entity.update(cx, |this, cx| {
                                this.poster_margin_enabled = *checked;
                                this.schedule_preview_update(cx);
                                cx.notify();
                            });
                        }
                    }),
            );
        if self.poster_margin_enabled {
            poster_section = poster_section.child(
                v_flex()
                    .gap_1()
                    .child(section_title(&format!(
                        "Margin: {} px",
                        self.poster_margin_px
                    )))
                    .child(Slider::new(&self.poster_margin_slider).w_full()),
            );
        }

        let mut watermark_section = v_flex().gap_2().child(section_title("Watermark")).child(
            Checkbox::new("watermark-enabled")
                .checked(self.watermark_enabled)
                .label("Add watermark")
                .on_click({
                    let entity = entity.clone();
                    move |checked, _, cx| {
                        entity.update(cx, |this, cx| {
                            this.watermark_enabled = *checked;
                            this.schedule_preview_update(cx);
                            cx.notify();
                        });
                    }
                }),
        );
        if self.watermark_enabled {
            let watermark_label: SharedString = self
                .watermark_image_path
                .as_ref()
                .and_then(|p| p.file_name().and_then(|n| n.to_str()))
                .unwrap_or("Not selected")
                .to_string()
                .into();
            watermark_section = watermark_section
                .child(
                    h_flex()
                        .gap_2()
                        .items_center()
                        .child(div().text_sm().child(watermark_label))
                        .child(div().flex_1())
                        .child(
                            Button::new("watermark-browse-btn")
                                .label("Choose Image")
                                .small()
                                .on_click({
                                    let entity = entity.clone();
                                    move |_, _, cx| {
                                        entity
                                            .update(cx, |this, cx| this.choose_watermark_image(cx));
                                    }
                                }),
                        ),
                )
                .child(
                    v_flex()
                        .gap_1()
                        .child(section_title("Position"))
                        .child(self.watermark_position_select.clone()),
                )
                .child(
                    v_flex()
                        .gap_1()
                        .child(section_title("Rotation"))
                        .child(self.watermark_rotation_select.clone()),
                )
                .child(
                    v_flex()
                        .gap_1()
                        .child(section_title(&format!(
                            "Edge Margin: {} px",
                            self.watermark_edge_margin
                        )))
                        .child(Slider::new(&self.watermark_margin_slider).w_full()),
                )
                .child(
                    v_flex()
                        .gap_1()
                        .child(section_title(&format!(
                            "Size: {:.0}%",
                            self.watermark_scale_percent
                        )))
                        .child(Slider::new(&self.watermark_scale_slider).w_full()),
                )
                .child(
                    v_flex()
                        .gap_1()
                        .child(section_title(&format!(
                            "Opacity: {:.0}%",
                            self.watermark_opacity * 100.0
                        )))
                        .child(Slider::new(&self.watermark_opacity_slider).w_full()),
                );
        }

        let output_label: SharedString = if let Some(ref dir) = self.output_dir {
            dir.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("Selected")
                .to_string()
                .into()
        } else {
            "Not set".into()
        };

        let output_section = v_flex()
            .gap_1()
            .child(section_title("Output Folder"))
            .child(
                h_flex()
                    .gap_2()
                    .items_center()
                    .child(div().text_sm().child(output_label))
                    .child(div().flex_1())
                    .child(Button::new("browse-btn").label("Browse").small().on_click({
                        let entity = entity.clone();
                        move |_, _, cx| {
                            entity.update(cx, |this, cx| this.choose_output_dir(cx));
                        }
                    })),
            );

        let process_button = if self.is_processing {
            Button::new("process-btn")
                .label("Processing...")
                .w_full()
                .disabled(true)
        } else {
            Button::new("process-btn")
                .label("Process All")
                .primary()
                .w_full()
                .on_click({
                    let entity = entity.clone();
                    move |_, _, cx| {
                        entity.update(cx, |this, cx| this.start_batch(cx));
                    }
                })
        };

        let results_el = if self.batch_results.is_empty() {
            div().into_any_element()
        } else {
            let failed: Vec<_> = self.batch_results.iter().filter(|r| !r.success).collect();
            if failed.is_empty() {
                div()
                    .text_sm()
                    .text_color(gpui::hsla(0.33, 0.9, 0.4, 1.0))
                    .child("All images processed successfully!")
                    .into_any_element()
            } else {
                v_flex()
                    .gap_0p5()
                    .children(failed.iter().take(5).map(|r| {
                        let name = r.source.file_name().and_then(|n| n.to_str()).unwrap_or("?");
                        let err = r.error.as_deref().unwrap_or("unknown");
                        div()
                            .text_xs()
                            .text_color(gpui::hsla(0.0, 0.8, 0.5, 1.0))
                            .child(format!("❌ {name}: {err}"))
                    }))
                    .into_any_element()
            }
        };

        let divider = || div().h(px(1.)).bg(cx.theme().border);

        let settings_col = v_flex()
            .flex_none()
            .w_full()
            .gap_3()
            .p_3()
            .pb_12()
            .child(format_section)
            .child(divider())
            .child(quality_section)
            .child(divider())
            .child(rotation_section)
            .child(divider())
            .child(text_section)
            .child(divider())
            .child(poster_section)
            .child(divider())
            .child(watermark_section)
            .child(divider())
            .child(output_section)
            .child(divider())
            .child(process_button)
            .child(results_el);

        v_flex()
            .w(px(280.))
            .flex_none()
            .self_stretch()
            .min_h(px(0.))
            .border_l_1()
            .border_color(cx.theme().border)
            .bg(cx.theme().background)
            .relative()
            .overflow_hidden()
            .child(
                div()
                    .id("settings-scroll-area")
                    .size_full()
                    .overflow_y_scroll()
                    .track_scroll(&self.settings_scroll_handle)
                    .child(settings_col),
            )
            .child(
                div().absolute().top_0().right_0().bottom_0().child(
                    Scrollbar::vertical(&self.settings_scroll_handle)
                        .id("settings-scrollbar")
                        .scrollbar_show(ScrollbarShow::Always),
                ),
            )
    }
}

impl Render for App {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let mode = self.mode;
        let entity = cx.entity().clone();
        let is_batch = mode == AppMode::BatchProcessing;
        let is_numbering = mode == AppMode::Numbering;
        let is_sorting = mode == AppMode::Sorting;

        let tab_style = |active: bool, theme: &gpui_component::theme::Theme| {
            let base = div().px_4().py_2().text_sm().cursor_pointer().border_b_2();
            if active {
                base.border_color(theme.accent)
                    .text_color(theme.foreground)
                    .font_weight(FontWeight::SEMIBOLD)
            } else {
                base.border_color(transparent_black())
                    .text_color(theme.muted_foreground)
                    .hover(|el| el.text_color(theme.foreground))
            }
        };

        let theme = cx.theme().clone();
        let tabs = h_flex()
            .bg(theme.background)
            .border_b_1()
            .border_color(theme.border)
            .child(
                tab_style(is_batch, &theme)
                    .id("tab-batch")
                    .child("Batch Processing")
                    .on_click({
                        let entity = entity.clone();
                        move |_, _, cx| {
                            entity
                                .update(cx, |this, cx| this.set_mode(AppMode::BatchProcessing, cx));
                        }
                    }),
            )
            .child(
                tab_style(is_numbering, &theme)
                    .id("tab-numbering")
                    .child("Numbering")
                    .on_click({
                        let entity = entity.clone();
                        move |_, _, cx| {
                            entity.update(cx, |this, cx| this.set_mode(AppMode::Numbering, cx));
                        }
                    }),
            )
            .child(
                tab_style(is_sorting, &theme)
                    .id("tab-sorting")
                    .child("Sorting")
                    .on_click({
                        let entity = entity.clone();
                        move |_, _, cx| {
                            entity.update(cx, |this, cx| this.set_mode(AppMode::Sorting, cx));
                        }
                    }),
            );

        let content = match mode {
            AppMode::BatchProcessing => {
                let main_row = h_flex()
                    .flex_1()
                    .min_w(px(0.))
                    .min_h(px(0.))
                    .overflow_hidden()
                    .child(self.render_sidebar(cx))
                    .child(self.render_preview(cx))
                    .child(self.render_settings(cx));

                let status_bar = h_flex()
                    .px_2()
                    .py_1()
                    .gap_2()
                    .items_center()
                    .bg(cx.theme().background)
                    .border_t_1()
                    .border_color(cx.theme().border)
                    .child(
                        div()
                            .text_xs()
                            .text_color(cx.theme().muted_foreground)
                            .child(self.status_message.clone()),
                    )
                    .child(div().flex_1())
                    .when(self.is_processing, |el| {
                        el.child(
                            div()
                                .w(px(200.))
                                .child(Progress::new("progress").value(self.progress * 100.0)),
                        )
                    });

                v_flex()
                    .flex_1()
                    .min_h(px(0.))
                    .overflow_hidden()
                    .child(main_row)
                    .child(status_bar.flex_none())
                    .into_any_element()
            }
            AppMode::Numbering => div()
                .flex_1()
                .child(self.numbering_mode.clone())
                .into_any_element(),
            AppMode::Sorting => self.render_sorting(cx).into_any_element(),
        };

        v_flex()
            .size_full()
            .bg(cx.theme().background)
            .text_color(cx.theme().foreground)
            .child(tabs)
            .child(content)
    }
}

fn hsla_to_rgb(h: f32, s: f32, l: f32) -> (u8, u8, u8) {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h6 = h * 6.0;
    let x = c * (1.0 - (h6 % 2.0 - 1.0).abs());
    let m = l - c / 2.0;

    let (r1, g1, b1) = if h6 < 1.0 {
        (c, x, 0.0)
    } else if h6 < 2.0 {
        (x, c, 0.0)
    } else if h6 < 3.0 {
        (0.0, c, x)
    } else if h6 < 4.0 {
        (0.0, x, c)
    } else if h6 < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    (
        ((r1 + m) * 255.0).round() as u8,
        ((g1 + m) * 255.0).round() as u8,
        ((b1 + m) * 255.0).round() as u8,
    )
}
