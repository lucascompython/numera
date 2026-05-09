use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use super::image_ops::{self, Rotation, TextOverlayConfig, TextPosition};

/// What format to export as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Jpeg,
    Pdf,
}

#[derive(Default)]
pub struct PosterOptions {
    pub resize_to_33x66cm: bool,
    pub margin_px: u32,
}

pub struct WatermarkConfig {
    pub image_path: PathBuf,
    pub position: TextPosition,
    pub margin: u32,
    pub scale_percent: f32,
    pub opacity: f32, // 0.0..=1.0
    pub rotation: WatermarkRotation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatermarkRotation {
    MatchImage,
    Fixed(Rotation),
}

impl Default for WatermarkRotation {
    fn default() -> Self {
        Self::MatchImage
    }
}

impl WatermarkRotation {
    pub fn resolve(self, image_rotation: Rotation) -> Rotation {
        match self {
            Self::MatchImage => image_rotation,
            Self::Fixed(rotation) => rotation,
        }
    }
}

/// Full configuration for a batch run.
pub struct BatchConfig {
    pub quality: u8,
    pub rotation: Rotation,
    pub text_overlay: Option<TextOverlayConfig>,
    pub poster: PosterOptions,
    pub watermark: Option<WatermarkConfig>,
    pub output_format: OutputFormat,
    pub output_dir: PathBuf,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            quality: 70,
            rotation: Rotation::None,
            text_overlay: None,
            poster: PosterOptions::default(),
            watermark: None,
            output_format: OutputFormat::Pdf,
            output_dir: PathBuf::new(),
        }
    }
}

/// Result of processing a single image.
#[derive(Debug, Clone)]
pub struct ProcessResult {
    pub source: PathBuf,
    pub success: bool,
    pub error: Option<String>,
}

/// Process a batch of images in parallel.
///
/// Returns a vector of results (one per input path) and calls `progress_callback`
/// after each image is done, with (completed_count, total_count).
pub fn process_batch(
    paths: &[PathBuf],
    config: &BatchConfig,
    progress_callback: impl Fn(usize, usize) + Send + Sync,
) -> Vec<ProcessResult> {
    let total = paths.len();
    let completed = Arc::new(AtomicUsize::new(0));

    // Pre-render a shared text stamp if the template doesn't contain {filename}
    let shared_stamp = config.text_overlay.as_ref().and_then(|tc| {
        if !tc.text_template.contains("{filename}") {
            Some(image_ops::render_text_stamp(tc, ""))
        } else {
            None
        }
    });
    let shared_watermark_stamp = if let Some(wm) = config.watermark.as_ref() {
        match image_ops::load_watermark_stamp(
            &wm.image_path,
            wm.opacity,
            wm.scale_percent,
            wm.rotation.resolve(config.rotation),
        ) {
            Ok(stamp) => Some(Arc::new(stamp)),
            Err(err) => {
                return paths
                    .iter()
                    .map(|source| ProcessResult {
                        source: source.clone(),
                        success: false,
                        error: Some(err.clone()),
                    })
                    .collect();
            }
        }
    } else {
        None
    };

    // Process all images in parallel natively without pre-loading into a massive memory vector.
    paths
        .par_iter()
        .map(|source| {
            let load_result = image_ops::load_image(source);
            let result = process_single(
                source,
                load_result,
                config,
                shared_stamp.as_ref(),
                shared_watermark_stamp.as_deref(),
            );
            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            progress_callback(done, total);
            result
        })
        .collect()
}

fn process_single(
    source: &Path,
    load_result: Result<image::DynamicImage, String>,
    config: &BatchConfig,
    shared_stamp: Option<&image::RgbaImage>,
    shared_watermark_stamp: Option<&image::RgbaImage>,
) -> ProcessResult {
    let stem = source
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("image");

    let mut img = match load_result {
        Ok(img) => img,
        Err(e) => {
            return ProcessResult {
                source: source.to_path_buf(),
                success: false,
                error: Some(e),
            };
        }
    };

    // Apply rotation
    if config.rotation != Rotation::None {
        img = image_ops::rotate_image(&img, config.rotation);
    }
    img = apply_export_effects(img, stem, config, shared_stamp, shared_watermark_stamp);

    // Export
    let output_path = match config.output_format {
        OutputFormat::Jpeg => config.output_dir.join(format!("{stem}.jpg")),
        OutputFormat::Pdf => config.output_dir.join(format!("{stem}.pdf")),
    };

    let result = match config.output_format {
        OutputFormat::Jpeg => image_ops::save_jpeg(&img, &output_path, config.quality),
        OutputFormat::Pdf => {
            image_ops::export_single_image_to_pdf(&img, &output_path, config.quality, 300.0)
        }
    };

    match result {
        Ok(()) => ProcessResult {
            source: source.to_path_buf(),
            success: true,
            error: None,
        },
        Err(e) => ProcessResult {
            source: source.to_path_buf(),
            success: false,
            error: Some(e),
        },
    }
}

pub fn apply_export_effects(
    img: image::DynamicImage,
    filename: &str,
    config: &BatchConfig,
    shared_text_stamp: Option<&image::RgbaImage>,
    shared_watermark_stamp: Option<&image::RgbaImage>,
) -> image::DynamicImage {
    apply_effects_internal(
        img,
        filename,
        config,
        shared_text_stamp,
        shared_watermark_stamp,
        false,
    )
}

pub fn apply_preview_effects(
    img: image::DynamicImage,
    filename: &str,
    config: &BatchConfig,
    shared_text_stamp: Option<&image::RgbaImage>,
    shared_watermark_stamp: Option<&image::RgbaImage>,
) -> image::DynamicImage {
    apply_effects_internal(
        img,
        filename,
        config,
        shared_text_stamp,
        shared_watermark_stamp,
        true,
    )
}

fn apply_effects_internal(
    mut img: image::DynamicImage,
    filename: &str,
    config: &BatchConfig,
    shared_text_stamp: Option<&image::RgbaImage>,
    shared_watermark_stamp: Option<&image::RgbaImage>,
    preview_mode: bool,
) -> image::DynamicImage {
    if config.poster.resize_to_33x66cm {
        img = if preview_mode {
            // Preview fast-path: keep current decoded resolution and only apply centered aspect crop.
            image_ops::crop_to_33x66cm_poster_aspect(img)
        } else {
            image_ops::resize_to_33x66cm_poster(img)
        };
    }
    if config.poster.margin_px > 0 {
        img = image_ops::add_white_border(img, config.poster.margin_px);
    }

    // Apply text overlay using cached stamp when possible
    if let Some(ref text_config) = config.text_overlay {
        let effective = text_config.clone();
        if let Some(stamp) = shared_text_stamp {
            img = image_ops::overlay_text_with_stamp(img, &effective, stamp);
        } else {
            let stamp = image_ops::render_text_stamp(&effective, filename);
            img = image_ops::overlay_text_with_stamp(img, &effective, &stamp);
        }
    }

    // apply watermark (after text)
    if let Some(ref watermark) = config.watermark {
        if let Some(stamp) = shared_watermark_stamp {
            img = image_ops::overlay_stamp_with_position(
                img,
                watermark.position,
                watermark.margin,
                stamp,
            );
        } else if let Ok(stamp) = image_ops::load_watermark_stamp(
            &watermark.image_path,
            watermark.opacity,
            watermark.scale_percent,
            watermark.rotation.resolve(config.rotation),
        ) {
            img = image_ops::overlay_stamp_with_position(
                img,
                watermark.position,
                watermark.margin,
                &stamp,
            );
        }
    }

    img
}
