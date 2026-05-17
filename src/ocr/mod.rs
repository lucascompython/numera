//! OCR module for automatic motorcycle number recognition.
//!
//! Uses PaddleOCR ONNX models via oar-ocr (ONNX Runtime backend).

use image::imageops::FilterType;
use image::{DynamicImage, GrayImage};
use imageproc::template_matching::{MatchTemplateMethod, find_extremes, match_template_parallel};
use oar_ocr::core::config::{OrtExecutionProvider, OrtGraphOptimizationLevel, OrtSessionConfig};
use oar_ocr::oarocr::{OAROCR, OAROCRBuilder};
use rapidhash::fast::RapidHasher;
use scc::HashMap as SccHashMap;
use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::path::{Path, PathBuf};
use std::sync::{OnceLock, RwLock};

static OCR_ENGINE: OnceLock<Option<OAROCR>> = OnceLock::new();
static STICKER_TEMPLATE: OnceLock<RwLock<Option<StickerTemplate>>> = OnceLock::new();
static OCR_CACHE: OnceLock<
    SccHashMap<PathBuf, OcrResult, BuildHasherDefault<RapidHasher<'static>>>,
> = OnceLock::new();

const OCR_MAX_SIDE: u32 = 1800;
const MATCH_MAX_SIDE: u32 = 900;
const STICKER_MIN_SCORE: f32 = 0.27;
const STICKER_SCALES: &[f32] = &[0.22, 0.34, 0.48, 0.66];
const DEFAULT_NUMBER_ROI: NormalizedRect = NormalizedRect {
    x: 0.58,
    y: 0.04,
    w: 0.38,
    h: 0.45,
};
const OCR_CACHE_MAX_SIZE: usize = 16_384;

#[derive(Debug, Clone)]
pub struct OcrResult {
    /// Detected text (filtered to digits only for motorcycle numbers)
    pub text: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// All detected text boxes with their confidence
    pub all_detections: Vec<Detection>,
}

/// A single text detection
#[derive(Debug, Clone)]
pub struct Detection {
    pub text: String,
    pub confidence: f32,
}

#[derive(Debug)]
struct ModelPaths {
    det_model: PathBuf,
    rec_model: PathBuf,
    dictionary: PathBuf,
    text_line_ori_model: Option<PathBuf>,
    doc_ori_model: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct StickerTemplate {
    path: PathBuf,
    grayscale: GrayImage,
    number_roi: NormalizedRect,
}

#[derive(Debug, Clone, Copy)]
struct NormalizedRect {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

#[derive(Debug, Clone, Copy)]
struct MatchRect {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    score: f32,
}

#[derive(Debug, Clone, Copy)]
struct PixelRect {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

#[derive(Debug, Clone, Copy, Default)]
struct CandidateStats {
    max_confidence: f32,
    count: usize,
    template_hits: usize,
}

pub fn init_ocr() -> Result<(), String> {
    OCR_ENGINE.get_or_init(|| {
        let model_dir = model_dir();
        let paths = match resolve_model_paths(&model_dir) {
            Ok(paths) => paths,
            Err(e) => {
                eprintln!("{e}");
                return None;
            }
        };

        let preferred = ocr_ort_config_from_env();
        let preferred_provider_label = preferred
            .as_ref()
            .and_then(ort_provider_label)
            .unwrap_or("cpu");

        match build_ocr_engine(&paths, preferred) {
            Ok(engine) => {
                eprintln!(
                    "OCR initialized from {} (provider: {})",
                    model_dir.display(),
                    preferred_provider_label
                );
                Some(engine)
            }
            Err(err) => {
                if !preferred_provider_label.eq_ignore_ascii_case("cpu") {
                    eprintln!(
                        "OCR provider '{}' failed: {}. Falling back to CPU.",
                        preferred_provider_label, err
                    );
                    match build_ocr_engine(&paths, Some(cpu_ort_config())) {
                        Ok(engine) => {
                            eprintln!(
                                "OCR initialized from {} (provider: cpu fallback)",
                                model_dir.display()
                            );
                            return Some(engine);
                        }
                        Err(fallback_err) => {
                            eprintln!(
                                "Failed to initialize OCR engine (fallback): {}",
                                fallback_err
                            );
                            return None;
                        }
                    }
                }

                eprintln!("Failed to initialize OCR engine: {}", err);
                None
            }
        }
    });

    if OCR_ENGINE.get().map(|e| e.is_some()).unwrap_or(false) {
        Ok(())
    } else {
        Err("OCR engine initialization failed".into())
    }
}

fn build_ocr_engine(
    paths: &ModelPaths,
    ort_config: Option<OrtSessionConfig>,
) -> Result<OAROCR, String> {
    let mut builder = OAROCRBuilder::new(&paths.det_model, &paths.rec_model, &paths.dictionary)
        .image_batch_size(1)
        .region_batch_size(16);

    if let Some(config) = ort_config {
        builder = builder.ort_session(config);
    }

    if let Some(model) = paths.doc_ori_model.as_ref() {
        builder = builder.with_document_image_orientation_classification(model);
    } else {
        eprintln!("OCR: doc orientation model not found, continuing without it");
    }

    if let Some(model) = paths.text_line_ori_model.as_ref() {
        builder = builder.with_text_line_orientation_classification(model);
    } else {
        eprintln!("OCR: text-line orientation model not found, continuing without it");
    }

    builder
        .build()
        .map_err(|e| format!("failed to build OCR runtime: {e}"))
}

fn ocr_intra_threads() -> usize {
    std::env::var("BIP_OCR_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or_else(default_ocr_intra_threads)
}

fn default_ocr_intra_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 8)
}

fn cpu_ort_config() -> OrtSessionConfig {
    let threads = ocr_intra_threads();

    OrtSessionConfig::new()
        .with_optimization_level(OrtGraphOptimizationLevel::All)
        .with_intra_threads(threads)
        .with_inter_threads(1)
        .with_parallel_execution(true)
        .with_memory_pattern(true)
        .with_execution_providers(vec![OrtExecutionProvider::CPU])
}

fn ocr_ort_config_from_env() -> Option<OrtSessionConfig> {
    let device = std::env::var("BIP_OCR_DEVICE")
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "cpu".to_string());

    let threads = ocr_intra_threads();

    let mut cfg = OrtSessionConfig::new()
        .with_optimization_level(OrtGraphOptimizationLevel::All)
        .with_intra_threads(threads)
        .with_inter_threads(1)
        .with_parallel_execution(true)
        .with_memory_pattern(true);

    let providers = parse_execution_providers(&device)?;
    cfg = cfg.with_execution_providers(providers);
    Some(cfg)
}

fn parse_execution_providers(device: &str) -> Option<Vec<OrtExecutionProvider>> {
    let mut provider_split = device.split(':');
    let provider = provider_split.next()?.trim();
    let arg = provider_split.next().map(str::trim);

    let providers = match provider {
        "cpu" => vec![OrtExecutionProvider::CPU],
        "cuda" | "gpu" => {
            let device_id = arg.and_then(|id| id.parse::<i32>().ok()).or(Some(0));
            vec![
                OrtExecutionProvider::CUDA {
                    device_id,
                    gpu_mem_limit: None,
                    arena_extend_strategy: None,
                    cudnn_conv_algo_search: None,
                    cudnn_conv_use_max_workspace: None,
                },
                OrtExecutionProvider::CPU,
            ]
        }
        "directml" | "dml" => {
            let device_id = arg.and_then(|id| id.parse::<i32>().ok()).or(Some(0));
            vec![
                OrtExecutionProvider::DirectML { device_id },
                OrtExecutionProvider::CPU,
            ]
        }
        "openvino" => vec![
            OrtExecutionProvider::OpenVINO {
                device_type: arg.map(|s| s.to_string()),
                num_threads: None,
            },
            OrtExecutionProvider::CPU,
        ],
        "tensorrt" | "trt" => {
            let device_id = arg.and_then(|id| id.parse::<i32>().ok()).or(Some(0));
            vec![
                OrtExecutionProvider::TensorRT {
                    device_id,
                    max_workspace_size: None,
                    min_subgraph_size: None,
                    fp16_enable: Some(true),
                    timing_cache: Some(true),
                    timing_cache_path: None,
                    force_timing_cache: None,
                    engine_cache: Some(true),
                    engine_cache_path: None,
                    dump_ep_context_model: None,
                    ep_context_file_path: None,
                },
                OrtExecutionProvider::CPU,
            ]
        }
        "coreml" => vec![
            OrtExecutionProvider::CoreML {
                ane_only: None,
                subgraphs: Some(true),
            },
            OrtExecutionProvider::CPU,
        ],
        "webgpu" => vec![OrtExecutionProvider::WebGPU, OrtExecutionProvider::CPU],
        other => {
            eprintln!("OCR: unknown BIP_OCR_DEVICE '{}' - using CPU", other);
            vec![OrtExecutionProvider::CPU]
        }
    };

    Some(providers)
}

fn ort_provider_label(config: &OrtSessionConfig) -> Option<&'static str> {
    let first = config.get_execution_providers().into_iter().next()?;
    let label = match first {
        OrtExecutionProvider::CPU => "cpu",
        OrtExecutionProvider::CUDA { .. } => "cuda",
        OrtExecutionProvider::DirectML { .. } => "directml",
        OrtExecutionProvider::OpenVINO { .. } => "openvino",
        OrtExecutionProvider::TensorRT { .. } => "tensorrt",
        OrtExecutionProvider::CoreML { .. } => "coreml",
        OrtExecutionProvider::WebGPU => "webgpu",
    };
    Some(label)
}

/// Set a sticker template image to bias OCR toward the current event sticker.
pub fn set_sticker_template(path: &Path) -> Result<(), String> {
    let img = crate::processing::image_ops::load_image(path)?;
    let grayscale = prepare_template_image(&img);

    if grayscale.width() < 24 || grayscale.height() < 16 {
        return Err("Sticker template is too small".into());
    }

    let template = StickerTemplate {
        path: path.to_path_buf(),
        number_roi: infer_number_roi(&img),
        grayscale,
    };

    let mut lock = sticker_template_store()
        .write()
        .map_err(|_| "Failed to update sticker template".to_string())?;
    *lock = Some(template);
    Ok(())
}

/// Clear the current sticker template.
pub fn clear_sticker_template() {
    if let Ok(mut lock) = sticker_template_store().write() {
        *lock = None;
    }
}

/// Get the loaded sticker template filename (if any).
pub fn sticker_template_name() -> Option<String> {
    let lock = sticker_template_store().read().ok()?;
    lock.as_ref()?
        .path
        .file_name()
        .and_then(|n| n.to_str())
        .map(|s| s.to_string())
}

/// Check whether a sticker template is loaded.
pub fn has_sticker_template() -> bool {
    sticker_template_store()
        .read()
        .map(|lock| lock.is_some())
        .unwrap_or(false)
}

type OcrCacheMap = SccHashMap<PathBuf, OcrResult, BuildHasherDefault<RapidHasher<'static>>>;

fn ocr_cache() -> &'static OcrCacheMap {
    OCR_CACHE.get_or_init(OcrCacheMap::default)
}

pub fn get_cached_ocr(path: &Path) -> Option<OcrResult> {
    ocr_cache().read_sync(path, |_, v| v.clone())
}

fn cache_ocr_result(path: &Path, result: OcrResult) {
    let cache = ocr_cache();
    if cache.len() >= OCR_CACHE_MAX_SIZE {
        cache.clear_sync();
    }
    let _ = cache.insert_sync(path.to_path_buf(), result);
}

pub fn recognize_number_from_path(path: &Path) -> Option<OcrResult> {
    if let Some(cached) = get_cached_ocr(path) {
        return Some(cached);
    }

    let img = crate::processing::image_ops::load_image_for_preview(path, OCR_MAX_SIDE).ok()?;
    let result = recognize_number(&img)?;
    cache_ocr_result(path, result.clone());
    Some(result)
}

/// Run OCR on an image and extract likely motorcycle numbers.
///
/// Returns the best candidate number with confidence score.
pub fn recognize_number(img: &DynamicImage) -> Option<OcrResult> {
    let engine = OCR_ENGINE.get()?.as_ref()?;
    let resized = resize_for_ocr(img, OCR_MAX_SIDE);
    let working_img = resized.as_ref().unwrap_or(img);

    let mut grouped: HashMap<String, CandidateStats> = HashMap::new();
    let mut all_detections: Vec<Detection> = Vec::new();

    let template_crops = sticker_guided_crops(working_img);
    for crop in &template_crops {
        merge_detections(
            run_ocr_detections(engine, crop),
            true,
            &mut grouped,
            &mut all_detections,
        );
    }

    // If template-guided OCR found nothing usable, fallback to the full image.
    if grouped.is_empty() {
        merge_detections(
            run_ocr_detections(engine, working_img),
            false,
            &mut grouped,
            &mut all_detections,
        );
    }

    let best_number = grouped
        .into_iter()
        .max_by(|(digits_a, a), (digits_b, b)| {
            let score_a =
                candidate_score(digits_a.len(), a.max_confidence, a.count, a.template_hits);
            let score_b =
                candidate_score(digits_b.len(), b.max_confidence, b.count, b.template_hits);
            score_a.total_cmp(&score_b)
        })
        .map(|(digits, stats)| (digits, stats.max_confidence));

    best_number.map(|(text, confidence)| OcrResult {
        text,
        confidence,
        all_detections,
    })
}

fn resize_for_ocr(img: &DynamicImage, max_side: u32) -> Option<DynamicImage> {
    let max_dim = img.width().max(img.height());
    if max_dim <= max_side {
        return None;
    }

    let scale = max_side as f32 / max_dim as f32;
    let target_w = ((img.width() as f32 * scale).round() as u32).max(1);
    let target_h = ((img.height() as f32 * scale).round() as u32).max(1);
    Some(img.resize(target_w, target_h, FilterType::Triangle))
}

fn run_ocr_detections(engine: &OAROCR, img: &DynamicImage) -> Vec<Detection> {
    let Ok(results) = engine.predict(vec![img.to_rgb8()]) else {
        return Vec::new();
    };
    let Some(result) = results.first() else {
        return Vec::new();
    };

    result
        .text_regions
        .iter()
        .filter_map(|region| {
            region
                .text_with_confidence()
                .map(|(text, confidence)| Detection {
                    text: text.to_string(),
                    confidence,
                })
        })
        .collect()
}

fn merge_detections(
    detections: Vec<Detection>,
    from_template: bool,
    grouped: &mut HashMap<String, CandidateStats>,
    all_detections: &mut Vec<Detection>,
) {
    all_detections.extend(detections.iter().cloned());

    for detection in detections {
        let digits = digits_only(&detection.text);
        if !(1..=6).contains(&digits.len()) {
            continue;
        }

        let entry = grouped.entry(digits).or_default();
        entry.max_confidence = entry.max_confidence.max(detection.confidence);
        entry.count += 1;
        if from_template {
            entry.template_hits += 1;
        }
    }
}

fn candidate_score(len: usize, confidence: f32, count: usize, template_hits: usize) -> f32 {
    let len_bonus = match len {
        1 => 0.05,
        2 => 0.09,
        3 => 0.08,
        4 => 0.06,
        5 => 0.03,
        6 => 0.01,
        _ => 0.0,
    };
    let repeat_bonus = ((count.saturating_sub(1)) as f32 * 0.03).min(0.12);
    let template_bonus = ((template_hits as f32) * 0.10).min(0.35);
    confidence + len_bonus + repeat_bonus + template_bonus
}

/// Check if OCR is available
pub fn is_ocr_available() -> bool {
    OCR_ENGINE.get().map(|e| e.is_some()).unwrap_or(false)
}

fn sticker_guided_crops(img: &DynamicImage) -> Vec<DynamicImage> {
    let Some(template) = sticker_template_store()
        .read()
        .ok()
        .and_then(|lock| lock.clone())
    else {
        return Vec::new();
    };

    let input_gray = img.to_luma8();
    let (search_gray, scale_to_search) = downscale_for_matching(&input_gray, MATCH_MAX_SIDE);

    let Some(matched) = best_template_match(&search_gray, &template.grayscale) else {
        return Vec::new();
    };

    if matched.score < STICKER_MIN_SCORE {
        return Vec::new();
    }

    let inv_scale = if scale_to_search > 0.0 {
        1.0 / scale_to_search
    } else {
        1.0
    };
    let img_w = img.width();
    let img_h = img.height();

    let Some(sticker_rect) = clamp_rect(
        matched.x as f32 * inv_scale,
        matched.y as f32 * inv_scale,
        matched.w as f32 * inv_scale,
        matched.h as f32 * inv_scale,
        img_w,
        img_h,
    ) else {
        return Vec::new();
    };

    let mut crops: Vec<DynamicImage> = Vec::new();
    let mut seen: Vec<(u32, u32, u32, u32)> = Vec::new();

    let roi = template.number_roi;
    let roi_x = sticker_rect.x as f32 + roi.x * sticker_rect.w as f32;
    let roi_y = sticker_rect.y as f32 + roi.y * sticker_rect.h as f32;
    let roi_w = roi.w * sticker_rect.w as f32;
    let roi_h = roi.h * sticker_rect.h as f32;

    let candidate_rects = [
        clamp_rect(
            roi_x - roi_w * 0.30,
            roi_y - roi_h * 0.30,
            roi_w * 1.60,
            roi_h * 1.70,
            img_w,
            img_h,
        ),
        clamp_rect(
            roi_x - roi_w * 0.45,
            roi_y - roi_h * 0.45,
            roi_w * 1.95,
            roi_h * 2.00,
            img_w,
            img_h,
        ),
        clamp_rect(
            sticker_rect.x as f32 - sticker_rect.w as f32 * 0.10,
            sticker_rect.y as f32 - sticker_rect.h as f32 * 0.10,
            sticker_rect.w as f32 * 1.20,
            sticker_rect.h as f32 * 1.20,
            img_w,
            img_h,
        ),
    ];

    for rect in candidate_rects.into_iter().flatten() {
        let key = (rect.x, rect.y, rect.w, rect.h);
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        crops.push(img.crop_imm(rect.x, rect.y, rect.w, rect.h));
    }

    crops
}

fn best_template_match(image: &GrayImage, template: &GrayImage) -> Option<MatchRect> {
    let mut best: Option<MatchRect> = None;

    for scale in STICKER_SCALES {
        let tw = ((template.width() as f32 * scale).round() as u32).max(1);
        let th = ((template.height() as f32 * scale).round() as u32).max(1);

        if tw < 20 || th < 14 || tw >= image.width() || th >= image.height() {
            continue;
        }

        let resized_template = image::imageops::resize(template, tw, th, FilterType::Triangle);
        if resized_template.width() >= image.width() || resized_template.height() >= image.height()
        {
            continue;
        }

        let result = match_template_parallel(
            image,
            &resized_template,
            MatchTemplateMethod::CrossCorrelationNormalized,
        );
        let extremes = find_extremes(&result);

        let candidate = MatchRect {
            x: extremes.max_value_location.0,
            y: extremes.max_value_location.1,
            w: resized_template.width(),
            h: resized_template.height(),
            score: extremes.max_value,
        };

        if best
            .map(|current| candidate.score > current.score)
            .unwrap_or(true)
        {
            best = Some(candidate);
        }
    }

    best
}

fn downscale_for_matching(gray: &GrayImage, max_side: u32) -> (GrayImage, f32) {
    let max_dim = gray.width().max(gray.height());
    if max_dim <= max_side {
        return (gray.clone(), 1.0);
    }

    let scale = max_side as f32 / max_dim as f32;
    let target_w = ((gray.width() as f32 * scale).round() as u32).max(1);
    let target_h = ((gray.height() as f32 * scale).round() as u32).max(1);
    (
        image::imageops::resize(gray, target_w, target_h, FilterType::Triangle),
        scale,
    )
}

fn prepare_template_image(template: &DynamicImage) -> GrayImage {
    let gray = template.to_luma8();
    let max_dim = gray.width().max(gray.height());
    if max_dim <= 900 {
        gray
    } else {
        let scale = 900.0 / max_dim as f32;
        let w = ((gray.width() as f32 * scale).round() as u32).max(1);
        let h = ((gray.height() as f32 * scale).round() as u32).max(1);
        image::imageops::resize(&gray, w, h, FilterType::Triangle)
    }
}

fn infer_number_roi(template: &DynamicImage) -> NormalizedRect {
    let rgb = template.to_rgb8();
    let (w, h) = rgb.dimensions();
    if w < 8 || h < 8 {
        return DEFAULT_NUMBER_ROI;
    }

    let mut min_x = w;
    let mut min_y = h;
    let mut max_x = 0;
    let mut max_y = 0;
    let mut blue_pixels = 0usize;

    for (x, y, px) in rgb.enumerate_pixels() {
        let [r, g, b] = px.0;
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let saturation = max.saturating_sub(min);

        let likely_blue =
            b > 70 && b > r.saturating_add(18) && b > g.saturating_add(10) && saturation > 28;
        if !likely_blue {
            continue;
        }

        blue_pixels += 1;
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    if blue_pixels < 24 || min_x > max_x || min_y > max_y {
        return DEFAULT_NUMBER_ROI;
    }

    let bw = (max_x - min_x + 1) as f32;
    let bh = (max_y - min_y + 1) as f32;
    let x = clamp01((min_x as f32 - bw * 0.12) / w as f32);
    let y = clamp01((min_y as f32 - bh * 0.12) / h as f32);
    let x2 = clamp01((max_x as f32 + 1.0 + bw * 0.12) / w as f32);
    let y2 = clamp01((max_y as f32 + 1.0 + bh * 0.12) / h as f32);

    let roi = NormalizedRect {
        x,
        y,
        w: (x2 - x).max(0.08),
        h: (y2 - y).max(0.08),
    };

    if roi.w <= 0.0 || roi.h <= 0.0 {
        DEFAULT_NUMBER_ROI
    } else {
        roi
    }
}

fn clamp_rect(x: f32, y: f32, w: f32, h: f32, img_w: u32, img_h: u32) -> Option<PixelRect> {
    if w <= 1.0 || h <= 1.0 || img_w == 0 || img_h == 0 {
        return None;
    }

    let x0 = x.max(0.0).min(img_w.saturating_sub(1) as f32);
    let y0 = y.max(0.0).min(img_h.saturating_sub(1) as f32);
    let x1 = (x + w).max(x0 + 1.0).min(img_w as f32);
    let y1 = (y + h).max(y0 + 1.0).min(img_h as f32);

    let cw = (x1 - x0).floor() as u32;
    let ch = (y1 - y0).floor() as u32;
    if cw < 2 || ch < 2 {
        return None;
    }

    Some(PixelRect {
        x: x0.floor() as u32,
        y: y0.floor() as u32,
        w: cw,
        h: ch,
    })
}

fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

fn sticker_template_store() -> &'static RwLock<Option<StickerTemplate>> {
    STICKER_TEMPLATE.get_or_init(|| RwLock::new(None))
}

fn model_dir() -> PathBuf {
    std::env::var_os("BIP_OCR_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/ppocrv5"))
}

fn resolve_model_paths(model_dir: &Path) -> Result<ModelPaths, String> {
    let det_model = find_existing(model_dir, "detection model", &["det.onnx"])?;
    let rec_model = find_existing(model_dir, "recognition model", &["rec.onnx"])?;
    let dictionary = find_existing(
        model_dir,
        "OCR dictionary",
        &["ppocrv5_dict.txt", "dict.txt"],
    )?;

    let text_line_ori_model = find_optional(
        model_dir,
        &[
            "PP-LCNet_x0_25_textline_ori.onnx",
            "textline_ori.onnx",
            "text_line_ori.onnx",
        ],
    );
    let doc_ori_model = find_optional(model_dir, &["PP-LCNet_x1_0_doc_ori.onnx", "doc_ori.onnx"]);

    Ok(ModelPaths {
        det_model,
        rec_model,
        dictionary,
        text_line_ori_model,
        doc_ori_model,
    })
}

fn find_existing(model_dir: &Path, label: &str, candidates: &[&str]) -> Result<PathBuf, String> {
    for name in candidates {
        let path = model_dir.join(name);
        if path.exists() {
            return Ok(path);
        }
    }

    Err(format!(
        "OCR disabled: missing {label} in {} (tried: {}). Set BIP_OCR_MODEL_DIR to your model folder.",
        model_dir.display(),
        candidates.join(", ")
    ))
}

fn find_optional(model_dir: &Path, candidates: &[&str]) -> Option<PathBuf> {
    for name in candidates {
        let path = model_dir.join(name);
        if path.exists() {
            return Some(path);
        }
    }
    None
}

fn digits_only(text: &str) -> String {
    text.chars().filter(|c| c.is_ascii_digit()).collect()
}
