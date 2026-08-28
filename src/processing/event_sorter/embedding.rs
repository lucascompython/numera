use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{anyhow, bail, Context, Result};
use opencv::core::{Mat, Rect, Size, Vec3b};
use opencv::imgcodecs;
use opencv::imgproc;
use opencv::prelude::*;
use ort::session::Session;
use ort::value::{Tensor, TensorElementType, ValueType};

use super::preprocessing::write_debug_image;

pub const OPENCV_MODEL_NAME: &str = "opencv_color_texture_v1";
pub const MODEL_NAME: &str = OPENCV_MODEL_NAME;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualEmbeddingBackend {
    OpenCv,
    Onnx,
}

impl VisualEmbeddingBackend {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "opencv" => Ok(Self::OpenCv),
            "onnx" => Ok(Self::Onnx),
            other => bail!("unsupported visual embedding backend '{other}'"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VisualEmbeddingConfig {
    pub backend: VisualEmbeddingBackend,
    pub model_path: Option<PathBuf>,
    pub input_size: usize,
    pub normalize: bool,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub model_name: Option<String>,
}

impl Default for VisualEmbeddingConfig {
    fn default() -> Self {
        Self {
            backend: VisualEmbeddingBackend::OpenCv,
            model_path: None,
            input_size: 256,
            normalize: true,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            model_name: None,
        }
    }
}

impl VisualEmbeddingConfig {
    pub fn with_backend(mut self, backend: &str) -> Result<Self> {
        self.backend = VisualEmbeddingBackend::parse(backend)?;
        Ok(self)
    }

    pub fn resolved_model_name(&self) -> String {
        if let Some(name) = self.model_name.as_ref().filter(|name| !name.is_empty()) {
            return name.clone();
        }

        match self.backend {
            VisualEmbeddingBackend::OpenCv => {
                if self.input_size == 256 {
                    OPENCV_MODEL_NAME.to_string()
                } else {
                    format!("{}_{}px", OPENCV_MODEL_NAME, self.input_size)
                }
            }
            VisualEmbeddingBackend::Onnx => {
                let stem = self
                    .model_path
                    .as_ref()
                    .and_then(|path| path.file_stem())
                    .and_then(|stem| stem.to_str())
                    .unwrap_or("model");
                format!("onnx_{stem}_{}px", self.input_size)
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CropBox {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

#[derive(Debug, Clone)]
pub struct VisualEmbeddingResult {
    pub embedding: Vec<f32>,
    pub crop_path: Option<PathBuf>,
    pub crop_bbox: Option<CropBox>,
    pub quality: Option<f32>,
}

pub trait EmbeddingProvider: Send + Sync {
    fn model_name(&self) -> &str;
    fn embedding_dim(&self) -> usize;
    fn generate_embedding(&self, image_path: &Path) -> Result<VisualEmbeddingResult>;
}

#[derive(Debug, Clone)]
pub struct OpenCvEmbeddingProvider {
    model_name: String,
    input_size: i32,
}

impl Default for OpenCvEmbeddingProvider {
    fn default() -> Self {
        Self::new(256)
    }
}

impl OpenCvEmbeddingProvider {
    pub fn new(input_size: i32) -> Self {
        Self::with_model_name(input_size, OPENCV_MODEL_NAME.to_string())
    }

    pub fn with_model_name(input_size: i32, model_name: String) -> Self {
        Self {
            model_name,
            input_size: input_size.max(32),
        }
    }
}

impl EmbeddingProvider for OpenCvEmbeddingProvider {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn embedding_dim(&self) -> usize {
        96
    }

    fn generate_embedding(&self, image_path: &Path) -> Result<VisualEmbeddingResult> {
        let image = load_color_image(image_path)?;
        let (crop, bbox) = rider_motorcycle_proxy_crop(&image)?;

        let mut resized = Mat::default();
        imgproc::resize(
            &crop,
            &mut resized,
            Size::new(self.input_size, self.input_size),
            0.0,
            0.0,
            imgproc::INTER_AREA,
        )?;

        let mut embedding = Vec::with_capacity(self.embedding_dim());
        append_hsv_histograms(&resized, &mut embedding)?;
        append_bgr_grid_means(&resized, &mut embedding)?;
        append_edge_grid_density(&resized, &mut embedding)?;
        l2_normalize(&mut embedding);

        Ok(VisualEmbeddingResult {
            embedding,
            crop_path: None,
            crop_bbox: Some(bbox),
            quality: Some(proxy_crop_quality(&crop)?),
        })
    }
}

#[derive(Debug)]
pub struct OnnxEmbeddingProvider {
    model_name: String,
    session: Mutex<Session>,
    input_name: String,
    output_name: String,
    input_size: usize,
    input_layout: OnnxInputLayout,
    normalize_pixels: bool,
    mean: [f32; 3],
    std: [f32; 3],
    output_dim: usize,
}

impl OnnxEmbeddingProvider {
    pub fn new(config: &VisualEmbeddingConfig) -> Result<Self> {
        let model_path = config
            .model_path
            .as_ref()
            .context("visual_embedding_model_path is required when backend is 'onnx'")?;
        if !model_path.exists() {
            bail!(
                "ONNX embedding model does not exist: {}",
                model_path.display()
            );
        }

        let mut builder = Session::builder().with_context(|| {
            format!(
                "failed to create ONNX session builder for {}",
                model_path.display()
            )
        })?;
        let session = builder.commit_from_file(model_path).with_context(|| {
            format!(
                "failed to load ONNX embedding model {}",
                model_path.display()
            )
        })?;

        let input = session
            .inputs()
            .first()
            .context("ONNX embedding model has no inputs")?;
        let input_shape = tensor_shape(input.dtype())
            .with_context(|| format!("ONNX input '{}' is not a tensor", input.name()))?;
        let input_layout = OnnxInputLayout::from_shape(&input_shape).with_context(|| {
            format!(
                "unsupported ONNX input shape for '{}': {:?}; expected NCHW or NHWC image tensor",
                input.name(),
                input_shape
            )
        })?;
        validate_float_tensor(input.dtype())
            .with_context(|| format!("ONNX input '{}' must be f32", input.name()))?;

        let output = session
            .outputs()
            .first()
            .context("ONNX embedding model has no outputs")?;
        validate_float_tensor(output.dtype())
            .with_context(|| format!("ONNX output '{}' must be f32", output.name()))?;
        let output_dim = tensor_shape(output.dtype())
            .and_then(|shape| static_embedding_dim(&shape))
            .unwrap_or(0);

        Ok(Self {
            model_name: config.resolved_model_name(),
            input_name: input.name().to_string(),
            output_name: output.name().to_string(),
            session: Mutex::new(session),
            input_size: config.input_size.max(32),
            input_layout,
            normalize_pixels: config.normalize,
            mean: config.mean,
            std: config.std,
            output_dim,
        })
    }
}

impl EmbeddingProvider for OnnxEmbeddingProvider {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn embedding_dim(&self) -> usize {
        self.output_dim
    }

    fn generate_embedding(&self, image_path: &Path) -> Result<VisualEmbeddingResult> {
        let image = load_color_image(image_path)?;
        let (crop, bbox) = rider_motorcycle_proxy_crop(&image)?;
        let tensor_data = onnx_input_tensor(
            &crop,
            self.input_size,
            self.input_layout,
            self.normalize_pixels,
            self.mean,
            self.std,
        )?;
        let tensor_shape = self.input_layout.shape(self.input_size);
        let tensor = Tensor::from_array((tensor_shape, tensor_data))
            .context("failed to create ONNX embedding input tensor")?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow!("ONNX embedding session lock was poisoned"))?;
        let outputs = session
            .run(ort::inputs![self.input_name.as_str() => tensor])
            .context("ONNX embedding inference failed")?;
        let output = outputs
            .get(self.output_name.as_str())
            .with_context(|| format!("ONNX output '{}' was not returned", self.output_name))?;
        let (shape, data) = output
            .try_extract_tensor::<f32>()
            .with_context(|| format!("ONNX output '{}' is not an f32 tensor", self.output_name))?;
        if data.len() < 2 {
            bail!(
                "unsupported ONNX output shape {:?}; expected a non-empty embedding vector",
                shape.iter().copied().collect::<Vec<_>>()
            );
        }

        let mut embedding = data.to_vec();
        l2_normalize(&mut embedding);

        Ok(VisualEmbeddingResult {
            embedding,
            crop_path: None,
            crop_bbox: Some(bbox),
            quality: Some(proxy_crop_quality(&crop)?),
        })
    }
}

#[derive(Debug, Clone, Copy)]
enum OnnxInputLayout {
    Nchw,
    Nhwc,
}

impl OnnxInputLayout {
    fn from_shape(shape: &[i64]) -> Option<Self> {
        if shape.len() != 4 {
            return None;
        }

        if shape[1] == 3 {
            Some(Self::Nchw)
        } else if shape[3] == 3 {
            Some(Self::Nhwc)
        } else if shape[1] <= 0 && shape[3] != 3 {
            Some(Self::Nchw)
        } else {
            None
        }
    }

    fn shape(self, input_size: usize) -> Vec<i64> {
        let input_size = input_size as i64;
        match self {
            Self::Nchw => vec![1, 3, input_size, input_size],
            Self::Nhwc => vec![1, input_size, input_size, 3],
        }
    }
}

pub fn build_embedding_provider(
    config: &VisualEmbeddingConfig,
) -> Result<Box<dyn EmbeddingProvider>> {
    match config.backend {
        VisualEmbeddingBackend::OpenCv => Ok(Box::new(OpenCvEmbeddingProvider::with_model_name(
            config.input_size as i32,
            config.resolved_model_name(),
        ))),
        VisualEmbeddingBackend::Onnx => Ok(Box::new(OnnxEmbeddingProvider::new(config)?)),
    }
}

pub fn generate_visual_embedding_with_provider(
    provider: &dyn EmbeddingProvider,
    image_path: &Path,
    debug_crop_path: Option<&Path>,
) -> Result<VisualEmbeddingResult> {
    let mut result = provider.generate_embedding(image_path)?;
    if let Some(path) = debug_crop_path {
        save_proxy_crop_debug(image_path, path)?;
        result.crop_path = Some(path.to_path_buf());
    }
    Ok(result)
}

pub fn generate_visual_embedding(
    image_path: &Path,
    debug_crop_path: Option<&Path>,
) -> Result<VisualEmbeddingResult> {
    let provider = OpenCvEmbeddingProvider::default();
    generate_visual_embedding_with_provider(&provider, image_path, debug_crop_path)
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    a.iter()
        .zip(b.iter())
        .map(|(left, right)| left * right)
        .sum::<f32>()
        .clamp(-1.0, 1.0)
}

fn load_color_image(image_path: &Path) -> Result<Mat> {
    let image = imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR)?;
    if image.empty() {
        bail!("image is empty: {}", image_path.display());
    }
    Ok(image)
}

fn rider_motorcycle_proxy_crop(image: &Mat) -> Result<(Mat, CropBox)> {
    let cols = image.cols();
    let rows = image.rows();
    if cols <= 0 || rows <= 0 {
        bail!("image has invalid dimensions");
    }

    let x = (cols as f32 * 0.08).round() as i32;
    let y = (rows as f32 * 0.10).round() as i32;
    let w = (cols as f32 * 0.84).round() as i32;
    let h = (rows as f32 * 0.82).round() as i32;
    let x = x.clamp(0, cols - 1);
    let y = y.clamp(0, rows - 1);
    let w = w.min(cols - x).max(1);
    let h = h.min(rows - y).max(1);
    let rect = Rect::new(x, y, w, h);
    let crop = image.roi(rect)?.try_clone()?;
    Ok((crop, CropBox { x, y, w, h }))
}

fn save_proxy_crop_debug(image_path: &Path, debug_crop_path: &Path) -> Result<()> {
    let image = load_color_image(image_path)?;
    let (crop, _) = rider_motorcycle_proxy_crop(&image)?;
    write_debug_image(debug_crop_path, &crop)
}

fn onnx_input_tensor(
    crop_bgr: &Mat,
    input_size: usize,
    layout: OnnxInputLayout,
    normalize_pixels: bool,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<Vec<f32>> {
    let input_size_i32 = input_size as i32;
    let mut resized = Mat::default();
    imgproc::resize(
        crop_bgr,
        &mut resized,
        Size::new(input_size_i32, input_size_i32),
        0.0,
        0.0,
        imgproc::INTER_AREA,
    )?;

    let mut rgb = Mat::default();
    imgproc::cvt_color_def(&resized, &mut rgb, imgproc::COLOR_BGR2RGB)?;

    let pixel_count = input_size * input_size;
    let mut data = vec![0.0; pixel_count * 3];
    for y in 0..input_size_i32 {
        for x in 0..input_size_i32 {
            let pixel = *rgb.at_2d::<Vec3b>(y, x)?;
            for channel in 0..3 {
                let mut value = pixel[channel] as f32 / 255.0;
                if normalize_pixels {
                    value = (value - mean[channel]) / std[channel].max(f32::EPSILON);
                }

                let index = match layout {
                    OnnxInputLayout::Nchw => {
                        channel * pixel_count + y as usize * input_size + x as usize
                    }
                    OnnxInputLayout::Nhwc => (y as usize * input_size + x as usize) * 3 + channel,
                };
                data[index] = value;
            }
        }
    }

    Ok(data)
}

fn validate_float_tensor(dtype: &ValueType) -> Result<()> {
    match dtype.tensor_type() {
        Some(TensorElementType::Float32) => Ok(()),
        Some(other) => bail!("expected f32 tensor, got {other}"),
        None => bail!("expected tensor, got {dtype}"),
    }
}

fn tensor_shape(dtype: &ValueType) -> Option<Vec<i64>> {
    dtype
        .tensor_shape()
        .map(|shape| shape.iter().copied().collect())
}

fn static_embedding_dim(shape: &[i64]) -> Option<usize> {
    if shape.is_empty() || shape.iter().any(|dim| *dim == 0 || *dim < -1) {
        return None;
    }

    let dims = if shape.first().copied() == Some(1) || shape.first().copied() == Some(-1) {
        &shape[1..]
    } else {
        shape
    };

    if dims.iter().any(|dim| *dim <= 0) {
        return None;
    }

    Some(dims.iter().map(|dim| *dim as usize).product())
}

fn proxy_crop_quality(crop: &Mat) -> Result<f32> {
    let area_score = ((crop.cols() * crop.rows()) as f32 / (512.0 * 512.0)).clamp(0.05, 1.0);
    let mut gray = Mat::default();
    imgproc::cvt_color_def(crop, &mut gray, imgproc::COLOR_BGR2GRAY)?;
    let mut edges = Mat::default();
    imgproc::canny(&gray, &mut edges, 80.0, 160.0, 3, false)?;

    let mut edge_pixels = 0f32;
    let mut total = 0f32;
    for row in 0..edges.rows() {
        for col in 0..edges.cols() {
            if *edges.at_2d::<u8>(row, col)? != 0 {
                edge_pixels += 1.0;
            }
            total += 1.0;
        }
    }

    let edge_score = (edge_pixels / total.max(1.0) * 12.0).clamp(0.0, 1.0);
    Ok((area_score * 0.45 + edge_score * 0.55).clamp(0.0, 1.0))
}

fn append_hsv_histograms(image_bgr: &Mat, values: &mut Vec<f32>) -> Result<()> {
    let mut hsv = Mat::default();
    imgproc::cvt_color_def(image_bgr, &mut hsv, imgproc::COLOR_BGR2HSV)?;

    let mut hue = [0f32; 16];
    let mut sat = [0f32; 8];
    let mut val = [0f32; 8];
    let mut total = 0f32;

    for row in 0..hsv.rows() {
        for col in 0..hsv.cols() {
            let pixel = *hsv.at_2d::<Vec3b>(row, col)?;
            let h_bin = ((pixel[0] as usize * hue.len()) / 180).min(hue.len() - 1);
            let s_bin = ((pixel[1] as usize * sat.len()) / 256).min(sat.len() - 1);
            let v_bin = ((pixel[2] as usize * val.len()) / 256).min(val.len() - 1);
            hue[h_bin] += 1.0;
            sat[s_bin] += 1.0;
            val[v_bin] += 1.0;
            total += 1.0;
        }
    }

    let total = total.max(1.0);
    values.extend(hue.iter().map(|count| count / total));
    values.extend(sat.iter().map(|count| count / total));
    values.extend(val.iter().map(|count| count / total));
    Ok(())
}

fn append_bgr_grid_means(image_bgr: &Mat, values: &mut Vec<f32>) -> Result<()> {
    const GRID: i32 = 4;
    let cell_w = image_bgr.cols() / GRID;
    let cell_h = image_bgr.rows() / GRID;

    for cell_y in 0..GRID {
        for cell_x in 0..GRID {
            let x0 = cell_x * cell_w;
            let y0 = cell_y * cell_h;
            let x1 = if cell_x + 1 == GRID {
                image_bgr.cols()
            } else {
                (cell_x + 1) * cell_w
            };
            let y1 = if cell_y + 1 == GRID {
                image_bgr.rows()
            } else {
                (cell_y + 1) * cell_h
            };

            let mut b = 0f32;
            let mut g = 0f32;
            let mut r = 0f32;
            let mut count = 0f32;
            for row in y0..y1 {
                for col in x0..x1 {
                    let pixel = *image_bgr.at_2d::<Vec3b>(row, col)?;
                    b += pixel[0] as f32;
                    g += pixel[1] as f32;
                    r += pixel[2] as f32;
                    count += 1.0;
                }
            }

            let denom = count.max(1.0) * 255.0;
            values.push(b / denom);
            values.push(g / denom);
            values.push(r / denom);
        }
    }

    Ok(())
}

fn append_edge_grid_density(image_bgr: &Mat, values: &mut Vec<f32>) -> Result<()> {
    const GRID: i32 = 4;
    let mut gray = Mat::default();
    imgproc::cvt_color_def(image_bgr, &mut gray, imgproc::COLOR_BGR2GRAY)?;

    let mut edges = Mat::default();
    imgproc::canny(&gray, &mut edges, 80.0, 160.0, 3, false)?;

    let cell_w = edges.cols() / GRID;
    let cell_h = edges.rows() / GRID;
    for cell_y in 0..GRID {
        for cell_x in 0..GRID {
            let x0 = cell_x * cell_w;
            let y0 = cell_y * cell_h;
            let x1 = if cell_x + 1 == GRID {
                edges.cols()
            } else {
                (cell_x + 1) * cell_w
            };
            let y1 = if cell_y + 1 == GRID {
                edges.rows()
            } else {
                (cell_y + 1) * cell_h
            };

            let mut edge_pixels = 0f32;
            let mut count = 0f32;
            for row in y0..y1 {
                for col in x0..x1 {
                    if *edges.at_2d::<u8>(row, col)? != 0 {
                        edge_pixels += 1.0;
                    }
                    count += 1.0;
                }
            }
            values.push(edge_pixels / count.max(1.0));
        }
    }

    Ok(())
}

fn l2_normalize(values: &mut [f32]) {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm <= f32::EPSILON {
        return;
    }

    for value in values {
        *value /= norm;
    }
}

#[cfg(test)]
mod tests {
    use super::{cosine_similarity, VisualEmbeddingBackend, VisualEmbeddingConfig};

    #[test]
    fn cosine_similarity_requires_same_dimensions() {
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 0.0]), 0.0);
    }

    #[test]
    fn cosine_similarity_scores_normalized_vectors() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 0.001);
        assert!((cosine_similarity(&[1.0, 0.0], &[0.0, 1.0])).abs() < 0.001);
    }

    #[test]
    fn embedding_backend_parses_config_values() {
        assert_eq!(
            VisualEmbeddingBackend::parse("onnx").unwrap(),
            VisualEmbeddingBackend::Onnx
        );
        assert_eq!(
            VisualEmbeddingBackend::parse("opencv").unwrap(),
            VisualEmbeddingBackend::OpenCv
        );
        assert!(VisualEmbeddingBackend::parse("unknown").is_err());
    }

    #[test]
    fn default_embedding_config_uses_opencv_fallback() {
        let config = VisualEmbeddingConfig::default();
        assert_eq!(config.backend, VisualEmbeddingBackend::OpenCv);
        assert_eq!(config.resolved_model_name(), super::OPENCV_MODEL_NAME);
    }
}
