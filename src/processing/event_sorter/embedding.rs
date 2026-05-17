use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use opencv::core::{Mat, Rect, Size, Vec3b};
use opencv::imgcodecs;
use opencv::imgproc;
use opencv::prelude::*;

use super::preprocessing::write_debug_image;

pub const MODEL_NAME: &str = "opencv_color_texture_v1";

#[derive(Debug, Clone)]
pub struct GeneratedEmbedding {
    pub model_name: String,
    pub crop_path: Option<PathBuf>,
    pub values: Vec<f32>,
}

pub fn generate_visual_embedding(
    image_path: &Path,
    debug_crop_path: Option<&Path>,
) -> Result<GeneratedEmbedding> {
    let image_path_str = image_path.to_string_lossy();
    let image = imgcodecs::imread(&image_path_str, imgcodecs::IMREAD_COLOR)?;
    if image.empty() {
        return Err(anyhow!("image is empty: {}", image_path.display()));
    }

    let crop = rider_motorcycle_proxy_crop(&image)?;
    if let Some(path) = debug_crop_path {
        write_debug_image(path, &crop)?;
    }

    let mut resized = Mat::default();
    imgproc::resize(
        &crop,
        &mut resized,
        Size::new(256, 256),
        0.0,
        0.0,
        imgproc::INTER_AREA,
    )?;

    let mut values = Vec::with_capacity(96);
    append_hsv_histograms(&resized, &mut values)?;
    append_bgr_grid_means(&resized, &mut values)?;
    append_edge_grid_density(&resized, &mut values)?;
    l2_normalize(&mut values);

    Ok(GeneratedEmbedding {
        model_name: MODEL_NAME.to_string(),
        crop_path: debug_crop_path.map(Path::to_path_buf),
        values,
    })
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

fn rider_motorcycle_proxy_crop(image: &Mat) -> Result<Mat> {
    let cols = image.cols();
    let rows = image.rows();
    if cols <= 0 || rows <= 0 {
        return Err(anyhow!("image has invalid dimensions"));
    }

    // Until detector-backed crops are configured, bias toward the central/lower
    // part of the photo where the bike and rider usually dominate.
    let x = (cols as f32 * 0.08).round() as i32;
    let y = (rows as f32 * 0.10).round() as i32;
    let w = (cols as f32 * 0.84).round() as i32;
    let h = (rows as f32 * 0.82).round() as i32;
    let rect = Rect::new(
        x.clamp(0, cols - 1),
        y.clamp(0, rows - 1),
        w.min(cols - x),
        h.min(rows - y),
    );
    Ok(image.roi(rect)?.try_clone()?)
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
    use super::cosine_similarity;

    #[test]
    fn cosine_similarity_requires_same_dimensions() {
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 0.0]), 0.0);
    }

    #[test]
    fn cosine_similarity_scores_normalized_vectors() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 0.001);
        assert!((cosine_similarity(&[1.0, 0.0], &[0.0, 1.0])).abs() < 0.001);
    }
}
