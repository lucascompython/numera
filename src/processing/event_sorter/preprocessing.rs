use anyhow::{Result, anyhow};
use image::DynamicImage;
use opencv::core::{Mat, Size, Vector};
use opencv::imgcodecs;
use opencv::imgproc;
use opencv::prelude::*;

#[derive(Debug)]
pub struct PreprocessedCrop {
    pub variant: String,
    pub mat: Mat,
}

pub fn preprocess_number_crop(crop: &Mat) -> Result<PreprocessedCrop> {
    let mut gray = Mat::default();
    if crop.channels() > 1 {
        imgproc::cvt_color_def(crop, &mut gray, imgproc::COLOR_BGR2GRAY)?;
    } else {
        gray = crop.try_clone()?;
    }

    let mut equalized = Mat::default();
    imgproc::equalize_hist(&gray, &mut equalized)?;

    let mut blurred = Mat::default();
    imgproc::gaussian_blur_def(&equalized, &mut blurred, Size::new(3, 3), 0.0)?;

    let block_size = adaptive_block_size(blurred.cols(), blurred.rows());
    let mut thresholded = Mat::default();
    imgproc::adaptive_threshold(
        &blurred,
        &mut thresholded,
        255.0,
        imgproc::ADAPTIVE_THRESH_GAUSSIAN_C,
        imgproc::THRESH_BINARY,
        block_size,
        7.0,
    )?;

    let kernel = imgproc::get_structuring_element_def(imgproc::MORPH_RECT, Size::new(2, 2))?;
    let mut closed = Mat::default();
    imgproc::morphology_ex_def(&thresholded, &mut closed, imgproc::MORPH_CLOSE, &kernel)?;

    Ok(PreprocessedCrop {
        variant: "gray_equalized_adaptive_close".to_string(),
        mat: closed,
    })
}

pub fn mat_to_dynamic_image(mat: &Mat) -> Result<DynamicImage> {
    let mut encoded = Vector::<u8>::new();
    imgcodecs::imencode_def(".png", mat, &mut encoded)?;
    Ok(image::load_from_memory(encoded.as_slice())?)
}

pub fn write_debug_image(path: &std::path::Path, mat: &Mat) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if !imgcodecs::imwrite_def(&path.to_string_lossy(), mat)? {
        return Err(anyhow!("OpenCV failed to write {}", path.display()));
    }
    Ok(())
}

fn adaptive_block_size(width: i32, height: i32) -> i32 {
    let base = (width.min(height) / 8).clamp(11, 41);
    if base % 2 == 0 { base + 1 } else { base }
}
