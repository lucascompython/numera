use anyhow::{Result, anyhow};
use opencv::core::{Mat, Rect};
use opencv::prelude::*;

use super::event_config::NormalizedRect;

pub fn crop_number_region(warped_sticker: &Mat, region: NormalizedRect) -> Result<Mat> {
    let cols = warped_sticker.cols();
    let rows = warped_sticker.rows();
    if cols <= 0 || rows <= 0 {
        return Err(anyhow!("warped sticker is empty"));
    }

    let x = (region.x * cols as f32)
        .round()
        .clamp(0.0, (cols - 1) as f32) as i32;
    let y = (region.y * rows as f32)
        .round()
        .clamp(0.0, (rows - 1) as f32) as i32;
    let w = (region.w * cols as f32).round().max(1.0) as i32;
    let h = (region.h * rows as f32).round().max(1.0) as i32;
    let w = w.min(cols - x);
    let h = h.min(rows - y);

    if w <= 1 || h <= 1 {
        return Err(anyhow!(
            "configured number region is too small after warping"
        ));
    }

    let roi = warped_sticker.roi(Rect::new(x, y, w, h))?;
    Ok(roi.try_clone()?)
}
