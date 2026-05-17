use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};

#[derive(Debug, Clone, Copy)]
pub struct NormalizedRect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl NormalizedRect {
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Result<Self> {
        let rect = Self { x, y, w, h };
        rect.validate()?;
        Ok(rect)
    }

    pub fn validate(self) -> Result<()> {
        let values = [self.x, self.y, self.w, self.h];
        if values.iter().any(|value| !value.is_finite()) {
            return Err(anyhow!("number region contains a non-finite value"));
        }
        if self.x < 0.0 || self.y < 0.0 || self.w <= 0.0 || self.h <= 0.0 {
            return Err(anyhow!("number region must have positive size within 0..1"));
        }
        if self.x + self.w > 1.0 || self.y + self.h > 1.0 {
            return Err(anyhow!(
                "number region must fit inside the sticker template"
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct EventConfig {
    pub id: i64,
    pub name: String,
    pub sticker_template_path: PathBuf,
    pub template_width: i32,
    pub template_height: i32,
    pub number_region: NormalizedRect,
}

impl EventConfig {
    pub fn validate(&self) -> Result<()> {
        self.number_region.validate()?;
        if self.template_width <= 0 || self.template_height <= 0 {
            return Err(anyhow!("sticker template dimensions must be positive"));
        }
        if self.sticker_template_path.as_os_str().is_empty() {
            return Err(anyhow!("sticker template path is empty"));
        }
        Ok(())
    }
}

pub fn template_dimensions(path: &Path) -> Result<(i32, i32)> {
    let img = crate::processing::image_ops::load_image(path).map_err(anyhow::Error::msg)?;
    Ok((img.width() as i32, img.height() as i32))
}
