use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png"];

pub fn discover_images(source_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut images = Vec::new();
    visit_dir(source_dir, &mut images)?;
    images.sort();
    Ok(images)
}

fn visit_dir(dir: &Path, images: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;

        if file_type.is_dir() {
            if should_skip_dir(&path) {
                continue;
            }
            visit_dir(&path, images)?;
            continue;
        }

        if file_type.is_file() && is_supported_image(&path) {
            images.push(path);
        }
    }

    Ok(())
}

fn should_skip_dir(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.starts_with('_'))
        .unwrap_or(false)
}

fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            let ext = ext.to_ascii_lowercase();
            IMAGE_EXTENSIONS.contains(&ext.as_str())
        })
        .unwrap_or(false)
}
