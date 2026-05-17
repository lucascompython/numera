use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};

#[derive(Debug, Clone, Copy)]
pub enum SortMode {
    Copy,
    Move,
}

pub fn destination_folder(output_dir: &Path, status: &str, number: Option<&str>) -> PathBuf {
    match (status, number) {
        ("assigned_by_ocr", Some(number)) => output_dir.join(number),
        ("assigned_by_visual_match", Some(number)) => output_dir.join(number),
        ("ambiguous", _) => output_dir.join("_ambiguous"),
        ("no_sticker_found", _) => output_dir.join("_no_sticker"),
        _ => output_dir.join("_review"),
    }
}

pub fn place_file(
    source: &Path,
    output_dir: &Path,
    status: &str,
    number: Option<&str>,
    mode: SortMode,
) -> Result<PathBuf> {
    let folder = destination_folder(output_dir, status, number);
    fs::create_dir_all(&folder)
        .with_context(|| format!("failed to create output folder {}", folder.display()))?;

    let file_name = source
        .file_name()
        .ok_or_else(|| anyhow!("source path has no filename: {}", source.display()))?;

    for attempt in 0..10_000 {
        let destination = folder.join(conflict_safe_name(file_name, attempt));
        match mode {
            SortMode::Copy => {
                if copy_create_new(source, &destination)? {
                    return Ok(destination);
                }
            }
            SortMode::Move => {
                if destination.exists() {
                    continue;
                }
                match fs::rename(source, &destination) {
                    Ok(()) => return Ok(destination),
                    Err(err) if err.kind() == ErrorKind::CrossesDevices => {
                        if copy_create_new(source, &destination)? {
                            fs::remove_file(source).with_context(|| {
                                format!(
                                    "failed to remove source after cross-device move {}",
                                    source.display()
                                )
                            })?;
                            return Ok(destination);
                        }
                    }
                    Err(err) if destination.exists() => {
                        let _ = err;
                        continue;
                    }
                    Err(err) => {
                        return Err(err).with_context(|| {
                            format!(
                                "failed to move {} to {}",
                                source.display(),
                                destination.display()
                            )
                        });
                    }
                }
            }
        }
    }

    Err(anyhow!(
        "could not find a free destination filename for {}",
        source.display()
    ))
}

fn copy_create_new(source: &Path, destination: &Path) -> Result<bool> {
    let mut output = match OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)
    {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => return Ok(false),
        Err(err) => {
            return Err(err).with_context(|| {
                format!("failed to create destination {}", destination.display())
            });
        }
    };

    let mut input =
        File::open(source).with_context(|| format!("failed to open {}", source.display()))?;
    let mut buffer = [0u8; 128 * 1024];
    loop {
        let read = input.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        output.write_all(&buffer[..read])?;
    }
    output.sync_all()?;
    Ok(true)
}

fn conflict_safe_name(file_name: &std::ffi::OsStr, attempt: usize) -> std::ffi::OsString {
    if attempt == 0 {
        return file_name.to_os_string();
    }

    let path = Path::new(file_name);
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("image");
    let extension = path.extension().and_then(|extension| extension.to_str());

    match extension {
        Some(extension) => format!("{stem}_{attempt}.{extension}").into(),
        None => format!("{stem}_{attempt}").into(),
    }
}
