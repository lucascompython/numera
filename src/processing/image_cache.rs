//! High-performance image caching with LRU eviction and background preloading.
//!
//! This module provides a thread-safe cache for decoded image thumbnails,
//! eliminating redundant decoding when navigating through images.

use gpui::RenderImage;
use image::{DynamicImage, Frame, RgbaImage};
use lru::LruCache;
use parking_lot::Mutex;
use rapidhash::fast::RapidHasher;
use rayon::prelude::*;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::image_ops::{self, Rotation, TextOverlayConfig};

/// Cached thumbnail data ready for display.
#[derive(Clone)]
pub struct CachedImage {
    /// RGBA pixel data
    pub rgba: Arc<RgbaImage>,
    /// GPUI render image ready for direct texture upload/reuse during navigation.
    pub preview_image: Arc<RenderImage>,
    /// Preview dimensions.
    pub width: u32,
    pub height: u32,
}

/// Cache key combining path and rendering parameters.
#[derive(Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    path: PathBuf,
    rotation: u8, // Rotation enum as u8
    max_size: u32,
    overlay_signature: u64,
}

impl CacheKey {
    fn new(
        path: &Path,
        rotation: Rotation,
        max_size: u32,
        text_config: Option<&TextOverlayConfig>,
    ) -> Self {
        Self {
            path: path.to_path_buf(),
            rotation: rotation as u8,
            max_size,
            overlay_signature: text_signature(text_config),
        }
    }
}

type RapidBuildHasher = BuildHasherDefault<RapidHasher<'static>>;

const DEFAULT_PREVIEW_MAX_SIDE: u32 = 3200;
const MIN_PREVIEW_MAX_SIDE: u32 = 512;
const MAX_PREVIEW_MAX_SIDE: u32 = 8192;

fn preview_max_side_from_env() -> u32 {
    std::env::var("BIP_PREVIEW_MAX_SIDE")
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .filter(|&value| value > 0)
        .map(|value| value.clamp(MIN_PREVIEW_MAX_SIDE, MAX_PREVIEW_MAX_SIDE))
        .unwrap_or(DEFAULT_PREVIEW_MAX_SIDE)
}

/// Thread-safe LRU cache for image thumbnails.
pub struct ImageCache {
    cache: Mutex<LruCache<CacheKey, CachedImage, RapidBuildHasher>>,
    max_size: u32,
}

impl ImageCache {
    /// Create a new cache with the given capacity (number of images).
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: Mutex::new(LruCache::with_hasher(
                NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::new(10).unwrap()),
                RapidBuildHasher::default(),
            )),
            max_size: preview_max_side_from_env(),
        }
    }

    /// Get a cached image or decode it if not present.
    pub fn get_or_decode(
        &self,
        path: &Path,
        rotation: Rotation,
        text_config: Option<&TextOverlayConfig>,
    ) -> Option<CachedImage> {
        let key = CacheKey::new(path, rotation, self.max_size, text_config);

        // Check cache first
        {
            let mut cache = self.cache.lock();
            if let Some(cached) = cache.get(&key) {
                return Some(cached.clone());
            }
        }

        // Decode image (outside lock)
        let cached = self.decode_image(path, rotation, text_config)?;

        // Store in cache
        {
            let mut cache = self.cache.lock();
            cache.put(key, cached.clone());
        }

        Some(cached)
    }

    /// Decode an image without using the cache.
    fn decode_image(
        &self,
        path: &Path,
        rotation: Rotation,
        text_config: Option<&TextOverlayConfig>,
    ) -> Option<CachedImage> {
        let mut img = image_ops::load_image_for_preview(path, self.max_size).ok()?;

        // Apply rotation
        if rotation != Rotation::None {
            img = image_ops::rotate_image(&img, rotation);
        }

        let mut rgba = if img.width().max(img.height()) > self.max_size {
            image_ops::generate_thumbnail(&img, self.max_size)
        } else {
            match img {
                DynamicImage::ImageRgba8(existing) => existing,
                other => other.to_rgba8(),
            }
        };
        let mut final_img = DynamicImage::ImageRgba8(rgba);

        // Apply text overlay if configured
        if let Some(tc) = text_config {
            let filename = path.file_stem().and_then(|n| n.to_str()).unwrap_or("image");
            final_img = image_ops::overlay_text(final_img, tc, filename);
        }

        rgba = final_img.into_rgba8();
        let width = rgba.width();
        let height = rgba.height();
        let preview_image = Arc::new(render_preview_image(&rgba));

        Some(CachedImage {
            rgba: Arc::new(rgba),
            preview_image,
            width,
            height,
        })
    }

    /// Preload images in background (for adjacent images).
    /// Returns immediately, decoding happens in parallel.
    pub fn preload(
        &self,
        paths: &[PathBuf],
        rotation: Rotation,
        text_config: Option<&TextOverlayConfig>,
    ) {
        let max_size = self.max_size;

        // Check which paths need decoding
        let to_decode: Vec<PathBuf> = {
            let cache = self.cache.lock();
            paths
                .iter()
                .filter(|p| {
                    let key = CacheKey::new(p, rotation, max_size, text_config);
                    !cache.contains(&key)
                })
                .cloned()
                .collect()
        };

        if to_decode.is_empty() {
            return;
        }

        // Clone text_config once for parallel use
        let text_config = text_config.cloned();

        // Decode in parallel using rayon
        let results: Vec<_> = to_decode
            .par_iter()
            .filter_map(|path| {
                let cached = self.decode_image(path, rotation, text_config.as_ref())?;
                let key = CacheKey::new(path, rotation, max_size, text_config.as_ref());
                Some((key, cached))
            })
            .collect();

        // Store results in cache
        let mut cache = self.cache.lock();
        for (key, cached) in results {
            cache.put(key, cached);
        }
    }
}

impl Default for ImageCache {
    fn default() -> Self {
        Self::new(10)
    }
}

fn text_signature(text_config: Option<&TextOverlayConfig>) -> u64 {
    let Some(cfg) = text_config else {
        return 0;
    };

    let mut hasher = RapidHasher::default();
    cfg.text_template.hash(&mut hasher);
    cfg.position.hash(&mut hasher);
    cfg.font_size.to_bits().hash(&mut hasher);
    cfg.color.r.hash(&mut hasher);
    cfg.color.g.hash(&mut hasher);
    cfg.color.b.hash(&mut hasher);
    cfg.color.a.hash(&mut hasher);
    cfg.margin.hash(&mut hasher);
    hasher.finish()
}

pub(crate) fn render_preview_image(rgba: &RgbaImage) -> RenderImage {
    let mut bgra = rgba.clone();

    // GPUI RenderImage expects BGRA pixels. Keep CachedImage::rgba in RGBA for
    // callers that still need normal image processing, and convert only the
    // render copy to avoid PNG encode + GPUI decode on every cached preview.
    for pixel in bgra.chunks_exact_mut(4) {
        pixel.swap(0, 2);
    }

    RenderImage::new(vec![Frame::new(bgra)])
}
