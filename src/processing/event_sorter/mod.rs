//! Resumable event-sticker based sorter pipeline.
//!
//! This module implements the event-sticker batch sorter pipeline. It persists
//! state in SQLite, detects the event sticker with OpenCV, runs OCR on the
//! configured number region, generates visual embeddings for fallback matching,
//! and sorts only high-confidence assignments automatically.

#![allow(dead_code)]

pub mod db;
pub mod embedding;
pub mod event_config;
pub mod image_loader;
pub mod number_cropper;
pub mod pipeline;
pub mod preprocessing;
pub mod sorter;
pub mod sticker_matcher;
pub mod visual_matcher;
