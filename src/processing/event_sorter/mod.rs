//! Resumable event-sticker based sorter pipeline.
//!
//! This module is the first-stage implementation for a more robust batch
//! sorter. It persists state in SQLite, detects the event sticker with OpenCV,
//! crops the configured number region, runs the existing OCR engine on that
//! crop, and sorts only high-confidence OCR results automatically.

#![allow(dead_code)]

pub mod db;
pub mod event_config;
pub mod image_loader;
pub mod number_cropper;
pub mod pipeline;
pub mod preprocessing;
pub mod sorter;
pub mod sticker_matcher;
