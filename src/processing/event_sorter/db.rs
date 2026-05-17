use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, params};

use super::event_config::{EventConfig, NormalizedRect, template_dimensions};

#[derive(Debug, Clone)]
pub struct ProcessingDb {
    path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct ImageIdentity {
    pub id: i64,
    pub already_processed: bool,
}

#[derive(Debug, Clone)]
pub struct StickerMatchRecord {
    pub found: bool,
    pub match_confidence: f32,
    pub good_match_count: i32,
    pub homography_valid: bool,
    pub warped_sticker_path: Option<PathBuf>,
    pub number_crop_path: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct OcrRecord {
    pub raw_text: String,
    pub digits_only: String,
    pub confidence: f32,
    pub preprocessing_variant: String,
    pub is_high_confidence: bool,
}

#[derive(Debug, Clone)]
pub struct AssignmentRecord {
    pub final_number: Option<String>,
    pub assignment_method: String,
    pub confidence: f32,
    pub needs_review: bool,
    pub notes: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ImageProcessingRecord {
    pub image_id: i64,
    pub status: String,
    pub sticker_match: StickerMatchRecord,
    pub ocr_result: Option<OcrRecord>,
    pub assignment: AssignmentRecord,
}

impl ProcessingDb {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let db = Self { path: path.into() };
        let conn = db.connect()?;
        init_schema(&conn)?;
        Ok(db)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn connect(&self) -> Result<Connection> {
        connect(&self.path)
    }

    pub fn create_event(
        &self,
        name: &str,
        sticker_template_path: &Path,
        number_region: NormalizedRect,
    ) -> Result<EventConfig> {
        number_region.validate()?;
        let (template_width, template_height) = template_dimensions(sticker_template_path)?;
        let conn = self.connect()?;
        conn.execute(
            "INSERT INTO events (
                name,
                sticker_template_path,
                template_width,
                template_height,
                number_region_x,
                number_region_y,
                number_region_w,
                number_region_h
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                name,
                sticker_template_path.to_string_lossy(),
                template_width,
                template_height,
                number_region.x,
                number_region.y,
                number_region.w,
                number_region.h,
            ],
        )
        .context("failed to insert event")?;

        let id = conn.last_insert_rowid();
        self.get_event(id)
    }

    pub fn get_event(&self, event_id: i64) -> Result<EventConfig> {
        get_event(&self.connect()?, event_id)
    }
}

pub fn connect(path: &Path) -> Result<Connection> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create database folder {}", parent.display()))?;
    }

    let conn = Connection::open(path)
        .with_context(|| format!("failed to open processing database {}", path.display()))?;
    conn.busy_timeout(Duration::from_secs(30))?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    Ok(conn)
}

pub fn init_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            sticker_template_path TEXT NOT NULL,
            template_width INTEGER NOT NULL,
            template_height INTEGER NOT NULL,
            number_region_x REAL NOT NULL,
            number_region_y REAL NOT NULL,
            number_region_w REAL NOT NULL,
            number_region_h REAL NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY,
            event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
            file_path TEXT NOT NULL,
            file_name TEXT NOT NULL,
            width INTEGER,
            height INTEGER,
            exif_datetime TEXT,
            processed_at TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            UNIQUE(event_id, file_path)
        );

        CREATE TABLE IF NOT EXISTS sticker_matches (
            id INTEGER PRIMARY KEY,
            image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            found INTEGER NOT NULL,
            match_confidence REAL NOT NULL,
            good_match_count INTEGER NOT NULL,
            homography_valid INTEGER NOT NULL,
            warped_sticker_path TEXT,
            number_crop_path TEXT,
            UNIQUE(image_id)
        );

        CREATE TABLE IF NOT EXISTS ocr_results (
            id INTEGER PRIMARY KEY,
            image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            raw_text TEXT NOT NULL,
            digits_only TEXT NOT NULL,
            ocr_confidence REAL NOT NULL,
            preprocessing_variant TEXT NOT NULL,
            is_high_confidence INTEGER NOT NULL,
            UNIQUE(image_id, preprocessing_variant)
        );

        CREATE TABLE IF NOT EXISTS visual_embeddings (
            id INTEGER PRIMARY KEY,
            image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            model_name TEXT NOT NULL,
            crop_path TEXT,
            embedding_blob BLOB NOT NULL,
            embedding_dim INTEGER NOT NULL,
            UNIQUE(image_id, model_name)
        );

        CREATE TABLE IF NOT EXISTS assignments (
            id INTEGER PRIMARY KEY,
            image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            final_number TEXT,
            assignment_method TEXT NOT NULL,
            confidence REAL NOT NULL,
            needs_review INTEGER NOT NULL,
            notes TEXT,
            UNIQUE(image_id)
        );

        CREATE TABLE IF NOT EXISTS visual_matches (
            id INTEGER PRIMARY KEY,
            image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            matched_anchor_image_id INTEGER REFERENCES images(id) ON DELETE SET NULL,
            matched_number TEXT,
            similarity REAL NOT NULL,
            rank INTEGER NOT NULL,
            UNIQUE(image_id, rank)
        );

        CREATE INDEX IF NOT EXISTS idx_images_event_status ON images(event_id, status);
        CREATE INDEX IF NOT EXISTS idx_assignments_number ON assignments(final_number);
        CREATE INDEX IF NOT EXISTS idx_ocr_high_confidence ON ocr_results(is_high_confidence);
        ",
    )
    .context("failed to initialize processing schema")?;
    Ok(())
}

pub fn get_event(conn: &Connection, event_id: i64) -> Result<EventConfig> {
    let event = conn
        .query_row(
            "SELECT
                id,
                name,
                sticker_template_path,
                template_width,
                template_height,
                number_region_x,
                number_region_y,
                number_region_w,
                number_region_h
            FROM events
            WHERE id = ?1",
            [event_id],
            |row| {
                Ok(EventConfig {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    sticker_template_path: PathBuf::from(row.get::<_, String>(2)?),
                    template_width: row.get(3)?,
                    template_height: row.get(4)?,
                    number_region: NormalizedRect {
                        x: row.get(5)?,
                        y: row.get(6)?,
                        w: row.get(7)?,
                        h: row.get(8)?,
                    },
                })
            },
        )
        .with_context(|| format!("event {event_id} not found"))?;
    event.validate()?;
    Ok(event)
}

pub fn upsert_image(
    conn: &Connection,
    event_id: i64,
    path: &Path,
    width: Option<i32>,
    height: Option<i32>,
) -> Result<ImageIdentity> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("image")
        .to_string();
    let file_path = path.to_string_lossy().to_string();

    conn.execute(
        "INSERT INTO images (event_id, file_path, file_name, width, height, status)
         VALUES (?1, ?2, ?3, ?4, ?5, 'pending')
         ON CONFLICT(event_id, file_path) DO UPDATE SET
            file_name = excluded.file_name,
            width = COALESCE(excluded.width, images.width),
            height = COALESCE(excluded.height, images.height)",
        params![event_id, file_path, file_name, width, height],
    )
    .with_context(|| format!("failed to upsert image {}", path.display()))?;

    let (id, processed_at): (i64, Option<String>) = conn.query_row(
        "SELECT id, processed_at FROM images WHERE event_id = ?1 AND file_path = ?2",
        params![event_id, path.to_string_lossy()],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;

    Ok(ImageIdentity {
        id,
        already_processed: processed_at.is_some(),
    })
}

pub fn is_image_processed(conn: &Connection, event_id: i64, path: &Path) -> Result<bool> {
    let processed_at: Option<Option<String>> = conn
        .query_row(
            "SELECT processed_at FROM images WHERE event_id = ?1 AND file_path = ?2",
            params![event_id, path.to_string_lossy()],
            |row| row.get(0),
        )
        .optional()?;
    Ok(processed_at.flatten().is_some())
}

pub fn save_processing_result(conn: &mut Connection, record: &ImageProcessingRecord) -> Result<()> {
    let tx = conn.transaction()?;
    tx.execute(
        "INSERT INTO sticker_matches (
            image_id,
            found,
            match_confidence,
            good_match_count,
            homography_valid,
            warped_sticker_path,
            number_crop_path
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
         ON CONFLICT(image_id) DO UPDATE SET
            found = excluded.found,
            match_confidence = excluded.match_confidence,
            good_match_count = excluded.good_match_count,
            homography_valid = excluded.homography_valid,
            warped_sticker_path = excluded.warped_sticker_path,
            number_crop_path = excluded.number_crop_path",
        params![
            record.image_id,
            record.sticker_match.found as i32,
            record.sticker_match.match_confidence,
            record.sticker_match.good_match_count,
            record.sticker_match.homography_valid as i32,
            optional_path(&record.sticker_match.warped_sticker_path),
            optional_path(&record.sticker_match.number_crop_path),
        ],
    )?;

    tx.execute(
        "DELETE FROM ocr_results WHERE image_id = ?1",
        [record.image_id],
    )?;

    if let Some(ocr) = record.ocr_result.as_ref() {
        tx.execute(
            "INSERT INTO ocr_results (
                image_id,
                raw_text,
                digits_only,
                ocr_confidence,
                preprocessing_variant,
                is_high_confidence
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(image_id, preprocessing_variant) DO UPDATE SET
                raw_text = excluded.raw_text,
                digits_only = excluded.digits_only,
                ocr_confidence = excluded.ocr_confidence,
                is_high_confidence = excluded.is_high_confidence",
            params![
                record.image_id,
                ocr.raw_text,
                ocr.digits_only,
                ocr.confidence,
                ocr.preprocessing_variant,
                ocr.is_high_confidence as i32,
            ],
        )?;
    }

    tx.execute(
        "INSERT INTO assignments (
            image_id,
            final_number,
            assignment_method,
            confidence,
            needs_review,
            notes
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)
         ON CONFLICT(image_id) DO UPDATE SET
            final_number = excluded.final_number,
            assignment_method = excluded.assignment_method,
            confidence = excluded.confidence,
            needs_review = excluded.needs_review,
            notes = excluded.notes",
        params![
            record.image_id,
            record.assignment.final_number,
            record.assignment.assignment_method,
            record.assignment.confidence,
            record.assignment.needs_review as i32,
            record.assignment.notes,
        ],
    )?;

    tx.execute(
        "UPDATE images
         SET status = ?1, processed_at = CURRENT_TIMESTAMP
         WHERE id = ?2",
        params![record.status, record.image_id],
    )?;

    tx.commit()?;
    Ok(())
}

fn optional_path(path: &Option<PathBuf>) -> Option<String> {
    path.as_ref().map(|path| path.to_string_lossy().to_string())
}
