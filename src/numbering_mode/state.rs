//! State management for numbering mode.

use std::path::PathBuf;

/// Represents a completed file move operation for undo support.
#[derive(Clone, Debug)]
pub struct MoveOperation {
    pub original_path: PathBuf,
    pub new_path: PathBuf,
    pub number: String,
}

/// State for the numbering mode.
pub struct NumberingState {
    /// Source folder path
    pub source_folder: Option<PathBuf>,

    /// All image paths in the folder
    pub image_paths: Vec<PathBuf>,

    /// Current image index
    pub current_index: usize,

    /// Number input buffer (what the user is typing)
    pub input_buffer: String,

    /// Zoom level (1.0 = fit to view, >1.0 = zoomed in)
    pub zoom_level: f32,

    /// Pan offset in viewport pixels.
    pub pan_x: f32,
    pub pan_y: f32,

    /// Whether we're currently dragging to pan
    pub is_dragging: bool,
    pub drag_start_x: f32,
    pub drag_start_y: f32,

    /// Undo stack
    pub undo_stack: Vec<MoveOperation>,

    /// Status message
    pub status_message: String,
}

impl Default for NumberingState {
    fn default() -> Self {
        Self {
            source_folder: None,
            image_paths: Vec::new(),
            current_index: 0,
            input_buffer: String::new(),
            zoom_level: 1.0,
            pan_x: 0.0,
            pan_y: 0.0,
            is_dragging: false,
            drag_start_x: 0.0,
            drag_start_y: 0.0,
            undo_stack: Vec::new(),
            status_message: "Open a folder to begin numbering".into(),
        }
    }
}

impl NumberingState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the current image path, if any.
    pub fn current_image(&self) -> Option<&PathBuf> {
        self.image_paths.get(self.current_index)
    }

    /// Move to the next image (without saving).
    pub fn next_image(&mut self) {
        if self.current_index + 1 < self.image_paths.len() {
            self.current_index += 1;
            self.input_buffer.clear();
        }
    }

    /// Move to the previous image (without saving).
    pub fn prev_image(&mut self) {
        if self.current_index > 0 {
            self.current_index -= 1;
            self.input_buffer.clear();
        }
    }

    /// Reset zoom to fit view.
    pub fn reset_zoom(&mut self) {
        self.zoom_level = 1.0;
        self.pan_x = 0.0;
        self.pan_y = 0.0;
    }

    /// Zoom in by a factor.
    pub fn zoom_in(&mut self) {
        self.zoom_level = (self.zoom_level * 1.20).min(8.0);
    }

    /// Zoom out by a factor.
    pub fn zoom_out(&mut self) {
        self.zoom_level = (self.zoom_level / 1.20).max(1.0);
    }

    /// Apply pan delta in pixels.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        self.pan_x += dx;
        self.pan_y += dy;
    }

    /// Process number input and move file.
    /// Returns Ok(()) if successful, Err with message otherwise.
    pub fn confirm_number(&mut self) -> Result<(), String> {
        let number = self.input_buffer.trim().to_string();
        if number.is_empty() {
            return Err("Please enter a number".into());
        }

        let source_folder = self.source_folder.as_ref().ok_or("No folder selected")?;
        let current_path = self.current_image().ok_or("No image selected")?.clone();

        // Create target folder: <source_folder>/<number>/
        let target_folder = source_folder.join(&number);
        if !target_folder.exists() {
            std::fs::create_dir_all(&target_folder)
                .map_err(|e| format!("Failed to create folder: {e}"))?;
        }

        // Move file keeping original filename
        let filename = current_path.file_name().ok_or("Invalid filename")?;
        let target_path = target_folder.join(filename);

        // Check if target already exists
        if target_path.exists() {
            return Err(format!("File already exists: {}", target_path.display()));
        }

        std::fs::rename(&current_path, &target_path)
            .map_err(|e| format!("Failed to move file: {e}"))?;

        // Record for undo
        self.undo_stack.push(MoveOperation {
            original_path: current_path.clone(),
            new_path: target_path,
            number: number.clone(),
        });

        // Remove from list and advance
        self.image_paths.remove(self.current_index);

        // Adjust index if needed
        if self.current_index >= self.image_paths.len() && self.current_index > 0 {
            self.current_index -= 1;
        }

        // Clear input for next image
        self.input_buffer.clear();

        if self.image_paths.is_empty() {
            self.status_message = "All images processed!".into();
        } else {
            self.status_message =
                format!("Moved to {}/. {} remaining", number, self.image_paths.len());
        }

        Ok(())
    }

    /// Undo the last move operation.
    pub fn undo(&mut self) -> Result<(), String> {
        let op = self.undo_stack.pop().ok_or("Nothing to undo")?;

        // Move file back
        std::fs::rename(&op.new_path, &op.original_path)
            .map_err(|e| format!("Failed to undo: {e}"))?;

        // Re-add to image list at current position
        self.image_paths
            .insert(self.current_index, op.original_path);

        self.status_message = format!("Undid move of {}", op.number);
        Ok(())
    }

    /// Get progress as (current, total).
    pub fn progress(&self) -> (usize, usize) {
        let total = self.image_paths.len() + self.undo_stack.len();
        let done = self.undo_stack.len();
        (done, total)
    }

    /// Remaining count.
    pub fn remaining(&self) -> usize {
        self.image_paths.len()
    }
}
