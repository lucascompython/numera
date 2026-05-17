use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use opencv::calib3d;
use opencv::core::{DMatch, KeyPoint, Mat, NORM_HAMMING, Point2f, Size, Vector};
use opencv::features2d::{self, Feature2DTrait};
use opencv::imgcodecs;
use opencv::imgproc;
use opencv::prelude::*;

use super::event_config::EventConfig;

const LOWE_RATIO: f32 = 0.75;
const MIN_GOOD_MATCHES_FOR_HOMOGRAPHY: usize = 8;

pub struct StickerMatcher {
    event: EventConfig,
    template_gray: Mat,
    template_keypoints: Vector<KeyPoint>,
    template_descriptors: Mat,
}

#[derive(Debug)]
pub struct StickerDetection {
    pub source_path: PathBuf,
    pub source_width: i32,
    pub source_height: i32,
    pub found: bool,
    pub match_confidence: f32,
    pub good_match_count: usize,
    pub homography_valid: bool,
    pub warped_sticker: Option<Mat>,
    pub note: Option<String>,
}

impl StickerMatcher {
    pub fn new(event: EventConfig) -> Result<Self> {
        let template_path = event.sticker_template_path.to_string_lossy();
        let template_color = imgcodecs::imread(&template_path, imgcodecs::IMREAD_COLOR)
            .with_context(|| format!("failed to load sticker template {}", template_path))?;
        if template_color.empty() {
            return Err(anyhow!("sticker template is empty: {}", template_path));
        }

        let mut template_gray = Mat::default();
        imgproc::cvt_color_def(&template_color, &mut template_gray, imgproc::COLOR_BGR2GRAY)?;

        let mut orb = features2d::ORB::create_def()?;
        let mut template_keypoints = Vector::new();
        let mut template_descriptors = Mat::default();
        orb.detect_and_compute_def(
            &template_gray,
            &Mat::default(),
            &mut template_keypoints,
            &mut template_descriptors,
        )?;

        if template_keypoints.len() < MIN_GOOD_MATCHES_FOR_HOMOGRAPHY {
            return Err(anyhow!(
                "sticker template has too few ORB features ({})",
                template_keypoints.len()
            ));
        }

        Ok(Self {
            event,
            template_gray,
            template_keypoints,
            template_descriptors,
        })
    }

    pub fn detect(&mut self, image_path: &Path) -> Result<StickerDetection> {
        let image_path_str = image_path.to_string_lossy();
        let image_color = imgcodecs::imread(&image_path_str, imgcodecs::IMREAD_COLOR)
            .with_context(|| format!("failed to load image {}", image_path.display()))?;
        if image_color.empty() {
            return Err(anyhow!("image is empty: {}", image_path.display()));
        }

        let source_width = image_color.cols();
        let source_height = image_color.rows();

        let mut image_gray = Mat::default();
        imgproc::cvt_color_def(&image_color, &mut image_gray, imgproc::COLOR_BGR2GRAY)?;

        let mut orb = features2d::ORB::create_def()?;
        let mut image_keypoints = Vector::new();
        let mut image_descriptors = Mat::default();
        orb.detect_and_compute_def(
            &image_gray,
            &Mat::default(),
            &mut image_keypoints,
            &mut image_descriptors,
        )?;

        if image_keypoints.len() < MIN_GOOD_MATCHES_FOR_HOMOGRAPHY || image_descriptors.empty() {
            return Ok(self.not_found(
                image_path,
                source_width,
                source_height,
                "not enough image features",
            ));
        }

        let matcher = features2d::BFMatcher::create(NORM_HAMMING, false)?;
        let mut knn_matches: Vector<Vector<DMatch>> = Vector::new();
        matcher.knn_train_match_def(
            &self.template_descriptors,
            &image_descriptors,
            &mut knn_matches,
            2,
        )?;

        let mut good_matches = Vec::new();
        for pair in knn_matches.iter() {
            if pair.len() < 2 {
                continue;
            }
            let first = pair.get(0)?;
            let second = pair.get(1)?;
            if first.distance < LOWE_RATIO * second.distance {
                good_matches.push(first);
            }
        }

        if good_matches.len() < MIN_GOOD_MATCHES_FOR_HOMOGRAPHY {
            return Ok(self.not_found(
                image_path,
                source_width,
                source_height,
                "not enough good sticker matches",
            ));
        }

        let mut image_points: Vector<Point2f> = Vector::new();
        let mut template_points: Vector<Point2f> = Vector::new();
        for matched in &good_matches {
            let template_keypoint = self.template_keypoints.get(matched.query_idx as usize)?;
            let image_keypoint = image_keypoints.get(matched.train_idx as usize)?;
            template_points.push(template_keypoint.pt());
            image_points.push(image_keypoint.pt());
        }

        let mut inlier_mask = Mat::default();
        let homography = calib3d::find_homography(
            &image_points,
            &template_points,
            &mut inlier_mask,
            calib3d::RANSAC,
            4.0,
        )?;

        if homography.empty() {
            return Ok(self.not_found(
                image_path,
                source_width,
                source_height,
                "homography could not be estimated",
            ));
        }

        let inlier_count = count_inliers(&inlier_mask).unwrap_or(good_matches.len());
        let inlier_ratio = inlier_count as f32 / good_matches.len().max(1) as f32;
        let match_confidence = ((inlier_count as f32 / 28.0).min(1.0) * 0.65
            + inlier_ratio.min(1.0) * 0.35)
            .clamp(0.0, 1.0);

        let mut warped = Mat::default();
        imgproc::warp_perspective_def(
            &image_color,
            &mut warped,
            &homography,
            Size::new(self.event.template_width, self.event.template_height),
        )?;

        Ok(StickerDetection {
            source_path: image_path.to_path_buf(),
            source_width,
            source_height,
            found: true,
            match_confidence,
            good_match_count: good_matches.len(),
            homography_valid: true,
            warped_sticker: Some(warped),
            note: None,
        })
    }

    fn not_found(
        &self,
        image_path: &Path,
        source_width: i32,
        source_height: i32,
        note: impl Into<String>,
    ) -> StickerDetection {
        StickerDetection {
            source_path: image_path.to_path_buf(),
            source_width,
            source_height,
            found: false,
            match_confidence: 0.0,
            good_match_count: 0,
            homography_valid: false,
            warped_sticker: None,
            note: Some(note.into()),
        }
    }
}

fn count_inliers(mask: &Mat) -> Result<usize> {
    if mask.empty() {
        return Ok(0);
    }
    let mut count = 0usize;
    for row in 0..mask.rows() {
        let value = *mask.at_2d::<u8>(row, 0)?;
        if value != 0 {
            count += 1;
        }
    }
    Ok(count)
}
