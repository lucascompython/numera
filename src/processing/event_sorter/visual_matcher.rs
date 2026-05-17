use std::collections::HashMap;

use anyhow::Result;
use rusqlite::Connection;

use super::db::{
    self, AnchorEmbedding, CandidateEmbedding, VisualAssignmentRecord, VisualMatchRecord,
};
use super::embedding::{MODEL_NAME, cosine_similarity};

#[derive(Debug, Clone)]
pub struct VisualMatcherConfig {
    pub model_name: String,
    pub top_k: usize,
    pub min_similarity: f32,
    pub min_margin: f32,
    pub min_anchor_agreement: f32,
    pub min_confidence: f32,
}

impl Default for VisualMatcherConfig {
    fn default() -> Self {
        Self {
            model_name: MODEL_NAME.to_string(),
            top_k: 5,
            min_similarity: 0.90,
            min_margin: 0.035,
            min_anchor_agreement: 0.60,
            min_confidence: 0.88,
        }
    }
}

#[derive(Debug, Default)]
pub struct VisualMatchSummary {
    pub anchors: usize,
    pub candidates: usize,
    pub assigned: usize,
    pub ambiguous: usize,
    pub weak: usize,
}

#[derive(Debug, Clone)]
struct ScoredAnchor<'a> {
    anchor: &'a AnchorEmbedding,
    similarity: f32,
}

pub fn run_anchor_visual_matching(
    conn: &mut Connection,
    event_id: i64,
    config: &VisualMatcherConfig,
) -> Result<VisualMatchSummary> {
    let anchors = db::load_anchor_embeddings(conn, event_id, &config.model_name)?;
    let candidates = db::load_visual_candidates(conn, event_id, &config.model_name)?;
    let mut summary = VisualMatchSummary {
        anchors: anchors.len(),
        candidates: candidates.len(),
        ..VisualMatchSummary::default()
    };

    if anchors.is_empty() || candidates.is_empty() {
        return Ok(summary);
    }

    for candidate in candidates {
        let decision = evaluate_candidate(&candidate, &anchors, config);
        let top_matches = top_matches(&candidate, &anchors, config.top_k);

        if let Some(decision) = decision {
            db::save_visual_assignment(conn, &decision)?;
            summary.assigned += 1;
        } else {
            db::save_visual_match_candidates(conn, candidate.image_id, &top_matches)?;
            if is_conflicting(&top_matches) {
                summary.ambiguous += 1;
            } else {
                summary.weak += 1;
            }
        }
    }

    Ok(summary)
}

fn evaluate_candidate(
    candidate: &CandidateEmbedding,
    anchors: &[AnchorEmbedding],
    config: &VisualMatcherConfig,
) -> Option<VisualAssignmentRecord> {
    let scored = scored_anchors(candidate, anchors, config.top_k);
    let best = scored.first()?;
    if best.similarity < config.min_similarity {
        return None;
    }

    let mut by_number: HashMap<&str, (usize, f32, f32)> = HashMap::new();
    for scored_anchor in &scored {
        let entry = by_number
            .entry(scored_anchor.anchor.number.as_str())
            .or_insert((0, 0.0, 0.0));
        entry.0 += 1;
        entry.1 += scored_anchor.similarity;
        entry.2 = f32::max(entry.2, scored_anchor.similarity);
    }

    let mut ranked_numbers: Vec<_> = by_number.into_iter().collect();
    ranked_numbers.sort_by(|(_, left), (_, right)| {
        let left_avg = left.1 / left.0 as f32;
        let right_avg = right.1 / right.0 as f32;
        right_avg.total_cmp(&left_avg)
    });

    let (winner_number, (winner_votes, winner_sum, winner_best)) = ranked_numbers.first()?;
    let agreement = *winner_votes as f32 / scored.len().max(1) as f32;
    if agreement < config.min_anchor_agreement {
        return None;
    }

    let next_best = ranked_numbers
        .iter()
        .skip(1)
        .map(|(_, (_, _, best))| *best)
        .fold(0.0f32, f32::max);
    let margin = *winner_best - next_best;
    if next_best > 0.0 && margin < config.min_margin {
        return None;
    }

    let avg_similarity = *winner_sum / *winner_votes as f32;
    let anchor_confidence = scored
        .iter()
        .filter(|scored_anchor| scored_anchor.anchor.number == *winner_number)
        .map(|scored_anchor| scored_anchor.anchor.assignment_confidence)
        .fold(0.0f32, f32::max);
    let confidence =
        (avg_similarity * 0.70 + agreement * 0.20 + anchor_confidence * 0.10).clamp(0.0, 1.0);
    if confidence < config.min_confidence {
        return None;
    }

    let matches = top_matches(candidate, anchors, config.top_k);
    Some(VisualAssignmentRecord {
        image_id: candidate.image_id,
        final_number: (*winner_number).to_string(),
        confidence,
        notes: Some(format!(
            "visual match from {} top anchors; avg_similarity={avg_similarity:.3}, agreement={agreement:.2}, margin={margin:.3}, previous_status={}",
            scored.len(),
            candidate.status
        )),
        matches,
    })
}

fn top_matches(
    candidate: &CandidateEmbedding,
    anchors: &[AnchorEmbedding],
    top_k: usize,
) -> Vec<VisualMatchRecord> {
    scored_anchors(candidate, anchors, top_k)
        .into_iter()
        .enumerate()
        .map(|(rank, scored)| VisualMatchRecord {
            image_id: candidate.image_id,
            matched_anchor_image_id: scored.anchor.image_id,
            matched_number: scored.anchor.number.clone(),
            similarity: scored.similarity,
            rank: rank as i32 + 1,
        })
        .collect()
}

fn scored_anchors<'a>(
    candidate: &CandidateEmbedding,
    anchors: &'a [AnchorEmbedding],
    top_k: usize,
) -> Vec<ScoredAnchor<'a>> {
    let mut scored: Vec<_> = anchors
        .iter()
        .map(|anchor| ScoredAnchor {
            anchor,
            similarity: cosine_similarity(&candidate.embedding, &anchor.embedding),
        })
        .collect();

    scored.sort_by(|left, right| right.similarity.total_cmp(&left.similarity));
    scored.truncate(top_k.max(1));
    scored
}

fn is_conflicting(matches: &[VisualMatchRecord]) -> bool {
    let Some(first) = matches.first() else {
        return false;
    };
    matches
        .iter()
        .any(|matched| matched.matched_number != first.matched_number)
}
