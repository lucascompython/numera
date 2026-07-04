use rapidhash::fast::HashMapExt;
use rapidhash::fast::RapidHashMap as HashMap;

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
    pub min_visual_similarity: f32,
    pub min_topk_agreement: f32,
    pub max_conflicting_anchor_similarity: f32,
    pub min_anchor_count: usize,
    pub min_confidence: f32,
    pub debug_logging: bool,
}

impl Default for VisualMatcherConfig {
    fn default() -> Self {
        Self {
            model_name: MODEL_NAME.to_string(),
            top_k: 5,
            min_visual_similarity: 0.90,
            min_topk_agreement: 0.60,
            max_conflicting_anchor_similarity: 0.88,
            min_anchor_count: 2,
            min_confidence: 0.88,
            debug_logging: false,
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

#[derive(Debug)]
enum VisualDecision {
    Assign(VisualAssignmentRecord),
    Ambiguous {
        image_id: i64,
        confidence: f32,
        notes: String,
        matches: Vec<VisualMatchRecord>,
    },
    NeedsReview {
        image_id: i64,
        notes: String,
        matches: Vec<VisualMatchRecord>,
    },
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

    debug_log(
        config,
        format!(
            "model={} anchors={} candidates={} min_anchor_count={}",
            config.model_name,
            anchors.len(),
            candidates.len(),
            config.min_anchor_count
        ),
    );

    if anchors.len() < config.min_anchor_count || candidates.is_empty() {
        if anchors.len() < config.min_anchor_count {
            summary.weak = candidates.len();
            debug_log(
                config,
                format!(
                    "skipping visual assignment: only {} anchors available",
                    anchors.len()
                ),
            );
        }
        return Ok(summary);
    }

    for candidate in candidates {
        let decision = evaluate_candidate(&candidate, &anchors, config);
        match decision {
            VisualDecision::Assign(assignment) => {
                debug_log(
                    config,
                    format!(
                        "image={} assigned number={} confidence={:.3}; {}",
                        assignment.image_id,
                        assignment.final_number,
                        assignment.confidence,
                        format_matches(&assignment.matches)
                    ),
                );
                db::save_visual_assignment(conn, &assignment)?;
                summary.assigned += 1;
            }
            VisualDecision::Ambiguous {
                image_id,
                confidence,
                notes,
                matches,
            } => {
                debug_log(
                    config,
                    format!(
                        "image={} ambiguous confidence={:.3}: {}; {}",
                        image_id,
                        confidence,
                        notes,
                        format_matches(&matches)
                    ),
                );
                db::save_ambiguous_visual_assignment(
                    conn,
                    image_id,
                    confidence,
                    Some(notes),
                    &matches,
                )?;
                summary.ambiguous += 1;
            }
            VisualDecision::NeedsReview {
                image_id,
                notes,
                matches,
            } => {
                debug_log(
                    config,
                    format!(
                        "image={} not visually assigned: {}; {}",
                        image_id,
                        notes,
                        format_matches(&matches)
                    ),
                );
                db::save_visual_match_candidates(conn, image_id, &matches)?;
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
) -> VisualDecision {
    let scored = scored_anchors(candidate, anchors, config.top_k);
    let matches = matches_from_scored(candidate.image_id, &scored);
    let Some(best) = scored.first() else {
        return VisualDecision::NeedsReview {
            image_id: candidate.image_id,
            notes: "no anchors available after scoring".to_string(),
            matches,
        };
    };

    if best.similarity < config.min_visual_similarity {
        return VisualDecision::NeedsReview {
            image_id: candidate.image_id,
            notes: format!(
                "best visual similarity {:.3} is below min_visual_similarity {:.3}; previous_status={}",
                best.similarity, config.min_visual_similarity, candidate.status
            ),
            matches,
        };
    }

    let ranked_numbers = ranked_numbers(&scored);
    let Some(winner) = ranked_numbers.first() else {
        return VisualDecision::NeedsReview {
            image_id: candidate.image_id,
            notes: "top-k anchors did not produce a number vote".to_string(),
            matches,
        };
    };

    let agreement = winner.votes as f32 / scored.len().max(1) as f32;
    let conflict = strongest_conflict(&ranked_numbers, winner.number.as_str());
    if let Some(conflict) = conflict.as_ref()
        && conflict.best_similarity >= config.max_conflicting_anchor_similarity
    {
        return VisualDecision::Ambiguous {
            image_id: candidate.image_id,
            confidence: conflict.best_similarity,
            notes: format!(
                "conflicting anchor number {} similarity {:.3} exceeds max_conflicting_anchor_similarity {:.3}; winner={} agreement={:.2}; previous_status={}",
                conflict.number,
                conflict.best_similarity,
                config.max_conflicting_anchor_similarity,
                winner.number,
                agreement,
                candidate.status
            ),
            matches,
        };
    }

    if agreement < config.min_topk_agreement {
        let notes = format!(
            "winner={} agreement {:.2} is below min_topk_agreement {:.2}; previous_status={}",
            winner.number, agreement, config.min_topk_agreement, candidate.status
        );
        return if conflict.is_some() {
            VisualDecision::Ambiguous {
                image_id: candidate.image_id,
                confidence: winner.best_similarity,
                notes,
                matches,
            }
        } else {
            VisualDecision::NeedsReview {
                image_id: candidate.image_id,
                notes,
                matches,
            }
        };
    }

    let avg_similarity = winner.sum_similarity / winner.votes as f32;
    let anchor_confidence = scored
        .iter()
        .filter(|scored_anchor| scored_anchor.anchor.number == winner.number)
        .map(|scored_anchor| scored_anchor.anchor.assignment_confidence)
        .fold(0.0f32, f32::max);
    let confidence =
        (avg_similarity * 0.70 + agreement * 0.20 + anchor_confidence * 0.10).clamp(0.0, 1.0);
    if confidence < config.min_confidence {
        return VisualDecision::NeedsReview {
            image_id: candidate.image_id,
            notes: format!(
                "visual confidence {:.3} is below min_confidence {:.3}; winner={} avg_similarity={:.3} agreement={:.2}; previous_status={}",
                confidence,
                config.min_confidence,
                winner.number,
                avg_similarity,
                agreement,
                candidate.status
            ),
            matches,
        };
    }

    VisualDecision::Assign(VisualAssignmentRecord {
        image_id: candidate.image_id,
        final_number: winner.number.clone(),
        confidence,
        notes: Some(format!(
            "visual match from {} top anchors; avg_similarity={avg_similarity:.3}, agreement={agreement:.2}, anchor_confidence={anchor_confidence:.3}, previous_status={}, file={}",
            scored.len(),
            candidate.status,
            candidate.file_path.display()
        )),
        matches,
    })
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

#[derive(Debug, Clone)]
struct NumberVote {
    number: String,
    votes: usize,
    sum_similarity: f32,
    best_similarity: f32,
}

fn ranked_numbers(scored: &[ScoredAnchor<'_>]) -> Vec<NumberVote> {
    let mut by_number: HashMap<&str, NumberVote> = HashMap::new();
    for scored_anchor in scored {
        let entry = by_number
            .entry(scored_anchor.anchor.number.as_str())
            .or_insert_with(|| NumberVote {
                number: scored_anchor.anchor.number.clone(),
                votes: 0,
                sum_similarity: 0.0,
                best_similarity: 0.0,
            });
        entry.votes += 1;
        entry.sum_similarity += scored_anchor.similarity;
        entry.best_similarity = f32::max(entry.best_similarity, scored_anchor.similarity);
    }

    let mut ranked: Vec<_> = by_number.into_values().collect();
    ranked.sort_by(|left, right| {
        right
            .votes
            .cmp(&left.votes)
            .then_with(|| {
                let right_avg = right.sum_similarity / right.votes.max(1) as f32;
                let left_avg = left.sum_similarity / left.votes.max(1) as f32;
                right_avg.total_cmp(&left_avg)
            })
            .then_with(|| right.best_similarity.total_cmp(&left.best_similarity))
    });
    ranked
}

fn strongest_conflict<'a>(
    ranked_numbers: &'a [NumberVote],
    winner_number: &str,
) -> Option<&'a NumberVote> {
    ranked_numbers
        .iter()
        .filter(|vote| vote.number != winner_number)
        .max_by(|left, right| left.best_similarity.total_cmp(&right.best_similarity))
}

fn matches_from_scored(image_id: i64, scored: &[ScoredAnchor<'_>]) -> Vec<VisualMatchRecord> {
    scored
        .iter()
        .enumerate()
        .map(|(rank, scored)| VisualMatchRecord {
            image_id,
            matched_anchor_image_id: scored.anchor.image_id,
            matched_number: scored.anchor.number.clone(),
            similarity: scored.similarity,
            rank: rank as i32 + 1,
        })
        .collect()
}

fn format_matches(matches: &[VisualMatchRecord]) -> String {
    if matches.is_empty() {
        return "top_k=[]".to_string();
    }

    let parts = matches
        .iter()
        .map(|matched| {
            format!(
                "#{} image={} number={} sim={:.3}",
                matched.rank,
                matched.matched_anchor_image_id,
                matched.matched_number,
                matched.similarity
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("top_k=[{parts}]")
}

fn debug_log(config: &VisualMatcherConfig, message: String) {
    if config.debug_logging {
        eprintln!("[event_sorter::visual_matcher] {message}");
    }
}
