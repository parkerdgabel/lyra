//! Sequence Alignment Algorithms
//!
//! This module implements various sequence alignment algorithms including:
//! - Needleman-Wunsch global alignment (dynamic programming)
//! - Smith-Waterman local alignment (dynamic programming)
//! - BLAST-like local search algorithm
//! - Multiple sequence alignment (progressive method)

use crate::{
    foreign::{Foreign, ForeignError, LyObj},
    vm::{Value, VmError, VmResult},
};
use std::any::Any;
use std::cmp;

/// Scoring parameters for sequence alignment
#[derive(Debug, Clone)]
pub struct ScoringMatrix {
    pub match_score: i32,
    pub mismatch_score: i32,
    pub gap_penalty: i32,
}

impl Default for ScoringMatrix {
    fn default() -> Self {
        Self {
            match_score: 2,
            mismatch_score: -1,
            gap_penalty: -2,
        }
    }
}

/// Foreign object representing an alignment result
#[derive(Debug, Clone, PartialEq)]
pub struct AlignmentResult {
    pub aligned_seq1: String,
    pub aligned_seq2: String,
    pub score: f64,
    pub start1: Option<usize>,
    pub end1: Option<usize>,
    pub start2: Option<usize>, 
    pub end2: Option<usize>,
    pub alignment_type: AlignmentType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AlignmentType {
    Global,
    Local,
}

impl Foreign for AlignmentResult {
    fn type_name(&self) -> &'static str {
        "AlignmentResult"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "score" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Real(self.score))
            }
            "alignedSequence1" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::String(self.aligned_seq1.clone()))
            }
            "alignedSequence2" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::String(self.aligned_seq2.clone()))
            }
            "start1" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.start1.unwrap_or(0) as i64))
            }
            "end1" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.end1.unwrap_or(self.aligned_seq1.len()) as i64))
            }
            "start2" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.start2.unwrap_or(0) as i64))
            }
            "end2" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.end2.unwrap_or(self.aligned_seq2.len()) as i64))
            }
            "alignmentType" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let type_str = match self.alignment_type {
                    AlignmentType::Global => "Global",
                    AlignmentType::Local => "Local",
                };
                Ok(Value::String(type_str.to_string()))
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Foreign object representing multiple sequence alignment result
#[derive(Debug, Clone, PartialEq)]
pub struct MultipleAlignmentResult {
    pub aligned_sequences: Vec<String>,
    pub total_score: f64,
    pub consensus: String,
}

impl Foreign for MultipleAlignmentResult {
    fn type_name(&self) -> &'static str {
        "MultipleAlignmentResult"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "alignedSequences" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let sequences: Vec<Value> = self.aligned_sequences
                    .iter()
                    .map(|s| Value::String(s.clone()))
                    .collect();
                Ok(Value::List(sequences))
            }
            "totalScore" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Real(self.total_score))
            }
            "consensus" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::String(self.consensus.clone()))
            }
            "numSequences" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.aligned_sequences.len() as i64))
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Foreign object representing a BLAST hit
#[derive(Debug, Clone, PartialEq)]
pub struct BlastHit {
    pub query_start: usize,
    pub query_end: usize,
    pub subject_start: usize,
    pub subject_end: usize,
    pub subject_index: usize,
    pub score: f64,
    pub e_value: f64,
    pub identity: f64,
}

impl Foreign for BlastHit {
    fn type_name(&self) -> &'static str {
        "BlastHit"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "score" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Real(self.score))
            }
            "eValue" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Real(self.e_value))
            }
            "position" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.subject_start as i64))
            }
            "queryStart" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.query_start as i64))
            }
            "queryEnd" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.query_end as i64))
            }
            "subjectStart" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.subject_start as i64))
            }
            "subjectEnd" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.subject_end as i64))
            }
            "subjectIndex" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.subject_index as i64))
            }
            "identity" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Real(self.identity))
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Needleman-Wunsch global alignment algorithm
fn needleman_wunsch(seq1: &str, seq2: &str, scoring: &ScoringMatrix) -> AlignmentResult {
    let seq1_chars: Vec<char> = seq1.chars().collect();
    let seq2_chars: Vec<char> = seq2.chars().collect();
    let m = seq1_chars.len();
    let n = seq2_chars.len();
    
    // Initialize scoring matrix
    let mut score_matrix = vec![vec![0i32; n + 1]; m + 1];
    
    // Initialize first row and column
    for i in 0..=m {
        score_matrix[i][0] = (i as i32) * scoring.gap_penalty;
    }
    for j in 0..=n {
        score_matrix[0][j] = (j as i32) * scoring.gap_penalty;
    }
    
    // Fill the scoring matrix
    for i in 1..=m {
        for j in 1..=n {
            let match_score = if seq1_chars[i-1] == seq2_chars[j-1] {
                scoring.match_score
            } else {
                scoring.mismatch_score
            };
            
            score_matrix[i][j] = cmp::max(
                cmp::max(
                    score_matrix[i-1][j-1] + match_score, // diagonal
                    score_matrix[i-1][j] + scoring.gap_penalty, // from above
                ),
                score_matrix[i][j-1] + scoring.gap_penalty, // from left
            );
        }
    }
    
    // Traceback to construct alignment
    let mut aligned_seq1 = String::new();
    let mut aligned_seq2 = String::new();
    let mut i = m;
    let mut j = n;
    
    while i > 0 || j > 0 {
        if i > 0 && j > 0 {
            let match_score = if seq1_chars[i-1] == seq2_chars[j-1] {
                scoring.match_score
            } else {
                scoring.mismatch_score
            };
            
            if score_matrix[i][j] == score_matrix[i-1][j-1] + match_score {
                aligned_seq1.push(seq1_chars[i-1]);
                aligned_seq2.push(seq2_chars[j-1]);
                i -= 1;
                j -= 1;
            } else if score_matrix[i][j] == score_matrix[i-1][j] + scoring.gap_penalty {
                aligned_seq1.push(seq1_chars[i-1]);
                aligned_seq2.push('-');
                i -= 1;
            } else {
                aligned_seq1.push('-');
                aligned_seq2.push(seq2_chars[j-1]);
                j -= 1;
            }
        } else if i > 0 {
            aligned_seq1.push(seq1_chars[i-1]);
            aligned_seq2.push('-');
            i -= 1;
        } else {
            aligned_seq1.push('-');
            aligned_seq2.push(seq2_chars[j-1]);
            j -= 1;
        }
    }
    
    // Reverse the aligned sequences
    aligned_seq1 = aligned_seq1.chars().rev().collect();
    aligned_seq2 = aligned_seq2.chars().rev().collect();
    
    AlignmentResult {
        aligned_seq1,
        aligned_seq2,
        score: score_matrix[m][n] as f64,
        start1: Some(0),
        end1: Some(m),
        start2: Some(0),
        end2: Some(n),
        alignment_type: AlignmentType::Global,
    }
}

/// Smith-Waterman local alignment algorithm
fn smith_waterman(seq1: &str, seq2: &str, scoring: &ScoringMatrix) -> AlignmentResult {
    let seq1_chars: Vec<char> = seq1.chars().collect();
    let seq2_chars: Vec<char> = seq2.chars().collect();
    let m = seq1_chars.len();
    let n = seq2_chars.len();
    
    // Initialize scoring matrix
    let mut score_matrix = vec![vec![0i32; n + 1]; m + 1];
    let mut max_score = 0i32;
    let mut max_i = 0;
    let mut max_j = 0;
    
    // Fill the scoring matrix
    for i in 1..=m {
        for j in 1..=n {
            let match_score = if seq1_chars[i-1] == seq2_chars[j-1] {
                scoring.match_score
            } else {
                scoring.mismatch_score
            };
            
            score_matrix[i][j] = cmp::max(
                0, // Local alignment allows zero score
                cmp::max(
                    cmp::max(
                        score_matrix[i-1][j-1] + match_score, // diagonal
                        score_matrix[i-1][j] + scoring.gap_penalty, // from above
                    ),
                    score_matrix[i][j-1] + scoring.gap_penalty, // from left
                ),
            );
            
            // Track maximum score for optimal local alignment
            if score_matrix[i][j] > max_score {
                max_score = score_matrix[i][j];
                max_i = i;
                max_j = j;
            }
        }
    }
    
    // Traceback from the maximum score position
    let mut aligned_seq1 = String::new();
    let mut aligned_seq2 = String::new();
    let mut i = max_i;
    let mut j = max_j;
    let end1 = i;
    let end2 = j;
    
    while i > 0 && j > 0 && score_matrix[i][j] > 0 {
        let match_score = if seq1_chars[i-1] == seq2_chars[j-1] {
            scoring.match_score
        } else {
            scoring.mismatch_score
        };
        
        if score_matrix[i][j] == score_matrix[i-1][j-1] + match_score {
            aligned_seq1.push(seq1_chars[i-1]);
            aligned_seq2.push(seq2_chars[j-1]);
            i -= 1;
            j -= 1;
        } else if i > 0 && score_matrix[i][j] == score_matrix[i-1][j] + scoring.gap_penalty {
            aligned_seq1.push(seq1_chars[i-1]);
            aligned_seq2.push('-');
            i -= 1;
        } else if j > 0 && score_matrix[i][j] == score_matrix[i][j-1] + scoring.gap_penalty {
            aligned_seq1.push('-');
            aligned_seq2.push(seq2_chars[j-1]);
            j -= 1;
        } else {
            break;
        }
    }
    
    let start1 = i;
    let start2 = j;
    
    // Reverse the aligned sequences
    aligned_seq1 = aligned_seq1.chars().rev().collect();
    aligned_seq2 = aligned_seq2.chars().rev().collect();
    
    AlignmentResult {
        aligned_seq1,
        aligned_seq2,
        score: max_score as f64,
        start1: Some(start1),
        end1: Some(end1),
        start2: Some(start2),
        end2: Some(end2),
        alignment_type: AlignmentType::Local,
    }
}

/// BLAST-like heuristic search algorithm
fn blast_search_impl(query: &str, database: &[String], word_size: usize) -> Vec<BlastHit> {
    let mut hits = Vec::new();
    
    if query.len() < word_size {
        return hits;
    }
    
    // Generate all k-mers (words) from the query
    let query_chars: Vec<char> = query.chars().collect();
    let mut words = std::collections::HashMap::new();
    
    for i in 0..=(query_chars.len() - word_size) {
        let word: String = query_chars[i..(i + word_size)].iter().collect();
        words.entry(word).or_insert(Vec::new()).push(i);
    }
    
    // Search for matches in each database sequence
    for (db_idx, db_seq) in database.iter().enumerate() {
        let db_chars: Vec<char> = db_seq.chars().collect();
        
        for i in 0..=(db_chars.len().saturating_sub(word_size)) {
            let db_word: String = db_chars[i..(i + word_size)].iter().collect();
            
            if let Some(query_positions) = words.get(&db_word) {
                // Found a matching word, try to extend the alignment
                for &query_pos in query_positions {
                    let hit = extend_alignment(
                        query,
                        db_seq,
                        query_pos,
                        i,
                        word_size,
                        db_idx,
                    );
                    
                    // Only include hits above a threshold score
                    if hit.score >= word_size as f64 {
                        hits.push(hit);
                    }
                }
            }
        }
    }
    
    // Sort hits by score (descending)
    hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    
    // Remove redundant hits (keep best non-overlapping hits)
    remove_overlapping_hits(&mut hits);
    
    hits
}

/// Extend a BLAST hit in both directions
fn extend_alignment(
    query: &str,
    subject: &str,
    query_start: usize,
    subject_start: usize,
    word_size: usize,
    subject_index: usize,
) -> BlastHit {
    let query_chars: Vec<char> = query.chars().collect();
    let subject_chars: Vec<char> = subject.chars().collect();
    
    let mut score = word_size as f64; // Initial score for the matching word
    let mut identity_matches = word_size;
    let mut total_length = word_size;
    
    // Extend to the right
    let mut q_pos = query_start + word_size;
    let mut s_pos = subject_start + word_size;
    
    while q_pos < query_chars.len() && s_pos < subject_chars.len() {
        if query_chars[q_pos] == subject_chars[s_pos] {
            score += 2.0; // Match
            identity_matches += 1;
        } else {
            score -= 1.0; // Mismatch
        }
        total_length += 1;
        
        // Stop if score drops too much
        if score < (word_size as f64) / 2.0 {
            break;
        }
        
        q_pos += 1;
        s_pos += 1;
    }
    
    let query_end = q_pos;
    let subject_end = s_pos;
    
    // Extend to the left
    let mut q_pos = query_start;
    let mut s_pos = subject_start;
    
    while q_pos > 0 && s_pos > 0 {
        q_pos -= 1;
        s_pos -= 1;
        
        if query_chars[q_pos] == subject_chars[s_pos] {
            score += 2.0; // Match
            identity_matches += 1;
        } else {
            score -= 1.0; // Mismatch
        }
        total_length += 1;
        
        // Stop if score drops too much
        if score < (word_size as f64) / 2.0 {
            q_pos += 1;
            s_pos += 1;
            break;
        }
    }
    
    let query_start_final = q_pos;
    let subject_start_final = s_pos;
    
    // Calculate E-value (simplified)
    let e_value = calculate_e_value(score, query.len(), subject.len());
    let identity = identity_matches as f64 / total_length as f64;
    
    BlastHit {
        query_start: query_start_final,
        query_end,
        subject_start: subject_start_final,
        subject_end,
        subject_index,
        score,
        e_value,
        identity,
    }
}

/// Calculate E-value for a BLAST hit (simplified calculation)
fn calculate_e_value(score: f64, query_len: usize, subject_len: usize) -> f64 {
    // Simplified E-value calculation
    // In practice, this would use more sophisticated statistics
    let lambda = 0.318;
    let k = 0.13;
    let n = query_len * subject_len;
    
    (k * n as f64) * (-lambda * score).exp()
}

/// Remove overlapping BLAST hits, keeping the best scoring ones
fn remove_overlapping_hits(hits: &mut Vec<BlastHit>) {
    hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    
    let mut filtered: Vec<BlastHit> = Vec::new();
    
    for hit in hits.iter() {
        let mut overlaps = false;
        
        for existing in &filtered {
            if hit.subject_index == existing.subject_index && hits_overlap(hit, existing) {
                overlaps = true;
                break;
            }
        }
        
        if !overlaps {
            filtered.push(hit.clone());
        }
    }
    
    *hits = filtered;
}

/// Check if two BLAST hits overlap significantly
fn hits_overlap(hit1: &BlastHit, hit2: &BlastHit) -> bool {
    let q_overlap = !(hit1.query_end <= hit2.query_start || hit2.query_end <= hit1.query_start);
    let s_overlap = !(hit1.subject_end <= hit2.subject_start || hit2.subject_end <= hit1.subject_start);
    
    q_overlap && s_overlap
}

/// Progressive multiple sequence alignment
fn progressive_multiple_alignment(sequences: &[String], scoring: &ScoringMatrix) -> MultipleAlignmentResult {
    if sequences.len() < 2 {
        return MultipleAlignmentResult {
            aligned_sequences: sequences.to_vec(),
            total_score: 0.0,
            consensus: sequences.first().unwrap_or(&String::new()).clone(),
        };
    }
    
    // Start with pairwise alignment of first two sequences
    let first_alignment = needleman_wunsch(&sequences[0], &sequences[1], scoring);
    let mut aligned_sequences = vec![
        first_alignment.aligned_seq1.clone(),
        first_alignment.aligned_seq2.clone(),
    ];
    let mut total_score = first_alignment.score;
    
    // Add remaining sequences one by one
    for i in 2..sequences.len() {
        // Align current sequence to the consensus of existing alignment
        let consensus = build_consensus(&aligned_sequences);
        let pairwise_alignment = needleman_wunsch(&consensus, &sequences[i], scoring);
        
        // Update existing alignments to match the new alignment length
        let new_length = pairwise_alignment.aligned_seq1.len();
        aligned_sequences = update_multiple_alignment(&aligned_sequences, &pairwise_alignment.aligned_seq1, new_length);
        aligned_sequences.push(pairwise_alignment.aligned_seq2);
        
        total_score += pairwise_alignment.score;
    }
    
    let consensus = build_consensus(&aligned_sequences);
    
    MultipleAlignmentResult {
        aligned_sequences,
        total_score,
        consensus,
    }
}

/// Build consensus sequence from multiple aligned sequences
fn build_consensus(sequences: &[String]) -> String {
    if sequences.is_empty() {
        return String::new();
    }
    
    let seq_len = sequences[0].len();
    let mut consensus = String::new();
    
    for pos in 0..seq_len {
        let mut counts = std::collections::HashMap::new();
        
        for seq in sequences {
            if pos < seq.len() {
                let ch = seq.chars().nth(pos).unwrap_or('-');
                *counts.entry(ch).or_insert(0) += 1;
            }
        }
        
        // Find most frequent character (excluding gaps when possible)
        let most_frequent = counts.iter()
            .filter(|(&ch, _)| ch != '-')
            .max_by_key(|(_, &count)| count)
            .or_else(|| counts.iter().max_by_key(|(_, &count)| count))
            .map(|(&ch, _)| ch)
            .unwrap_or('-');
        
        consensus.push(most_frequent);
    }
    
    consensus
}

/// Update existing multiple alignment when adding a new sequence
fn update_multiple_alignment(
    existing: &[String],
    consensus_alignment: &str,
    new_length: usize,
) -> Vec<String> {
    let mut updated = Vec::new();
    
    for seq in existing {
        let mut new_seq = String::new();
        let mut seq_pos = 0;
        
        for ch in consensus_alignment.chars() {
            if ch == '-' {
                new_seq.push('-');
            } else {
                if seq_pos < seq.len() {
                    new_seq.push(seq.chars().nth(seq_pos).unwrap());
                    seq_pos += 1;
                } else {
                    new_seq.push('-');
                }
            }
        }
        
        // Pad to new length if necessary
        while new_seq.len() < new_length {
            new_seq.push('-');
        }
        
        updated.push(new_seq);
    }
    
    updated
}

// Public API functions for the Lyra stdlib

/// GlobalAlignment[seq1, seq2] or GlobalAlignment[seq1, seq2, match, mismatch, gap]
pub fn global_alignment(args: &[Value]) -> VmResult<Value> {
    match args.len() {
        2 => {
            let seq1 = crate::stdlib::bioinformatics::validate_sequence_string(&args[0])?;
            let seq2 = crate::stdlib::bioinformatics::validate_sequence_string(&args[1])?;
            
            if seq1.is_empty() || seq2.is_empty() {
                return Err(VmError::Runtime("Sequences cannot be empty".to_string()));
            }
            
            let scoring = ScoringMatrix::default();
            let result = needleman_wunsch(&seq1, &seq2, &scoring);
            Ok(Value::LyObj(LyObj::new(Box::new(result))))
        }
        5 => {
            let seq1 = crate::stdlib::bioinformatics::validate_sequence_string(&args[0])?;
            let seq2 = crate::stdlib::bioinformatics::validate_sequence_string(&args[1])?;
            let match_score = crate::stdlib::bioinformatics::validate_integer(&args[2])? as i32;
            let mismatch_score = crate::stdlib::bioinformatics::validate_integer(&args[3])? as i32;
            let gap_penalty = crate::stdlib::bioinformatics::validate_integer(&args[4])? as i32;
            
            if seq1.is_empty() || seq2.is_empty() {
                return Err(VmError::Runtime("Sequences cannot be empty".to_string()));
            }
            
            let scoring = ScoringMatrix {
                match_score,
                mismatch_score,
                gap_penalty,
            };
            let result = needleman_wunsch(&seq1, &seq2, &scoring);
            Ok(Value::LyObj(LyObj::new(Box::new(result))))
        }
        _ => Err(VmError::Runtime(format!("Invalid number of arguments: expected 2, got {}", args.len()))),
    }
}

/// LocalAlignment[seq1, seq2] or LocalAlignment[seq1, seq2, match, mismatch, gap]
pub fn local_alignment(args: &[Value]) -> VmResult<Value> {
    match args.len() {
        2 => {
            let seq1 = crate::stdlib::bioinformatics::validate_sequence_string(&args[0])?;
            let seq2 = crate::stdlib::bioinformatics::validate_sequence_string(&args[1])?;
            
            if seq1.is_empty() || seq2.is_empty() {
                return Err(VmError::Runtime("Sequences cannot be empty".to_string()));
            }
            
            let scoring = ScoringMatrix::default();
            let result = smith_waterman(&seq1, &seq2, &scoring);
            Ok(Value::LyObj(LyObj::new(Box::new(result))))
        }
        5 => {
            let seq1 = crate::stdlib::bioinformatics::validate_sequence_string(&args[0])?;
            let seq2 = crate::stdlib::bioinformatics::validate_sequence_string(&args[1])?;
            let match_score = crate::stdlib::bioinformatics::validate_integer(&args[2])? as i32;
            let mismatch_score = crate::stdlib::bioinformatics::validate_integer(&args[3])? as i32;
            let gap_penalty = crate::stdlib::bioinformatics::validate_integer(&args[4])? as i32;
            
            if seq1.is_empty() || seq2.is_empty() {
                return Err(VmError::Runtime("Sequences cannot be empty".to_string()));
            }
            
            let scoring = ScoringMatrix {
                match_score,
                mismatch_score,
                gap_penalty,
            };
            let result = smith_waterman(&seq1, &seq2, &scoring);
            Ok(Value::LyObj(LyObj::new(Box::new(result))))
        }
        _ => Err(VmError::Runtime(format!("Invalid number of arguments: expected 2, got {}", args.len()))),
    }
}

/// MultipleAlignment[{seq1, seq2, seq3, ...}]
pub fn multiple_alignment(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!("Invalid number of arguments: expected 1, got {}", args.len())));
    }
    
    let sequences = crate::stdlib::bioinformatics::validate_sequence_list(&args[0])?;
    
    if sequences.len() < 2 {
        return Err(VmError::Runtime("Need at least 2 sequences for multiple alignment".to_string()));
    }
    
    for seq in &sequences {
        if seq.is_empty() {
            return Err(VmError::Runtime("Sequences cannot be empty".to_string()));
        }
    }
    
    let scoring = ScoringMatrix::default();
    let result = progressive_multiple_alignment(&sequences, &scoring);
    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}

/// BlastSearch[query, database] or BlastSearch[query, database, word_size]
pub fn blast_search(args: &[Value]) -> VmResult<Value> {
    let (query, database, word_size) = match args.len() {
        2 => {
            let query = crate::stdlib::bioinformatics::validate_sequence_string(&args[0])?;
            let database = crate::stdlib::bioinformatics::validate_sequence_list(&args[1])?;
            (query, database, 3) // Default word size
        }
        3 => {
            let query = crate::stdlib::bioinformatics::validate_sequence_string(&args[0])?;
            let database = crate::stdlib::bioinformatics::validate_sequence_list(&args[1])?;
            let word_size = crate::stdlib::bioinformatics::validate_integer(&args[2])? as usize;
            (query, database, word_size)
        }
        _ => {
            return Err(VmError::Runtime(format!("Invalid number of arguments: expected 2 or 3, got {}", args.len())));
        }
    };
    
    if query.is_empty() {
        return Err(VmError::Runtime("Query sequence cannot be empty".to_string()));
    }
    
    if database.is_empty() {
        return Err(VmError::Runtime("Database cannot be empty".to_string()));
    }
    
    if word_size == 0 || word_size > query.len() {
        return Err(VmError::Runtime(format!("Invalid word size: {}. Must be > 0 and <= query length", word_size)));
    }
    
    let hits = blast_search_impl(&query, &database, word_size);
    let hit_values: Vec<Value> = hits.into_iter()
        .map(|hit| Value::LyObj(LyObj::new(Box::new(hit))))
        .collect();
    
    Ok(Value::List(hit_values))
}