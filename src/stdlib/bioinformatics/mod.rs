//! Bioinformatics Module
//!
//! This module provides comprehensive bioinformatics algorithms for sequence analysis,
//! phylogenetics, and genomics tools. All biological data types are implemented as
//! Foreign objects following the Foreign trait pattern.
//!
//! ## Main Areas:
//! - Sequence Alignment: Needleman-Wunsch, Smith-Waterman, BLAST-like algorithms
//! - Phylogenetics: Neighbor-joining trees, maximum likelihood estimation
//! - Genomics: ORF finding, GC content, translation, reverse complement
//!
//! ## Foreign Objects:
//! - BiologicalSequence: DNA, RNA, and protein sequences
//! - AlignmentResult: Pairwise and multiple sequence alignments
//! - PhylogeneticTree: Evolutionary trees and distances
//! - GenomicFeature: ORFs, motifs, and other genomic elements

pub mod alignment;
pub mod phylogenetics;
pub mod genomics;

use crate::vm::{Value, VmResult};

/// Register all bioinformatics functions with the standard library
pub fn register_bioinformatics_functions() -> std::collections::HashMap<String, crate::stdlib::StdlibFunction> {
    let mut functions = std::collections::HashMap::new();
    
    // Sequence Alignment Functions
    functions.insert("GlobalAlignment".to_string(), alignment::global_alignment);
    functions.insert("LocalAlignment".to_string(), alignment::local_alignment);
    functions.insert("MultipleAlignment".to_string(), alignment::multiple_alignment);
    functions.insert("BlastSearch".to_string(), alignment::blast_search);
    
    // Phylogenetic Functions
    functions.insert("PhylogeneticTree".to_string(), phylogenetics::phylogenetic_tree);
    functions.insert("NeighborJoining".to_string(), phylogenetics::neighbor_joining);
    functions.insert("MaximumLikelihood".to_string(), phylogenetics::maximum_likelihood);
    functions.insert("PairwiseDistance".to_string(), phylogenetics::pairwise_distance);
    
    // Genomics Functions
    functions.insert("BiologicalSequence".to_string(), genomics::biological_sequence);
    functions.insert("ReverseComplement".to_string(), genomics::reverse_complement);
    functions.insert("Translate".to_string(), genomics::translate);
    functions.insert("Transcribe".to_string(), genomics::transcribe);
    functions.insert("GCContent".to_string(), genomics::gc_content);
    functions.insert("FindORFs".to_string(), genomics::find_orfs);
    functions.insert("FindMotifs".to_string(), genomics::find_motifs);
    functions.insert("CodonUsage".to_string(), genomics::codon_usage);
    
    functions
}

/// Validates that input is a string (nucleotide or protein sequence)
pub fn validate_sequence_string(value: &Value) -> VmResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        _ => Err(crate::vm::VmError::TypeError {
            expected: "String sequence".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

/// Validates that input is a list of strings (multiple sequences)
pub fn validate_sequence_list(value: &Value) -> VmResult<Vec<String>> {
    match value {
        Value::List(items) => {
            let mut sequences = Vec::new();
            for item in items {
                sequences.push(validate_sequence_string(item)?);
            }
            Ok(sequences)
        }
        _ => Err(crate::vm::VmError::TypeError {
            expected: "List of String sequences".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

/// Validates that input is an integer (position, length, etc.)
pub fn validate_integer(value: &Value) -> VmResult<i64> {
    match value {
        Value::Integer(n) => Ok(*n),
        _ => Err(crate::vm::VmError::TypeError {
            expected: "Integer".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

/// Validates that input is a real number (score, probability, etc.)
pub fn validate_real(value: &Value) -> VmResult<f64> {
    match value {
        Value::Real(r) => Ok(*r),
        Value::Integer(n) => Ok(*n as f64),
        _ => Err(crate::vm::VmError::TypeError {
            expected: "Real or Integer".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}