//! Genomics Analysis Module
//!
//! This module implements core genomics analysis functions:
//! - Biological sequence management (DNA, RNA, protein)
//! - Sequence transformations (reverse complement, transcription, translation)
//! - ORF (Open Reading Frame) finding
//! - GC content calculation
//! - Motif finding
//! - Codon usage analysis

use crate::{
    foreign::{Foreign, ForeignError, LyObj},
    vm::{Value, VmError, VmResult},
};
use std::any::Any;
use std::collections::HashMap;

/// Sequence types supported by the bioinformatics module
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SequenceType {
    DNA,
    RNA,
    Protein,
}

impl SequenceType {
    fn from_string(s: &str) -> Result<Self, String> {
        match s.to_uppercase().as_str() {
            "DNA" => Ok(SequenceType::DNA),
            "RNA" => Ok(SequenceType::RNA),
            "PROTEIN" | "AMINO_ACID" | "AA" => Ok(SequenceType::Protein),
            _ => Err(format!("Unknown sequence type: {}", s)),
        }
    }
    
    fn to_string(&self) -> String {
        match self {
            SequenceType::DNA => "DNA".to_string(),
            SequenceType::RNA => "RNA".to_string(),
            SequenceType::Protein => "Protein".to_string(),
        }
    }
    
    fn valid_characters(&self) -> &'static str {
        match self {
            SequenceType::DNA => "ATCGN",
            SequenceType::RNA => "AUCGN",
            SequenceType::Protein => "ACDEFGHIKLMNPQRSTVWYUX*",
        }
    }
}

/// Foreign object representing a biological sequence
#[derive(Debug, Clone, PartialEq)]
pub struct BiologicalSequence {
    pub sequence: String,
    pub sequence_type: SequenceType,
    pub description: Option<String>,
}

impl Foreign for BiologicalSequence {
    fn type_name(&self) -> &'static str {
        "BiologicalSequence"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "sequence" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::String(self.sequence.clone()))
            }
            "sequenceType" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::String(self.sequence_type.to_string()))
            }
            "length" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.sequence.len() as i64))
            }
            "description" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                match &self.description {
                    Some(desc) => Ok(Value::String(desc.clone())),
                    None => Ok(Value::Missing),
                }
            }
            "gcContent" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                match self.sequence_type {
                    SequenceType::DNA | SequenceType::RNA => {
                        let gc_content = calculate_gc_content(&self.sequence);
                        Ok(Value::Real(gc_content))
                    }
                    SequenceType::Protein => Err(ForeignError::RuntimeError {
                        message: "GC content not applicable to protein sequences".to_string(),
                    }),
                }
            }
            "reverseComplement" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                match self.sequence_type {
                    SequenceType::DNA => {
                        let rev_comp = reverse_complement_dna(&self.sequence);
                        Ok(Value::String(rev_comp))
                    }
                    _ => Err(ForeignError::RuntimeError {
                        message: "Reverse complement only applicable to DNA sequences".to_string(),
                    }),
                }
            }
            "transcribe" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                match self.sequence_type {
                    SequenceType::DNA => {
                        let rna = transcribe_dna(&self.sequence);
                        Ok(Value::String(rna))
                    }
                    _ => Err(ForeignError::RuntimeError {
                        message: "Transcription only applicable to DNA sequences".to_string(),
                    }),
                }
            }
            "translate" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                match self.sequence_type {
                    SequenceType::DNA => {
                        let protein = translate_dna(&self.sequence);
                        Ok(Value::String(protein))
                    }
                    SequenceType::RNA => {
                        let protein = translate_rna(&self.sequence);
                        Ok(Value::String(protein))
                    }
                    SequenceType::Protein => Err(ForeignError::RuntimeError {
                        message: "Cannot translate protein sequences".to_string(),
                    }),
                }
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

/// Foreign object representing a genomic feature (ORF, motif, etc.)
#[derive(Debug, Clone, PartialEq)]
pub struct GenomicFeature {
    pub feature_type: String,
    pub start: usize,
    pub end: usize,
    pub sequence: String,
    pub frame: Option<i32>,
    pub strand: Option<String>,
    pub score: Option<f64>,
}

impl Foreign for GenomicFeature {
    fn type_name(&self) -> &'static str {
        "GenomicFeature"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "featureType" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::String(self.feature_type.clone()))
            }
            "start" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.start as i64))
            }
            "end" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.end as i64))
            }
            "sequence" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::String(self.sequence.clone()))
            }
            "length" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer((self.end - self.start) as i64))
            }
            "frame" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                match self.frame {
                    Some(frame) => Ok(Value::Integer(frame as i64)),
                    None => Ok(Value::Missing),
                }
            }
            "strand" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                match &self.strand {
                    Some(strand) => Ok(Value::String(strand.clone())),
                    None => Ok(Value::Missing),
                }
            }
            "score" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                match self.score {
                    Some(score) => Ok(Value::Real(score)),
                    None => Ok(Value::Missing),
                }
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

/// Foreign object representing codon usage statistics
#[derive(Debug, Clone, PartialEq)]
pub struct CodonUsageTable {
    pub codon_counts: HashMap<String, i32>,
    pub total_codons: i32,
    pub amino_acid_counts: HashMap<char, i32>,
}

impl Foreign for CodonUsageTable {
    fn type_name(&self) -> &'static str {
        "CodonUsageTable"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "totalCodons" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.total_codons as i64))
            }
            "mostFrequent" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let most_frequent = self.codon_counts
                    .iter()
                    .max_by_key(|(_, &count)| count)
                    .map(|(codon, _)| codon.clone())
                    .unwrap_or_else(|| "".to_string());
                Ok(Value::String(most_frequent))
            }
            "getCodonCount" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let codon = match &args[0] {
                    Value::String(s) => s.clone(),
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "String".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };
                
                let count = self.codon_counts.get(&codon).unwrap_or(&0);
                Ok(Value::Integer(*count as i64))
            }
            "getCodonFrequency" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let codon = match &args[0] {
                    Value::String(s) => s.clone(),
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "String".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };
                
                let count = self.codon_counts.get(&codon).unwrap_or(&0);
                let frequency = if self.total_codons > 0 {
                    *count as f64 / self.total_codons as f64
                } else {
                    0.0
                };
                Ok(Value::Real(frequency))
            }
            "allCodons" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let codon_list: Vec<Value> = self.codon_counts
                    .keys()
                    .map(|codon| Value::String(codon.clone()))
                    .collect();
                Ok(Value::List(codon_list))
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

/// Genetic code table for translation
fn get_genetic_code() -> HashMap<String, char> {
    let mut code = HashMap::new();
    
    // Standard genetic code
    code.insert("TTT".to_string(), 'F'); code.insert("TTC".to_string(), 'F');
    code.insert("TTA".to_string(), 'L'); code.insert("TTG".to_string(), 'L');
    code.insert("TCT".to_string(), 'S'); code.insert("TCC".to_string(), 'S');
    code.insert("TCA".to_string(), 'S'); code.insert("TCG".to_string(), 'S');
    code.insert("TAT".to_string(), 'Y'); code.insert("TAC".to_string(), 'Y');
    code.insert("TAA".to_string(), '*'); code.insert("TAG".to_string(), '*');
    code.insert("TGT".to_string(), 'C'); code.insert("TGC".to_string(), 'C');
    code.insert("TGA".to_string(), '*'); code.insert("TGG".to_string(), 'W');
    
    code.insert("CTT".to_string(), 'L'); code.insert("CTC".to_string(), 'L');
    code.insert("CTA".to_string(), 'L'); code.insert("CTG".to_string(), 'L');
    code.insert("CCT".to_string(), 'P'); code.insert("CCC".to_string(), 'P');
    code.insert("CCA".to_string(), 'P'); code.insert("CCG".to_string(), 'P');
    code.insert("CAT".to_string(), 'H'); code.insert("CAC".to_string(), 'H');
    code.insert("CAA".to_string(), 'Q'); code.insert("CAG".to_string(), 'Q');
    code.insert("CGT".to_string(), 'R'); code.insert("CGC".to_string(), 'R');
    code.insert("CGA".to_string(), 'R'); code.insert("CGG".to_string(), 'R');
    
    code.insert("ATT".to_string(), 'I'); code.insert("ATC".to_string(), 'I');
    code.insert("ATA".to_string(), 'I'); code.insert("ATG".to_string(), 'M');
    code.insert("ACT".to_string(), 'T'); code.insert("ACC".to_string(), 'T');
    code.insert("ACA".to_string(), 'T'); code.insert("ACG".to_string(), 'T');
    code.insert("AAT".to_string(), 'N'); code.insert("AAC".to_string(), 'N');
    code.insert("AAA".to_string(), 'K'); code.insert("AAG".to_string(), 'K');
    code.insert("AGT".to_string(), 'S'); code.insert("AGC".to_string(), 'S');
    code.insert("AGA".to_string(), 'R'); code.insert("AGG".to_string(), 'R');
    
    code.insert("GTT".to_string(), 'V'); code.insert("GTC".to_string(), 'V');
    code.insert("GTA".to_string(), 'V'); code.insert("GTG".to_string(), 'V');
    code.insert("GCT".to_string(), 'A'); code.insert("GCC".to_string(), 'A');
    code.insert("GCA".to_string(), 'A'); code.insert("GCG".to_string(), 'A');
    code.insert("GAT".to_string(), 'D'); code.insert("GAC".to_string(), 'D');
    code.insert("GAA".to_string(), 'E'); code.insert("GAG".to_string(), 'E');
    code.insert("GGT".to_string(), 'G'); code.insert("GGC".to_string(), 'G');
    code.insert("GGA".to_string(), 'G'); code.insert("GGG".to_string(), 'G');
    
    code
}

/// Validate sequence contains only valid characters for its type
fn validate_sequence_characters(sequence: &str, seq_type: &SequenceType) -> Result<(), String> {
    let valid_chars = seq_type.valid_characters();
    let sequence_upper = sequence.to_uppercase();
    
    for ch in sequence_upper.chars() {
        if !valid_chars.contains(ch) {
            return Err(format!(
                "Invalid character '{}' for {} sequence. Valid characters: {}",
                ch, seq_type.to_string(), valid_chars
            ));
        }
    }
    
    Ok(())
}

/// Calculate GC content of a DNA or RNA sequence
fn calculate_gc_content(sequence: &str) -> f64 {
    if sequence.is_empty() {
        return 0.0;
    }
    
    let sequence_upper = sequence.to_uppercase();
    let gc_count = sequence_upper.chars()
        .filter(|&ch| ch == 'G' || ch == 'C')
        .count();
    
    gc_count as f64 / sequence.len() as f64
}

/// Generate reverse complement of a DNA sequence
fn reverse_complement_dna(dna: &str) -> String {
    let complement_map = [
        ('A', 'T'), ('T', 'A'), ('C', 'G'), ('G', 'C'),
        ('a', 't'), ('t', 'a'), ('c', 'g'), ('g', 'c'),
        ('N', 'N'), ('n', 'n'),
    ].iter().cloned().collect::<HashMap<char, char>>();
    
    dna.chars()
        .rev()
        .map(|ch| *complement_map.get(&ch).unwrap_or(&ch))
        .collect()
}

/// Transcribe DNA to RNA (T -> U)
fn transcribe_dna(dna: &str) -> String {
    dna.replace('T', "U").replace('t', "u")
}

/// Translate DNA to protein sequence
fn translate_dna(dna: &str) -> String {
    let rna = transcribe_dna(dna);
    translate_rna(&rna)
}

/// Translate RNA to protein sequence
fn translate_rna(rna: &str) -> String {
    let genetic_code = get_genetic_code();
    let rna_upper = rna.to_uppercase().replace('U', "T"); // Convert back to T for lookup
    let mut protein = String::new();
    
    let chars: Vec<char> = rna_upper.chars().collect();
    for i in (0..chars.len()).step_by(3) {
        if i + 2 < chars.len() {
            let codon: String = chars[i..i+3].iter().collect();
            if let Some(&amino_acid) = genetic_code.get(&codon) {
                protein.push(amino_acid);
            } else {
                protein.push('X'); // Unknown amino acid
            }
        }
    }
    
    protein
}

/// Find Open Reading Frames (ORFs) in a DNA sequence
fn find_orfs_impl(dna: &str, min_length: usize) -> Vec<GenomicFeature> {
    let mut orfs = Vec::new();
    let genetic_code = get_genetic_code();
    
    // Search in all 6 reading frames (3 forward, 3 reverse)
    for frame in 0..3 {
        // Forward strand
        orfs.extend(find_orfs_in_frame(dna, frame, "+", &genetic_code, min_length));
        
        // Reverse strand
        let rev_comp = reverse_complement_dna(dna);
        orfs.extend(find_orfs_in_frame(&rev_comp, frame, "-", &genetic_code, min_length));
    }
    
    // Sort ORFs by start position
    orfs.sort_by_key(|orf| orf.start);
    
    orfs
}

/// Find ORFs in a specific reading frame
fn find_orfs_in_frame(
    sequence: &str,
    frame: usize,
    strand: &str,
    genetic_code: &HashMap<String, char>,
    min_length: usize,
) -> Vec<GenomicFeature> {
    let mut orfs = Vec::new();
    let sequence_upper = sequence.to_uppercase().replace('U', "T");
    let chars: Vec<char> = sequence_upper.chars().collect();
    
    let mut in_orf = false;
    let mut orf_start = 0;
    
    for i in (frame..chars.len()).step_by(3) {
        if i + 2 >= chars.len() {
            break;
        }
        
        let codon: String = chars[i..i+3].iter().collect();
        
        if let Some(&amino_acid) = genetic_code.get(&codon) {
            if amino_acid == 'M' && !in_orf {
                // Start codon found
                in_orf = true;
                orf_start = i;
            } else if amino_acid == '*' && in_orf {
                // Stop codon found
                let orf_end = i + 3;
                let orf_length = orf_end - orf_start;
                
                if orf_length >= min_length {
                    let orf_sequence: String = chars[orf_start..orf_end].iter().collect();
                    orfs.push(GenomicFeature {
                        feature_type: "ORF".to_string(),
                        start: orf_start,
                        end: orf_end,
                        sequence: orf_sequence,
                        frame: Some(frame as i32 + 1),
                        strand: Some(strand.to_string()),
                        score: Some(orf_length as f64),
                    });
                }
                
                in_orf = false;
            }
        }
    }
    
    orfs
}

/// Find motifs (simple substring search) in a sequence
fn find_motifs_impl(sequence: &str, motif: &str) -> Vec<usize> {
    let mut positions = Vec::new();
    let sequence_upper = sequence.to_uppercase();
    let motif_upper = motif.to_uppercase();
    
    if motif.is_empty() {
        return positions;
    }
    
    let sequence_chars: Vec<char> = sequence_upper.chars().collect();
    let motif_chars: Vec<char> = motif_upper.chars().collect();
    
    for i in 0..=sequence_chars.len().saturating_sub(motif_chars.len()) {
        if sequence_chars[i..i + motif_chars.len()] == motif_chars {
            positions.push(i);
        }
    }
    
    positions
}

/// Calculate codon usage statistics
fn calculate_codon_usage(dna: &str) -> CodonUsageTable {
    let mut codon_counts = HashMap::new();
    let mut amino_acid_counts = HashMap::new();
    let genetic_code = get_genetic_code();
    
    let sequence_upper = dna.to_uppercase().replace('U', "T");
    let chars: Vec<char> = sequence_upper.chars().collect();
    let mut total_codons = 0;
    
    for i in (0..chars.len()).step_by(3) {
        if i + 2 < chars.len() {
            let codon: String = chars[i..i+3].iter().collect();
            
            if genetic_code.contains_key(&codon) {
                *codon_counts.entry(codon.clone()).or_insert(0) += 1;
                total_codons += 1;
                
                if let Some(&amino_acid) = genetic_code.get(&codon) {
                    *amino_acid_counts.entry(amino_acid).or_insert(0) += 1;
                }
            }
        }
    }
    
    CodonUsageTable {
        codon_counts,
        total_codons,
        amino_acid_counts,
    }
}

// Public API functions for the Lyra stdlib

/// BiologicalSequence[sequence, type] -> BiologicalSequence
pub fn biological_sequence(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!("Invalid number of arguments: expected 2, got {}", args.len())));
    }
    
    let sequence = crate::stdlib::bioinformatics::validate_sequence_string(&args[0])?;
    let type_str = crate::stdlib::bioinformatics::validate_sequence_string(&args[1])?;
    
    if sequence.is_empty() {
        return Err(VmError::Runtime("Sequence cannot be empty".to_string()));
    }
    
    let sequence_type = SequenceType::from_string(&type_str)
        .map_err(|e| VmError::Runtime(e))?;
    
    // Validate sequence characters
    validate_sequence_characters(&sequence, &sequence_type)
        .map_err(|e| VmError::Runtime(e))?;
    
    let bio_seq = BiologicalSequence {
        sequence: sequence.to_uppercase(),
        sequence_type,
        description: None,
    };
    
    Ok(Value::LyObj(LyObj::new(Box::new(bio_seq))))
}

/// ReverseComplement[dna_sequence] -> String
pub fn reverse_complement(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!("Invalid number of arguments: expected 1, got {}", args.len())));
    }
    
    let sequence = crate::stdlib::bioinformatics::validate_sequence_string(&args[0])?;
    
    if sequence.is_empty() {
        return Err(VmError::Runtime("Sequence cannot be empty".to_string()));
    }
    
    // Validate it's a DNA sequence
    validate_sequence_characters(&sequence, &SequenceType::DNA)
        .map_err(|e| VmError::Runtime(e))?;
    
    let rev_comp = reverse_complement_dna(&sequence);
    Ok(Value::String(rev_comp))
}

/// Translate[sequence] -> String
pub fn translate(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!("Invalid number of arguments: expected 1, got {}", args.len())));
    }
    
    let sequence = crate::stdlib::bioinformatics::validate_sequence_string(&args[0])?;
    
    if sequence.is_empty() {
        return Err(VmError::Runtime("Sequence cannot be empty".to_string()));
    }
    
    if sequence.len() % 3 != 0 {
        return Err(VmError::Runtime("Sequence length must be divisible by 3 for translation".to_string()));
    }
    
    // Try to determine if it's DNA or RNA based on presence of U/T
    let protein = if sequence.to_uppercase().contains('U') {
        // Assume RNA
        validate_sequence_characters(&sequence, &SequenceType::RNA)
            .map_err(|e| VmError::Runtime(e))?;
        translate_rna(&sequence)
    } else {
        // Assume DNA
        validate_sequence_characters(&sequence, &SequenceType::DNA)
            .map_err(|e| VmError::Runtime(e))?;
        translate_dna(&sequence)
    };
    
    Ok(Value::String(protein))
}

/// Transcribe[dna_sequence] -> String
pub fn transcribe(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!("Invalid number of arguments: expected 1, got {}", args.len())));
    }
    
    let sequence = crate::stdlib::bioinformatics::validate_sequence_string(&args[0])?;
    
    if sequence.is_empty() {
        return Err(VmError::Runtime("Sequence cannot be empty".to_string()));
    }
    
    // Validate it's a DNA sequence
    validate_sequence_characters(&sequence, &SequenceType::DNA)
        .map_err(|e| VmError::Runtime(e))?;
    
    let rna = transcribe_dna(&sequence);
    Ok(Value::String(rna))
}

/// GCContent[sequence] -> Real
pub fn gc_content(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!("Invalid number of arguments: expected 1, got {}", args.len())));
    }
    
    let sequence = crate::stdlib::bioinformatics::validate_sequence_string(&args[0])?;
    
    if sequence.is_empty() {
        return Err(VmError::Runtime("Sequence cannot be empty".to_string()));
    }
    
    let gc_content = calculate_gc_content(&sequence);
    Ok(Value::Real(gc_content))
}

/// FindORFs[dna_sequence] or FindORFs[dna_sequence, min_length]
pub fn find_orfs(args: &[Value]) -> VmResult<Value> {
    let (sequence, min_length) = match args.len() {
        1 => {
            let sequence = crate::stdlib::bioinformatics::validate_sequence_string(&args[0])?;
            (sequence, 60) // Default minimum length of 60 nucleotides (20 amino acids)
        }
        2 => {
            let sequence = crate::stdlib::bioinformatics::validate_sequence_string(&args[0])?;
            let min_length = crate::stdlib::bioinformatics::validate_integer(&args[1])? as usize;
            (sequence, min_length)
        }
        _ => {
            return Err(VmError::Runtime(format!("Invalid number of arguments: expected 1 or 2, got {}", args.len())));
        }
    };
    
    if sequence.is_empty() {
        return Err(VmError::Runtime("Sequence cannot be empty".to_string()));
    }
    
    if min_length == 0 {
        return Err(VmError::Runtime("Minimum length must be greater than 0".to_string()));
    }
    
    // Validate it's a DNA sequence
    validate_sequence_characters(&sequence, &SequenceType::DNA)
        .map_err(|e| VmError::Runtime(e))?;
    
    let orfs = find_orfs_impl(&sequence, min_length);
    let orf_values: Vec<Value> = orfs.into_iter()
        .map(|orf| Value::LyObj(LyObj::new(Box::new(orf))))
        .collect();
    
    Ok(Value::List(orf_values))
}

/// FindMotifs[sequence, motif] -> List[Integer]
pub fn find_motifs(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!("Invalid number of arguments: expected 2, got {}", args.len())));
    }
    
    let sequence = crate::stdlib::bioinformatics::validate_sequence_string(&args[0])?;
    let motif = crate::stdlib::bioinformatics::validate_sequence_string(&args[1])?;
    
    if sequence.is_empty() {
        return Err(VmError::Runtime("Sequence cannot be empty".to_string()));
    }
    
    if motif.is_empty() {
        return Err(VmError::Runtime("Motif cannot be empty".to_string()));
    }
    
    let positions = find_motifs_impl(&sequence, &motif);
    let position_values: Vec<Value> = positions.into_iter()
        .map(|pos| Value::Integer(pos as i64))
        .collect();
    
    Ok(Value::List(position_values))
}

/// CodonUsage[dna_sequence] -> CodonUsageTable
pub fn codon_usage(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!("Invalid number of arguments: expected 1, got {}", args.len())));
    }
    
    let sequence = crate::stdlib::bioinformatics::validate_sequence_string(&args[0])?;
    
    if sequence.is_empty() {
        return Err(VmError::Runtime("Sequence cannot be empty".to_string()));
    }
    
    // Validate it's a DNA sequence
    validate_sequence_characters(&sequence, &SequenceType::DNA)
        .map_err(|e| VmError::Runtime(e))?;
    
    let codon_usage_table = calculate_codon_usage(&sequence);
    Ok(Value::LyObj(LyObj::new(Box::new(codon_usage_table))))
}