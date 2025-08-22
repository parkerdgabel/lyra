//! Comprehensive tests for bioinformatics algorithms
//! 
//! Following TDD approach - these tests define the expected behavior
//! before implementation (RED phase)

use lyra::vm::{Value, VmResult, Vm};
use lyra::stdlib::StandardLibrary;

/// Test sequence alignment algorithms
#[cfg(test)]
mod alignment_tests {
    use super::*;

    #[test]
    fn test_global_alignment_basic() {
        let mut vm = Vm::new();
        let stdlib = StandardLibrary::new();
        
        // GlobalAlignment[seq1, seq2] -> AlignmentResult
        let result = stdlib.get_function("GlobalAlignment").unwrap()(&[
            Value::String("ACGT".to_string()),
            Value::String("ACT".to_string()),
        ]);
        
        // Should succeed and return AlignmentResult Foreign object
        assert!(result.is_ok());
        let alignment = result.unwrap();
        
        // Should be LyObj containing AlignmentResult
        if let Value::LyObj(obj) = &alignment {
            assert_eq!(obj.type_name(), "AlignmentResult");
            
            // Test method calls on alignment
            let score = obj.call_method("score", &[]).unwrap();
            assert!(matches!(score, Value::Real(_) | Value::Integer(_)));
            
            let aligned_seq1 = obj.call_method("alignedSequence1", &[]).unwrap();
            assert!(matches!(aligned_seq1, Value::String(_)));
            
            let aligned_seq2 = obj.call_method("alignedSequence2", &[]).unwrap();
            assert!(matches!(aligned_seq2, Value::String(_)));
        } else {
            panic!("Expected LyObj, got {:?}", alignment);
        }
    }

    #[test]
    fn test_global_alignment_with_scoring_matrix() {
        let stdlib = StandardLibrary::new();
        
        // GlobalAlignment[seq1, seq2, match, mismatch, gap]
        let result = stdlib.get_function("GlobalAlignment").unwrap()(&[
            Value::String("ACGT".to_string()),
            Value::String("ACT".to_string()),
            Value::Integer(2),  // match score
            Value::Integer(-1), // mismatch score
            Value::Integer(-2), // gap penalty
        ]);
        
        assert!(result.is_ok());
        let alignment = result.unwrap();
        
        if let Value::LyObj(obj) = &alignment {
            let score = obj.call_method("score", &[]).unwrap();
            // With custom scoring: A-A(+2), C-C(+2), G-T(-1), T-gap(-2) = +1
            if let Value::Real(score_val) = score {
                assert!((score_val - 1.0).abs() < 1e-10);
            } else if let Value::Integer(score_val) = score {
                assert_eq!(score_val, 1);
            }
        }
    }

    #[test]
    fn test_local_alignment_smith_waterman() {
        let stdlib = StandardLibrary::new();
        
        // LocalAlignment[seq1, seq2] -> AlignmentResult
        let result = stdlib.get_function("LocalAlignment").unwrap()(&[
            Value::String("AAAGACGT".to_string()),
            Value::String("GACGT".to_string()),
        ]);
        
        assert!(result.is_ok());
        let alignment = result.unwrap();
        
        if let Value::LyObj(obj) = &alignment {
            assert_eq!(obj.type_name(), "AlignmentResult");
            
            let score = obj.call_method("score", &[]).unwrap();
            assert!(matches!(score, Value::Real(_) | Value::Integer(_)));
            
            // Local alignment should find best matching subsequence
            let start1 = obj.call_method("start1", &[]).unwrap();
            let start2 = obj.call_method("start2", &[]).unwrap();
            assert!(matches!(start1, Value::Integer(_)));
            assert!(matches!(start2, Value::Integer(_)));
        }
    }

    #[test]
    fn test_multiple_sequence_alignment() {
        let stdlib = StandardLibrary::new();
        
        // MultipleAlignment[{seq1, seq2, seq3}] -> MultipleAlignmentResult
        let sequences = vec![
            Value::String("ACGT".to_string()),
            Value::String("ACT".to_string()),
            Value::String("AGGT".to_string()),
        ];
        
        let result = stdlib.get_function("MultipleAlignment").unwrap()(&[
            Value::List(sequences),
        ]);
        
        assert!(result.is_ok());
        let alignment = result.unwrap();
        
        if let Value::LyObj(obj) = &alignment {
            assert_eq!(obj.type_name(), "MultipleAlignmentResult");
            
            let aligned_sequences = obj.call_method("alignedSequences", &[]).unwrap();
            if let Value::List(seqs) = aligned_sequences {
                assert_eq!(seqs.len(), 3);
                for seq in seqs {
                    assert!(matches!(seq, Value::String(_)));
                }
            }
        }
    }

    #[test]
    fn test_blast_search() {
        let stdlib = StandardLibrary::new();
        
        // BlastSearch[query, database] -> List[BlastHit]
        let database = vec![
            Value::String("ACGTACGTACGT".to_string()),
            Value::String("TTTTACGTAAAA".to_string()),
            Value::String("GGGGCCCCTTTT".to_string()),
        ];
        
        let result = stdlib.get_function("BlastSearch").unwrap()(&[
            Value::String("ACGT".to_string()),
            Value::List(database),
        ]);
        
        assert!(result.is_ok());
        let hits = result.unwrap();
        
        if let Value::List(hit_list) = hits {
            assert!(hit_list.len() >= 1); // Should find matches
            
            for hit in hit_list {
                if let Value::LyObj(obj) = hit {
                    assert_eq!(obj.type_name(), "BlastHit");
                    
                    let score = obj.call_method("score", &[]).unwrap();
                    let evalue = obj.call_method("eValue", &[]).unwrap();
                    let position = obj.call_method("position", &[]).unwrap();
                    
                    assert!(matches!(score, Value::Real(_) | Value::Integer(_)));
                    assert!(matches!(evalue, Value::Real(_)));
                    assert!(matches!(position, Value::Integer(_)));
                }
            }
        }
    }

    #[test]
    fn test_alignment_error_cases() {
        let stdlib = StandardLibrary::new();
        
        // Empty sequences should return error
        let result = stdlib.get_function("GlobalAlignment").unwrap()(&[
            Value::String("".to_string()),
            Value::String("ACT".to_string()),
        ]);
        assert!(result.is_err());
        
        // Wrong argument types should return error
        let result = stdlib.get_function("GlobalAlignment").unwrap()(&[
            Value::Integer(123),
            Value::String("ACT".to_string()),
        ]);
        assert!(result.is_err());
        
        // Wrong number of arguments should return error
        let result = stdlib.get_function("GlobalAlignment").unwrap()(&[
            Value::String("ACGT".to_string()),
        ]);
        assert!(result.is_err());
    }
}

/// Test phylogenetic analysis functions
#[cfg(test)]
mod phylogenetics_tests {
    use super::*;

    #[test]
    fn test_neighbor_joining_tree() {
        let stdlib = StandardLibrary::new();
        
        // NeighborJoining[sequences] -> PhylogeneticTree
        let sequences = vec![
            Value::String("ACGT".to_string()),
            Value::String("ACCT".to_string()),
            Value::String("AAGT".to_string()),
            Value::String("TTGT".to_string()),
        ];
        
        let result = stdlib.get_function("NeighborJoining").unwrap()(&[
            Value::List(sequences),
        ]);
        
        assert!(result.is_ok());
        let tree = result.unwrap();
        
        if let Value::LyObj(obj) = &tree {
            assert_eq!(obj.type_name(), "PhylogeneticTree");
            
            let newick = obj.call_method("newick", &[]).unwrap();
            assert!(matches!(newick, Value::String(_)));
            
            let leaves = obj.call_method("leaves", &[]).unwrap();
            if let Value::List(leaf_list) = leaves {
                assert_eq!(leaf_list.len(), 4); // 4 input sequences
            }
            
            let distance_matrix = obj.call_method("distanceMatrix", &[]).unwrap();
            assert!(matches!(distance_matrix, Value::LyObj(_)));
        }
    }

    #[test]
    fn test_maximum_likelihood_tree() {
        let stdlib = StandardLibrary::new();
        
        // MaximumLikelihood[sequences] -> PhylogeneticTree
        let sequences = vec![
            Value::String("ACGTACGT".to_string()),
            Value::String("ACCTACCT".to_string()),
            Value::String("AAGTAAGT".to_string()),
        ];
        
        let result = stdlib.get_function("MaximumLikelihood").unwrap()(&[
            Value::List(sequences),
        ]);
        
        assert!(result.is_ok());
        let tree = result.unwrap();
        
        if let Value::LyObj(obj) = &tree {
            assert_eq!(obj.type_name(), "PhylogeneticTree");
            
            let likelihood = obj.call_method("likelihood", &[]).unwrap();
            assert!(matches!(likelihood, Value::Real(_)));
            
            let branch_lengths = obj.call_method("branchLengths", &[]).unwrap();
            assert!(matches!(branch_lengths, Value::List(_)));
        }
    }

    #[test]
    fn test_pairwise_distance_calculation() {
        let stdlib = StandardLibrary::new();
        
        // PairwiseDistance[seq1, seq2] -> Real
        let result = stdlib.get_function("PairwiseDistance").unwrap()(&[
            Value::String("ACGT".to_string()),
            Value::String("ACCT".to_string()),
        ]);
        
        assert!(result.is_ok());
        let distance = result.unwrap();
        
        // Should return a Real number representing evolutionary distance
        assert!(matches!(distance, Value::Real(_)));
        if let Value::Real(dist_val) = distance {
            assert!(dist_val >= 0.0); // Distance should be non-negative
            assert!(dist_val <= 1.0); // Normalized distance should be <= 1
        }
    }

    #[test]
    fn test_phylogenetic_tree_construction() {
        let stdlib = StandardLibrary::new();
        
        // PhylogeneticTree[newick_string] -> PhylogeneticTree
        let newick = "(A:0.1,B:0.2,(C:0.3,D:0.4):0.5);";
        
        let result = stdlib.get_function("PhylogeneticTree").unwrap()(&[
            Value::String(newick.to_string()),
        ]);
        
        assert!(result.is_ok());
        let tree = result.unwrap();
        
        if let Value::LyObj(obj) = &tree {
            assert_eq!(obj.type_name(), "PhylogeneticTree");
            
            let is_rooted = obj.call_method("isRooted", &[]).unwrap();
            assert!(matches!(is_rooted, Value::Boolean(_)));
            
            let num_leaves = obj.call_method("numLeaves", &[]).unwrap();
            if let Value::Integer(n) = num_leaves {
                assert_eq!(n, 4); // A, B, C, D
            }
        }
    }
}

/// Test genomics analysis functions
#[cfg(test)]
mod genomics_tests {
    use super::*;

    #[test]
    fn test_biological_sequence_creation() {
        let stdlib = StandardLibrary::new();
        
        // BiologicalSequence[sequence, type] -> BiologicalSequence
        let result = stdlib.get_function("BiologicalSequence").unwrap()(&[
            Value::String("ACGTACGT".to_string()),
            Value::String("DNA".to_string()),
        ]);
        
        assert!(result.is_ok());
        let seq = result.unwrap();
        
        if let Value::LyObj(obj) = &seq {
            assert_eq!(obj.type_name(), "BiologicalSequence");
            
            let sequence = obj.call_method("sequence", &[]).unwrap();
            if let Value::String(s) = sequence {
                assert_eq!(s, "ACGTACGT");
            }
            
            let seq_type = obj.call_method("sequenceType", &[]).unwrap();
            if let Value::String(t) = seq_type {
                assert_eq!(t, "DNA");
            }
            
            let length = obj.call_method("length", &[]).unwrap();
            if let Value::Integer(len) = length {
                assert_eq!(len, 8);
            }
        }
    }

    #[test]
    fn test_reverse_complement() {
        let stdlib = StandardLibrary::new();
        
        // ReverseComplement[dna_sequence] -> String
        let result = stdlib.get_function("ReverseComplement").unwrap()(&[
            Value::String("ACGTACGT".to_string()),
        ]);
        
        assert!(result.is_ok());
        let rev_comp = result.unwrap();
        
        if let Value::String(s) = rev_comp {
            assert_eq!(s, "ACGTACGT"); // Palindrome in this case
        }
        
        // Test non-palindromic sequence
        let result = stdlib.get_function("ReverseComplement").unwrap()(&[
            Value::String("ATCG".to_string()),
        ]);
        
        if let Value::String(s) = result.unwrap() {
            assert_eq!(s, "CGAT"); // Reverse of TACG (complement of ATCG)
        }
    }

    #[test]
    fn test_translation() {
        let stdlib = StandardLibrary::new();
        
        // Translate[dna_sequence] -> String (protein)
        let result = stdlib.get_function("Translate").unwrap()(&[
            Value::String("ATGAAATAA".to_string()), // ATG(M) AAA(K) TAA(*)
        ]);
        
        assert!(result.is_ok());
        let protein = result.unwrap();
        
        if let Value::String(p) = protein {
            assert!(p.starts_with("MK")); // Met-Lys-Stop
        }
    }

    #[test]
    fn test_transcription() {
        let stdlib = StandardLibrary::new();
        
        // Transcribe[dna_sequence] -> String (RNA)
        let result = stdlib.get_function("Transcribe").unwrap()(&[
            Value::String("ATCG".to_string()),
        ]);
        
        assert!(result.is_ok());
        let rna = result.unwrap();
        
        if let Value::String(r) = rna {
            assert_eq!(r, "AUCG"); // T -> U
        }
    }

    #[test]
    fn test_gc_content() {
        let stdlib = StandardLibrary::new();
        
        // GCContent[sequence] -> Real
        let result = stdlib.get_function("GCContent").unwrap()(&[
            Value::String("GCGC".to_string()), // 100% GC
        ]);
        
        assert!(result.is_ok());
        let gc_content = result.unwrap();
        
        if let Value::Real(gc) = gc_content {
            assert!((gc - 1.0).abs() < 1e-10); // Should be 1.0 (100%)
        }
        
        // Test 50% GC content
        let result = stdlib.get_function("GCContent").unwrap()(&[
            Value::String("ATGC".to_string()), // 50% GC
        ]);
        
        if let Value::Real(gc) = result.unwrap() {
            assert!((gc - 0.5).abs() < 1e-10); // Should be 0.5 (50%)
        }
    }

    #[test]
    fn test_find_orfs() {
        let stdlib = StandardLibrary::new();
        
        // FindORFs[dna_sequence] -> List[ORF]
        let sequence = "ATGAAATAAATGCCCTAGATGGGGTAA"; // Multiple ORFs
        let result = stdlib.get_function("FindORFs").unwrap()(&[
            Value::String(sequence.to_string()),
        ]);
        
        assert!(result.is_ok());
        let orfs = result.unwrap();
        
        if let Value::List(orf_list) = orfs {
            assert!(orf_list.len() >= 1); // Should find at least one ORF
            
            for orf in orf_list {
                if let Value::LyObj(obj) = orf {
                    assert_eq!(obj.type_name(), "GenomicFeature");
                    
                    let start = obj.call_method("start", &[]).unwrap();
                    let end = obj.call_method("end", &[]).unwrap();
                    let frame = obj.call_method("frame", &[]).unwrap();
                    let sequence = obj.call_method("sequence", &[]).unwrap();
                    
                    assert!(matches!(start, Value::Integer(_)));
                    assert!(matches!(end, Value::Integer(_)));
                    assert!(matches!(frame, Value::Integer(_)));
                    assert!(matches!(sequence, Value::String(_)));
                }
            }
        }
    }

    #[test]
    fn test_find_motifs() {
        let stdlib = StandardLibrary::new();
        
        // FindMotifs[sequence, motif] -> List[Integer] (positions)
        let result = stdlib.get_function("FindMotifs").unwrap()(&[
            Value::String("ACGTACGTACGT".to_string()),
            Value::String("ACGT".to_string()),
        ]);
        
        assert!(result.is_ok());
        let positions = result.unwrap();
        
        if let Value::List(pos_list) = positions {
            assert!(pos_list.len() >= 2); // Should find multiple matches
            
            for pos in pos_list {
                if let Value::Integer(p) = pos {
                    assert!(p >= 0);
                    assert!(p < 12); // Within sequence bounds
                }
            }
        }
    }

    #[test]
    fn test_codon_usage() {
        let stdlib = StandardLibrary::new();
        
        // CodonUsage[dna_sequence] -> Association[codon -> count]
        let result = stdlib.get_function("CodonUsage").unwrap()(&[
            Value::String("ATGAAATAAATGCCC".to_string()),
        ]);
        
        assert!(result.is_ok());
        let codon_counts = result.unwrap();
        
        // Should return a structured representation of codon usage
        if let Value::LyObj(obj) = &codon_counts {
            assert_eq!(obj.type_name(), "CodonUsageTable");
            
            let total_codons = obj.call_method("totalCodons", &[]).unwrap();
            assert!(matches!(total_codons, Value::Integer(_)));
            
            let most_frequent = obj.call_method("mostFrequent", &[]).unwrap();
            assert!(matches!(most_frequent, Value::String(_)));
        }
    }

    #[test]
    fn test_genomics_error_cases() {
        let stdlib = StandardLibrary::new();
        
        // Invalid sequence type should error
        let result = stdlib.get_function("BiologicalSequence").unwrap()(&[
            Value::String("ACGT".to_string()),
            Value::String("INVALID".to_string()),
        ]);
        assert!(result.is_err());
        
        // Empty sequence should error
        let result = stdlib.get_function("GCContent").unwrap()(&[
            Value::String("".to_string()),
        ]);
        assert!(result.is_err());
        
        // Invalid nucleotide characters should error
        let result = stdlib.get_function("ReverseComplement").unwrap()(&[
            Value::String("ACGTXYZ".to_string()),
        ]);
        assert!(result.is_err());
    }
}

/// Integration tests combining multiple bioinformatics operations
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_sequence_analysis_pipeline() {
        let stdlib = StandardLibrary::new();
        
        // Create biological sequence
        let dna = stdlib.get_function("BiologicalSequence").unwrap()(&[
            Value::String("ATGAAATAAATGCCCTAGATGGGGTAA".to_string()),
            Value::String("DNA".to_string()),
        ]).unwrap();
        
        // Calculate GC content
        if let Value::LyObj(seq_obj) = &dna {
            let sequence = seq_obj.call_method("sequence", &[]).unwrap();
            let gc_content = stdlib.get_function("GCContent").unwrap()(&[sequence]).unwrap();
            assert!(matches!(gc_content, Value::Real(_)));
        }
        
        // Find ORFs in the sequence
        if let Value::LyObj(seq_obj) = &dna {
            let sequence = seq_obj.call_method("sequence", &[]).unwrap();
            let orfs = stdlib.get_function("FindORFs").unwrap()(&[sequence]).unwrap();
            assert!(matches!(orfs, Value::List(_)));
        }
    }

    #[test]
    fn test_phylogenetic_analysis_pipeline() {
        let stdlib = StandardLibrary::new();
        
        // Create multiple sequences
        let sequences = vec![
            Value::String("ACGTACGTACGT".to_string()),
            Value::String("ACCTACCTACCT".to_string()),
            Value::String("AAGTAAGTAAGT".to_string()),
        ];
        
        // Build neighbor-joining tree
        let tree = stdlib.get_function("NeighborJoining").unwrap()(&[
            Value::List(sequences.clone())
        ]).unwrap();
        
        // Verify tree properties
        if let Value::LyObj(tree_obj) = &tree {
            let newick = tree_obj.call_method("newick", &[]).unwrap();
            assert!(matches!(newick, Value::String(_)));
            
            let num_leaves = tree_obj.call_method("numLeaves", &[]).unwrap();
            if let Value::Integer(n) = num_leaves {
                assert_eq!(n, 3);
            }
        }
    }

    #[test]
    fn test_alignment_and_phylogeny() {
        let stdlib = StandardLibrary::new();
        
        let sequences = vec![
            Value::String("ACGT".to_string()),
            Value::String("ACCT".to_string()),
        ];
        
        // Perform pairwise alignment
        let alignment = stdlib.get_function("GlobalAlignment").unwrap()(&[
            sequences[0].clone(),
            sequences[1].clone(),
        ]).unwrap();
        
        // Calculate evolutionary distance
        let distance = stdlib.get_function("PairwiseDistance").unwrap()(&[
            sequences[0].clone(),
            sequences[1].clone(),
        ]).unwrap();
        
        // Both should succeed
        assert!(matches!(alignment, Value::LyObj(_)));
        assert!(matches!(distance, Value::Real(_)));
    }
}

/// Performance tests for bioinformatics algorithms
#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_alignment_performance() {
        let stdlib = StandardLibrary::new();
        
        // Test with moderately long sequences
        let seq1 = "A".repeat(1000);
        let seq2 = "T".repeat(1000);
        
        let start = Instant::now();
        let result = stdlib.get_function("GlobalAlignment").unwrap()(&[
            Value::String(seq1),
            Value::String(seq2),
        ]);
        let duration = start.elapsed();
        
        assert!(result.is_ok());
        assert!(duration.as_secs() < 5); // Should complete within 5 seconds
    }

    #[test]
    fn test_large_sequence_gc_content() {
        let stdlib = StandardLibrary::new();
        
        // Test GC content calculation on large sequence
        let large_sequence = "ATCG".repeat(10000);
        
        let start = Instant::now();
        let result = stdlib.get_function("GCContent").unwrap()(&[
            Value::String(large_sequence),
        ]);
        let duration = start.elapsed();
        
        assert!(result.is_ok());
        if let Value::Real(gc) = result.unwrap() {
            assert!((gc - 0.5).abs() < 1e-10); // Should be exactly 0.5
        }
        assert!(duration.as_millis() < 100); // Should be fast
    }
}