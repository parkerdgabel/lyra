//! Bio-ML Integration: Bioinformatics-Machine Learning Bridge
//!
//! This module provides seamless integration between bioinformatics data
//! (DNA, RNA, protein sequences) and machine learning pipelines, enabling
//! sequence-based ML workflows for genomics, proteomics, and drug discovery.

use crate::stdlib::ml::{MLResult, MLError};
use crate::stdlib::ml::layers::Tensor;
use crate::stdlib::ml::preprocessing::{MLPreprocessor, AutoPreprocessor};
use crate::stdlib::ml::NetChain;
use crate::stdlib::ml::training::{NetTrain, TrainingConfig, TrainingResult};
use crate::stdlib::bioinformatics::genomics::{BiologicalSequence, SequenceType};
use crate::stdlib::autodiff::Dual;
use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmError, VmResult};
use std::collections::HashMap;
use std::any::Any;

/// Sequence encoding strategies for ML input
#[derive(Debug, Clone, PartialEq)]
pub enum SequenceEncoding {
    /// One-hot encoding: A=[1,0,0,0], C=[0,1,0,0], G=[0,0,1,0], T=[0,0,0,1]
    OneHot,
    /// Ordinal encoding: A=0, C=1, G=2, T=3
    Ordinal,
    /// K-mer frequency vectors
    KMer { k: usize },
    /// Physicochemical properties (for proteins)
    Physicochemical,
    /// BLOSUM/PAM matrices (for proteins)
    SubstitutionMatrix { matrix_type: String },
    /// Word2Vec-style embeddings
    Embedding { dimension: usize },
}

/// Bioinformatics-specific preprocessor for sequence data
#[derive(Debug, Clone)]
pub struct SequencePreprocessor {
    encoding: SequenceEncoding,
    normalize: bool,
    padding_length: Option<usize>,
    sequence_type: Option<SequenceType>,
}

impl SequencePreprocessor {
    /// Create new sequence preprocessor
    pub fn new(encoding: SequenceEncoding) -> Self {
        Self {
            encoding,
            normalize: true,
            padding_length: None,
            sequence_type: None,
        }
    }
    
    /// Set sequence padding length for fixed-size inputs
    pub fn with_padding(mut self, length: usize) -> Self {
        self.padding_length = Some(length);
        self
    }
    
    /// Set expected sequence type for validation
    pub fn with_sequence_type(mut self, seq_type: SequenceType) -> Self {
        self.sequence_type = Some(seq_type);
        self
    }
    
    /// Enable/disable normalization
    pub fn with_normalization(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
    
    /// Encode a single sequence according to the encoding strategy
    pub fn encode_sequence(&self, sequence: &BiologicalSequence) -> MLResult<Tensor> {
        // Validate sequence type if specified
        if let Some(expected_type) = &self.sequence_type {
            if &sequence.sequence_type != expected_type {
                return Err(MLError::DataError {
                    reason: format!("Expected {:?} sequence, got {:?}", 
                                  expected_type, sequence.sequence_type),
                });
            }
        }
        
        match &self.encoding {
            SequenceEncoding::OneHot => self.encode_one_hot(sequence),
            SequenceEncoding::Ordinal => self.encode_ordinal(sequence),
            SequenceEncoding::KMer { k } => self.encode_kmer(sequence, *k),
            SequenceEncoding::Physicochemical => self.encode_physicochemical(sequence),
            SequenceEncoding::SubstitutionMatrix { matrix_type } => {
                self.encode_substitution_matrix(sequence, matrix_type)
            }
            SequenceEncoding::Embedding { dimension } => {
                self.encode_embedding(sequence, *dimension)
            }
        }
    }
    
    /// One-hot encoding: each nucleotide/amino acid becomes a binary vector
    fn encode_one_hot(&self, sequence: &BiologicalSequence) -> MLResult<Tensor> {
        let alphabet = self.get_alphabet(&sequence.sequence_type);
        let seq_len = sequence.sequence.len();
        let alphabet_size = alphabet.len();
        
        // Apply padding if specified
        let target_length = self.padding_length.unwrap_or(seq_len);
        let mut encoded = vec![vec![0.0; alphabet_size]; target_length];
        
        for (i, char) in sequence.sequence.chars().enumerate() {
            if i >= target_length { break; }
            
            if let Some(pos) = alphabet.iter().position(|&c| c == char.to_ascii_uppercase()) {
                encoded[i][pos] = 1.0;
            } else {
                // Unknown character - use uniform distribution or special encoding
                for val in &mut encoded[i] {
                    *val = 1.0 / alphabet_size as f64;
                }
            }
        }
        
        // Convert to tensor
        let data: Vec<f64> = encoded.into_iter().flatten().collect();
        let dual_data: Vec<Dual> = data.into_iter().map(crate::stdlib::autodiff::Dual::variable).collect();
        Tensor::new(dual_data, vec![target_length, alphabet_size])
    }
    
    /// Ordinal encoding: map each character to an integer
    fn encode_ordinal(&self, sequence: &BiologicalSequence) -> MLResult<Tensor> {
        let alphabet = self.get_alphabet(&sequence.sequence_type);
        let seq_len = sequence.sequence.len();
        
        let target_length = self.padding_length.unwrap_or(seq_len);
        let mut encoded = vec![0.0; target_length];
        
        for (i, char) in sequence.sequence.chars().enumerate() {
            if i >= target_length { break; }
            
            let ordinal = alphabet.iter().position(|&c| c == char.to_ascii_uppercase())
                .unwrap_or(0) as f64;
            encoded[i] = ordinal;
        }
        
        // Normalize if requested
        if self.normalize {
            let max_val = alphabet.len() as f64 - 1.0;
            for val in &mut encoded {
                *val /= max_val;
            }
        }
        
        let dual_data: Vec<Dual> = encoded.into_iter().map(crate::stdlib::autodiff::Dual::variable).collect();
        Tensor::new(dual_data, vec![target_length])
    }
    
    /// K-mer frequency encoding: count k-length subsequences
    fn encode_kmer(&self, sequence: &BiologicalSequence, k: usize) -> MLResult<Tensor> {
        if k == 0 || k > 10 {
            return Err(MLError::DataError {
                reason: format!("Invalid k-mer size: {}. Must be between 1 and 10", k),
            });
        }
        
        let alphabet = self.get_alphabet(&sequence.sequence_type);
        let kmer_count = alphabet.len().pow(k as u32);
        let mut frequencies = vec![0.0; kmer_count];
        
        // Generate all possible k-mers
        let kmers = self.generate_kmers(&alphabet, k);
        let kmer_to_index: HashMap<String, usize> = kmers.iter()
            .enumerate()
            .map(|(i, kmer)| (kmer.clone(), i))
            .collect();
        
        // Count k-mers in sequence
        let seq = sequence.sequence.to_uppercase();
        let mut total_kmers = 0;
        
        for i in 0..=seq.len().saturating_sub(k) {
            if let Some(kmer) = seq.get(i..i+k) {
                if let Some(&index) = kmer_to_index.get(kmer) {
                    frequencies[index] += 1.0;
                    total_kmers += 1;
                }
            }
        }
        
        // Normalize to frequencies if requested
        if self.normalize && total_kmers > 0 {
            for freq in &mut frequencies {
                *freq /= total_kmers as f64;
            }
        }
        
        let dual_data: Vec<Dual> = frequencies.into_iter().map(crate::stdlib::autodiff::Dual::variable).collect();
        Tensor::new(dual_data, vec![kmer_count])
    }
    
    /// Physicochemical property encoding for proteins
    fn encode_physicochemical(&self, sequence: &BiologicalSequence) -> MLResult<Tensor> {
        if sequence.sequence_type != SequenceType::Protein {
            return Err(MLError::DataError {
                reason: "Physicochemical encoding only supports protein sequences".to_string(),
            });
        }
        
        // Physicochemical properties: hydrophobicity, molecular weight, charge, etc.
        let properties = self.get_amino_acid_properties();
        let property_count = properties.values().next().map(|v| v.len()).unwrap_or(0);
        
        let seq_len = sequence.sequence.len();
        let target_length = self.padding_length.unwrap_or(seq_len);
        let mut encoded = vec![vec![0.0; property_count]; target_length];
        
        for (i, char) in sequence.sequence.chars().enumerate() {
            if i >= target_length { break; }
            
            if let Some(props) = properties.get(&char.to_ascii_uppercase()) {
                encoded[i] = props.clone();
            }
        }
        
        let data: Vec<f64> = encoded.into_iter().flatten().collect();
        let dual_data: Vec<Dual> = data.into_iter().map(crate::stdlib::autodiff::Dual::variable).collect();
        Tensor::new(dual_data, vec![target_length, property_count])
    }
    
    /// Substitution matrix encoding (BLOSUM, PAM)
    fn encode_substitution_matrix(&self, sequence: &BiologicalSequence, matrix_type: &str) -> MLResult<Tensor> {
        if sequence.sequence_type != SequenceType::Protein {
            return Err(MLError::DataError {
                reason: "Substitution matrix encoding only supports protein sequences".to_string(),
            });
        }
        
        let matrix = self.get_substitution_matrix(matrix_type)?;
        let seq_len = sequence.sequence.len();
        let target_length = self.padding_length.unwrap_or(seq_len);
        let alphabet_size = 20; // Standard amino acids
        
        let mut encoded = vec![vec![0.0; alphabet_size]; target_length];
        
        for (i, char) in sequence.sequence.chars().enumerate() {
            if i >= target_length { break; }
            
            if let Some(scores) = matrix.get(&char.to_ascii_uppercase()) {
                encoded[i] = scores.clone();
            }
        }
        
        let data: Vec<f64> = encoded.into_iter().flatten().collect();
        let dual_data: Vec<Dual> = data.into_iter().map(crate::stdlib::autodiff::Dual::variable).collect();
        Tensor::new(dual_data, vec![target_length, alphabet_size])
    }
    
    /// Embedding encoding (placeholder for learned embeddings)
    fn encode_embedding(&self, sequence: &BiologicalSequence, dimension: usize) -> MLResult<Tensor> {
        // This would use pre-trained embeddings in a real implementation
        let seq_len = sequence.sequence.len();
        let target_length = self.padding_length.unwrap_or(seq_len);
        
        // Generate pseudo-embeddings based on character hash
        let mut encoded = vec![vec![0.0; dimension]; target_length];
        
        for (i, char) in sequence.sequence.chars().enumerate() {
            if i >= target_length { break; }
            
            // Simple hash-based pseudo-embedding
            let hash = char as usize;
            for j in 0..dimension {
                encoded[i][j] = ((hash + j) % 100) as f64 / 100.0;
            }
        }
        
        let data: Vec<f64> = encoded.into_iter().flatten().collect();
        let dual_data: Vec<Dual> = data.into_iter().map(crate::stdlib::autodiff::Dual::variable).collect();
        Tensor::new(dual_data, vec![target_length, dimension])
    }
    
    /// Get alphabet for sequence type
    fn get_alphabet(&self, seq_type: &SequenceType) -> Vec<char> {
        match seq_type {
            SequenceType::DNA => vec!['A', 'T', 'C', 'G'],
            SequenceType::RNA => vec!['A', 'U', 'C', 'G'],
            SequenceType::Protein => vec![
                'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
            ],
        }
    }
    
    /// Generate all possible k-mers for an alphabet
    fn generate_kmers(&self, alphabet: &[char], k: usize) -> Vec<String> {
        if k == 0 {
            return vec![];
        }
        if k == 1 {
            return alphabet.iter().map(|&c| c.to_string()).collect();
        }
        
        let smaller_kmers = self.generate_kmers(alphabet, k - 1);
        let mut kmers = Vec::new();
        
        for base in alphabet {
            for kmer in &smaller_kmers {
                kmers.push(format!("{}{}", base, kmer));
            }
        }
        
        kmers
    }
    
    /// Get amino acid physicochemical properties
    fn get_amino_acid_properties(&self) -> HashMap<char, Vec<f64>> {
        // Simplified physicochemical properties
        // Real implementation would use complete property databases
        let mut properties = HashMap::new();
        
        // Properties: [hydrophobicity, molecular_weight, charge, polar, aromatic]
        properties.insert('A', vec![1.8, 89.0, 0.0, 0.0, 0.0]);   // Alanine
        properties.insert('C', vec![2.5, 121.0, 0.0, 0.0, 0.0]);  // Cysteine
        properties.insert('D', vec![-3.5, 133.0, -1.0, 1.0, 0.0]); // Aspartic acid
        properties.insert('E', vec![-3.5, 147.0, -1.0, 1.0, 0.0]); // Glutamic acid
        properties.insert('F', vec![2.8, 165.0, 0.0, 0.0, 1.0]);  // Phenylalanine
        properties.insert('G', vec![-0.4, 75.0, 0.0, 0.0, 0.0]);  // Glycine
        properties.insert('H', vec![-3.2, 155.0, 1.0, 1.0, 1.0]); // Histidine
        properties.insert('I', vec![4.5, 131.0, 0.0, 0.0, 0.0]);  // Isoleucine
        properties.insert('K', vec![-3.9, 146.0, 1.0, 1.0, 0.0]); // Lysine
        properties.insert('L', vec![3.8, 131.0, 0.0, 0.0, 0.0]);  // Leucine
        properties.insert('M', vec![1.9, 149.0, 0.0, 0.0, 0.0]);  // Methionine
        properties.insert('N', vec![-3.5, 132.0, 0.0, 1.0, 0.0]); // Asparagine
        properties.insert('P', vec![-1.6, 115.0, 0.0, 0.0, 0.0]); // Proline
        properties.insert('Q', vec![-3.5, 146.0, 0.0, 1.0, 0.0]); // Glutamine
        properties.insert('R', vec![-4.5, 174.0, 1.0, 1.0, 0.0]); // Arginine
        properties.insert('S', vec![-0.8, 105.0, 0.0, 1.0, 0.0]); // Serine
        properties.insert('T', vec![-0.7, 119.0, 0.0, 1.0, 0.0]); // Threonine
        properties.insert('V', vec![4.2, 117.0, 0.0, 0.0, 0.0]);  // Valine
        properties.insert('W', vec![-0.9, 204.0, 0.0, 0.0, 1.0]); // Tryptophan
        properties.insert('Y', vec![-1.3, 181.0, 0.0, 1.0, 1.0]); // Tyrosine
        
        properties
    }
    
    /// Get substitution matrix (BLOSUM62 as example)
    fn get_substitution_matrix(&self, matrix_type: &str) -> MLResult<HashMap<char, Vec<f64>>> {
        match matrix_type.to_uppercase().as_str() {
            "BLOSUM62" => Ok(self.get_blosum62_matrix()),
            "PAM250" => Ok(self.get_pam250_matrix()),
            _ => Err(MLError::DataError {
                reason: format!("Unknown substitution matrix: {}", matrix_type),
            }),
        }
    }
    
    /// Simplified BLOSUM62 matrix
    fn get_blosum62_matrix(&self) -> HashMap<char, Vec<f64>> {
        // This is a simplified version - real implementation would use full matrix
        let amino_acids = vec!['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                              'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'];
        let mut matrix = HashMap::new();
        
        // Simplified scoring - real BLOSUM62 has specific values
        for (i, &aa1) in amino_acids.iter().enumerate() {
            let mut scores = vec![0.0; 20];
            for (j, &_aa2) in amino_acids.iter().enumerate() {
                scores[j] = if i == j { 5.0 } else { -1.0 }; // Simplified
            }
            matrix.insert(aa1, scores);
        }
        
        matrix
    }
    
    /// Simplified PAM250 matrix  
    fn get_pam250_matrix(&self) -> HashMap<char, Vec<f64>> {
        // Placeholder - similar to BLOSUM62 but different values
        self.get_blosum62_matrix()
    }
}

impl MLPreprocessor for SequencePreprocessor {
    fn preprocess(&self, data: &Value) -> MLResult<Value> {
        // Handle single sequence
        if let Value::LyObj(obj) = data {
            if let Some(sequence) = obj.downcast_ref::<BiologicalSequence>() {
                let tensor = self.encode_sequence(sequence)?;
                // Convert tensor back to Value (list of numbers)
                let values: Vec<Value> = tensor.data.iter()
                    .map(|dual| Value::Real(dual.value()))
                    .collect();
                return Ok(Value::List(values));
            }
        }
        
        // Handle list of sequences
        if let Value::List(items) = data {
            let mut encoded_sequences = Vec::new();
            
            for item in items {
                if let Value::LyObj(obj) = item {
                    if let Some(sequence) = obj.downcast_ref::<BiologicalSequence>() {
                        let tensor = self.encode_sequence(sequence)?;
                        // Convert tensor to Value (list of numbers)
                        let values: Vec<Value> = tensor.data.iter()
                            .map(|dual| Value::Real(dual.value()))
                            .collect();
                        encoded_sequences.push(Value::List(values));
                    } else {
                        return Err(MLError::DataError {
                            reason: "Expected BiologicalSequence objects in list".to_string(),
                        });
                    }
                } else {
                    return Err(MLError::DataError {
                        reason: "Expected BiologicalSequence objects in list".to_string(),
                    });
                }
            }
            
            return Ok(Value::List(encoded_sequences));
        }
        
        Err(MLError::DataError {
            reason: "Expected BiologicalSequence or List of BiologicalSequence".to_string(),
        })
    }
    
    fn name(&self) -> &str {
        "SequencePreprocessor"
    }
    
    fn config(&self) -> HashMap<String, Value> {
        let mut config = HashMap::new();
        config.insert("encoding".to_string(), Value::String(format!("{:?}", self.encoding)));
        config.insert("normalize".to_string(), Value::Boolean(self.normalize));
        if let Some(padding) = self.padding_length {
            config.insert("padding_length".to_string(), Value::Integer(padding as i64));
        }
        if let Some(seq_type) = &self.sequence_type {
            config.insert("sequence_type".to_string(), Value::String(format!("{:?}", seq_type)));
        }
        config
    }
    
    fn clone_boxed(&self) -> Box<dyn MLPreprocessor> {
        Box::new(self.clone())
    }
}

/// Sequence dataset for bioinformatics ML workflows
#[derive(Debug, Clone)]
pub struct SequenceDataset {
    sequences: Vec<BiologicalSequence>,
    labels: Option<Vec<Value>>,
    metadata: HashMap<String, Value>,
}

impl SequenceDataset {
    /// Create new sequence dataset
    pub fn new(sequences: Vec<BiologicalSequence>) -> Self {
        Self {
            sequences,
            labels: None,
            metadata: HashMap::new(),
        }
    }
    
    /// Add labels for supervised learning
    pub fn with_labels(mut self, labels: Vec<Value>) -> MLResult<Self> {
        if labels.len() != self.sequences.len() {
            return Err(MLError::DataError {
                reason: format!("Label count ({}) doesn't match sequence count ({})", 
                              labels.len(), self.sequences.len()),
            });
        }
        self.labels = Some(labels);
        Ok(self)
    }
    
    /// Add metadata
    pub fn with_metadata(mut self, metadata: HashMap<String, Value>) -> Self {
        self.metadata = metadata;
        self
    }
    
    /// Get sequence count
    pub fn len(&self) -> usize {
        self.sequences.len()
    }
    
    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }
    
    /// Get sequences and labels as tensor data for ML training
    pub fn to_ml_data(&self, preprocessor: &SequencePreprocessor) -> MLResult<(Tensor, Option<Tensor>)> {
        // Process sequences
        let mut sequence_data = Vec::new();
        for sequence in &self.sequences {
            let encoded = preprocessor.encode_sequence(sequence)?;
            sequence_data.push(encoded);
        }
        
        // Stack sequences into batch tensor
        let batch_tensor = self.stack_tensors(sequence_data)?;
        
        // Process labels if available
        let label_tensor = if let Some(labels) = &self.labels {
            Some(self.process_labels(labels)?)
        } else {
            None
        };
        
        Ok((batch_tensor, label_tensor))
    }
    
    /// Stack individual tensors into batch tensor
    fn stack_tensors(&self, tensors: Vec<Tensor>) -> MLResult<Tensor> {
        if tensors.is_empty() {
            return Err(MLError::DataError {
                reason: "Cannot stack empty tensor list".to_string(),
            });
        }
        
        // Get dimensions from first tensor
        let first_shape = tensors[0].shape.clone();
        let batch_size = tensors.len();
        
        // Verify all tensors have same shape
        for (i, tensor) in tensors.iter().enumerate() {
            if tensor.shape != first_shape {
                return Err(MLError::DataError {
                    reason: format!("Tensor {} has incompatible shape", i),
                });
            }
        }
        
        // Stack tensors
        let element_count = first_shape.iter().product::<usize>();
        let mut stacked_data = Vec::with_capacity(batch_size * element_count);
        
        for tensor in tensors {
            stacked_data.extend_from_slice(&tensor.data);
        }
        
        let mut batch_shape = vec![batch_size];
        batch_shape.extend(first_shape);
        
        Tensor::new(stacked_data, batch_shape)
    }
    
    /// Process labels into tensor format
    fn process_labels(&self, labels: &[Value]) -> MLResult<Tensor> {
        let mut label_data = Vec::new();
        
        for label in labels {
            match label {
                Value::Integer(i) => label_data.push(*i as f64),
                Value::Real(r) => label_data.push(*r),
                Value::Boolean(b) => label_data.push(if *b { 1.0 } else { 0.0 }),
                _ => return Err(MLError::DataError {
                    reason: "Labels must be numeric or boolean".to_string(),
                }),
            }
        }
        
        let dual_data: Vec<Dual> = label_data.into_iter().map(crate::stdlib::autodiff::Dual::variable).collect();
        let data_len = dual_data.len();
        Tensor::new(dual_data, vec![data_len])
    }
}

impl Foreign for SequenceDataset {
    fn type_name(&self) -> &'static str {
        "SequenceDataset"
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "length" => Ok(Value::Integer(self.len() as i64)),
            "sequenceType" => {
                if let Some(first_seq) = self.sequences.first() {
                    Ok(Value::String(format!("{:?}", first_seq.sequence_type)))
                } else {
                    Ok(Value::String("Empty".to_string()))
                }
            }
            "hasLabels" => Ok(Value::Boolean(self.labels.is_some())),
            "metadata" => {
                let metadata_list: Vec<Value> = self.metadata.iter()
                    .map(|(k, v)| Value::List(vec![Value::String(k.clone()), v.clone()]))
                    .collect();
                Ok(Value::List(metadata_list))
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }
}

/// Bio-ML workflow orchestrator
pub struct BioMLWorkflow {
    preprocessor: SequencePreprocessor,
    auto_preprocessor: AutoPreprocessor,
}

impl BioMLWorkflow {
    /// Create new Bio-ML workflow
    pub fn new(encoding: SequenceEncoding) -> Self {
        Self {
            preprocessor: SequencePreprocessor::new(encoding),
            auto_preprocessor: AutoPreprocessor::new(),
        }
    }
    
    /// Train ML model on biological sequence data
    pub fn train_sequence_classifier(
        &self,
        dataset: &SequenceDataset,
        config: TrainingConfig,
    ) -> MLResult<TrainingResult> {
        // Convert sequences to ML tensors
        let (sequence_tensor, label_tensor) = dataset.to_ml_data(&self.preprocessor)?;
        
        let labels = label_tensor.ok_or_else(|| MLError::DataError {
            reason: "Sequence classification requires labeled data".to_string(),
        })?;
        
        // Create neural network for sequence classification
        let input_size = sequence_tensor.shape[1..].iter().product::<usize>();
        let output_size = self.infer_output_size(&labels)?;
        
        let mut network = self.create_sequence_classifier(input_size, output_size)?;
        
        // Train the network
        let trainer = NetTrain::with_config(config);
        // Convert tensors to training pairs
        let training_data = vec![(sequence_tensor, labels)];
        trainer.train(&mut network, &training_data, &crate::stdlib::ml::losses::MSELoss, &mut crate::stdlib::ml::optimizers::SGD::new(0.001))
    }
    
    /// Create sequence classifier network architecture
    fn create_sequence_classifier(&self, _input_size: usize, _output_size: usize) -> MLResult<NetChain> {
        // Use NetChain builder pattern to create appropriate architecture
        let network = NetChain::new(vec![]);
        
        // Add layers appropriate for sequence data
        // This would be implemented with the layer system once it's available
        // For now, return a placeholder network
        
        Ok(network)
    }
    
    /// Infer output size from labels
    fn infer_output_size(&self, labels: &Tensor) -> MLResult<usize> {
        // Find unique label values to determine number of classes
        let unique_values: std::collections::HashSet<i64> = labels.data
            .iter()
            .map(|x| x.value() as i64)
            .collect();
        
        Ok(unique_values.len())
    }
}

/// Standard library function registrations for Bio-ML integration
pub fn biological_sequence_to_ml(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(VmError::ArityError {
            function_name: "BiologicalSequenceToML".to_string(),
            expected: 2,
            actual: args.len(),
        });
    }
    
    // Extract sequence and encoding type
    let sequence = if let Value::LyObj(obj) = &args[0] {
        obj.downcast_ref::<BiologicalSequence>()
            .ok_or_else(|| VmError::TypeError {
                expected: "BiologicalSequence".to_string(),
                actual: "other".to_string(),
            })?
    } else {
        return Err(VmError::TypeError {
            expected: "BiologicalSequence".to_string(),
            actual: format!("{:?}", args[0]),
        });
    };
    
    let encoding_str = if let Value::String(s) = &args[1] {
        s
    } else {
        return Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[1]),
        });
    };
    
    // Parse encoding type
    let encoding = match encoding_str.to_lowercase().as_str() {
        "onehot" => SequenceEncoding::OneHot,
        "ordinal" => SequenceEncoding::Ordinal,
        "kmer" => {
            let k = if args.len() > 2 {
                if let Value::Integer(k_val) = &args[2] {
                    *k_val as usize
                } else {
                    3 // Default k=3
                }
            } else {
                3
            };
            SequenceEncoding::KMer { k }
        }
        "physicochemical" => SequenceEncoding::Physicochemical,
        "blosum62" => SequenceEncoding::SubstitutionMatrix { 
            matrix_type: "BLOSUM62".to_string() 
        },
        _ => return Err(VmError::ArgumentTypeError {
            function_name: "BiologicalSequenceToML".to_string(),
            param_index: 1,
            expected: "valid encoding type (onehot, ordinal, kmer, etc.)".to_string(),
            actual: "unknown encoding".to_string(),
        }),
    };
    
    // Create preprocessor and encode sequence
    let preprocessor = SequencePreprocessor::new(encoding);
    let tensor = preprocessor.encode_sequence(sequence)
        .map_err(|e| VmError::Runtime(format!("Sequence encoding failed: {:?}", e)))?;
    
    // Convert tensor to Value (list of numbers)
    let values: Vec<Value> = tensor.data.iter()
        .map(|dual| Value::Real(dual.value()))
        .collect();
    Ok(Value::List(values))
}

/// Create sequence dataset from list of sequences and optional labels
pub fn sequence_dataset(args: &[Value]) -> VmResult<Value> {
    if args.is_empty() {
        return Err(VmError::ArityError {
            function_name: "SequenceDataset".to_string(),
            expected: 1,
            actual: 0,
        });
    }
    
    // Extract sequences from list
    let sequences = if let Value::List(items) = &args[0] {
        let mut bio_sequences = Vec::new();
        for item in items {
            if let Value::LyObj(obj) = item {
                if let Some(sequence) = obj.downcast_ref::<BiologicalSequence>() {
                    bio_sequences.push(sequence.clone());
                } else {
                    return Err(VmError::TypeError {
                        expected: "BiologicalSequence".to_string(),
                        actual: "other".to_string(),
                    });
                }
            } else {
                return Err(VmError::TypeError {
                    expected: "BiologicalSequence".to_string(),
                    actual: format!("{:?}", item),
                });
            }
        }
        bio_sequences
    } else {
        return Err(VmError::TypeError {
            expected: "List of BiologicalSequence".to_string(),
            actual: format!("{:?}", args[0]),
        });
    };
    
    let mut dataset = SequenceDataset::new(sequences);
    
    // Add labels if provided
    if args.len() > 1 {
        if let Value::List(label_items) = &args[1] {
            dataset = dataset.with_labels(label_items.clone())
                .map_err(|e| VmError::Runtime(format!("Label processing failed: {:?}", e)))?;
        }
    }
    
    Ok(Value::LyObj(LyObj::new(Box::new(dataset))))
}

/// Train sequence classifier with automatic preprocessing
pub fn train_sequence_classifier(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(VmError::ArityError {
            function_name: "TrainSequenceClassifier".to_string(),
            expected: 2,
            actual: args.len(),
        });
    }
    
    // Extract dataset
    let dataset = if let Value::LyObj(obj) = &args[0] {
        obj.downcast_ref::<SequenceDataset>()
            .ok_or_else(|| VmError::TypeError {
                expected: "SequenceDataset".to_string(),
                actual: "other".to_string(),
            })?
    } else {
        return Err(VmError::TypeError {
            expected: "SequenceDataset".to_string(),
            actual: format!("{:?}", args[0]),
        });
    };
    
    // Extract encoding type
    let encoding_str = if let Value::String(s) = &args[1] {
        s
    } else {
        return Err(VmError::TypeError {
            expected: "String encoding type".to_string(),
            actual: format!("{:?}", args[1]),
        });
    };
    
    // Parse encoding
    let encoding = match encoding_str.to_lowercase().as_str() {
        "onehot" => SequenceEncoding::OneHot,
        "ordinal" => SequenceEncoding::Ordinal,
        "kmer" => SequenceEncoding::KMer { k: 3 },
        "physicochemical" => SequenceEncoding::Physicochemical,
        _ => return Err(VmError::ArgumentTypeError {
            function_name: "TrainSequenceClassifier".to_string(),
            param_index: 1,
            expected: "valid encoding type (onehot, ordinal, kmer, etc.)".to_string(),
            actual: "unknown encoding".to_string(),
        }),
    };
    
    // Create workflow and train model
    let workflow = BioMLWorkflow::new(encoding);
    let config = TrainingConfig::default();
    
    let result = workflow.train_sequence_classifier(dataset, config)
        .map_err(|e| VmError::Runtime(format!("Training failed: {:?}", e)))?;
    
    // Convert TrainingResult to Value (simplified)
    let mut result_map = HashMap::new();
    result_map.insert("final_loss".to_string(), Value::Real(result.final_loss));
    result_map.insert("epochs_completed".to_string(), Value::Integer(result.epochs_completed as i64));
    
    Ok(Value::List(result_map.into_iter()
        .map(|(k, v)| Value::List(vec![Value::String(k), v]))
        .collect()))
}

/// Register Bio-ML integration functions
pub fn register_bio_ml_functions() -> HashMap<String, fn(&[Value]) -> VmResult<Value>> {
    let mut functions = HashMap::new();
    
    functions.insert("BiologicalSequenceToML".to_string(), biological_sequence_to_ml as fn(&[Value]) -> VmResult<Value>);
    functions.insert("SequenceDataset".to_string(), sequence_dataset as fn(&[Value]) -> VmResult<Value>);
    functions.insert("TrainSequenceClassifier".to_string(), train_sequence_classifier as fn(&[Value]) -> VmResult<Value>);
    
    functions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stdlib::bioinformatics::genomics::BiologicalSequence;

    #[test]
    fn test_sequence_one_hot_encoding() {
        let sequence = BiologicalSequence {
            sequence: "ATCG".to_string(),
            sequence_type: SequenceType::DNA,
            description: None,
        };
        
        let preprocessor = SequencePreprocessor::new(SequenceEncoding::OneHot);
        let tensor = preprocessor.encode_sequence(&sequence).unwrap();
        
        // Should be 4x4 matrix (4 nucleotides, 4 positions)
        assert_eq!(&tensor.shape, &[4, 4]);
        
        // Check encoding: A=[1,0,0,0], T=[0,1,0,0], C=[0,0,1,0], G=[0,0,0,1]
        let data = &tensor.data;
        assert_eq!(data[0].value(), 1.0); // A position 0
        assert_eq!(data[1].value(), 0.0);
        assert_eq!(data[2].value(), 0.0);
        assert_eq!(data[3].value(), 0.0);
    }

    #[test]
    fn test_sequence_dataset_creation() {
        let sequences = vec![
            BiologicalSequence {
                sequence: "ATCG".to_string(),
                sequence_type: SequenceType::DNA,
                description: None,
            },
            BiologicalSequence {
                sequence: "GCTA".to_string(),
                sequence_type: SequenceType::DNA,
                description: None,
            },
        ];
        
        let labels = vec![Value::Integer(0), Value::Integer(1)];
        let dataset = SequenceDataset::new(sequences)
            .with_labels(labels).unwrap();
        
        assert_eq!(dataset.len(), 2);
        assert!(dataset.labels.is_some());
    }

    #[test]
    fn test_kmer_encoding() {
        let sequence = BiologicalSequence {
            sequence: "ATCGATCG".to_string(),
            sequence_type: SequenceType::DNA,
            description: None,
        };
        
        let preprocessor = SequencePreprocessor::new(SequenceEncoding::KMer { k: 3 });
        let tensor = preprocessor.encode_sequence(&sequence).unwrap();
        
        // Should have 4^3 = 64 possible 3-mers
        assert_eq!(&tensor.shape, &[64]);
    }
}
