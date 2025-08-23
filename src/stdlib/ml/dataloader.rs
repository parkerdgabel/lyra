//! ML DataLoader Infrastructure
//!
//! This module provides memory-efficient batch processing for ML training,
//! with integrated preprocessing pipeline support and shuffling capabilities.

use crate::stdlib::ml::{MLResult, MLError};
use crate::stdlib::ml::layers::Tensor;
use crate::stdlib::ml::preprocessing::MLPreprocessor;
use crate::stdlib::data::{ForeignDataset, ForeignTable};
use crate::stdlib::autodiff::Dual;
use crate::vm::Value;
use std::collections::HashMap;
use rand::seq::SliceRandom;
use rand::thread_rng;

/// Configuration for DataLoader behavior
#[derive(Debug, Clone)]
pub struct DataLoaderConfig {
    pub batch_size: usize,
    pub shuffle: bool,
    pub drop_last: bool,
    pub num_workers: usize,
    pub pin_memory: bool,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            shuffle: true,
            drop_last: false,
            num_workers: 1,
            pin_memory: false,
        }
    }
}

/// DataLoader: Memory-efficient batch iterator for ML training
/// Supports various data sources with integrated preprocessing
pub struct DataLoader {
    config: DataLoaderConfig,
    data_source: DataSource,
    preprocessor: Option<Box<dyn MLPreprocessor>>,
    indices: Vec<usize>,
    current_epoch: usize,
}

/// Supported data source types
#[derive(Debug)]
pub enum DataSource {
    Dataset {
        dataset: ForeignDataset,
        target_extraction: crate::stdlib::ml::training::DatasetTargetExtraction,
    },
    Table {
        table: ForeignTable,
        feature_columns: Vec<String>,
        target_column: String,
    },
    TensorPairs {
        data: Vec<(Tensor, Tensor)>,
    },
    ValuePairs {
        data: Vec<(Value, Value)>,
    },
}

impl DataLoader {
    /// Create DataLoader from ForeignDataset
    pub fn from_dataset(
        dataset: ForeignDataset,
        target_extraction: crate::stdlib::ml::training::DatasetTargetExtraction,
        config: DataLoaderConfig,
    ) -> MLResult<Self> {
        let data_len = Self::get_dataset_length(&dataset)?;
        
        Ok(Self {
            config,
            data_source: DataSource::Dataset { dataset, target_extraction },
            preprocessor: None,
            indices: (0..data_len).collect(),
            current_epoch: 0,
        })
    }
    
    /// Create DataLoader from ForeignTable
    pub fn from_table(
        table: ForeignTable,
        feature_columns: Vec<String>,
        target_column: String,
        config: DataLoaderConfig,
    ) -> MLResult<Self> {
        if table.length == 0 {
            return Err(MLError::DataError {
                reason: "Cannot create DataLoader from empty table".to_string(),
            });
        }
        
        // Validate columns exist
        for col_name in &feature_columns {
            if table.get_column(col_name).is_none() {
                return Err(MLError::DataError {
                    reason: format!("Feature column '{}' not found in table", col_name),
                });
            }
        }
        
        if table.get_column(&target_column).is_none() {
            return Err(MLError::DataError {
                reason: format!("Target column '{}' not found in table", target_column),
            });
        }
        
        let table_length = table.length;
        
        Ok(Self {
            config,
            data_source: DataSource::Table { table, feature_columns, target_column },
            preprocessor: None,
            indices: (0..table_length).collect(),
            current_epoch: 0,
        })
    }
    
    /// Create DataLoader from pre-computed tensor pairs
    pub fn from_tensor_pairs(
        data: Vec<(Tensor, Tensor)>,
        config: DataLoaderConfig,
    ) -> MLResult<Self> {
        if data.is_empty() {
            return Err(MLError::DataError {
                reason: "Cannot create DataLoader from empty tensor data".to_string(),
            });
        }
        
        let data_len = data.len();
        
        Ok(Self {
            config,
            data_source: DataSource::TensorPairs { data },
            preprocessor: None,
            indices: (0..data_len).collect(),
            current_epoch: 0,
        })
    }
    
    /// Create DataLoader from Value pairs
    pub fn from_value_pairs(
        data: Vec<(Value, Value)>,
        config: DataLoaderConfig,
    ) -> MLResult<Self> {
        if data.is_empty() {
            return Err(MLError::DataError {
                reason: "Cannot create DataLoader from empty Value data".to_string(),
            });
        }
        
        let data_len = data.len();
        
        Ok(Self {
            config,
            data_source: DataSource::ValuePairs { data },
            preprocessor: None,
            indices: (0..data_len).collect(),
            current_epoch: 0,
        })
    }
    
    /// Set preprocessing pipeline for this DataLoader
    pub fn with_preprocessing(mut self, preprocessor: Box<dyn MLPreprocessor>) -> Self {
        self.preprocessor = Some(preprocessor);
        self
    }
    
    /// Get the total number of samples in the dataset
    pub fn len(&self) -> usize {
        self.indices.len()
    }
    
    /// Check if the dataset is empty
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
    
    /// Get the number of batches per epoch
    pub fn num_batches(&self) -> usize {
        if self.config.drop_last {
            self.len() / self.config.batch_size
        } else {
            (self.len() + self.config.batch_size - 1) / self.config.batch_size
        }
    }
    
    /// Start a new epoch (reshuffles if shuffle=true)
    pub fn new_epoch(&mut self) {
        self.current_epoch += 1;
        
        if self.config.shuffle {
            let mut rng = thread_rng();
            self.indices.shuffle(&mut rng);
        }
    }
    
    /// Get an iterator over batches for current epoch
    pub fn iter_batches(&self) -> DataLoaderIterator {
        DataLoaderIterator {
            loader: self,
            batch_index: 0,
        }
    }
    
    /// Load a single batch by index
    pub fn get_batch(&self, batch_index: usize) -> MLResult<Vec<(Tensor, Tensor)>> {
        let start_idx = batch_index * self.config.batch_size;
        let end_idx = if self.config.drop_last {
            std::cmp::min(start_idx + self.config.batch_size, self.len())
        } else {
            std::cmp::min(start_idx + self.config.batch_size, self.len())
        };
        
        if start_idx >= self.len() {
            return Ok(Vec::new()); // Empty batch for out-of-bounds
        }
        
        let mut batch_data = Vec::new();
        
        for &data_idx in &self.indices[start_idx..end_idx] {
            let (input, target) = self.load_sample(data_idx)?;
            batch_data.push((input, target));
        }
        
        Ok(batch_data)
    }
    
    /// Load a single sample by index
    fn load_sample(&self, index: usize) -> MLResult<(Tensor, Tensor)> {
        match &self.data_source {
            DataSource::Dataset { dataset, target_extraction } => {
                self.load_dataset_sample(dataset, index, *target_extraction)
            },
            DataSource::Table { table, feature_columns, target_column } => {
                self.load_table_sample(table, index, feature_columns, target_column)
            },
            DataSource::TensorPairs { data } => {
                if index >= data.len() {
                    return Err(MLError::DataError {
                        reason: format!("Sample index {} out of bounds for tensor data", index),
                    });
                }
                Ok(data[index].clone())
            },
            DataSource::ValuePairs { data } => {
                if index >= data.len() {
                    return Err(MLError::DataError {
                        reason: format!("Sample index {} out of bounds for Value data", index),
                    });
                }
                let (input_value, target_value) = &data[index];
                let input_tensor = self.value_to_tensor_with_preprocessing(input_value)?;
                let target_tensor = self.value_to_tensor(target_value)?;
                Ok((input_tensor, target_tensor))
            },
        }
    }
    
    /// Load sample from ForeignDataset
    fn load_dataset_sample(
        &self,
        dataset: &ForeignDataset,
        index: usize,
        target_extraction: crate::stdlib::ml::training::DatasetTargetExtraction,
    ) -> MLResult<(Tensor, Tensor)> {
        // For now, assume dataset contains list of lists
        // In a more complete implementation, this would handle various dataset formats
        let data_value = dataset.get_value();
        
        match data_value {
            Value::List(elements) => {
                if index >= elements.len() {
                    return Err(MLError::DataError {
                        reason: format!("Sample index {} out of bounds for dataset", index),
                    });
                }
                
                let sample = &elements[index];
                match sample {
                    Value::List(sample_elements) => {
                        self.split_sample_features_target(sample_elements, target_extraction)
                    },
                    _ => Err(MLError::DataError {
                        reason: "Dataset samples must be lists".to_string(),
                    }),
                }
            },
            _ => Err(MLError::DataError {
                reason: "Dataset must contain List data".to_string(),
            }),
        }
    }
    
    /// Load sample from ForeignTable
    fn load_table_sample(
        &self,
        table: &ForeignTable,
        index: usize,
        feature_columns: &[String],
        target_column: &str,
    ) -> MLResult<(Tensor, Tensor)> {
        if index >= table.length {
            return Err(MLError::DataError {
                reason: format!("Sample index {} out of bounds for table", index),
            });
        }
        
        // Extract features
        let mut feature_values = Vec::new();
        for col_name in feature_columns {
            if let Some(series) = table.get_column(col_name) {
                if let Ok(value) = series.get(index) {
                    feature_values.push(value.clone());
                } else {
                    return Err(MLError::DataError {
                        reason: format!("Cannot access row {} in column '{}'", index, col_name),
                    });
                }
            } else {
                return Err(MLError::DataError {
                    reason: format!("Feature column '{}' not found", col_name),
                });
            }
        }
        
        // Extract target
        let target_value = if let Some(target_series) = table.get_column(target_column) {
            if let Ok(value) = target_series.get(index) {
                value.clone()
            } else {
                return Err(MLError::DataError {
                    reason: format!("Cannot access row {} in target column '{}'", index, target_column),
                });
            }
        } else {
            return Err(MLError::DataError {
                reason: format!("Target column '{}' not found", target_column),
            });
        };
        
        // Convert to tensors
        let features_list = Value::List(feature_values);
        let input_tensor = self.value_to_tensor_with_preprocessing(&features_list)?;
        let target_tensor = self.value_to_tensor(&target_value)?;
        
        Ok((input_tensor, target_tensor))
    }
    
    /// Split sample into features and target based on extraction strategy
    fn split_sample_features_target(
        &self,
        sample_elements: &[Value],
        target_extraction: crate::stdlib::ml::training::DatasetTargetExtraction,
    ) -> MLResult<(Tensor, Tensor)> {
        use crate::stdlib::ml::training::DatasetTargetExtraction;
        
        match target_extraction {
            DatasetTargetExtraction::LastElement => {
                if sample_elements.len() < 2 {
                    return Err(MLError::DataError {
                        reason: "Sample must have at least 2 elements for feature-target split".to_string(),
                    });
                }
                
                let features = &sample_elements[..sample_elements.len()-1];
                let target = &sample_elements[sample_elements.len()-1];
                
                let features_value = Value::List(features.to_vec());
                let input_tensor = self.value_to_tensor_with_preprocessing(&features_value)?;
                let target_tensor = self.value_to_tensor(target)?;
                
                Ok((input_tensor, target_tensor))
            },
            DatasetTargetExtraction::FirstElement => {
                if sample_elements.len() < 2 {
                    return Err(MLError::DataError {
                        reason: "Sample must have at least 2 elements for target-feature split".to_string(),
                    });
                }
                
                let target = &sample_elements[0];
                let features = &sample_elements[1..];
                
                let features_value = Value::List(features.to_vec());
                let input_tensor = self.value_to_tensor_with_preprocessing(&features_value)?;
                let target_tensor = self.value_to_tensor(target)?;
                
                Ok((input_tensor, target_tensor))
            },
            DatasetTargetExtraction::EvenOdd => {
                if sample_elements.len() % 2 != 0 {
                    return Err(MLError::DataError {
                        reason: "Sample must have even number of elements for even-odd split".to_string(),
                    });
                }
                
                // For single sample, assume first two elements
                if sample_elements.len() < 2 {
                    return Err(MLError::DataError {
                        reason: "Sample must have at least 2 elements".to_string(),
                    });
                }
                
                let input_tensor = self.value_to_tensor_with_preprocessing(&sample_elements[0])?;
                let target_tensor = self.value_to_tensor(&sample_elements[1])?;
                
                Ok((input_tensor, target_tensor))
            },
        }
    }
    
    /// Convert Value to Tensor with optional preprocessing
    fn value_to_tensor_with_preprocessing(&self, value: &Value) -> MLResult<Tensor> {
        let processed_value = if let Some(ref preprocessor) = self.preprocessor {
            preprocessor.preprocess(value)?
        } else {
            value.clone()
        };
        
        crate::stdlib::ml::preprocessing::preprocessed_value_to_tensor(&processed_value)
    }
    
    /// Convert Value to Tensor without preprocessing (for targets)
    fn value_to_tensor(&self, value: &Value) -> MLResult<Tensor> {
        match value {
            Value::Real(n) => {
                let data = vec![Dual::variable(*n)];
                Tensor::new(data, vec![1])
            },
            Value::Integer(n) => {
                let data = vec![Dual::variable(*n as f64)];
                Tensor::new(data, vec![1])
            },
            Value::List(elements) => {
                let dual_values: Result<Vec<Dual>, MLError> = elements.iter()
                    .map(|v| match v {
                        Value::Real(n) => Ok(Dual::variable(*n)),
                        Value::Integer(n) => Ok(Dual::variable(*n as f64)),
                        _ => Err(MLError::DataError {
                            reason: format!("Cannot convert {:?} to tensor", v),
                        }),
                    })
                    .collect();
                
                let data = dual_values?;
                let shape = vec![data.len()];
                Tensor::new(data, shape)
            },
            _ => Err(MLError::DataError {
                reason: format!("Cannot convert {:?} to tensor", value),
            }),
        }
    }
    
    /// Get the dataset length based on data source type
    fn get_dataset_length(dataset: &ForeignDataset) -> MLResult<usize> {
        let data_value = dataset.get_value();
        match data_value {
            Value::List(elements) => Ok(elements.len()),
            _ => Err(MLError::DataError {
                reason: "Dataset must contain List data".to_string(),
            }),
        }
    }
    
    /// Get configuration parameters
    pub fn config(&self) -> &DataLoaderConfig {
        &self.config
    }
    
    /// Get current epoch number
    pub fn epoch(&self) -> usize {
        self.current_epoch
    }
    
    /// Get data source information
    pub fn data_source_info(&self) -> HashMap<String, Value> {
        let mut info = HashMap::new();
        
        match &self.data_source {
            DataSource::Dataset { .. } => {
                info.insert("type".to_string(), Value::String("dataset".to_string()));
            },
            DataSource::Table { feature_columns, target_column, .. } => {
                info.insert("type".to_string(), Value::String("table".to_string()));
                info.insert("feature_columns".to_string(), Value::Integer(feature_columns.len() as i64));
                info.insert("target_column".to_string(), Value::String(target_column.clone()));
            },
            DataSource::TensorPairs { data } => {
                info.insert("type".to_string(), Value::String("tensor_pairs".to_string()));
                info.insert("samples".to_string(), Value::Integer(data.len() as i64));
            },
            DataSource::ValuePairs { data } => {
                info.insert("type".to_string(), Value::String("value_pairs".to_string()));
                info.insert("samples".to_string(), Value::Integer(data.len() as i64));
            },
        }
        
        info.insert("batch_size".to_string(), Value::Integer(self.config.batch_size as i64));
        info.insert("shuffle".to_string(), Value::Boolean(self.config.shuffle));
        info.insert("drop_last".to_string(), Value::Boolean(self.config.drop_last));
        info.insert("total_samples".to_string(), Value::Integer(self.len() as i64));
        info.insert("total_batches".to_string(), Value::Integer(self.num_batches() as i64));
        info.insert("current_epoch".to_string(), Value::Integer(self.current_epoch as i64));
        
        if self.preprocessor.is_some() {
            info.insert("has_preprocessing".to_string(), Value::Boolean(true));
        }
        
        info
    }
}

/// Iterator over DataLoader batches
pub struct DataLoaderIterator<'a> {
    loader: &'a DataLoader,
    batch_index: usize,
}

impl<'a> Iterator for DataLoaderIterator<'a> {
    type Item = MLResult<Vec<(Tensor, Tensor)>>;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.batch_index >= self.loader.num_batches() {
            return None;
        }
        
        let batch_result = self.loader.get_batch(self.batch_index);
        self.batch_index += 1;
        
        match batch_result {
            Ok(batch) if batch.is_empty() => None,
            Ok(batch) => Some(Ok(batch)),
            Err(e) => Some(Err(e)),
        }
    }
}

/// DataLoader Factory: Convenient creation methods
pub struct DataLoaderFactory;

impl DataLoaderFactory {
    /// Create DataLoader with automatic preprocessing from ForeignDataset
    pub fn auto_dataset(
        dataset: ForeignDataset,
        target_extraction: crate::stdlib::ml::training::DatasetTargetExtraction,
        batch_size: usize,
    ) -> MLResult<DataLoader> {
        let config = DataLoaderConfig {
            batch_size,
            ..Default::default()
        };
        
        let mut loader = DataLoader::from_dataset(dataset, target_extraction, config)?;
        
        // Auto-infer preprocessing from first sample
        if let Ok(first_batch) = loader.get_batch(0) {
            if let Some((first_input, _)) = first_batch.first() {
                // Convert tensor back to Value for preprocessing analysis
                let input_value = Self::tensor_to_value(first_input)?;
                if let Ok(auto_preprocessor) = crate::stdlib::ml::preprocessing::AutoPreprocessor::infer_pipeline(&input_value) {
                    loader = loader.with_preprocessing(Box::new(auto_preprocessor));
                }
            }
        }
        
        Ok(loader)
    }
    
    /// Create DataLoader with automatic preprocessing from ForeignTable
    pub fn auto_table(
        table: ForeignTable,
        feature_columns: Vec<String>,
        target_column: String,
        batch_size: usize,
    ) -> MLResult<DataLoader> {
        let config = DataLoaderConfig {
            batch_size,
            ..Default::default()
        };
        
        let mut loader = DataLoader::from_table(table, feature_columns, target_column, config)?;
        
        // Auto-infer preprocessing from first sample
        if let Ok(first_batch) = loader.get_batch(0) {
            if let Some((first_input, _)) = first_batch.first() {
                let input_value = Self::tensor_to_value(first_input)?;
                if let Ok(auto_preprocessor) = crate::stdlib::ml::preprocessing::AutoPreprocessor::infer_pipeline(&input_value) {
                    loader = loader.with_preprocessing(Box::new(auto_preprocessor));
                }
            }
        }
        
        Ok(loader)
    }
    
    /// Create simple DataLoader without preprocessing
    pub fn simple_tensors(
        data: Vec<(Tensor, Tensor)>,
        batch_size: usize,
        shuffle: bool,
    ) -> MLResult<DataLoader> {
        let config = DataLoaderConfig {
            batch_size,
            shuffle,
            ..Default::default()
        };
        
        DataLoader::from_tensor_pairs(data, config)
    }
    
    /// Helper: Convert tensor back to Value for preprocessing analysis
    fn tensor_to_value(tensor: &Tensor) -> MLResult<Value> {
        let values: Vec<Value> = tensor.data.iter()
            .map(|dual| Value::Real(dual.value()))
            .collect();
        
        if values.len() == 1 {
            Ok(values[0].clone())
        } else {
            Ok(Value::List(values))
        }
    }
}

/// Memory-efficient batch processing for very large datasets
pub struct StreamingDataLoader {
    config: DataLoaderConfig,
    chunk_size: usize,
    current_chunk: usize,
    total_chunks: usize,
    preprocessor: Option<Box<dyn MLPreprocessor>>,
}

impl StreamingDataLoader {
    /// Create streaming loader for very large datasets
    pub fn new(
        total_samples: usize,
        chunk_size: usize,
        config: DataLoaderConfig,
    ) -> Self {
        let total_chunks = (total_samples + chunk_size - 1) / chunk_size;
        
        Self {
            config,
            chunk_size,
            current_chunk: 0,
            total_chunks,
            preprocessor: None,
        }
    }
    
    /// Set preprocessing for streaming loader
    pub fn with_preprocessing(mut self, preprocessor: Box<dyn MLPreprocessor>) -> Self {
        self.preprocessor = Some(preprocessor);
        self
    }
    
    /// Load next chunk of data (to be implemented with specific data sources)
    pub fn next_chunk(&mut self) -> MLResult<Option<Vec<(Tensor, Tensor)>>> {
        if self.current_chunk >= self.total_chunks {
            return Ok(None);
        }
        
        // Placeholder implementation - in practice this would load from disk/database
        // and apply preprocessing to the chunk
        self.current_chunk += 1;
        
        // Return empty for now - real implementation would load actual data
        Ok(Some(Vec::new()))
    }
    
    /// Reset streaming loader to beginning
    pub fn reset(&mut self) {
        self.current_chunk = 0;
    }
    
    /// Get streaming progress info
    pub fn progress(&self) -> (usize, usize) {
        (self.current_chunk, self.total_chunks)
    }
}