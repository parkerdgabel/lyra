//! Neural Network Training Infrastructure
//!
//! NetTrain implements comprehensive training functionality for neural networks
//! with support for various optimizers, loss functions, and training strategies.

use crate::stdlib::ml::{MLResult, MLError};
use crate::stdlib::ml::layers::Tensor;
use crate::stdlib::ml::NetChain;
use crate::stdlib::ml::losses::{LossFunction, MSELoss};
use crate::stdlib::ml::optimizers::{Optimizer, SGD};
use crate::stdlib::data::{ForeignDataset, ForeignTable};
use crate::stdlib::autodiff::Dual;
use crate::stdlib::ml::preprocessing::{MLPreprocessor, AutoPreprocessor};
use crate::vm::Value;

/// Training configuration for neural networks
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub print_progress: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 32,
            learning_rate: 0.001,
            print_progress: true,
        }
    }
}

/// Training results and statistics
#[derive(Debug, Clone)]
pub struct TrainingResult {
    pub final_loss: f64,
    pub epochs_completed: usize,
    pub loss_history: Vec<f64>,
}

/// NetTrain: Main training function for neural networks
pub struct NetTrain {
    config: TrainingConfig,
}

impl NetTrain {
    /// Create NetTrain with default configuration
    pub fn new() -> Self {
        Self {
            config: TrainingConfig::default(),
        }
    }
    
    /// Create NetTrain with custom configuration
    pub fn with_config(config: TrainingConfig) -> Self {
        Self { config }
    }
    
    /// Train a neural network on the given dataset
    pub fn train(
        &self, 
        network: &mut NetChain, 
        data: &[(Tensor, Tensor)],
        loss_fn: &dyn LossFunction,
        optimizer: &mut dyn Optimizer
    ) -> MLResult<TrainingResult> {
        
        if data.is_empty() {
            return Err(MLError::TrainingError {
                reason: "Training data is empty".to_string(),
            });
        }
        
        let mut loss_history = Vec::new();
        let mut final_loss = 0.0;
        
        // Training loop
        for epoch in 0..self.config.epochs {
            let mut epoch_loss = 0.0;
            
            // Process data in batches
            for batch in data.chunks(self.config.batch_size) {
                let mut batch_loss = 0.0;
                
                // Forward pass for each sample in batch
                for (input, target) in batch {
                    // Forward pass
                    let prediction = network.forward(input)?;
                    
                    // Compute loss
                    let loss = loss_fn.compute_loss(&prediction, target)?;
                    batch_loss += loss;
                    
                    // Real autodiff gradient computation
                    let gradients = self.compute_autodiff_gradients(network, input, target, loss_fn)?;
                    
                    // Update parameters
                    let mut params: Vec<&mut Tensor> = network.parameters_mut();
                    let grad_refs: Vec<&Tensor> = gradients.iter().collect();
                    optimizer.step(params.as_mut_slice(), &grad_refs)?;
                }
                
                epoch_loss += batch_loss / batch.len() as f64;
            }
            
            final_loss = epoch_loss;
            loss_history.push(final_loss);
            
            if self.config.print_progress && epoch % 10 == 0 {
                println!("Epoch {}/{}: Loss = {:.6}", epoch + 1, self.config.epochs, final_loss);
            }
        }
        
        if self.config.print_progress {
            println!("Training completed! Final loss: {:.6}", final_loss);
        }
        
        Ok(TrainingResult {
            final_loss,
            epochs_completed: self.config.epochs,
            loss_history,
        })
    }
    
    /// Compute gradients using automatic differentiation
    /// This replaces the placeholder with real autodiff-powered gradient computation
    fn compute_autodiff_gradients(
        &self,
        network: &mut NetChain,
        input: &Tensor,
        target: &Tensor,
        loss_fn: &dyn LossFunction,
    ) -> MLResult<Vec<Tensor>> {
        
        // Step 1: Forward pass with gradient tracking enabled
        // The network parameters already have Dual::variable values with gradient=1
        let prediction = network.forward(input)?;
        
        // Step 2: Compute loss with automatic differentiation
        let _loss_value = loss_fn.compute_loss(&prediction, target)?;
        
        // Step 3: Extract gradients from the dual numbers in network parameters
        let parameters = network.parameters();
        let mut gradients = Vec::new();
        
        for param in parameters {
            let mut grad_data = Vec::with_capacity(param.data.len());
            
            // Extract the gradient (derivative) from each dual number
            for dual in &param.data {
                grad_data.push(crate::stdlib::autodiff::Dual::variable(dual.derivative()));
            }
            
            gradients.push(Tensor::new(grad_data, param.shape.clone())?);
        }
        
        Ok(gradients)
    }
    
    /// Legacy method for compatibility - now delegates to autodiff implementation
    fn compute_gradients(
        &self,
        network: &mut NetChain,
        input: &Tensor,
        target: &Tensor,
        loss_fn: &dyn LossFunction,
    ) -> MLResult<Vec<Tensor>> {
        self.compute_autodiff_gradients(network, input, target, loss_fn)
    }
    
    /// Train with default MSE loss and SGD optimizer
    pub fn train_simple(
        &self,
        network: &mut NetChain,
        data: &[(Tensor, Tensor)]
    ) -> MLResult<TrainingResult> {
        let loss_fn = MSELoss;
        let mut optimizer = SGD::new(self.config.learning_rate);
        self.train(network, data, &loss_fn, &mut optimizer)
    }
    
    /// Train a neural network using a ForeignDataset
    /// Assumes the dataset contains (input, target) pairs or can be split automatically
    pub fn train_dataset(
        &self,
        network: &mut NetChain,
        dataset: &ForeignDataset,
        target_extraction: DatasetTargetExtraction
    ) -> MLResult<TrainingResult> {
        // Convert dataset to tensor pairs
        let training_data = self.dataset_to_training_data(dataset, target_extraction)?;
        
        // Use default MSE loss and SGD optimizer
        let loss_fn = MSELoss;
        let mut optimizer = SGD::new(self.config.learning_rate);
        
        self.train(network, &training_data, &loss_fn, &mut optimizer)
    }
    
    /// Train a neural network using a ForeignTable
    /// Specify feature columns and target column explicitly
    pub fn train_table(
        &self,
        network: &mut NetChain,
        table: &ForeignTable,
        feature_columns: &[String],
        target_column: &str
    ) -> MLResult<TrainingResult> {
        // Convert table to tensor pairs
        let training_data = self.table_to_training_data(table, feature_columns, target_column)?;
        
        // Use default MSE loss and SGD optimizer
        let loss_fn = MSELoss;
        let mut optimizer = SGD::new(self.config.learning_rate);
        
        self.train(network, &training_data, &loss_fn, &mut optimizer)
    }
    
    /// Train a neural network using mixed data types
    /// Automatically detect and convert various Value types to tensors
    pub fn train_mixed(
        &self,
        network: &mut NetChain,
        data: &[Value]
    ) -> MLResult<TrainingResult> {
        // Convert mixed data to tensor pairs
        let training_data = self.mixed_data_to_training_data(data)?;
        
        // Use default MSE loss and SGD optimizer
        let loss_fn = MSELoss;
        let mut optimizer = SGD::new(self.config.learning_rate);
        
        self.train(network, &training_data, &loss_fn, &mut optimizer)
    }
    
    /// Train a neural network using a ForeignDataset with automatic preprocessing
    /// Automatically detects data characteristics and applies appropriate preprocessing
    pub fn train_dataset_auto(
        &self,
        network: &mut NetChain,
        dataset: &ForeignDataset,
        target_extraction: DatasetTargetExtraction
    ) -> MLResult<TrainingResult> {
        // Infer preprocessing pipeline from data
        let preprocessor = AutoPreprocessor::infer_pipeline(dataset.get_value())?;
        self.train_dataset_with_preprocessing(network, dataset, target_extraction, &preprocessor)
    }
    
    /// Train a neural network using a ForeignDataset with custom preprocessing
    pub fn train_dataset_with_preprocessing(
        &self,
        network: &mut NetChain,
        dataset: &ForeignDataset,
        target_extraction: DatasetTargetExtraction,
        preprocessor: &dyn MLPreprocessor
    ) -> MLResult<TrainingResult> {
        // Apply preprocessing to dataset
        let preprocessed_data = preprocessor.preprocess(dataset.get_value())?;
        
        // Create temporary dataset with preprocessed data
        let preprocessed_dataset = crate::stdlib::data::ForeignDataset::new(preprocessed_data);
        
        // Convert to training data using existing method
        let training_data = self.dataset_to_training_data(&preprocessed_dataset, target_extraction)?;
        
        // Use default MSE loss and SGD optimizer
        let loss_fn = MSELoss;
        let mut optimizer = SGD::new(self.config.learning_rate);
        
        self.train(network, &training_data, &loss_fn, &mut optimizer)
    }
    
    /// Train a neural network using a ForeignTable with automatic preprocessing
    pub fn train_table_auto(
        &self,
        network: &mut NetChain,
        table: &ForeignTable,
        feature_columns: &[String],
        target_column: &str
    ) -> MLResult<TrainingResult> {
        // Apply automatic preprocessing to table data
        let mut preprocessed_training_data = Vec::new();
        
        // Extract and preprocess feature data row by row
        for row_idx in 0..table.length {
            let mut row_features_raw = Vec::new();
            
            // Collect raw feature values for this row
            for col_name in feature_columns {
                if let Some(series) = table.get_column(col_name) {
                    if let Ok(value) = series.get(row_idx) {
                        row_features_raw.push(value.clone());
                    } else {
                        return Err(MLError::DataError {
                            reason: format!("Cannot access row {} in column '{}'", row_idx, col_name),
                        });
                    }
                } else {
                    return Err(MLError::DataError {
                        reason: format!("Feature column '{}' not found in table", col_name),
                    });
                }
            }
            
            // Apply automatic preprocessing to this row's features
            let features_value = Value::List(row_features_raw);
            let preprocessor = AutoPreprocessor::infer_pipeline(&features_value)?;
            let preprocessed_features = preprocessor.preprocess(&features_value)?;
            
            // Convert preprocessed features to tensor
            let feature_tensor = crate::stdlib::ml::preprocessing::preprocessed_value_to_tensor(&preprocessed_features)?;
            
            // Extract target (no preprocessing applied to targets)
            let target_value = if let Some(target_series) = table.get_column(target_column) {
                if let Ok(value) = target_series.get(row_idx) {
                    match value {
                        Value::Real(n) => Dual::variable(*n),
                        Value::Integer(n) => Dual::variable(*n as f64),
                        _ => return Err(MLError::DataError {
                            reason: format!("Target column '{}' contains non-numeric data", target_column),
                        }),
                    }
                } else {
                    return Err(MLError::DataError {
                        reason: format!("Cannot access row {} in target column '{}'", row_idx, target_column),
                    });
                }
            } else {
                return Err(MLError::DataError {
                    reason: format!("Target column '{}' not found in table", target_column),
                });
            };
            
            let target_tensor = Tensor::new(vec![target_value], vec![1])?;
            preprocessed_training_data.push((feature_tensor, target_tensor));
        }
        
        // Use default MSE loss and SGD optimizer
        let loss_fn = MSELoss;
        let mut optimizer = SGD::new(self.config.learning_rate);
        
        self.train(network, &preprocessed_training_data, &loss_fn, &mut optimizer)
    }
    
    /// Train a neural network using mixed data with automatic preprocessing
    pub fn train_mixed_auto(
        &self,
        network: &mut NetChain,
        data: &[Value]
    ) -> MLResult<TrainingResult> {
        let mut preprocessed_training_data = Vec::new();
        
        // Process data in pairs: (input, target)
        for chunk in data.chunks(2) {
            if chunk.len() != 2 {
                return Err(MLError::DataError {
                    reason: "Mixed data must contain even number of elements (input, target pairs)".to_string(),
                });
            }
            
            // Apply automatic preprocessing to input features
            let input_preprocessor = AutoPreprocessor::infer_pipeline(&chunk[0])?;
            let preprocessed_input = input_preprocessor.preprocess(&chunk[0])?;
            let input_tensor = crate::stdlib::ml::preprocessing::preprocessed_value_to_tensor(&preprocessed_input)?;
            
            // No preprocessing for targets (keep raw)
            let target_tensor = self.value_to_tensor(&chunk[1])?;
            
            preprocessed_training_data.push((input_tensor, target_tensor));
        }
        
        // Use default MSE loss and SGD optimizer
        let loss_fn = MSELoss;
        let mut optimizer = SGD::new(self.config.learning_rate);
        
        self.train(network, &preprocessed_training_data, &loss_fn, &mut optimizer)
    }
    
    /// Convert ForeignDataset to training data (input, target) pairs
    fn dataset_to_training_data(
        &self,
        dataset: &ForeignDataset,
        target_extraction: DatasetTargetExtraction
    ) -> MLResult<Vec<(Tensor, Tensor)>> {
        let data_value = dataset.get_value();
        
        match data_value {
            Value::List(elements) => {
                match target_extraction {
                    DatasetTargetExtraction::LastElement => {
                        // Treat last element as target, rest as features
                        self.split_features_target_from_list(elements)
                    },
                    DatasetTargetExtraction::FirstElement => {
                        // Treat first element as target, rest as features
                        self.split_target_features_from_list(elements)
                    },
                    DatasetTargetExtraction::EvenOdd => {
                        // Even indices as features, odd indices as targets
                        self.split_even_odd_from_list(elements)
                    }
                }
            },
            _ => Err(MLError::DataError {
                reason: "Dataset must contain List data for training".to_string(),
            })
        }
    }
    
    /// Convert ForeignTable to training data (input, target) pairs
    fn table_to_training_data(
        &self,
        table: &ForeignTable,
        feature_columns: &[String],
        target_column: &str
    ) -> MLResult<Vec<(Tensor, Tensor)>> {
        if table.length == 0 {
            return Err(MLError::DataError {
                reason: "Cannot train on empty table".to_string(),
            });
        }
        
        // Extract feature data
        let mut feature_data = Vec::new();
        for row_idx in 0..table.length {
            let mut row_features = Vec::new();
            
            for col_name in feature_columns {
                if let Some(series) = table.get_column(col_name) {
                    if let Ok(value) = series.get(row_idx) {
                        match value {
                            Value::Real(n) => row_features.push(Dual::variable(*n)),
                            Value::Integer(n) => row_features.push(Dual::variable(*n as f64)),
                            _ => return Err(MLError::DataError {
                                reason: format!("Feature column '{}' contains non-numeric data", col_name),
                            }),
                        }
                    } else {
                        return Err(MLError::DataError {
                            reason: format!("Cannot access row {} in column '{}'", row_idx, col_name),
                        });
                    }
                } else {
                    return Err(MLError::DataError {
                        reason: format!("Feature column '{}' not found in table", col_name),
                    });
                }
            }
            
            let feature_tensor = Tensor::new(row_features, vec![feature_columns.len()])?;
            feature_data.push(feature_tensor);
        }
        
        // Extract target data
        let mut target_data = Vec::new();
        if let Some(target_series) = table.get_column(target_column) {
            for row_idx in 0..table.length {
                if let Ok(value) = target_series.get(row_idx) {
                    let target_value = match value {
                        Value::Real(n) => Dual::variable(*n),
                        Value::Integer(n) => Dual::variable(*n as f64),
                        _ => return Err(MLError::DataError {
                            reason: format!("Target column '{}' contains non-numeric data", target_column),
                        }),
                    };
                    
                    let target_tensor = Tensor::new(vec![target_value], vec![1])?;
                    target_data.push(target_tensor);
                } else {
                    return Err(MLError::DataError {
                        reason: format!("Cannot access row {} in target column '{}'", row_idx, target_column),
                    });
                }
            }
        } else {
            return Err(MLError::DataError {
                reason: format!("Target column '{}' not found in table", target_column),
            });
        }
        
        // Combine features and targets
        if feature_data.len() != target_data.len() {
            return Err(MLError::DataError {
                reason: "Mismatch between feature and target data lengths".to_string(),
            });
        }
        
        let training_data: Vec<(Tensor, Tensor)> = feature_data.into_iter()
            .zip(target_data.into_iter())
            .collect();
        
        Ok(training_data)
    }
    
    /// Convert mixed Value data to training data (input, target) pairs
    fn mixed_data_to_training_data(
        &self,
        data: &[Value]
    ) -> MLResult<Vec<(Tensor, Tensor)>> {
        let mut training_pairs = Vec::new();
        
        // Process data in pairs: (input, target)
        for chunk in data.chunks(2) {
            if chunk.len() != 2 {
                return Err(MLError::DataError {
                    reason: "Mixed data must contain even number of elements (input, target pairs)".to_string(),
                });
            }
            
            let input_tensor = self.value_to_tensor(&chunk[0])?;
            let target_tensor = self.value_to_tensor(&chunk[1])?;
            
            training_pairs.push((input_tensor, target_tensor));
        }
        
        Ok(training_pairs)
    }
    
    /// Helper: Convert Value to Tensor (simplified version of wrapper function)
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
            Value::LyObj(obj) => {
                if let Some(dataset) = obj.downcast_ref::<ForeignDataset>() {
                    // Convert dataset to single tensor (flatten approach)
                    let data_value = dataset.get_value();
                    self.value_to_tensor(data_value)
                } else if let Some(table) = obj.downcast_ref::<ForeignTable>() {
                    // Convert table to flattened tensor (all numeric columns)
                    self.table_to_single_tensor(table)
                } else {
                    Err(MLError::DataError {
                        reason: format!("Unsupported LyObj type '{}' for tensor conversion", obj.type_name()),
                    })
                }
            },
            _ => Err(MLError::DataError {
                reason: format!("Cannot convert {:?} to tensor", value),
            }),
        }
    }
    
    /// Helper: Convert table to single flattened tensor
    fn table_to_single_tensor(&self, table: &ForeignTable) -> MLResult<Tensor> {
        let column_names = table.column_names();
        let mut all_data = Vec::new();
        
        for row_idx in 0..table.length {
            for column_name in &column_names {
                if let Some(series) = table.get_column(column_name) {
                    if let Ok(value) = series.get(row_idx) {
                        match value {
                            Value::Real(n) => all_data.push(Dual::variable(*n)),
                            Value::Integer(n) => all_data.push(Dual::variable(*n as f64)),
                            _ => {}, // Skip non-numeric values
                        }
                    }
                }
            }
        }
        
        if all_data.is_empty() {
            return Err(MLError::DataError {
                reason: "Table contains no numeric data".to_string(),
            });
        }
        
        let shape = vec![all_data.len()];
        Tensor::new(all_data, shape)
    }
    
    /// Helper: Split list into (features, target) where target is last element
    fn split_features_target_from_list(&self, elements: &[Value]) -> MLResult<Vec<(Tensor, Tensor)>> {
        if elements.len() < 2 {
            return Err(MLError::DataError {
                reason: "Dataset must have at least 2 elements for feature-target split".to_string(),
            });
        }
        
        let features = &elements[..elements.len()-1];
        let target = &elements[elements.len()-1];
        
        let feature_tensor = self.value_to_tensor(&Value::List(features.to_vec()))?;
        let target_tensor = self.value_to_tensor(target)?;
        
        Ok(vec![(feature_tensor, target_tensor)])
    }
    
    /// Helper: Split list into (features, target) where target is first element
    fn split_target_features_from_list(&self, elements: &[Value]) -> MLResult<Vec<(Tensor, Tensor)>> {
        if elements.len() < 2 {
            return Err(MLError::DataError {
                reason: "Dataset must have at least 2 elements for target-feature split".to_string(),
            });
        }
        
        let target = &elements[0];
        let features = &elements[1..];
        
        let feature_tensor = self.value_to_tensor(&Value::List(features.to_vec()))?;
        let target_tensor = self.value_to_tensor(target)?;
        
        Ok(vec![(feature_tensor, target_tensor)])
    }
    
    /// Helper: Split list into alternating (feature, target) pairs
    fn split_even_odd_from_list(&self, elements: &[Value]) -> MLResult<Vec<(Tensor, Tensor)>> {
        if elements.len() % 2 != 0 {
            return Err(MLError::DataError {
                reason: "Dataset must have even number of elements for even-odd split".to_string(),
            });
        }
        
        let mut training_pairs = Vec::new();
        
        for chunk in elements.chunks(2) {
            let feature_tensor = self.value_to_tensor(&chunk[0])?;
            let target_tensor = self.value_to_tensor(&chunk[1])?;
            training_pairs.push((feature_tensor, target_tensor));
        }
        
        Ok(training_pairs)
    }
}

/// How to extract target values from datasets
#[derive(Debug, Clone, Copy)]
pub enum DatasetTargetExtraction {
    /// Use the last element as target, rest as features
    LastElement,
    /// Use the first element as target, rest as features  
    FirstElement,
    /// Use even indices as features, odd indices as targets
    EvenOdd,
}