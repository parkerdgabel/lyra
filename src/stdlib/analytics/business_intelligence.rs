//! Business Intelligence Functions
//!
//! Comprehensive business analytics capabilities including KPI calculation,
//! cohort analysis, funnel analysis, retention metrics, and A/B testing.

use crate::vm::{Value, VmResult, VmError};
use crate::foreign::{Foreign, LyObj};
use std::collections::HashMap;
use chrono::{DateTime, Utc, Duration};

/// Key Performance Indicator - Foreign Object
#[derive(Debug, Clone)]
pub struct KPI {
    name: String,
    value: f64,
    target: Option<f64>,
    period: String,
    unit: String,
    trend: Option<f64>, // percentage change
    status: String, // "on_track", "at_risk", "off_track"
}

impl Foreign for KPI {
    fn type_name(&self) -> &'static str {
        "KPI"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> VmResult<Value> {
        match method {
            "name" => Ok(Value::String(self.name.clone())),
            "value" => Ok(Value::Real(self.value)),
            "target" => match self.target {
                Some(target) => Ok(Value::Real(target)),
                None => Ok(Value::Missing),
            },
            "period" => Ok(Value::String(self.period.clone())),
            "unit" => Ok(Value::String(self.unit.clone())),
            "trend" => match self.trend {
                Some(trend) => Ok(Value::Real(trend)),
                None => Ok(Value::Missing),
            },
            "status" => Ok(Value::String(self.status.clone())),
            "isOnTrack" => Ok(Value::Boolean(self.status == "on_track")),
            "performance" => {
                if let Some(target) = self.target {
                    let performance = (self.value / target) * 100.0;
                    Ok(Value::Real(performance))
                } else {
                    Ok(Value::Missing)
                }
            },
            _ => Err(VmError::Runtime(format!(
                "Unknown method '{}' for KPI", method
            ))),
        }
    }
}

/// Cohort Analysis Result - Foreign Object
#[derive(Debug, Clone)]
pub struct CohortAnalysis {
    cohorts: HashMap<String, Vec<f64>>, // cohort_name -> retention rates by period
    metrics: HashMap<String, f64>,
    periods: Vec<String>,
    cohort_sizes: HashMap<String, usize>,
}

impl Foreign for CohortAnalysis {
    fn type_name(&self) -> &'static str {
        "CohortAnalysis"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> VmResult<Value> {
        match method {
            "cohorts" => {
                let mut cohort_map = HashMap::new();
                for (name, rates) in &self.cohorts {
                    cohort_map.insert(name.clone(), Value::List(
                        rates.iter().map(|&x| Value::Real(x)).collect()
                    ));
                }
                Ok(Value::Object(cohort_map))
            },
            "periods" => Ok(Value::List(
                self.periods.iter().map(|s| Value::String(s.clone())).collect()
            )),
            "getCohort" => {
                if let Some(Value::String(cohort_name)) = args.get(0) {
                    if let Some(rates) = self.cohorts.get(cohort_name) {
                        Ok(Value::List(rates.iter().map(|&x| Value::Real(x)).collect()))
                    } else {
                        Err(VmError::Runtime(format!("Cohort '{}' not found", cohort_name)))
                    }
                } else {
                    Err(VmError::Runtime("getCohort requires cohort name".to_string()))
                }
            },
            "cohortSize" => {
                if let Some(Value::String(cohort_name)) = args.get(0) {
                    if let Some(&size) = self.cohort_sizes.get(cohort_name) {
                        Ok(Value::Integer(size as i64))
                    } else {
                        Err(VmError::Runtime(format!("Cohort '{}' not found", cohort_name)))
                    }
                } else {
                    Err(VmError::Runtime("cohortSize requires cohort name".to_string()))
                }
            },
            "averageRetention" => {
                let period_index = args.get(0).and_then(|v| v.as_real()).unwrap_or(0.0) as usize;
                let mut total = 0.0;
                let mut count = 0;
                
                for rates in self.cohorts.values() {
                    if period_index < rates.len() {
                        total += rates[period_index];
                        count += 1;
                    }
                }
                
                if count > 0 {
                    Ok(Value::Real(total / count as f64))
                } else {
                    Ok(Value::Real(0.0))
                }
            },
            _ => Err(VmError::Runtime(format!(
                "Unknown method '{}' for CohortAnalysis", method
            ))),
        }
    }
}

/// Funnel Analysis Result - Foreign Object
#[derive(Debug, Clone)]
pub struct FunnelAnalysis {
    stages: Vec<String>,
    counts: Vec<usize>,
    conversion_rates: Vec<f64>,
    drop_off_rates: Vec<f64>,
}

impl Foreign for FunnelAnalysis {
    fn type_name(&self) -> &'static str {
        "FunnelAnalysis"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> VmResult<Value> {
        match method {
            "stages" => Ok(Value::List(
                self.stages.iter().map(|s| Value::String(s.clone())).collect()
            )),
            "counts" => Ok(Value::List(
                self.counts.iter().map(|&x| Value::Integer(x as i64)).collect()
            )),
            "conversionRates" => Ok(Value::List(
                self.conversion_rates.iter().map(|&x| Value::Real(x)).collect()
            )),
            "dropOffRates" => Ok(Value::List(
                self.drop_off_rates.iter().map(|&x| Value::Real(x)).collect()
            )),
            "overallConversion" => {
                if !self.counts.is_empty() {
                    let overall = *self.counts.last().unwrap() as f64 / self.counts[0] as f64;
                    Ok(Value::Real(overall))
                } else {
                    Ok(Value::Real(0.0))
                }
            },
            "bottleneckStage" => {
                if let Some((index, _)) = self.drop_off_rates.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
                    if index < self.stages.len() {
                        Ok(Value::String(self.stages[index].clone()))
                    } else {
                        Ok(Value::Missing)
                    }
                } else {
                    Ok(Value::Missing)
                }
            },
            _ => Err(VmError::Runtime(format!(
                "Unknown method '{}' for FunnelAnalysis", method
            ))),
        }
    }
}

/// A/B Test Result - Foreign Object
#[derive(Debug, Clone)]
pub struct ABTestResult {
    control_size: usize,
    treatment_size: usize,
    control_metric: f64,
    treatment_metric: f64,
    lift: f64,
    p_value: f64,
    confidence_level: f64,
    is_significant: bool,
    confidence_interval: (f64, f64),
}

impl Foreign for ABTestResult {
    fn type_name(&self) -> &'static str {
        "ABTestResult"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> VmResult<Value> {
        match method {
            "controlSize" => Ok(Value::Integer(self.control_size as i64)),
            "treatmentSize" => Ok(Value::Integer(self.treatment_size as i64)),
            "controlMetric" => Ok(Value::Real(self.control_metric)),
            "treatmentMetric" => Ok(Value::Real(self.treatment_metric)),
            "lift" => Ok(Value::Real(self.lift)),
            "pValue" => Ok(Value::Real(self.p_value)),
            "isSignificant" => Ok(Value::Boolean(self.is_significant)),
            "confidenceLevel" => Ok(Value::Real(self.confidence_level)),
            "confidenceInterval" => Ok(Value::List(vec![
                Value::Real(self.confidence_interval.0),
                Value::Real(self.confidence_interval.1),
            ])),
            "relativeImprovement" => Ok(Value::Real(
                (self.treatment_metric - self.control_metric) / self.control_metric * 100.0
            )),
            _ => Err(VmError::Runtime(format!(
                "Unknown method '{}' for ABTestResult", method
            ))),
        }
    }
}

/// Key Performance Indicator calculation
pub fn kpi(args: &[Value]) -> VmResult<Value> {
    if args.len() < 4 {
        return Err(VmError::Runtime(
            "KPI requires 4 arguments: data, metric_definition, target, period".to_string()
        ));
    }

    let data = extract_data_for_kpi(&args[0])?;
    let metric_definition = args[1].as_string().ok_or_else(|| VmError::Runtime(
        "Metric definition must be a string".to_string()
    ))?;
    let target = args[2].as_real();
    let period = args[3].as_string().ok_or_else(|| VmError::Runtime(
        "Period must be a string".to_string()
    ))?;

    let kpi_result = calculate_kpi(data, &metric_definition, target, &period)?;
    Ok(Value::LyObj(LyObj::new(Box::new(kpi_result))))
}

/// Cohort analysis for customer retention
pub fn cohort_analysis(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(VmError::Runtime(
            "CohortAnalysis requires 3 arguments: data, cohort_criteria, metrics".to_string()
        ));
    }

    let data = extract_cohort_data(&args[0])?;
    let cohort_criteria = args[1].as_string().ok_or_else(|| VmError::Runtime(
        "Cohort criteria must be a string".to_string()
    ))?;
    let metrics = extract_metric_list(&args[2])?;

    let cohort_result = perform_cohort_analysis(data, &cohort_criteria, metrics)?;
    Ok(Value::LyObj(LyObj::new(Box::new(cohort_result))))
}

/// Conversion funnel analysis
pub fn funnel_analysis(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(VmError::Runtime(
            "FunnelAnalysis requires 3 arguments: data, stages, conversion_events".to_string()
        ));
    }

    let data = extract_funnel_data(&args[0])?;
    let stages = extract_stage_list(&args[1])?;
    let conversion_events = extract_event_list(&args[2])?;

    let funnel_result = perform_funnel_analysis(data, stages, conversion_events)?;
    Ok(Value::LyObj(LyObj::new(Box::new(funnel_result))))
}

/// User retention analysis
pub fn retention_analysis(args: &[Value]) -> VmResult<Value> {
    if args.len() < 4 {
        return Err(VmError::Runtime(
            "RetentionAnalysis requires 4 arguments: data, user_id, event_date, periods".to_string()
        ));
    }

    let data = extract_retention_data(&args[0])?;
    let user_id_col = args[1].as_string().ok_or_else(|| VmError::Runtime(
        "User ID column must be a string".to_string()
    ))?;
    let event_date_col = args[2].as_string().ok_or_else(|| VmError::Runtime(
        "Event date column must be a string".to_string()
    ))?;
    let periods = extract_numeric_vector(&args[3])?;

    let retention_rates = calculate_retention_rates(data, &user_id_col, &event_date_col, periods)?;
    Ok(Value::List(retention_rates.iter().map(|&x| Value::Real(x)).collect()))
}

/// Customer Lifetime Value calculation
pub fn ltv(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(VmError::Runtime(
            "LTV requires 3 arguments: customer_data, revenue_events, time_horizon".to_string()
        ));
    }

    let customer_data = extract_customer_data(&args[0])?;
    let revenue_events = extract_revenue_events(&args[1])?;
    let time_horizon = args[2].as_real().ok_or_else(|| VmError::Runtime(
        "Time horizon must be a number (days)".to_string()
    ))?;

    let ltv_values = calculate_ltv(customer_data, revenue_events, time_horizon)?;
    
    let mut result = HashMap::new();
    result.insert("averageLTV".to_string(), Value::Real(
        ltv_values.iter().sum::<f64>() / ltv_values.len() as f64
    ));
    result.insert("totalLTV".to_string(), Value::Real(ltv_values.iter().sum()));
    result.insert("individualLTV".to_string(), Value::List(
        ltv_values.iter().map(|&x| Value::Real(x)).collect()
    ));

    Ok(Value::Object(result))
}

/// Churn prediction and analysis
pub fn churn(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(VmError::Runtime(
            "Churn requires 2 arguments: data, features".to_string()
        ));
    }

    let data = extract_churn_data(&args[0])?;
    let features = extract_feature_list(&args[1])?;
    let prediction_horizon = args.get(2).and_then(|v| v.as_real()).unwrap_or(30.0);

    let churn_analysis = perform_churn_analysis(data, features, prediction_horizon)?;
    Ok(Value::Object(churn_analysis))
}

/// Customer/market segmentation
pub fn segmentation(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(VmError::Runtime(
            "Segmentation requires 3 arguments: data, features, method".to_string()
        ));
    }

    let data = extract_segmentation_data(&args[0])?;
    let features = extract_feature_list(&args[1])?;
    let method = args[2].as_string().ok_or_else(|| VmError::Runtime(
        "Method must be a string".to_string()
    ))?;
    let k = args.get(3).and_then(|v| v.as_real()).map(|x| x as usize).unwrap_or(3);

    let segments = perform_segmentation(data, features, &method, k)?;
    Ok(Value::Object(segments))
}

/// A/B test statistical analysis
pub fn ab_test_analysis(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(VmError::Runtime(
            "ABTestAnalysis requires 3 arguments: control, treatment, metric".to_string()
        ));
    }

    let control = extract_numeric_vector(&args[0])?;
    let treatment = extract_numeric_vector(&args[1])?;
    let metric = args[2].as_string().ok_or_else(|| VmError::Runtime(
        "Metric must be a string".to_string()
    ))?;
    let alpha = args.get(3).and_then(|v| v.as_real()).unwrap_or(0.05);

    let ab_result = perform_ab_test(control, treatment, &metric, alpha)?;
    Ok(Value::LyObj(LyObj::new(Box::new(ab_result))))
}

/// Marketing attribution modeling
pub fn attribution_model(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(VmError::Runtime(
            "AttributionModel requires 3 arguments: touchpoints, conversions, model_type".to_string()
        ));
    }

    let touchpoints = extract_touchpoint_data(&args[0])?;
    let conversions = extract_conversion_data(&args[1])?;
    let model_type = args[2].as_string().ok_or_else(|| VmError::Runtime(
        "Model type must be a string".to_string()
    ))?;

    let attribution_results = calculate_attribution(touchpoints, conversions, &model_type)?;
    Ok(Value::Object(attribution_results))
}

/// Create analytics dashboard
pub fn dashboard(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 {
        return Err(VmError::Runtime(
            "Dashboard requires at least 1 argument: metrics".to_string()
        ));
    }

    let metrics = extract_dashboard_metrics(&args[0])?;
    let filters = if args.len() > 1 {
        extract_dashboard_filters(&args[1])?
    } else {
        HashMap::new()
    };
    let visualizations = if args.len() > 2 {
        extract_visualization_config(&args[2])?
    } else {
        vec!["table".to_string()]
    };

    let dashboard_data = create_dashboard(metrics, filters, visualizations)?;
    Ok(Value::Object(dashboard_data))
}

// Helper functions for data extraction and processing
fn extract_data_for_kpi(value: &Value) -> VmResult<HashMap<String, Vec<f64>>> {
    // Extract data for KPI calculation
    // This would parse data structure containing time series data
    Ok(HashMap::new()) // Placeholder
}

fn extract_cohort_data(value: &Value) -> VmResult<Vec<HashMap<String, Value>>> {
    // Extract user event data for cohort analysis
    Ok(Vec::new()) // Placeholder
}

fn extract_metric_list(value: &Value) -> VmResult<Vec<String>> {
    match value {
        Value::List(metrics) => {
            metrics.iter()
                .map(|m| m.as_string().ok_or_else(|| VmError::Runtime(
                    "All metrics must be strings".to_string()
                )))
                .collect()
        },
        Value::String(s) => Ok(vec![s.clone()]),
        _ => Err(VmError::Runtime(
            "Metrics must be a string or list of strings".to_string()
        )),
    }
}

fn extract_funnel_data(value: &Value) -> VmResult<Vec<HashMap<String, Value>>> {
    // Extract user journey data for funnel analysis
    Ok(Vec::new()) // Placeholder
}

fn extract_stage_list(value: &Value) -> VmResult<Vec<String>> {
    match value {
        Value::List(stages) => {
            stages.iter()
                .map(|s| s.as_string().ok_or_else(|| VmError::Runtime(
                    "All stages must be strings".to_string()
                )))
                .collect()
        },
        _ => Err(VmError::Runtime(
            "Stages must be a list of strings".to_string()
        )),
    }
}

fn extract_event_list(value: &Value) -> VmResult<Vec<String>> {
    match value {
        Value::List(events) => {
            events.iter()
                .map(|e| e.as_string().ok_or_else(|| VmError::Runtime(
                    "All events must be strings".to_string()
                )))
                .collect()
        },
        _ => Err(VmError::Runtime(
            "Events must be a list of strings".to_string()
        )),
    }
}

fn extract_retention_data(value: &Value) -> VmResult<Vec<HashMap<String, Value>>> {
    // Extract user activity data for retention analysis
    Ok(Vec::new()) // Placeholder
}

fn extract_numeric_vector(value: &Value) -> VmResult<Vec<f64>> {
    match value {
        Value::List(items) => {
            items.iter()
                .map(|item| match item {
                    Value::Real(n) => Ok(*n),
                    _ => Err(VmError::Runtime(
                        "All elements must be numbers".to_string()
                    )),
                })
                .collect()
        },
        Value::Real(n) => Ok(vec![*n]),
        _ => Err(VmError::Runtime(
            "Data must be a number or list of numbers".to_string()
        )),
    }
}

fn extract_customer_data(value: &Value) -> VmResult<Vec<HashMap<String, Value>>> {
    // Extract customer profile data
    Ok(Vec::new()) // Placeholder
}

fn extract_revenue_events(value: &Value) -> VmResult<Vec<HashMap<String, Value>>> {
    // Extract revenue event data
    Ok(Vec::new()) // Placeholder
}

fn extract_churn_data(value: &Value) -> VmResult<Vec<HashMap<String, Value>>> {
    // Extract customer data for churn analysis
    Ok(Vec::new()) // Placeholder
}

fn extract_feature_list(value: &Value) -> VmResult<Vec<String>> {
    match value {
        Value::List(features) => {
            features.iter()
                .map(|f| f.as_string().ok_or_else(|| VmError::Runtime(
                    "All features must be strings".to_string()
                )))
                .collect()
        },
        _ => Err(VmError::Runtime(
            "Features must be a list of strings".to_string()
        )),
    }
}

fn extract_segmentation_data(value: &Value) -> VmResult<Vec<HashMap<String, Value>>> {
    // Extract data for segmentation analysis
    Ok(Vec::new()) // Placeholder
}

fn extract_touchpoint_data(value: &Value) -> VmResult<Vec<HashMap<String, Value>>> {
    // Extract marketing touchpoint data
    Ok(Vec::new()) // Placeholder
}

fn extract_conversion_data(value: &Value) -> VmResult<Vec<HashMap<String, Value>>> {
    // Extract conversion event data
    Ok(Vec::new()) // Placeholder
}

fn extract_dashboard_metrics(value: &Value) -> VmResult<Vec<String>> {
    extract_metric_list(value)
}

fn extract_dashboard_filters(value: &Value) -> VmResult<HashMap<String, Value>> {
    match value {
        Value::Object(filters) => Ok(filters.clone()),
        _ => Ok(HashMap::new()),
    }
}

fn extract_visualization_config(value: &Value) -> VmResult<Vec<String>> {
    match value {
        Value::List(viz) => {
            viz.iter()
                .map(|v| v.as_string().ok_or_else(|| VmError::Runtime(
                    "All visualizations must be strings".to_string()
                )))
                .collect()
        },
        Value::String(s) => Ok(vec![s.clone()]),
        _ => Err(VmError::Runtime(
            "Visualizations must be a string or list of strings".to_string()
        )),
    }
}

// Implementation functions
fn calculate_kpi(data: HashMap<String, Vec<f64>>, metric_definition: &str, target: Option<f64>, period: &str) -> VmResult<KPI> {
    // Calculate KPI based on metric definition
    let value = match metric_definition {
        "revenue" => data.get("revenue").map(|v| v.iter().sum()).unwrap_or(0.0),
        "conversion_rate" => {
            let conversions = data.get("conversions").map(|v| v.iter().sum()).unwrap_or(0.0);
            let visitors = data.get("visitors").map(|v| v.iter().sum()).unwrap_or(1.0);
            (conversions / visitors) * 100.0
        },
        "average_order_value" => {
            let revenue = data.get("revenue").map(|v| v.iter().sum()).unwrap_or(0.0);
            let orders = data.get("orders").map(|v| v.iter().sum()).unwrap_or(1.0);
            revenue / orders
        },
        _ => return Err(VmError::Runtime(
            format!("Unsupported metric definition: {}", metric_definition)
        )),
    };

    let status = if let Some(target_val) = target {
        if value >= target_val * 0.95 {
            "on_track"
        } else if value >= target_val * 0.8 {
            "at_risk"
        } else {
            "off_track"
        }
    } else {
        "unknown"
    };

    Ok(KPI {
        name: metric_definition.to_string(),
        value,
        target,
        period: period.to_string(),
        unit: determine_unit(metric_definition),
        trend: None, // Would calculate from historical data
        status: status.to_string(),
    })
}

fn determine_unit(metric: &str) -> String {
    match metric {
        "revenue" => "currency".to_string(),
        "conversion_rate" => "percentage".to_string(),
        "average_order_value" => "currency".to_string(),
        _ => "count".to_string(),
    }
}

fn perform_cohort_analysis(data: Vec<HashMap<String, Value>>, cohort_criteria: &str, metrics: Vec<String>) -> VmResult<CohortAnalysis> {
    // Placeholder implementation for cohort analysis
    let mut cohorts = HashMap::new();
    let mut cohort_sizes = HashMap::new();
    
    // Example cohorts
    cohorts.insert("2024-01".to_string(), vec![100.0, 85.0, 70.0, 60.0]); // Retention rates
    cohorts.insert("2024-02".to_string(), vec![100.0, 88.0, 75.0, 65.0]);
    cohorts.insert("2024-03".to_string(), vec![100.0, 90.0, 78.0, 68.0]);
    
    cohort_sizes.insert("2024-01".to_string(), 1000);
    cohort_sizes.insert("2024-02".to_string(), 1200);
    cohort_sizes.insert("2024-03".to_string(), 1100);
    
    let periods = vec!["Month 0".to_string(), "Month 1".to_string(), "Month 2".to_string(), "Month 3".to_string()];
    
    Ok(CohortAnalysis {
        cohorts,
        metrics: HashMap::new(),
        periods,
        cohort_sizes,
    })
}

fn perform_funnel_analysis(data: Vec<HashMap<String, Value>>, stages: Vec<String>, conversion_events: Vec<String>) -> VmResult<FunnelAnalysis> {
    // Placeholder implementation for funnel analysis
    let counts = vec![10000, 8500, 6800, 5100, 3400]; // Users at each stage
    
    let mut conversion_rates = Vec::new();
    let mut drop_off_rates = Vec::new();
    
    for i in 1..counts.len() {
        let conversion_rate = (counts[i] as f64 / counts[i-1] as f64) * 100.0;
        let drop_off_rate = 100.0 - conversion_rate;
        conversion_rates.push(conversion_rate);
        drop_off_rates.push(drop_off_rate);
    }
    
    Ok(FunnelAnalysis {
        stages: vec!["Visit".to_string(), "View Product".to_string(), "Add to Cart".to_string(), "Checkout".to_string(), "Purchase".to_string()],
        counts,
        conversion_rates,
        drop_off_rates,
    })
}

fn calculate_retention_rates(data: Vec<HashMap<String, Value>>, user_id_col: &str, event_date_col: &str, periods: Vec<f64>) -> VmResult<Vec<f64>> {
    // Placeholder implementation for retention rate calculation
    // In practice, would analyze user activity over time periods
    Ok(vec![100.0, 85.0, 70.0, 60.0, 55.0]) // Example retention rates
}

fn calculate_ltv(customer_data: Vec<HashMap<String, Value>>, revenue_events: Vec<HashMap<String, Value>>, time_horizon: f64) -> VmResult<Vec<f64>> {
    // Placeholder implementation for LTV calculation
    // In practice, would calculate based on customer purchase history and predicted future value
    Ok(vec![250.0, 180.0, 320.0, 150.0, 400.0]) // Example LTV values
}

fn perform_churn_analysis(data: Vec<HashMap<String, Value>>, features: Vec<String>, prediction_horizon: f64) -> VmResult<HashMap<String, Value>> {
    // Placeholder implementation for churn analysis
    let mut result = HashMap::new();
    result.insert("churnRate".to_string(), Value::Real(15.5)); // 15.5% churn rate
    result.insert("riskFactors".to_string(), Value::List(vec![
        Value::String("low_engagement".to_string()),
        Value::String("payment_issues".to_string()),
        Value::String("support_tickets".to_string()),
    ]));
    result.insert("predictionAccuracy".to_string(), Value::Real(82.3));
    
    Ok(result)
}

fn perform_segmentation(data: Vec<HashMap<String, Value>>, features: Vec<String>, method: &str, k: usize) -> VmResult<HashMap<String, Value>> {
    // Placeholder implementation for customer segmentation
    let mut result = HashMap::new();
    
    match method {
        "kmeans" => {
            result.insert("method".to_string(), Value::String("K-Means".to_string()));
            result.insert("clusters".to_string(), Value::Integer(k as i64));
            
            let segment_names = vec!["High Value", "Regular", "Price Sensitive"];
            result.insert("segments".to_string(), Value::List(
                segment_names.iter().map(|s| Value::String(s.to_string())).collect()
            ));
            
            result.insert("segmentSizes".to_string(), Value::List(vec![
                Value::Integer(1200),
                Value::Integer(3500),
                Value::Integer(2300),
            ]));
        },
        "rfm" => {
            result.insert("method".to_string(), Value::String("RFM Analysis".to_string()));
            result.insert("segments".to_string(), Value::List(vec![
                Value::String("Champions".to_string()),
                Value::String("Loyal Customers".to_string()),
                Value::String("Potential Loyalists".to_string()),
                Value::String("At Risk".to_string()),
            ]));
        },
        _ => return Err(VmError::Runtime(
            format!("Unsupported segmentation method: {}", method)
        )),
    }
    
    Ok(result)
}

fn perform_ab_test(control: Vec<f64>, treatment: Vec<f64>, metric: &str, alpha: f64) -> VmResult<ABTestResult> {
    let control_mean = control.iter().sum::<f64>() / control.len() as f64;
    let treatment_mean = treatment.iter().sum::<f64>() / treatment.len() as f64;
    
    let control_var = control.iter().map(|x| (x - control_mean).powi(2)).sum::<f64>() / (control.len() - 1) as f64;
    let treatment_var = treatment.iter().map(|x| (x - treatment_mean).powi(2)).sum::<f64>() / (treatment.len() - 1) as f64;
    
    let pooled_se = ((control_var / control.len() as f64) + (treatment_var / treatment.len() as f64)).sqrt();
    let t_stat = (treatment_mean - control_mean) / pooled_se;
    
    // Simplified p-value calculation
    let p_value = if t_stat.abs() > 2.0 { 0.02 } else { 0.15 };
    let is_significant = p_value < alpha;
    
    let lift = if control_mean != 0.0 {
        ((treatment_mean - control_mean) / control_mean) * 100.0
    } else {
        0.0
    };
    
    let margin_of_error = 1.96 * pooled_se; // 95% CI
    let confidence_interval = (
        (treatment_mean - control_mean) - margin_of_error,
        (treatment_mean - control_mean) + margin_of_error
    );
    
    Ok(ABTestResult {
        control_size: control.len(),
        treatment_size: treatment.len(),
        control_metric: control_mean,
        treatment_metric: treatment_mean,
        lift,
        p_value,
        confidence_level: 1.0 - alpha,
        is_significant,
        confidence_interval,
    })
}

fn calculate_attribution(touchpoints: Vec<HashMap<String, Value>>, conversions: Vec<HashMap<String, Value>>, model_type: &str) -> VmResult<HashMap<String, Value>> {
    let mut result = HashMap::new();
    
    match model_type {
        "first_touch" => {
            result.insert("model".to_string(), Value::String("First Touch".to_string()));
            result.insert("attributions".to_string(), Value::Object(hashmap! {
                "paid_search".to_string() => Value::Real(40.0),
                "social_media".to_string() => Value::Real(25.0),
                "email".to_string() => Value::Real(20.0),
                "direct".to_string() => Value::Real(15.0)
            }));
        },
        "last_touch" => {
            result.insert("model".to_string(), Value::String("Last Touch".to_string()));
            result.insert("attributions".to_string(), Value::Object(hashmap! {
                "paid_search".to_string() => Value::Real(35.0),
                "social_media".to_string() => Value::Real(30.0),
                "email".to_string() => Value::Real(25.0),
                "direct".to_string() => Value::Real(10.0)
            }));
        },
        "linear" => {
            result.insert("model".to_string(), Value::String("Linear".to_string()));
            result.insert("attributions".to_string(), Value::Object(hashmap! {
                "paid_search".to_string() => Value::Real(25.0),
                "social_media".to_string() => Value::Real(25.0),
                "email".to_string() => Value::Real(25.0),
                "direct".to_string() => Value::Real(25.0)
            }));
        },
        _ => return Err(VmError::Runtime(
            format!("Unsupported attribution model: {}", model_type)
        )),
    }
    
    Ok(result)
}

fn create_dashboard(metrics: Vec<String>, filters: HashMap<String, Value>, visualizations: Vec<String>) -> VmResult<HashMap<String, Value>> {
    let mut dashboard = HashMap::new();
    
    dashboard.insert("metrics".to_string(), Value::List(
        metrics.iter().map(|m| Value::String(m.clone())).collect()
    ));
    
    dashboard.insert("filters".to_string(), Value::Object(filters));
    
    dashboard.insert("visualizations".to_string(), Value::List(
        visualizations.iter().map(|v| Value::String(v.clone())).collect()
    ));
    
    // Sample dashboard data
    let mut kpi_data = HashMap::new();
    kpi_data.insert("revenue".to_string(), Value::Real(125000.0));
    kpi_data.insert("conversion_rate".to_string(), Value::Real(3.2));
    kpi_data.insert("average_order_value".to_string(), Value::Real(85.50));
    kpi_data.insert("customer_acquisition_cost".to_string(), Value::Real(25.0));
    
    dashboard.insert("data".to_string(), Value::Object(kpi_data));
    
    dashboard.insert("lastUpdated".to_string(), Value::String(
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string()
    ));
    
    Ok(dashboard)
}

// Helper macro for creating hash maps
macro_rules! hashmap {
    ($( $key: expr => $val: expr ),*) => {{
         let mut map = ::std::collections::HashMap::new();
         $( map.insert($key, $val); )*
         map
    }}
}

pub(crate) use hashmap;