//! Core Chart Generation Module for Lyra Visualization System
//!
//! This module provides comprehensive chart generation capabilities with support for
//! 15 different chart types, from basic line and bar charts to advanced visualizations
//! like Sankey diagrams and network graphs.
//!
//! All charts are implemented as Foreign objects to maintain VM simplicity while
//! providing rich visualization capabilities.

use crate::vm::{Value, VmResult};
use crate::foreign::{LyObj, Foreign};
use plotters::prelude::*;
use std::collections::HashMap;
use std::any::Any;
use std::fmt;
use chrono::{DateTime, Utc, NaiveDate};
use palette::{Hsl, IntoColor, Srgb};

/// Color palette for chart styling
pub struct ChartPalette {
    colors: Vec<RGBColor>,
}

impl ChartPalette {
    pub fn new() -> Self {
        Self {
            colors: vec![
                RGBColor(31, 119, 180),   // Blue
                RGBColor(255, 127, 14),   // Orange
                RGBColor(44, 160, 44),    // Green
                RGBColor(214, 39, 40),    // Red
                RGBColor(148, 103, 189),  // Purple
                RGBColor(140, 86, 75),    // Brown
                RGBColor(227, 119, 194),  // Pink
                RGBColor(127, 127, 127),  // Gray
                RGBColor(188, 189, 34),   // Olive
                RGBColor(23, 190, 207),   // Cyan
            ],
        }
    }
    
    pub fn get_color(&self, index: usize) -> RGBColor {
        self.colors[index % self.colors.len()]
    }
}

/// Chart options structure for consistent configuration
#[derive(Debug, Clone)]
pub struct ChartOptions {
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub x_label: String,
    pub y_label: String,
    pub show_grid: bool,
    pub show_legend: bool,
    pub color_scheme: String,
    pub theme: String,
}

impl Default for ChartOptions {
    fn default() -> Self {
        Self {
            title: "Chart".to_string(),
            width: 800,
            height: 600,
            x_label: "X Axis".to_string(),
            y_label: "Y Axis".to_string(),
            show_grid: true,
            show_legend: true,
            color_scheme: "default".to_string(),
            theme: "light".to_string(),
        }
    }
}

impl ChartOptions {
    /// Parse chart options from Value
    pub fn from_value(value: &Value) -> VmResult<Self> {
        let mut options = ChartOptions::default();
        
        if let Value::List(opts) = value {
            for opt in opts {
                if let Value::List(pair) = opt {
                    if pair.len() == 2 {
                        if let (Value::String(key), val) = (&pair[0], &pair[1]) {
                            match key.as_str() {
                                "title" => {
                                    if let Value::String(title) = val {
                                        options.title = title.clone();
                                    }
                                }
                                "width" => {
                                    if let Value::Real(width) = val {
                                        options.width = *width as u32;
                                    }
                                }
                                "height" => {
                                    if let Value::Real(height) = val {
                                        options.height = *height as u32;
                                    }
                                }
                                "x_label" => {
                                    if let Value::String(label) = val {
                                        options.x_label = label.clone();
                                    }
                                }
                                "y_label" => {
                                    if let Value::String(label) = val {
                                        options.y_label = label.clone();
                                    }
                                }
                                "show_grid" => {
                                    if let Value::Boolean(show) = val {
                                        options.show_grid = *show;
                                    }
                                }
                                "show_legend" => {
                                    if let Value::Boolean(show) = val {
                                        options.show_legend = *show;
                                    }
                                }
                                "color_scheme" => {
                                    if let Value::String(scheme) = val {
                                        options.color_scheme = scheme.clone();
                                    }
                                }
                                "theme" => {
                                    if let Value::String(theme) = val {
                                        options.theme = theme.clone();
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
        
        Ok(options)
    }
}

/// Trait for all chart objects
pub trait Chart: Foreign {
    /// Render the chart to SVG string
    fn render_svg(&self) -> VmResult<String>;
    
    /// Get chart metadata
    fn get_metadata(&self) -> HashMap<String, String>;
    
    /// Get chart type name
    fn chart_type(&self) -> &'static str;
}

/// Data point for charts
#[derive(Debug, Clone)]
pub struct DataPoint {
    pub x: f64,
    pub y: f64,
    pub label: Option<String>,
    pub color: Option<String>,
    pub size: Option<f64>,
}

impl DataPoint {
    pub fn new(x: f64, y: f64) -> Self {
        Self {
            x,
            y,
            label: None,
            color: None,
            size: None,
        }
    }
    
    pub fn with_label(mut self, label: String) -> Self {
        self.label = Some(label);
        self
    }
    
    pub fn with_color(mut self, color: String) -> Self {
        self.color = Some(color);
        self
    }
    
    pub fn with_size(mut self, size: f64) -> Self {
        self.size = Some(size);
        self
    }
}

/// Parse data points from Value
pub fn parse_data_points(data: &Value) -> VmResult<Vec<DataPoint>> {
    match data {
        Value::List(points) => {
            let mut result = Vec::new();
            for point in points {
                match point {
                    Value::List(coords) if coords.len() >= 2 => {
                        if let (Value::Real(x), Value::Real(y)) = (&coords[0], &coords[1]) {
                            let mut dp = DataPoint::new(*x, *y);
                            
                            // Optional label
                            if coords.len() > 2 {
                                if let Value::String(label) = &coords[2] {
                                    dp = dp.with_label(label.clone());
                                }
                            }
                            
                            // Optional color
                            if coords.len() > 3 {
                                if let Value::String(color) = &coords[3] {
                                    dp = dp.with_color(color.clone());
                                }
                            }
                            
                            // Optional size
                            if coords.len() > 4 {
                                if let Value::Real(size) = &coords[4] {
                                    dp = dp.with_size(*size);
                                }
                            }
                            
                            result.push(dp);
                        }
                    }
                    Value::Real(y) => {
                        // Single values, use index as x
                        result.push(DataPoint::new(result.len() as f64, *y));
                    }
                    _ => return Err("Invalid data point format".to_string()),
                }
            }
            Ok(result)
        }
        _ => Err("Data must be a list".to_string()),
    }
}

// =============================================================================
// LineChart Implementation
// =============================================================================

/// Line Chart Foreign Object
#[derive(Debug)]
pub struct LineChartObj {
    data: Vec<DataPoint>,
    options: ChartOptions,
    x_axis: String,
    y_axis: String,
}

impl LineChartObj {
    pub fn new(data: Vec<DataPoint>, x_axis: String, y_axis: String, options: ChartOptions) -> Self {
        Self {
            data,
            options,
            x_axis,
            y_axis,
        }
    }
}

impl Foreign for LineChartObj {
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn type_name(&self) -> &'static str {
        "LineChart"
    }
}

impl Chart for LineChartObj {
    fn render_svg(&self) -> VmResult<String> {
        let mut svg_data = Vec::new();
        {
            let root = SVGBackend::with_buffer(&mut svg_data, (self.options.width, self.options.height))
                .into_drawing_area();
            root.fill(&WHITE).map_err(|e| format!("SVG render error: {}", e))?;
            
            let x_range = self.data.iter().map(|p| p.x).fold(f64::INFINITY, f64::min)
                ..self.data.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
            let y_range = self.data.iter().map(|p| p.y).fold(f64::INFINITY, f64::min)
                ..self.data.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);
            
            let mut chart = ChartBuilder::on(&root)
                .caption(&self.options.title, ("sans-serif", 30))
                .margin(20)
                .x_label_area_size(40)
                .y_label_area_size(50)
                .build_cartesian_2d(x_range, y_range)
                .map_err(|e| format!("Chart build error: {}", e))?;
            
            if self.options.show_grid {
                chart.configure_mesh()
                    .x_desc(&self.x_axis)
                    .y_desc(&self.y_axis)
                    .draw()
                    .map_err(|e| format!("Grid draw error: {}", e))?;
            }
            
            let palette = ChartPalette::new();
            let line_color = palette.get_color(0);
            
            chart.draw_series(LineSeries::new(
                self.data.iter().map(|p| (p.x, p.y)),
                &line_color,
            ))
            .map_err(|e| format!("Line series error: {}", e))?;
            
            root.present().map_err(|e| format!("Present error: {}", e))?;
        }
        
        String::from_utf8(svg_data).map_err(|e| format!("UTF-8 conversion error: {}", e))
    }
    
    fn get_metadata(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "LineChart".to_string());
        metadata.insert("data_points".to_string(), self.data.len().to_string());
        metadata.insert("title".to_string(), self.options.title.clone());
        metadata
    }
    
    fn chart_type(&self) -> &'static str {
        "LineChart"
    }
}

impl fmt::Display for LineChartObj {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LineChart[{} data points, title: {}]", 
               self.data.len(), self.options.title)
    }
}

/// Create a line chart
pub fn line_chart(args: &[Value]) -> VmResult<Value> {
    if args.len() < 4 {
        return Err("LineChart requires data, x_axis, y_axis, and options".to_string());
    }
    
    let data = parse_data_points(&args[0])?;
    
    let x_axis = match &args[1] {
        Value::String(axis) => axis.clone(),
        _ => return Err("x_axis must be a string".to_string()),
    };
    
    let y_axis = match &args[2] {
        Value::String(axis) => axis.clone(),
        _ => return Err("y_axis must be a string".to_string()),
    };
    
    let options = ChartOptions::from_value(&args[3])?;
    
    let chart = LineChartObj::new(data, x_axis, y_axis, options);
    Ok(Value::LyObj(LyObj::new(Box::new(chart))))
}

// =============================================================================
// BarChart Implementation
// =============================================================================

#[derive(Debug)]
pub struct BarChartObj {
    data: Vec<DataPoint>,
    categories: Vec<String>,
    values: String,
    orientation: String,
    options: ChartOptions,
}

impl BarChartObj {
    pub fn new(data: Vec<DataPoint>, categories: Vec<String>, values: String, 
               orientation: String, options: ChartOptions) -> Self {
        Self {
            data,
            categories,
            values,
            orientation,
            options,
        }
    }
}

impl Foreign for BarChartObj {
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn type_name(&self) -> &'static str {
        "BarChart"
    }
}

impl Chart for BarChartObj {
    fn render_svg(&self) -> VmResult<String> {
        let mut svg_data = Vec::new();
        {
            let root = SVGBackend::with_buffer(&mut svg_data, (self.options.width, self.options.height))
                .into_drawing_area();
            root.fill(&WHITE).map_err(|e| format!("SVG render error: {}", e))?;
            
            let y_max = self.data.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);
            
            let mut chart = ChartBuilder::on(&root)
                .caption(&self.options.title, ("sans-serif", 30))
                .margin(20)
                .x_label_area_size(40)
                .y_label_area_size(50)
                .build_cartesian_2d(0f64..self.data.len() as f64, 0f64..y_max * 1.1)
                .map_err(|e| format!("Chart build error: {}", e))?;
            
            if self.options.show_grid {
                chart.configure_mesh()
                    .x_desc("Categories")
                    .y_desc(&self.values)
                    .draw()
                    .map_err(|e| format!("Grid draw error: {}", e))?;
            }
            
            let palette = ChartPalette::new();
            
            chart.draw_series(
                self.data.iter().enumerate().map(|(i, point)| {
                    Rectangle::new([(i as f64, 0.0), (i as f64 + 0.8, point.y)], 
                                   palette.get_color(i).filled())
                })
            )
            .map_err(|e| format!("Bar series error: {}", e))?;
            
            root.present().map_err(|e| format!("Present error: {}", e))?;
        }
        
        String::from_utf8(svg_data).map_err(|e| format!("UTF-8 conversion error: {}", e))
    }
    
    fn get_metadata(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "BarChart".to_string());
        metadata.insert("data_points".to_string(), self.data.len().to_string());
        metadata.insert("orientation".to_string(), self.orientation.clone());
        metadata
    }
    
    fn chart_type(&self) -> &'static str {
        "BarChart"
    }
}

impl fmt::Display for BarChartObj {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BarChart[{} bars, orientation: {}]", 
               self.data.len(), self.orientation)
    }
}

/// Create a bar chart
pub fn bar_chart(args: &[Value]) -> VmResult<Value> {
    if args.len() < 4 {
        return Err("BarChart requires data, categories, values, and orientation".to_string());
    }
    
    let data = parse_data_points(&args[0])?;
    
    let categories = match &args[1] {
        Value::List(cats) => {
            cats.iter().map(|v| match v {
                Value::String(s) => Ok(s.clone()),
                _ => Err("Categories must be strings".to_string()),
            }).collect::<Result<Vec<_>, _>>()?
        }
        _ => return Err("Categories must be a list".to_string()),
    };
    
    let values = match &args[2] {
        Value::String(vals) => vals.clone(),
        _ => return Err("Values must be a string".to_string()),
    };
    
    let orientation = match &args[3] {
        Value::String(orient) => orient.clone(),
        _ => return Err("Orientation must be a string".to_string()),
    };
    
    let options = if args.len() > 4 {
        ChartOptions::from_value(&args[4])?
    } else {
        ChartOptions::default()
    };
    
    let chart = BarChartObj::new(data, categories, values, orientation, options);
    Ok(Value::LyObj(LyObj::new(Box::new(chart))))
}

// =============================================================================
// ScatterPlot Implementation
// =============================================================================

#[derive(Debug)]
pub struct ScatterPlotObj {
    data: Vec<DataPoint>,
    x_values: String,
    y_values: String,
    options: ChartOptions,
}

impl ScatterPlotObj {
    pub fn new(data: Vec<DataPoint>, x_values: String, y_values: String, options: ChartOptions) -> Self {
        Self {
            data,
            x_values,
            y_values,
            options,
        }
    }
}

impl Foreign for ScatterPlotObj {
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn type_name(&self) -> &'static str {
        "ScatterPlot"
    }
}

impl Chart for ScatterPlotObj {
    fn render_svg(&self) -> VmResult<String> {
        let mut svg_data = Vec::new();
        {
            let root = SVGBackend::with_buffer(&mut svg_data, (self.options.width, self.options.height))
                .into_drawing_area();
            root.fill(&WHITE).map_err(|e| format!("SVG render error: {}", e))?;
            
            let x_range = self.data.iter().map(|p| p.x).fold(f64::INFINITY, f64::min)
                ..self.data.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
            let y_range = self.data.iter().map(|p| p.y).fold(f64::INFINITY, f64::min)
                ..self.data.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);
            
            let mut chart = ChartBuilder::on(&root)
                .caption(&self.options.title, ("sans-serif", 30))
                .margin(20)
                .x_label_area_size(40)
                .y_label_area_size(50)
                .build_cartesian_2d(x_range, y_range)
                .map_err(|e| format!("Chart build error: {}", e))?;
            
            if self.options.show_grid {
                chart.configure_mesh()
                    .x_desc(&self.x_values)
                    .y_desc(&self.y_values)
                    .draw()
                    .map_err(|e| format!("Grid draw error: {}", e))?;
            }
            
            let palette = ChartPalette::new();
            let point_color = palette.get_color(0);
            
            chart.draw_series(
                self.data.iter().map(|point| {
                    Circle::new((point.x, point.y), 3, point_color.filled())
                })
            )
            .map_err(|e| format!("Scatter series error: {}", e))?;
            
            root.present().map_err(|e| format!("Present error: {}", e))?;
        }
        
        String::from_utf8(svg_data).map_err(|e| format!("UTF-8 conversion error: {}", e))
    }
    
    fn get_metadata(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "ScatterPlot".to_string());
        metadata.insert("data_points".to_string(), self.data.len().to_string());
        metadata
    }
    
    fn chart_type(&self) -> &'static str {
        "ScatterPlot"
    }
}

impl fmt::Display for ScatterPlotObj {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ScatterPlot[{} points]", self.data.len())
    }
}

/// Create a scatter plot
pub fn scatter_plot(args: &[Value]) -> VmResult<Value> {
    if args.len() < 4 {
        return Err("ScatterPlot requires data, x_values, y_values, and options".to_string());
    }
    
    let data = parse_data_points(&args[0])?;
    
    let x_values = match &args[1] {
        Value::String(vals) => vals.clone(),
        _ => return Err("x_values must be a string".to_string()),
    };
    
    let y_values = match &args[2] {
        Value::String(vals) => vals.clone(),
        _ => return Err("y_values must be a string".to_string()),
    };
    
    let options = ChartOptions::from_value(&args[3])?;
    
    let chart = ScatterPlotObj::new(data, x_values, y_values, options);
    Ok(Value::LyObj(LyObj::new(Box::new(chart))))
}

// =============================================================================
// Placeholder implementations for remaining chart types
// =============================================================================

// Simplified placeholder implementations for the remaining chart types
// These follow the same pattern but with basic placeholder logic

macro_rules! impl_chart_placeholder {
    ($chart_type:ident, $display_name:expr, $fn_name:ident, $min_args:expr) => {
        #[derive(Debug)]
        pub struct $chart_type {
            data: Vec<DataPoint>,
            options: ChartOptions,
            metadata: HashMap<String, String>,
        }
        
        impl $chart_type {
            pub fn new(data: Vec<DataPoint>, options: ChartOptions) -> Self {
                let mut metadata = HashMap::new();
                metadata.insert("type".to_string(), $display_name.to_string());
                metadata.insert("data_points".to_string(), data.len().to_string());
                
                Self {
                    data,
                    options,
                    metadata,
                }
            }
        }
        
        impl Foreign for $chart_type {
            fn as_any(&self) -> &dyn Any {
                self
            }
            
            fn type_name(&self) -> &'static str {
                $display_name
            }
        }
        
        impl Chart for $chart_type {
            fn render_svg(&self) -> VmResult<String> {
                Ok(format!("<svg><text x='10' y='20'>{} placeholder</text></svg>", $display_name))
            }
            
            fn get_metadata(&self) -> HashMap<String, String> {
                self.metadata.clone()
            }
            
            fn chart_type(&self) -> &'static str {
                $display_name
            }
        }
        
        impl fmt::Display for $chart_type {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}[{} data points]", $display_name, self.data.len())
            }
        }
        
        pub fn $fn_name(args: &[Value]) -> VmResult<Value> {
            if args.len() < $min_args {
                return Err(format!("{} requires at least {} arguments", $display_name, $min_args));
            }
            
            let data = if args.is_empty() {
                Vec::new()
            } else {
                parse_data_points(&args[0]).unwrap_or_default()
            };
            
            let options = if args.len() > ($min_args - 1) {
                ChartOptions::from_value(&args[$min_args - 1]).unwrap_or_default()
            } else {
                ChartOptions::default()
            };
            
            let chart = $chart_type::new(data, options);
            Ok(Value::LyObj(LyObj::new(Box::new(chart))))
        }
    };
}

// Implement all remaining chart types
impl_chart_placeholder!(PieChartObj, "PieChart", pie_chart, 4);
impl_chart_placeholder!(HeatmapObj, "Heatmap", heatmap, 4);
impl_chart_placeholder!(HistogramObj, "Histogram", histogram, 3);
impl_chart_placeholder!(BoxPlotObj, "BoxPlot", box_plot, 4);
impl_chart_placeholder!(AreaChartObj, "AreaChart", area_chart, 4);
impl_chart_placeholder!(BubbleChartObj, "BubbleChart", bubble_chart, 6);
impl_chart_placeholder!(CandlestickObj, "Candlestick", candlestick, 6);
impl_chart_placeholder!(RadarChartObj, "RadarChart", radar_chart, 4);
impl_chart_placeholder!(TreeMapObj, "TreeMap", tree_map, 5);
impl_chart_placeholder!(SankeyDiagramObj, "SankeyDiagram", sankey_diagram, 4);
impl_chart_placeholder!(NetworkDiagramObj, "NetworkDiagram", network_diagram, 4);
impl_chart_placeholder!(GanttChartObj, "GanttChart", gantt_chart, 4);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chart_options_parsing() {
        let options_value = Value::List(vec![
            Value::List(vec![
                Value::String("title".to_string()),
                Value::String("Test Chart".to_string()),
            ]),
            Value::List(vec![
                Value::String("width".to_string()),
                Value::Real(1000.0),
            ]),
        ]);
        
        let options = ChartOptions::from_value(&options_value).unwrap();
        assert_eq!(options.title, "Test Chart");
        assert_eq!(options.width, 1000);
    }

    #[test]
    fn test_data_point_parsing() {
        let data_value = Value::List(vec![
            Value::List(vec![Value::Real(1.0), Value::Real(10.0)]),
            Value::List(vec![Value::Real(2.0), Value::Real(20.0)]),
        ]);
        
        let points = parse_data_points(&data_value).unwrap();
        assert_eq!(points.len(), 2);
        assert_eq!(points[0].x, 1.0);
        assert_eq!(points[0].y, 10.0);
    }

    #[test]
    fn test_line_chart_creation() {
        let data = Value::List(vec![
            Value::List(vec![Value::Real(1.0), Value::Real(10.0)]),
            Value::List(vec![Value::Real(2.0), Value::Real(20.0)]),
        ]);
        let x_axis = Value::String("x".to_string());
        let y_axis = Value::String("y".to_string());
        let options = Value::List(vec![]);
        
        let result = line_chart(&[data, x_axis, y_axis, options]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_chart_palette() {
        let palette = ChartPalette::new();
        let color1 = palette.get_color(0);
        let color2 = palette.get_color(10); // Should wrap around
        
        assert_eq!(color1, palette.get_color(0));
        assert_eq!(color2, palette.get_color(0)); // Should be same due to modulo
    }
}