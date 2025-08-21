//! Lyra Visualization System - Phase 17A: Data Visualization & Dashboards
//!
//! This module provides comprehensive data visualization capabilities for the Lyra
//! symbolic computation engine, including:
//!
//! - 15 different chart types (line, bar, scatter, pie, heatmap, etc.)
//! - Interactive dashboard creation and management
//! - Multiple export formats (SVG, PNG, PDF, HTML)
//! - Interactive features (tooltips, zoom, selection, animations)
//! - Cross-chart filtering and real-time updates
//!
//! All visualization components are implemented as Foreign objects to maintain
//! VM simplicity while providing rich graphical capabilities.

use crate::vm::{Value, VmResult};

pub mod charts;
pub mod dashboard;
pub mod export;
pub mod interactive;

// Re-export key types for convenience
pub use charts::{
    Chart, ChartOptions, DataPoint, ChartPalette,
    LineChartObj, BarChartObj, ScatterPlotObj, PieChartObj,
    HeatmapObj, HistogramObj, BoxPlotObj, AreaChartObj,
    BubbleChartObj, CandlestickObj, RadarChartObj, TreeMapObj,
    SankeyDiagramObj, NetworkDiagramObj, GanttChartObj,
};

pub use dashboard::{
    DashboardObj, DashboardWidgetObj, RealTimeChartObj,
    DashboardLayout, WidgetConfig, DashboardFilter,
    DrillDownConfig, WidgetInteraction,
};

pub use export::{
    ExportOptions, ChartTemplate, VisualizationTheme,
    MultiPageReportObj,
};

pub use interactive::{
    TooltipObj, ZoomObj, SelectionObj, AnimationObj, CrossFilterObj,
    TooltipConfig, ZoomConfig, SelectionConfig, AnimationConfig,
    CrossFilterConfig,
};

// =============================================================================
// Chart Generation Functions (15 functions)
// =============================================================================

/// Create a line chart with multiple series support
pub fn line_chart(args: &[Value]) -> VmResult<Value> {
    charts::line_chart(args)
}

/// Create a bar chart with horizontal/vertical orientation
pub fn bar_chart(args: &[Value]) -> VmResult<Value> {
    charts::bar_chart(args)
}

/// Create a scatter plot with optional regression lines
pub fn scatter_plot(args: &[Value]) -> VmResult<Value> {
    charts::scatter_plot(args)
}

/// Create a pie chart with customizable segments
pub fn pie_chart(args: &[Value]) -> VmResult<Value> {
    charts::pie_chart(args)
}

/// Create a heatmap with color scale customization
pub fn heatmap(args: &[Value]) -> VmResult<Value> {
    charts::heatmap(args)
}

/// Create a histogram with binning options
pub fn histogram(args: &[Value]) -> VmResult<Value> {
    charts::histogram(args)
}

/// Create a box plot for statistical distribution visualization
pub fn box_plot(args: &[Value]) -> VmResult<Value> {
    charts::box_plot(args)
}

/// Create an area chart with stacking options
pub fn area_chart(args: &[Value]) -> VmResult<Value> {
    charts::area_chart(args)
}

/// Create a bubble chart with size and color dimensions
pub fn bubble_chart(args: &[Value]) -> VmResult<Value> {
    charts::bubble_chart(args)
}

/// Create a candlestick chart for financial data
pub fn candlestick(args: &[Value]) -> VmResult<Value> {
    charts::candlestick(args)
}

/// Create a radar/spider chart for multi-dimensional data
pub fn radar_chart(args: &[Value]) -> VmResult<Value> {
    charts::radar_chart(args)
}

/// Create a tree map for hierarchical data visualization
pub fn tree_map(args: &[Value]) -> VmResult<Value> {
    charts::tree_map(args)
}

/// Create a Sankey diagram for flow visualization
pub fn sankey_diagram(args: &[Value]) -> VmResult<Value> {
    charts::sankey_diagram(args)
}

/// Create a network diagram for graph visualization
pub fn network_diagram(args: &[Value]) -> VmResult<Value> {
    charts::network_diagram(args)
}

/// Create a Gantt chart for project timeline visualization
pub fn gantt_chart(args: &[Value]) -> VmResult<Value> {
    charts::gantt_chart(args)
}

// =============================================================================
// Dashboard Functions (8 functions)
// =============================================================================

/// Create an interactive dashboard with multiple widgets
pub fn dashboard(args: &[Value]) -> VmResult<Value> {
    dashboard::dashboard(args)
}

/// Create a dashboard widget with specific configuration
pub fn dashboard_widget(args: &[Value]) -> VmResult<Value> {
    dashboard::dashboard_widget(args)
}

/// Add interactive filters to dashboard
pub fn filter(args: &[Value]) -> VmResult<Value> {
    dashboard::filter(args)
}

/// Configure drill-down navigation for dashboard widgets
pub fn drill_down(args: &[Value]) -> VmResult<Value> {
    dashboard::drill_down(args)
}

/// Create a real-time updating chart
pub fn real_time_chart(args: &[Value]) -> VmResult<Value> {
    dashboard::real_time_chart(args)
}

/// Configure dashboard layout and responsive design
pub fn dashboard_layout(args: &[Value]) -> VmResult<Value> {
    dashboard::dashboard_layout(args)
}

/// Set up widget interactions and cross-widget communication
pub fn widget_interaction(args: &[Value]) -> VmResult<Value> {
    dashboard::widget_interaction(args)
}

/// Export dashboard to various formats
pub fn dashboard_export(args: &[Value]) -> VmResult<Value> {
    dashboard::dashboard_export(args)
}

// =============================================================================
// Export & Rendering Functions (7 functions)
// =============================================================================

/// Export chart to SVG format with customization options
pub fn chart_to_svg(args: &[Value]) -> VmResult<Value> {
    export::chart_to_svg(args)
}

/// Export chart to PNG format with resolution settings
pub fn chart_to_png(args: &[Value]) -> VmResult<Value> {
    export::chart_to_png(args)
}

/// Export chart to PDF format with page layout options
pub fn chart_to_pdf(args: &[Value]) -> VmResult<Value> {
    export::chart_to_pdf(args)
}

/// Generate interactive HTML with JavaScript libraries
pub fn interactive_html(args: &[Value]) -> VmResult<Value> {
    export::interactive_html(args)
}

/// Create reusable chart templates
pub fn chart_template(args: &[Value]) -> VmResult<Value> {
    export::chart_template(args)
}

/// Apply visual themes to charts
pub fn theme_apply(args: &[Value]) -> VmResult<Value> {
    export::theme_apply(args)
}

/// Generate multi-page reports with multiple charts
pub fn multi_page_report(args: &[Value]) -> VmResult<Value> {
    export::multi_page_report(args)
}

// =============================================================================
// Interactive Features Functions (5 functions)
// =============================================================================

/// Add interactive tooltips to charts
pub fn tooltip(args: &[Value]) -> VmResult<Value> {
    interactive::tooltip(args)
}

/// Enable zoom and pan functionality for charts
pub fn zoom(args: &[Value]) -> VmResult<Value> {
    interactive::zoom(args)
}

/// Add data point selection capabilities
pub fn selection(args: &[Value]) -> VmResult<Value> {
    interactive::selection(args)
}

/// Add animations and transitions to charts
pub fn animation(args: &[Value]) -> VmResult<Value> {
    interactive::animation(args)
}

/// Enable cross-chart filtering and synchronization
pub fn cross_filter(args: &[Value]) -> VmResult<Value> {
    interactive::cross_filter(args)
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Get information about available chart types
pub fn get_chart_types() -> Vec<&'static str> {
    vec![
        "LineChart",
        "BarChart", 
        "ScatterPlot",
        "PieChart",
        "Heatmap",
        "Histogram",
        "BoxPlot",
        "AreaChart",
        "BubbleChart",
        "Candlestick",
        "RadarChart",
        "TreeMap",
        "SankeyDiagram",
        "NetworkDiagram",
        "GanttChart",
    ]
}

/// Get available export formats
pub fn get_export_formats() -> Vec<&'static str> {
    vec!["SVG", "PNG", "PDF", "HTML", "JSON"]
}

/// Get available visualization themes
pub fn get_themes() -> Vec<&'static str> {
    vec!["light", "dark", "business", "scientific", "presentation"]
}

/// Validate chart data format
pub fn validate_chart_data(data: &Value) -> VmResult<bool> {
    match data {
        Value::List(points) => {
            for point in points {
                match point {
                    Value::List(coords) if coords.len() >= 2 => {
                        if !matches!(coords[0], Value::Real(_)) || !matches!(coords[1], Value::Real(_)) {
                            return Ok(false);
                        }
                    }
                    Value::Real(_) => {
                        // Single value is valid (will use index as x)
                    }
                    _ => return Ok(false),
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// Create sample data for testing and demos
pub fn create_sample_data(chart_type: &str, num_points: usize) -> Value {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    match chart_type {
        "line" | "scatter" | "area" => {
            let points: Vec<Value> = (0..num_points)
                .map(|i| {
                    let x = i as f64;
                    let y = rng.gen_range(0.0..100.0);
                    Value::List(vec![Value::Real(x), Value::Real(y)])
                })
                .collect();
            Value::List(points)
        }
        "bar" | "pie" => {
            let values: Vec<Value> = (0..num_points)
                .map(|_| Value::Real(rng.gen_range(10.0..100.0)))
                .collect();
            Value::List(values)
        }
        "heatmap" => {
            let size = (num_points as f64).sqrt() as usize;
            let matrix: Vec<Value> = (0..size)
                .map(|_| {
                    let row: Vec<Value> = (0..size)
                        .map(|_| Value::Real(rng.gen_range(0.0..1.0)))
                        .collect();
                    Value::List(row)
                })
                .collect();
            Value::List(matrix)
        }
        _ => {
            // Default to simple numeric series
            let values: Vec<Value> = (0..num_points)
                .map(|_| Value::Real(rng.gen_range(0.0..100.0)))
                .collect();
            Value::List(values)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chart_types_availability() {
        let types = get_chart_types();
        assert_eq!(types.len(), 15);
        assert!(types.contains(&"LineChart"));
        assert!(types.contains(&"BarChart"));
        assert!(types.contains(&"GanttChart"));
    }

    #[test]
    fn test_export_formats_availability() {
        let formats = get_export_formats();
        assert!(formats.contains(&"SVG"));
        assert!(formats.contains(&"PNG"));
        assert!(formats.contains(&"PDF"));
        assert!(formats.contains(&"HTML"));
    }

    #[test]
    fn test_data_validation() {
        // Valid data
        let valid_data = Value::List(vec![
            Value::List(vec![Value::Real(1.0), Value::Real(10.0)]),
            Value::List(vec![Value::Real(2.0), Value::Real(20.0)]),
        ]);
        assert!(validate_chart_data(&valid_data).unwrap());

        // Invalid data
        let invalid_data = Value::List(vec![
            Value::String("invalid".to_string()),
        ]);
        assert!(!validate_chart_data(&invalid_data).unwrap());

        // Single values (valid)
        let single_values = Value::List(vec![
            Value::Real(10.0),
            Value::Real(20.0),
            Value::Real(15.0),
        ]);
        assert!(validate_chart_data(&single_values).unwrap());
    }

    #[test]
    fn test_sample_data_generation() {
        let line_data = create_sample_data("line", 5);
        if let Value::List(points) = line_data {
            assert_eq!(points.len(), 5);
            // Each point should be a [x, y] pair
            if let Value::List(first_point) = &points[0] {
                assert_eq!(first_point.len(), 2);
                assert!(matches!(first_point[0], Value::Real(_)));
                assert!(matches!(first_point[1], Value::Real(_)));
            }
        } else {
            panic!("Expected list of data points");
        }

        let bar_data = create_sample_data("bar", 3);
        if let Value::List(values) = bar_data {
            assert_eq!(values.len(), 3);
            assert!(matches!(values[0], Value::Real(_)));
        } else {
            panic!("Expected list of values");
        }
    }

    #[test]
    fn test_theme_availability() {
        let themes = get_themes();
        assert!(themes.contains(&"light"));
        assert!(themes.contains(&"dark"));
        assert!(themes.contains(&"business"));
    }
}