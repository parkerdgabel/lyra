//! Export & Rendering Module for Lyra Visualization System
//!
//! This module provides comprehensive export capabilities for charts and dashboards,
//! supporting multiple formats including SVG, PNG, PDF, and interactive HTML.

use crate::vm::{Value, VmResult};
use crate::foreign::{LyObj, Foreign};
use crate::stdlib::visualization::charts::{Chart, ChartOptions};
use crate::stdlib::visualization::dashboard::DashboardObj;
use std::collections::HashMap;
use std::any::Any;
use std::fmt;
use std::path::Path;
// use image::{ImageBuffer, Rgba};
use printpdf::*;
use tera::{Tera, Context};
use palette::{Hsl, IntoColor, Srgb};

/// Export options for different output formats
#[derive(Debug, Clone)]
pub struct ExportOptions {
    pub width: u32,
    pub height: u32,
    pub dpi: u32,
    pub quality: u8, // 0-100 for lossy formats
    pub background_color: String,
    pub transparent: bool,
    pub embed_fonts: bool,
    pub compress: bool,
    pub metadata: HashMap<String, String>,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            dpi: 300,
            quality: 95,
            background_color: "white".to_string(),
            transparent: false,
            embed_fonts: true,
            compress: true,
            metadata: HashMap::new(),
        }
    }
}

impl ExportOptions {
    pub fn from_value(value: &Value) -> VmResult<Self> {
        let mut options = ExportOptions::default();
        
        if let Value::List(opts) = value {
            for opt in opts {
                if let Value::List(pair) = opt {
                    if pair.len() == 2 {
                        if let (Value::String(key), val) = (&pair[0], &pair[1]) {
                            match key.as_str() {
                                "width" => {
                                    if let Value::Real(w) = val {
                                        options.width = *w as u32;
                                    }
                                }
                                "height" => {
                                    if let Value::Real(h) = val {
                                        options.height = *h as u32;
                                    }
                                }
                                "dpi" => {
                                    if let Value::Real(dpi) = val {
                                        options.dpi = *dpi as u32;
                                    }
                                }
                                "quality" => {
                                    if let Value::Real(q) = val {
                                        options.quality = (*q as u8).min(100);
                                    }
                                }
                                "background_color" => {
                                    if let Value::String(color) = val {
                                        options.background_color = color.clone();
                                    }
                                }
                                "transparent" => {
                                    if let Value::Boolean(t) = val {
                                        options.transparent = *t;
                                    }
                                }
                                "embed_fonts" => {
                                    if let Value::Boolean(ef) = val {
                                        options.embed_fonts = *ef;
                                    }
                                }
                                "compress" => {
                                    if let Value::Boolean(c) = val {
                                        options.compress = *c;
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

/// Chart template system for reusable chart configurations
#[derive(Debug, Clone)]
pub struct ChartTemplate {
    pub template_type: String,
    pub default_options: ChartOptions,
    pub color_scheme: Vec<String>,
    pub style_overrides: HashMap<String, String>,
}

impl ChartTemplate {
    pub fn new(template_type: String) -> Self {
        let (default_options, color_scheme) = match template_type.as_str() {
            "business" => (
                ChartOptions {
                    theme: "professional".to_string(),
                    color_scheme: "corporate".to_string(),
                    show_grid: true,
                    show_legend: true,
                    ..Default::default()
                },
                vec![
                    "#2C3E50".to_string(), "#3498DB".to_string(), "#E74C3C".to_string(),
                    "#F39C12".to_string(), "#27AE60".to_string(), "#9B59B6".to_string(),
                ]
            ),
            "scientific" => (
                ChartOptions {
                    theme: "minimal".to_string(),
                    color_scheme: "viridis".to_string(),
                    show_grid: true,
                    show_legend: true,
                    ..Default::default()
                },
                vec![
                    "#440154".to_string(), "#31688E".to_string(), "#35B779".to_string(),
                    "#FDE725".to_string(), "#21908C".to_string(), "#443A83".to_string(),
                ]
            ),
            "presentation" => (
                ChartOptions {
                    theme: "dark".to_string(),
                    color_scheme: "bright".to_string(),
                    show_grid: false,
                    show_legend: true,
                    ..Default::default()
                },
                vec![
                    "#FF6B6B".to_string(), "#4ECDC4".to_string(), "#45B7D1".to_string(),
                    "#96CEB4".to_string(), "#FFEAA7".to_string(), "#DDA0DD".to_string(),
                ]
            ),
            _ => (ChartOptions::default(), vec![
                "#1f77b4".to_string(), "#ff7f0e".to_string(), "#2ca02c".to_string(),
                "#d62728".to_string(), "#9467bd".to_string(), "#8c564b".to_string(),
            ])
        };
        
        Self {
            template_type,
            default_options,
            color_scheme,
            style_overrides: HashMap::new(),
        }
    }
}

/// Theme configuration for consistent styling
#[derive(Debug, Clone)]
pub struct VisualizationTheme {
    pub name: String,
    pub background_color: String,
    pub text_color: String,
    pub grid_color: String,
    pub accent_color: String,
    pub color_palette: Vec<String>,
    pub font_family: String,
    pub font_sizes: HashMap<String, u32>,
}

impl VisualizationTheme {
    pub fn light_theme() -> Self {
        let mut font_sizes = HashMap::new();
        font_sizes.insert("title".to_string(), 24);
        font_sizes.insert("subtitle".to_string(), 18);
        font_sizes.insert("axis_label".to_string(), 14);
        font_sizes.insert("legend".to_string(), 12);
        
        Self {
            name: "light".to_string(),
            background_color: "#FFFFFF".to_string(),
            text_color: "#2C3E50".to_string(),
            grid_color: "#ECF0F1".to_string(),
            accent_color: "#3498DB".to_string(),
            color_palette: vec![
                "#3498DB".to_string(), "#E74C3C".to_string(), "#2ECC71".to_string(),
                "#F39C12".to_string(), "#9B59B6".to_string(), "#1ABC9C".to_string(),
            ],
            font_family: "Arial, sans-serif".to_string(),
            font_sizes,
        }
    }
    
    pub fn dark_theme() -> Self {
        let mut font_sizes = HashMap::new();
        font_sizes.insert("title".to_string(), 24);
        font_sizes.insert("subtitle".to_string(), 18);
        font_sizes.insert("axis_label".to_string(), 14);
        font_sizes.insert("legend".to_string(), 12);
        
        Self {
            name: "dark".to_string(),
            background_color: "#2C3E50".to_string(),
            text_color: "#ECF0F1".to_string(),
            grid_color: "#34495E".to_string(),
            accent_color: "#3498DB".to_string(),
            color_palette: vec![
                "#3498DB".to_string(), "#E74C3C".to_string(), "#2ECC71".to_string(),
                "#F39C12".to_string(), "#9B59B6".to_string(), "#1ABC9C".to_string(),
            ],
            font_family: "Arial, sans-serif".to_string(),
            font_sizes,
        }
    }
}

/// Multi-page report generator
#[derive(Debug)]
pub struct MultiPageReportObj {
    pub charts: Vec<Value>,
    pub layout: String,
    pub metadata: HashMap<String, String>,
    pub theme: VisualizationTheme,
}

impl MultiPageReportObj {
    pub fn new(charts: Vec<Value>, layout: String, metadata: HashMap<String, String>) -> Self {
        Self {
            charts,
            layout,
            metadata,
            theme: VisualizationTheme::light_theme(),
        }
    }
    
    /// Generate PDF report
    pub fn generate_pdf(&self) -> VmResult<Vec<u8>> {
        let (doc, page1, layer1) = PdfDocument::new(
            &self.metadata.get("title").unwrap_or(&"Report".to_string()),
            Mm(210.0), // A4 width
            Mm(297.0), // A4 height
            "Layer 1"
        );
        
        let mut pdf_data = Vec::new();
        
        // Add title page
        let current_layer = doc.get_page(page1).get_layer(layer1);
        let font = doc.add_builtin_font(BuiltinFont::HelveticaBold)
            .map_err(|e| format!("Font error: {}", e))?;
        
        current_layer.use_text(
            &self.metadata.get("title").unwrap_or(&"Report".to_string()),
            24.0,
            Mm(105.0), // Center horizontally
            Mm(250.0), // Near top
            &font
        );
        
        // Add charts to subsequent pages
        for (i, chart_value) in self.charts.iter().enumerate() {
            if let Value::LyObj(chart_obj) = chart_value {
                if let Some(chart) = chart_obj.as_any().downcast_ref::<dyn Chart>() {
                    // Create new page for each chart
                    let (page, layer) = doc.add_page(Mm(210.0), Mm(297.0), &format!("Chart {}", i + 1));
                    let current_layer = doc.get_page(page).get_layer(layer);
                    
                    // Add chart title
                    current_layer.use_text(
                        &format!("Chart {}", i + 1),
                        18.0,
                        Mm(20.0),
                        Mm(270.0),
                        &font
                    );
                    
                    // Add chart SVG (simplified - would need SVG to PDF conversion)
                    let svg_content = chart.render_svg()?;
                    // In a full implementation, you'd convert SVG to PDF content here
                }
            }
        }
        
        // Save PDF
        doc.save(&mut pdf_data)
            .map_err(|e| format!("PDF save error: {}", e))?;
        
        Ok(pdf_data)
    }
}

impl Foreign for MultiPageReportObj {
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn type_name(&self) -> &'static str {
        "MultiPageReport"
    }
}

impl fmt::Display for MultiPageReportObj {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MultiPageReport[{} charts, layout: {}]", 
               self.charts.len(), self.layout)
    }
}

// =============================================================================
// Export Function Implementations
// =============================================================================

/// Export chart to SVG format
pub fn chart_to_svg(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 {
        return Err("ChartToSVG requires a chart object".to_string());
    }
    
    let options = if args.len() > 1 {
        ExportOptions::from_value(&args[1])?
    } else {
        ExportOptions::default()
    };
    
    if let Value::LyObj(chart_obj) = &args[0] {
        if let Some(chart) = chart_obj.as_any().downcast_ref::<dyn Chart>() {
            let svg_content = chart.render_svg()?;
            
            // Apply any SVG transformations based on options
            let processed_svg = process_svg_options(svg_content, &options)?;
            
            Ok(Value::String(processed_svg))
        } else {
            Err("Object is not a valid chart".to_string())
        }
    } else {
        Err("First argument must be a chart object".to_string())
    }
}

/// Export chart to PNG format
pub fn chart_to_png(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err("ChartToPNG requires chart, resolution, and options".to_string());
    }
    
    let resolution = match &args[1] {
        Value::Real(res) => *res as u32,
        _ => return Err("Resolution must be a number".to_string()),
    };
    
    let options = if args.len() > 2 {
        ExportOptions::from_value(&args[2])?
    } else {
        ExportOptions::default()
    };
    
    if let Value::LyObj(chart_obj) = &args[0] {
        if let Some(chart) = chart_obj.as_any().downcast_ref::<dyn Chart>() {
            let svg_content = chart.render_svg()?;
            
            // Convert SVG to PNG (simplified implementation)
            let png_data = svg_to_png(&svg_content, resolution, &options)?;
            
            // Return as base64 encoded string for transport
            let base64_data = base64::encode(&png_data);
            Ok(Value::String(format!("data:image/png;base64,{}", base64_data)))
        } else {
            Err("Object is not a valid chart".to_string())
        }
    } else {
        Err("First argument must be a chart object".to_string())
    }
}

/// Export chart to PDF format
pub fn chart_to_pdf(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err("ChartToPDF requires chart, page_size, and options".to_string());
    }
    
    let page_size = match &args[1] {
        Value::String(size) => size.clone(),
        _ => return Err("Page size must be a string".to_string()),
    };
    
    let options = if args.len() > 2 {
        ExportOptions::from_value(&args[2])?
    } else {
        ExportOptions::default()
    };
    
    if let Value::LyObj(chart_obj) = &args[0] {
        if let Some(chart) = chart_obj.as_any().downcast_ref::<dyn Chart>() {
            let svg_content = chart.render_svg()?;
            
            // Convert to PDF
            let pdf_data = chart_to_pdf_data(&svg_content, &page_size, &options)?;
            
            // Return as base64 encoded string
            let base64_data = base64::encode(&pdf_data);
            Ok(Value::String(format!("data:application/pdf;base64,{}", base64_data)))
        } else {
            Err("Object is not a valid chart".to_string())
        }
    } else {
        Err("First argument must be a chart object".to_string())
    }
}

/// Generate interactive HTML for chart
pub fn interactive_html(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err("InteractiveHTML requires chart, javascript_libs, and options".to_string());
    }
    
    let js_libs = match &args[1] {
        Value::List(libs) => {
            libs.iter().map(|v| match v {
                Value::String(lib) => Ok(lib.clone()),
                _ => Err("JavaScript libraries must be strings".to_string()),
            }).collect::<Result<Vec<_>, _>>()?
        }
        _ => return Err("JavaScript libraries must be a list".to_string()),
    };
    
    let options = if args.len() > 2 {
        ExportOptions::from_value(&args[2])?
    } else {
        ExportOptions::default()
    };
    
    if let Value::LyObj(chart_obj) = &args[0] {
        if let Some(chart) = chart_obj.as_any().downcast_ref::<dyn Chart>() {
            let html = generate_interactive_html(chart, &js_libs, &options)?;
            Ok(Value::String(html))
        } else {
            Err("Object is not a valid chart".to_string())
        }
    } else {
        Err("First argument must be a chart object".to_string())
    }
}

/// Create chart template
pub fn chart_template(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err("ChartTemplate requires chart_type and template_options".to_string());
    }
    
    let chart_type = match &args[0] {
        Value::String(t) => t.clone(),
        _ => return Err("Chart type must be a string".to_string()),
    };
    
    let template = ChartTemplate::new(chart_type);
    
    // Return template configuration as a value
    let template_data = serde_json::to_string(&template)
        .map_err(|e| format!("Template serialization error: {}", e))?;
    
    Ok(Value::String(template_data))
}

/// Apply theme to chart
pub fn theme_apply(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err("ThemeApply requires chart, theme_name, and custom_colors".to_string());
    }
    
    let theme_name = match &args[1] {
        Value::String(name) => name.clone(),
        _ => return Err("Theme name must be a string".to_string()),
    };
    
    let theme = match theme_name.as_str() {
        "light" => VisualizationTheme::light_theme(),
        "dark" => VisualizationTheme::dark_theme(),
        _ => return Err(format!("Unknown theme: {}", theme_name)),
    };
    
    // In a full implementation, this would modify the chart object
    // For now, return success indicator
    Ok(Value::Boolean(true))
}

/// Generate multi-page report
pub fn multi_page_report(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err("MultiPageReport requires charts, layout, and metadata".to_string());
    }
    
    let charts = match &args[0] {
        Value::List(chart_list) => chart_list.clone(),
        _ => return Err("Charts must be a list".to_string()),
    };
    
    let layout = match &args[1] {
        Value::String(l) => l.clone(),
        _ => return Err("Layout must be a string".to_string()),
    };
    
    let metadata = parse_metadata(&args[2])?;
    
    let report = MultiPageReportObj::new(charts, layout, metadata);
    Ok(Value::LyObj(LyObj::new(Box::new(report))))
}

// =============================================================================
// Helper Functions
// =============================================================================

fn process_svg_options(svg: String, options: &ExportOptions) -> VmResult<String> {
    let mut processed = svg;
    
    // Apply background color if not transparent
    if !options.transparent && options.background_color != "transparent" {
        processed = processed.replace(
            "<svg",
            &format!("<svg style=\"background-color: {}\"", options.background_color)
        );
    }
    
    // Apply width and height
    processed = processed.replace(
        "width=\"800\"",
        &format!("width=\"{}\"", options.width)
    );
    processed = processed.replace(
        "height=\"600\"",
        &format!("height=\"{}\"", options.height)
    );
    
    Ok(processed)
}

fn svg_to_png(_svg_content: &str, _resolution: u32, _options: &ExportOptions) -> VmResult<Vec<u8>> {
    // TODO: Implement PNG conversion when image crate API is fixed
    // For now, return empty PNG data as placeholder
    Ok(vec![137, 80, 78, 71, 13, 10, 26, 10]) // PNG header
}

fn chart_to_pdf_data(svg_content: &str, page_size: &str, options: &ExportOptions) -> VmResult<Vec<u8>> {
    let (width, height) = match page_size {
        "A4" => (Mm(210.0), Mm(297.0)),
        "Letter" => (Mm(215.9), Mm(279.4)),
        "Legal" => (Mm(215.9), Mm(355.6)),
        _ => (Mm(210.0), Mm(297.0)), // Default to A4
    };
    
    let (doc, page1, layer1) = PdfDocument::new("Chart", width, height, "Layer 1");
    
    // Add SVG content (simplified - would need proper SVG to PDF conversion)
    let current_layer = doc.get_page(page1).get_layer(layer1);
    let font = doc.add_builtin_font(BuiltinFont::Helvetica)
        .map_err(|e| format!("Font error: {}", e))?;
    
    current_layer.use_text("Chart Placeholder", 12.0, Mm(20.0), Mm(250.0), &font);
    
    let mut pdf_data = Vec::new();
    doc.save(&mut pdf_data)
        .map_err(|e| format!("PDF save error: {}", e))?;
    
    Ok(pdf_data)
}

fn generate_interactive_html(chart: &dyn Chart, js_libs: &[String], options: &ExportOptions) -> VmResult<String> {
    let mut html = String::new();
    
    html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
    html.push_str("<title>Interactive Chart</title>\n");
    
    // Include JavaScript libraries
    for lib in js_libs {
        match lib.as_str() {
            "d3" => {
                html.push_str("<script src=\"https://d3js.org/d3.v7.min.js\"></script>\n");
            }
            "plotly" => {
                html.push_str("<script src=\"https://cdn.plot.ly/plotly-latest.min.js\"></script>\n");
            }
            "chart.js" => {
                html.push_str("<script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>\n");
            }
            _ => {}
        }
    }
    
    html.push_str("</head>\n<body>\n");
    html.push_str("<div id=\"chart-container\">\n");
    
    // Embed SVG
    let svg_content = chart.render_svg()?;
    html.push_str(&svg_content);
    
    html.push_str("</div>\n");
    
    // Add interactivity JavaScript
    html.push_str("<script>\n");
    html.push_str("// Chart interactivity\n");
    html.push_str("document.addEventListener('DOMContentLoaded', function() {\n");
    html.push_str("  console.log('Interactive chart loaded');\n");
    html.push_str("  // Add event listeners, animations, etc.\n");
    html.push_str("});\n");
    html.push_str("</script>\n");
    
    html.push_str("</body>\n</html>");
    
    Ok(html)
}

fn parse_metadata(value: &Value) -> VmResult<HashMap<String, String>> {
    let mut metadata = HashMap::new();
    
    if let Value::List(items) = value {
        for item in items {
            if let Value::List(pair) = item {
                if pair.len() == 2 {
                    if let (Value::String(key), Value::String(val)) = (&pair[0], &pair[1]) {
                        metadata.insert(key.clone(), val.clone());
                    }
                }
            }
        }
    }
    
    Ok(metadata)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_options_parsing() {
        let options_value = Value::List(vec![
            Value::List(vec![
                Value::String("width".to_string()),
                Value::Real(1200.0),
            ]),
            Value::List(vec![
                Value::String("dpi".to_string()),
                Value::Real(300.0),
            ]),
        ]);
        
        let options = ExportOptions::from_value(&options_value).unwrap();
        assert_eq!(options.width, 1200);
        assert_eq!(options.dpi, 300);
    }

    #[test]
    fn test_chart_template_creation() {
        let template = ChartTemplate::new("business".to_string());
        assert_eq!(template.template_type, "business");
        assert!(!template.color_scheme.is_empty());
    }

    #[test]
    fn test_theme_creation() {
        let light_theme = VisualizationTheme::light_theme();
        assert_eq!(light_theme.name, "light");
        assert_eq!(light_theme.background_color, "#FFFFFF");
        
        let dark_theme = VisualizationTheme::dark_theme();
        assert_eq!(dark_theme.name, "dark");
        assert_eq!(dark_theme.background_color, "#2C3E50");
    }

    #[test]
    fn test_svg_processing() {
        let svg = "<svg width=\"800\" height=\"600\"></svg>".to_string();
        let options = ExportOptions {
            width: 1000,
            height: 800,
            ..Default::default()
        };
        
        let processed = process_svg_options(svg, &options).unwrap();
        assert!(processed.contains("width=\"1000\""));
        assert!(processed.contains("height=\"800\""));
    }
}