use lyra::vm::{Value, VmResult};
use lyra::stdlib::visualization::export::*;
use lyra::stdlib::visualization::charts::*;
use lyra::foreign::{LyObj, Foreign};

fn create_test_chart() -> Value {
    let data = Value::List(vec![
        Value::List(vec![Value::Number(1.0), Value::Number(10.0)]),
        Value::List(vec![Value::Number(2.0), Value::Number(20.0)]),
        Value::List(vec![Value::Number(3.0), Value::Number(15.0)]),
    ]);
    let x_axis = Value::String("x".to_string());
    let y_axis = Value::String("y".to_string());
    let options = Value::List(vec![
        Value::List(vec![
            Value::String("title".to_string()),
            Value::String("Test Chart".to_string()),
        ]),
    ]);
    
    line_chart(&[data, x_axis, y_axis, options]).unwrap()
}

fn create_test_export_options() -> Value {
    Value::List(vec![
        Value::List(vec![
            Value::String("width".to_string()),
            Value::Number(1000.0),
        ]),
        Value::List(vec![
            Value::String("height".to_string()),
            Value::Number(800.0),
        ]),
        Value::List(vec![
            Value::String("dpi".to_string()),
            Value::Number(300.0),
        ]),
        Value::List(vec![
            Value::String("quality".to_string()),
            Value::Number(95.0),
        ]),
    ])
}

#[test]
fn test_chart_to_svg_export() {
    let chart = create_test_chart();
    let options = create_test_export_options();
    
    let result = chart_to_svg(&[chart, options]);
    assert!(result.is_ok());
    
    if let Ok(Value::String(svg_content)) = result {
        assert!(svg_content.contains("<svg"));
        assert!(svg_content.contains("width=\"1000\""));
        assert!(svg_content.contains("height=\"800\""));
    } else {
        panic!("Expected SVG content as string");
    }
}

#[test]
fn test_chart_to_png_export() {
    let chart = create_test_chart();
    let resolution = Value::Number(300.0);
    let options = create_test_export_options();
    
    let result = chart_to_png(&[chart, resolution, options]);
    assert!(result.is_ok());
    
    if let Ok(Value::String(png_data)) = result {
        assert!(png_data.starts_with("data:image/png;base64,"));
    } else {
        panic!("Expected PNG data as base64 string");
    }
}

#[test]
fn test_chart_to_pdf_export() {
    let chart = create_test_chart();
    let page_size = Value::String("A4".to_string());
    let options = create_test_export_options();
    
    let result = chart_to_pdf(&[chart, page_size, options]);
    assert!(result.is_ok());
    
    if let Ok(Value::String(pdf_data)) = result {
        assert!(pdf_data.starts_with("data:application/pdf;base64,"));
    } else {
        panic!("Expected PDF data as base64 string");
    }
}

#[test]
fn test_interactive_html_generation() {
    let chart = create_test_chart();
    let js_libs = Value::List(vec![
        Value::String("d3".to_string()),
        Value::String("plotly".to_string()),
    ]);
    let options = create_test_export_options();
    
    let result = interactive_html(&[chart, js_libs, options]);
    assert!(result.is_ok());
    
    if let Ok(Value::String(html_content)) = result {
        assert!(html_content.contains("<!DOCTYPE html>"));
        assert!(html_content.contains("<script src=\"https://d3js.org"));
        assert!(html_content.contains("<script src=\"https://cdn.plot.ly"));
        assert!(html_content.contains("<div id=\"chart-container\""));
        assert!(html_content.contains("Interactive chart loaded"));
    } else {
        panic!("Expected HTML content as string");
    }
}

#[test]
fn test_chart_template_creation() {
    let chart_type = Value::String("business".to_string());
    let template_options = Value::List(vec![]);
    
    let result = chart_template(&[chart_type, template_options]);
    assert!(result.is_ok());
    
    if let Ok(Value::String(template_json)) = result {
        assert!(template_json.contains("business"));
        assert!(template_json.contains("color_scheme"));
    } else {
        panic!("Expected template configuration as JSON string");
    }
}

#[test]
fn test_theme_application() {
    let chart = create_test_chart();
    let theme_name = Value::String("dark".to_string());
    let custom_colors = Value::List(vec![]);
    
    let result = theme_apply(&[chart, theme_name, custom_colors]);
    assert!(result.is_ok());
    
    if let Ok(Value::Boolean(success)) = result {
        assert!(success);
    } else {
        panic!("Expected Boolean success indicator");
    }
}

#[test]
fn test_multi_page_report_creation() {
    let chart1 = create_test_chart();
    let chart2 = create_test_chart();
    let charts = Value::List(vec![chart1, chart2]);
    let layout = Value::String("A4".to_string());
    let metadata = Value::List(vec![
        Value::List(vec![
            Value::String("title".to_string()),
            Value::String("Monthly Report".to_string()),
        ]),
        Value::List(vec![
            Value::String("author".to_string()),
            Value::String("Analytics Team".to_string()),
        ]),
    ]);
    
    let result = multi_page_report(&[charts, layout, metadata]);
    assert!(result.is_ok());
    
    if let Ok(Value::LyObj(obj)) = result {
        assert!(obj.as_any().is::<MultiPageReportObj>());
        
        if let Some(report) = obj.as_any().downcast_ref::<MultiPageReportObj>() {
            assert_eq!(report.charts.len(), 2);
            assert_eq!(report.layout, "A4");
            assert!(report.metadata.contains_key("title"));
            assert!(report.metadata.contains_key("author"));
        }
    } else {
        panic!("Expected LyObj with MultiPageReportObj");
    }
}

#[test]
fn test_export_options_parsing() {
    let options_value = Value::List(vec![
        Value::List(vec![
            Value::String("width".to_string()),
            Value::Number(1200.0),
        ]),
        Value::List(vec![
            Value::String("height".to_string()),
            Value::Number(900.0),
        ]),
        Value::List(vec![
            Value::String("dpi".to_string()),
            Value::Number(150.0),
        ]),
        Value::List(vec![
            Value::String("transparent".to_string()),
            Value::Boolean(true),
        ]),
    ]);
    
    let options = ExportOptions::from_value(&options_value).unwrap();
    assert_eq!(options.width, 1200);
    assert_eq!(options.height, 900);
    assert_eq!(options.dpi, 150);
    assert!(options.transparent);
}

#[test]
fn test_chart_template_types() {
    let business_template = ChartTemplate::new("business".to_string());
    assert_eq!(business_template.template_type, "business");
    assert_eq!(business_template.default_options.theme, "professional");
    assert!(!business_template.color_scheme.is_empty());
    
    let scientific_template = ChartTemplate::new("scientific".to_string());
    assert_eq!(scientific_template.template_type, "scientific");
    assert_eq!(scientific_template.default_options.theme, "minimal");
    
    let presentation_template = ChartTemplate::new("presentation".to_string());
    assert_eq!(presentation_template.template_type, "presentation");
    assert_eq!(presentation_template.default_options.theme, "dark");
}

#[test]
fn test_visualization_themes() {
    let light_theme = VisualizationTheme::light_theme();
    assert_eq!(light_theme.name, "light");
    assert_eq!(light_theme.background_color, "#FFFFFF");
    assert_eq!(light_theme.text_color, "#2C3E50");
    assert!(!light_theme.color_palette.is_empty());
    
    let dark_theme = VisualizationTheme::dark_theme();
    assert_eq!(dark_theme.name, "dark");
    assert_eq!(dark_theme.background_color, "#2C3E50");
    assert_eq!(dark_theme.text_color, "#ECF0F1");
}

#[test]
fn test_svg_options_processing() {
    let svg = "<svg width=\"800\" height=\"600\"></svg>".to_string();
    let options = ExportOptions {
        width: 1000,
        height: 800,
        background_color: "blue".to_string(),
        transparent: false,
        ..Default::default()
    };
    
    let processed = process_svg_options(svg, &options).unwrap();
    assert!(processed.contains("width=\"1000\""));
    assert!(processed.contains("height=\"800\""));
    assert!(processed.contains("background-color: blue"));
}

#[test]
fn test_multi_page_report_pdf_generation() {
    let chart = create_test_chart();
    let charts = vec![chart];
    let layout = "A4".to_string();
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("title".to_string(), "Test Report".to_string());
    metadata.insert("author".to_string(), "Test Author".to_string());
    
    let report = MultiPageReportObj::new(charts, layout, metadata);
    let pdf_result = report.generate_pdf();
    assert!(pdf_result.is_ok());
    
    let pdf_data = pdf_result.unwrap();
    assert!(!pdf_data.is_empty());
    // PDF should start with PDF header
    assert!(pdf_data.starts_with(b"%PDF"));
}

#[test]
fn test_invalid_chart_export() {
    let invalid_chart = Value::String("not a chart".to_string());
    let options = create_test_export_options();
    
    let result = chart_to_svg(&[invalid_chart, options]);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("must be a chart object"));
}

#[test]
fn test_invalid_export_format() {
    // Test with insufficient arguments
    let chart = create_test_chart();
    
    let result = chart_to_png(&[chart]); // Missing resolution
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("requires chart, resolution"));
}

#[test]
fn test_html_generation_with_different_libraries() {
    let chart = create_test_chart();
    let options = create_test_export_options();
    
    // Test with Chart.js
    let chartjs_libs = Value::List(vec![Value::String("chart.js".to_string())]);
    let result = interactive_html(&[chart.clone(), chartjs_libs, options.clone()]);
    assert!(result.is_ok());
    if let Ok(Value::String(html)) = result {
        assert!(html.contains("chart.js"));
    }
    
    // Test with D3
    let d3_libs = Value::List(vec![Value::String("d3".to_string())]);
    let result = interactive_html(&[chart.clone(), d3_libs, options.clone()]);
    assert!(result.is_ok());
    if let Ok(Value::String(html)) = result {
        assert!(html.contains("d3js.org"));
    }
    
    // Test with Plotly
    let plotly_libs = Value::List(vec![Value::String("plotly".to_string())]);
    let result = interactive_html(&[chart, plotly_libs, options]);
    assert!(result.is_ok());
    if let Ok(Value::String(html)) = result {
        assert!(html.contains("plotly"));
    }
}

#[test]
fn test_export_options_defaults() {
    let default_options = ExportOptions::default();
    assert_eq!(default_options.width, 800);
    assert_eq!(default_options.height, 600);
    assert_eq!(default_options.dpi, 300);
    assert_eq!(default_options.quality, 95);
    assert_eq!(default_options.background_color, "white");
    assert!(!default_options.transparent);
    assert!(default_options.embed_fonts);
    assert!(default_options.compress);
}

#[test]
fn test_page_size_configurations() {
    let chart = create_test_chart();
    let options = create_test_export_options();
    
    // Test different page sizes
    let page_sizes = vec!["A4", "Letter", "Legal"];
    
    for page_size in page_sizes {
        let size_value = Value::String(page_size.to_string());
        let result = chart_to_pdf(&[chart.clone(), size_value, options.clone()]);
        assert!(result.is_ok());
    }
}