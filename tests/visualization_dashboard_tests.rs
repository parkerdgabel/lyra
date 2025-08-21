use lyra::vm::{Value, VmResult};
use lyra::stdlib::visualization::dashboard::*;
use lyra::foreign::{LyObj, Foreign};

fn create_test_dashboard_data() -> (Value, Value, Value) {
    let name = Value::String("Test Dashboard".to_string());
    let layout = Value::String("grid".to_string());
    let widgets = Value::List(vec![
        Value::List(vec![
            Value::String("line_chart".to_string()),
            Value::String("sales_data".to_string()),
            Value::List(vec![Value::Number(0.0), Value::Number(0.0)]), // position
        ]),
        Value::List(vec![
            Value::String("bar_chart".to_string()),
            Value::String("category_data".to_string()),
            Value::List(vec![Value::Number(4.0), Value::Number(0.0)]), // position
        ]),
    ]);
    (name, layout, widgets)
}

#[test]
fn test_dashboard_creation() {
    let (name, layout, widgets) = create_test_dashboard_data();
    
    let result = dashboard(&[name, layout, widgets]);
    assert!(result.is_ok());
    
    if let Ok(Value::LyObj(obj)) = result {
        assert!(obj.as_any().is::<DashboardObj>());
        
        if let Some(dashboard_obj) = obj.as_any().downcast_ref::<DashboardObj>() {
            assert_eq!(dashboard_obj.name, "Test Dashboard");
            assert!(!dashboard_obj.widgets.is_empty());
        }
    } else {
        panic!("Expected LyObj with DashboardObj");
    }
}

#[test]
fn test_dashboard_widget_creation() {
    let widget_type = Value::String("line_chart".to_string());
    let data_source = Value::String("sales_data".to_string());
    let config = Value::List(vec![
        Value::List(vec![
            Value::String("title".to_string()),
            Value::String("Sales Chart".to_string()),
        ]),
    ]);
    let position = Value::List(vec![Value::Number(0.0), Value::Number(0.0)]);
    
    let result = dashboard_widget(&[widget_type, data_source, config, position]);
    assert!(result.is_ok());
    
    if let Ok(Value::LyObj(obj)) = result {
        assert!(obj.as_any().is::<DashboardWidgetObj>());
    } else {
        panic!("Expected LyObj with DashboardWidgetObj");
    }
}

#[test]
fn test_filter_creation() {
    let dashboard_mock = Value::String("dashboard".to_string()); // Mock dashboard
    let field = Value::String("category".to_string());
    let values = Value::List(vec![
        Value::String("A".to_string()),
        Value::String("B".to_string()),
        Value::String("C".to_string()),
    ]);
    let operation = Value::String("equals".to_string());
    
    let result = filter(&[dashboard_mock, field, values, operation]);
    assert!(result.is_ok());
    
    if let Ok(Value::Boolean(success)) = result {
        assert!(success);
    } else {
        panic!("Expected Boolean success indicator");
    }
}

#[test]
fn test_drill_down_configuration() {
    let widget = Value::String("sales_chart".to_string());
    let hierarchy = Value::List(vec![
        Value::String("year".to_string()),
        Value::String("month".to_string()),
        Value::String("day".to_string()),
    ]);
    let action = Value::String("navigate".to_string());
    
    let result = drill_down(&[widget, hierarchy, action]);
    assert!(result.is_ok());
    
    if let Ok(Value::Boolean(success)) = result {
        assert!(success);
    } else {
        panic!("Expected Boolean success indicator");
    }
}

#[test]
fn test_real_time_chart_creation() {
    let data_stream = Value::String("live_sales_data".to_string());
    let chart_type = Value::String("line".to_string());
    let update_interval = Value::Number(1000.0); // 1 second
    
    let result = real_time_chart(&[data_stream, chart_type, update_interval]);
    assert!(result.is_ok());
    
    if let Ok(Value::LyObj(obj)) = result {
        assert!(obj.as_any().is::<RealTimeChartObj>());
        
        if let Some(rt_chart) = obj.as_any().downcast_ref::<RealTimeChartObj>() {
            assert_eq!(rt_chart.config.chart_type, "line");
            assert_eq!(rt_chart.config.update_interval, 1000);
        }
    } else {
        panic!("Expected LyObj with RealTimeChartObj");
    }
}

#[test]
fn test_dashboard_layout_configuration() {
    let grid_size = Value::List(vec![Value::Number(12.0), Value::Number(8.0)]);
    let responsive = Value::Boolean(true);
    let sections = Value::List(vec![]);
    
    let result = dashboard_layout(&[grid_size, responsive, sections]);
    assert!(result.is_ok());
    
    if let Ok(Value::Boolean(success)) = result {
        assert!(success);
    } else {
        panic!("Expected Boolean success indicator");
    }
}

#[test]
fn test_widget_interaction_setup() {
    let source_widget = Value::String("chart1".to_string());
    let target_widget = Value::String("chart2".to_string());
    let event = Value::String("click".to_string());
    let action = Value::String("filter".to_string());
    
    let result = widget_interaction(&[source_widget, target_widget, event, action]);
    assert!(result.is_ok());
    
    if let Ok(Value::Boolean(success)) = result {
        assert!(success);
    } else {
        panic!("Expected Boolean success indicator");
    }
}

#[test]
fn test_dashboard_export() {
    // Create a test dashboard first
    let (name, layout, widgets) = create_test_dashboard_data();
    let dashboard_result = dashboard(&[name, layout, widgets]);
    assert!(dashboard_result.is_ok());
    
    let dashboard_obj = dashboard_result.unwrap();
    let format = Value::String("html".to_string());
    let options = Value::List(vec![]);
    
    let result = dashboard_export(&[dashboard_obj, format, options]);
    assert!(result.is_ok());
    
    if let Ok(Value::String(html_content)) = result {
        assert!(html_content.contains("<!DOCTYPE html>"));
        assert!(html_content.contains("Test Dashboard"));
        assert!(html_content.contains("dashboard"));
    } else {
        panic!("Expected String with HTML content");
    }
}

#[test]
fn test_dashboard_config_parsing() {
    let config = WidgetConfig::new(
        "line_chart".to_string(),
        "test_data".to_string(),
        (2, 3),
    );
    
    assert_eq!(config.widget_type, "line_chart");
    assert_eq!(config.data_source, "test_data");
    assert_eq!(config.position, (2, 3));
    assert_eq!(config.size, (4, 3)); // Default size
}

#[test]
fn test_dashboard_filter_configuration() {
    let filter_config = DashboardFilter {
        field: "category".to_string(),
        values: vec!["A".to_string(), "B".to_string()],
        operation: "equals".to_string(),
        widget_targets: vec!["widget1".to_string(), "widget2".to_string()],
    };
    
    assert_eq!(filter_config.field, "category");
    assert_eq!(filter_config.values.len(), 2);
    assert_eq!(filter_config.operation, "equals");
    assert_eq!(filter_config.widget_targets.len(), 2);
}

#[test]
fn test_drill_down_config() {
    let drill_config = DrillDownConfig {
        source_widget: "sales_chart".to_string(),
        hierarchy: vec!["year".to_string(), "quarter".to_string(), "month".to_string()],
        action: "expand".to_string(),
        target_dashboard: Some("detailed_view".to_string()),
    };
    
    assert_eq!(drill_config.source_widget, "sales_chart");
    assert_eq!(drill_config.hierarchy.len(), 3);
    assert_eq!(drill_config.action, "expand");
    assert!(drill_config.target_dashboard.is_some());
}

#[test]
fn test_widget_interaction_config() {
    let interaction = WidgetInteraction {
        source_widget: "chart1".to_string(),
        target_widget: "chart2".to_string(),
        event: "click".to_string(),
        action: "highlight".to_string(),
        parameters: std::collections::HashMap::new(),
    };
    
    assert_eq!(interaction.source_widget, "chart1");
    assert_eq!(interaction.target_widget, "chart2");
    assert_eq!(interaction.event, "click");
    assert_eq!(interaction.action, "highlight");
}

#[test]
fn test_dashboard_with_insufficient_args() {
    let name = Value::String("Test".to_string());
    let layout = Value::String("grid".to_string());
    
    // Missing widgets argument
    let result = dashboard(&[name, layout]);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Dashboard requires"));
}

#[test]
fn test_real_time_chart_with_invalid_args() {
    let data_stream = Value::Number(123.0); // Should be string
    let chart_type = Value::String("line".to_string());
    let interval = Value::Number(1000.0);
    
    let result = real_time_chart(&[data_stream, chart_type, interval]);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Data stream must be a string"));
}

#[test]
fn test_dashboard_html_export_structure() {
    // Create a dashboard with multiple widgets
    let (name, layout, widgets) = create_test_dashboard_data();
    let dashboard_result = dashboard(&[name, layout, widgets]).unwrap();
    
    if let Value::LyObj(obj) = dashboard_result {
        if let Some(dashboard_obj) = obj.as_any().downcast_ref::<DashboardObj>() {
            let html = dashboard_obj.export_to_html().unwrap();
            
            // Check HTML structure
            assert!(html.contains("<html>"));
            assert!(html.contains("<head>"));
            assert!(html.contains("<body>"));
            assert!(html.contains("<div class='dashboard'"));
            assert!(html.contains("<div class='widget-grid'"));
            assert!(html.contains("<script>"));
            
            // Check dashboard title
            assert!(html.contains("Test Dashboard"));
            
            // Check JavaScript functionality
            assert!(html.contains("class Dashboard"));
            assert!(html.contains("setupFilters"));
            assert!(html.contains("setupWidgetInteractions"));
        }
    }
}