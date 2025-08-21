use lyra::vm::{Value, VmResult};
use lyra::stdlib::visualization::interactive::*;
use lyra::foreign::{LyObj, Foreign};

fn create_mock_chart() -> Value {
    // Create a mock chart object for testing
    Value::String("mock_chart_id".to_string())
}

#[test]
fn test_tooltip_creation() {
    let chart = create_mock_chart();
    let content_template = Value::String("X: {x}, Y: {y}".to_string());
    let position = Value::String("mouse".to_string());
    
    let result = tooltip(&[chart, content_template, position]);
    assert!(result.is_ok());
    
    if let Ok(Value::LyObj(obj)) = result {
        assert!(obj.as_any().is::<TooltipObj>());
        
        if let Some(tooltip_obj) = obj.as_any().downcast_ref::<TooltipObj>() {
            assert_eq!(tooltip_obj.config.content_template, "X: {x}, Y: {y}");
            assert!(matches!(tooltip_obj.config.position, TooltipPosition::Mouse));
            assert!(tooltip_obj.config.enabled);
        }
    } else {
        panic!("Expected LyObj with TooltipObj");
    }
}

#[test]
fn test_tooltip_with_fixed_position() {
    let chart = create_mock_chart();
    let content_template = Value::String("{value}".to_string());
    let position = Value::List(vec![Value::Number(100.0), Value::Number(200.0)]);
    
    let result = tooltip(&[chart, content_template, position]);
    assert!(result.is_ok());
    
    if let Ok(Value::LyObj(obj)) = result {
        if let Some(tooltip_obj) = obj.as_any().downcast_ref::<TooltipObj>() {
            assert!(matches!(tooltip_obj.config.position, TooltipPosition::Fixed { x: 100, y: 200 }));
        }
    }
}

#[test]
fn test_zoom_creation() {
    let chart = create_mock_chart();
    let axes = Value::List(vec![
        Value::String("x".to_string()),
        Value::String("y".to_string()),
    ]);
    let constraints = Value::List(vec![]); // Empty constraints for simplicity
    
    let result = zoom(&[chart, axes, constraints]);
    assert!(result.is_ok());
    
    if let Ok(Value::LyObj(obj)) = result {
        assert!(obj.as_any().is::<ZoomObj>());
        
        if let Some(zoom_obj) = obj.as_any().downcast_ref::<ZoomObj>() {
            assert_eq!(zoom_obj.config.axes.len(), 2);
            assert!(zoom_obj.config.axes.contains(&"x".to_string()));
            assert!(zoom_obj.config.axes.contains(&"y".to_string()));
            assert!(zoom_obj.config.pan_enabled);
            assert_eq!(zoom_obj.current_zoom, 1.0);
        }
    } else {
        panic!("Expected LyObj with ZoomObj");
    }
}

#[test]
fn test_selection_creation() {
    let chart = create_mock_chart();
    let selection_mode = Value::String("rectangle".to_string());
    let callback = Value::String("onSelection".to_string());
    
    let result = selection(&[chart, selection_mode, callback]);
    assert!(result.is_ok());
    
    if let Ok(Value::LyObj(obj)) = result {
        assert!(obj.as_any().is::<SelectionObj>());
        
        if let Some(selection_obj) = obj.as_any().downcast_ref::<SelectionObj>() {
            assert!(matches!(selection_obj.config.mode, SelectionMode::Rectangle));
            assert_eq!(selection_obj.config.callback, Some("onSelection".to_string()));
            assert!(selection_obj.selected_items.is_empty());
        }
    } else {
        panic!("Expected LyObj with SelectionObj");
    }
}

#[test]
fn test_selection_modes() {
    let chart = create_mock_chart();
    let callback = Value::String("callback".to_string());
    
    let modes = vec![
        ("point", SelectionMode::Point),
        ("rectangle", SelectionMode::Rectangle),
        ("lasso", SelectionMode::Lasso),
        ("brush", SelectionMode::Brush),
        ("none", SelectionMode::None),
    ];
    
    for (mode_str, expected_mode) in modes {
        let selection_mode = Value::String(mode_str.to_string());
        let result = selection(&[chart.clone(), selection_mode, callback.clone()]);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(obj)) = result {
            if let Some(selection_obj) = obj.as_any().downcast_ref::<SelectionObj>() {
                match (&selection_obj.config.mode, &expected_mode) {
                    (SelectionMode::Point, SelectionMode::Point) => {},
                    (SelectionMode::Rectangle, SelectionMode::Rectangle) => {},
                    (SelectionMode::Lasso, SelectionMode::Lasso) => {},
                    (SelectionMode::Brush, SelectionMode::Brush) => {},
                    (SelectionMode::None, SelectionMode::None) => {},
                    _ => panic!("Mode mismatch for {}", mode_str),
                }
            }
        }
    }
}

#[test]
fn test_animation_creation() {
    let chart = create_mock_chart();
    let transition_type = Value::String("fade_in".to_string());
    let duration = Value::Number(1500.0);
    
    let result = animation(&[chart, transition_type, duration]);
    assert!(result.is_ok());
    
    if let Ok(Value::LyObj(obj)) = result {
        assert!(obj.as_any().is::<AnimationObj>());
        
        if let Some(animation_obj) = obj.as_any().downcast_ref::<AnimationObj>() {
            assert!(matches!(animation_obj.config.transition_type, TransitionType::FadeIn));
            assert_eq!(animation_obj.config.duration, 1500);
            assert!(!animation_obj.is_playing);
        }
    } else {
        panic!("Expected LyObj with AnimationObj");
    }
}

#[test]
fn test_animation_transition_types() {
    let chart = create_mock_chart();
    let duration = Value::Number(1000.0);
    
    let transitions = vec![
        ("fade_in", TransitionType::FadeIn),
        ("fade_out", TransitionType::FadeOut),
        ("slide_left", TransitionType::SlideIn { direction: Direction::Left }),
        ("slide_right", TransitionType::SlideIn { direction: Direction::Right }),
        ("scale", TransitionType::Scale { from: 0.0, to: 1.0 }),
    ];
    
    for (transition_str, expected_transition) in transitions {
        let transition_type = Value::String(transition_str.to_string());
        let result = animation(&[chart.clone(), transition_type, duration.clone()]);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(obj)) = result {
            if let Some(animation_obj) = obj.as_any().downcast_ref::<AnimationObj>() {
                match (&animation_obj.config.transition_type, &expected_transition) {
                    (TransitionType::FadeIn, TransitionType::FadeIn) => {},
                    (TransitionType::FadeOut, TransitionType::FadeOut) => {},
                    (TransitionType::SlideIn { direction: Direction::Left }, 
                     TransitionType::SlideIn { direction: Direction::Left }) => {},
                    (TransitionType::SlideIn { direction: Direction::Right }, 
                     TransitionType::SlideIn { direction: Direction::Right }) => {},
                    (TransitionType::Scale { from: f1, to: t1 }, 
                     TransitionType::Scale { from: f2, to: t2 }) => {
                        assert_eq!(f1, f2);
                        assert_eq!(t1, t2);
                    },
                    _ => panic!("Transition type mismatch for {}", transition_str),
                }
            }
        }
    }
}

#[test]
fn test_cross_filter_creation() {
    let charts = Value::List(vec![
        Value::String("chart1".to_string()),
        Value::String("chart2".to_string()),
        Value::String("chart3".to_string()),
    ]);
    let data_field = Value::String("category".to_string());
    let filter_interactions = Value::String("equals".to_string());
    
    let result = cross_filter(&[charts, data_field, filter_interactions]);
    assert!(result.is_ok());
    
    if let Ok(Value::LyObj(obj)) = result {
        assert!(obj.as_any().is::<CrossFilterObj>());
        
        if let Some(cross_filter_obj) = obj.as_any().downcast_ref::<CrossFilterObj>() {
            assert_eq!(cross_filter_obj.config.source_charts.len(), 3);
            assert_eq!(cross_filter_obj.config.filter_field, "category");
            assert!(matches!(cross_filter_obj.config.filter_operation, FilterOperation::Equals));
            assert!(cross_filter_obj.active_filters.is_empty());
        }
    } else {
        panic!("Expected LyObj with CrossFilterObj");
    }
}

#[test]
fn test_filter_operations() {
    let charts = Value::List(vec![Value::String("chart1".to_string())]);
    let field = Value::String("value".to_string());
    
    let operations = vec![
        ("equals", FilterOperation::Equals),
        ("contains", FilterOperation::Contains),
        ("range", FilterOperation::Range),
        ("greater_than", FilterOperation::GreaterThan),
        ("less_than", FilterOperation::LessThan),
        ("in", FilterOperation::In),
        ("not_in", FilterOperation::NotIn),
    ];
    
    for (op_str, expected_op) in operations {
        let operation = Value::String(op_str.to_string());
        let result = cross_filter(&[charts.clone(), field.clone(), operation]);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(obj)) = result {
            if let Some(cross_filter_obj) = obj.as_any().downcast_ref::<CrossFilterObj>() {
                match (&cross_filter_obj.config.filter_operation, &expected_op) {
                    (FilterOperation::Equals, FilterOperation::Equals) => {},
                    (FilterOperation::Contains, FilterOperation::Contains) => {},
                    (FilterOperation::Range, FilterOperation::Range) => {},
                    (FilterOperation::GreaterThan, FilterOperation::GreaterThan) => {},
                    (FilterOperation::LessThan, FilterOperation::LessThan) => {},
                    (FilterOperation::In, FilterOperation::In) => {},
                    (FilterOperation::NotIn, FilterOperation::NotIn) => {},
                    _ => panic!("Filter operation mismatch for {}", op_str),
                }
            }
        }
    }
}

#[test]
fn test_tooltip_config_defaults() {
    let config = TooltipConfig::default();
    assert_eq!(config.content_template, "{x}: {y}");
    assert!(matches!(config.position, TooltipPosition::Mouse));
    assert!(matches!(config.trigger, TooltipTrigger::Hover));
    assert_eq!(config.delay, 100);
    assert!(config.enabled);
    assert_eq!(config.style.background_color, "rgba(0, 0, 0, 0.8)");
    assert_eq!(config.style.text_color, "#ffffff");
}

#[test]
fn test_zoom_config_defaults() {
    let config = ZoomConfig::default();
    assert_eq!(config.axes.len(), 2);
    assert!(config.axes.contains(&"x".to_string()));
    assert!(config.axes.contains(&"y".to_string()));
    assert_eq!(config.min_zoom, 0.1);
    assert_eq!(config.max_zoom, 10.0);
    assert_eq!(config.zoom_factor, 1.1);
    assert!(config.pan_enabled);
    assert_eq!(config.animation_duration, 300);
}

#[test]
fn test_selection_config_defaults() {
    let config = SelectionConfig::default();
    assert!(matches!(config.mode, SelectionMode::Point));
    assert!(!config.multi_select);
    assert!(config.callback.is_none());
    assert_eq!(config.highlight_style.color, "#ff6b6b");
    assert_eq!(config.highlight_style.opacity, 0.8);
}

#[test]
fn test_animation_config_defaults() {
    let config = AnimationConfig::default();
    assert!(matches!(config.transition_type, TransitionType::FadeIn));
    assert_eq!(config.duration, 1000);
    assert!(matches!(config.easing, EasingFunction::EaseInOut));
    assert_eq!(config.delay, 0);
    assert!(!config.loop_animation);
    assert!(config.auto_play);
}

#[test]
fn test_cross_filter_config_defaults() {
    let config = CrossFilterConfig::default();
    assert!(config.source_charts.is_empty());
    assert!(config.target_charts.is_empty());
    assert_eq!(config.filter_field, "value");
    assert!(matches!(config.filter_operation, FilterOperation::Equals));
    assert!(!config.sync_zoom);
    assert!(config.sync_selection);
    assert_eq!(config.debounce_delay, 300);
}

#[test]
fn test_tooltip_javascript_generation() {
    let config = TooltipConfig::default();
    let tooltip = TooltipObj::new(config, "test_chart".to_string());
    
    let js_code = tooltip.generate_js();
    assert!(js_code.contains("class Tooltip"));
    assert!(js_code.contains("constructor(chartId, config)"));
    assert!(js_code.contains("attachEvents"));
    assert!(js_code.contains("show(event)"));
    assert!(js_code.contains("hide()"));
    assert!(js_code.contains("test_chart"));
}

#[test]
fn test_zoom_javascript_generation() {
    let config = ZoomConfig::default();
    let zoom = ZoomObj::new(config, "test_chart".to_string());
    
    let js_code = zoom.generate_js();
    assert!(js_code.contains("class ZoomController"));
    assert!(js_code.contains("handleWheel(event)"));
    assert!(js_code.contains("handleMouseDown(event)"));
    assert!(js_code.contains("applyTransform()"));
    assert!(js_code.contains("reset()"));
    assert!(js_code.contains("test_chart"));
}

#[test]
fn test_invalid_interactive_arguments() {
    // Test tooltip with insufficient arguments
    let chart = create_mock_chart();
    let content = Value::String("test".to_string());
    
    let result = tooltip(&[chart, content]); // Missing position
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("requires chart, content_template, and position"));
    
    // Test zoom with insufficient arguments
    let chart = create_mock_chart();
    let axes = Value::List(vec![Value::String("x".to_string())]);
    
    let result = zoom(&[chart, axes]); // Missing constraints
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("requires chart, axes, and constraints"));
    
    // Test animation with insufficient arguments
    let chart = create_mock_chart();
    let transition = Value::String("fade_in".to_string());
    
    let result = animation(&[chart, transition]); // Missing duration
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("requires chart, transition_type, and duration"));
}

#[test]
fn test_interactive_object_display() {
    let tooltip_config = TooltipConfig::default();
    let tooltip = TooltipObj::new(tooltip_config, "chart1".to_string());
    let display = format!("{}", tooltip);
    assert!(display.contains("Tooltip"));
    assert!(display.contains("chart1"));
    assert!(display.contains("active: false"));
    
    let zoom_config = ZoomConfig::default();
    let zoom = ZoomObj::new(zoom_config, "chart2".to_string());
    let display = format!("{}", zoom);
    assert!(display.contains("Zoom"));
    assert!(display.contains("chart2"));
    assert!(display.contains("zoom: 1.00x"));
    
    let selection_config = SelectionConfig::default();
    let selection = SelectionObj::new(selection_config, "chart3".to_string());
    let display = format!("{}", selection);
    assert!(display.contains("Selection"));
    assert!(display.contains("chart3"));
    assert!(display.contains("items: 0"));
}