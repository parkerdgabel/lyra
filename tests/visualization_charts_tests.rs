use lyra::vm::{Value, VmResult};
use lyra::stdlib::visualization::charts::*;
use lyra::foreign::{LyObj, Foreign};

// Test data structures for chart validation
fn create_test_chart_data() -> Value {
    // Create test data: [(x, y), ...]
    Value::List(vec![
        Value::List(vec![Value::Number(1.0), Value::Number(10.0)]),
        Value::List(vec![Value::Number(2.0), Value::Number(20.0)]),
        Value::List(vec![Value::Number(3.0), Value::Number(15.0)]),
        Value::List(vec![Value::Number(4.0), Value::Number(25.0)]),
        Value::List(vec![Value::Number(5.0), Value::Number(30.0)]),
    ])
}

fn create_test_series_data() -> Value {
    Value::List(vec![
        Value::Number(10.0),
        Value::Number(20.0),
        Value::Number(15.0),
        Value::Number(25.0),
        Value::Number(30.0),
    ])
}

fn create_test_labels() -> Value {
    Value::List(vec![
        Value::String("A".to_string()),
        Value::String("B".to_string()),
        Value::String("C".to_string()),
        Value::String("D".to_string()),
        Value::String("E".to_string()),
    ])
}

fn create_chart_options() -> Value {
    Value::List(vec![
        Value::List(vec![
            Value::String("title".to_string()),
            Value::String("Test Chart".to_string()),
        ]),
        Value::List(vec![
            Value::String("width".to_string()),
            Value::Number(800.0),
        ]),
        Value::List(vec![
            Value::String("height".to_string()),
            Value::Number(600.0),
        ]),
    ])
}

#[test]
fn test_line_chart_creation() {
    let data = create_test_chart_data();
    let x_axis = Value::String("x".to_string());
    let y_axis = Value::String("y".to_string());
    let options = create_chart_options();
    
    let result = line_chart(&[data, x_axis, y_axis, options]);
    assert!(result.is_ok());
    
    // Verify result is a LyObj containing a Chart
    if let Ok(Value::LyObj(obj)) = result {
        assert!(obj.as_any().is::<LineChartObj>());
    } else {
        panic!("Expected LyObj with LineChart");
    }
}

#[test]
fn test_bar_chart_creation() {
    let data = create_test_series_data();
    let categories = create_test_labels();
    let values = Value::String("values".to_string());
    let orientation = Value::String("vertical".to_string());
    
    let result = bar_chart(&[data, categories, values, orientation]);
    assert!(result.is_ok());
    
    if let Ok(Value::LyObj(obj)) = result {
        assert!(obj.as_any().is::<BarChartObj>());
    } else {
        panic!("Expected LyObj with BarChart");
    }
}

#[test]
fn test_scatter_plot_creation() {
    let data = create_test_chart_data();
    let x_values = Value::String("x".to_string());
    let y_values = Value::String("y".to_string());
    let options = create_chart_options();
    
    let result = scatter_plot(&[data, x_values, y_values, options]);
    assert!(result.is_ok());
    
    if let Ok(Value::LyObj(obj)) = result {
        assert!(obj.as_any().is::<ScatterPlotObj>());
    } else {
        panic!("Expected LyObj with ScatterPlot");
    }
}

#[test]
fn test_pie_chart_creation() {
    let data = create_test_series_data();
    let labels = create_test_labels();
    let values = Value::String("values".to_string());
    let options = create_chart_options();
    
    let result = pie_chart(&[data, labels, values, options]);
    assert!(result.is_ok());
    
    if let Ok(Value::LyObj(obj)) = result {
        assert!(obj.as_any().is::<PieChartObj>());
    } else {
        panic!("Expected LyObj with PieChart");
    }
}

#[test]
fn test_heatmap_creation() {
    // Create 2D data for heatmap
    let data = Value::List(vec![
        Value::List(vec![Value::Number(1.0), Value::Number(2.0), Value::Number(3.0)]),
        Value::List(vec![Value::Number(4.0), Value::Number(5.0), Value::Number(6.0)]),
        Value::List(vec![Value::Number(7.0), Value::Number(8.0), Value::Number(9.0)]),
    ]);
    let x_labels = Value::List(vec![
        Value::String("X1".to_string()),
        Value::String("X2".to_string()),
        Value::String("X3".to_string()),
    ]);
    let y_labels = Value::List(vec![
        Value::String("Y1".to_string()),
        Value::String("Y2".to_string()),
        Value::String("Y3".to_string()),
    ]);
    let color_scale = Value::String("viridis".to_string());
    
    let result = heatmap(&[data, x_labels, y_labels, color_scale]);
    assert!(result.is_ok());
    
    if let Ok(Value::LyObj(obj)) = result {
        assert!(obj.as_any().is::<HeatmapObj>());
    } else {
        panic!("Expected LyObj with Heatmap");
    }
}

#[test]
fn test_histogram_creation() {
    let data = create_test_series_data();
    let bins = Value::Number(10.0);
    let options = create_chart_options();
    
    let result = histogram(&[data, bins, options]);
    assert!(result.is_ok());
    
    if let Ok(Value::LyObj(obj)) = result {
        assert!(obj.as_any().is::<HistogramObj>());
    } else {
        panic!("Expected LyObj with Histogram");
    }
}

#[test]
fn test_box_plot_creation() {
    let data = create_test_chart_data();
    let groups = create_test_labels();
    let values = Value::String("values".to_string());
    let outliers = Value::Boolean(true);
    
    let result = box_plot(&[data, groups, values, outliers]);
    assert!(result.is_ok());
    
    if let Ok(Value::LyObj(obj)) = result {
        assert!(obj.as_any().is::<BoxPlotObj>());
    } else {
        panic!("Expected LyObj with BoxPlot");
    }
}

#[test]
fn test_area_chart_creation() {
    let data = create_test_chart_data();
    let x_axis = Value::String("x".to_string());
    let y_series = Value::String("y".to_string());
    let stacking = Value::String("normal".to_string());
    
    let result = area_chart(&[data, x_axis, y_series, stacking]);
    assert!(result.is_ok());
    
    if let Ok(Value::LyObj(obj)) = result {
        assert!(obj.as_any().is::<AreaChartObj>());
    } else {
        panic!("Expected LyObj with AreaChart");
    }
}

#[test]
fn test_bubble_chart_creation() {
    // Create bubble data with x, y, size, color
    let data = Value::List(vec![
        Value::List(vec![
            Value::Number(1.0), Value::Number(10.0), 
            Value::Number(5.0), Value::String("red".to_string())
        ]),
        Value::List(vec![
            Value::Number(2.0), Value::Number(20.0), 
            Value::Number(10.0), Value::String("blue".to_string())
        ]),
    ]);
    let x = Value::String("x".to_string());
    let y = Value::String("y".to_string());
    let size = Value::String("size".to_string());
    let color = Value::String("color".to_string());
    let options = create_chart_options();
    
    let result = bubble_chart(&[data, x, y, size, color, options]);
    assert!(result.is_ok());
    
    if let Ok(Value::LyObj(obj)) = result {
        assert!(obj.as_any().is::<BubbleChartObj>());
    } else {
        panic!("Expected LyObj with BubbleChart");
    }
}

#[test]
fn test_candlestick_creation() {
    // Create OHLC data
    let data = Value::List(vec![
        Value::List(vec![
            Value::Number(100.0), // open
            Value::Number(110.0), // high
            Value::Number(95.0),  // low
            Value::Number(105.0), // close
            Value::Number(1000.0) // volume
        ]),
    ]);
    let open = Value::String("open".to_string());
    let high = Value::String("high".to_string());
    let low = Value::String("low".to_string());
    let close = Value::String("close".to_string());
    let volume = Value::String("volume".to_string());
    
    let result = candlestick(&[data, open, high, low, close, volume]);
    assert!(result.is_ok());
    
    if let Ok(Value::LyObj(obj)) = result {
        assert!(obj.as_any().is::<CandlestickObj>());
    } else {
        panic!("Expected LyObj with Candlestick");
    }
}

#[test]
fn test_radar_chart_creation() {
    let data = create_test_chart_data();
    let dimensions = create_test_labels();
    let series = Value::String("series".to_string());
    let options = create_chart_options();
    
    let result = radar_chart(&[data, dimensions, series, options]);
    assert!(result.is_ok());
    
    if let Ok(Value::LyObj(obj)) = result {
        assert!(obj.as_any().is::<RadarChartObj>());
    } else {
        panic!("Expected LyObj with RadarChart");
    }
}

#[test]
fn test_tree_map_creation() {
    // Create hierarchical data
    let data = Value::List(vec![
        Value::List(vec![
            Value::String("Root".to_string()),
            Value::String("Child1".to_string()),
            Value::Number(100.0),
            Value::String("red".to_string()),
        ]),
    ]);
    let hierarchy = Value::String("hierarchy".to_string());
    let size = Value::String("size".to_string());
    let color = Value::String("color".to_string());
    let options = create_chart_options();
    
    let result = tree_map(&[data, hierarchy, size, color, options]);
    assert!(result.is_ok());
    
    if let Ok(Value::LyObj(obj)) = result {
        assert!(obj.as_any().is::<TreeMapObj>());
    } else {
        panic!("Expected LyObj with TreeMap");
    }
}

#[test]
fn test_sankey_diagram_creation() {
    // Create flow data
    let data = Value::List(vec![]);
    let nodes = Value::List(vec![
        Value::String("Source".to_string()),
        Value::String("Target".to_string()),
    ]);
    let links = Value::List(vec![
        Value::List(vec![
            Value::String("Source".to_string()),
            Value::String("Target".to_string()),
            Value::Number(100.0),
        ]),
    ]);
    let options = create_chart_options();
    
    let result = sankey_diagram(&[data, nodes, links, options]);
    assert!(result.is_ok());
    
    if let Ok(Value::LyObj(obj)) = result {
        assert!(obj.as_any().is::<SankeyDiagramObj>());
    } else {
        panic!("Expected LyObj with SankeyDiagram");
    }
}

#[test]
fn test_network_diagram_creation() {
    let nodes = Value::List(vec![
        Value::String("Node1".to_string()),
        Value::String("Node2".to_string()),
    ]);
    let edges = Value::List(vec![
        Value::List(vec![
            Value::String("Node1".to_string()),
            Value::String("Node2".to_string()),
        ]),
    ]);
    let layout = Value::String("force_directed".to_string());
    let options = create_chart_options();
    
    let result = network_diagram(&[nodes, edges, layout, options]);
    assert!(result.is_ok());
    
    if let Ok(Value::LyObj(obj)) = result {
        assert!(obj.as_any().is::<NetworkDiagramObj>());
    } else {
        panic!("Expected LyObj with NetworkDiagram");
    }
}

#[test]
fn test_gantt_chart_creation() {
    // Create task data
    let tasks = Value::List(vec![
        Value::List(vec![
            Value::String("Task1".to_string()),
            Value::String("2024-01-01".to_string()), // start
            Value::String("2024-01-10".to_string()), // end
            Value::List(vec![]), // dependencies
        ]),
    ]);
    let start_date = Value::String("start".to_string());
    let end_date = Value::String("end".to_string());
    let dependencies = Value::String("deps".to_string());
    
    let result = gantt_chart(&[tasks, start_date, end_date, dependencies]);
    assert!(result.is_ok());
    
    if let Ok(Value::LyObj(obj)) = result {
        assert!(obj.as_any().is::<GanttChartObj>());
    } else {
        panic!("Expected LyObj with GanttChart");
    }
}

#[test]
fn test_invalid_chart_data() {
    // Test with invalid data
    let invalid_data = Value::String("invalid".to_string());
    let x_axis = Value::String("x".to_string());
    let y_axis = Value::String("y".to_string());
    let options = create_chart_options();
    
    let result = line_chart(&[invalid_data, x_axis, y_axis, options]);
    assert!(result.is_err());
}

#[test]
fn test_missing_chart_arguments() {
    // Test with insufficient arguments
    let data = create_test_chart_data();
    
    let result = line_chart(&[data]);
    assert!(result.is_err());
}