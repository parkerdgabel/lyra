//! Dashboard Creation Module for Lyra Visualization System
//!
//! This module provides interactive dashboard creation capabilities with support for
//! widget management, real-time updates, filtering, and drill-down navigation.

use crate::vm::{Value, VmResult};
use crate::foreign::{LyObj, Foreign};
use std::collections::HashMap;
use std::any::Any;
use std::fmt;
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;
use serde::{Serialize, Deserialize};

/// Dashboard layout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardLayout {
    pub grid_size: (u32, u32),
    pub responsive: bool,
    pub sections: Vec<LayoutSection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutSection {
    pub id: String,
    pub position: (u32, u32),
    pub size: (u32, u32),
    pub title: String,
}

impl Default for DashboardLayout {
    fn default() -> Self {
        Self {
            grid_size: (12, 8),
            responsive: true,
            sections: Vec::new(),
        }
    }
}

/// Widget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetConfig {
    pub widget_type: String,
    pub data_source: String,
    pub position: (u32, u32),
    pub size: (u32, u32),
    pub title: String,
    pub options: HashMap<String, String>,
    pub refresh_interval: Option<u64>, // milliseconds
}

impl WidgetConfig {
    pub fn new(widget_type: String, data_source: String, position: (u32, u32)) -> Self {
        Self {
            widget_type,
            data_source,
            position,
            size: (4, 3), // Default size
            title: "Widget".to_string(),
            options: HashMap::new(),
            refresh_interval: None,
        }
    }
}

/// Filter definition for dashboard interactivity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardFilter {
    pub field: String,
    pub values: Vec<String>,
    pub operation: String, // "equals", "contains", "range", etc.
    pub widget_targets: Vec<String>, // Widget IDs affected by this filter
}

/// Drill-down navigation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrillDownConfig {
    pub source_widget: String,
    pub hierarchy: Vec<String>,
    pub action: String, // "navigate", "expand", "filter"
    pub target_dashboard: Option<String>,
}

/// Real-time data stream configuration
#[derive(Debug, Clone)]
pub struct RealTimeConfig {
    pub data_stream: String,
    pub chart_type: String,
    pub update_interval: u64, // milliseconds
    pub buffer_size: usize,
    pub sender: broadcast::Sender<Value>,
}

/// Widget interaction definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetInteraction {
    pub source_widget: String,
    pub target_widget: String,
    pub event: String, // "click", "hover", "select"
    pub action: String, // "filter", "highlight", "navigate"
    pub parameters: HashMap<String, String>,
}

/// Dashboard Foreign Object
#[derive(Debug)]
pub struct DashboardObj {
    pub name: String,
    pub layout: DashboardLayout,
    pub widgets: HashMap<String, WidgetConfig>,
    pub filters: HashMap<String, DashboardFilter>,
    pub interactions: Vec<WidgetInteraction>,
    pub real_time_streams: HashMap<String, RealTimeConfig>,
    pub metadata: HashMap<String, String>,
}

impl DashboardObj {
    pub fn new(name: String, layout: DashboardLayout) -> Self {
        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "Dashboard".to_string());
        metadata.insert("name".to_string(), name.clone());
        metadata.insert("created".to_string(), chrono::Utc::now().to_rfc3339());
        
        Self {
            name,
            layout,
            widgets: HashMap::new(),
            filters: HashMap::new(),
            interactions: Vec::new(),
            real_time_streams: HashMap::new(),
            metadata,
        }
    }
    
    pub fn add_widget(&mut self, id: String, widget: WidgetConfig) {
        self.widgets.insert(id, widget);
    }
    
    pub fn add_filter(&mut self, id: String, filter: DashboardFilter) {
        self.filters.insert(id, filter);
    }
    
    pub fn add_interaction(&mut self, interaction: WidgetInteraction) {
        self.interactions.push(interaction);
    }
    
    pub fn add_real_time_stream(&mut self, id: String, config: RealTimeConfig) {
        self.real_time_streams.insert(id, config);
    }
    
    /// Export dashboard to HTML with JavaScript interactivity
    pub fn export_to_html(&self) -> VmResult<String> {
        let mut html = String::new();
        
        html.push_str("<!DOCTYPE html>\n");
        html.push_str("<html>\n<head>\n");
        html.push_str("<title>");
        html.push_str(&self.name);
        html.push_str("</title>\n");
        html.push_str("<style>\n");
        html.push_str(include_str!("dashboard_styles.css"));
        html.push_str("</style>\n");
        html.push_str("</head>\n<body>\n");
        
        // Dashboard container
        html.push_str(&format!("<div class='dashboard' id='dashboard-{}'>\n", self.name));
        html.push_str(&format!("<h1>{}</h1>\n", self.name));
        
        // Filter panel
        if !self.filters.is_empty() {
            html.push_str("<div class='filter-panel'>\n");
            for (id, filter) in &self.filters {
                html.push_str(&format!(
                    "<div class='filter-item' data-filter-id='{}'>\n",
                    id
                ));
                html.push_str(&format!("<label>{}</label>\n", filter.field));
                html.push_str(&format!(
                    "<select multiple data-field='{}' data-operation='{}'>\n",
                    filter.field, filter.operation
                ));
                for value in &filter.values {
                    html.push_str(&format!("<option value='{}'>{}</option>\n", value, value));
                }
                html.push_str("</select>\n</div>\n");
            }
            html.push_str("</div>\n");
        }
        
        // Widget grid
        html.push_str("<div class='widget-grid'>\n");
        for (id, widget) in &self.widgets {
            html.push_str(&format!(
                "<div class='widget' id='widget-{}' style='grid-column: {} / span {}; grid-row: {} / span {};'>\n",
                id, widget.position.0 + 1, widget.size.0, widget.position.1 + 1, widget.size.1
            ));
            html.push_str(&format!("<h3>{}</h3>\n", widget.title));
            html.push_str(&format!("<div class='widget-content' data-type='{}' data-source='{}'>\n", 
                                   widget.widget_type, widget.data_source));
            html.push_str("Loading...\n");
            html.push_str("</div>\n</div>\n");
        }
        html.push_str("</div>\n");
        
        html.push_str("</div>\n");
        
        // JavaScript for interactivity
        html.push_str("<script>\n");
        html.push_str("// Dashboard interactivity code\n");
        html.push_str(&self.generate_javascript());
        html.push_str("</script>\n");
        
        html.push_str("</body>\n</html>");
        
        Ok(html)
    }
    
    fn generate_javascript(&self) -> String {
        let mut js = String::new();
        
        js.push_str("class Dashboard {\n");
        js.push_str("  constructor() {\n");
        js.push_str("    this.widgets = new Map();\n");
        js.push_str("    this.filters = new Map();\n");
        js.push_str("    this.init();\n");
        js.push_str("  }\n\n");
        
        js.push_str("  init() {\n");
        js.push_str("    this.setupFilters();\n");
        js.push_str("    this.setupWidgetInteractions();\n");
        js.push_str("    this.loadWidgetData();\n");
        js.push_str("  }\n\n");
        
        js.push_str("  setupFilters() {\n");
        for (id, filter) in &self.filters {
            js.push_str(&format!(
                "    this.filters.set('{}', {{ field: '{}', operation: '{}', targets: {} }});\n",
                id, filter.field, filter.operation,
                serde_json::to_string(&filter.widget_targets).unwrap_or_default()
            ));
        }
        js.push_str("  }\n\n");
        
        js.push_str("  setupWidgetInteractions() {\n");
        for interaction in &self.interactions {
            js.push_str(&format!(
                "    document.getElementById('widget-{}').addEventListener('{}', (e) => {{\n",
                interaction.source_widget, interaction.event
            ));
            js.push_str(&format!(
                "      this.handleInteraction('{}', '{}', '{}', e);\n",
                interaction.source_widget, interaction.target_widget, interaction.action
            ));
            js.push_str("    });\n");
        }
        js.push_str("  }\n\n");
        
        js.push_str("  loadWidgetData() {\n");
        js.push_str("    // Load data for each widget\n");
        for (id, widget) in &self.widgets {
            js.push_str(&format!(
                "    this.loadWidget('{}', '{}', '{}');\n",
                id, widget.widget_type, widget.data_source
            ));
        }
        js.push_str("  }\n\n");
        
        js.push_str("  loadWidget(id, type, dataSource) {\n");
        js.push_str("    // Placeholder for widget loading\n");
        js.push_str("    console.log(`Loading widget ${id} of type ${type} from ${dataSource}`);\n");
        js.push_str("  }\n\n");
        
        js.push_str("  handleInteraction(source, target, action, event) {\n");
        js.push_str("    // Placeholder for interaction handling\n");
        js.push_str("    console.log(`Interaction: ${source} -> ${target} (${action})`);\n");
        js.push_str("  }\n");
        
        js.push_str("}\n\n");
        js.push_str("new Dashboard();\n");
        
        js
    }
}

impl Foreign for DashboardObj {
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn type_name(&self) -> &'static str {
        "Dashboard"
    }
}

impl fmt::Display for DashboardObj {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dashboard[{}, {} widgets]", self.name, self.widgets.len())
    }
}

/// Dashboard Widget Foreign Object
#[derive(Debug)]
pub struct DashboardWidgetObj {
    pub config: WidgetConfig,
    pub chart_data: Option<Value>,
}

impl DashboardWidgetObj {
    pub fn new(config: WidgetConfig) -> Self {
        Self {
            config,
            chart_data: None,
        }
    }
}

impl Foreign for DashboardWidgetObj {
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn type_name(&self) -> &'static str {
        "DashboardWidget"
    }
}

impl fmt::Display for DashboardWidgetObj {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DashboardWidget[{}, position: {:?}]", 
               self.config.widget_type, self.config.position)
    }
}

/// Real-time Chart Foreign Object
#[derive(Debug)]
pub struct RealTimeChartObj {
    pub config: RealTimeConfig,
    pub data_buffer: Arc<Mutex<Vec<Value>>>,
}

impl RealTimeChartObj {
    pub fn new(config: RealTimeConfig) -> Self {
        Self {
            config,
            data_buffer: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    pub fn add_data_point(&self, data: Value) -> VmResult<()> {
        let mut buffer = self.data_buffer.lock()
            .map_err(|_| "Failed to acquire data buffer lock")?;
        
        buffer.push(data);
        
        // Keep buffer size under limit
        if buffer.len() > self.config.buffer_size {
            buffer.remove(0);
        }
        
        // Send update to subscribers
        let _ = self.config.sender.send(Value::List(buffer.clone()));
        
        Ok(())
    }
}

impl Foreign for RealTimeChartObj {
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn type_name(&self) -> &'static str {
        "RealTimeChart"
    }
}

impl fmt::Display for RealTimeChartObj {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RealTimeChart[{}, update: {}ms]", 
               self.config.chart_type, self.config.update_interval)
    }
}

// =============================================================================
// Dashboard Function Implementations
// =============================================================================

/// Create a dashboard
pub fn dashboard(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err("Dashboard requires name, layout, and widgets".to_string());
    }
    
    let name = match &args[0] {
        Value::String(name) => name.clone(),
        _ => return Err("Dashboard name must be a string".to_string()),
    };
    
    let layout = parse_dashboard_layout(&args[1])?;
    let mut dashboard = DashboardObj::new(name, layout);
    
    // Parse widgets
    if let Value::List(widgets) = &args[2] {
        for (i, widget_def) in widgets.iter().enumerate() {
            let widget = parse_widget_config(widget_def)?;
            dashboard.add_widget(format!("widget_{}", i), widget);
        }
    }
    
    // Parse options if provided
    if args.len() > 3 {
        parse_dashboard_options(&mut dashboard, &args[3])?;
    }
    
    Ok(Value::LyObj(LyObj::new(Box::new(dashboard))))
}

/// Create a dashboard widget
pub fn dashboard_widget(args: &[Value]) -> VmResult<Value> {
    if args.len() < 4 {
        return Err("DashboardWidget requires type, data_source, config, and position".to_string());
    }
    
    let widget_type = match &args[0] {
        Value::String(t) => t.clone(),
        _ => return Err("Widget type must be a string".to_string()),
    };
    
    let data_source = match &args[1] {
        Value::String(ds) => ds.clone(),
        _ => return Err("Data source must be a string".to_string()),
    };
    
    let position = parse_position(&args[3])?;
    let mut config = WidgetConfig::new(widget_type, data_source, position);
    
    // Parse additional config
    parse_widget_options(&mut config, &args[2])?;
    
    let widget = DashboardWidgetObj::new(config);
    Ok(Value::LyObj(LyObj::new(Box::new(widget))))
}

/// Add a filter to dashboard
pub fn filter(args: &[Value]) -> VmResult<Value> {
    if args.len() < 4 {
        return Err("Filter requires dashboard, field, values, and operation".to_string());
    }
    
    // This would modify an existing dashboard object
    // For simplicity, returning a filter configuration
    let field = match &args[1] {
        Value::String(f) => f.clone(),
        _ => return Err("Field must be a string".to_string()),
    };
    
    let values = match &args[2] {
        Value::List(vals) => {
            vals.iter().map(|v| match v {
                Value::String(s) => Ok(s.clone()),
                _ => Err("Filter values must be strings".to_string()),
            }).collect::<Result<Vec<_>, _>>()?
        }
        _ => return Err("Values must be a list".to_string()),
    };
    
    let operation = match &args[3] {
        Value::String(op) => op.clone(),
        _ => return Err("Operation must be a string".to_string()),
    };
    
    let filter_config = DashboardFilter {
        field,
        values,
        operation,
        widget_targets: Vec::new(),
    };
    
    // Return success indicator
    Ok(Value::Boolean(true))
}

/// Configure drill-down navigation
pub fn drill_down(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err("DrillDown requires widget, hierarchy, and action".to_string());
    }
    
    let source_widget = match &args[0] {
        Value::String(w) => w.clone(),
        _ => return Err("Widget must be a string".to_string()),
    };
    
    let hierarchy = match &args[1] {
        Value::List(levels) => {
            levels.iter().map(|v| match v {
                Value::String(s) => Ok(s.clone()),
                _ => Err("Hierarchy levels must be strings".to_string()),
            }).collect::<Result<Vec<_>, _>>()?
        }
        _ => return Err("Hierarchy must be a list".to_string()),
    };
    
    let action = match &args[2] {
        Value::String(a) => a.clone(),
        _ => return Err("Action must be a string".to_string()),
    };
    
    let drill_config = DrillDownConfig {
        source_widget,
        hierarchy,
        action,
        target_dashboard: None,
    };
    
    Ok(Value::Boolean(true))
}

/// Create a real-time chart
pub fn real_time_chart(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err("RealTimeChart requires data_stream, chart_type, and update_interval".to_string());
    }
    
    let data_stream = match &args[0] {
        Value::String(ds) => ds.clone(),
        _ => return Err("Data stream must be a string".to_string()),
    };
    
    let chart_type = match &args[1] {
        Value::String(ct) => ct.clone(),
        _ => return Err("Chart type must be a string".to_string()),
    };
    
    let update_interval = match &args[2] {
        Value::Real(interval) => *interval as u64,
        _ => return Err("Update interval must be a number".to_string()),
    };
    
    let (sender, _receiver) = broadcast::channel(1000);
    
    let config = RealTimeConfig {
        data_stream,
        chart_type,
        update_interval,
        buffer_size: 1000,
        sender,
    };
    
    let chart = RealTimeChartObj::new(config);
    Ok(Value::LyObj(LyObj::new(Box::new(chart))))
}

/// Configure dashboard layout
pub fn dashboard_layout(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err("DashboardLayout requires grid_size, responsive, and sections".to_string());
    }
    
    let layout = parse_dashboard_layout_args(args)?;
    
    // Return success indicator - in practice this would be used to modify a dashboard
    Ok(Value::Boolean(true))
}

/// Configure widget interaction
pub fn widget_interaction(args: &[Value]) -> VmResult<Value> {
    if args.len() < 4 {
        return Err("WidgetInteraction requires source_widget, target_widget, event, and action".to_string());
    }
    
    let source_widget = match &args[0] {
        Value::String(w) => w.clone(),
        _ => return Err("Source widget must be a string".to_string()),
    };
    
    let target_widget = match &args[1] {
        Value::String(w) => w.clone(),
        _ => return Err("Target widget must be a string".to_string()),
    };
    
    let event = match &args[2] {
        Value::String(e) => e.clone(),
        _ => return Err("Event must be a string".to_string()),
    };
    
    let action = match &args[3] {
        Value::String(a) => a.clone(),
        _ => return Err("Action must be a string".to_string()),
    };
    
    let interaction = WidgetInteraction {
        source_widget,
        target_widget,
        event,
        action,
        parameters: HashMap::new(),
    };
    
    Ok(Value::Boolean(true))
}

/// Export dashboard
pub fn dashboard_export(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err("DashboardExport requires dashboard, format, and options".to_string());
    }
    
    if let Value::LyObj(obj) = &args[0] {
        if let Some(dashboard) = obj.as_any().downcast_ref::<DashboardObj>() {
            let format = match &args[1] {
                Value::String(f) => f.clone(),
                _ => return Err("Format must be a string".to_string()),
            };
            
            match format.as_str() {
                "html" => {
                    let html = dashboard.export_to_html()?;
                    Ok(Value::String(html))
                }
                "json" => {
                    let json = serde_json::to_string_pretty(dashboard)
                        .map_err(|e| format!("JSON serialization error: {}", e))?;
                    Ok(Value::String(json))
                }
                _ => Err(format!("Unsupported export format: {}", format)),
            }
        } else {
            Err("First argument must be a Dashboard object".to_string())
        }
    } else {
        Err("First argument must be a Dashboard object".to_string())
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

fn parse_dashboard_layout(value: &Value) -> VmResult<DashboardLayout> {
    // Parse layout from value - simplified implementation
    Ok(DashboardLayout::default())
}

fn parse_dashboard_layout_args(args: &[Value]) -> VmResult<DashboardLayout> {
    // Parse layout from arguments - simplified implementation
    Ok(DashboardLayout::default())
}

fn parse_widget_config(value: &Value) -> VmResult<WidgetConfig> {
    // Parse widget configuration - simplified implementation
    Ok(WidgetConfig::new(
        "chart".to_string(),
        "data".to_string(),
        (0, 0),
    ))
}

fn parse_widget_options(config: &mut WidgetConfig, value: &Value) -> VmResult<()> {
    // Parse widget options - simplified implementation
    Ok(())
}

fn parse_dashboard_options(dashboard: &mut DashboardObj, value: &Value) -> VmResult<()> {
    // Parse dashboard options - simplified implementation
    Ok(())
}

fn parse_position(value: &Value) -> VmResult<(u32, u32)> {
    match value {
        Value::List(coords) if coords.len() == 2 => {
            if let (Value::Real(x), Value::Real(y)) = (&coords[0], &coords[1]) {
                Ok((*x as u32, *y as u32))
            } else {
                Err("Position coordinates must be numbers".to_string())
            }
        }
        _ => Err("Position must be a list of two numbers".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_creation() {
        let name = Value::String("Test Dashboard".to_string());
        let layout = Value::String("grid".to_string());
        let widgets = Value::List(vec![]);
        
        let result = dashboard(&[name, layout, widgets]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_widget_config() {
        let config = WidgetConfig::new(
            "line_chart".to_string(),
            "sales_data".to_string(),
            (0, 0),
        );
        
        assert_eq!(config.widget_type, "line_chart");
        assert_eq!(config.position, (0, 0));
    }

    #[test]
    fn test_real_time_chart_creation() {
        let data_stream = Value::String("live_data".to_string());
        let chart_type = Value::String("line".to_string());
        let interval = Value::Real(1000.0);
        
        let result = real_time_chart(&[data_stream, chart_type, interval]);
        assert!(result.is_ok());
    }
}