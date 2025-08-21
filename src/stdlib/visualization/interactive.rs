//! Interactive Features Module for Lyra Visualization System
//!
//! This module provides interactive visualization capabilities including tooltips,
//! zooming, selection, animations, and cross-chart filtering for modern web applications.

use crate::vm::{Value, VmResult};
use crate::foreign::{LyObj, Foreign};
use crate::stdlib::visualization::charts::Chart;
use std::collections::HashMap;
use std::any::Any;
use std::fmt;
use serde::{Serialize, Deserialize};
use wasm_bindgen::prelude::*;

/// Tooltip configuration and behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TooltipConfig {
    pub content_template: String,
    pub position: TooltipPosition,
    pub style: TooltipStyle,
    pub trigger: TooltipTrigger,
    pub delay: u32, // milliseconds
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TooltipPosition {
    Mouse,
    Fixed { x: i32, y: i32 },
    Relative { offset_x: i32, offset_y: i32 },
    Auto,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TooltipStyle {
    pub background_color: String,
    pub text_color: String,
    pub border_color: String,
    pub border_width: u32,
    pub border_radius: u32,
    pub padding: u32,
    pub font_size: u32,
    pub max_width: Option<u32>,
    pub opacity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TooltipTrigger {
    Hover,
    Click,
    Focus,
    Always,
}

impl Default for TooltipConfig {
    fn default() -> Self {
        Self {
            content_template: "{x}: {y}".to_string(),
            position: TooltipPosition::Mouse,
            style: TooltipStyle {
                background_color: "rgba(0, 0, 0, 0.8)".to_string(),
                text_color: "#ffffff".to_string(),
                border_color: "#cccccc".to_string(),
                border_width: 1,
                border_radius: 4,
                padding: 8,
                font_size: 12,
                max_width: Some(200),
                opacity: 0.9,
            },
            trigger: TooltipTrigger::Hover,
            delay: 100,
            enabled: true,
        }
    }
}

/// Zoom and pan configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoomConfig {
    pub axes: Vec<String>, // Which axes can be zoomed
    pub min_zoom: f64,
    pub max_zoom: f64,
    pub zoom_factor: f64,
    pub pan_enabled: bool,
    pub constraints: ZoomConstraints,
    pub animation_duration: u32, // milliseconds
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoomConstraints {
    pub x_min: Option<f64>,
    pub x_max: Option<f64>,
    pub y_min: Option<f64>,
    pub y_max: Option<f64>,
}

impl Default for ZoomConfig {
    fn default() -> Self {
        Self {
            axes: vec!["x".to_string(), "y".to_string()],
            min_zoom: 0.1,
            max_zoom: 10.0,
            zoom_factor: 1.1,
            pan_enabled: true,
            constraints: ZoomConstraints {
                x_min: None,
                x_max: None,
                y_min: None,
                y_max: None,
            },
            animation_duration: 300,
        }
    }
}

/// Selection configuration and behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionConfig {
    pub mode: SelectionMode,
    pub multi_select: bool,
    pub callback: Option<String>, // JavaScript callback function name
    pub highlight_style: HighlightStyle,
    pub brush_style: BrushStyle,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionMode {
    Point,
    Rectangle,
    Lasso,
    Brush,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighlightStyle {
    pub color: String,
    pub opacity: f32,
    pub stroke_width: u32,
    pub stroke_color: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrushStyle {
    pub fill_color: String,
    pub fill_opacity: f32,
    pub stroke_color: String,
    pub stroke_width: u32,
    pub stroke_dash: Vec<u32>,
}

impl Default for SelectionConfig {
    fn default() -> Self {
        Self {
            mode: SelectionMode::Point,
            multi_select: false,
            callback: None,
            highlight_style: HighlightStyle {
                color: "#ff6b6b".to_string(),
                opacity: 0.8,
                stroke_width: 2,
                stroke_color: "#d63031".to_string(),
            },
            brush_style: BrushStyle {
                fill_color: "#74b9ff".to_string(),
                fill_opacity: 0.3,
                stroke_color: "#0984e3".to_string(),
                stroke_width: 1,
                stroke_dash: vec![5, 5],
            },
        }
    }
}

/// Animation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationConfig {
    pub transition_type: TransitionType,
    pub duration: u32, // milliseconds
    pub easing: EasingFunction,
    pub delay: u32,
    pub loop_animation: bool,
    pub auto_play: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransitionType {
    FadeIn,
    FadeOut,
    SlideIn { direction: Direction },
    SlideOut { direction: Direction },
    Scale { from: f64, to: f64 },
    Rotate { from: f64, to: f64 },
    Morph,
    Custom { keyframes: Vec<AnimationKeyframe> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Direction {
    Left,
    Right,
    Up,
    Down,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EasingFunction {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    Bounce,
    Elastic,
    Back,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationKeyframe {
    pub time: f64, // 0.0 to 1.0
    pub properties: HashMap<String, f64>,
}

impl Default for AnimationConfig {
    fn default() -> Self {
        Self {
            transition_type: TransitionType::FadeIn,
            duration: 1000,
            easing: EasingFunction::EaseInOut,
            delay: 0,
            loop_animation: false,
            auto_play: true,
        }
    }
}

/// Cross-chart filtering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossFilterConfig {
    pub source_charts: Vec<String>,
    pub target_charts: Vec<String>,
    pub filter_field: String,
    pub filter_operation: FilterOperation,
    pub sync_zoom: bool,
    pub sync_selection: bool,
    pub debounce_delay: u32, // milliseconds
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterOperation {
    Equals,
    Contains,
    Range,
    GreaterThan,
    LessThan,
    In,
    NotIn,
}

impl Default for CrossFilterConfig {
    fn default() -> Self {
        Self {
            source_charts: Vec::new(),
            target_charts: Vec::new(),
            filter_field: "value".to_string(),
            filter_operation: FilterOperation::Equals,
            sync_zoom: false,
            sync_selection: true,
            debounce_delay: 300,
        }
    }
}

// =============================================================================
// Foreign Object Implementations
// =============================================================================

/// Tooltip Foreign Object
#[derive(Debug)]
pub struct TooltipObj {
    pub config: TooltipConfig,
    pub chart_id: String,
    pub is_active: bool,
}

impl TooltipObj {
    pub fn new(config: TooltipConfig, chart_id: String) -> Self {
        Self {
            config,
            chart_id,
            is_active: false,
        }
    }
    
    /// Generate JavaScript code for tooltip functionality
    pub fn generate_js(&self) -> String {
        format!(
            r#"
            class Tooltip {{
                constructor(chartId, config) {{
                    this.chartId = chartId;
                    this.config = config;
                    this.element = null;
                    this.init();
                }}
                
                init() {{
                    this.element = document.createElement('div');
                    this.element.className = 'chart-tooltip';
                    this.element.style.position = 'absolute';
                    this.element.style.pointerEvents = 'none';
                    this.element.style.background = this.config.style.background_color;
                    this.element.style.color = this.config.style.text_color;
                    this.element.style.padding = this.config.style.padding + 'px';
                    this.element.style.borderRadius = this.config.style.border_radius + 'px';
                    this.element.style.fontSize = this.config.style.font_size + 'px';
                    this.element.style.opacity = '0';
                    this.element.style.transition = 'opacity 0.3s ease';
                    document.body.appendChild(this.element);
                    
                    this.attachEvents();
                }}
                
                attachEvents() {{
                    const chart = document.getElementById(this.chartId);
                    if (!chart) return;
                    
                    if (this.config.trigger === 'Hover') {{
                        chart.addEventListener('mousemove', (e) => this.show(e));
                        chart.addEventListener('mouseleave', () => this.hide());
                    }} else if (this.config.trigger === 'Click') {{
                        chart.addEventListener('click', (e) => this.show(e));
                    }}
                }}
                
                show(event) {{
                    if (!this.config.enabled) return;
                    
                    const data = this.extractDataFromEvent(event);
                    const content = this.formatContent(data);
                    
                    this.element.innerHTML = content;
                    this.positionTooltip(event);
                    
                    setTimeout(() => {{
                        this.element.style.opacity = this.config.style.opacity;
                    }}, this.config.delay);
                }}
                
                hide() {{
                    this.element.style.opacity = '0';
                }}
                
                extractDataFromEvent(event) {{
                    // Extract data point information from mouse event
                    return {{
                        x: event.offsetX,
                        y: event.offsetY,
                        value: 'Sample Value'
                    }};
                }}
                
                formatContent(data) {{
                    return this.config.content_template
                        .replace('{{x}}', data.x)
                        .replace('{{y}}', data.y)
                        .replace('{{value}}', data.value);
                }}
                
                positionTooltip(event) {{
                    const rect = event.target.getBoundingClientRect();
                    
                    switch (this.config.position) {{
                        case 'Mouse':
                            this.element.style.left = (event.clientX + 10) + 'px';
                            this.element.style.top = (event.clientY - 10) + 'px';
                            break;
                        default:
                            this.element.style.left = (rect.left + event.offsetX + 10) + 'px';
                            this.element.style.top = (rect.top + event.offsetY - 10) + 'px';
                    }}
                }}
            }}
            
            new Tooltip('{}', {});
            "#,
            self.chart_id,
            serde_json::to_string(&self.config).unwrap_or_default()
        )
    }
}

impl Foreign for TooltipObj {
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn type_name(&self) -> &'static str {
        "Tooltip"
    }
}

impl fmt::Display for TooltipObj {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tooltip[chart: {}, active: {}]", self.chart_id, self.is_active)
    }
}

/// Zoom Foreign Object
#[derive(Debug)]
pub struct ZoomObj {
    pub config: ZoomConfig,
    pub chart_id: String,
    pub current_zoom: f64,
    pub current_pan: (f64, f64),
}

impl ZoomObj {
    pub fn new(config: ZoomConfig, chart_id: String) -> Self {
        Self {
            config,
            chart_id,
            current_zoom: 1.0,
            current_pan: (0.0, 0.0),
        }
    }
    
    pub fn generate_js(&self) -> String {
        format!(
            r#"
            class ZoomController {{
                constructor(chartId, config) {{
                    this.chartId = chartId;
                    this.config = config;
                    this.zoom = 1.0;
                    this.pan = {{x: 0, y: 0}};
                    this.init();
                }}
                
                init() {{
                    const chart = document.getElementById(this.chartId);
                    if (!chart) return;
                    
                    chart.addEventListener('wheel', (e) => this.handleWheel(e));
                    chart.addEventListener('mousedown', (e) => this.handleMouseDown(e));
                }}
                
                handleWheel(event) {{
                    event.preventDefault();
                    
                    const delta = event.deltaY > 0 ? 1 / this.config.zoom_factor : this.config.zoom_factor;
                    const newZoom = Math.max(this.config.min_zoom, 
                                           Math.min(this.config.max_zoom, this.zoom * delta));
                    
                    if (newZoom !== this.zoom) {{
                        this.zoom = newZoom;
                        this.applyTransform();
                    }}
                }}
                
                handleMouseDown(event) {{
                    if (!this.config.pan_enabled) return;
                    
                    const startX = event.clientX - this.pan.x;
                    const startY = event.clientY - this.pan.y;
                    
                    const handleMouseMove = (e) => {{
                        this.pan.x = e.clientX - startX;
                        this.pan.y = e.clientY - startY;
                        this.applyTransform();
                    }};
                    
                    const handleMouseUp = () => {{
                        document.removeEventListener('mousemove', handleMouseMove);
                        document.removeEventListener('mouseup', handleMouseUp);
                    }};
                    
                    document.addEventListener('mousemove', handleMouseMove);
                    document.addEventListener('mouseup', handleMouseUp);
                }}
                
                applyTransform() {{
                    const chart = document.getElementById(this.chartId);
                    if (!chart) return;
                    
                    const svg = chart.querySelector('svg');
                    if (svg) {{
                        svg.style.transform = `translate(${{this.pan.x}}px, ${{this.pan.y}}px) scale(${{this.zoom}})`;
                        svg.style.transformOrigin = 'center center';
                        svg.style.transition = this.config.animation_duration + 'ms ease';
                    }}
                }}
                
                reset() {{
                    this.zoom = 1.0;
                    this.pan = {{x: 0, y: 0}};
                    this.applyTransform();
                }}
            }}
            
            new ZoomController('{}', {});
            "#,
            self.chart_id,
            serde_json::to_string(&self.config).unwrap_or_default()
        )
    }
}

impl Foreign for ZoomObj {
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn type_name(&self) -> &'static str {
        "Zoom"
    }
}

impl fmt::Display for ZoomObj {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Zoom[chart: {}, zoom: {:.2}x]", self.chart_id, self.current_zoom)
    }
}

/// Selection Foreign Object
#[derive(Debug)]
pub struct SelectionObj {
    pub config: SelectionConfig,
    pub chart_id: String,
    pub selected_items: Vec<String>,
}

impl SelectionObj {
    pub fn new(config: SelectionConfig, chart_id: String) -> Self {
        Self {
            config,
            chart_id,
            selected_items: Vec::new(),
        }
    }
}

impl Foreign for SelectionObj {
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn type_name(&self) -> &'static str {
        "Selection"
    }
}

impl fmt::Display for SelectionObj {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Selection[chart: {}, items: {}]", self.chart_id, self.selected_items.len())
    }
}

/// Animation Foreign Object
#[derive(Debug)]
pub struct AnimationObj {
    pub config: AnimationConfig,
    pub chart_id: String,
    pub is_playing: bool,
}

impl AnimationObj {
    pub fn new(config: AnimationConfig, chart_id: String) -> Self {
        Self {
            config,
            chart_id,
            is_playing: false,
        }
    }
}

impl Foreign for AnimationObj {
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn type_name(&self) -> &'static str {
        "Animation"
    }
}

impl fmt::Display for AnimationObj {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Animation[chart: {}, playing: {}]", self.chart_id, self.is_playing)
    }
}

/// Cross Filter Foreign Object
#[derive(Debug)]
pub struct CrossFilterObj {
    pub config: CrossFilterConfig,
    pub active_filters: HashMap<String, Value>,
}

impl CrossFilterObj {
    pub fn new(config: CrossFilterConfig) -> Self {
        Self {
            config,
            active_filters: HashMap::new(),
        }
    }
}

impl Foreign for CrossFilterObj {
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn type_name(&self) -> &'static str {
        "CrossFilter"
    }
}

impl fmt::Display for CrossFilterObj {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CrossFilter[{} source charts, {} active filters]", 
               self.config.source_charts.len(), self.active_filters.len())
    }
}

// =============================================================================
// Interactive Function Implementations
// =============================================================================

/// Create tooltip for chart
pub fn tooltip(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err("Tooltip requires chart, content_template, and position".to_string());
    }
    
    let chart_id = extract_chart_id(&args[0])?;
    
    let content_template = match &args[1] {
        Value::String(template) => template.clone(),
        _ => return Err("Content template must be a string".to_string()),
    };
    
    let position = parse_tooltip_position(&args[2])?;
    
    let mut config = TooltipConfig::default();
    config.content_template = content_template;
    config.position = position;
    
    let tooltip = TooltipObj::new(config, chart_id);
    Ok(Value::LyObj(LyObj::new(Box::new(tooltip))))
}

/// Add zoom and pan functionality to chart
pub fn zoom(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err("Zoom requires chart, axes, and constraints".to_string());
    }
    
    let chart_id = extract_chart_id(&args[0])?;
    
    let axes = match &args[1] {
        Value::List(axis_list) => {
            axis_list.iter().map(|v| match v {
                Value::String(axis) => Ok(axis.clone()),
                _ => Err("Axes must be strings".to_string()),
            }).collect::<Result<Vec<_>, _>>()?
        }
        _ => return Err("Axes must be a list".to_string()),
    };
    
    let constraints = parse_zoom_constraints(&args[2])?;
    
    let mut config = ZoomConfig::default();
    config.axes = axes;
    config.constraints = constraints;
    
    let zoom = ZoomObj::new(config, chart_id);
    Ok(Value::LyObj(LyObj::new(Box::new(zoom))))
}

/// Add selection functionality to chart
pub fn selection(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err("Selection requires chart, selection_mode, and callback".to_string());
    }
    
    let chart_id = extract_chart_id(&args[0])?;
    
    let mode = parse_selection_mode(&args[1])?;
    
    let callback = match &args[2] {
        Value::String(cb) => Some(cb.clone()),
        Value::Missing => None,
        _ => return Err("Callback must be a string or Missing".to_string()),
    };
    
    let mut config = SelectionConfig::default();
    config.mode = mode;
    config.callback = callback;
    
    let selection = SelectionObj::new(config, chart_id);
    Ok(Value::LyObj(LyObj::new(Box::new(selection))))
}

/// Add animation to chart
pub fn animation(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err("Animation requires chart, transition_type, and duration".to_string());
    }
    
    let chart_id = extract_chart_id(&args[0])?;
    
    let transition_type = parse_transition_type(&args[1])?;
    
    let duration = match &args[2] {
        Value::Real(d) => *d as u32,
        _ => return Err("Duration must be a number".to_string()),
    };
    
    let mut config = AnimationConfig::default();
    config.transition_type = transition_type;
    config.duration = duration;
    
    let animation = AnimationObj::new(config, chart_id);
    Ok(Value::LyObj(LyObj::new(Box::new(animation))))
}

/// Create cross-chart filtering
pub fn cross_filter(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err("CrossFilter requires charts, data, and filter_interactions".to_string());
    }
    
    let source_charts = parse_chart_list(&args[0])?;
    let filter_field = parse_filter_field(&args[1])?;
    let filter_operation = parse_filter_operation(&args[2])?;
    
    let mut config = CrossFilterConfig::default();
    config.source_charts = source_charts.clone();
    config.target_charts = source_charts; // Use same charts as targets by default
    config.filter_field = filter_field;
    config.filter_operation = filter_operation;
    
    let cross_filter = CrossFilterObj::new(config);
    Ok(Value::LyObj(LyObj::new(Box::new(cross_filter))))
}

// =============================================================================
// Helper Functions
// =============================================================================

fn extract_chart_id(value: &Value) -> VmResult<String> {
    // In a real implementation, this would extract the chart ID from the chart object
    // For now, return a placeholder ID
    Ok("chart_1".to_string())
}

fn parse_tooltip_position(value: &Value) -> VmResult<TooltipPosition> {
    match value {
        Value::String(pos) => match pos.as_str() {
            "mouse" => Ok(TooltipPosition::Mouse),
            "auto" => Ok(TooltipPosition::Auto),
            _ => Ok(TooltipPosition::Mouse),
        },
        Value::List(coords) if coords.len() == 2 => {
            if let (Value::Real(x), Value::Real(y)) = (&coords[0], &coords[1]) {
                Ok(TooltipPosition::Fixed { x: *x as i32, y: *y as i32 })
            } else {
                Err("Position coordinates must be numbers".to_string())
            }
        }
        _ => Ok(TooltipPosition::Mouse),
    }
}

fn parse_zoom_constraints(value: &Value) -> VmResult<ZoomConstraints> {
    // Simplified constraint parsing
    Ok(ZoomConstraints {
        x_min: None,
        x_max: None,
        y_min: None,
        y_max: None,
    })
}

fn parse_selection_mode(value: &Value) -> VmResult<SelectionMode> {
    match value {
        Value::String(mode) => match mode.as_str() {
            "point" => Ok(SelectionMode::Point),
            "rectangle" => Ok(SelectionMode::Rectangle),
            "lasso" => Ok(SelectionMode::Lasso),
            "brush" => Ok(SelectionMode::Brush),
            "none" => Ok(SelectionMode::None),
            _ => Ok(SelectionMode::Point),
        },
        _ => Ok(SelectionMode::Point),
    }
}

fn parse_transition_type(value: &Value) -> VmResult<TransitionType> {
    match value {
        Value::String(transition) => match transition.as_str() {
            "fade_in" => Ok(TransitionType::FadeIn),
            "fade_out" => Ok(TransitionType::FadeOut),
            "slide_left" => Ok(TransitionType::SlideIn { direction: Direction::Left }),
            "slide_right" => Ok(TransitionType::SlideIn { direction: Direction::Right }),
            "scale" => Ok(TransitionType::Scale { from: 0.0, to: 1.0 }),
            _ => Ok(TransitionType::FadeIn),
        },
        _ => Ok(TransitionType::FadeIn),
    }
}

fn parse_chart_list(value: &Value) -> VmResult<Vec<String>> {
    match value {
        Value::List(charts) => {
            charts.iter().map(|v| match v {
                Value::String(id) => Ok(id.clone()),
                _ => Ok("chart_placeholder".to_string()),
            }).collect()
        }
        _ => Ok(vec!["chart_placeholder".to_string()]),
    }
}

fn parse_filter_field(value: &Value) -> VmResult<String> {
    match value {
        Value::String(field) => Ok(field.clone()),
        _ => Ok("value".to_string()),
    }
}

fn parse_filter_operation(value: &Value) -> VmResult<FilterOperation> {
    match value {
        Value::String(op) => match op.as_str() {
            "equals" => Ok(FilterOperation::Equals),
            "contains" => Ok(FilterOperation::Contains),
            "range" => Ok(FilterOperation::Range),
            "greater_than" => Ok(FilterOperation::GreaterThan),
            "less_than" => Ok(FilterOperation::LessThan),
            "in" => Ok(FilterOperation::In),
            "not_in" => Ok(FilterOperation::NotIn),
            _ => Ok(FilterOperation::Equals),
        },
        _ => Ok(FilterOperation::Equals),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tooltip_config_creation() {
        let config = TooltipConfig::default();
        assert_eq!(config.content_template, "{x}: {y}");
        assert!(config.enabled);
        assert_eq!(config.delay, 100);
    }

    #[test]
    fn test_zoom_config_creation() {
        let config = ZoomConfig::default();
        assert_eq!(config.min_zoom, 0.1);
        assert_eq!(config.max_zoom, 10.0);
        assert!(config.pan_enabled);
    }

    #[test]
    fn test_selection_mode_parsing() {
        let point_mode = parse_selection_mode(&Value::String("point".to_string())).unwrap();
        assert!(matches!(point_mode, SelectionMode::Point));
        
        let rect_mode = parse_selection_mode(&Value::String("rectangle".to_string())).unwrap();
        assert!(matches!(rect_mode, SelectionMode::Rectangle));
    }

    #[test]
    fn test_tooltip_position_parsing() {
        let mouse_pos = parse_tooltip_position(&Value::String("mouse".to_string())).unwrap();
        assert!(matches!(mouse_pos, TooltipPosition::Mouse));
        
        let fixed_pos = parse_tooltip_position(&Value::List(vec![
            Value::Real(100.0),
            Value::Real(200.0),
        ])).unwrap();
        assert!(matches!(fixed_pos, TooltipPosition::Fixed { x: 100, y: 200 }));
    }

    #[test]
    fn test_cross_filter_creation() {
        let charts = Value::List(vec![
            Value::String("chart1".to_string()),
            Value::String("chart2".to_string()),
        ]);
        let field = Value::String("category".to_string());
        let operation = Value::String("equals".to_string());
        
        let result = cross_filter(&[charts, field, operation]);
        assert!(result.is_ok());
    }
}