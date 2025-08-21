//! Color themes for syntax highlighting
//!
//! This module provides configurable color themes for the Lyra REPL syntax highlighter,
//! supporting both dark and light modes with ANSI color codes.

use crate::repl::config::ReplConfig;
use std::collections::HashMap;

/// Available highlight colors for different syntax elements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HighlightColor {
    Function,
    Variable,
    Number,
    String,
    Bracket,
    Punctuation,
    Operator,
    Error,
    Comment,
}

/// ANSI color codes for terminal output
#[derive(Debug, Clone)]
pub struct AnsiColor {
    /// Opening color code
    pub open: &'static str,
    /// Closing reset code
    pub close: &'static str,
}

impl AnsiColor {
    const fn new(open: &'static str) -> Self {
        Self {
            open,
            close: "\x1b[0m", // Reset code
        }
    }
    
    /// Apply color to text
    pub fn colorize(&self, text: &str) -> String {
        format!("{}{}{}", self.open, text, self.close)
    }
}

/// Predefined ANSI colors
pub struct AnsiColors;

impl AnsiColors {
    pub const RESET: AnsiColor = AnsiColor::new("\x1b[0m");
    pub const BOLD: AnsiColor = AnsiColor::new("\x1b[1m");
    
    // Basic colors
    pub const BLACK: AnsiColor = AnsiColor::new("\x1b[30m");
    pub const RED: AnsiColor = AnsiColor::new("\x1b[31m");
    pub const GREEN: AnsiColor = AnsiColor::new("\x1b[32m");
    pub const YELLOW: AnsiColor = AnsiColor::new("\x1b[33m");
    pub const BLUE: AnsiColor = AnsiColor::new("\x1b[34m");
    pub const MAGENTA: AnsiColor = AnsiColor::new("\x1b[35m");
    pub const CYAN: AnsiColor = AnsiColor::new("\x1b[36m");
    pub const WHITE: AnsiColor = AnsiColor::new("\x1b[37m");
    
    // Bright colors
    pub const BRIGHT_BLACK: AnsiColor = AnsiColor::new("\x1b[90m");
    pub const BRIGHT_RED: AnsiColor = AnsiColor::new("\x1b[91m");
    pub const BRIGHT_GREEN: AnsiColor = AnsiColor::new("\x1b[92m");
    pub const BRIGHT_YELLOW: AnsiColor = AnsiColor::new("\x1b[93m");
    pub const BRIGHT_BLUE: AnsiColor = AnsiColor::new("\x1b[94m");
    pub const BRIGHT_MAGENTA: AnsiColor = AnsiColor::new("\x1b[95m");
    pub const BRIGHT_CYAN: AnsiColor = AnsiColor::new("\x1b[96m");
    pub const BRIGHT_WHITE: AnsiColor = AnsiColor::new("\x1b[97m");
    
    // Styles
    pub const DIM: AnsiColor = AnsiColor::new("\x1b[2m");
    pub const ITALIC: AnsiColor = AnsiColor::new("\x1b[3m");
    pub const UNDERLINE: AnsiColor = AnsiColor::new("\x1b[4m");
}

/// A complete color theme for syntax highlighting
#[derive(Debug, Clone)]
pub struct ColorTheme {
    name: String,
    colors: HashMap<HighlightColor, AnsiColor>,
    enabled: bool,
}

impl ColorTheme {
    /// Create a new color theme
    pub fn new(name: String, colors: HashMap<HighlightColor, AnsiColor>, enabled: bool) -> Self {
        Self {
            name,
            colors,
            enabled,
        }
    }
    
    /// Create theme from REPL configuration
    pub fn from_config(config: &ReplConfig) -> Self {
        let enabled = config.display.colors;
        
        // For now, we'll use a single default theme, but this can be extended
        // to support multiple themes based on configuration
        if enabled {
            Self::default_dark_theme()
        } else {
            Self::no_color_theme()
        }
    }
    
    /// Default dark theme optimized for dark terminal backgrounds
    pub fn default_dark_theme() -> Self {
        let mut colors = HashMap::new();
        
        colors.insert(HighlightColor::Function, AnsiColors::BRIGHT_BLUE);
        colors.insert(HighlightColor::Variable, AnsiColors::BRIGHT_WHITE);
        colors.insert(HighlightColor::Number, AnsiColors::BRIGHT_MAGENTA);
        colors.insert(HighlightColor::String, AnsiColors::BRIGHT_GREEN);
        colors.insert(HighlightColor::Bracket, AnsiColors::BRIGHT_YELLOW);
        colors.insert(HighlightColor::Punctuation, AnsiColors::WHITE);
        colors.insert(HighlightColor::Operator, AnsiColors::BRIGHT_CYAN);
        colors.insert(HighlightColor::Error, AnsiColors::BRIGHT_RED);
        colors.insert(HighlightColor::Comment, AnsiColors::BRIGHT_BLACK);
        
        Self::new("Dark".to_string(), colors, true)
    }
    
    /// Light theme optimized for light terminal backgrounds
    pub fn default_light_theme() -> Self {
        let mut colors = HashMap::new();
        
        colors.insert(HighlightColor::Function, AnsiColors::BLUE);
        colors.insert(HighlightColor::Variable, AnsiColors::BLACK);
        colors.insert(HighlightColor::Number, AnsiColors::MAGENTA);
        colors.insert(HighlightColor::String, AnsiColors::GREEN);
        colors.insert(HighlightColor::Bracket, AnsiColors::RED);
        colors.insert(HighlightColor::Punctuation, AnsiColors::BLACK);
        colors.insert(HighlightColor::Operator, AnsiColors::CYAN);
        colors.insert(HighlightColor::Error, AnsiColors::RED);
        colors.insert(HighlightColor::Comment, AnsiColors::BRIGHT_BLACK);
        
        Self::new("Light".to_string(), colors, true)
    }
    
    /// No-color theme for when colors are disabled
    pub fn no_color_theme() -> Self {
        Self::new("None".to_string(), HashMap::new(), false)
    }
    
    /// Apply color to text based on highlight type
    pub fn apply_color(&self, text: &str, color_type: HighlightColor) -> String {
        if !self.enabled {
            return text.to_string();
        }
        
        if let Some(color) = self.colors.get(&color_type) {
            color.colorize(text)
        } else {
            text.to_string()
        }
    }
    
    /// Get the theme name
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Check if colors are enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    /// Get available color types in this theme
    pub fn available_colors(&self) -> Vec<HighlightColor> {
        self.colors.keys().copied().collect()
    }
    
    /// Create a custom theme builder
    pub fn builder(name: String) -> ColorThemeBuilder {
        ColorThemeBuilder::new(name)
    }
}

/// Builder for creating custom color themes
#[derive(Debug)]
pub struct ColorThemeBuilder {
    name: String,
    colors: HashMap<HighlightColor, AnsiColor>,
    enabled: bool,
}

impl ColorThemeBuilder {
    /// Create a new theme builder
    pub fn new(name: String) -> Self {
        Self {
            name,
            colors: HashMap::new(),
            enabled: true,
        }
    }
    
    /// Set color for a specific highlight type
    pub fn color(mut self, color_type: HighlightColor, color: AnsiColor) -> Self {
        self.colors.insert(color_type, color);
        self
    }
    
    /// Enable or disable colors
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
    
    /// Build the color theme
    pub fn build(self) -> ColorTheme {
        ColorTheme::new(self.name, self.colors, self.enabled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repl::config::ReplConfig;

    #[test]
    fn test_ansi_color_colorize() {
        let red = AnsiColors::RED;
        let result = red.colorize("test");
        assert_eq!(result, "\x1b[31mtest\x1b[0m");
    }
    
    #[test]
    fn test_default_dark_theme() {
        let theme = ColorTheme::default_dark_theme();
        assert_eq!(theme.name(), "Dark");
        assert!(theme.is_enabled());
        
        // Test function coloring
        let colored = theme.apply_color("Sin", HighlightColor::Function);
        assert!(colored.contains("Sin"));
        assert!(colored.contains("\x1b[94m")); // Bright blue
    }
    
    #[test]
    fn test_no_color_theme() {
        let theme = ColorTheme::no_color_theme();
        assert_eq!(theme.name(), "None");
        assert!(!theme.is_enabled());
        
        // Should return text unchanged
        let result = theme.apply_color("test", HighlightColor::Function);
        assert_eq!(result, "test");
    }
    
    #[test]
    fn test_theme_from_config() {
        // Test with colors enabled
        let mut config = ReplConfig::default();
        config.display.colors = true;
        let theme = ColorTheme::from_config(&config);
        assert!(theme.is_enabled());
        
        // Test with colors disabled
        config.display.colors = false;
        let theme = ColorTheme::from_config(&config);
        assert!(!theme.is_enabled());
    }
    
    #[test]
    fn test_theme_builder() {
        let theme = ColorTheme::builder("Custom".to_string())
            .color(HighlightColor::Function, AnsiColors::RED)
            .color(HighlightColor::Number, AnsiColors::GREEN)
            .enabled(true)
            .build();
        
        assert_eq!(theme.name(), "Custom");
        assert!(theme.is_enabled());
        
        let colored = theme.apply_color("test", HighlightColor::Function);
        assert!(colored.contains("\x1b[31m")); // Red color
    }
    
    #[test]
    fn test_all_highlight_colors() {
        let theme = ColorTheme::default_dark_theme();
        let available = theme.available_colors();
        
        // Should have colors for all major highlight types
        assert!(available.contains(&HighlightColor::Function));
        assert!(available.contains(&HighlightColor::Variable));
        assert!(available.contains(&HighlightColor::Number));
        assert!(available.contains(&HighlightColor::String));
        assert!(available.contains(&HighlightColor::Bracket));
        assert!(available.contains(&HighlightColor::Operator));
        assert!(available.contains(&HighlightColor::Error));
    }
    
    #[test]
    fn test_light_theme() {
        let theme = ColorTheme::default_light_theme();
        assert_eq!(theme.name(), "Light");
        assert!(theme.is_enabled());
        
        // Light theme should use different colors than dark theme
        let dark_theme = ColorTheme::default_dark_theme();
        let light_func = theme.apply_color("Sin", HighlightColor::Function);
        let dark_func = dark_theme.apply_color("Sin", HighlightColor::Function);
        
        // Should produce different colored output
        assert_ne!(light_func, dark_func);
    }
}