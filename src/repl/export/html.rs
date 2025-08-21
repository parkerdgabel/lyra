//! HTML Export Engine
//! 
//! Converts Lyra REPL sessions to interactive HTML documents with mathematical
//! rendering, syntax highlighting, data visualizations, and modern web features

use crate::repl::export::engine::{ExportEngine, ExportConfig, ExportResult, ExportError, ExportFeature};
use crate::repl::export::{SessionSnapshot, SessionEntry, CellType};
use crate::vm::Value;
use std::path::PathBuf;
use std::fs;
use std::collections::HashMap;

/// HTML export engine with interactive features
pub struct HtmlExporter {
    template_engine: TemplateEngine,
    math_renderer: WebMathRenderer,
    syntax_highlighter: WebHighlighter,
    chart_generator: ChartGenerator,
    asset_manager: AssetManager,
}

impl HtmlExporter {
    /// Create a new HTML exporter
    pub fn new() -> Self {
        Self {
            template_engine: TemplateEngine::new(),
            math_renderer: WebMathRenderer::new(),
            syntax_highlighter: WebHighlighter::new(),
            chart_generator: ChartGenerator::new(),
            asset_manager: AssetManager::new(),
        }
    }
    
    /// Convert session snapshot to HTML document
    fn convert_to_html(&self, snapshot: &SessionSnapshot, config: &ExportConfig) -> ExportResult<String> {
        let mut html = String::new();
        
        // Build HTML5 document structure
        html.push_str(&self.build_document_head(snapshot, config)?);
        html.push_str("<body>\n");
        
        // Build navigation and header
        html.push_str(&self.build_navigation(snapshot, config)?);
        html.push_str(&self.build_header(snapshot, config)?);
        
        // Main content container
        html.push_str("<main class=\"main-content\">\n");
        
        // Session information section
        if config.include_metadata {
            html.push_str(&self.build_session_info(snapshot, config)?);
        }
        
        // Process each session entry
        html.push_str("<section class=\"session-expressions\" id=\"expressions\">\n");
        html.push_str("<h2>Session Expressions</h2>\n");
        
        for (index, entry) in snapshot.entries.iter().enumerate() {
            html.push_str(&self.convert_entry(entry, index, config)?);
        }
        
        html.push_str("</section>\n");
        
        // Environment variables section
        if config.include_environment && !snapshot.environment.is_empty() {
            html.push_str(&self.build_environment_section(&snapshot.environment, config)?);
        }
        
        // Performance visualization
        if config.include_performance_data {
            html.push_str(&self.build_performance_section(snapshot, config)?);
        }
        
        // Session summary
        html.push_str(&self.build_summary_section(snapshot, config)?);
        
        html.push_str("</main>\n");
        
        // Footer and scripts
        html.push_str(&self.build_footer(config)?);
        html.push_str(&self.build_scripts(config)?);
        
        html.push_str("</body>\n</html>\n");
        
        Ok(html)
    }
    
    /// Build HTML document head with meta tags, CSS, and external libraries
    fn build_document_head(&self, snapshot: &SessionSnapshot, config: &ExportConfig) -> ExportResult<String> {
        let title = config.title.as_deref().unwrap_or("Lyra REPL Session");
        let theme = config.format_options
            .get("theme")
            .and_then(|v| v.as_str())
            .unwrap_or("auto");
        
        let head = format!(r#"<!DOCTYPE html>
<html lang="en" data-theme="{theme}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Lyra REPL Session Export - Interactive Mathematical Computing">
    <meta name="generator" content="Lyra {version}">
    <title>{title}</title>
    
    {css}
    {math_libraries}
    {chart_libraries}
</head>"#,
            theme = theme,
            title = escape_html(title),
            version = snapshot.metadata.lyra_version,
            css = self.build_css_styles(config)?,
            math_libraries = self.build_math_libraries(config)?,
            chart_libraries = self.build_chart_libraries(config)?
        );
        
        Ok(head)
    }
    
    /// Build comprehensive CSS styles
    fn build_css_styles(&self, config: &ExportConfig) -> ExportResult<String> {
        let embed_assets = config.format_options
            .get("embed_assets")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        
        if embed_assets {
            Ok(format!("<style>\n{}\n</style>", self.get_embedded_css(config)?))
        } else {
            Ok(r#"<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/line-numbers/prism-line-numbers.min.css">
    <link rel="stylesheet" href="lyra-export.css">"#.to_string())
        }
    }
    
    /// Get embedded CSS styles
    fn get_embedded_css(&self, config: &ExportConfig) -> ExportResult<String> {
        let css = r#"
/* CSS Variables for theming */
:root {
    --primary-color: #2563eb;
    --secondary-color: #64748b;
    --background-color: #ffffff;
    --surface-color: #f8fafc;
    --text-color: #1e293b;
    --text-muted: #64748b;
    --border-color: #e2e8f0;
    --code-background: #f1f5f9;
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --error-color: #ef4444;
    --shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
    --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
}

[data-theme="dark"] {
    --primary-color: #3b82f6;
    --secondary-color: #94a3b8;
    --background-color: #0f172a;
    --surface-color: #1e293b;
    --text-color: #f1f5f9;
    --text-muted: #94a3b8;
    --border-color: #334155;
    --code-background: #1e293b;
    --success-color: #22c55e;
    --warning-color: #fbbf24;
    --error-color: #f87171;
}

/* Reset and base styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    line-height: 1.6;
    color: var(--text-color);
    background-color: var(--background-color);
    font-size: 16px;
}

/* Navigation */
.navigation {
    position: sticky;
    top: 0;
    background: var(--surface-color);
    border-bottom: 1px solid var(--border-color);
    padding: 1rem 0;
    z-index: 100;
    backdrop-filter: blur(10px);
}

.nav-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.nav-brand {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--primary-color);
    text-decoration: none;
}

.nav-menu {
    display: flex;
    list-style: none;
    gap: 2rem;
}

.nav-item a {
    color: var(--text-muted);
    text-decoration: none;
    font-weight: 500;
    transition: color 0.2s;
}

.nav-item a:hover {
    color: var(--primary-color);
}

.theme-toggle {
    background: none;
    border: 1px solid var(--border-color);
    color: var(--text-color);
    padding: 0.5rem;
    border-radius: 0.375rem;
    cursor: pointer;
    transition: all 0.2s;
}

.theme-toggle:hover {
    background: var(--surface-color);
}

/* Header */
.header {
    background: linear-gradient(135deg, var(--primary-color), #1d4ed8);
    color: white;
    padding: 4rem 0;
    text-align: center;
}

.header-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 2rem;
}

.header h1 {
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 1rem;
}

.header p {
    font-size: 1.25rem;
    opacity: 0.9;
    max-width: 600px;
    margin: 0 auto;
}

/* Main content */
.main-content {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}

/* Session information */
.session-info {
    background: var(--surface-color);
    border-radius: 0.75rem;
    padding: 2rem;
    margin-bottom: 3rem;
    border: 1px solid var(--border-color);
    box-shadow: var(--shadow);
}

.session-info h2 {
    color: var(--primary-color);
    margin-bottom: 1.5rem;
    font-size: 1.5rem;
}

.info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1.5rem;
}

.info-card {
    background: var(--background-color);
    padding: 1.5rem;
    border-radius: 0.5rem;
    border: 1px solid var(--border-color);
}

.info-card h3 {
    color: var(--text-color);
    font-size: 0.875rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}

.info-card .value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--primary-color);
}

/* Expression entries */
.expression-entry {
    background: var(--surface-color);
    border: 1px solid var(--border-color);
    border-radius: 0.75rem;
    margin-bottom: 2rem;
    overflow: hidden;
    box-shadow: var(--shadow);
    transition: all 0.2s;
}

.expression-entry:hover {
    box-shadow: var(--shadow-lg);
}

.expression-header {
    background: var(--background-color);
    padding: 1rem 1.5rem;
    border-bottom: 1px solid var(--border-color);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.expression-title {
    font-weight: 600;
    color: var(--text-color);
    font-size: 1.125rem;
}

.expression-meta {
    display: flex;
    gap: 1rem;
    align-items: center;
    font-size: 0.875rem;
    color: var(--text-muted);
}

.execution-time {
    background: var(--success-color);
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    font-weight: 500;
}

.copy-button {
    background: none;
    border: 1px solid var(--border-color);
    color: var(--text-muted);
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    cursor: pointer;
    font-size: 0.75rem;
    transition: all 0.2s;
}

.copy-button:hover {
    background: var(--surface-color);
    color: var(--text-color);
}

/* Code blocks */
.code-section {
    padding: 1.5rem;
}

.code-block {
    background: var(--code-background);
    border-radius: 0.5rem;
    overflow-x: auto;
    position: relative;
}

.code-block pre {
    margin: 0;
    padding: 1rem;
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    font-size: 0.875rem;
    line-height: 1.5;
}

.code-block code {
    background: none;
    padding: 0;
    border-radius: 0;
}

/* Output sections */
.output-section {
    padding: 1.5rem;
    border-top: 1px solid var(--border-color);
    background: var(--background-color);
}

.output-section h4 {
    color: var(--text-color);
    margin-bottom: 1rem;
    font-size: 1rem;
    font-weight: 600;
}

.math-output {
    font-size: 1.125rem;
    text-align: center;
    padding: 1rem;
    background: var(--surface-color);
    border-radius: 0.5rem;
    border: 1px solid var(--border-color);
}

.text-output {
    font-family: monospace;
    background: var(--code-background);
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid var(--border-color);
    white-space: pre-wrap;
}

/* Error styling */
.error-section {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid var(--error-color);
    border-radius: 0.5rem;
    padding: 1rem;
    margin-top: 1rem;
}

.error-title {
    color: var(--error-color);
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.error-message {
    font-family: monospace;
    color: var(--error-color);
    font-size: 0.875rem;
}

/* Environment variables */
.environment-section {
    background: var(--surface-color);
    border-radius: 0.75rem;
    padding: 2rem;
    margin-bottom: 3rem;
    border: 1px solid var(--border-color);
    box-shadow: var(--shadow);
}

.environment-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 1rem;
}

.environment-table th,
.environment-table td {
    padding: 0.75rem;
    text-align: left;
    border-bottom: 1px solid var(--border-color);
}

.environment-table th {
    background: var(--background-color);
    font-weight: 600;
    color: var(--text-color);
}

.environment-table code {
    background: var(--code-background);
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.875rem;
}

/* Table of contents */
.toc {
    position: fixed;
    left: 2rem;
    top: 50%;
    transform: translateY(-50%);
    background: var(--surface-color);
    border: 1px solid var(--border-color);
    border-radius: 0.5rem;
    padding: 1rem;
    max-height: 70vh;
    overflow-y: auto;
    width: 250px;
    box-shadow: var(--shadow-lg);
    z-index: 50;
}

.toc h3 {
    font-size: 0.875rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: var(--text-color);
}

.toc ul {
    list-style: none;
}

.toc li {
    margin-bottom: 0.5rem;
}

.toc a {
    color: var(--text-muted);
    text-decoration: none;
    font-size: 0.875rem;
    transition: color 0.2s;
}

.toc a:hover,
.toc a.active {
    color: var(--primary-color);
}

/* Responsive design */
@media (max-width: 1024px) {
    .toc {
        display: none;
    }
    
    .main-content {
        padding: 1rem;
    }
    
    .header h1 {
        font-size: 2rem;
    }
    
    .nav-container {
        padding: 0 1rem;
    }
}

@media (max-width: 768px) {
    .nav-menu {
        display: none;
    }
    
    .header {
        padding: 2rem 0;
    }
    
    .header h1 {
        font-size: 1.75rem;
    }
    
    .info-grid {
        grid-template-columns: 1fr;
    }
    
    .expression-header {
        flex-direction: column;
        gap: 1rem;
        align-items: flex-start;
    }
    
    .expression-meta {
        flex-wrap: wrap;
    }
}

/* Print styles */
@media print {
    .navigation,
    .toc,
    .copy-button,
    .theme-toggle {
        display: none !important;
    }
    
    .main-content {
        max-width: none;
        padding: 0;
    }
    
    .expression-entry {
        break-inside: avoid;
        box-shadow: none;
        border: 1px solid #ccc;
    }
    
    .header {
        background: none !important;
        color: black !important;
        padding: 1rem 0;
    }
}

/* Utility classes */
.sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
}

.text-center {
    text-align: center;
}

.text-muted {
    color: var(--text-muted);
}

.mb-4 {
    margin-bottom: 2rem;
}

.mt-4 {
    margin-top: 2rem;
}
"#;
        
        Ok(css.to_string())
    }
    
    /// Build mathematical libraries (MathJax)
    fn build_math_libraries(&self, config: &ExportConfig) -> ExportResult<String> {
        let math_renderer = config.format_options
            .get("math_renderer")
            .and_then(|v| v.as_str())
            .unwrap_or("mathjax");
        
        match math_renderer {
            "mathjax" => Ok(r#"
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']],
                processEscapes: true,
                processEnvironments: true
            },
            options: {
                skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
            }
        };
    </script>"#.to_string()),
            "katex" => Ok(r#"
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/contrib/auto-render.min.js"></script>"#.to_string()),
            _ => Ok(String::new()),
        }
    }
    
    /// Build chart libraries (Chart.js)
    fn build_chart_libraries(&self, config: &ExportConfig) -> ExportResult<String> {
        if config.include_performance_data {
            Ok(r#"
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.min.js"></script>"#.to_string())
        } else {
            Ok(String::new())
        }
    }
    
    /// Build navigation bar
    fn build_navigation(&self, _snapshot: &SessionSnapshot, _config: &ExportConfig) -> ExportResult<String> {
        Ok("<nav class=\"navigation\">
    <div class=\"nav-container\">
        <a href=\"#\" class=\"nav-brand\">📊 Lyra Session</a>
        <ul class=\"nav-menu\">
            <li class=\"nav-item\"><a href=\"#session-info\">Info</a></li>
            <li class=\"nav-item\"><a href=\"#expressions\">Expressions</a></li>
            <li class=\"nav-item\"><a href=\"#environment\">Environment</a></li>
            <li class=\"nav-item\"><a href=\"#summary\">Summary</a></li>
        </ul>
        <button class=\"theme-toggle\" onclick=\"toggleTheme()\" aria-label=\"Toggle theme\">
            🌓
        </button>
    </div>
</nav>".to_string())
    }
    
    /// Build header section
    fn build_header(&self, snapshot: &SessionSnapshot, config: &ExportConfig) -> ExportResult<String> {
        let title = config.title.as_deref().unwrap_or("Lyra REPL Session");
        let description = format!(
            "Interactive mathematical computing session with {} expressions",
            snapshot.entries.len()
        );
        
        Ok(format!(r#"<header class="header">
    <div class="header-container">
        <h1>{}</h1>
        <p>{}</p>
    </div>
</header>"#, escape_html(title), escape_html(&description)))
    }
    
    /// Convert a single session entry to HTML
    fn convert_entry(&self, entry: &SessionEntry, index: usize, config: &ExportConfig) -> ExportResult<String> {
        let mut html = String::new();
        
        let entry_id = format!("entry-{}", index);
        let entry_type = match entry.cell_type {
            CellType::Code => "Code Expression",
            CellType::Meta => "Meta Command",
        };
        
        html.push_str(&format!("<article class=\"expression-entry\" id=\"{}\">\n", entry_id));
        
        // Entry header
        html.push_str("<header class=\"expression-header\">\n");
        html.push_str(&format!("<h3 class=\"expression-title\">{} {}</h3>\n", 
            entry_type, 
            entry.execution_count.map(|c| c.to_string()).unwrap_or_else(|| "".to_string())
        ));
        
        html.push_str("<div class=\"expression-meta\">\n");
        if let Some(execution_time) = entry.execution_time {
            html.push_str(&format!("<span class=\"execution-time\">{:.3}ms</span>\n", 
                execution_time.as_millis()));
        }
        html.push_str(&format!("<button class=\"copy-button\" onclick=\"copyCode('{}')\" aria-label=\"Copy code\">📋 Copy</button>\n", entry_id));
        html.push_str("</div>\n");
        html.push_str("</header>\n");
        
        // Code input section
        html.push_str("<section class=\"code-section\">\n");
        html.push_str("<h4>Input</h4>\n");
        html.push_str(&format!("<div class=\"code-block\" id=\"{}-code\">\n", entry_id));
        html.push_str(&format!("<pre><code class=\"language-lyra\">{}</code></pre>\n", 
            escape_html(&entry.input)));
        html.push_str("</div>\n");
        html.push_str("</section>\n");
        
        // Output section
        html.push_str("<section class=\"output-section\">\n");
        html.push_str("<h4>Output</h4>\n");
        html.push_str(&self.convert_output_to_html(&entry.output)?);
        html.push_str("</section>\n");
        
        // Error section if present
        if config.include_errors {
            if let Some(ref error) = entry.error {
                html.push_str("<section class=\"error-section\">\n");
                html.push_str("<div class=\"error-title\">Error</div>\n");
                html.push_str(&format!("<div class=\"error-message\">{}</div>\n", escape_html(error)));
                html.push_str("</section>\n");
            }
        }
        
        html.push_str("</article>\n");
        
        Ok(html)
    }
    
    /// Convert VM Value to HTML
    fn convert_output_to_html(&self, output: &Value) -> ExportResult<String> {
        match output {
            Value::Integer(n) => Ok(format!("<div class=\"math-output\">${}$</div>", n)),
            Value::Real(f) => {
                let formatted = if f.fract() == 0.0 {
                    format!("{:.0}", f)
                } else {
                    format!("{:.6}", f)
                };
                Ok(format!("<div class=\"math-output\">${}$</div>", formatted))
            }
            Value::String(s) => {
                if self.math_renderer.is_mathematical_expression(s) {
                    Ok(format!("<div class=\"math-output\">{}</div>", 
                        self.math_renderer.convert_to_web_math(s)?))
                } else {
                    Ok(format!("<div class=\"text-output\">{}</div>", escape_html(s)))
                }
            }
            Value::Symbol(sym) => {
                Ok(format!("<div class=\"math-output\">{}</div>", 
                    self.math_renderer.convert_symbol_to_web_math(sym)?))
            }
            Value::List(items) => {
                let mut html = String::new();
                html.push_str("<div class=\"math-output\">$\\{");
                
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        html.push_str(", ");
                    }
                    let item_str = self.value_to_string(item);
                    html.push_str(&escape_html(&item_str));
                }
                
                html.push_str("\\}$</div>");
                Ok(html)
            }
            _ => Ok(format!("<div class=\"text-output\">{}</div>", 
                escape_html(&self.value_to_string(output)))),
        }
    }
    
    /// Build session information section
    fn build_session_info(&self, snapshot: &SessionSnapshot, _config: &ExportConfig) -> ExportResult<String> {
        let code_cells = snapshot.entries.iter().filter(|e| matches!(e.cell_type, CellType::Code)).count();
        let meta_cells = snapshot.entries.iter().filter(|e| matches!(e.cell_type, CellType::Meta)).count();
        let error_cells = snapshot.entries.iter().filter(|e| e.error.is_some()).count();
        let total_time: std::time::Duration = snapshot.entries.iter()
            .filter_map(|e| e.execution_time)
            .sum();
        
        let html = format!(r#"<section class="session-info" id="session-info">
    <h2>Session Information</h2>
    <div class="info-grid">
        <div class="info-card">
            <h3>Session ID</h3>
            <div class="value">{}</div>
        </div>
        <div class="info-card">
            <h3>Lyra Version</h3>
            <div class="value">{}</div>
        </div>
        <div class="info-card">
            <h3>Total Expressions</h3>
            <div class="value">{}</div>
        </div>
        <div class="info-card">
            <h3>Code Expressions</h3>
            <div class="value">{}</div>
        </div>
        <div class="info-card">
            <h3>Meta Commands</h3>
            <div class="value">{}</div>
        </div>
        <div class="info-card">
            <h3>Errors</h3>
            <div class="value">{}</div>
        </div>
        <div class="info-card">
            <h3>Total Time</h3>
            <div class="value">{:.3}ms</div>
        </div>
        <div class="info-card">
            <h3>Success Rate</h3>
            <div class="value">{:.1}%</div>
        </div>
    </div>
</section>"#,
            escape_html(&snapshot.metadata.session_id),
            escape_html(&snapshot.metadata.lyra_version),
            snapshot.entries.len(),
            code_cells,
            meta_cells,
            error_cells,
            total_time.as_millis(),
            if snapshot.entries.is_empty() { 100.0 } else {
                ((snapshot.entries.len() - error_cells) as f64 / snapshot.entries.len() as f64) * 100.0
            }
        );
        
        Ok(html)
    }
    
    /// Build environment variables section
    fn build_environment_section(&self, environment: &HashMap<String, Value>, _config: &ExportConfig) -> ExportResult<String> {
        let mut html = String::new();
        
        html.push_str("<section class=\"environment-section\" id=\"environment\">\n");
        html.push_str("<h2>Environment Variables</h2>\n");
        html.push_str("<table class=\"environment-table\">\n");
        html.push_str("<thead><tr><th>Variable</th><th>Value</th><th>Type</th></tr></thead>\n");
        html.push_str("<tbody>\n");
        
        for (key, value) in environment {
            let value_str = self.value_to_string(value);
            let type_str = self.get_value_type(value);
            html.push_str(&format!("<tr><td><code>{}</code></td><td>{}</td><td>{}</td></tr>\n",
                escape_html(key),
                escape_html(&value_str),
                escape_html(&type_str)
            ));
        }
        
        html.push_str("</tbody>\n");
        html.push_str("</table>\n");
        html.push_str("</section>\n");
        
        Ok(html)
    }
    
    /// Build performance section placeholder
    fn build_performance_section(&self, _snapshot: &SessionSnapshot, _config: &ExportConfig) -> ExportResult<String> {
        Ok(r#"<section class="performance-section" id="performance">
    <h2>Performance Analysis</h2>
    <div class="chart-container">
        <canvas id="performanceChart" width="400" height="200"></canvas>
    </div>
</section>"#.to_string())
    }
    
    /// Build summary section
    fn build_summary_section(&self, _snapshot: &SessionSnapshot, _config: &ExportConfig) -> ExportResult<String> {
        Ok(r#"<section class="summary-section" id="summary">
    <h2>Session Summary</h2>
    <p>This interactive HTML export provides a comprehensive view of your Lyra REPL session with mathematical rendering, syntax highlighting, and performance analysis.</p>
</section>"#.to_string())
    }
    
    /// Build footer
    fn build_footer(&self, _config: &ExportConfig) -> ExportResult<String> {
        Ok(format!(r#"<footer class="footer">
    <div class="footer-container">
        <p>Generated by Lyra v{} • <a href="https://github.com/lyra-lang/lyra">Learn more</a></p>
    </div>
</footer>"#, env!("CARGO_PKG_VERSION")))
    }
    
    /// Build JavaScript functionality
    fn build_scripts(&self, _config: &ExportConfig) -> ExportResult<String> {
        Ok(r#"<script>
// Theme toggle functionality
function toggleTheme() {
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
}

// Copy code functionality
function copyCode(entryId) {
    const codeElement = document.querySelector(`#${entryId}-code code`);
    if (codeElement) {
        navigator.clipboard.writeText(codeElement.textContent).then(() => {
            // Show success feedback
            const button = document.querySelector(`#${entryId} .copy-button`);
            const originalText = button.textContent;
            button.textContent = '✅ Copied!';
            setTimeout(() => {
                button.textContent = originalText;
            }, 2000);
        });
    }
}

// Initialize theme from localStorage
document.addEventListener('DOMContentLoaded', () => {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    
    // Initialize MathJax rendering if available
    if (window.MathJax) {
        MathJax.typesetPromise();
    }
    
    // Initialize KaTeX rendering if available
    if (window.renderMathInElement) {
        renderMathInElement(document.body, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false}
            ]
        });
    }
});

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="\\#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});
</script>"#.to_string())
    }
    
    /// Helper: Convert VM Value to string
    fn value_to_string(&self, value: &Value) -> String {
        match value {
            Value::Integer(n) => n.to_string(),
            Value::Real(f) => f.to_string(),
            Value::String(s) => s.clone(),
            Value::Symbol(sym) => sym.clone(),
            Value::Boolean(b) => b.to_string(),
            Value::List(items) => {
                let items_str: Vec<String> = items.iter().map(|v| self.value_to_string(v)).collect();
                format!("{{{}}}", items_str.join(", "))
            }
            Value::Function { name, .. } => format!("Function[{}]", name),
            Value::LyObj(_) => "Foreign[Object]".to_string(),
        }
    }
    
    /// Helper: Get value type string
    fn get_value_type(&self, value: &Value) -> String {
        match value {
            Value::Integer(_) => "Integer",
            Value::Real(_) => "Real",
            Value::String(_) => "String",
            Value::Symbol(_) => "Symbol",
            Value::Boolean(_) => "Boolean",
            Value::List(_) => "List",
            Value::Function { .. } => "Function",
            Value::LyObj(_) => "Foreign",
        }.to_string()
    }
}

impl ExportEngine for HtmlExporter {
    fn export(
        &self,
        snapshot: &SessionSnapshot,
        config: &ExportConfig,
        output_path: &PathBuf,
    ) -> ExportResult<()> {
        let html_content = self.convert_to_html(snapshot, config)?;
        fs::write(output_path, html_content).map_err(ExportError::from)
    }
    
    fn preview(
        &self,
        snapshot: &SessionSnapshot,
        config: &ExportConfig,
    ) -> ExportResult<String> {
        self.convert_to_html(snapshot, config)
    }
    
    fn validate_config(&self, _config: &ExportConfig) -> ExportResult<()> {
        Ok(())
    }
    
    fn default_extension(&self) -> &'static str {
        "html"
    }
    
    fn mime_type(&self) -> &'static str {
        "text/html"
    }
    
    fn description(&self) -> &'static str {
        "Interactive HTML document with mathematical rendering and visualizations"
    }
    
    fn supports_feature(&self, feature: ExportFeature) -> bool {
        match feature {
            ExportFeature::SyntaxHighlighting => true,
            ExportFeature::MathRendering => true,
            ExportFeature::EmbeddedImages => true,
            ExportFeature::Interactive => true,
            ExportFeature::Metadata => true,
            ExportFeature::PerformanceData => true,
            ExportFeature::ErrorDetails => true,
            ExportFeature::CrossReferences => true,
            ExportFeature::CustomStyling => true,
            ExportFeature::TableOfContents => true,
        }
    }
    
    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "theme": {
                    "type": "string",
                    "enum": ["auto", "light", "dark"],
                    "default": "auto"
                },
                "math_renderer": {
                    "type": "string",
                    "enum": ["mathjax", "katex"],
                    "default": "mathjax"
                },
                "embed_assets": {
                    "type": "boolean",
                    "default": true
                },
                "include_charts": {
                    "type": "boolean",
                    "default": true
                },
                "responsive": {
                    "type": "boolean",
                    "default": true
                }
            }
        })
    }
}

/// Template engine for HTML generation
struct TemplateEngine;

impl TemplateEngine {
    fn new() -> Self {
        Self
    }
}

/// Web mathematical renderer
struct WebMathRenderer {
    function_mappings: HashMap<String, String>,
    symbol_mappings: HashMap<String, String>,
}

impl WebMathRenderer {
    fn new() -> Self {
        let mut function_mappings = HashMap::new();
        function_mappings.insert("Sin".to_string(), "\\sin".to_string());
        function_mappings.insert("Cos".to_string(), "\\cos".to_string());
        function_mappings.insert("Tan".to_string(), "\\tan".to_string());
        function_mappings.insert("Log".to_string(), "\\log".to_string());
        function_mappings.insert("Exp".to_string(), "\\exp".to_string());
        function_mappings.insert("Sqrt".to_string(), "\\sqrt".to_string());
        
        let mut symbol_mappings = HashMap::new();
        symbol_mappings.insert("Pi".to_string(), "\\pi".to_string());
        symbol_mappings.insert("Alpha".to_string(), "\\alpha".to_string());
        symbol_mappings.insert("Beta".to_string(), "\\beta".to_string());
        symbol_mappings.insert("Infinity".to_string(), "\\infty".to_string());
        
        Self {
            function_mappings,
            symbol_mappings,
        }
    }
    
    fn is_mathematical_expression(&self, s: &str) -> bool {
        s.contains('^') || s.contains('*') || s.contains('/') || 
        s.contains('+') || s.contains('-') || s.contains('=') ||
        self.function_mappings.keys().any(|f| s.contains(f)) ||
        self.symbol_mappings.keys().any(|f| s.contains(f))
    }
    
    fn convert_to_web_math(&self, expr: &str) -> ExportResult<String> {
        let mut math = format!("${expr}$");
        
        // Replace function calls
        for (lyra_func, math_func) in &self.function_mappings {
            let pattern = format!("{}[", lyra_func);
            if math.contains(&pattern) {
                math = math.replace(&pattern, &format!("{}(", math_func));
                math = math.replace(']', ")");
            }
        }
        
        // Replace symbols
        for (lyra_sym, math_sym) in &self.symbol_mappings {
            math = math.replace(lyra_sym, math_sym);
        }
        
        Ok(math)
    }
    
    fn convert_symbol_to_web_math(&self, symbol: &str) -> ExportResult<String> {
        if let Some(math_symbol) = self.symbol_mappings.get(symbol) {
            Ok(format!("${math_symbol}$"))
        } else {
            Ok(format!("$\\text{{{}}}$", escape_html(symbol)))
        }
    }
}

/// Web syntax highlighter
struct WebHighlighter;

impl WebHighlighter {
    fn new() -> Self {
        Self
    }
}

/// Chart generator for data visualizations
struct ChartGenerator;

impl ChartGenerator {
    fn new() -> Self {
        Self
    }
}

/// Asset manager for handling CSS, JS, and other resources
struct AssetManager;

impl AssetManager {
    fn new() -> Self {
        Self
    }
}

/// Escape HTML special characters
fn escape_html(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

impl Default for HtmlExporter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repl::export::SessionMetadata;
    use std::time::SystemTime;
    
    fn create_test_snapshot() -> SessionSnapshot {
        SessionSnapshot {
            entries: vec![
                SessionEntry {
                    cell_id: "cell-0".to_string(),
                    cell_type: CellType::Code,
                    input: "x = Sin[Pi/2]".to_string(),
                    output: Value::String("Sin[Pi/2]".to_string()),
                    execution_count: Some(1),
                    timestamp: SystemTime::now(),
                    execution_time: Some(std::time::Duration::from_millis(10)),
                    error: None,
                },
            ],
            environment: std::collections::HashMap::new(),
            metadata: SessionMetadata {
                session_id: "test-session".to_string(),
                start_time: SystemTime::now(),
                end_time: SystemTime::now(),
                lyra_version: "0.1.0".to_string(),
                user: Some("test-user".to_string()),
                working_directory: PathBuf::from("/test"),
                configuration: HashMap::new(),
            },
        }
    }
    
    #[test]
    fn test_html_exporter_creation() {
        let exporter = HtmlExporter::new();
        assert_eq!(exporter.default_extension(), "html");
        assert_eq!(exporter.mime_type(), "text/html");
    }
    
    #[test]
    fn test_html_export_preview() {
        let exporter = HtmlExporter::new();
        let snapshot = create_test_snapshot();
        let config = ExportConfig::default();
        
        let preview = exporter.preview(&snapshot, &config).unwrap();
        assert!(preview.contains("<!DOCTYPE html>"));
        assert!(preview.contains("<html"));
        assert!(preview.contains("</html>"));
        assert!(preview.contains("Sin[Pi/2]"));
        assert!(preview.contains("Lyra Session"));
    }
    
    #[test]
    fn test_feature_support() {
        let exporter = HtmlExporter::new();
        assert!(exporter.supports_feature(ExportFeature::Interactive));
        assert!(exporter.supports_feature(ExportFeature::MathRendering));
        assert!(exporter.supports_feature(ExportFeature::SyntaxHighlighting));
    }
    
    #[test]
    fn test_math_renderer() {
        let renderer = WebMathRenderer::new();
        assert!(renderer.is_mathematical_expression("Sin[x] + 2"));
        assert!(!renderer.is_mathematical_expression("Hello World"));
        
        let math = renderer.convert_to_web_math("Sin[x]").unwrap();
        assert!(math.contains("\\sin"));
    }
    
    #[test]
    fn test_html_escaping() {
        let escaped = escape_html("<script>alert('xss')</script>");
        assert!(escaped.contains("&lt;script&gt;"));
        assert!(!escaped.contains("<script>"));
    }
}