//! LaTeX Export Engine
//! 
//! Converts Lyra REPL sessions to properly formatted LaTeX documents
//! with mathematical notation, syntax highlighting, and document structure

use crate::repl::export::engine::{ExportEngine, ExportConfig, ExportResult, ExportError, ExportFeature};
use crate::repl::export::{SessionSnapshot, SessionEntry, CellType};
use crate::vm::Value;
use std::path::PathBuf;
use std::fs;
use std::collections::HashMap;

/// LaTeX export engine
pub struct LaTeXExporter {
    math_converter: MathConverter,
    syntax_highlighter: LaTeXHighlighter,
    document_builder: DocumentBuilder,
}

impl LaTeXExporter {
    /// Create a new LaTeX exporter
    pub fn new() -> Self {
        Self {
            math_converter: MathConverter::new(),
            syntax_highlighter: LaTeXHighlighter::new(),
            document_builder: DocumentBuilder::new(),
        }
    }
    
    /// Convert session snapshot to LaTeX document
    fn convert_to_latex(&self, snapshot: &SessionSnapshot, config: &ExportConfig) -> ExportResult<String> {
        let mut latex = String::new();
        
        // Build document header
        latex.push_str(&self.document_builder.build_header(config)?);
        
        // Add title and metadata
        if config.include_metadata {
            latex.push_str(&self.build_title_section(snapshot, config)?);
        }
        
        // Process each session entry
        for entry in &snapshot.entries {
            latex.push_str(&self.convert_entry(entry, config)?);
        }
        
        // Add environment section if requested
        if config.include_environment && !snapshot.environment.is_empty() {
            latex.push_str(&self.build_environment_section(&snapshot.environment)?);
        }
        
        // Build document footer
        latex.push_str(&self.document_builder.build_footer(config)?);
        
        Ok(latex)
    }
    
    /// Convert a single session entry to LaTeX
    fn convert_entry(&self, entry: &SessionEntry, config: &ExportConfig) -> ExportResult<String> {
        let mut latex = String::new();
        
        match entry.cell_type {
            CellType::Code => {
                latex.push_str(&self.convert_code_cell(entry, config)?);
            }
            CellType::Meta => {
                latex.push_str(&self.convert_meta_cell(entry, config)?);
            }
            CellType::Markdown => {
                latex.push_str(&self.convert_markdown_cell(entry, config)?);
            }
            CellType::Raw => {
                latex.push_str(&self.convert_raw_cell(entry, config)?);
            }
        }
        
        latex.push('\n');
        Ok(latex)
    }
    
    /// Convert a code cell to LaTeX
    fn convert_code_cell(&self, entry: &SessionEntry, config: &ExportConfig) -> ExportResult<String> {
        let mut latex = String::new();
        
        // Add subsection for code cell
        if let Some(count) = entry.execution_count {
            latex.push_str(&format!("\\subsection{{Expression {}}}\n", count));
        }
        
        // Input code block
        latex.push_str("\\subsubsection{Input}\n");
        if config.syntax_highlighting {
            latex.push_str(&self.syntax_highlighter.highlight_code(&entry.input)?);
        } else {
            latex.push_str(&format!("\\begin{{verbatim}}\n{}\n\\end{{verbatim}}\n", entry.input));
        }
        
        // Output section
        latex.push_str("\n\\subsubsection{Output}\n");
        let output_latex = self.convert_output(&entry.output, config)?;
        latex.push_str(&output_latex);
        
        // Add timing information if requested
        if config.include_timing {
            if let Some(execution_time) = entry.execution_time {
                latex.push_str(&format!(
                    "\n\\textit{{Execution time: {:.3}ms}}\n",
                    execution_time.as_millis()
                ));
            }
        }
        
        // Add error information if present
        if config.include_errors {
            if let Some(ref error) = entry.error {
                latex.push_str("\n\\subsubsection{Error}\n");
                latex.push_str(&format!("\\textcolor{{red}}{{\\texttt{{{}}}}}\n", escape_latex(error)));
            }
        }
        
        Ok(latex)
    }
    
    /// Convert a meta command cell to LaTeX
    fn convert_meta_cell(&self, entry: &SessionEntry, _config: &ExportConfig) -> ExportResult<String> {
        let mut latex = String::new();
        
        latex.push_str("\\subsubsection{Meta Command}\n");
        latex.push_str(&format!("\\texttt{{{}}}\n", escape_latex(&entry.input)));
        
        // Output of meta command
        let output_str = self.value_to_string(&entry.output);
        if !output_str.trim().is_empty() {
            latex.push_str("\n\\paragraph{Output:}\n");
            latex.push_str(&format!("\\begin{{quote}}\n{}\\end{{quote}}\n", escape_latex(&output_str)));
        }
        
        Ok(latex)
    }
    
    /// Convert output value to LaTeX
    fn convert_output(&self, output: &Value, config: &ExportConfig) -> ExportResult<String> {
        match output {
            // Mathematical expressions get LaTeX math formatting
            Value::Integer(n) => Ok(format!("${n}$")),
            Value::Real(f) => {
                if f.fract() == 0.0 {
                    Ok(format!("${:.0}$", f))
                } else {
                    Ok(format!("${:.6}$", f))
                }
            }
            Value::String(s) => {
                // Check if string looks like a mathematical expression
                if self.math_converter.is_mathematical_expression(s) {
                    Ok(self.math_converter.convert_to_latex(s)?)
                } else {
                    Ok(escape_latex(s))
                }
            }
            Value::Symbol(sym) => {
                // Mathematical symbols get special treatment
                Ok(self.math_converter.convert_symbol_to_latex(sym)?)
            }
            Value::List(items) => {
                if config.math_rendering == crate::repl::export::engine::MathRenderingStyle::LaTeX {
                    self.convert_list_to_latex(items, config)
                } else {
                    Ok(format!("\\texttt{{{}}}", escape_latex(&self.value_to_string(output))))
                }
            }
            _ => Ok(format!("\\texttt{{{}}}", escape_latex(&self.value_to_string(output)))),
        }
    }
    
    /// Convert a list to LaTeX mathematical notation
    fn convert_list_to_latex(&self, items: &[Value], config: &ExportConfig) -> ExportResult<String> {
        let mut latex = String::new();
        latex.push_str("$\\{");
        
        for (i, item) in items.iter().enumerate() {
            if i > 0 {
                latex.push_str(", ");
            }
            let item_latex = self.convert_output(item, config)?;
            // Remove outer $ signs if present
            let item_latex = item_latex.trim_start_matches('$').trim_end_matches('$');
            latex.push_str(item_latex);
        }
        
        latex.push_str("\\}$");
        Ok(latex)
    }
    
    /// Build title section
    fn build_title_section(&self, snapshot: &SessionSnapshot, config: &ExportConfig) -> ExportResult<String> {
        let mut latex = String::new();
        
        let title = config.title.as_deref().unwrap_or("Lyra REPL Session");
        latex.push_str(&format!("\\title{{{}}}\n", escape_latex(title)));
        
        if let Some(ref author) = config.author {
            latex.push_str(&format!("\\author{{{}}}\n", escape_latex(author)));
        } else if let Some(ref user) = snapshot.metadata.user {
            latex.push_str(&format!("\\author{{{}}}\n", escape_latex(user)));
        }
        
        if config.include_export_timestamp {
            latex.push_str(&format!("\\date{{{}}}\n", 
                chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
        }
        
        latex.push_str("\\maketitle\n\n");
        
        // Add session metadata table
        latex.push_str("\\section{Session Information}\n");
        latex.push_str("\\begin{tabular}{ll}\n");
        latex.push_str(&format!("Session ID & \\texttt{{{}}} \\\\\n", 
            escape_latex(&snapshot.metadata.session_id)));
        latex.push_str(&format!("Lyra Version & {} \\\\\n", 
            escape_latex(&snapshot.metadata.lyra_version)));
        latex.push_str(&format!("Total Expressions & {} \\\\\n", snapshot.entries.len()));
        
        let code_cells = snapshot.entries.iter().filter(|e| matches!(e.cell_type, CellType::Code)).count();
        latex.push_str(&format!("Code Expressions & {} \\\\\n", code_cells));
        
        if config.include_timing {
            let total_time: std::time::Duration = snapshot.entries.iter()
                .filter_map(|e| e.execution_time)
                .sum();
            latex.push_str(&format!("Total Execution Time & {:.3}ms \\\\\n", total_time.as_millis()));
        }
        
        latex.push_str("\\end{tabular}\n\n");
        
        Ok(latex)
    }
    
    /// Build environment variables section
    fn build_environment_section(&self, environment: &HashMap<String, Value>) -> ExportResult<String> {
        let mut latex = String::new();
        
        latex.push_str("\\section{Environment Variables}\n");
        latex.push_str("\\begin{tabular}{ll}\n");
        latex.push_str("\\textbf{Variable} & \\textbf{Value} \\\\\n");
        latex.push_str("\\hline\n");
        
        for (key, value) in environment {
            let value_str = self.value_to_string(value);
            latex.push_str(&format!("\\texttt{{{}}} & {} \\\\\n", 
                escape_latex(key), 
                escape_latex(&value_str)));
        }
        
        latex.push_str("\\end{tabular}\n\n");
        Ok(latex)
    }
    
    /// Convert VM Value to string representation
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
            Value::Function(name) => format!("Function[{}]", name),
            Value::Missing => "Missing[]".to_string(),
            Value::Object(_) => "Object[...]".to_string(),
            Value::LyObj(_) => "Foreign[Object]".to_string(),
            Value::Quote(expr) => format!("Hold[{:?}]", expr),
            Value::Pattern(pattern) => format!("{}", pattern),
            Value::Rule { lhs, rhs } => format!("{} -> {}", self.value_to_string(lhs), self.value_to_string(rhs)),
            Value::PureFunction { body } => format!("{} \\&", self.value_to_string(body)),
            Value::Slot { number } => match number {
                Some(n) => format!("\\#{}", n),
                None => "\\#".to_string(),
            },
        }
    }

    /// Convert markdown cell to LaTeX
    fn convert_markdown_cell(&self, entry: &SessionEntry, _config: &ExportConfig) -> ExportResult<String> {
        let mut latex = String::new();
        latex.push_str("\\subsection{Markdown Cell}\n");
        latex.push_str("\\begin{quote}\n");
        latex.push_str(&escape_latex(&entry.input));
        latex.push_str("\n\\end{quote}\n");
        Ok(latex)
    }

    /// Convert raw cell to LaTeX
    fn convert_raw_cell(&self, entry: &SessionEntry, _config: &ExportConfig) -> ExportResult<String> {
        let mut latex = String::new();
        latex.push_str("\\subsection{Raw Text}\n");
        latex.push_str("\\begin{verbatim}\n");
        latex.push_str(&entry.input);
        latex.push_str("\n\\end{verbatim}\n");
        Ok(latex)
    }
}

impl ExportEngine for LaTeXExporter {
    fn export(
        &self,
        snapshot: &SessionSnapshot,
        config: &ExportConfig,
        output_path: &PathBuf,
    ) -> ExportResult<()> {
        let latex_content = self.convert_to_latex(snapshot, config)?;
        fs::write(output_path, latex_content).map_err(ExportError::from)
    }
    
    fn preview(
        &self,
        snapshot: &SessionSnapshot,
        config: &ExportConfig,
    ) -> ExportResult<String> {
        self.convert_to_latex(snapshot, config)
    }
    
    fn validate_config(&self, _config: &ExportConfig) -> ExportResult<()> {
        // LaTeX exporter supports all standard configurations
        Ok(())
    }
    
    fn default_extension(&self) -> &'static str {
        "tex"
    }
    
    fn mime_type(&self) -> &'static str {
        "application/x-latex"
    }
    
    fn description(&self) -> &'static str {
        "LaTeX document with mathematical formatting and syntax highlighting"
    }
    
    fn supports_feature(&self, feature: ExportFeature) -> bool {
        match feature {
            ExportFeature::SyntaxHighlighting => true,
            ExportFeature::MathRendering => true,
            ExportFeature::Metadata => true,
            ExportFeature::PerformanceData => true,
            ExportFeature::ErrorDetails => true,
            ExportFeature::CrossReferences => true,
            ExportFeature::CustomStyling => true,
            ExportFeature::TableOfContents => true,
            ExportFeature::EmbeddedImages => false, // TODO: implement
            ExportFeature::Interactive => false,
        }
    }
    
    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "document_class": {
                    "type": "string",
                    "enum": ["article", "report", "book", "amsart"],
                    "default": "article"
                },
                "geometry": {
                    "type": "string",
                    "default": "margin=1in"
                },
                "font_size": {
                    "type": "string",
                    "enum": ["10pt", "11pt", "12pt"],
                    "default": "11pt"
                },
                "math_packages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["amsmath", "amssymb", "amsfonts"]
                }
            }
        })
    }
}

/// Mathematical expression converter
struct MathConverter {
    function_mappings: HashMap<String, String>,
    symbol_mappings: HashMap<String, String>,
}

impl MathConverter {
    fn new() -> Self {
        let mut function_mappings = HashMap::new();
        
        // Basic mathematical functions
        function_mappings.insert("Sin".to_string(), "\\sin".to_string());
        function_mappings.insert("Cos".to_string(), "\\cos".to_string());
        function_mappings.insert("Tan".to_string(), "\\tan".to_string());
        function_mappings.insert("Log".to_string(), "\\log".to_string());
        function_mappings.insert("Ln".to_string(), "\\ln".to_string());
        function_mappings.insert("Exp".to_string(), "\\exp".to_string());
        function_mappings.insert("Sqrt".to_string(), "\\sqrt".to_string());
        function_mappings.insert("Abs".to_string(), "\\left|".to_string());
        
        // Calculus functions
        function_mappings.insert("D".to_string(), "\\frac{\\partial}{\\partial}".to_string());
        function_mappings.insert("Integrate".to_string(), "\\int".to_string());
        function_mappings.insert("Sum".to_string(), "\\sum".to_string());
        function_mappings.insert("Product".to_string(), "\\prod".to_string());
        function_mappings.insert("Limit".to_string(), "\\lim".to_string());
        
        let mut symbol_mappings = HashMap::new();
        
        // Greek letters
        symbol_mappings.insert("Pi".to_string(), "\\pi".to_string());
        symbol_mappings.insert("Alpha".to_string(), "\\alpha".to_string());
        symbol_mappings.insert("Beta".to_string(), "\\beta".to_string());
        symbol_mappings.insert("Gamma".to_string(), "\\gamma".to_string());
        symbol_mappings.insert("Delta".to_string(), "\\delta".to_string());
        symbol_mappings.insert("Epsilon".to_string(), "\\epsilon".to_string());
        symbol_mappings.insert("Theta".to_string(), "\\theta".to_string());
        symbol_mappings.insert("Lambda".to_string(), "\\lambda".to_string());
        symbol_mappings.insert("Mu".to_string(), "\\mu".to_string());
        symbol_mappings.insert("Nu".to_string(), "\\nu".to_string());
        symbol_mappings.insert("Xi".to_string(), "\\xi".to_string());
        symbol_mappings.insert("Rho".to_string(), "\\rho".to_string());
        symbol_mappings.insert("Sigma".to_string(), "\\sigma".to_string());
        symbol_mappings.insert("Tau".to_string(), "\\tau".to_string());
        symbol_mappings.insert("Phi".to_string(), "\\phi".to_string());
        symbol_mappings.insert("Chi".to_string(), "\\chi".to_string());
        symbol_mappings.insert("Psi".to_string(), "\\psi".to_string());
        symbol_mappings.insert("Omega".to_string(), "\\omega".to_string());
        
        // Special symbols
        symbol_mappings.insert("Infinity".to_string(), "\\infty".to_string());
        symbol_mappings.insert("E".to_string(), "e".to_string());
        symbol_mappings.insert("I".to_string(), "i".to_string());
        
        Self {
            function_mappings,
            symbol_mappings,
        }
    }
    
    /// Check if a string represents a mathematical expression
    fn is_mathematical_expression(&self, s: &str) -> bool {
        // Simple heuristic: contains mathematical operators or function names
        s.contains('^') || s.contains('*') || s.contains('/') || 
        s.contains('+') || s.contains('-') || s.contains('=') ||
        self.function_mappings.keys().any(|f| s.contains(f)) ||
        self.symbol_mappings.keys().any(|f| s.contains(f))
    }
    
    /// Convert Lyra expression to LaTeX
    fn convert_to_latex(&self, expr: &str) -> ExportResult<String> {
        let mut latex = format!("${expr}$");
        
        // Replace function calls: Sin[x] -> \sin(x)
        for (lyra_func, latex_func) in &self.function_mappings {
            let pattern = format!("{}[", lyra_func);
            if latex.contains(&pattern) {
                latex = latex.replace(&pattern, &format!("{}(", latex_func));
                latex = latex.replace(']', ")"); // Replace closing bracket
            }
        }
        
        // Replace symbols
        for (lyra_sym, latex_sym) in &self.symbol_mappings {
            latex = latex.replace(lyra_sym, latex_sym);
        }
        
        // Handle exponentiation
        latex = latex.replace('^', "^{");
        // Count braces to balance them properly
        let open_braces = latex.matches("^{").count();
        for _ in 0..open_braces {
            if let Some(pos) = latex.rfind("^{") {
                if let Some(space_pos) = latex[pos..].find(' ') {
                    latex.insert(pos + space_pos, '}');
                } else if let Some(end_pos) = latex[pos..].find('$') {
                    latex.insert(pos + end_pos, '}');
                }
            }
        }
        
        // Handle fractions: a/b -> \frac{a}{b}
        // This is a simplified version - a full parser would be better
        
        Ok(latex)
    }
    
    /// Convert symbol to LaTeX
    fn convert_symbol_to_latex(&self, symbol: &str) -> ExportResult<String> {
        if let Some(latex_symbol) = self.symbol_mappings.get(symbol) {
            Ok(format!("${latex_symbol}$"))
        } else {
            Ok(format!("$\\text{{{}}}$", escape_latex(symbol)))
        }
    }
}

/// LaTeX syntax highlighter using listings package
struct LaTeXHighlighter {
    lyra_language_def: String,
}

impl LaTeXHighlighter {
    fn new() -> Self {
        let lyra_language_def = r#"
\lstdefinelanguage{Lyra}{
    keywords={If, While, For, Function, Module, True, False, Null},
    keywordstyle=\color{blue}\bfseries,
    ndkeywords={Sin, Cos, Tan, Log, Exp, Sqrt, D, Integrate, Sum, Product},
    ndkeywordstyle=\color{green}\bfseries,
    sensitive=true,
    comment=[l]{\%},
    commentstyle=\color{gray}\itshape,
    string=[b]",
    stringstyle=\color{red},
    basicstyle=\ttfamily\small,
    numbers=left,
    numberstyle=\tiny\color{gray},
    stepnumber=1,
    tabsize=2,
    showstringspaces=false,
    breaklines=true,
    frame=single,
    backgroundcolor=\color{gray!10}
}
"#.to_string();
        
        Self { lyra_language_def }
    }
    
    /// Generate syntax-highlighted code block
    fn highlight_code(&self, code: &str) -> ExportResult<String> {
        let mut latex = String::new();
        
        // Include language definition if not already done
        latex.push_str(&self.lyra_language_def);
        latex.push('\n');
        
        // Create highlighted code block
        latex.push_str("\\begin{lstlisting}[language=Lyra]\n");
        latex.push_str(code);
        if !code.ends_with('\n') {
            latex.push('\n');
        }
        latex.push_str("\\end{lstlisting}\n");
        
        Ok(latex)
    }
}

/// LaTeX document builder
struct DocumentBuilder;

impl DocumentBuilder {
    fn new() -> Self {
        Self
    }
    
    /// Build document header with packages and settings
    fn build_header(&self, config: &ExportConfig) -> ExportResult<String> {
        let mut header = String::new();
        
        // Document class
        let doc_class = config.format_options
            .get("document_class")
            .and_then(|v| v.as_str())
            .unwrap_or("article");
        let font_size = config.format_options
            .get("font_size")
            .and_then(|v| v.as_str())
            .unwrap_or("11pt");
        
        header.push_str(&format!("\\documentclass[{}]{{{}}}\n", font_size, doc_class));
        
        // Packages
        header.push_str("\\usepackage[utf8]{inputenc}\n");
        header.push_str("\\usepackage[T1]{fontenc}\n");
        
        // Geometry
        let geometry = config.format_options
            .get("geometry")
            .and_then(|v| v.as_str())
            .unwrap_or("margin=1in");
        header.push_str(&format!("\\usepackage[{}]{{geometry}}\n", geometry));
        
        // Math packages
        header.push_str("\\usepackage{amsmath}\n");
        header.push_str("\\usepackage{amssymb}\n");
        header.push_str("\\usepackage{amsfonts}\n");
        
        // Syntax highlighting
        if config.syntax_highlighting {
            header.push_str("\\usepackage{listings}\n");
            header.push_str("\\usepackage{xcolor}\n");
        }
        
        // Graphics and tables
        header.push_str("\\usepackage{graphicx}\n");
        header.push_str("\\usepackage{array}\n");
        header.push_str("\\usepackage{tabularx}\n");
        
        // Hyperlinks
        header.push_str("\\usepackage{hyperref}\n");
        header.push_str("\\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}\n");
        
        header.push_str("\n\\begin{document}\n");
        
        Ok(header)
    }
    
    /// Build document footer
    fn build_footer(&self, _config: &ExportConfig) -> ExportResult<String> {
        Ok("\\end{document}\n".to_string())
    }
}

/// Escape special LaTeX characters
fn escape_latex(text: &str) -> String {
    text.replace('\\', "\\textbackslash{}")
        .replace('{', "\\{")
        .replace('}', "\\}")
        .replace('$', "\\$")
        .replace('&', "\\&")
        .replace('%', "\\%")
        .replace('#', "\\#")
        .replace('^', "\\textasciicircum{}")
        .replace('_', "\\_")
        .replace('~', "\\textasciitilde{}")
}

impl Default for LaTeXExporter {
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
                    output: Value::Integer(1),
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
    fn test_latex_exporter_creation() {
        let exporter = LaTeXExporter::new();
        assert_eq!(exporter.default_extension(), "tex");
        assert_eq!(exporter.mime_type(), "application/x-latex");
    }
    
    #[test]
    fn test_math_converter() {
        let converter = MathConverter::new();
        assert!(converter.is_mathematical_expression("Sin[x] + 2"));
        assert!(!converter.is_mathematical_expression("Hello World"));
        
        let latex = converter.convert_to_latex("Sin[x]").unwrap();
        assert!(latex.contains("\\sin"));
    }
    
    #[test]
    fn test_symbol_conversion() {
        let converter = MathConverter::new();
        let result = converter.convert_symbol_to_latex("Pi").unwrap();
        assert!(result.contains("\\pi"));
    }
    
    #[test]
    fn test_latex_export_preview() {
        let exporter = LaTeXExporter::new();
        let snapshot = create_test_snapshot();
        let config = ExportConfig::default();
        
        let preview = exporter.preview(&snapshot, &config).unwrap();
        assert!(preview.contains("\\documentclass"));
        assert!(preview.contains("\\begin{document}"));
        assert!(preview.contains("\\end{document}"));
        assert!(preview.contains("\\sin"));
    }
    
    #[test]
    fn test_feature_support() {
        let exporter = LaTeXExporter::new();
        assert!(exporter.supports_feature(ExportFeature::MathRendering));
        assert!(exporter.supports_feature(ExportFeature::SyntaxHighlighting));
        assert!(!exporter.supports_feature(ExportFeature::Interactive));
    }
    
    #[test]
    fn test_latex_escaping() {
        let escaped = escape_latex("$100 & 50% profit");
        assert!(escaped.contains("\\$"));
        assert!(escaped.contains("\\&"));
        assert!(escaped.contains("\\%"));
    }
    
    #[test]
    fn test_document_builder() {
        let builder = DocumentBuilder::new();
        let config = ExportConfig::default();
        
        let header = builder.build_header(&config).unwrap();
        assert!(header.contains("\\documentclass"));
        assert!(header.contains("\\usepackage{amsmath}"));
        
        let footer = builder.build_footer(&config).unwrap();
        assert!(footer.contains("\\end{document}"));
    }
}