# Lyra Language Support for VS Code

A comprehensive Visual Studio Code extension providing rich language support for the Lyra programming language, including syntax highlighting, IntelliSense, REPL integration, and export capabilities.

## Features

### 🎨 Syntax Highlighting
- Comprehensive syntax highlighting for Lyra language constructs
- Built-in functions, mathematical constants, and Greek letters
- Pattern matching and rule syntax highlighting
- Meta commands and control structures
- Two beautiful themes: Lyra Dark and Lyra Light

### 🧠 IntelliSense & Code Completion
- Smart auto-completion for built-in functions
- Mathematical constants and Greek letters
- Context-aware parameter suggestions
- Pattern and rule completion
- Variable and function completion from document

### 📟 REPL Integration
- Start, stop, and restart Lyra REPL directly from VS Code
- Evaluate current selection or entire file
- Real-time interaction with Lyra interpreter
- Terminal integration with proper output formatting

### 📤 Export Capabilities
- Export to Jupyter notebooks (.ipynb)
- Export to LaTeX documents (.tex) with automatic PDF compilation
- Export to interactive HTML with mathematical rendering
- Quick access via command palette and context menus

### 🔍 Error Detection & Diagnostics
- Real-time syntax error detection
- Bracket balance checking
- Undefined function warnings
- Pattern syntax validation
- Rule syntax validation

### 📖 Documentation & Help
- Hover documentation for built-in functions
- Function signatures with parameter descriptions
- Examples and usage patterns
- Cross-references to related functions

### 🎯 Code Formatting
- Automatic code formatting
- Proper indentation for nested expressions
- Operator spacing normalization
- Function call and list formatting

## Installation

### From VS Code Marketplace
1. Open VS Code
2. Go to Extensions view (Ctrl+Shift+X)
3. Search for "Lyra Language Support"
4. Click Install

### Manual Installation
1. Download the .vsix file from releases
2. Open VS Code
3. Press Ctrl+Shift+P
4. Type "Extensions: Install from VSIX"
5. Select the downloaded .vsix file

## Requirements

- Visual Studio Code 1.74.0 or higher
- Lyra interpreter installed and accessible in PATH

## Configuration

The extension can be configured through VS Code settings:

```json
{
  "lyra.repl.path": "lyra",                    // Path to Lyra executable
  "lyra.repl.args": [],                        // Additional REPL arguments
  "lyra.repl.autoStart": false,               // Auto-start REPL for .lyra files
  "lyra.syntax.enableMathSymbols": true,      // Enable math symbol rendering
  "lyra.completion.enableIntelliSense": true, // Enable auto-completion
  "lyra.completion.includeBultins": true,     // Include built-in functions
  "lyra.export.defaultFormat": "jupyter",     // Default export format
  "lyra.diagnostics.enable": true             // Enable error diagnostics
}
```

## Usage

### Quick Start
1. Create a new file with `.lyra` extension
2. Start typing Lyra code with syntax highlighting
3. Press `Ctrl+Shift+L` to start REPL
4. Use `Ctrl+Shift+E` to evaluate selection
5. Use `Ctrl+Shift+R` to evaluate entire file

### Available Commands
- **Lyra: Start REPL** - Start interactive Lyra session
- **Lyra: Stop REPL** - Stop current REPL session
- **Lyra: Restart REPL** - Restart REPL session
- **Lyra: Evaluate Selection in REPL** - Execute selected code
- **Lyra: Evaluate File in REPL** - Execute entire file
- **Lyra: Export to Jupyter Notebook** - Create .ipynb file
- **Lyra: Export to LaTeX** - Create .tex document
- **Lyra: Export to HTML** - Create interactive HTML
- **Lyra: Open Function Documentation** - Show function help
- **Lyra: Format Lyra Document** - Format code

### Keyboard Shortcuts
- `Ctrl+Shift+L` (Cmd+Shift+L on Mac) - Start REPL
- `Ctrl+Shift+E` (Cmd+Shift+E on Mac) - Evaluate selection
- `Ctrl+Shift+R` (Cmd+Shift+R on Mac) - Evaluate file

## Language Features

### Syntax Highlighting
The extension provides comprehensive syntax highlighting for:
- Mathematical functions (Sin, Cos, Log, etc.)
- List operations (Map, Select, Apply, etc.)
- Pattern matching (Cases, MatchQ, Replace, etc.)
- Control structures (If, Which, For, While, etc.)
- Mathematical constants (Pi, E, Golden Ratio, etc.)
- Greek letters (Alpha, Beta, Gamma, etc.)
- Operators and rules (→, :>, /., etc.)
- Meta commands (%help, %load, etc.)

### Code Snippets
Built-in snippets for common patterns:
- `func` - Function definition
- `if` - Conditional expression
- `for` - For loop
- `while` - While loop
- `module` - Module with local variables
- `rule` - Replacement rule
- `map` - Map operation
- And many more...

## Troubleshooting

### REPL Not Starting
- Ensure Lyra is installed and in your PATH
- Check the `lyra.repl.path` setting
- Verify Lyra executable permissions

### Export Not Working
- Ensure Lyra supports export functionality
- Check file permissions for output directory
- For LaTeX export, ensure pdflatex is installed

### Syntax Highlighting Issues
- Restart VS Code after installation
- Check file is saved with `.lyra` extension
- Verify no conflicting extensions

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This extension is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Support

- 📚 [Documentation](https://lyra-lang.org/docs)
- 🐛 [Issue Tracker](https://github.com/lyra-lang/lyra/issues)
- 💬 [Discussions](https://github.com/lyra-lang/lyra/discussions)
- 📧 [Contact](mailto:support@lyra-lang.org)