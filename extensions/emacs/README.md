# Lyra Mode for Emacs

A comprehensive Emacs package providing full support for the Lyra programming language, including major mode, org-babel integration, REPL support, and export capabilities.

## Features

### 🎨 Major Mode (`lyra-mode`)
- **Syntax Highlighting**: Complete font-lock support for all Lyra constructs
- **Smart Indentation**: Context-aware indentation for nested expressions
- **Code Completion**: Built-in function and symbol completion
- **REPL Integration**: Interactive development with persistent sessions
- **Export Functions**: Export to Jupyter, LaTeX, and HTML formats

### 📚 Org-Babel Integration (`ob-lyra`)
- **Literate Programming**: Execute Lyra code blocks in org-mode documents
- **Session Support**: Persistent execution environment across code blocks
- **Variable Passing**: Pass data between org and Lyra seamlessly
- **Result Handling**: Multiple output formats and display options
- **Export Integration**: Include Lyra results in org-mode exports

### 🔧 Additional Features
- **Menu Integration**: Easy access via Emacs menu system
- **Customizable Settings**: Configure behavior via Emacs customization
- **Documentation Support**: Built-in help and function documentation
- **File Association**: Automatic mode activation for `.lyra` files

## Installation

### Via Package Manager (Recommended)

#### MELPA (when available)
```elisp
(use-package lyra-mode
  :ensure t)
```

#### Manual Package Installation
1. Download the package files
2. Add to your `load-path`:
```elisp
(add-to-list 'load-path "/path/to/lyra-mode")
(require 'lyra-mode)
(require 'ob-lyra)
```

### Manual Installation
1. Clone or download the Lyra repository
2. Copy the Emacs files to your configuration directory:
```bash
cp extensions/emacs/*.el ~/.emacs.d/lisp/
```

3. Add to your Emacs configuration:
```elisp
(add-to-list 'load-path "~/.emacs.d/lisp")
(require 'lyra-mode)
(require 'ob-lyra)
```

## Configuration

### Basic Configuration
Add to your Emacs init file (`.emacs` or `init.el`):

```elisp
;; Basic Lyra mode setup
(require 'lyra-mode)
(require 'ob-lyra)

;; Auto-mode association
(add-to-list 'auto-mode-alist '("\\.lyra\\'" . lyra-mode))

;; Org-babel language support
(org-babel-do-load-languages
 'org-babel-load-languages
 '((lyra . t)
   (emacs-lisp . t)
   ;; other languages...
   ))
```

### Advanced Configuration
```elisp
(use-package lyra-mode
  :mode "\\.lyra\\'"
  :config
  ;; Customize Lyra program path
  (setq lyra-program-name "/usr/local/bin/lyra")
  
  ;; Customize indentation
  (setq lyra-indent-offset 2)
  
  ;; Auto-start REPL for Lyra files
  (setq lyra-auto-start-repl t)
  
  ;; Custom REPL buffer name
  (setq lyra-repl-buffer-name "*My-Lyra-REPL*")
  
  ;; Add hooks
  (add-hook 'lyra-mode-hook
            (lambda ()
              (electric-pair-mode 1)
              (show-paren-mode 1)))
  
  ;; Org-babel configuration
  (with-eval-after-load 'org
    (org-babel-do-load-languages
     'org-babel-load-languages
     '((lyra . t))))
  
  ;; Custom keybindings
  :bind (:map lyra-mode-map
              ("C-c C-c" . lyra-eval-region)
              ("C-c C-k" . lyra-eval-buffer)))
```

## Usage

### Basic Editing

#### File Type Detection
The mode automatically activates for files with `.lyra` extension:
```lyra
// This is a Lyra file
x = Sin[Pi/2] + Cos[0]
Print[x]
```

#### Syntax Highlighting
Comprehensive highlighting for:
- **Functions**: `Sin`, `Cos`, `Map`, `Select`, etc.
- **Constants**: `Pi`, `E`, `Infinity`, Greek letters
- **Keywords**: `If`, `While`, `Module`, `Function`
- **Operators**: `->`, `:>`, `/.`, `==`, etc.
- **Comments**: `//` line comments and `/* */` blocks

### REPL Integration

#### Starting the REPL
```
M-x lyra-start-repl  (or C-c C-z)
```

#### Evaluation Commands
- `C-c C-l` - Evaluate current line
- `C-c C-e` - Evaluate region
- `C-c C-b` - Evaluate entire buffer
- `C-c C-f` - Load current file in REPL

#### REPL Management
- `C-c C-z` - Start REPL
- `C-c C-q` - Stop REPL
- `C-c C-r` - Restart REPL

### Code Completion
Use standard Emacs completion:
- `M-TAB` or `C-M-i` - Complete symbol at point
- Works with built-in functions, constants, and buffer symbols

### Export Functions
- `C-c C-x j` - Export to Jupyter notebook
- `C-c C-x l` - Export to LaTeX
- `C-c C-x h` - Export to HTML

### Documentation
- `C-c C-h` - Get help for symbol at point

## Org-Babel Integration

### Basic Usage
```org
#+BEGIN_SRC lyra
x = 2 + 3
Print["Result: ", x]
#+END_SRC

#+RESULTS:
: Result: 5
```

### Session-Based Execution
```org
#+BEGIN_SRC lyra :session my-session
x = 10
y = 20
#+END_SRC

#+BEGIN_SRC lyra :session my-session
z = x + y
Print[z]
#+END_SRC

#+RESULTS:
: 30
```

### Variable Passing
```org
#+NAME: input-data
| 1 | 2 | 3 | 4 | 5 |

#+BEGIN_SRC lyra :var data=input-data
mean = Mean[data]
Print["Mean: ", mean]
#+END_SRC

#+RESULTS:
: Mean: 3
```

### File Output
```org
#+BEGIN_SRC lyra :file plot.png
data = Range[1, 10]
Plot[Sin[x], {x, 0, 2*Pi}]
Export["plot.png"]
#+END_SRC

#+RESULTS:
[[file:plot.png]]
```

### Advanced Examples

#### Mathematical Computation
```org
#+BEGIN_SRC lyra :exports both
(* Solve a quadratic equation *)
eq = x^2 - 5*x + 6 == 0
solutions = Solve[eq, x]
Print["Solutions: ", solutions]

(* Verify solutions *)
Map[Function[sol, eq /. sol], solutions]
#+END_SRC
```

#### Data Analysis
```org
#+BEGIN_SRC lyra :session analysis :var dataset=data-table
(* Load and analyze data *)
stats = {
  "mean" -> Mean[dataset],
  "median" -> Median[dataset],
  "std" -> StandardDeviation[dataset]
}
Print["Statistics: ", stats]
#+END_SRC
```

#### Literate Programming
```org
* Data Processing Pipeline

First, we load our data:
#+BEGIN_SRC lyra :session pipeline
rawData = Import["data.csv"]
Print["Loaded ", Length[rawData], " records"]
#+END_SRC

Then we clean and process it:
#+BEGIN_SRC lyra :session pipeline
cleanData = Select[rawData, Function[row, Length[row] == 5]]
processedData = Map[Function[row, Take[row, 3]], cleanData]
#+END_SRC

Finally, we analyze the results:
#+BEGIN_SRC lyra :session pipeline
results = Map[Mean, Transpose[processedData]]
Print["Column means: ", results]
#+END_SRC
```

## Customization

### Available Options
- `lyra-program-name`: Path to Lyra executable (default: "lyra")
- `lyra-program-args`: Arguments for Lyra interpreter (default: nil)
- `lyra-indent-offset`: Indentation width (default: 2)
- `lyra-auto-start-repl`: Auto-start REPL for Lyra files (default: nil)
- `lyra-repl-buffer-name`: REPL buffer name (default: "*Lyra*")

### Customization Groups
Use `M-x customize-group RET lyra RET` to customize via Emacs interface.

### Hooks
- `lyra-mode-hook`: Run when entering Lyra mode
- `org-babel-after-execute-hook`: Run after executing org-babel blocks

## Troubleshooting

### REPL Not Starting
1. Ensure Lyra is installed and in PATH
2. Check `lyra-program-name` setting
3. Verify Lyra executable permissions

### Syntax Highlighting Issues
1. Ensure file has `.lyra` extension
2. Try `M-x lyra-mode` manually
3. Check for conflicting modes

### Org-Babel Not Working
1. Ensure `ob-lyra` is loaded
2. Check org-babel language configuration
3. Verify Lyra is available in PATH

### Completion Not Working
1. Ensure point is on a symbol
2. Try `M-x completion-at-point`
3. Check completion functions are loaded

### Performance Issues
1. Disable auto-start REPL if not needed
2. Use sessions for multiple evaluations
3. Consider increasing `gc-cons-threshold`

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with various Emacs versions
5. Submit a pull request

### Development Setup
```bash
git clone https://github.com/lyra-lang/lyra.git
cd lyra/extensions/emacs

# Test with Emacs
emacs -Q -l lyra-mode.el
```

## License

This package is licensed under the GPL-3.0 License. See LICENSE for details.

## Support

- 📚 [Lyra Documentation](https://lyra-lang.org/docs)
- 🐛 [Issue Tracker](https://github.com/lyra-lang/lyra/issues)
- 💬 [Discussions](https://github.com/lyra-lang/lyra/discussions)
- 📧 [Mailing List](mailto:lyra-emacs@lyra-lang.org)

## Changelog

### Version 1.0.0
- Initial release
- Complete major mode with syntax highlighting
- REPL integration with session support
- Org-babel integration for literate programming
- Export functionality
- Code completion and documentation
- Comprehensive test suite

## Related Packages

- **[org-mode](https://orgmode.org/)**: Literate programming support
- **[company-mode](https://company-mode.github.io/)**: Enhanced completion
- **[flycheck](https://www.flycheck.org/)**: Syntax checking (with Lyra support)
- **[smartparens](https://github.com/Fuco1/smartparens)**: Advanced parentheses handling

## Tips and Tricks

### Efficient Workflow
1. Use sessions for interactive development
2. Leverage org-babel for documentation
3. Export to multiple formats for sharing
4. Use completion extensively

### Integration with Other Tools
```elisp
;; Company mode integration
(use-package company
  :config
  (add-hook 'lyra-mode-hook 'company-mode))

;; Projectile integration
(use-package projectile
  :config
  (add-to-list 'projectile-project-root-files "lyra.toml"))

;; Which-key for keybinding help
(use-package which-key
  :config
  (which-key-mode))
```