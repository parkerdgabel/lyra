# Lyra Vim Plugin

A comprehensive Vim plugin for the Lyra programming language, providing syntax highlighting, REPL integration, code completion, and various editing enhancements.

## Features

### 🎨 Syntax Highlighting
- Complete syntax highlighting for Lyra language constructs
- Mathematical functions, constants, and Greek letters
- Pattern matching and rule syntax
- Meta commands and control structures
- Mathematical symbol concealing (optional)

### 📟 REPL Integration
- Start, stop, and restart Lyra REPL from within Vim
- Evaluate current line, selection, or entire buffer
- Load files directly into REPL
- Real-time interaction with Lyra interpreter

### 🧠 Code Completion
- Built-in function completion
- Mathematical constants and Greek letters
- Variable completion from current buffer
- Context-aware suggestions

### 📚 Documentation
- Built-in help for Lyra functions
- Preview window documentation
- Quick help with `K` key

### 🔧 Code Tools
- Automatic indentation
- Code formatting
- Basic syntax checking/linting
- Export to Jupyter, LaTeX, and HTML

## Installation

### Using vim-plug
Add to your `.vimrc`:
```vim
Plug 'lyra-lang/lyra', {'rtp': 'extensions/vim'}
```

### Using Vundle
Add to your `.vimrc`:
```vim
Plugin 'lyra-lang/lyra', {'rtp': 'extensions/vim'}
```

### Using Pathogen
```bash
cd ~/.vim/bundle
git clone https://github.com/lyra-lang/lyra.git
```

### Manual Installation
Copy the plugin files to your Vim configuration directory:
```bash
cp -r extensions/vim/* ~/.vim/
```

## Configuration

Add these settings to your `.vimrc` to customize the plugin:

```vim
" Lyra REPL configuration
let g:lyra_repl_command = 'lyra'           " Path to Lyra executable
let g:lyra_repl_args = []                  " Additional REPL arguments
let g:lyra_auto_start_repl = 0             " Auto-start REPL for .lyra files
let g:lyra_split_direction = 'vertical'    " REPL split direction ('vertical' or 'horizontal')
let g:lyra_repl_size = 40                  " Size of REPL window

" Enable mathematical symbol concealing
set conceallevel=2
set concealcursor=nc
```

## Usage

### File Type Detection
The plugin automatically detects Lyra files by:
- File extension: `.lyra`
- Shebang: `#!/usr/bin/env lyra`
- Content patterns

### Key Mappings (in Lyra files)
- `<leader>rs` - Start REPL
- `<leader>rq` - Stop REPL  
- `<leader>rr` - Restart REPL
- `<leader>el` - Evaluate current line
- `<leader>es` - Evaluate visual selection
- `<leader>eb` - Evaluate entire buffer
- `<leader>lf` - Load current file in REPL
- `K` - Show help for word under cursor
- `<F5>` - Evaluate entire buffer

### Commands

#### REPL Management
- `:LyraREPLStart` - Start Lyra REPL
- `:LyraREPLStop` - Stop Lyra REPL
- `:LyraREPLRestart` - Restart Lyra REPL
- `:LyraREPLShow` - Show REPL window

#### Code Evaluation
- `:LyraEvaluateLine` - Evaluate current line
- `:LyraEvaluateSelection` - Evaluate visual selection
- `:LyraEvaluateBuffer` - Evaluate entire buffer
- `:LyraLoadFile` - Load current file in REPL

#### Utilities
- `:LyraHelp` - Get help for word under cursor

### Code Completion
The plugin provides omni-completion (`<C-x><C-o>`) with:
- Built-in Lyra functions
- Mathematical constants
- Greek letters  
- Variables from current buffer

### Code Formatting
Basic code formatting is available through the autoload functions:
```vim
:call lyra#format()
```

### Syntax Checking
Basic syntax checking with quickfix integration:
```vim
:call lyra#lint()
```

### Export Functions
Export current file to different formats:
```vim
:call lyra#export('jupyter')  " Export to Jupyter notebook
:call lyra#export('latex')    " Export to LaTeX
:call lyra#export('html')     " Export to HTML
```

## Syntax Highlighting Features

### Function Categories
- **Mathematical functions**: `Sin`, `Cos`, `Log`, `Exp`, etc.
- **List functions**: `Map`, `Select`, `Apply`, `Length`, etc.
- **Pattern functions**: `MatchQ`, `Cases`, `Replace`, etc.
- **Control functions**: `If`, `Which`, `For`, `While`, etc.

### Constants and Symbols
- **Mathematical constants**: `Pi`, `E`, `Infinity`, `GoldenRatio`
- **Greek letters**: `Alpha`, `Beta`, `Gamma`, `Delta`, etc.
- **Boolean values**: `True`, `False`

### Operators and Patterns
- **Assignment**: `=`, `:=`
- **Comparison**: `==`, `!=`, `<`, `>`, `<=`, `>=`
- **Logical**: `&&`, `||`, `!`
- **Rules**: `->`, `:>`, `/.`
- **Patterns**: `_`, `__`, `___`

### Mathematical Symbol Concealing
When enabled, common mathematical symbols are displayed as Unicode:
- `Pi` → π
- `Alpha` → α  
- `Infinity` → ∞
- `I` → 𝑢
- `E` → ℯ

## Troubleshooting

### REPL Not Starting
1. Ensure Lyra is installed and in your PATH
2. Check `g:lyra_repl_command` setting
3. Verify Lyra executable permissions

### Syntax Highlighting Not Working
1. Ensure file has `.lyra` extension
2. Check `:set filetype?` output
3. Restart Vim after installation

### Completion Not Working
1. Ensure you're using `<C-x><C-o>` for omni-completion
2. Check that the file type is detected as 'lyra'
3. Try `:call lyra#init_completion()` manually

### Performance Issues
1. Disable concealing if not needed: `set conceallevel=0`
2. Reduce REPL window size
3. Use `:LyraREPLStop` when not actively using REPL

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This plugin is licensed under the MIT License. See LICENSE for details.

## Support

- 📚 [Lyra Documentation](https://lyra-lang.org/docs)
- 🐛 [Issue Tracker](https://github.com/lyra-lang/lyra/issues)
- 💬 [Discussions](https://github.com/lyra-lang/lyra/discussions)

## Changelog

### Version 1.0
- Initial release
- Complete syntax highlighting
- REPL integration
- Code completion
- Basic documentation support
- Export functionality