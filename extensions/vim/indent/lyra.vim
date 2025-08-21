" Vim indent file
" Language: Lyra
" Maintainer: Lyra Language Team

if exists("b:did_indent")
  finish
endif
let b:did_indent = 1

setlocal autoindent
setlocal smartindent
setlocal expandtab
setlocal tabstop=2
setlocal shiftwidth=2
setlocal softtabstop=2

" Enable automatic indentation
setlocal indentexpr=GetLyraIndent(v:lnum)
setlocal indentkeys=!^F,o,O,e,},],),=Module,=Block,=If,=Which,=For,=While

" Only define the function once
if exists("*GetLyraIndent")
  finish
endif

function! GetLyraIndent(lnum)
  " Find a non-blank line above the current line
  let prevlnum = prevnonblank(a:lnum - 1)
  
  " If there's no previous line, don't indent
  if prevlnum == 0
    return 0
  endif
  
  " Get the previous line
  let prevline = getline(prevlnum)
  let curline = getline(a:lnum)
  
  " Start with the indent of the previous line
  let indent = indent(prevlnum)
  
  " Increase indent after opening brackets or control structures
  if prevline =~ '\v[\[{(]\s*$'
    let indent += &shiftwidth
  endif
  
  " Increase indent after control structure keywords
  if prevline =~ '\v<(If|Which|For|While|Module|Block|Function)\s*\['
    let indent += &shiftwidth
  endif
  
  " Increase indent after assignment with multiline expressions
  if prevline =~ '\v:?\=\s*$' || prevline =~ '\v:?\=.*[\[{(]\s*$'
    let indent += &shiftwidth
  endif
  
  " Increase indent after rule operators
  if prevline =~ '\v(-\>|:\>)\s*$'
    let indent += &shiftwidth
  endif
  
  " Decrease indent for closing brackets
  if curline =~ '\v^\s*[\]})]'
    let indent -= &shiftwidth
  endif
  
  " Decrease indent for 'else' and similar keywords
  if curline =~ '\v^\s*(Else|ElseIf)\>'
    let indent -= &shiftwidth
  endif
  
  " Special handling for list elements
  if prevline =~ '\v,\s*$' && curline !~ '\v^\s*[\]})]'
    " Keep same indent as previous line for continued list elements
    return indent(prevlnum)
  endif
  
  " Special handling for function parameters
  if prevline =~ '\v\[\s*$' && curline !~ '\v^\s*\]'
    let indent += &shiftwidth
  endif
  
  " Ensure we don't go negative
  if indent < 0
    let indent = 0
  endif
  
  return indent
endfunction

" Set up bracket matching
setlocal matchpairs+=<:>

" Configure comment formatting
setlocal formatoptions-=t
setlocal formatoptions+=croql

" Set comment strings
setlocal commentstring=//\ %s
setlocal comments=://,s:/*,mb:*,ex:*/