" Vim autoload functions for Lyra
" Language: Lyra
" Maintainer: Lyra Language Team

" Completion function for Lyra
function! lyra#complete(findstart, base)
  if a:findstart
    " Find the start of the current word
    let line = getline('.')
    let start = col('.') - 1
    while start > 0 && line[start - 1] =~ '\a\|_\|\d'
      let start -= 1
    endwhile
    return start
  else
    " Return completion matches
    let completions = []
    
    " Add built-in functions
    let builtin_functions = [
          \ 'Sin', 'Cos', 'Tan', 'ArcSin', 'ArcCos', 'ArcTan',
          \ 'Sinh', 'Cosh', 'Tanh', 'Log', 'Ln', 'Exp', 'Sqrt',
          \ 'Abs', 'Floor', 'Ceil', 'Round', 'Max', 'Min',
          \ 'Sum', 'Product', 'Mean', 'Median', 'Variance',
          \ 'Length', 'First', 'Last', 'Rest', 'Most', 'Take', 'Drop',
          \ 'Join', 'Sort', 'Reverse', 'Union', 'Intersection',
          \ 'Map', 'Apply', 'Select', 'Cases', 'Count', 'Fold',
          \ 'MatchQ', 'Replace', 'ReplaceAll', 'Position',
          \ 'If', 'Which', 'Switch', 'While', 'For', 'Do',
          \ 'Module', 'Block', 'With', 'Function',
          \ 'Print', 'Echo', 'Input', 'Get', 'Put'
          \ ]
    
    " Add mathematical constants
    let math_constants = [
          \ 'Pi', 'E', 'I', 'Infinity', 'ComplexInfinity',
          \ 'GoldenRatio', 'EulerGamma', 'Catalan'
          \ ]
    
    " Add Greek letters
    let greek_letters = [
          \ 'Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta',
          \ 'Eta', 'Theta', 'Iota', 'Kappa', 'Lambda', 'Mu',
          \ 'Nu', 'Xi', 'Omicron', 'Rho', 'Sigma', 'Tau',
          \ 'Upsilon', 'Phi', 'Chi', 'Psi', 'Omega'
          \ ]
    
    " Filter completions based on input
    for item in builtin_functions + math_constants + greek_letters
      if item =~? '^' . a:base
        call add(completions, {
              \ 'word': item,
              \ 'menu': '[Lyra]',
              \ 'kind': item =~? '^\(Sin\|Cos\|Tan\|Log\|Exp\|Sqrt\|Abs\|Floor\|Ceil\|Round\|Max\|Min\)' ? 'f' : 
              \         item =~? '^\(Pi\|E\|I\|Infinity\|Alpha\|Beta\|Gamma\)' ? 'v' : 'f'
              \ })
      endif
    endfor
    
    " Add variables from current buffer
    let variables = lyra#get_buffer_variables()
    for var in variables
      if var =~? '^' . a:base
        call add(completions, {
              \ 'word': var,
              \ 'menu': '[var]',
              \ 'kind': 'v'
              \ })
      endif
    endfor
    
    return completions
  endif
endfunction

" Get variables defined in current buffer
function! lyra#get_buffer_variables()
  let variables = []
  let lines = getline(1, '$')
  
  for line in lines
    " Match variable assignments: varName = value or varName := value
    let matches = matchlist(line, '\([a-zA-Z][a-zA-Z0-9]*\)\s*:*=')
    if !empty(matches)
      let var = matches[1]
      if index(variables, var) == -1
        call add(variables, var)
      endif
    endif
    
    " Match function definitions: funcName[args_] := body
    let matches = matchlist(line, '\([a-zA-Z][a-zA-Z0-9]*\)\[.*\]\s*:*=')
    if !empty(matches)
      let func = matches[1]
      if index(variables, func) == -1
        call add(variables, func)
      endif
    endif
  endfor
  
  return variables
endfunction

" Format Lyra code
function! lyra#format()
  " Save cursor position
  let save_cursor = getcurpos()
  
  " Format the entire buffer
  silent! execute '%s/\s\+$//e'  " Remove trailing whitespace
  silent! execute '%s/\(\S\)\s*\([=!<>]=\?\|&&\|||\)/\1 \2/ge'  " Add spaces around operators
  silent! execute '%s/\([=!<>]=\?\|&&\|||\)\s*\(\S\)/\1 \2/ge'  " Add spaces around operators
  silent! execute '%s/,\s*/, /ge'  " Normalize comma spacing
  silent! execute '%s/{\s*/, /ge'  " Normalize brace spacing
  silent! execute '%s/\s*}/}/ge'  " Normalize brace spacing
  
  " Restore cursor position
  call setpos('.', save_cursor)
  
  echo "Lyra code formatted"
endfunction

" Lint Lyra code (basic syntax checking)
function! lyra#lint()
  let errors = []
  let lines = getline(1, '$')
  let line_num = 0
  
  for line in lines
    let line_num += 1
    
    " Check for unbalanced brackets
    let brackets = {'[': 0, '{': 0, '(': 0}
    let i = 0
    while i < len(line)
      let char = line[i]
      if has_key(brackets, char)
        let brackets[char] += 1
      elseif char == ']'
        let brackets['['] -= 1
      elseif char == '}'
        let brackets['{'] -= 1
      elseif char == ')'
        let brackets['('] -= 1
      endif
      let i += 1
    endwhile
    
    for [bracket, count] in items(brackets)
      if count != 0
        call add(errors, {
              \ 'lnum': line_num,
              \ 'col': 1,
              \ 'text': 'Unbalanced ' . bracket . ' bracket',
              \ 'type': 'E'
              \ })
      endif
    endfor
    
    " Check for invalid operators
    if line =~ '\(===\|&&&\||||\)'
      call add(errors, {
            \ 'lnum': line_num,
            \ 'col': match(line, '\(===\|&&&\||||\)') + 1,
            \ 'text': 'Invalid operator',
            \ 'type': 'E'
            \ })
    endif
    
    " Check for unterminated strings
    if line =~ '"\([^"]\|\\.\)*$\|''\([^'']\|\\.\)*$'
      call add(errors, {
            \ 'lnum': line_num,
            \ 'col': match(line, '"\([^"]\|\\.\)*$\|''\([^'']\|\\.\)*$') + 1,
            \ 'text': 'Unterminated string',
            \ 'type': 'E'
            \ })
    endif
  endfor
  
  " Set quickfix list
  call setqflist(errors, 'r')
  
  if len(errors) > 0
    echo 'Found ' . len(errors) . ' errors. Use :copen to view them.'
    copen
  else
    echo 'No syntax errors found.'
    cclose
  endif
endfunction

" Export buffer to different formats
function! lyra#export(format)
  if &filetype != 'lyra'
    echo "Current buffer is not a Lyra file"
    return
  endif
  
  let filename = expand('%')
  if filename == ''
    echo "Buffer has no associated file. Please save first."
    return
  endif
  
  " Save file if modified
  if &modified
    write
  endif
  
  let base_name = fnamemodify(filename, ':r')
  
  if a:format == 'jupyter'
    let output_file = base_name . '.ipynb'
  elseif a:format == 'latex'
    let output_file = base_name . '.tex'
  elseif a:format == 'html'
    let output_file = base_name . '.html'
  else
    echo "Unknown format: " . a:format
    return
  endif
  
  " Execute export command
  let cmd = g:lyra_repl_command . ' --export ' . a:format . ' --input "' . filename . '" --output "' . output_file . '"'
  let result = system(cmd)
  
  if v:shell_error == 0
    echo "Exported to " . a:format . ": " . output_file
  else
    echo "Export failed: " . result
  endif
endfunction

" Get function documentation
function! lyra#get_doc(word)
  let docs = {
        \ 'Sin': 'Sin[x] - Computes the sine of x (x in radians)',
        \ 'Cos': 'Cos[x] - Computes the cosine of x (x in radians)',
        \ 'Tan': 'Tan[x] - Computes the tangent of x (x in radians)',
        \ 'Log': 'Log[x] - Computes the natural logarithm of x',
        \ 'Exp': 'Exp[x] - Computes the exponential function e^x',
        \ 'Sqrt': 'Sqrt[x] - Computes the square root of x',
        \ 'Abs': 'Abs[x] - Computes the absolute value of x',
        \ 'Length': 'Length[list] - Returns the number of elements in list',
        \ 'Map': 'Map[f, list] - Applies function f to each element of list',
        \ 'Select': 'Select[list, pred] - Returns elements for which pred is True',
        \ 'If': 'If[condition, trueExpr, falseExpr] - Conditional expression',
        \ 'Pi': 'Pi - The mathematical constant π ≈ 3.14159',
        \ 'E': 'E - The mathematical constant e ≈ 2.71828'
        \ }
  
  if has_key(docs, a:word)
    return docs[a:word]
  else
    return 'No documentation available for: ' . a:word
  endif
endfunction

" Show function documentation in preview window
function! lyra#show_doc(word)
  let doc = lyra#get_doc(a:word)
  
  " Open preview window
  pedit __LyraDoc__
  wincmd P
  
  " Set up the preview buffer
  setlocal buftype=nofile
  setlocal noswapfile
  setlocal modifiable
  
  " Clear and add documentation
  silent %delete
  call append(0, [
        \ 'Lyra Documentation: ' . a:word,
        \ repeat('=', len('Lyra Documentation: ' . a:word)),
        \ '',
        \ doc
        \ ])
  
  " Remove the last empty line
  silent $delete
  
  " Make buffer read-only
  setlocal nomodifiable
  setlocal filetype=text
  
  " Return to original window
  wincmd p
endfunction

" Initialize completion for Lyra files
function! lyra#init_completion()
  setlocal omnifunc=lyra#complete
  setlocal completefunc=lyra#complete
endfunction