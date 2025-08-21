" Vim plugin for Lyra language support
" Language: Lyra
" Maintainer: Lyra Language Team
" Version: 1.0

if exists("g:loaded_lyra") || &cp
  finish
endif
let g:loaded_lyra = 1

" Default configuration
if !exists('g:lyra_repl_command')
  let g:lyra_repl_command = 'lyra'
endif

if !exists('g:lyra_repl_args')
  let g:lyra_repl_args = []
endif

if !exists('g:lyra_auto_start_repl')
  let g:lyra_auto_start_repl = 0
endif

if !exists('g:lyra_split_direction')
  let g:lyra_split_direction = 'vertical'
endif

if !exists('g:lyra_repl_size')
  let g:lyra_repl_size = 40
endif

" REPL management variables
let s:lyra_repl_job = 0
let s:lyra_repl_buffer = 0
let s:lyra_repl_window = 0

" Function to start Lyra REPL
function! s:StartLyraREPL()
  if s:lyra_repl_job != 0 && job_status(s:lyra_repl_job) == 'run'
    echo "Lyra REPL is already running"
    call s:ShowREPLWindow()
    return
  endif
  
  " Create new buffer for REPL
  if g:lyra_split_direction == 'horizontal'
    execute 'split'
    execute 'resize ' . g:lyra_repl_size
  else
    execute 'vsplit'
    execute 'vertical resize ' . g:lyra_repl_size
  endif
  
  " Create terminal with Lyra REPL
  let l:cmd = [g:lyra_repl_command] + g:lyra_repl_args
  
  if has('nvim')
    let s:lyra_repl_job = termopen(l:cmd)
    let s:lyra_repl_buffer = bufnr('%')
  else
    let s:lyra_repl_job = term_start(l:cmd, {
          \ 'term_name': 'Lyra REPL',
          \ 'vertical': g:lyra_split_direction == 'vertical',
          \ 'term_finish': 'close'
          \ })
    let s:lyra_repl_buffer = term_getbuf(s:lyra_repl_job)
  endif
  
  let s:lyra_repl_window = win_getid()
  
  " Set buffer options
  setlocal buftype=terminal
  setlocal noswapfile
  setlocal nomodified
  
  echo "Lyra REPL started"
endfunction

" Function to stop Lyra REPL
function! s:StopLyraREPL()
  if s:lyra_repl_job == 0
    echo "No Lyra REPL is running"
    return
  endif
  
  if has('nvim')
    call jobstop(s:lyra_repl_job)
  else
    call job_stop(s:lyra_repl_job)
  endif
  
  " Close REPL window if it exists
  if s:lyra_repl_window != 0 && win_id2win(s:lyra_repl_window) != 0
    call win_gotoid(s:lyra_repl_window)
    close
  endif
  
  let s:lyra_repl_job = 0
  let s:lyra_repl_buffer = 0
  let s:lyra_repl_window = 0
  
  echo "Lyra REPL stopped"
endfunction

" Function to restart Lyra REPL
function! s:RestartLyraREPL()
  call s:StopLyraREPL()
  sleep 500m
  call s:StartLyraREPL()
endfunction

" Function to show REPL window
function! s:ShowREPLWindow()
  if s:lyra_repl_buffer == 0
    echo "No Lyra REPL buffer available"
    return
  endif
  
  " Find if REPL window is already visible
  let l:winid = bufwinid(s:lyra_repl_buffer)
  if l:winid != -1
    call win_gotoid(l:winid)
    return
  endif
  
  " Create new window for existing REPL buffer
  if g:lyra_split_direction == 'horizontal'
    execute 'split'
    execute 'resize ' . g:lyra_repl_size
  else
    execute 'vsplit'
    execute 'vertical resize ' . g:lyra_repl_size
  endif
  
  execute 'buffer ' . s:lyra_repl_buffer
  let s:lyra_repl_window = win_getid()
endfunction

" Function to send text to REPL
function! s:SendToREPL(text)
  if s:lyra_repl_job == 0 || (has('nvim') ? 0 : job_status(s:lyra_repl_job) != 'run')
    echo "No running Lyra REPL found. Starting new one..."
    call s:StartLyraREPL()
    sleep 1000m  " Wait for REPL to start
  endif
  
  " Ensure REPL window is visible
  call s:ShowREPLWindow()
  
  " Send text to terminal
  if has('nvim')
    call chansend(s:lyra_repl_job, a:text . "\n")
  else
    call term_sendkeys(s:lyra_repl_buffer, a:text . "\n")
  endif
endfunction

" Function to evaluate current line
function! s:EvaluateLine()
  let l:line = getline('.')
  if l:line =~ '^\s*$'
    echo "Empty line, nothing to evaluate"
    return
  endif
  
  call s:SendToREPL(l:line)
  echo "Sent line to REPL: " . l:line
endfunction

" Function to evaluate visual selection
function! s:EvaluateSelection() range
  let l:lines = getline(a:firstline, a:lastline)
  let l:text = join(l:lines, "\n")
  
  if l:text =~ '^\s*$'
    echo "Empty selection, nothing to evaluate"
    return
  endif
  
  call s:SendToREPL(l:text)
  echo "Sent selection to REPL (" . len(l:lines) . " lines)"
endfunction

" Function to evaluate entire buffer
function! s:EvaluateBuffer()
  let l:lines = getline(1, '$')
  let l:text = join(l:lines, "\n")
  
  if l:text =~ '^\s*$'
    echo "Empty buffer, nothing to evaluate"
    return
  endif
  
  call s:SendToREPL(l:text)
  echo "Sent entire buffer to REPL (" . len(l:lines) . " lines)"
endfunction

" Function to load current file in REPL
function! s:LoadFile()
  if &filetype != 'lyra'
    echo "Current buffer is not a Lyra file"
    return
  endif
  
  let l:filename = expand('%')
  if l:filename == ''
    echo "Buffer has no associated file"
    return
  endif
  
  " Save file first if modified
  if &modified
    write
  endif
  
  call s:SendToREPL('%load "' . l:filename . '"')
  echo "Loading file in REPL: " . l:filename
endfunction

" Function to get help for word under cursor
function! s:GetHelp()
  let l:word = expand('<cword>')
  if l:word == ''
    echo "No word under cursor"
    return
  endif
  
  call s:SendToREPL('%help ' . l:word)
  echo "Getting help for: " . l:word
endfunction

" Commands
command! LyraREPLStart call s:StartLyraREPL()
command! LyraREPLStop call s:StopLyraREPL()
command! LyraREPLRestart call s:RestartLyraREPL()
command! LyraREPLShow call s:ShowREPLWindow()
command! LyraEvaluateLine call s:EvaluateLine()
command! -range LyraEvaluateSelection <line1>,<line2>call s:EvaluateSelection()
command! LyraEvaluateBuffer call s:EvaluateBuffer()
command! LyraLoadFile call s:LoadFile()
command! LyraHelp call s:GetHelp()

" Key mappings for Lyra files
augroup LyraKeyMappings
  autocmd!
  autocmd FileType lyra nnoremap <buffer> <leader>rs :LyraREPLStart<CR>
  autocmd FileType lyra nnoremap <buffer> <leader>rq :LyraREPLStop<CR>
  autocmd FileType lyra nnoremap <buffer> <leader>rr :LyraREPLRestart<CR>
  autocmd FileType lyra nnoremap <buffer> <leader>el :LyraEvaluateLine<CR>
  autocmd FileType lyra vnoremap <buffer> <leader>es :LyraEvaluateSelection<CR>
  autocmd FileType lyra nnoremap <buffer> <leader>eb :LyraEvaluateBuffer<CR>
  autocmd FileType lyra nnoremap <buffer> <leader>lf :LyraLoadFile<CR>
  autocmd FileType lyra nnoremap <buffer> K :LyraHelp<CR>
  autocmd FileType lyra nnoremap <buffer> <F5> :LyraEvaluateBuffer<CR>
  autocmd FileType lyra inoremap <buffer> <F5> <Esc>:LyraEvaluateBuffer<CR>
augroup END

" Auto-start REPL if configured
augroup LyraAutoStart
  autocmd!
  if g:lyra_auto_start_repl
    autocmd FileType lyra call s:StartLyraREPL()
  endif
augroup END

" Folding support
augroup LyraFolding
  autocmd!
  autocmd FileType lyra setlocal foldmethod=syntax
  autocmd FileType lyra setlocal foldlevel=99
augroup END

" Set up menu items (for GUI Vim)
if has("gui_running")
  amenu &Lyra.Start\ REPL :LyraREPLStart<CR>
  amenu &Lyra.Stop\ REPL :LyraREPLStop<CR>
  amenu &Lyra.Restart\ REPL :LyraREPLRestart<CR>
  amenu &Lyra.-sep1- :
  amenu &Lyra.Evaluate\ Line :LyraEvaluateLine<CR>
  amenu &Lyra.Evaluate\ Selection :LyraEvaluateSelection<CR>
  amenu &Lyra.Evaluate\ Buffer :LyraEvaluateBuffer<CR>
  amenu &Lyra.-sep2- :
  amenu &Lyra.Load\ File :LyraLoadFile<CR>
  amenu &Lyra.Help\ for\ Word :LyraHelp<CR>
endif

" Status line integration
function! LyraREPLStatus()
  if s:lyra_repl_job != 0
    if has('nvim') || job_status(s:lyra_repl_job) == 'run'
      return '[REPL:ON]'
    else
      return '[REPL:OFF]'
    endif
  else
    return '[REPL:OFF]'
  endif
endfunction

" Add to status line for Lyra files
augroup LyraStatusLine
  autocmd!
  autocmd FileType lyra setlocal statusline+=%{LyraREPLStatus()}
augroup END