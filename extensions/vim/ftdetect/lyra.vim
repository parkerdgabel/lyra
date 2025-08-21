" Vim filetype detection for Lyra
" Language: Lyra
" Maintainer: Lyra Language Team

" Detect Lyra files by extension
autocmd BufNewFile,BufRead *.lyra set filetype=lyra

" Detect Lyra files by shebang
autocmd BufNewFile,BufRead * if getline(1) =~ '^#!.*\blyra\b' | set filetype=lyra | endif

" Detect Lyra files by content patterns
autocmd BufNewFile,BufRead * if getline(1) =~ '^\s*%' && getline(2) =~ 'Lyra' | set filetype=lyra | endif

" Set default file type for new .lyra files
autocmd BufNewFile *.lyra 0put ='// Lyra mathematical computing script' | set filetype=lyra