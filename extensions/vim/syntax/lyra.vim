" Vim syntax file
" Language: Lyra
" Maintainer: Lyra Language Team
" Last Change: 2024
" Version: 1.0

if exists("b:current_syntax")
  finish
endif

" Comments
syn match lyraComment "//.*$" contains=@Spell
syn region lyraBlockComment start="/\*" end="\*/" contains=@Spell fold
hi def link lyraComment Comment
hi def link lyraBlockComment Comment

" Strings
syn region lyraString start=/"/ skip=/\\"/ end=/"/ contains=@Spell
syn region lyraString start=/'/ skip=/\\'/ end=/'/ contains=@Spell
hi def link lyraString String

" Numbers
syn match lyraNumber "\<\d\+\>"
syn match lyraFloat "\<\d\+\.\d\+\([eE][+-]\?\d\+\)\?\>"
syn match lyraComplex "\<\d\+\(\.\d\+\)\?[iI]\>"
hi def link lyraNumber Number
hi def link lyraFloat Float
hi def link lyraComplex Number

" Keywords and Control Structures
syn keyword lyraKeyword If Then Else While For Do Break Continue Return Module Block With
syn keyword lyraKeyword Function True False Null Undefined Missing All None
hi def link lyraKeyword Keyword

" Mathematical Functions
syn keyword lyraMathFunc Sin Cos Tan ArcSin ArcCos ArcTan Sinh Cosh Tanh
syn keyword lyraMathFunc Log Ln Exp Sqrt Abs Floor Ceil Round
syn keyword lyraMathFunc Max Min Sum Product Mean Median Variance StandardDeviation
syn keyword lyraMathFunc Integrate Differentiate Limit Series Solve
hi def link lyraMathFunc Function

" List and Data Functions
syn keyword lyraListFunc Length First Last Rest Most Take Drop Join Sort Reverse
syn keyword lyraListFunc Union Intersection Complement Select Map Apply Fold
syn keyword lyraListFunc FoldLeft FoldRight Scan Transpose Range Table
hi def link lyraListFunc Function

" Pattern Matching Functions
syn keyword lyraPatternFunc MatchQ Cases Count Position Replace ReplaceAll
syn keyword lyraPatternFunc DeleteCases Extract Part Flatten Partition
hi def link lyraPatternFunc Function

" Control Flow Functions
syn keyword lyraControlFunc Which Switch While For Do Module Block With Function
syn keyword lyraControlFunc If Condition TryCatch Throw Catch Finally
hi def link lyraControlFunc Statement

" I/O Functions
syn keyword lyraIOFunc Print Echo Input Get Put Import Export
syn keyword lyraIOFunc ReadString WriteString OpenRead OpenWrite Close
hi def link lyraIOFunc Function

" Mathematical Constants
syn keyword lyraMathConst Pi E I Infinity ComplexInfinity GoldenRatio
syn keyword lyraMathConst EulerGamma Catalan Degree
hi def link lyraMathConst Constant

" Greek Letters
syn keyword lyraGreek Alpha Beta Gamma Delta Epsilon Zeta Eta Theta
syn keyword lyraGreek Iota Kappa Lambda Mu Nu Xi Omicron Rho
syn keyword lyraGreek Sigma Tau Upsilon Phi Chi Psi Omega
hi def link lyraGreek Constant

" Operators
syn match lyraOperator ":="
syn match lyraOperator "="
syn match lyraOperator "+="
syn match lyraOperator "-="
syn match lyraOperator "\*="
syn match lyraOperator "/="
syn match lyraOperator "=="
syn match lyraOperator "!="
syn match lyraOperator "<="
syn match lyraOperator ">="
syn match lyraOperator "<"
syn match lyraOperator ">"
syn match lyraOperator "&&"
syn match lyraOperator "||"
syn match lyraOperator "!"
syn match lyraOperator "+"
syn match lyraOperator "-"
syn match lyraOperator "\*"
syn match lyraOperator "/"
syn match lyraOperator "\^"
syn match lyraOperator "%"
syn match lyraOperator "&"
syn match lyraOperator "@"
syn match lyraOperator "@@"
syn match lyraOperator "/@"
syn match lyraOperator "//@"
hi def link lyraOperator Operator

" Rule Operators
syn match lyraRule "->"
syn match lyraRule ":>"
syn match lyraRule "/\."
hi def link lyraRule Special

" Pattern Operators
syn match lyraPattern "_"
syn match lyraPattern "__"
syn match lyraPattern "___"
syn match lyraPattern "\?"
hi def link lyraPattern Special

" Brackets
syn match lyraBracket "[\[\]{}()]"
hi def link lyraBracket Delimiter

" Function Calls
syn match lyraFunctionCall "\<[A-Z][a-zA-Z0-9]*\ze\["
hi def link lyraFunctionCall Function

" Variables and Symbols
syn match lyraVariable "\<[a-z][a-zA-Z0-9]*\>"
hi def link lyraVariable Identifier

" Pattern Variables
syn match lyraPatternVar "\<[a-zA-Z][a-zA-Z0-9]*_\+\>"
hi def link lyraPatternVar Special

" Meta Commands
syn match lyraMetaCommand "%\w\+"
hi def link lyraMetaCommand PreProc

" Type Annotations
syn keyword lyraType Integer Real String Symbol Boolean List Function Pattern
syn keyword lyraType Complex Matrix Vector Set Association Rule
hi def link lyraType Type

" Folding
syn region lyraFold start="{" end="}" transparent fold
syn region lyraFold start="\[" end="\]" transparent fold
syn region lyraFold start="(" end=")" transparent fold

" Error highlighting for common mistakes
syn match lyraError "\v\=\=\="
syn match lyraError "\v\&\&\&"
syn match lyraError "\v\|\|\|"
syn match lyraError "\v\_\{4,}"
hi def link lyraError Error

" Spell checking in comments and strings
syn cluster lyraSpellCheck contains=lyraComment,lyraBlockComment,lyraString

" Define syntax regions for better highlighting
syn region lyraList start="{" end="}" contains=ALL
syn region lyraFunctionArgs start="\[" end="\]" contains=ALL
syn region lyraParens start="(" end=")" contains=ALL

" Conditional syntax highlighting based on context
syn match lyraConditional "\<If\>\ze\s*\["
syn match lyraLoop "\<\(For\|While\|Do\)\>\ze\s*\["
hi def link lyraConditional Conditional
hi def link lyraLoop Repeat

" Mathematical expressions
syn match lyraMathExpr "\<\(Sin\|Cos\|Tan\|Log\|Exp\|Sqrt\)\>\ze\["
hi def link lyraMathExpr Function

" Special highlighting for important patterns
syn match lyraImportant "\<\(Module\|Block\|Function\)\>\ze\s*\["
hi def link lyraImportant Structure

" Set filetype options
setlocal commentstring=//\ %s
setlocal comments=://,s:/*,mb:*,ex:*/
setlocal iskeyword+=_

" Indentation
setlocal autoindent
setlocal smartindent
setlocal expandtab
setlocal tabstop=2
setlocal shiftwidth=2
setlocal softtabstop=2

" Matching brackets
setlocal matchpairs+=<:>

" Set syntax name
let b:current_syntax = "lyra"

" Conceal support for mathematical symbols (optional)
if has('conceal')
  syn match lyraConcealed "Pi" conceal cchar=π
  syn match lyraConcealed "Alpha" conceal cchar=α
  syn match lyraConcealed "Beta" conceal cchar=β
  syn match lyraConcealed "Gamma" conceal cchar=γ
  syn match lyraConcealed "Delta" conceal cchar=δ
  syn match lyraConcealed "Epsilon" conceal cchar=ε
  syn match lyraConcealed "Theta" conceal cchar=θ
  syn match lyraConcealed "Lambda" conceal cchar=λ
  syn match lyraConcealed "Mu" conceal cchar=μ
  syn match lyraConcealed "Sigma" conceal cchar=σ
  syn match lyraConcealed "Phi" conceal cchar=φ
  syn match lyraConcealed "Omega" conceal cchar=ω
  syn match lyraConcealed "Infinity" conceal cchar=∞
  syn match lyraConcealed "\<I\>" conceal cchar=𝑖
  syn match lyraConcealed "\<E\>" conceal cchar=ℯ
  
  hi def link lyraConcealed Conceal
  setlocal conceallevel=2
  setlocal concealcursor=nc
endif