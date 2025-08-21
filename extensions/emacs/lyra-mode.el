;;; lyra-mode.el --- Major mode for Lyra language  -*- lexical-binding: t; -*-

;; Copyright (C) 2024 Lyra Language Team

;; Author: Lyra Language Team
;; URL: https://github.com/lyra-lang/lyra
;; Version: 1.0.0
;; Package-Requires: ((emacs "24.3"))
;; Keywords: languages, lyra, mathematical, symbolic

;; This file is not part of GNU Emacs.

;; This program is free software; you can redistribute it and/or modify
;; it under the terms of the GNU General Public License as published by
;; the Free Software Foundation, either version 3 of the License, or
;; (at your option) any later version.

;; This program is distributed in the hope that it will be useful,
;; but WITHOUT ANY WARRANTY; without even the implied warranty of
;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;; GNU General Public License for more details.

;;; Commentary:

;; This package provides a major mode for editing Lyra programming language files.
;; Lyra is a mathematical and symbolic computing language.
;;
;; Features:
;; - Syntax highlighting with font-lock
;; - Proper indentation
;; - REPL integration
;; - Code completion
;; - Export capabilities
;; - org-babel support

;;; Code:

(require 'comint)
(require 'thingatpt)

;;; Customization

(defgroup lyra nil
  "Major mode for Lyra language."
  :group 'languages
  :prefix "lyra-")

(defcustom lyra-program-name "lyra"
  "Name of the Lyra interpreter program."
  :type 'string
  :group 'lyra)

(defcustom lyra-program-args '()
  "Arguments to pass to the Lyra interpreter."
  :type '(repeat string)
  :group 'lyra)

(defcustom lyra-indent-offset 2
  "Number of spaces for each indentation step in Lyra mode."
  :type 'integer
  :group 'lyra)

(defcustom lyra-auto-start-repl nil
  "Whether to automatically start REPL when opening Lyra files."
  :type 'boolean
  :group 'lyra)

(defcustom lyra-repl-buffer-name "*Lyra*"
  "Name of the Lyra REPL buffer."
  :type 'string
  :group 'lyra)

;;; Syntax

(defconst lyra-font-lock-keywords
  `(
    ;; Comments
    ("//.*$" . font-lock-comment-face)
    ("/\\*.*?\\*/" . font-lock-comment-face)
    
    ;; Strings
    ("\"\\(?:[^\"\\\\]\\|\\\\.\\)*\"" . font-lock-string-face)
    ("'\\(?:[^'\\\\]\\|\\\\.\\)*'" . font-lock-string-face)
    
    ;; Numbers
    ("\\<\\d+\\(?:\\.\\d+\\)?\\(?:[eE][+-]?\\d+\\)?[iI]?\\>" . font-lock-constant-face)
    
    ;; Mathematical functions
    (,(concat "\\<\\("
              (mapconcat 'identity
                         '("Sin" "Cos" "Tan" "ArcSin" "ArcCos" "ArcTan"
                           "Sinh" "Cosh" "Tanh" "Log" "Ln" "Exp" "Sqrt"
                           "Abs" "Floor" "Ceil" "Round" "Max" "Min"
                           "Sum" "Product" "Mean" "Median" "Variance"
                           "StandardDeviation" "Integrate" "Differentiate")
                         "\\|")
              "\\)\\>")
     . font-lock-function-name-face)
    
    ;; List functions
    (,(concat "\\<\\("
              (mapconcat 'identity
                         '("Length" "First" "Last" "Rest" "Most" "Take" "Drop"
                           "Join" "Sort" "Reverse" "Union" "Intersection"
                           "Map" "Apply" "Select" "Cases" "Count" "Fold"
                           "FoldLeft" "FoldRight" "Scan" "Transpose"
                           "Range" "Table" "Partition" "Flatten")
                         "\\|")
              "\\)\\>")
     . font-lock-function-name-face)
    
    ;; Pattern matching functions
    (,(concat "\\<\\("
              (mapconcat 'identity
                         '("MatchQ" "Replace" "ReplaceAll" "Position"
                           "Extract" "DeleteCases" "Part")
                         "\\|")
              "\\)\\>")
     . font-lock-function-name-face)
    
    ;; Control structures
    (,(concat "\\<\\("
              (mapconcat 'identity
                         '("If" "Which" "Switch" "While" "For" "Do"
                           "Module" "Block" "With" "Function"
                           "Return" "Break" "Continue")
                         "\\|")
              "\\)\\>")
     . font-lock-keyword-face)
    
    ;; I/O functions
    (,(concat "\\<\\("
              (mapconcat 'identity
                         '("Print" "Echo" "Input" "Get" "Put"
                           "Import" "Export" "ReadString" "WriteString")
                         "\\|")
              "\\)\\>")
     . font-lock-function-name-face)
    
    ;; Mathematical constants
    (,(concat "\\<\\("
              (mapconcat 'identity
                         '("Pi" "E" "I" "Infinity" "ComplexInfinity"
                           "GoldenRatio" "EulerGamma" "Catalan" "Degree")
                         "\\|")
              "\\)\\>")
     . font-lock-constant-face)
    
    ;; Greek letters
    (,(concat "\\<\\("
              (mapconcat 'identity
                         '("Alpha" "Beta" "Gamma" "Delta" "Epsilon" "Zeta"
                           "Eta" "Theta" "Iota" "Kappa" "Lambda" "Mu"
                           "Nu" "Xi" "Omicron" "Rho" "Sigma" "Tau"
                           "Upsilon" "Phi" "Chi" "Psi" "Omega")
                         "\\|")
              "\\)\\>")
     . font-lock-constant-face)
    
    ;; Boolean constants
    ("\\<\\(True\\|False\\|Null\\|Undefined\\|Missing\\)\\>" . font-lock-constant-face)
    
    ;; Function calls
    ("\\<\\([A-Z][a-zA-Z0-9]*\\)\\s-*\\[" 1 font-lock-function-name-face)
    
    ;; Variables
    ("\\<\\([a-z][a-zA-Z0-9]*\\)\\>" . font-lock-variable-name-face)
    
    ;; Pattern variables
    ("\\<\\([a-zA-Z][a-zA-Z0-9]*\\)_+\\>" . font-lock-type-face)
    
    ;; Meta commands
    ("%\\w+" . font-lock-preprocessor-face)
    
    ;; Operators
    ("\\(:=\\|->\\|:>\\|/\\.\\|==\\|!=\\|<=\\|>=\\|&&\\|||\\)" . font-lock-builtin-face)
    
    ;; Brackets and delimiters
    ("\\[\\|\\]\\|{\\|}\\|(\\|)" . font-lock-delimiter-face)
    )
  "Keyword highlighting specification for Lyra mode.")

;;; Indentation

(defun lyra-indent-line ()
  "Indent current line as Lyra code."
  (interactive)
  (let ((indent-col 0)
        (prev-line-indent 0))
    (save-excursion
      (beginning-of-line)
      (if (bobp)
          (setq indent-col 0)
        (let ((cur-line (thing-at-point 'line t))
              (prev-line nil))
          ;; Get previous non-empty line
          (forward-line -1)
          (while (and (not (bobp)) 
                      (string-match "^\\s-*$" (thing-at-point 'line t)))
            (forward-line -1))
          (setq prev-line (thing-at-point 'line t))
          (setq prev-line-indent (current-indentation))
          
          ;; Calculate indentation
          (setq indent-col prev-line-indent)
          
          ;; Increase indent after opening brackets or control structures
          (when (string-match "\\[\\s-*$\\|{\\s-*$\\|(\\s-*$" prev-line)
            (setq indent-col (+ indent-col lyra-indent-offset)))
          
          ;; Increase indent after control structures
          (when (string-match "\\<\\(If\\|Which\\|For\\|While\\|Module\\|Block\\|Function\\)\\s-*\\[" prev-line)
            (setq indent-col (+ indent-col lyra-indent-offset)))
          
          ;; Increase indent after assignment
          (when (string-match ":?=\\s-*$" prev-line)
            (setq indent-col (+ indent-col lyra-indent-offset)))
          
          ;; Decrease indent for closing brackets
          (when (string-match "^\\s-*\\(\\]\\|}\\|)\\)" cur-line)
            (setq indent-col (max 0 (- indent-col lyra-indent-offset))))
          
          ;; Ensure non-negative
          (setq indent-col (max 0 indent-col)))))
    
    ;; Apply indentation
    (save-excursion
      (beginning-of-line)
      (delete-horizontal-space)
      (indent-to indent-col))
    
    ;; Move point to first non-whitespace character if at beginning
    (when (< (current-column) (current-indentation))
      (back-to-indentation))))

;;; REPL Integration

(defvar lyra-repl-process nil
  "The current Lyra REPL process.")

(defun lyra-start-repl ()
  "Start a Lyra REPL in a buffer."
  (interactive)
  (if (and lyra-repl-process (process-live-p lyra-repl-process))
      (switch-to-buffer-other-window (process-buffer lyra-repl-process))
    (let ((buffer (get-buffer-create lyra-repl-buffer-name)))
      (with-current-buffer buffer
        (comint-mode)
        (setq lyra-repl-process
              (apply #'start-process "lyra-repl" buffer
                     lyra-program-name lyra-program-args))
        (set-process-query-on-exit-flag lyra-repl-process nil))
      (switch-to-buffer-other-window buffer))))

(defun lyra-stop-repl ()
  "Stop the Lyra REPL."
  (interactive)
  (when (and lyra-repl-process (process-live-p lyra-repl-process))
    (process-send-eof lyra-repl-process)
    (setq lyra-repl-process nil)
    (message "Lyra REPL stopped")))

(defun lyra-restart-repl ()
  "Restart the Lyra REPL."
  (interactive)
  (lyra-stop-repl)
  (sleep-for 1)
  (lyra-start-repl))

(defun lyra-send-to-repl (text)
  "Send TEXT to the Lyra REPL."
  (unless (and lyra-repl-process (process-live-p lyra-repl-process))
    (lyra-start-repl)
    (sleep-for 1))
  (with-current-buffer (process-buffer lyra-repl-process)
    (goto-char (point-max))
    (insert text)
    (comint-send-input)))

(defun lyra-eval-line ()
  "Evaluate the current line in the REPL."
  (interactive)
  (let ((line (thing-at-point 'line t)))
    (lyra-send-to-repl line)
    (message "Sent line to REPL")))

(defun lyra-eval-region (start end)
  "Evaluate the region from START to END in the REPL."
  (interactive "r")
  (let ((text (buffer-substring-no-properties start end)))
    (lyra-send-to-repl text)
    (message "Sent region to REPL")))

(defun lyra-eval-buffer ()
  "Evaluate the entire buffer in the REPL."
  (interactive)
  (lyra-eval-region (point-min) (point-max))
  (message "Sent buffer to REPL"))

(defun lyra-load-file ()
  "Load the current file in the REPL."
  (interactive)
  (when buffer-file-name
    (save-buffer)
    (lyra-send-to-repl (format "%%load \"%s\"" buffer-file-name))
    (message "Loading file in REPL: %s" buffer-file-name)))

;;; Completion

(defvar lyra-builtin-functions
  '("Sin" "Cos" "Tan" "ArcSin" "ArcCos" "ArcTan" "Sinh" "Cosh" "Tanh"
    "Log" "Ln" "Exp" "Sqrt" "Abs" "Floor" "Ceil" "Round" "Max" "Min"
    "Sum" "Product" "Mean" "Median" "Variance" "StandardDeviation"
    "Length" "First" "Last" "Rest" "Most" "Take" "Drop" "Join" "Sort"
    "Reverse" "Union" "Intersection" "Map" "Apply" "Select" "Cases"
    "Count" "Fold" "FoldLeft" "FoldRight" "Scan" "Transpose" "Range"
    "Table" "MatchQ" "Replace" "ReplaceAll" "Position" "Extract"
    "DeleteCases" "Part" "If" "Which" "Switch" "While" "For" "Do"
    "Module" "Block" "With" "Function" "Print" "Echo" "Input" "Get" "Put"
    "Import" "Export" "ReadString" "WriteString")
  "List of built-in Lyra functions.")

(defvar lyra-math-constants
  '("Pi" "E" "I" "Infinity" "ComplexInfinity" "GoldenRatio" "EulerGamma" "Catalan")
  "List of mathematical constants.")

(defvar lyra-greek-letters
  '("Alpha" "Beta" "Gamma" "Delta" "Epsilon" "Zeta" "Eta" "Theta"
    "Iota" "Kappa" "Lambda" "Mu" "Nu" "Xi" "Omicron" "Rho"
    "Sigma" "Tau" "Upsilon" "Phi" "Chi" "Psi" "Omega")
  "List of Greek letters.")

(defun lyra-completion-at-point ()
  "Function for `completion-at-point-functions'."
  (let ((bounds (bounds-of-thing-at-point 'symbol)))
    (when bounds
      (list (car bounds) (cdr bounds)
            (append lyra-builtin-functions
                    lyra-math-constants
                    lyra-greek-letters)
            :company-docsig #'lyra-get-function-doc))))

(defun lyra-get-function-doc (candidate)
  "Get documentation for CANDIDATE function."
  (cond
   ((string= candidate "Sin") "Sin[x] - Computes the sine of x")
   ((string= candidate "Cos") "Cos[x] - Computes the cosine of x")
   ((string= candidate "Log") "Log[x] - Natural logarithm of x")
   ((string= candidate "Map") "Map[f, list] - Apply function f to each element")
   ((string= candidate "Pi") "Pi - Mathematical constant π")
   (t (format "%s - Lyra function" candidate))))

;;; Export Functions

(defun lyra-export-to-jupyter ()
  "Export current buffer to Jupyter notebook."
  (interactive)
  (lyra-export-buffer "jupyter"))

(defun lyra-export-to-latex ()
  "Export current buffer to LaTeX."
  (interactive)
  (lyra-export-buffer "latex"))

(defun lyra-export-to-html ()
  "Export current buffer to HTML."
  (interactive)
  (lyra-export-buffer "html"))

(defun lyra-export-buffer (format)
  "Export current buffer to FORMAT."
  (when buffer-file-name
    (save-buffer)
    (let* ((base-name (file-name-sans-extension buffer-file-name))
           (output-file (concat base-name "." 
                               (cond ((string= format "jupyter") "ipynb")
                                     ((string= format "latex") "tex")
                                     ((string= format "html") "html"))))
           (cmd (format "%s --export %s --input \"%s\" --output \"%s\""
                       lyra-program-name format buffer-file-name output-file)))
      (shell-command cmd)
      (message "Exported to %s: %s" format output-file))))

;;; Help and Documentation

(defun lyra-help-at-point ()
  "Get help for symbol at point."
  (interactive)
  (let ((symbol (thing-at-point 'symbol t)))
    (when symbol
      (lyra-send-to-repl (format "%%help %s" symbol)))))

;;; Keymap

(defvar lyra-mode-map
  (let ((map (make-sparse-keymap)))
    ;; REPL commands
    (define-key map (kbd "C-c C-z") 'lyra-start-repl)
    (define-key map (kbd "C-c C-q") 'lyra-stop-repl)
    (define-key map (kbd "C-c C-r") 'lyra-restart-repl)
    
    ;; Evaluation commands
    (define-key map (kbd "C-c C-l") 'lyra-eval-line)
    (define-key map (kbd "C-c C-e") 'lyra-eval-region)
    (define-key map (kbd "C-c C-b") 'lyra-eval-buffer)
    (define-key map (kbd "C-c C-f") 'lyra-load-file)
    
    ;; Export commands
    (define-key map (kbd "C-c C-x j") 'lyra-export-to-jupyter)
    (define-key map (kbd "C-c C-x l") 'lyra-export-to-latex)
    (define-key map (kbd "C-c C-x h") 'lyra-export-to-html)
    
    ;; Help
    (define-key map (kbd "C-c C-h") 'lyra-help-at-point)
    
    map)
  "Keymap for Lyra mode.")

;;; Syntax Table

(defvar lyra-mode-syntax-table
  (let ((table (make-syntax-table)))
    ;; Comments
    (modify-syntax-entry ?/ ". 124b" table)
    (modify-syntax-entry ?* ". 23" table)
    (modify-syntax-entry ?\n "> b" table)
    
    ;; Strings
    (modify-syntax-entry ?\" "\"" table)
    (modify-syntax-entry ?\' "\"" table)
    
    ;; Operators
    (modify-syntax-entry ?+ "." table)
    (modify-syntax-entry ?- "." table)
    (modify-syntax-entry ?* "." table)
    (modify-syntax-entry ?/ "." table)
    (modify-syntax-entry ?^ "." table)
    (modify-syntax-entry ?= "." table)
    (modify-syntax-entry ?< "." table)
    (modify-syntax-entry ?> "." table)
    (modify-syntax-entry ?! "." table)
    (modify-syntax-entry ?& "." table)
    (modify-syntax-entry ?| "." table)
    
    ;; Brackets
    (modify-syntax-entry ?\[ "(]" table)
    (modify-syntax-entry ?\] ")[" table)
    (modify-syntax-entry ?\{ "(}" table)
    (modify-syntax-entry ?\} "){" table)
    (modify-syntax-entry ?\( "()" table)
    (modify-syntax-entry ?\) ")(" table)
    
    ;; Underscore is part of symbol
    (modify-syntax-entry ?_ "_" table)
    
    table)
  "Syntax table for Lyra mode.")

;;; Mode Definition

;;;###autoload
(define-derived-mode lyra-mode prog-mode "Lyra"
  "Major mode for editing Lyra language files.
\\{lyra-mode-map}"
  :syntax-table lyra-mode-syntax-table
  
  ;; Font lock
  (setq font-lock-defaults '(lyra-font-lock-keywords))
  
  ;; Indentation
  (setq-local indent-line-function 'lyra-indent-line)
  (setq-local tab-width lyra-indent-offset)
  (setq-local indent-tabs-mode nil)
  
  ;; Comments
  (setq-local comment-start "// ")
  (setq-local comment-end "")
  (setq-local comment-start-skip "//+\\s-*")
  
  ;; Completion
  (add-hook 'completion-at-point-functions 'lyra-completion-at-point nil t)
  
  ;; Auto-start REPL if configured
  (when lyra-auto-start-repl
    (lyra-start-repl)))

;;; File Association

;;;###autoload
(add-to-list 'auto-mode-alist '("\\.lyra\\'" . lyra-mode))

;;; Menu

(easy-menu-define lyra-mode-menu lyra-mode-map
  "Menu for Lyra mode."
  '("Lyra"
    ["Start REPL" lyra-start-repl]
    ["Stop REPL" lyra-stop-repl]
    ["Restart REPL" lyra-restart-repl]
    "---"
    ["Evaluate Line" lyra-eval-line]
    ["Evaluate Region" lyra-eval-region]
    ["Evaluate Buffer" lyra-eval-buffer]
    ["Load File" lyra-load-file]
    "---"
    ["Export to Jupyter" lyra-export-to-jupyter]
    ["Export to LaTeX" lyra-export-to-latex]
    ["Export to HTML" lyra-export-to-html]
    "---"
    ["Help at Point" lyra-help-at-point]))

(provide 'lyra-mode)

;;; lyra-mode.el ends here