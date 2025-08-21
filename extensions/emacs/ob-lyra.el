;;; ob-lyra.el --- Babel Functions for Lyra  -*- lexical-binding: t; -*-

;; Copyright (C) 2024 Lyra Language Team

;; Author: Lyra Language Team
;; URL: https://github.com/lyra-lang/lyra
;; Version: 1.0.0
;; Package-Requires: ((emacs "24.3") (org "8.0"))
;; Keywords: literate programming, reproducible research, lyra

;; This file is not part of GNU Emacs.

;; This program is free software; you can redistribute it and/or modify
;; it under the terms of the GNU General Public License as published by
;; the Free Software Foundation, either version 3 of the License, or
;; (at your option) any later version.

;;; Commentary:

;; Org-babel support for Lyra language.
;; This enables execution of Lyra code blocks within org-mode documents.
;;
;; Usage:
;; #+BEGIN_SRC lyra
;; x = Sin[Pi/2] + Cos[0]
;; Print[x]
;; #+END_SRC

;;; Code:

(require 'ob)
(require 'ob-eval)

;; Define Lyra language for org-babel
(defvar org-babel-default-header-args:lyra '()
  "Default header arguments for Lyra code blocks.")

(defcustom org-babel-lyra-command "lyra"
  "Command to execute Lyra code."
  :group 'org-babel
  :type 'string)

(defcustom org-babel-lyra-command-args '()
  "Arguments to pass to the Lyra command."
  :group 'org-babel
  :type '(repeat string))

(defun org-babel-execute:lyra (body params)
  "Execute a block of Lyra code with org-babel.
BODY is the code to execute.
PARAMS is a plist of parameters."
  (let* ((session (cdr (assq :session params)))
         (result-type (cdr (assq :result-type params)))
         (result-params (cdr (assq :result-params params)))
         (graphics-file (cdr (assq :file params)))
         (dir (cdr (assq :dir params)))
         (default-directory (or dir default-directory))
         (full-body (org-babel-expand-body:lyra body params))
         (cmd (mapconcat #'shell-quote-argument
                        (append (list org-babel-lyra-command)
                               org-babel-lyra-command-args)
                        " ")))
    
    (if (string= session "none")
        ;; Non-session execution
        (org-babel-eval cmd full-body)
      ;; Session execution
      (org-babel-lyra-evaluate-session session full-body result-type result-params))))

(defun org-babel-expand-body:lyra (body params)
  "Expand BODY according to PARAMS, return the expanded body."
  (let ((vars (org-babel--get-vars params)))
    (concat
     ;; Variable assignments
     (mapconcat
      (lambda (pair)
        (format "%s = %s;" 
                (car pair)
                (org-babel-lyra-var-to-lyra (cdr pair))))
      vars "\n")
     (when vars "\n")
     ;; Body
     body)))

(defun org-babel-lyra-var-to-lyra (var)
  "Convert an Emacs Lisp value to a Lyra representation."
  (cond
   ((stringp var) (format "\"%s\"" var))
   ((numberp var) (number-to-string var))
   ((listp var) (format "{%s}" (mapconcat #'org-babel-lyra-var-to-lyra var ", ")))
   (t (format "%s" var))))

(defvar org-babel-lyra-buffers nil
  "Association list of session names and their buffers.")

(defun org-babel-lyra-initiate-session (&optional session params)
  "Initiate a Lyra session.
If SESSION is nil, create a new session.
PARAMS are additional parameters."
  (unless (string= session "none")
    (let* ((session (or session "*Lyra*"))
           (buffer (cdr (assoc session org-babel-lyra-buffers))))
      (unless (and buffer (buffer-live-p buffer))
        (setq buffer (get-buffer-create session))
        (with-current-buffer buffer
          (comint-mode)
          (let ((process (apply #'start-process "lyra" buffer
                               org-babel-lyra-command
                               org-babel-lyra-command-args)))
            (set-process-query-on-exit-flag process nil)
            (comint-send-string process "// Org-babel session started\n")))
        (push (cons session buffer) org-babel-lyra-buffers))
      buffer)))

(defun org-babel-lyra-evaluate-session (session body &optional result-type result-params)
  "Evaluate BODY in SESSION.
RESULT-TYPE and RESULT-PARAMS control output format."
  (let ((buffer (org-babel-lyra-initiate-session session)))
    (with-current-buffer buffer
      (let ((start-point (point-max)))
        ;; Send code to session
        (comint-send-string (get-buffer-process buffer) 
                           (concat body "\n"))
        
        ;; Wait for completion and capture output
        (while (accept-process-output (get-buffer-process buffer) 0.1))
        (sleep-for 0.1) ; Brief pause to ensure output is complete
        
        ;; Extract output
        (let ((output (buffer-substring-no-properties start-point (point-max))))
          ;; Clean up output
          (string-trim
           (replace-regexp-in-string
            "^.*> " "" ; Remove prompts
            (replace-regexp-in-string
             "\r" "" ; Remove carriage returns
             output))))))))

(defun org-babel-prep-session:lyra (session params)
  "Prepare SESSION according to PARAMS."
  (let ((buffer (org-babel-lyra-initiate-session session)))
    (org-babel-comint-in-buffer buffer
      (goto-char (point-max)))
    buffer))

(defun org-babel-lyra-strip-output (output)
  "Strip extraneous output from Lyra OUTPUT."
  (string-trim
   (replace-regexp-in-string
    "^Lyra>\\|^In\\[[0-9]+\\]:=\\|^Out\\[[0-9]+\\]=" ""
    output)))

;; Graphics support
(defun org-babel-lyra-graphical-output-file (params)
  "Return the name of the Lyra graphics output file."
  (and (member "graphics" (cdr (assq :result-params params)))
       (cdr (assq :file params))))

;; Add lyra to supported languages
(add-to-list 'org-src-lang-modes '("lyra" . lyra))

;; Provide file extension mapping
(setq org-babel-tangle-lang-exts
      (cons '("lyra" . "lyra") org-babel-tangle-lang-exts))

;; Support for various result types
(defun org-babel-lyra-table-or-string (results)
  "Convert RESULTS to appropriate format for org-mode."
  (let ((tmp-file (org-babel-temp-file "lyra-")))
    (with-temp-file tmp-file (insert results))
    (org-babel-import-elisp-from-file tmp-file '(16))))

;; Header argument documentation
(defvar org-babel-header-args:lyra
  '((session . :any)
    (file . :any)
    (dir . :any)
    (exports . ((code results both none)))
    (results . ((file list vector table scalar verbatim)
                (raw html latex org code pp drawer)
                (replace silent none append prepend)
                (output value))))
  "Lyra-specific header arguments.")

;; Add example usage and help
(defun org-babel-lyra-help ()
  "Display help for org-babel Lyra integration."
  (interactive)
  (with-help-window "*Org-Babel Lyra Help*"
    (princ "Org-Babel Lyra Integration\n")
    (princ "===========================\n\n")
    (princ "Basic usage:\n")
    (princ "#+BEGIN_SRC lyra\n")
    (princ "x = 2 + 3\n")
    (princ "Print[x]\n")
    (princ "#+END_SRC\n\n")
    (princ "With variables:\n")
    (princ "#+BEGIN_SRC lyra :var data='(1 2 3 4 5)\n")
    (princ "mean = Mean[data]\n")
    (princ "Print[mean]\n")
    (princ "#+END_SRC\n\n")
    (princ "Session-based execution:\n")
    (princ "#+BEGIN_SRC lyra :session *lyra-session*\n")
    (princ "x = 10\n")
    (princ "#+END_SRC\n\n")
    (princ "#+BEGIN_SRC lyra :session *lyra-session*\n")
    (princ "y = x * 2\n")
    (princ "Print[y]\n")
    (princ "#+END_SRC\n\n")
    (princ "Export to file:\n")
    (princ "#+BEGIN_SRC lyra :file output.txt\n")
    (princ "result = Solve[x^2 - 4 == 0, x]\n")
    (princ "Export[result, \"output.txt\"]\n")
    (princ "#+END_SRC\n\n")
    (princ "Header arguments:\n")
    (princ ":session - Session name for persistent execution\n")
    (princ ":file    - Output file for results\n")
    (princ ":dir     - Working directory\n")
    (princ ":exports - What to export (code, results, both, none)\n")
    (princ ":results - How to handle results (value, output, etc.)\n")))

;; Integration with org-mode export
(defun org-babel-lyra-process-result (result params)
  "Process RESULT according to PARAMS for export."
  (let ((file (cdr (assq :file params))))
    (if file
        (if (file-exists-p file)
            file
          result)
      result)))

(provide 'ob-lyra)

;;; ob-lyra.el ends here