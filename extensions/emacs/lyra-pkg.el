;;; lyra-pkg.el --- Lyra language package definition  -*- no-byte-compile: t; lexical-binding: t; -*-

;; Copyright (C) 2024 Lyra Language Team

;; Author: Lyra Language Team
;; URL: https://github.com/lyra-lang/lyra
;; Version: 1.0.0
;; Package-Requires: ((emacs "24.3") (org "8.0"))
;; Keywords: languages, lyra, mathematical, symbolic

;; This file is not part of GNU Emacs.

;;; Commentary:

;; This package provides comprehensive Emacs support for the Lyra programming
;; language, including:
;;
;; - lyra-mode: Major mode with syntax highlighting and editing features
;; - ob-lyra: Org-babel integration for literate programming
;; - REPL integration for interactive development
;; - Export capabilities to multiple formats
;; - Code completion and documentation

;;; Code:

;; Package definition
(define-package "lyra-mode" "1.0.0"
  "Major mode for Lyra programming language"
  '((emacs "24.3")
    (org "8.0"))
  :url "https://github.com/lyra-lang/lyra"
  :keywords '("languages" "lyra" "mathematical" "symbolic"))

;;; lyra-pkg.el ends here