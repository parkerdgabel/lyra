//! Multi-line input support for the REPL
//! 
//! Handles incomplete expressions that span multiple lines, providing
//! intelligent continuation prompts and bracket matching.

use std::collections::VecDeque;

/// Tracks bracket nesting for incomplete expression detection
#[derive(Debug, Clone, Default)]
pub struct BracketTracker {
    /// Stack of open brackets: (, [, {
    open_brackets: VecDeque<char>,
    /// Whether we're inside a string literal
    in_string: bool,
    /// Character that started current string (for escape handling)
    string_delimiter: Option<char>,
}

impl BracketTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Process a character and update bracket state
    pub fn process_char(&mut self, ch: char, prev_char: Option<char>) {
        // Handle string literals first
        if self.in_string {
            if Some(ch) == self.string_delimiter && prev_char != Some('\\') {
                self.in_string = false;
                self.string_delimiter = None;
            }
            return;
        }

        match ch {
            '"' | '\'' => {
                self.in_string = true;
                self.string_delimiter = Some(ch);
            }
            '(' | '[' | '{' => {
                self.open_brackets.push_back(ch);
            }
            ')' => {
                if self.open_brackets.back() == Some(&'(') {
                    self.open_brackets.pop_back();
                }
                // Note: We don't error on mismatched brackets here
                // That's handled by the parser later
            }
            ']' => {
                if self.open_brackets.back() == Some(&'[') {
                    self.open_brackets.pop_back();
                }
            }
            '}' => {
                if self.open_brackets.back() == Some(&'{') {
                    self.open_brackets.pop_back();
                }
            }
            _ => {}
        }
    }

    /// Check if all brackets are balanced and no string is open
    pub fn is_complete(&self) -> bool {
        self.open_brackets.is_empty() && !self.in_string
    }

    /// Get the expected closing bracket for current nesting
    pub fn expected_closing(&self) -> Option<char> {
        self.open_brackets.back().map(|&open| match open {
            '(' => ')',
            '[' => ']',
            '{' => '}',
            _ => open, // shouldn't happen
        })
    }

    /// Reset bracket tracking state
    pub fn reset(&mut self) {
        self.open_brackets.clear();
        self.in_string = false;
        self.string_delimiter = None;
    }
}

/// Buffer for accumulating multi-line input
#[derive(Debug, Clone)]
pub struct MultilineBuffer {
    /// Lines of input so far
    lines: Vec<String>,
    /// Bracket tracking state
    tracker: BracketTracker,
}

impl MultilineBuffer {
    pub fn new() -> Self {
        Self {
            lines: Vec::new(),
            tracker: BracketTracker::new(),
        }
    }

    /// Add a line to the buffer and check if expression is complete
    pub fn add_line(&mut self, line: &str) -> bool {
        // Process each character to update bracket state
        let mut prev_char = None;
        for ch in line.chars() {
            self.tracker.process_char(ch, prev_char);
            prev_char = Some(ch);
        }

        self.lines.push(line.to_string());

        // Check if expression is complete
        self.is_complete()
    }

    /// Check if the current buffer represents a complete expression
    pub fn is_complete(&self) -> bool {
        if self.lines.is_empty() {
            return true;
        }

        // Check bracket balance
        if !self.tracker.is_complete() {
            return false;
        }

        // Additional heuristics for Lyra syntax
        let last_line = self.lines.last().unwrap().trim();
        
        // Lines ending with operators suggest continuation
        if last_line.ends_with(['+', '-', '*', '/', '^', '=', ',', ';']) {
            return false;
        }

        // Lines ending with '->' or ':>' (rule operators) suggest continuation
        if last_line.ends_with("->") || last_line.ends_with(":>") {
            return false;
        }

        // Check for function definitions that might continue
        if last_line.contains("]:=") || last_line.contains("]=") {
            // Function definition might continue on next line
            return !last_line.trim_end().ends_with([',', '+', '-', '*', '/', '^']);
        }

        true
    }

    /// Get the complete input as a single string
    pub fn get_complete_input(&self) -> String {
        self.lines.join("\n")
    }

    /// Get the current number of lines
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.lines.clear();
        self.tracker.reset();
    }

    /// Get suggestion for what's expected to complete the expression
    pub fn completion_hint(&self) -> Option<String> {
        if let Some(expected) = self.tracker.expected_closing() {
            Some(format!("Expected: '{}'", expected))
        } else if self.tracker.in_string {
            Some("Unclosed string literal".to_string())
        } else {
            None
        }
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    /// Get the last line for editing
    pub fn last_line(&self) -> Option<&String> {
        self.lines.last()
    }
}

impl Default for MultilineBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_complete_expression() {
        let mut buffer = MultilineBuffer::new();
        assert!(buffer.add_line("x = 5"));
        assert_eq!(buffer.get_complete_input(), "x = 5");
    }

    #[test]
    fn test_unclosed_brackets() {
        let mut buffer = MultilineBuffer::new();
        
        // Unclosed parentheses
        assert!(!buffer.add_line("Sin[Pi/2"));
        assert!(buffer.add_line("]"));
        assert_eq!(buffer.get_complete_input(), "Sin[Pi/2\n]");
    }

    #[test]
    fn test_nested_brackets() {
        let mut buffer = MultilineBuffer::new();
        
        assert!(!buffer.add_line("Map[Function[x,"));
        assert!(!buffer.add_line("  x^2 + 1"));
        assert!(buffer.add_line("], {1, 2, 3}]"));
        
        let expected = "Map[Function[x,\n  x^2 + 1\n], {1, 2, 3}]";
        assert_eq!(buffer.get_complete_input(), expected);
    }

    #[test]
    fn test_string_literals() {
        let mut buffer = MultilineBuffer::new();
        
        // Unclosed string
        assert!(!buffer.add_line("message = \"Hello"));
        assert!(buffer.add_line("World\""));
        
        assert_eq!(buffer.get_complete_input(), "message = \"Hello\nWorld\"");
    }

    #[test]
    fn test_operator_continuation() {
        let mut buffer = MultilineBuffer::new();
        
        // Expression ending with operator
        assert!(!buffer.add_line("result = x +"));
        assert!(buffer.add_line("y * 2"));
        
        assert_eq!(buffer.get_complete_input(), "result = x +\ny * 2");
    }

    #[test]
    fn test_rule_continuation() {
        let mut buffer = MultilineBuffer::new();
        
        // Rule definition
        assert!(!buffer.add_line("x_ ->"));
        assert!(buffer.add_line("x^2 + 1"));
        
        assert_eq!(buffer.get_complete_input(), "x_ ->\nx^2 + 1");
    }

    #[test]
    fn test_clear_buffer() {
        let mut buffer = MultilineBuffer::new();
        
        buffer.add_line("incomplete = {1, 2,");
        assert!(!buffer.is_complete());
        
        buffer.clear();
        assert!(buffer.is_empty());
        assert!(buffer.is_complete());
    }

    #[test]
    fn test_completion_hints() {
        let mut buffer = MultilineBuffer::new();
        
        buffer.add_line("list = {1, 2");
        
        if let Some(hint) = buffer.completion_hint() {
            assert!(hint.contains("}"));
        }
    }

    #[test]
    fn test_bracket_tracker() {
        let mut tracker = BracketTracker::new();
        
        // Process "Sin[Pi/2"
        for ch in "Sin[Pi/2".chars() {
            tracker.process_char(ch, None);
        }
        assert!(!tracker.is_complete());
        assert_eq!(tracker.expected_closing(), Some(']'));
        
        // Add closing bracket
        tracker.process_char(']', None);
        assert!(tracker.is_complete());
    }
}