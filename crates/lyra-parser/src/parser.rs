use lyra_core::value::Value;
use lyra_core::error::{Result, LyraError};

pub type ParseResult<T> = Result<T>;

pub struct Parser { src: Vec<char>, pos: usize }

impl Parser {
    pub fn from_source(source: &str) -> Self { Self { src: source.chars().collect(), pos: 0 } }

    fn peekc(&self) -> Option<char> { self.src.get(self.pos).cloned() }
    fn nextc(&mut self) -> Option<char> { let c = self.src.get(self.pos).cloned(); if c.is_some() { self.pos += 1; } c }
    fn starts_with(&self, s: &str) -> bool {
        self.src[self.pos..].iter().zip(s.chars()).all(|(a,b)| *a==b) && self.pos + s.len() <= self.src.len()
    }
    fn eat_str(&mut self, s: &str) -> bool { if self.starts_with(s) { self.pos += s.len(); true } else { false } }
    fn skip_ws(&mut self) {
        loop {
            while matches!(self.peekc(), Some(c) if c.is_whitespace()) { self.pos += 1; }
            if self.starts_with("(*") {
                self.pos += 2;
                while self.pos+1 < self.src.len() && !self.starts_with("*)") { self.pos += 1; }
                if self.starts_with("*)") { self.pos += 2; }
                continue;
            }
            break;
        }
    }

    pub fn parse_all(&mut self) -> ParseResult<Vec<Value>> {
        let mut out = Vec::new();
        while self.pos < self.src.len() {
            self.skip_ws();
            if self.pos >= self.src.len() { break; }
            out.push(self.expr()?);
            self.skip_ws();
            if self.peekc()==Some(';') { self.nextc(); }
        }
        Ok(out)
    }

    fn expr(&mut self) -> ParseResult<Value> {
        if let Some(mut lambda) = self.try_parse_lambda()? {
            // Allow immediate call: (x)=>body [args]
            self.skip_ws();
            if self.peekc()==Some('[') {
                self.nextc();
                let mut args = Vec::new();
                self.skip_ws();
                if self.peekc()!=Some(']') {
                    loop {
                        args.push(self.expr()?);
                        self.skip_ws();
                        if self.peekc()==Some(',') { self.nextc(); self.skip_ws(); continue; }
                        break;
                    }
                }
                if self.nextc()!=Some(']') { return Err(LyraError::Parse("expected ']'".into())); }
                lambda = Value::expr(lambda, args);
            }
            return Ok(lambda);
        }
        let mut v = self.parse_rule()?;
        self.skip_ws();
        if self.peekc()==Some('&') {
            self.nextc();
            v = Value::pure_function(None, v);
        }
        Ok(v)
    }

    fn parse_rule(&mut self) -> ParseResult<Value> {
        let mut lhs = self.parse_logical()?;
        self.skip_ws();
        if self.starts_with("/;") {
            self.pos += 2; self.skip_ws();
            let cond = self.parse_logical()?;
            lhs = Value::expr(Value::Symbol("Condition".into()), vec![lhs, cond]);
            self.skip_ws();
        }
        // Alternatives with '|'
        let mut alt_args: Vec<Value> = vec![lhs];
        loop {
            self.skip_ws();
            // avoid '|>' and '||', and only treat '|' as Alternative when the next
            // non-whitespace token can start an expression
            if self.starts_with("|>") || self.starts_with("||") { break; }
            if self.peekc()==Some('|') {
                // Lookahead: only consume as Alternative if a valid expr follows
                let save = self.pos;
                self.nextc();
                self.skip_ws();
                let c = self.peekc();
                let expr_start = match c {
                    Some('(') | Some('{') | Some('"') | Some('_') | Some('#') => true,
                    Some('<') => self.starts_with("<|"),
                    Some(ch) if ch.is_ascii_digit() => true,
                    Some(ch) if is_ident_start(ch) => true,
                    _ => false,
                };
                if !expr_start { self.pos = save; break; }
                let rhs_alt = self.parse_logical()?;
                alt_args.push(rhs_alt);
                continue;
            }
            break;
        }
        let lhs_final = if alt_args.len()>1 { Value::expr(Value::Symbol("Alternative".into()), alt_args) } else { alt_args.pop().unwrap() };
        self.skip_ws();
        if self.starts_with("->") { self.pos += 2; self.skip_ws(); let rhs = self.parse_logical()?; return Ok(Value::expr(Value::Symbol("Rule".into()), vec![lhs_final, rhs])); }
        Ok(lhs_final)
    }

    fn parse_logical(&mut self) -> ParseResult<Value> {
        let mut lhs = self.parse_comparison()?;
        loop {
            self.skip_ws();
            if self.starts_with("||") { self.pos += 2; let rhs = self.parse_comparison()?; lhs = Value::expr(Value::Symbol("Or".into()), vec![lhs, rhs]); continue; }
            break;
        }
        Ok(lhs)
    }

    fn parse_comparison(&mut self) -> ParseResult<Value> {
        let mut lhs = self.parse_additive()?;
        loop {
            self.skip_ws();
            if self.starts_with("&&") { self.pos += 2; let rhs = self.parse_additive()?; lhs = Value::expr(Value::Symbol("And".into()), vec![lhs, rhs]); continue; }
            if self.starts_with("==") { self.pos += 2; let rhs = self.parse_additive()?; lhs = Value::expr(Value::Symbol("Equal".into()), vec![lhs, rhs]); continue; }
            if self.starts_with("!=") { self.pos += 2; let rhs = self.parse_additive()?; let eq = Value::expr(Value::Symbol("Equal".into()), vec![lhs, rhs]); lhs = Value::expr(Value::Symbol("Not".into()), vec![eq]); continue; }
            if self.starts_with("<=") { self.pos += 2; let rhs = self.parse_additive()?; lhs = Value::expr(Value::Symbol("LessEqual".into()), vec![lhs, rhs]); continue; }
            if self.starts_with(">=") { self.pos += 2; let rhs = self.parse_additive()?; lhs = Value::expr(Value::Symbol("GreaterEqual".into()), vec![lhs, rhs]); continue; }
            match self.peekc() {
                Some('<') => { self.nextc(); let rhs = self.parse_additive()?; lhs = Value::expr(Value::Symbol("Less".into()), vec![lhs, rhs]); }
                Some('>') => { self.nextc(); let rhs = self.parse_additive()?; lhs = Value::expr(Value::Symbol("Greater".into()), vec![lhs, rhs]); }
                _ => break,
            }
        }
        Ok(lhs)
    }

    // Operator precedence: + - | * / | ^ (right-assoc) with unary +/-
    fn parse_additive(&mut self) -> ParseResult<Value> {
        let mut lhs = self.parse_multiplicative()?;
        loop {
            self.skip_ws();
            match self.peekc() {
                Some('+') => { self.nextc(); let rhs = self.parse_multiplicative()?; lhs = Value::expr(Value::Symbol("Plus".into()), vec![lhs, rhs]); }
                Some('-') => {
                    // Do not consume '-' if part of '->' (rule)
                    let mut i = self.pos + 1;
                    while i < self.src.len() && self.src[i].is_whitespace() { i += 1; }
                    if i < self.src.len() && self.src[i] == '>' { break; }
                    self.nextc(); let rhs = self.parse_multiplicative()?; lhs = Value::expr(Value::Symbol("Minus".into()), vec![lhs, rhs]);
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn parse_multiplicative(&mut self) -> ParseResult<Value> {
        let mut lhs = self.parse_power()?;
        loop {
            self.skip_ws();
            match self.peekc() {
                Some('*') => { self.nextc(); let rhs = self.parse_power()?; lhs = Value::expr(Value::Symbol("Times".into()), vec![lhs, rhs]); }
                Some('/') => { self.nextc(); let rhs = self.parse_power()?; lhs = Value::expr(Value::Symbol("Divide".into()), vec![lhs, rhs]); }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn parse_power(&mut self) -> ParseResult<Value> {
        self.skip_ws();
        // unary +/- and !
        if self.peekc()==Some('!') { self.nextc(); let v = self.parse_power()?; return Ok(Value::expr(Value::Symbol("Not".into()), vec![v])); }
        if self.peekc()==Some('-') { self.nextc(); let v = self.parse_power()?; return Ok(Value::expr(Value::Symbol("Minus".into()), vec![v])); }
        if self.peekc()==Some('+') { self.nextc(); return self.parse_power(); }
        let lhs = self.parse_postfix()?;
        self.skip_ws();
        if self.peekc()==Some('^') { self.nextc(); let rhs = self.parse_power()?; return Ok(Value::expr(Value::Symbol("Power".into()), vec![lhs, rhs])); }
        Ok(lhs)
    }

    fn parse_postfix(&mut self) -> ParseResult<Value> {
        self.parse_call_or_atom()
    }

    fn parse_call_or_atom(&mut self) -> ParseResult<Value> {
        // Assignment: symbol = expr
        self.skip_ws();
        if is_ident_start(self.peekc().unwrap_or('\0')) {
            let save = self.pos;
            if let Value::Symbol(name) = self.parse_symbol()? {
                self.skip_ws();
                if self.peekc()==Some('=') { self.nextc(); self.skip_ws(); let rhs = self.expr()?; return Ok(Value::expr(Value::Symbol("Set".into()), vec![Value::Symbol(name), rhs])); }
            }
            // rewind if not assignment
            self.pos = save;
        }

        let mut base = self.atom()?;
        loop {
            self.skip_ws();
            // Head Alternatives: f | g [args] => Alternative[f, g][args]
            if !(self.starts_with("|>") || self.starts_with("||")) && self.peekc()==Some('|') {
                // Only treat as head-alternative if we have not yet consumed args
                // Combine consecutive alternatives: f|g|h
                let mut alts: Vec<Value> = vec![base];
                while !(self.starts_with("|>") || self.starts_with("||")) && self.peekc()==Some('|') {
                    self.nextc(); self.skip_ws();
                    let _save = self.pos;
                    let rhs_head = self.atom()?;
                    alts.push(rhs_head);
                    self.skip_ws();
                    // Stop if next token starts args '[' or end
                    // Continue loop to collect multiple alternatives
                }
                base = Value::expr(Value::Symbol("Alternative".into()), alts);
                continue;
            }
            // dot-call chaining: obj.method[args] or obj.method
            if self.peekc()==Some('.') {
                self.nextc(); self.skip_ws();
                let method = match self.parse_symbol()? { Value::Symbol(s)=>s, _=>return Err(LyraError::Parse("expected method name".into())) };
                self.skip_ws();
                let mut args = Vec::new();
                if self.peekc()==Some('[') {
                    self.nextc(); self.skip_ws();
                    if self.peekc()!=Some(']') {
                        loop {
                            args.push(self.expr()?);
                            self.skip_ws();
                            if self.peekc()==Some(',') { self.nextc(); self.skip_ws(); continue; }
                            break;
                        }
                    }
                    if self.nextc()!=Some(']') { return Err(LyraError::Parse("expected ']'".into())); }
                }
                if args.is_empty() {
                    // Property access → Lookup[base, "method"]
                    base = Value::expr(Value::Symbol("Lookup".into()), vec![base, Value::String(method)]);
                } else {
                    let mut all_args = Vec::with_capacity(1+args.len());
                    all_args.push(base);
                    all_args.extend(args);
                    base = Value::expr(Value::Symbol(method), all_args);
                }
                continue;
            }
            // normal call: head[args]
            if self.peekc()==Some('[') {
                self.nextc();
                let mut args = Vec::new();
                self.skip_ws();
                if self.peekc()!=Some(']') {
                    loop {
                        args.push(self.expr()?);
                        self.skip_ws();
                        if self.peekc()==Some(',') { self.nextc(); self.skip_ws(); continue; }
                        break;
                    }
                }
                if self.nextc()!=Some(']') { return Err(LyraError::Parse("expected ']'".into())); }
                base = Value::expr(base, args);
                continue;
            }
            // pipelines: a |> f[args] or a |> f
            if self.starts_with("|>") {
                self.pos += 2; self.skip_ws();
                // parse RHS as call or symbol
                let rhs = self.parse_call_or_atom()?; // recursive use to get f[...] or symbol
                let call = match rhs {
                    Value::Expr { head, mut args } => { args.insert(0, base); Value::Expr { head, args } }
                    Value::Symbol(s) => Value::expr(Value::Symbol(s), vec![base]),
                    Value::PureFunction { .. } => Value::expr(rhs, vec![base]),
                    other => Value::expr(other, vec![base]),
                };
                base = call;
                continue;
            }
            break;
        }
        Ok(base)
    }

    fn atom(&mut self) -> ParseResult<Value> {
        self.skip_ws();
        match self.peekc() {
            Some('{') => { self.nextc(); self.parse_list() }
            Some('<') if self.starts_with("<|") => { self.eat_str("<|"); self.parse_assoc() }
            Some('(') => { self.nextc(); let v = self.expr()?; self.skip_ws(); if self.nextc()!=Some(')') { return Err(LyraError::Parse("expected ')'".into())); } Ok(v) }
            Some('_') => self.parse_blank(),
            Some('#') => self.parse_slot(),
            Some('"') => self.parse_string(),
            Some(c) if c.is_ascii_digit() => self.parse_number(),
            Some(c) if is_ident_start(c) => self.parse_symbol_or_named_blank(),
            other => Err(LyraError::Parse(format!("unexpected char: {:?}", other))),
        }
    }

    fn parse_blank(&mut self) -> ParseResult<Value> {
        // _, __, ___ with optional Type, then optional ?pred
        if self.nextc()!=Some('_') { return Err(LyraError::Parse("expected '_'".into())); }
        let kind = if self.peekc()==Some('_') {
            self.nextc();
            if self.peekc()==Some('_') { self.nextc(); 3 } else { 2 }
        } else { 1 };
        self.skip_ws();
        let base_head = match kind { 1=>"Blank", 2=>"BlankSequence", _=>"BlankNullSequence" };
        let mut pat = if matches!(self.peekc(), Some(c) if is_ident_start(c)) {
            let sym = self.parse_symbol()?;
            Value::expr(Value::Symbol(base_head.into()), vec![sym])
        } else { Value::expr(Value::Symbol(base_head.into()), vec![]) };
        self.skip_ws();
        if self.peekc()==Some('?') { self.nextc(); self.skip_ws(); let pred = self.parse_call_or_atom()?; pat = Value::expr(Value::Symbol("PatternTest".into()), vec![pat, pred]); }
        Ok(pat)
    }

    fn parse_symbol_or_named_blank(&mut self) -> ParseResult<Value> {
        // symbol or named blank: name_ or name_Type, with optional ?pred
        let name_sym = self.parse_symbol()?;
        if let Value::Symbol(name) = &name_sym {
            self.skip_ws();
            if self.peekc()==Some('_') { // named blank or named sequence
                self.nextc();
                let kind = if self.peekc()==Some('_') { self.nextc(); if self.peekc()==Some('_') { self.nextc(); 3 } else { 2 } } else { 1 };
                self.skip_ws();
                let head_name = match kind { 1=>"NamedBlank", 2=>"NamedBlankSequence", _=>"NamedBlankNullSequence" };
                let mut pat = if matches!(self.peekc(), Some(c) if is_ident_start(c)) {
                    let type_sym = self.parse_symbol()?;
                    Value::expr(Value::Symbol(head_name.into()), vec![Value::Symbol(name.clone()), type_sym])
                } else {
                    Value::expr(Value::Symbol(head_name.into()), vec![Value::Symbol(name.clone())])
                };
                self.skip_ws();
                if self.peekc()==Some('?') { self.nextc(); self.skip_ws(); let pred = self.parse_call_or_atom()?; pat = Value::expr(Value::Symbol("PatternTest".into()), vec![pat, pred]); }
                // alternatives support on named blank
                let mut alt_args: Vec<Value> = vec![pat];
                loop {
                    self.skip_ws();
                    if self.starts_with("|>") || self.starts_with("||") { break; }
                    if self.peekc()==Some('|') { self.nextc(); self.skip_ws(); let rhs_alt = self.parse_logical()?; alt_args.push(rhs_alt); continue; }
                    break;
                }
                if alt_args.len()>1 { return Ok(Value::expr(Value::Symbol("Alternative".into()), alt_args)); }
                else { return Ok(alt_args.pop().unwrap()); }
            }
        }
        Ok(name_sym)
    }

    fn parse_list(&mut self) -> ParseResult<Value> {
        let mut items = Vec::new();
        self.skip_ws();
        if self.peekc()!=Some('}') {
            loop {
                items.push(self.expr()?);
                self.skip_ws();
                if self.peekc()==Some(',') { self.nextc(); self.skip_ws(); continue; }
                break;
            }
        }
        if self.nextc()!=Some('}') { return Err(LyraError::Parse("expected '}'".into())); }
        Ok(Value::List(items))
    }

    fn parse_assoc(&mut self) -> ParseResult<Value> {
        let mut pairs: Vec<(String, Value)> = Vec::new();
        self.skip_ws();
        // Handle empty association
        if self.starts_with("|>") { self.pos += 2; return Ok(Value::assoc(Vec::<(String, Value)>::new())); }
        loop {
            // Key: string or symbol
            let key = match self.peekc() {
                Some('"') => match self.parse_string()? { Value::String(s) => s, _ => unreachable!() },
                Some(c) if is_ident_start(c) => match self.parse_symbol()? { Value::Symbol(s) => s, _ => unreachable!() },
                _ => return Err(LyraError::Parse("expected association key".into())),
            };
            self.skip_ws();
            if !self.eat_str("->") { return Err(LyraError::Parse("expected '->'".into())); }
            self.skip_ws();
            // Parse value up to next ',' or '|>' at top level (respect nesting)
            let (val, ended_assoc) = self.parse_value_until_assoc_delim()?;
            pairs.push((key, val));
            if ended_assoc { return Ok(Value::assoc(pairs)); }
            // Else a comma was consumed; continue to next pair
            self.skip_ws();
        }
    }

    // Parse a value expression within an Association, stopping at the next top-level
    // comma or at the closing '|>' of the current association. Returns (value, ended_assoc)
    fn parse_value_until_assoc_delim(&mut self) -> ParseResult<(Value, bool)> {
        let start_pos = self.pos;
        let mut i = self.pos;
        let len = self.src.len();
        let mut depth_paren = 0i32;
        let mut depth_brack = 0i32;
        let mut depth_brace = 0i32;
        let mut depth_assoc = 0i32; // nested associations inside value
        let mut in_string = false;
        while i < len {
            let c = self.src[i];
            if in_string {
                if c == '\\' { // escape next char if any
                    i += 1; if i < len { i += 1; } else { break; }
                    continue;
                } else if c == '"' { in_string = false; i += 1; continue; }
                i += 1; continue;
            }
            // Comments: (* ... *)
            if c == '(' && i + 1 < len && self.src[i+1] == '*' { i += 2; while i + 1 < len && !(self.src[i] == '*' && self.src[i+1] == ')') { i += 1; } if i + 1 < len { i += 2; } continue; }
            // Association open/close tokens
            if c == '<' && i + 1 < len && self.src[i+1] == '|' { depth_assoc += 1; i += 2; continue; }
            if c == '|' && i + 1 < len && self.src[i+1] == '>' {
                if depth_paren == 0 && depth_brack == 0 && depth_brace == 0 && depth_assoc == 0 {
                    // reached end of current association value
                    break;
                } else {
                    depth_assoc = (depth_assoc - 1).max(0);
                    i += 2; continue;
                }
            }
            match c {
                '"' => { in_string = true; i += 1; }
                '(' => { depth_paren += 1; i += 1; }
                ')' => { depth_paren -= 1; i += 1; }
                '[' => { depth_brack += 1; i += 1; }
                ']' => { depth_brack -= 1; i += 1; }
                '{' => { depth_brace += 1; i += 1; }
                '}' => { depth_brace -= 1; i += 1; }
                ',' => {
                    if depth_paren == 0 && depth_brack == 0 && depth_brace == 0 && depth_assoc == 0 {
                        // Stop at comma separating pairs
                        break;
                    } else { i += 1; }
                }
                _ => { i += 1; }
            }
        }
        // Now slice out [start_pos, i) and parse as a standalone expression
        let slice: String = self.src[start_pos..i].iter().collect();
        let mut sub = Parser::from_source(&slice);
        let vals = sub.parse_all()?;
        if vals.is_empty() { return Err(LyraError::Parse("expected association value".into())); }
        let val = vals.last().unwrap().clone();
        // Advance main parser to delimiter
        self.pos = i;
        // Determine delimiter kind
        // Skip whitespace
        self.skip_ws();
        if self.starts_with("|>") { self.pos += 2; return Ok((val, true)); }
        if self.peekc() == Some(',') { self.nextc(); return Ok((val, false)); }
        // If end-of-input or unexpected, surface a helpful error
        if self.peekc().is_none() { return Err(LyraError::Parse("unexpected end while parsing association value".into())); }
        Err(LyraError::Parse(format!("expected ',' or '|>' after association value, found {:?}", self.peekc())))
    }

    fn parse_string(&mut self) -> ParseResult<Value> {
        let mut s = String::new();
        if self.nextc()!=Some('"') { return Err(LyraError::Parse("expected '\"'".into())); }
        while let Some(c) = self.nextc() {
            match c {
                '"' => break,
                '\\' => {
                    if let Some(n) = self.nextc() { s.push(n); } else { break; }
                }
                _ => s.push(c),
            }
        }
        Ok(Value::String(s))
    }

    fn parse_number(&mut self) -> ParseResult<Value> {
        let start = self.pos;
        while matches!(self.peekc(), Some(c) if c.is_ascii_digit()) { self.nextc(); }
        if self.peekc()==Some('.') {
            self.nextc();
            while matches!(self.peekc(), Some(c) if c.is_ascii_digit()) { self.nextc(); }
            let s: String = self.src[start..self.pos].iter().collect();
            let f: f64 = s.parse().map_err(|e| LyraError::Parse(format!("invalid real: {e}")))?;
            Ok(Value::Real(f))
        } else {
            let s: String = self.src[start..self.pos].iter().collect();
            let n: i64 = s.parse().map_err(|e| LyraError::Parse(format!("invalid int: {e}")))?;
            Ok(Value::Integer(n))
        }
    }

    fn parse_symbol(&mut self) -> ParseResult<Value> {
        let mut s = String::new();
        if let Some(c)=self.peekc() { if is_ident_start(c) { s.push(c); self.nextc(); } }
        while let Some(c)=self.peekc() { if is_ident_continue(c) { s.push(c); self.nextc(); } else { break; } }
        match s.as_str() {
            "True" => Ok(Value::Boolean(true)),
            "False" => Ok(Value::Boolean(false)),
            _ => Ok(Value::Symbol(s)),
        }
    }

    fn parse_slot(&mut self) -> ParseResult<Value> {
        if self.nextc()!=Some('#') { return Err(LyraError::Parse("expected '#'".into())); }
        let start = self.pos;
        while matches!(self.peekc(), Some(c) if c.is_ascii_digit()) { self.nextc(); }
        if self.pos==start { return Ok(Value::slot(None)); }
        let s: String = self.src[start..self.pos].iter().collect();
        let n: usize = s.parse().map_err(|e| LyraError::Parse(format!("invalid slot: {e}")))?;
        Ok(Value::slot(Some(n)))
    }

    fn try_parse_lambda(&mut self) -> ParseResult<Option<Value>> {
        self.skip_ws();
        let save = self.pos;
        // (x, y) => body
        if self.peekc()==Some('(') {
            self.nextc();
            let mut params = Vec::new();
            self.skip_ws();
            if self.peekc()!=Some(')') {
                loop {
                    let sym = self.parse_symbol()?;
                    if let Value::Symbol(s) = sym { params.push(s); } else { self.pos = save; return Ok(None); }
                    self.skip_ws();
                    if self.peekc()==Some(',') { self.nextc(); self.skip_ws(); continue; }
                    break;
                }
            }
            if self.nextc()!=Some(')') { self.pos = save; return Ok(None); }
            self.skip_ws();
            if !self.eat_str("=>") { self.pos = save; return Ok(None); }
            self.skip_ws();
            let body = self.expr()?;
            return Ok(Some(Value::pure_function(Some(params), body)));
        }
        // x => body
        if is_ident_start(self.peekc().unwrap_or('\0')) {
            let save2 = self.pos;
            let sym = self.parse_symbol()?;
            if let Value::Symbol(s) = sym {
                self.skip_ws();
                if self.eat_str("=>") {
                    self.skip_ws();
                    let body = self.expr()?;
                    return Ok(Some(Value::pure_function(Some(vec![s]), body)));
                } else { self.pos = save2; }
            }
        }
        Ok(None)
    }
}

fn is_ident_start(c: char) -> bool { c.is_alphabetic() || c=='_' || c=='$' }
fn is_ident_continue(c: char) -> bool { c.is_alphanumeric() || c=='_' || c=='$' }
