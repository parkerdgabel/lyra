use crate::register_if;
use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;

#[cfg(feature = "text_fuzzy")]
use fuzzy_matcher::skim::SkimMatcherV2;
#[cfg(feature = "text_fuzzy")]
use fuzzy_matcher::FuzzyMatcher;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

/// Register fuzzy text matching and scoring helpers.
pub fn register_text_fuzzy(ev: &mut Evaluator) {
    ev.register("FuzzyFindInText", fuzzy_find_in_text as NativeFn, Attributes::empty());
    ev.register("FuzzyFindInList", fuzzy_find_in_list as NativeFn, Attributes::empty());
    ev.register("FuzzyFindInFiles", fuzzy_find_in_files as NativeFn, Attributes::empty());
}

/// Conditionally register fuzzy text functions based on `pred`.
pub fn register_text_fuzzy_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str) -> bool) {
    register_if(ev, pred, "FuzzyFindInText", fuzzy_find_in_text as NativeFn, Attributes::empty());
    register_if(ev, pred, "FuzzyFindInList", fuzzy_find_in_list as NativeFn, Attributes::empty());
    register_if(ev, pred, "FuzzyFindInFiles", fuzzy_find_in_files as NativeFn, Attributes::empty());
}

fn matcher_from_opts(_opts: &std::collections::HashMap<String, Value>) -> SkimMatcherV2 {
    SkimMatcherV2::default()
}

fn fuzzy_find_in_text(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // FuzzyFindInText(text, needle, opts?)
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("FuzzyFindInText".into())), args };
    }
    let text = match ev.eval(args[0].clone()) {
        Value::String(s) => s,
        v => lyra_core::pretty::format_value(&v),
    };
    let needle = match ev.eval(args[1].clone()) {
        Value::String(s) => s,
        v => lyra_core::pretty::format_value(&v),
    };
    let opts = args
        .get(2)
        .map(|v| ev.eval(v.clone()))
        .and_then(|v| match v {
            Value::Assoc(m) => Some(m),
            _ => None,
        })
        .unwrap_or_default();
    let matcher = matcher_from_opts(&opts);
    let mut out: Vec<Value> = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        if let Some((score, indices)) = matcher.fuzzy_indices(line, &needle) {
            out.push(Value::Assoc(
                [
                    ("value".into(), Value::String(line.to_string())),
                    ("score".into(), Value::Integer(score as i64)),
                    (
                        "indices".into(),
                        Value::List(
                            indices.into_iter().map(|i| Value::Integer(i as i64)).collect(),
                        ),
                    ),
                    ("lineNumber".into(), Value::Integer((idx + 1) as i64)),
                ]
                .into_iter()
                .collect(),
            ));
        }
    }
    Value::Assoc(
        [("items".into(), Value::List(out)), ("durationMs".into(), Value::Integer(0))]
            .into_iter()
            .collect(),
    )
}

fn fuzzy_find_in_list(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // FuzzyFindInList(items, needle, opts?)
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("FuzzyFindInList".into())), args };
    }
    let items = match ev.eval(args[0].clone()) {
        Value::List(vs) => vs,
        v => {
            return Value::Expr {
                head: Box::new(Value::Symbol("FuzzyFindInList".into())),
                args: vec![v],
            }
        }
    };
    let needle = match ev.eval(args[1].clone()) {
        Value::String(s) => s,
        v => lyra_core::pretty::format_value(&v),
    };
    let opts = args
        .get(2)
        .map(|v| ev.eval(v.clone()))
        .and_then(|v| match v {
            Value::Assoc(m) => Some(m),
            _ => None,
        })
        .unwrap_or_default();
    let matcher = matcher_from_opts(&opts);
    let mut out: Vec<(i64, Value)> = Vec::new();
    for it in items {
        let s = match it {
            Value::String(s) => s,
            v => lyra_core::pretty::format_value(&v),
        };
        if let Some((score, indices)) = matcher.fuzzy_indices(&s, &needle) {
            let rec = Value::Assoc(
                [
                    ("value".into(), Value::String(s)),
                    ("score".into(), Value::Integer(score as i64)),
                    (
                        "indices".into(),
                        Value::List(
                            indices.into_iter().map(|i| Value::Integer(i as i64)).collect(),
                        ),
                    ),
                ]
                .into_iter()
                .collect(),
            );
            out.push((score as i64, rec));
        }
    }
    out.sort_by(|a, b| b.0.cmp(&a.0));
    Value::Assoc(
        [
            ("items".into(), Value::List(out.into_iter().map(|(_, v)| v).collect())),
            ("durationMs".into(), Value::Integer(0)),
        ]
        .into_iter()
        .collect(),
    )
}

fn fuzzy_find_in_files(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // FuzzyFindInFiles({paths|path}, needle)
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("FuzzyFindInFiles".into())), args };
    }
    let input = match super::text::parse_input(ev, args[0].clone()) {
        Ok(i) => i,
        Err(e) => return super::text::failure("Fuzzy::input", &e),
    };
    let needle = match ev.eval(args[1].clone()) {
        Value::String(s) => s,
        v => lyra_core::pretty::format_value(&v),
    };
    let matcher = SkimMatcherV2::default();
    let mut out: Vec<(i64, Value)> = Vec::new();
    match input {
        super::text::InputKind::Text(s) => {
            for (lnum, line) in s.lines().enumerate() {
                if let Some((score, indices)) = matcher.fuzzy_indices(line, &needle) {
                    out.push((
                        score as i64,
                        Value::Assoc(
                            [
                                ("value".into(), Value::String(line.to_string())),
                                ("file".into(), Value::String(String::new())),
                                ("lineNumber".into(), Value::Integer((lnum + 1) as i64)),
                                ("score".into(), Value::Integer(score as i64)),
                                (
                                    "indices".into(),
                                    Value::List(
                                        indices
                                            .into_iter()
                                            .map(|i| Value::Integer(i as i64))
                                            .collect(),
                                    ),
                                ),
                            ]
                            .into_iter()
                            .collect(),
                        ),
                    ));
                }
            }
        }
        super::text::InputKind::Files(paths) => {
            for p in paths {
                if let Ok(s) = super::text::read_file_to_string(&p) {
                    for (lnum, line) in s.lines().enumerate() {
                        if let Some((score, indices)) = matcher.fuzzy_indices(line, &needle) {
                            out.push((
                                score as i64,
                                Value::Assoc(
                                    [
                                        ("value".into(), Value::String(line.to_string())),
                                        ("file".into(), Value::String(p.clone())),
                                        ("lineNumber".into(), Value::Integer((lnum + 1) as i64)),
                                        ("score".into(), Value::Integer(score as i64)),
                                        (
                                            "indices".into(),
                                            Value::List(
                                                indices
                                                    .into_iter()
                                                    .map(|i| Value::Integer(i as i64))
                                                    .collect(),
                                            ),
                                        ),
                                    ]
                                    .into_iter()
                                    .collect(),
                                ),
                            ));
                        }
                    }
                }
            }
        }
    }
    out.sort_by(|a, b| b.0.cmp(&a.0));
    Value::Assoc(
        [
            ("items".into(), Value::List(out.into_iter().map(|(_, v)| v).collect())),
            ("durationMs".into(), Value::Integer(0)),
        ]
        .into_iter()
        .collect(),
    )
}
