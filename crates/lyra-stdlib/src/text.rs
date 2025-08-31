use crate::register_if;
#[cfg(feature = "text_multipattern")]
use aho_corasick::AhoCorasick;
#[cfg(feature = "text_encoding_detect")]
use chardetng::EncodingDetector;
#[cfg(feature = "text_glob")]
use ignore::{overrides::OverrideBuilder, WalkBuilder};
use lyra_core::value::Value;
use std::collections::HashMap;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
use regex::Regex;
// No direct use of encoding_rs::Encoding type; avoid unused import warning

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

/// Register text processing: find/count/search, line splitting, replace,
/// globbing-backed file search, and encoding detection.
pub fn register_text(ev: &mut Evaluator) {
    ev.register("TextFind", text_find as NativeFn, Attributes::empty());
    ev.register("TextCount", text_count as NativeFn, Attributes::empty());
    ev.register("TextFilesWithMatch", text_files_with_match as NativeFn, Attributes::empty());
    ev.register("TextLines", text_lines as NativeFn, Attributes::empty());
    ev.register("TextReplace", text_replace as NativeFn, Attributes::empty());
    ev.register("TextDetectEncoding", text_detect_encoding as NativeFn, Attributes::empty());
    ev.register("TextSearch", text_search as NativeFn, Attributes::empty());
    // NLP primitives
    ev.register("Tokenize", tokenize as NativeFn, Attributes::empty());
    ev.register("SentenceSplit", sentence_split as NativeFn, Attributes::empty());
    ev.register("NormalizeText", normalize_text as NativeFn, Attributes::empty());
    ev.register("Stopwords", stopwords as NativeFn, Attributes::empty());
    ev.register("RemoveStopwords", remove_stopwords as NativeFn, Attributes::empty());
    ev.register("Stem", stem as NativeFn, Attributes::empty());
    ev.register("Ngrams", ngrams as NativeFn, Attributes::empty());
    ev.register("TokenStats", token_stats as NativeFn, Attributes::empty());
    ev.register("BuildVocab", build_vocab as NativeFn, Attributes::empty());
    ev.register("BagOfWords", bag_of_words as NativeFn, Attributes::empty());
    ev.register("TfIdf", tfidf as NativeFn, Attributes::empty());
    ev.register("ChunkText", chunk_text as NativeFn, Attributes::empty());
    ev.register("Lemmatize", lemmatize as NativeFn, Attributes::empty());
}

/// Conditionally register text functions based on `pred`.
pub fn register_text_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str) -> bool) {
    register_if(ev, pred, "TextFind", text_find as NativeFn, Attributes::empty());
    register_if(ev, pred, "TextCount", text_count as NativeFn, Attributes::empty());
    register_if(
        ev,
        pred,
        "TextFilesWithMatch",
        text_files_with_match as NativeFn,
        Attributes::empty(),
    );
    register_if(ev, pred, "TextLines", text_lines as NativeFn, Attributes::empty());
    register_if(ev, pred, "TextReplace", text_replace as NativeFn, Attributes::empty());
    register_if(
        ev,
        pred,
        "TextDetectEncoding",
        text_detect_encoding as NativeFn,
        Attributes::empty(),
    );
    register_if(ev, pred, "TextSearch", text_search as NativeFn, Attributes::empty());
    // NLP primitives
    register_if(ev, pred, "Tokenize", tokenize as NativeFn, Attributes::empty());
    register_if(ev, pred, "SentenceSplit", sentence_split as NativeFn, Attributes::empty());
    register_if(ev, pred, "NormalizeText", normalize_text as NativeFn, Attributes::empty());
    register_if(ev, pred, "Stopwords", stopwords as NativeFn, Attributes::empty());
    register_if(ev, pred, "RemoveStopwords", remove_stopwords as NativeFn, Attributes::empty());
    register_if(ev, pred, "Stem", stem as NativeFn, Attributes::empty());
    register_if(ev, pred, "Ngrams", ngrams as NativeFn, Attributes::empty());
    register_if(ev, pred, "TokenStats", token_stats as NativeFn, Attributes::empty());
    register_if(ev, pred, "BuildVocab", build_vocab as NativeFn, Attributes::empty());
    register_if(ev, pred, "BagOfWords", bag_of_words as NativeFn, Attributes::empty());
    register_if(ev, pred, "TfIdf", tfidf as NativeFn, Attributes::empty());
    register_if(ev, pred, "ChunkText", chunk_text as NativeFn, Attributes::empty());
    register_if(ev, pred, "Lemmatize", lemmatize as NativeFn, Attributes::empty());
}

fn ok_summary(files: usize, files_with: usize, total: usize, dur_ms: u128) -> Value {
    Value::Assoc(
        [
            ("filesSearched".into(), Value::Integer(files as i64)),
            ("filesWithMatch".into(), Value::Integer(files_with as i64)),
            ("totalMatches".into(), Value::Integer(total as i64)),
            ("durationMs".into(), Value::Integer(dur_ms as i64)),
        ]
        .into_iter()
        .collect(),
    )
}

pub(crate) fn failure(tag: &str, msg: &str) -> Value {
    Value::Assoc(
        [
            ("message".to_string(), Value::String(msg.to_string())),
            ("tag".to_string(), Value::String(tag.to_string())),
        ]
        .into_iter()
        .collect(),
    )
}

#[derive(Clone)]
enum PatternKind {
    Regex(Regex),
    #[cfg(feature = "text_multipattern")]
    Multi {
        ac: AhoCorasick,
        #[allow(dead_code)]
        pats: Vec<String>,
    },
}

fn parse_pattern(
    ev: &mut Evaluator,
    v: Value,
    opts: Option<&std::collections::HashMap<String, Value>>,
) -> Result<PatternKind, String> {
    let mut case_insensitive = false;
    let mut multiline = false;
    if let Some(o) = opts {
        if let Some(Value::Assoc(ctx)) = o.get("context") {
            let _ = ctx;
        }
        if let Some(Value::Boolean(b)) = o.get("caseInsensitive") {
            case_insensitive = *b;
        }
        if let Some(Value::Boolean(b)) = o.get("multiline") {
            multiline = *b;
        }
    }
    let (mut regex_src, is_regex, _multi) = match ev.eval(v) {
        Value::Assoc(m) => {
            if let Some(Value::List(list)) = m.get("multi").or_else(|| m.get("Multi")) {
                let mut pats = Vec::new();
                for it in list {
                    match ev.eval(it.clone()) {
                        Value::String(s) => pats.push(s),
                        other => {
                            return Err(format!(
                                "Invalid multi item: {}",
                                lyra_core::pretty::format_value(&other)
                            ))
                        }
                    }
                }
                #[cfg(feature = "text_multipattern")]
                {
                    return Ok(PatternKind::Multi {
                        ac: AhoCorasick::new(&pats).map_err(|e| e.to_string())?,
                        pats,
                    });
                }
                #[cfg(not(feature = "text_multipattern"))]
                {
                    return Err("multi pattern requires feature 'text_multipattern'".into());
                }
            } else if let Some(Value::String(s)) = m.get("regex").or_else(|| m.get("Regex")) {
                (s.clone(), true, false)
            } else if let Some(Value::String(s)) = m.get("literal").or_else(|| m.get("Literal")) {
                (regex::escape(s), false, false)
            } else {
                return Err("Missing 'regex' or 'literal' in pattern".into());
            }
        }
        Value::String(s) | Value::Symbol(s) => (s, true, false),
        other => {
            return Err(format!("Invalid pattern: {}", lyra_core::pretty::format_value(&other)))
        }
    };
    if case_insensitive {
        regex_src = format!("(?i){}", regex_src);
    }
    if multiline && is_regex {
        regex_src = format!("(?m){}", regex_src);
    }
    let re = Regex::new(&regex_src).map_err(|e| e.to_string())?;
    Ok(PatternKind::Regex(re))
}

#[derive(Default)]
struct Ctx {
    before: usize,
    after: usize,
}

fn parse_opts(
    ev: &mut Evaluator,
    v: Option<Value>,
) -> (std::collections::HashMap<String, Value>, Ctx) {
    let m = v
        .map(|vv| ev.eval(vv))
        .and_then(|v| match v {
            Value::Assoc(m) => Some(m),
            _ => None,
        })
        .unwrap_or_default();
    let mut ctx = Ctx::default();
    if let Some(Value::Assoc(cm)) = m.get("context") {
        if let Some(Value::Integer(n)) = cm.get("before") {
            ctx.before = (*n).max(0) as usize;
        }
        if let Some(Value::Integer(n)) = cm.get("after") {
            ctx.after = (*n).max(0) as usize;
        }
    }
    (m, ctx)
}

#[derive(Clone)]
pub(crate) enum InputKind {
    Text(String),
    Files(Vec<String>),
}

pub(crate) fn parse_input(ev: &mut Evaluator, v: Value) -> Result<InputKind, String> {
    match ev.eval(v) {
        Value::Assoc(m) => {
            if let Some(Value::String(s)) = m.get("text").or_else(|| m.get("Text")) {
                Ok(InputKind::Text(s.clone()))
            } else if let Some(Value::String(p)) = m.get("path").or_else(|| m.get("Path")) {
                Ok(InputKind::Files(vec![p.clone()]))
            } else if let Some(Value::List(ps)) = m.get("paths").or_else(|| m.get("Paths")) {
                let mut out = Vec::new();
                for p in ps {
                    match ev.eval(p.clone()) {
                        Value::String(s) => out.push(s),
                        other => {
                            return Err(format!(
                                "Invalid path: {}",
                                lyra_core::pretty::format_value(&other)
                            ))
                        }
                    }
                }
                Ok(InputKind::Files(out))
            } else if let Some(Value::String(g)) = m.get("glob").or_else(|| m.get("Glob")) {
                Ok(InputKind::Files(resolve_glob_paths(&[g.clone()], &m)))
            } else if let Some(Value::List(gs)) = m.get("globs").or_else(|| m.get("Globs")) {
                let mut pats = Vec::new();
                for p in gs {
                    match ev.eval(p.clone()) {
                        Value::String(s) => pats.push(s),
                        other => {
                            return Err(format!(
                                "Invalid glob: {}",
                                lyra_core::pretty::format_value(&other)
                            ))
                        }
                    }
                }
                Ok(InputKind::Files(resolve_glob_paths(&pats, &m)))
            } else {
                Err("Provide {text}|{path}|{paths}".into())
            }
        }
        Value::String(s) => Ok(InputKind::Text(s)),
        other => Err(format!("Invalid input: {}", lyra_core::pretty::format_value(&other))),
    }
}

#[cfg_attr(not(feature = "text_glob"), allow(unused_variables))]
fn resolve_glob_paths(
    patterns: &[String],
    opts: &std::collections::HashMap<String, Value>,
) -> Vec<String> {
    #[cfg(feature = "text_glob")]
    {
        let mut ob = OverrideBuilder::new(".");
        for p in patterns {
            let _ = ob.add(p);
        }
        if let Some(Value::List(ex)) = opts.get("excludeGlobs").or_else(|| opts.get("ExcludeGlobs"))
        {
            for v in ex {
                if let Value::String(s) = v {
                    let _ = ob.add(&format!("!{}", s));
                }
            }
        }
        let ov = ob.build().unwrap();
        let mut wb = WalkBuilder::new(".");
        wb.overrides(ov);
        let follow = opts
            .get("followSymlinks")
            .and_then(|v| if let Value::Boolean(b) = v { Some(*b) } else { None })
            .unwrap_or(false);
        wb.follow_links(follow);
        let hidden = opts
            .get("hidden")
            .and_then(|v| if let Value::Boolean(b) = v { Some(*b) } else { None })
            .unwrap_or(false);
        wb.hidden(!hidden);
        let respect = opts
            .get("respectVCSIgnores")
            .and_then(|v| if let Value::Boolean(b) = v { Some(*b) } else { None })
            .unwrap_or(true);
        wb.git_global(respect).git_ignore(respect).git_exclude(respect);
        let mut out = Vec::new();
        for r in wb.build() {
            if let Ok(de) = r {
                if de.file_type().map(|t| t.is_file()).unwrap_or(false) {
                    if let Some(p) = de.path().to_str() {
                        out.push(p.to_string());
                    }
                }
            }
        }
        out
    }
    #[cfg(not(feature = "text_glob"))]
    {
        patterns.iter().cloned().collect()
    }
}

// ---------------- NLP: Tokenize, Normalize, Stopwords, Ngrams ----------------

fn to_string(ev: &mut Evaluator, v: Value) -> String { match ev.eval(v) { Value::String(s) | Value::Symbol(s) => s, other => lyra_core::pretty::format_value(&other) } }

fn normalize_opts_map(ev: &mut Evaluator, v: Option<Value>) -> std::collections::HashMap<String, Value> {
    v.map(|x| ev.eval(x)).and_then(|vv| if let Value::Assoc(m)=vv { Some(m) } else { None }).unwrap_or_default()
}

fn simple_word_tokens(s: &str, preserve_case: bool, strip_punct: bool, keep_numbers: bool) -> Vec<(String, usize, usize)> {
    let mut out: Vec<(String, usize, usize)> = Vec::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        while i < chars.len() {
            let c = chars[i];
            let is_alnum = c.is_alphanumeric();
            if is_alnum || (!strip_punct && (c=='\'' || c=='_')) { break; }
            i+=1;
        }
        if i>=chars.len() { break; }
        let start = i;
        while i < chars.len() {
            let c = chars[i];
            if c.is_alphanumeric() || (!strip_punct && (c=='\'' || c=='_')) { i+=1; } else { break; }
        }
        let end = i;
        if end>start {
            let mut t: String = chars[start..end].iter().collect();
            if !preserve_case { t = t.to_lowercase(); }
            if !keep_numbers && t.chars().all(|c| c.is_ascii_digit()) { continue; }
            out.push((t, start, end));
        }
    }
    out
}

fn tokenize(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Tokenize".into())), args }; }
    let text = to_string(ev, args[0].clone());
    let opts = normalize_opts_map(ev, args.get(1).cloned());
    let mode = opts.get("mode").and_then(|v| if let Value::String(s)|Value::Symbol(s)=v { Some(s.to_ascii_lowercase()) } else { None }).unwrap_or_else(|| "word".into());
    let preserve_case = matches!(opts.get("preserveCase"), Some(Value::Boolean(true)));
    let strip_punct = !matches!(opts.get("stripPunct"), Some(Value::Boolean(false)));
    let keep_numbers = !matches!(opts.get("keepNumbers"), Some(Value::Boolean(false)));
    let return_kind = opts.get("return").and_then(|v| if let Value::String(s)|Value::Symbol(s)=v { Some(s.to_ascii_lowercase()) } else { None }).unwrap_or_else(|| "tokens".into());
    let mut out: Vec<(String, usize, usize)> = match mode.as_str() {
        "char" => text.char_indices().map(|(i,c)| (c.to_string(), i, i+1)).collect(),
        _ => simple_word_tokens(&text, preserve_case, strip_punct, keep_numbers),
    };
    match return_kind.as_str() {
        "spans" => Value::List(out.into_iter().map(|(t,s,e)| Value::Assoc(HashMap::from([(String::from("text"), Value::String(t)), (String::from("start"), Value::Integer(s as i64)), (String::from("end"), Value::Integer(e as i64))]))).collect()),
        "assoc" => Value::Assoc(HashMap::from([(String::from("tokens"), Value::List(out.into_iter().map(|(t,_,_)| Value::String(t)).collect()))])),
        _ => Value::List(out.into_iter().map(|(t,_,_)| Value::String(t)).collect()),
    }
}

fn sentence_split(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("SentenceSplit".into())), args }; }
    let text = to_string(ev, args[0].clone());
    let _opts = normalize_opts_map(ev, args.get(1).cloned());
    // Simple heuristic: split on ., !, ? followed by space/newline; preserve punctuation
    let mut out: Vec<String> = Vec::new();
    let mut buf = String::new();
    for ch in text.chars() {
        buf.push(ch);
        if matches!(ch, '.'|'!'|'?') { out.push(buf.trim().to_string()); buf.clear(); }
        else if ch=='\n' && !buf.trim().is_empty() { out.push(buf.trim().to_string()); buf.clear(); }
    }
    if !buf.trim().is_empty() { out.push(buf.trim().to_string()); }
    Value::List(out.into_iter().map(Value::String).collect())
}

fn normalize_text(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("NormalizeText".into())), args }; }
    let mut s = to_string(ev, args[0].clone());
    let opts = normalize_opts_map(ev, args.get(1).cloned());
    if let Some(Value::String(mode))|Some(Value::Symbol(mode)) = opts.get("case") { match mode.as_str() { "lower" => s = s.to_lowercase(), "upper" => s = s.to_uppercase(), _ => {} } }
    if matches!(opts.get("whitespace"), Some(Value::String(ws)) if ws=="collapse") {
        let mut out = String::new(); let mut last_space = false;
        for ch in s.chars() { if ch.is_whitespace() { if !last_space { out.push(' '); last_space=true; } } else { out.push(ch); last_space=false; } }
        s = out.trim().to_string();
    }
    if matches!(opts.get("punctuation"), Some(Value::String(p)) if p=="remove") {
        s = s.chars().filter(|c| !c.is_ascii_punctuation()).collect();
    }
    if matches!(opts.get("digits"), Some(Value::String(d)) if d=="remove") {
        s = s.chars().filter(|c| !c.is_ascii_digit()).collect();
    }
    Value::String(s)
}

fn stopwords(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let lang = match args.get(0) { Some(Value::String(s))|Some(Value::Symbol(s)) => s.to_ascii_lowercase(), _ => String::from("en") };
    // Minimal built-in stopword lists for common languages
    const EN: &[&str] = &[
        "a","an","the","and","or","but","if","then","else","when","at","by","for","in","of","on","to","up","with","as","is","it","its","be","are","was","were","this","that","these","those","from","has","have","had","he","she","they","them","his","her","their","we","you","your","i","me","my","our","ours"
    ];
    const ES: &[&str] = &[
        "un","una","unos","unas","el","la","los","las","y","o","pero","si","entonces","sino","cuando","en","de","del","al","por","para","con","como","es","ser","son","fue","fueron","esto","esa","ese","estos","esas","desde","ha","han","tiene","tienen","él","ella","ellos","ellas","su","sus","nosotros","vosotros","usted","ustedes","yo","mi","mis","nuestro","nuestra","nuestros","nuestras"
    ];
    const FR: &[&str] = &[
        "un","une","des","le","la","les","et","ou","mais","si","alors","sinon","quand","à","au","aux","de","du","des","pour","par","avec","comme","est","être","sont","était","étaient","ceci","cela","ces","depuis","a","ont","il","elle","ils","elles","son","sa","ses","nous","vous","je","moi","mon","mes","notre","nos"
    ];
    const DE: &[&str] = &[
        "ein","eine","einer","eines","der","die","das","und","oder","aber","wenn","dann","sonst","wann","bei","von","für","mit","als","ist","sein","sind","war","waren","dies","das","diese","diese","seit","hat","haben","er","sie","sie","ihr","ihre","wir","ihr","Sie","ich","mich","mein","meine","unser","unsere"
    ];
    const IT: &[&str] = &[
        "un","una","uno","dei","degli","del","della","lo","la","il","e","o","ma","se","allora","altrimenti","quando","a","di","da","per","con","come","è","essere","sono","era","erano","questo","quello","questi","quelle","da","ha","hanno","lui","lei","essi","esse","suo","sua","suoi","noi","voi","io","me","mio","mia","nostro","nostri"
    ];
    const PT: &[&str] = &[
        "um","uma","uns","umas","o","a","os","as","e","ou","mas","se","então","senão","quando","em","de","do","da","dos","das","por","para","com","como","é","ser","são","foi","foram","isto","isso","estes","essas","desde","tem","têm","ele","ela","eles","elas","seu","sua","seus","nós","vós","você","vocês","eu","meu","minha","nossos","nossas"
    ];
    const NL: &[&str] = &[
        "een","de","het","en","of","maar","als","dan","anders","wanneer","bij","van","voor","met","als","is","zijn","waren","dit","dat","deze","die","sinds","heeft","hebben","hij","zij","ze","hun","wij","jullie","u","ik","mij","mijn","onze"
    ];
    let list_slice: &[&str] = match lang.as_str() {
        "en" => EN,
        "es" => ES,
        "fr" => FR,
        "de" => DE,
        "it" => IT,
        "pt" => PT,
        "nl" => NL,
        _ => EN,
    };
    let list: Vec<Value> = list_slice.iter().map(|s| Value::String((*s).into())).collect();
    Value::List(list)
}

fn remove_stopwords(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("RemoveStopwords".into())), args }; }
    let input = ev.eval(args[0].clone());
    let opts = normalize_opts_map(ev, args.get(1).cloned());
    let lang = opts.get("language").and_then(|v| if let Value::String(s)|Value::Symbol(s)=v { Some(s.as_str()) } else { None }).unwrap_or("en");
    let sw: std::collections::HashSet<String> = match opts.get("stopwords") {
        Some(Value::List(vs)) => vs.iter().filter_map(|v| if let Value::String(s)|Value::Symbol(s)=v { Some(s.to_ascii_lowercase()) } else { None }).collect(),
        _ => {
            let l = stopwords(ev, vec![Value::String(lang.into())]);
            if let Value::List(vs) = l { vs.into_iter().filter_map(|v| if let Value::String(s)=v { Some(s) } else { None }).collect() } else { std::collections::HashSet::new() }
        }
    };
    let toks: Vec<String> = match input {
        Value::List(vs) => vs.into_iter().filter_map(|v| if let Value::String(s)|Value::Symbol(s)=v { Some(s) } else { None }).collect(),
        Value::String(s) => simple_word_tokens(&s, false, true, true).into_iter().map(|(t,_,_)| t).collect(),
        other => vec![lyra_core::pretty::format_value(&other)],
    };
    let out: Vec<Value> = toks.into_iter().filter(|t| !sw.contains(&t.to_ascii_lowercase())).map(Value::String).collect();
    Value::List(out)
}

fn ngrams(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Ngrams".into())), args }; }
    let input = ev.eval(args[0].clone());
    let opts = normalize_opts_map(ev, args.get(1).cloned());
    let n = opts.get("n").and_then(|v| if let Value::Integer(i)=v { Some((*i).max(1) as usize) } else { None }).unwrap_or(2);
    let sep = opts.get("join").and_then(|v| if let Value::String(s)|Value::Symbol(s)=v { Some(s.clone()) } else { None }).unwrap_or_else(|| " ".into());
    let toks: Vec<String> = match input { Value::List(vs) => vs.into_iter().filter_map(|v| if let Value::String(s)|Value::Symbol(s)=v { Some(s) } else { None }).collect(), Value::String(s) => simple_word_tokens(&s, false, true, true).into_iter().map(|(t,_,_)| t).collect(), other => vec![lyra_core::pretty::format_value(&other)] };
    if toks.len() < n { return Value::List(vec![]); }
    let mut out: Vec<Value> = Vec::new();
    for i in 0..=(toks.len()-n) {
        let gram = toks[i..i+n].join(&sep);
        out.push(Value::String(gram));
    }
    Value::List(out)
}

fn token_stats(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("TokenStats".into())), args }; }
    let input = ev.eval(args[0].clone());
    let opts = normalize_opts_map(ev, args.get(1).cloned());
    let k = opts.get("k").and_then(|v| if let Value::Integer(i)=v { Some((*i).max(1) as usize) } else { None }).unwrap_or(10);
    let toks: Vec<String> = match input { Value::List(vs) => vs.into_iter().filter_map(|v| if let Value::String(s)|Value::Symbol(s)=v { Some(s) } else { None }).collect(), Value::String(s) => simple_word_tokens(&s, false, true, true).into_iter().map(|(t,_,_)| t).collect(), other => vec![lyra_core::pretty::format_value(&other)] };
    let total = toks.len();
    let mut map: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
    for t in toks { *map.entry(t).or_insert(0) += 1; }
    let mut items: Vec<(String, i64)> = map.into_iter().collect();
    items.sort_by(|a,b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let top_list: Vec<Value> = items.into_iter().take(k).map(|(term,count)| Value::Assoc(HashMap::from([(String::from("term"), Value::String(term)), (String::from("count"), Value::Integer(count)), (String::from("proportion"), Value::Real((count as f64)/ (total as f64 + 1e-12)))]))).collect();
    Value::Assoc(HashMap::from([(String::from("total"), Value::Integer(total as i64)), (String::from("unique"), Value::Integer(top_list.len() as i64)), (String::from("top"), Value::List(top_list))]))
}

fn build_vocab(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("BuildVocab".into())), args }; }
    let docs = match ev.eval(args[0].clone()) { Value::List(vs) => vs, other => vec![other] };
    let opts = normalize_opts_map(ev, args.get(1).cloned());
    let min_count = opts.get("minCount").and_then(|v| if let Value::Integer(i)=v { Some((*i).max(1) as i64) } else { None }).unwrap_or(1);
    let mut counts: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
    for d in docs {
        let toks: Vec<String> = match d { Value::List(vs) => vs.into_iter().filter_map(|v| if let Value::String(s)|Value::Symbol(s)=v { Some(s) } else { None }).collect(), Value::String(s) => simple_word_tokens(&s, false, true, true).into_iter().map(|(t,_,_)| t).collect(), other => vec![lyra_core::pretty::format_value(&other)] };
        for t in toks { *counts.entry(t).or_insert(0) += 1; }
    }
    let mut terms: Vec<String> = counts.into_iter().filter(|(_t,c)| *c >= min_count).map(|(t,_c)| t).collect();
    terms.sort();
    let vocab: std::collections::HashMap<String, Value> = terms.iter().enumerate().map(|(i,t)| (t.clone(), Value::Integer(i as i64))).collect();
    Value::Assoc(HashMap::from([(String::from("vocab"), Value::Assoc(vocab)), (String::from("size"), Value::Integer(terms.len() as i64))]))
}

fn bag_of_words(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("BagOfWords".into())), args }; }
    let input = ev.eval(args[0].clone());
    let opts = normalize_opts_map(ev, args.get(1).cloned());
    let toks: Vec<String> = match input { Value::List(vs) => vs.into_iter().filter_map(|v| if let Value::String(s)|Value::Symbol(s)=v { Some(s) } else { None }).collect(), Value::String(s) => simple_word_tokens(&s, false, true, true).into_iter().map(|(t,_,_)| t).collect(), other => vec![lyra_core::pretty::format_value(&other)] };
    if let Some(Value::Assoc(vmap)) = opts.get("vocab") {
        // produce dense vector
        let size = vmap.len();
        let mut vec: Vec<f64> = vec![0.0; size];
        for t in toks { if let Some(Value::Integer(ix)) = vmap.get(&t) { let i = (*ix).max(0) as usize; if i < size { vec[i] += 1.0; } } }
        return Value::PackedArray { shape: vec![size], data: vec };
    }
    // produce assoc term->count
    let mut map: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
    for t in toks { *map.entry(t).or_insert(0) += 1; }
    Value::Assoc(map.into_iter().map(|(k,v)| (k, Value::Integer(v))).collect())
}

fn tfidf(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("TfIdf".into())), args }; }
    let docs_v = ev.eval(args[0].clone());
    let docs = match docs_v { Value::List(vs) => vs, other => vec![other] };
    let opts = normalize_opts_map(ev, args.get(1).cloned());
    let vocab_val = build_vocab(ev, vec![Value::List(docs.clone()), Value::Assoc(opts.clone())]);
    let (vocab, size) = if let Value::Assoc(m) = vocab_val { (m.get("vocab").cloned().unwrap_or(Value::Assoc(HashMap::new())), m.get("size").cloned().unwrap_or(Value::Integer(0))) } else { (Value::Assoc(HashMap::new()), Value::Integer(0)) };
    let vocab_map = if let Value::Assoc(m) = vocab.clone() { m } else { HashMap::new() };
    let vsize = if let Value::Integer(n) = size { n.max(0) as usize } else { 0 };
    // df per term
    let mut df: Vec<i64> = vec![0; vsize];
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for d in docs.iter() {
        let toks: std::collections::HashSet<String> = match d { Value::List(vs) => vs.iter().filter_map(|v| if let Value::String(s)|Value::Symbol(s)=v { Some(s.clone()) } else { None }).collect(), Value::String(s) => simple_word_tokens(&s, false, true, true).into_iter().map(|(t,_,_)| t).collect(), other => vec![lyra_core::pretty::format_value(other)].into_iter().collect() };
        let mut row: Vec<f64> = vec![0.0; vsize];
        for t in toks.iter() {
            if let Some(Value::Integer(ix)) = vocab_map.get(t) { let i = (*ix).max(0) as usize; if i< vsize { df[i] += 1; } }
        }
        // term frequencies for this doc
        let toks_all: Vec<String> = match d { Value::List(vs) => vs.iter().filter_map(|v| if let Value::String(s)|Value::Symbol(s)=v { Some(s.clone()) } else { None }).collect(), Value::String(s) => simple_word_tokens(&s, false, true, true).into_iter().map(|(t,_,_)| t).collect(), other => vec![lyra_core::pretty::format_value(other)] };
        for t in toks_all { if let Some(Value::Integer(ix)) = vocab_map.get(&t) { let i = (*ix).max(0) as usize; if i < vsize { row[i] += 1.0; } } }
        rows.push(row);
    }
    let n_docs = docs.len().max(1) as f64;
    let idf: Vec<f64> = df.into_iter().map(|d| { let dfv = d as f64; if dfv>0.0 { (n_docs / dfv).ln() + 1.0 } else { 0.0 } }).collect();
    // tf-idf rows with l2 norm if requested
    let norm_l2 = !matches!(opts.get("norm"), Some(Value::String(s)) if s.eq_ignore_ascii_case("none"));
    let mut tfidf_rows: Vec<Value> = Vec::new();
    for mut row in rows.into_iter() {
        for i in 0..vsize { row[i] *= idf[i]; }
        if norm_l2 {
            let ss: f64 = row.iter().map(|x| x*x).sum();
            let denom = ss.sqrt();
            if denom > 0.0 { for i in 0..vsize { row[i] /= denom; } }
        }
        tfidf_rows.push(Value::PackedArray { shape: vec![vsize], data: row });
    }
    Value::Assoc(HashMap::from([
        (String::from("vocab"), vocab),
        (String::from("idf"), Value::PackedArray { shape: vec![vsize], data: idf }),
        (String::from("matrix"), Value::List(tfidf_rows)),
    ]))
}

fn chunk_text(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("ChunkText".into())), args }; }
    let text = to_string(ev, args[0].clone());
    let opts = normalize_opts_map(ev, args.get(1).cloned());
    let chunk_size = opts.get("chunkSize").and_then(|v| if let Value::Integer(i)=v { Some((*i).max(1) as usize) } else { None }).unwrap_or(1000);
    let chunk_overlap = opts.get("chunkOverlap").and_then(|v| if let Value::Integer(i)=v { Some((*i).max(0) as usize) } else { None }).unwrap_or(200);
    // naive recursive character strategy: break on double newline, then single, fallback to sliding window
    let mut chunks: Vec<(usize,usize)> = Vec::new();
    let mut start = 0usize;
    let bytes = text.as_bytes();
    while start < bytes.len() {
        let end_limit = (start + chunk_size).min(bytes.len());
        let slice = &text[start..end_limit];
        let mut cut = slice.rfind("\n\n").map(|i| i+start+2)
            .or_else(|| slice.rfind('\n').map(|i| i+start+1))
            .unwrap_or(end_limit);
        if cut <= start { cut = end_limit; }
        chunks.push((start, cut));
        if cut>=bytes.len() { break; }
        start = cut.saturating_sub(chunk_overlap);
    }
    let out: Vec<Value> = chunks.into_iter().map(|(s,e)| Value::Assoc(HashMap::from([(String::from("text"), Value::String(text[s..e].to_string())), (String::from("start"), Value::Integer(s as i64)), (String::from("end"), Value::Integer(e as i64))]))).collect();
    Value::List(out)
}

fn is_probably_binary(bytes: &[u8]) -> bool {
    // simple heuristic: any NUL byte indicates binary
    bytes.iter().any(|&b| b == 0)
}

pub(crate) fn read_file_to_string(path: &str) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if is_probably_binary(&bytes) {
        return Err("BinaryFileSkipped".into());
    }
    let s = String::from_utf8_lossy(&bytes).to_string();
    Ok(s)
}

fn text_find(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // TextFind(input, pattern, opts?) -> { matches: [...], summary: {...} }
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("TextFind".into())), args };
    }
    let start = std::time::Instant::now();
    let input = match parse_input(ev, args[0].clone()) {
        Ok(i) => i,
        Err(e) => return failure("Text::input", &e),
    };
    let (opts_map, ctx) = parse_opts(ev, args.get(2).cloned());
    let pat = match parse_pattern(ev, args[1].clone(), Some(&opts_map)) {
        Ok(p) => p,
        Err(e) => return failure("Text::pattern", &e),
    };

    let mut matches: Vec<Value> = Vec::new();
    let mut files_scanned = 0usize;
    let mut files_with = 0usize;
    let mut total = 0usize;
    let max_total = opts_map.get("maxMatches").and_then(|v| {
        if let Value::Integer(n) = v {
            Some(*n as usize)
        } else {
            None
        }
    });
    let max_per_file = opts_map.get("maxMatchesPerFile").and_then(|v| {
        if let Value::Integer(n) = v {
            Some(*n as usize)
        } else {
            None
        }
    });
    let timeout_ms = opts_map.get("timeoutMs").and_then(|v| {
        if let Value::Integer(n) = v {
            Some(*n as u128)
        } else {
            None
        }
    });

    match input {
        InputKind::Text(s) => {
            let (line_starts, lines) = line_indices(&s);
            for (lnum, line) in lines.iter().enumerate() {
                let base = line_starts[lnum];
                let mut per_line_count = 0usize;
                for (mstart, mend, caps) in iter_matches(&pat, line) {
                    total += 1;
                    per_line_count += 1;
                    let mut m = std::collections::HashMap::new();
                    m.insert("file".into(), Value::String(String::new()));
                    m.insert("line".into(), Value::String(line.clone()));
                    m.insert("lineNumber".into(), Value::Integer((lnum + 1) as i64));
                    m.insert(
                        "charRange".into(),
                        Value::List(vec![
                            Value::Integer(mstart as i64),
                            Value::Integer(mend as i64),
                        ]),
                    );
                    m.insert(
                        "byteRange".into(),
                        Value::List(vec![
                            Value::Integer((base + mstart) as i64),
                            Value::Integer((base + mend) as i64),
                        ]),
                    );
                    if !caps.is_empty() {
                        m.insert(
                            "captures".into(),
                            Value::List(
                                caps.iter()
                                    .map(|(s, e)| {
                                        Value::List(vec![
                                            Value::Integer(*s as i64),
                                            Value::Integer(*e as i64),
                                        ])
                                    })
                                    .collect(),
                            ),
                        );
                    }
                    if ctx.before > 0 || ctx.after > 0 {
                        m.insert(
                            "context".into(),
                            extract_context(&lines, lnum, ctx.before, ctx.after),
                        );
                    }
                    matches.push(Value::Assoc(m));
                    if let Some(limit) = max_per_file {
                        if per_line_count >= limit {
                            break;
                        }
                    }
                    if let Some(limit) = max_total {
                        if total >= limit {
                            break;
                        }
                    }
                    if let Some(ms) = timeout_ms {
                        if start.elapsed().as_millis() > ms {
                            break;
                        }
                    }
                }
                if let Some(limit) = max_total {
                    if total >= limit {
                        break;
                    }
                }
                if let Some(ms) = timeout_ms {
                    if start.elapsed().as_millis() > ms {
                        break;
                    }
                }
            }
            files_scanned = 1;
            files_with = if total > 0 { 1 } else { 0 };
        }
        InputKind::Files(paths) => {
            for p in paths {
                files_scanned += 1;
                match read_file_to_string(&p) {
                    Ok(s) => {
                        let (line_starts, lines) = line_indices(&s);
                        let mut had = false;
                        let mut per_file_count = 0usize;
                        for (lnum, line) in lines.iter().enumerate() {
                            let base = line_starts[lnum];
                            for (mstart, mend, caps) in iter_matches(&pat, line) {
                                total += 1;
                                had = true;
                                per_file_count += 1;
                                let mut m = std::collections::HashMap::new();
                                m.insert("file".into(), Value::String(p.clone()));
                                m.insert("line".into(), Value::String(line.clone()));
                                m.insert("lineNumber".into(), Value::Integer((lnum + 1) as i64));
                                m.insert(
                                    "charRange".into(),
                                    Value::List(vec![
                                        Value::Integer(mstart as i64),
                                        Value::Integer(mend as i64),
                                    ]),
                                );
                                m.insert(
                                    "byteRange".into(),
                                    Value::List(vec![
                                        Value::Integer((base + mstart) as i64),
                                        Value::Integer((base + mend) as i64),
                                    ]),
                                );
                                if !caps.is_empty() {
                                    m.insert(
                                        "captures".into(),
                                        Value::List(
                                            caps.iter()
                                                .map(|(s, e)| {
                                                    Value::List(vec![
                                                        Value::Integer(*s as i64),
                                                        Value::Integer(*e as i64),
                                                    ])
                                                })
                                                .collect(),
                                        ),
                                    );
                                }
                                if ctx.before > 0 || ctx.after > 0 {
                                    m.insert(
                                        "context".into(),
                                        extract_context(&lines, lnum, ctx.before, ctx.after),
                                    );
                                }
                                matches.push(Value::Assoc(m));
                                if let Some(limit) = max_per_file {
                                    if per_file_count >= limit {
                                        break;
                                    }
                                }
                                if let Some(limit) = max_total {
                                    if total >= limit {
                                        break;
                                    }
                                }
                                if let Some(ms) = timeout_ms {
                                    if start.elapsed().as_millis() > ms {
                                        break;
                                    }
                                }
                            }
                            if let Some(limit) = max_total {
                                if total >= limit {
                                    break;
                                }
                            }
                            if let Some(ms) = timeout_ms {
                                if start.elapsed().as_millis() > ms {
                                    break;
                                }
                            }
                        }
                        if had {
                            files_with += 1;
                        }
                    }
                    Err(e) => {
                        if e != "BinaryFileSkipped" {
                            return failure("Text::read", &format!("{}: {}", p, e));
                        }
                    }
                }
                if let Some(limit) = max_total {
                    if total >= limit {
                        break;
                    }
                }
                if let Some(ms) = timeout_ms {
                    if start.elapsed().as_millis() > ms {
                        break;
                    }
                }
            }
        }
    }

    Value::Assoc(
        [
            ("matches".into(), Value::List(matches)),
            (
                "summary".into(),
                ok_summary(files_scanned, files_with, total, start.elapsed().as_millis()),
            ),
        ]
        .into_iter()
        .collect(),
    )
}

fn text_count(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // TextCount(input, pattern, opts?) -> { perFile:[{file,count}], total, durationMs }
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("TextCount".into())), args };
    }
    let start = std::time::Instant::now();
    let input = match parse_input(ev, args[0].clone()) {
        Ok(i) => i,
        Err(e) => return failure("Text::input", &e),
    };
    let (opts_map, _) = parse_opts(ev, args.get(2).cloned());
    let pat = match parse_pattern(ev, args[1].clone(), Some(&opts_map)) {
        Ok(p) => p,
        Err(e) => return failure("Text::pattern", &e),
    };
    let mut per_file: Vec<Value> = Vec::new();
    let mut total = 0usize;
    match input {
        InputKind::Text(s) => {
            let mut c = 0usize;
            for (_, _, _) in iter_matches(&pat, &s) {
                c += 1;
            }
            total = c;
            per_file.push(Value::Assoc(
                [
                    ("file".into(), Value::String(String::new())),
                    ("count".into(), Value::Integer(c as i64)),
                ]
                .into_iter()
                .collect(),
            ));
        }
        InputKind::Files(paths) => {
            for p in paths {
                match read_file_to_string(&p) {
                    Ok(s) => {
                        let mut c = 0;
                        for (_, _, _) in iter_matches(&pat, &s) {
                            c += 1;
                        }
                        total += c;
                        per_file.push(Value::Assoc(
                            [
                                ("file".into(), Value::String(p)),
                                ("count".into(), Value::Integer(c as i64)),
                            ]
                            .into_iter()
                            .collect(),
                        ));
                    }
                    Err(e) => {
                        if e != "BinaryFileSkipped" {
                            return failure("Text::read", &format!("{}: {}", p, e));
                        }
                    }
                }
            }
        }
    }
    Value::Assoc(
        [
            ("perFile".into(), Value::List(per_file)),
            ("total".into(), Value::Integer(total as i64)),
            ("durationMs".into(), Value::Integer(start.elapsed().as_millis() as i64)),
        ]
        .into_iter()
        .collect(),
    )
}

fn text_files_with_match(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // TextFilesWithMatch(input, pattern, opts?) -> { files: [path], durationMs }
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("TextFilesWithMatch".into())), args };
    }
    let start = std::time::Instant::now();
    let input = match parse_input(ev, args[0].clone()) {
        Ok(i) => i,
        Err(e) => return failure("Text::input", &e),
    };
    let (opts_map, _) = parse_opts(ev, args.get(2).cloned());
    let pat = match parse_pattern(ev, args[1].clone(), Some(&opts_map)) {
        Ok(p) => p,
        Err(e) => return failure("Text::pattern", &e),
    };
    let mut out: Vec<Value> = Vec::new();
    match input {
        InputKind::Text(s) => {
            if iter_matches(&pat, &s).next().is_some() {
                out.push(Value::String(String::new()));
            }
        }
        InputKind::Files(paths) => {
            for p in paths {
                match read_file_to_string(&p) {
                    Ok(s) => {
                        if iter_matches(&pat, &s).next().is_some() {
                            out.push(Value::String(p));
                        }
                    }
                    Err(e) => {
                        if e != "BinaryFileSkipped" {
                            return failure("Text::read", &format!("{}: {}", p, e));
                        }
                    }
                }
            }
        }
    }
    Value::Assoc(
        [
            ("files".into(), Value::List(out)),
            ("durationMs".into(), Value::Integer(start.elapsed().as_millis() as i64)),
        ]
        .into_iter()
        .collect(),
    )
}

fn text_lines(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // TextLines(input, pattern, opts?) -> { lines: [{file,lineNumber,text,matches:[{charRange}]}], durationMs }
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("TextLines".into())), args };
    }
    let start = std::time::Instant::now();
    let input = match parse_input(ev, args[0].clone()) {
        Ok(i) => i,
        Err(e) => return failure("Text::input", &e),
    };
    let (opts_map, _) = parse_opts(ev, args.get(2).cloned());
    let pat = match parse_pattern(ev, args[1].clone(), Some(&opts_map)) {
        Ok(p) => p,
        Err(e) => return failure("Text::pattern", &e),
    };
    let mut lines_out: Vec<Value> = Vec::new();
    match input {
        InputKind::Text(s) => collect_lines_for_blob(None, &s, &pat, &mut lines_out),
        InputKind::Files(paths) => {
            for p in paths {
                match read_file_to_string(&p) {
                    Ok(s) => collect_lines_for_blob(Some(&p), &s, &pat, &mut lines_out),
                    Err(e) => {
                        if e != "BinaryFileSkipped" {
                            return failure("Text::read", &format!("{}: {}", p, e));
                        }
                    }
                }
            }
        }
    }
    Value::Assoc(
        [
            ("lines".into(), Value::List(lines_out)),
            ("durationMs".into(), Value::Integer(start.elapsed().as_millis() as i64)),
        ]
        .into_iter()
        .collect(),
    )
}

fn text_replace(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // TextReplace(input, pattern, replacement, {inPlace?:bool, dryRun?:bool, backupExt?:string})
    if args.len() < 3 {
        return Value::Expr { head: Box::new(Value::Symbol("TextReplace".into())), args };
    }
    let start = std::time::Instant::now();
    let input = match parse_input(ev, args[0].clone()) {
        Ok(i) => i,
        Err(e) => return failure("Text::input", &e),
    };
    let (opts_map, _) = parse_opts(ev, args.get(3).cloned());
    let pat = match parse_pattern(ev, args[1].clone(), Some(&opts_map)) {
        Ok(p) => p,
        Err(e) => return failure("Text::pattern", &e),
    };
    let replacement = match ev.eval(args[2].clone()) {
        Value::String(s) => s,
        v => lyra_core::pretty::format_value(&v),
    };
    let in_place = match opts_map.get("inPlace") {
        Some(Value::Boolean(b)) => *b,
        _ => false,
    };
    let dry_run = match opts_map.get("dryRun") {
        Some(Value::Boolean(b)) => *b,
        _ => !in_place,
    };
    let backup_ext = opts_map.get("backupExt").and_then(|v| {
        if let Value::String(s) = v {
            Some(s.clone())
        } else {
            None
        }
    });

    let mut files_changed = 0usize;
    let mut total_repls = 0usize;
    let mut edits: Vec<Value> = Vec::new();

    match input {
        InputKind::Text(s) => {
            let (new_s, n) = replace_all(&pat, &s, &replacement);
            total_repls = n;
            edits.push(Value::Assoc(
                [
                    ("file".into(), Value::String(String::new())),
                    ("replacements".into(), Value::Integer(n as i64)),
                    ("preview".into(), Value::String(new_s)),
                ]
                .into_iter()
                .collect(),
            ));
        }
        InputKind::Files(paths) => {
            for p in paths {
                match read_file_to_string(&p) {
                    Ok(orig) => {
                        let (new_s, n) = replace_all(&pat, &orig, &replacement);
                        if n > 0 {
                            files_changed += 1;
                            total_repls += n;
                        }
                        if !dry_run && n > 0 && in_place {
                            if let Some(ext) = &backup_ext {
                                let _ = std::fs::write(format!("{}{}", p, ext), orig.as_bytes());
                            }
                            if let Err(e) = std::fs::write(&p, new_s.as_bytes()) {
                                return failure("Text::write", &format!("{}: {}", p, e));
                            }
                            edits.push(Value::Assoc(
                                [
                                    ("file".into(), Value::String(p)),
                                    ("replacements".into(), Value::Integer(n as i64)),
                                ]
                                .into_iter()
                                .collect(),
                            ));
                        } else {
                            edits.push(Value::Assoc(
                                [
                                    ("file".into(), Value::String(p)),
                                    ("replacements".into(), Value::Integer(n as i64)),
                                    ("preview".into(), Value::String(new_s)),
                                ]
                                .into_iter()
                                .collect(),
                            ));
                        }
                    }
                    Err(e) => {
                        if e != "BinaryFileSkipped" {
                            return failure("Text::read", &format!("{}: {}", p, e));
                        }
                    }
                }
            }
        }
    }

    Value::Assoc(
        [
            ("edits".into(), Value::List(edits)),
            ("filesChanged".into(), Value::Integer(files_changed as i64)),
            ("totalReplacements".into(), Value::Integer(total_repls as i64)),
            ("durationMs".into(), Value::Integer(start.elapsed().as_millis() as i64)),
        ]
        .into_iter()
        .collect(),
    )
}

fn line_indices(s: &str) -> (Vec<usize>, Vec<String>) {
    let mut starts = vec![0usize];
    let mut lines: Vec<String> = Vec::new();
    let mut last = 0usize;
    for (i, ch) in s.char_indices() {
        if ch == '\n' {
            lines.push(s[last..i].to_string());
            starts.push(i + 1);
            last = i + 1;
        }
    }
    if last <= s.len() {
        lines.push(s[last..].to_string());
    }
    (starts, lines)
}

fn iter_matches<'a>(
    pat: &'a PatternKind,
    line: &'a str,
) -> Box<dyn Iterator<Item = (usize, usize, Vec<(usize, usize)>)> + 'a> {
    match pat {
        PatternKind::Regex(re) => Box::new(re.find_iter(line).map(move |m| {
            let mut caps = Vec::new();
            if let Some(c) = re.captures(line) {
                for i in 1..c.len() {
                    if let Some(m) = c.get(i) {
                        caps.push((m.start(), m.end()));
                    }
                }
            }
            (m.start(), m.end(), caps)
        })),
        #[cfg(feature = "text_multipattern")]
        PatternKind::Multi { ac, .. } => {
            Box::new(ac.find_iter(line.as_bytes()).map(|m| (m.start(), m.end(), Vec::new())))
        }
    }
}

fn extract_context(lines: &[String], idx: usize, before: usize, after: usize) -> Value {
    let start = idx.saturating_sub(before);
    let end = std::cmp::min(lines.len(), idx + 1 + after);
    let mut bm: Vec<Value> = Vec::new();
    let mut am: Vec<Value> = Vec::new();
    for i in start..idx {
        bm.push(Value::String(lines[i].clone()));
    }
    for i in (idx + 1)..end {
        am.push(Value::String(lines[i].clone()));
    }
    Value::Assoc(
        [("before".into(), Value::List(bm)), ("after".into(), Value::List(am))]
            .into_iter()
            .collect(),
    )
}

fn collect_lines_for_blob(file: Option<&str>, s: &str, pat: &PatternKind, out: &mut Vec<Value>) {
    for (lnum, line) in s.lines().enumerate() {
        let mut ms: Vec<Value> = Vec::new();
        for (mstart, mend, _) in iter_matches(pat, line) {
            ms.push(Value::Assoc(
                [(
                    "charRange".into(),
                    Value::List(vec![Value::Integer(mstart as i64), Value::Integer(mend as i64)]),
                )]
                .into_iter()
                .collect(),
            ));
        }
        if !ms.is_empty() {
            out.push(Value::Assoc(
                [
                    (
                        "file".into(),
                        match file {
                            Some(p) => Value::String(p.to_string()),
                            None => Value::String(String::new()),
                        },
                    ),
                    ("lineNumber".into(), Value::Integer((lnum + 1) as i64)),
                    ("text".into(), Value::String(line.to_string())),
                    ("matches".into(), Value::List(ms)),
                ]
                .into_iter()
                .collect(),
            ));
        }
    }
}

fn replace_all(pat: &PatternKind, s: &str, replacement: &str) -> (String, usize) {
    match pat {
        PatternKind::Regex(re) => {
            let mut count = 0usize;
            let out = re.replace_all(s, |_: &regex::Captures| {
                count += 1;
                replacement.to_string()
            });
            (out.into_owned(), count)
        }
        #[cfg(feature = "text_multipattern")]
        PatternKind::Multi { ac, .. } => {
            let mut count = 0usize;
            let mut out = String::with_capacity(s.len());
            let mut last = 0usize;
            for m in ac.find_iter(s.as_bytes()) {
                out.push_str(&s[last..m.start()]);
                out.push_str(replacement);
                last = m.end();
                count += 1;
            }
            out.push_str(&s[last..]);
            (out, count)
        }
    }
}

fn text_detect_encoding(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // TextDetectEncoding({paths|path}) -> { files: [{file, encoding}], durationMs }
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("TextDetectEncoding".into())), args };
    }
    let start = std::time::Instant::now();
    let input = match parse_input(ev, args[0].clone()) {
        Ok(i) => i,
        Err(e) => return failure("Text::input", &e),
    };
    let mut out: Vec<Value> = Vec::new();
    match input {
        InputKind::Text(s) => {
            let enc = detect_bytes_encoding(s.as_bytes());
            out.push(Value::Assoc(
                [
                    ("file".into(), Value::String(String::new())),
                    ("encoding".into(), Value::String(enc)),
                ]
                .into_iter()
                .collect(),
            ));
        }
        InputKind::Files(paths) => {
            for p in paths {
                match std::fs::read(&p) {
                    Ok(bytes) => {
                        let enc = detect_bytes_encoding(&bytes);
                        out.push(Value::Assoc(
                            [
                                ("file".into(), Value::String(p)),
                                ("encoding".into(), Value::String(enc)),
                            ]
                            .into_iter()
                            .collect(),
                        ));
                    }
                    Err(e) => return failure("Text::read", &format!("{}: {}", p, e)),
                }
            }
        }
    }
    Value::Assoc(
        [
            ("files".into(), Value::List(out)),
            ("durationMs".into(), Value::Integer(start.elapsed().as_millis() as i64)),
        ]
        .into_iter()
        .collect(),
    )
}

fn detect_bytes_encoding(bytes: &[u8]) -> String {
    // Try UTF-8 first
    if std::str::from_utf8(bytes).is_ok() {
        return "utf-8".into();
    }
    #[cfg(feature = "text_encoding_detect")]
    {
        let mut det = EncodingDetector::new();
        det.feed(bytes, true);
        let enc = det.guess(None, true);
        return enc.name().to_ascii_lowercase();
    }
    // Fallback
    "unknown".into()
}

fn text_search(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // TextSearch(input, query, opts?) -> { engine: "grep"|"fuzzy"|"index", data: <underlying result>, durationMs }
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("TextSearch".into())), args };
    }
    let start = std::time::Instant::now();
    let raw_input = ev.eval(args[0].clone());
    let query = ev.eval(args[1].clone());
    let opts = args
        .get(2)
        .map(|v| ev.eval(v.clone()))
        .and_then(|v| match v {
            Value::Assoc(m) => Some(m),
            _ => None,
        })
        .unwrap_or_default();

    let task = opts
        .get("task")
        .and_then(|v| if let Value::String(s) = v { Some(s.to_lowercase()) } else { None })
        .unwrap_or_else(|| "find".into());
    // Mode selection: explicit mode wins
    let explicit_mode = opts.get("mode").and_then(|v| {
        if let Value::String(s) = v {
            Some(s.to_lowercase())
        } else {
            None
        }
    });
    let index_path_opt = opts
        .get("indexPath")
        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
        .or_else(|| {
            if let Value::Assoc(m) = &raw_input {
                m.get("indexPath").and_then(|v| {
                    if let Value::String(s) = v {
                        Some(s.clone())
                    } else {
                        None
                    }
                })
            } else {
                None
            }
        });
    let fuzzy_flag = opts
        .get("fuzzy")
        .and_then(|v| if let Value::Boolean(b) = v { Some(*b) } else { None })
        .unwrap_or(false);

    let engine = if task != "find" {
        // Non-find tasks use grep engine
        "grep".into()
    } else if let Some(m) = explicit_mode {
        m
    } else if index_path_opt.is_some() {
        "index".into()
    } else if fuzzy_flag {
        "fuzzy".into()
    } else {
        // Heuristics: if query is assoc with 'regex' or 'literal' or 'multi' => grep; plain string => grep; otherwise fuzzy if enabled and small-ish needle
        match &query {
            Value::Assoc(m) => {
                if m.contains_key("regex")
                    || m.contains_key("Regex")
                    || m.contains_key("literal")
                    || m.contains_key("Literal")
                    || m.contains_key("multi")
                    || m.contains_key("Multi")
                {
                    "grep".into()
                } else {
                    "fuzzy".into()
                }
            }
            Value::String(s) | Value::Symbol(s) => {
                if s.len() <= 64 {
                    "grep".into()
                } else {
                    "fuzzy".into()
                }
            }
            _ => "grep".into(),
        }
    };

    let data = match engine.as_str() {
        "index" => {
            let idx = index_path_opt.unwrap();
            let q = match &query {
                Value::Assoc(m) => m
                    .get("q")
                    .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
                    .unwrap_or_default(),
                Value::String(s) => s.clone(),
                Value::Symbol(s) => s.clone(),
                _ => String::new(),
            };
            ev.eval(Value::expr(
                Value::Symbol("IndexSearch".into()),
                vec![Value::String(idx), Value::String(q)],
            ))
        }
        "fuzzy" => {
            // Use text_fuzzy if available; decide Text vs Files
            match &raw_input {
                Value::Assoc(m)
                    if m.contains_key("path")
                        || m.contains_key("paths")
                        || m.contains_key("Path")
                        || m.contains_key("Paths")
                        || m.contains_key("glob")
                        || m.contains_key("globs") =>
                {
                    let needle = match &query {
                        Value::Assoc(ma) => ma
                            .get("needle")
                            .and_then(
                                |v| if let Value::String(s) = v { Some(s.clone()) } else { None },
                            )
                            .unwrap_or_else(|| String::new()),
                        Value::String(s) => s.clone(),
                        Value::Symbol(s) => s.clone(),
                        _ => String::new(),
                    };
                    ev.eval(Value::expr(
                        Value::Symbol("FuzzyFindInFiles".into()),
                        vec![Value::Assoc(m.clone()), Value::String(needle)],
                    ))
                }
                Value::Assoc(m) if m.contains_key("text") || m.contains_key("Text") => {
                    let text = m.get("text").or_else(|| m.get("Text")).cloned().unwrap();
                    let needle = match &query {
                        Value::Assoc(ma) => ma
                            .get("needle")
                            .and_then(
                                |v| if let Value::String(s) = v { Some(s.clone()) } else { None },
                            )
                            .unwrap_or_else(|| String::new()),
                        Value::String(s) => s.clone(),
                        Value::Symbol(s) => s.clone(),
                        _ => String::new(),
                    };
                    ev.eval(Value::expr(
                        Value::Symbol("FuzzyFindInText".into()),
                        vec![text, Value::String(needle)],
                    ))
                }
                Value::String(s) => {
                    let needle = match &query {
                        Value::Assoc(ma) => ma
                            .get("needle")
                            .and_then(
                                |v| if let Value::String(s) = v { Some(s.clone()) } else { None },
                            )
                            .unwrap_or_else(|| String::new()),
                        Value::String(ss) => ss.clone(),
                        Value::Symbol(ss) => ss.clone(),
                        _ => String::new(),
                    };
                    ev.eval(Value::expr(
                        Value::Symbol("FuzzyFindInText".into()),
                        vec![Value::String(s.clone()), Value::String(needle)],
                    ))
                }
                other => failure(
                    "TextSearch::fuzzy",
                    &format!(
                        "Unsupported input for fuzzy: {}",
                        lyra_core::pretty::format_value(other)
                    ),
                ),
            }
        }
        _ => {
            // Grep-backed tasks
            let pattern = match &query {
                Value::Assoc(_) => query.clone(),
                Value::String(s) | Value::Symbol(s) => Value::Assoc(
                    [("literal".into(), Value::String(s.clone()))].into_iter().collect(),
                ),
                _ => Value::Assoc([].into_iter().collect()),
            };
            match task.as_str() {
                "count" => ev.eval(Value::expr(
                    Value::Symbol("TextCount".into()),
                    vec![raw_input.clone(), pattern, Value::Assoc(opts.clone())],
                )),
                "lines" => ev.eval(Value::expr(
                    Value::Symbol("TextLines".into()),
                    vec![raw_input.clone(), pattern, Value::Assoc(opts.clone())],
                )),
                "files" | "filesWithMatch" => ev.eval(Value::expr(
                    Value::Symbol("TextFilesWithMatch".into()),
                    vec![raw_input.clone(), pattern, Value::Assoc(opts.clone())],
                )),
                "replace" => {
                    let repl = if let Value::Assoc(m) = &query {
                        m.get("replacement").and_then(|v| {
                            if let Value::String(s) = v {
                                Some(s.clone())
                            } else {
                                None
                            }
                        })
                    } else {
                        None
                    }
                    .or_else(|| {
                        opts.get("replacement").and_then(|v| {
                            if let Value::String(s) = v {
                                Some(s.clone())
                            } else {
                                None
                            }
                        })
                    })
                    .unwrap_or_default();
                    ev.eval(Value::expr(
                        Value::Symbol("TextReplace".into()),
                        vec![
                            raw_input.clone(),
                            pattern,
                            Value::String(repl),
                            Value::Assoc(opts.clone()),
                        ],
                    ))
                }
                _ => ev.eval(Value::expr(
                    Value::Symbol("TextFind".into()),
                    vec![raw_input.clone(), pattern, Value::Assoc(opts.clone())],
                )),
            }
        }
    };
    Value::Assoc(
        [
            ("engine".into(), Value::String(engine)),
            ("data".into(), data),
            ("durationMs".into(), Value::Integer(start.elapsed().as_millis() as i64)),
        ]
        .into_iter()
        .collect(),
    )
}


// ---------------- Stemming (Porter-like for English; pass-through otherwise) ----------------

fn is_vowel(chars: &[char], i: usize) -> bool {
    let c = chars[i].to_ascii_lowercase();
    if "aeiou".contains(c) { return true; }
    if c == 'y' {
        if i == 0 { return false; }
        return !is_vowel(chars, i-1);
    }
    false
}

fn has_vowel(chars: &[char], end: usize) -> bool {
    for i in 0..end { if is_vowel(chars, i) { return true; } }
    false
}

fn ends_with(s: &str, suf: &str) -> bool { s.len() >= suf.len() && s[s.len()-suf.len()..].eq_ignore_ascii_case(suf) }

fn porter_en_stem(token: &str) -> String {
    let mut s = token.to_ascii_lowercase();
    if s.len() <= 2 { return s; }
    let mut chars: Vec<char> = s.chars().collect();
    // Step 1a
    if ends_with(&s, "sses") { s.truncate(s.len()-2); return s; } // sses -> ss
    if ends_with(&s, "ies") { s.truncate(s.len()-3); s.push('i'); return s; }
    if ends_with(&s, "ss") { return s; }
    if ends_with(&s, "s") { s.truncate(s.len()-1); }
    // Step 1b
    chars = s.chars().collect();
    if ends_with(&s, "eed") {
        s.truncate(s.len()-1); // eed -> ee (approx; ignoring m>0)
    } else if ends_with(&s, "ed") {
        let stem = &chars[..chars.len()-2];
        if has_vowel(stem, stem.len()) { s.truncate(s.len()-2); }
    } else if ends_with(&s, "ing") {
        let stem = &chars[..chars.len()-3];
        if has_vowel(stem, stem.len()) { s.truncate(s.len()-3); }
    }
    // After removing ed/ing: adjust endings
    if ends_with(&s, "at") { s.push('e'); } // at -> ate
    else if ends_with(&s, "bl") { s.push('e'); } // bl -> ble
    else if ends_with(&s, "iz") { s.push('e'); } // iz -> ize
    else {
        // remove double consonant (bb, dd, ff, gg, mm, nn, pp, rr, tt)
        let cs: Vec<char> = s.chars().collect();
        if cs.len() >= 2 {
            let a = cs[cs.len()-1]; let b = cs[cs.len()-2];
            if a==b && "bdfgmnprt".contains(a) { s.truncate(s.len()-1); }
        }
        // NOTE: Skip cvc + 'e' insertion to avoid over-stemming like 'running' -> 'rune'.
    }
    // Step 1c: y->i if vowel in stem
    if s.ends_with('y') {
        let cs: Vec<char> = s.chars().collect();
        if has_vowel(&cs, cs.len()-1) { s.pop(); s.push('i'); }
    }
    s
}

fn stem(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Stem".into())), args }; }
    let input = ev.eval(args[0].clone());
    let opts = if args.len()>1 { ev.eval(args[1].clone()) } else { Value::Assoc(HashMap::new()) };
    let (lang, algorithm) = if let Value::Assoc(m) = opts {
        let lang = m.get("language").and_then(|v| if let Value::String(s)|Value::Symbol(s)=v { Some(s.to_ascii_lowercase()) } else { None }).unwrap_or_else(|| "en".into());
        let algorithm = m.get("algorithm").and_then(|v| if let Value::String(s)|Value::Symbol(s)=v { Some(s.to_ascii_lowercase()) } else { None }).unwrap_or_else(|| "porter".into());
        (lang, algorithm)
    } else { (String::from("en"), String::from("porter")) };
    let stem_token = |t: &str| -> String {
        match lang.as_str() {
            "en" => match algorithm.as_str() { _ => porter_en_stem(t) },
            _ => t.to_string(),
        }
    };
    match input {
        Value::List(vs) => Value::List(vs.into_iter().map(|v| match v { Value::String(s)|Value::Symbol(s)=> Value::String(stem_token(&s)), other => other }).collect()),
        Value::String(s) => {
            let toks = simple_word_tokens(&s, false, true, true);
            Value::List(toks.into_iter().map(|(t,_,_)| Value::String(stem_token(&t))).collect())
        }
        other => Value::List(vec![other]),
    }
}

// ---------------- Lemmatization (rule-based English) ----------------

fn lemmatize_en(token: &str) -> String {
    let mut s = token.to_ascii_lowercase();
    if s.len() <= 2 { return s; }
    // Exceptions (irregulars)
    let mut exceptions: HashMap<&str,&str> = HashMap::new();
    for (a,b) in [
        ("went","go"),("gone","go"),("ran","run"),("eaten","eat"),("ate","eat"),("done","do"),("did","do"),("saw","see"),
        ("was","be"),("were","be"),("is","be"),("are","be"),("am","be"),("been","be"),("being","be"),
        ("had","have"),("has","have"),("bought","buy"),("brought","bring"),("better","good"),("best","good"),
    ] { exceptions.insert(a,b); }
    if let Some(&base) = exceptions.get(s.as_str()) { return base.to_string(); }
    // Adverbs -> adjective
    if s.ends_with("ly") && s.len()>4 { s.truncate(s.len()-2); return s; }
    // Past participles / past tense
    if s.ends_with("ied") && s.len()>4 { s.truncate(s.len()-3); s.push('y'); return s; }
    if s.ends_with("ed") && s.len()>3 { s.truncate(s.len()-2); return s; }
    // Gerund
    if s.ends_with("ing") && s.len()>4 {
        s.truncate(s.len()-3);
        // Collapse doubled consonant: running -> run
        let cs: Vec<char> = s.chars().collect();
        if cs.len()>=2 { let a=cs[cs.len()-1]; let b=cs[cs.len()-2]; if a==b && "bdfgmnprt".contains(a) { s.truncate(s.len()-1); } }
        return s;
    }
    // Plurals
    if s.ends_with("ies") && s.len()>4 { s.truncate(s.len()-3); s.push('y'); return s; }
    if s.ends_with("es") && s.len()>3 { s.truncate(s.len()-2); return s; }
    if s.ends_with('s') && !s.ends_with("ss") { s.truncate(s.len()-1); return s; }
    s
}

fn lemmatize(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Lemmatize".into())), args }; }
    let input = ev.eval(args[0].clone());
    let opts = normalize_opts_map(ev, args.get(1).cloned());
    let lang = opts.get("language").and_then(|v| if let Value::String(s)|Value::Symbol(s)=v { Some(s.to_ascii_lowercase()) } else { None }).unwrap_or_else(|| "en".into());
    let lemmatize_token = |t: &str| -> String { match lang.as_str() { "en" => lemmatize_en(t), _ => t.to_string() } };
    match input {
        Value::List(vs) => Value::List(vs.into_iter().map(|v| match v { Value::String(s)|Value::Symbol(s)=> Value::String(lemmatize_token(&s)), other => other }).collect()),
        Value::String(s) => {
            let toks = simple_word_tokens(&s, false, true, true);
            Value::List(toks.into_iter().map(|(t,_,_)| Value::String(lemmatize_token(&t))).collect())
        }
        other => Value::List(vec![other]),
    }
}
