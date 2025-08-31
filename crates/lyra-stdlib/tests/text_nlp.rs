use lyra_runtime::Evaluator;

fn eval_str(code: &str) -> lyra_core::value::Value {
    let mut ev = Evaluator::new();
    lyra_stdlib::register_all(&mut ev);
    let mut p = lyra_parser::Parser::from_source(code);
    let exprs = p.parse_all().unwrap();
    let mut out = lyra_core::value::Value::Symbol("Null".into());
    for e in exprs { out = ev.eval(e); }
    out
}

#[test]
fn tokenize_basic() {
    let v = eval_str("Tokenize[\"The quick brown fox.\"]");
    match v { lyra_core::value::Value::List(vs) => {
        assert!(vs.len()>=4);
    }, other => panic!("unexpected: {}", lyra_core::pretty::format_value(&other)) }
}

#[test]
fn ngrams_bigram() {
    let v = eval_str("Ngrams[{\"a\",\"b\",\"c\"}, <|\"n\"->2|>]");
    match v { lyra_core::value::Value::List(vs) => { assert_eq!(vs.len(), 2); }, other => panic!("unexpected: {}", lyra_core::pretty::format_value(&other)) }
}

#[test]
fn tfidf_shape() {
    let v = eval_str("TfIdf[{\"one two two\", \"two three\"}]");
    match v { lyra_core::value::Value::Assoc(m) => {
        assert!(m.get("vocab").is_some());
        assert!(m.get("idf").is_some());
        assert!(m.get("matrix").is_some());
    }, other => panic!("unexpected: {}", lyra_core::pretty::format_value(&other)) }
}

#[test]
fn remove_stopwords_spanish() {
    let v = eval_str("RemoveStopwords[\"el zorro y la zorra\", <|\"language\"->\"es\"|>]");
    match v { lyra_core::value::Value::List(vs) => {
        let s: Vec<String> = vs.into_iter().filter_map(|x| if let lyra_core::value::Value::String(t)=x { Some(t) } else { None }).collect();
        assert!(s.contains(&"zorro".to_string()) || s.contains(&"zorra".to_string()));
        assert!(!s.contains(&"el".to_string()));
        assert!(!s.contains(&"la".to_string()));
        assert!(!s.contains(&"y".to_string()));
    }, other => panic!("unexpected: {}", lyra_core::pretty::format_value(&other)) }
}


#[test]
fn stem_running() {
    let v = eval_str("Stem[\"running\"]");
    match v { lyra_core::value::Value::List(vs) => {
        let s: Vec<String> = vs.into_iter().filter_map(|x| if let lyra_core::value::Value::String(t)=x { Some(t) } else { None }).collect();
        assert_eq!(s, vec!["run".to_string()]);
    }, other => panic!("unexpected: {}", lyra_core::pretty::format_value(&other)) }
}


#[test]
fn lemmatize_examples() {
    // Adverb -> adjective
    let v = eval_str("Lemmatize[\"powerfully\"]");
    match v { lyra_core::value::Value::List(vs) => {
        let s: Vec<String> = vs.into_iter().filter_map(|x| if let lyra_core::value::Value::String(t)=x { Some(t) } else { None }).collect();
        assert_eq!(s, vec!["powerful".to_string()]);
    }, other => panic!("unexpected: {}", lyra_core::pretty::format_value(&other)) }
    // Mix of verb/noun plural
    let v2 = eval_str("Lemmatize[{\"running\",\"studies\",\"cars\"}]");
    match v2 { lyra_core::value::Value::List(vs) => {
        let s: Vec<String> = vs.into_iter().filter_map(|x| if let lyra_core::value::Value::String(t)=x { Some(t) } else { None }).collect();
        assert_eq!(s, vec!["run".to_string(), "study".to_string(), "car".to_string()]);
    }, other => panic!("unexpected: {}", lyra_core::pretty::format_value(&other)) }
}
