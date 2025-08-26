use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_runtime::attrs::Attributes;
use std::collections::HashMap;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

pub fn register_rag(ev: &mut Evaluator) {
    ev.register("RAGChunk", rag_chunk as NativeFn, Attributes::empty());
    ev.register("RAGIndex", rag_index as NativeFn, Attributes::empty());
    ev.register("RAGRetrieve", rag_retrieve as NativeFn, Attributes::empty());
    ev.register("RAGAssembleContext", rag_assemble_context as NativeFn, Attributes::empty());
    ev.register("RAGAnswer", rag_answer as NativeFn, Attributes::empty());
    ev.register("HybridSearch", hybrid_search as NativeFn, Attributes::empty());
    ev.register("Cite", cite as NativeFn, Attributes::empty());
    ev.register("Citations", citations as NativeFn, Attributes::empty());
}

pub fn register_rag_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str)->bool) {
    super::register_if(ev, pred, "RAGChunk", rag_chunk as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "RAGIndex", rag_index as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "RAGRetrieve", rag_retrieve as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "RAGAssembleContext", rag_assemble_context as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "RAGAnswer", rag_answer as NativeFn, Attributes::empty());
}

fn as_str(v: &Value) -> Option<String> { match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None } }
fn assoc_str(m: &HashMap<String, Value>, k: &str) -> Option<String> { m.get(k).and_then(as_str) }

fn rag_chunk(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // RAGChunk[text, <|Size->400, Overlap->40|>]
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("RAGChunk".into())), args } }
    let text = as_str(&args[0]).unwrap_or_default();
    let mut size: usize = 400; let mut overlap: usize = 40;
    if let Some(Value::Assoc(m)) = args.get(1) {
        if let Some(Value::Integer(n)) = m.get("Size") { size = (*n).max(1) as usize; }
        if let Some(Value::Integer(n)) = m.get("Overlap") { overlap = (*n).max(0) as usize; }
    }
    let mut chunks: Vec<Value> = Vec::new();
    let mut i = 0usize;
    let bytes = text.as_bytes();
    while i < bytes.len() {
        let end = (i+size).min(bytes.len());
        let s = String::from_utf8_lossy(&bytes[i..end]).to_string();
        chunks.push(Value::Assoc(HashMap::from([
            ("text".into(), Value::String(s)),
            ("ofs".into(), Value::Integer(i as i64)),
        ])));
        if end==bytes.len() { break; }
        if size > overlap { i += size - overlap; } else { i += size; }
    }
    Value::List(chunks)
}

fn rag_index(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // RAGIndex[store, docs, <|EmbedModel->Model[...], Chunk-><|...|>|>]
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("RAGIndex".into())), args } }
    let store = args[0].clone();
    let docs = args[1].clone();
    let mut chunk_opts: Value = Value::Assoc(HashMap::new());
    if let Some(Value::Assoc(m)) = args.get(2) { if let Some(co) = m.get("Chunk") { chunk_opts = co.clone(); } }
    let mut upserts: Vec<Value> = Vec::new();
    if let Value::List(list) = docs {
        for d in list {
            if let Value::Assoc(m) = d {
                let id = assoc_str(&m, "id").unwrap_or_else(|| "doc".into());
                let text = assoc_str(&m, "text").unwrap_or_default();
                let meta = m.get("meta").cloned().unwrap_or(Value::Assoc(HashMap::new()));
                let chunks = rag_chunk(ev, vec![Value::String(text), chunk_opts.clone()]);
                if let Value::List(chs) = chunks {
                    for (idx, ch) in chs.into_iter().enumerate() {
                        if let Value::Assoc(cm) = ch {
                            let cid = format!("{}#{}", id, idx);
                            let ctext = assoc_str(&cm, "text").unwrap_or_default();
                            let emb = super::model::embed(ev, vec![Value::Assoc(HashMap::from([(String::from("Text"), Value::String(ctext.clone()))]))]);
                            if let Value::List(vecs) = emb {
                                if let Some(Value::List(v)) = vecs.get(0) {
                                    upserts.push(Value::Assoc(HashMap::from([
                                        ("id".into(), Value::String(cid)),
                                        ("vector".into(), Value::List(v.clone())),
                                        ("meta".into(), Value::Assoc(HashMap::from([
                                            ("doc_id".into(), Value::String(id.clone())),
                                            ("text".into(), Value::String(ctext)),
                                        ]))),
                                    ])));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    super::vector::vs_upsert(ev, vec![store, Value::List(upserts)])
}

fn rag_retrieve(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // RAGRetrieve[store, query, <|K->n, Filter->...|>]
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("RAGRetrieve".into())), args } }
    super::vector::vs_query(ev, args)
}

fn rag_assemble_context(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // RAGAssembleContext[{matches...}, <|MaxChars->2000|>]
    let matches = args.get(0).cloned().unwrap_or(Value::List(vec![]));
    let mut max_chars: usize = 2000;
    if let Some(Value::Assoc(m)) = args.get(1) { if let Some(Value::Integer(n)) = m.get("MaxChars") { max_chars = *n as usize; } }
    let mut out = String::new();
    if let Value::List(mm) = matches {
        for m in mm {
            if let Value::Assoc(am) = m {
                if let Some(Value::Assoc(meta)) = am.get("meta") { if let Some(Value::String(t)) = meta.get("text") {
                    if out.len() + t.len() + 2 > max_chars { break; }
                    if !out.is_empty() { out.push_str("\n\n"); }
                    out.push_str(t);
                }}
            }
        }
    }
    Value::String(out)
}

fn rag_answer(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // RAGAnswer[store, query, <|Model->Model[...], K->n, Cite->True|>]
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("RAGAnswer".into())), args } }
    let store = args[0].clone();
    let query = args[1].clone();
    let mut k: i64 = 5; let mut cite = true;
    if let Some(Value::Assoc(m)) = args.get(2) { if let Some(Value::Integer(n)) = m.get("K") { k = *n; } if let Some(Value::Boolean(b)) = m.get("Cite") { cite = *b; } }
    let matches = rag_retrieve(ev, vec![store, query.clone(), Value::Assoc(HashMap::from([(String::from("K"), Value::Integer(k))]))]);
    let context = rag_assemble_context(ev, vec![matches.clone(), Value::Assoc(HashMap::from([(String::from("MaxChars"), Value::Integer(2000))]))]);
    // Compose a simple prompt and call Chat
    let messages = Value::List(vec![
        Value::Assoc(HashMap::from([(String::from("role"), Value::String(String::from("system"))), (String::from("content"), Value::String(String::from("You are a helpful assistant.")))])),
        Value::Assoc(HashMap::from([(String::from("role"), Value::String(String::from("user"))), (String::from("content"), Value::String(format!("Context:\n{}\n\nQuestion: {}", match &context { Value::String(s)=>s.clone(), _=>String::new() }, match &query { Value::String(s)=>s.clone(), Value::Symbol(s)=>s.clone(), _=>String::new() })))])),
    ]);
    let reply = super::model::chat(ev, vec![Value::Assoc(HashMap::from([(String::from("Messages"), messages)]))]);
    let cites = citations(ev, vec![matches.clone()]);
    Value::Assoc(HashMap::from([
        ("answer".into(), reply),
        ("matches".into(), matches),
        ("citations".into(), cites),
    ]))
}



fn hybrid_search(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // HybridSearch[store, query, <|K->n, Alpha->a|>]
    let mut opts = std::collections::HashMap::new();
    if let Some(Value::Assoc(m)) = args.get(2) { opts = m.clone(); }
    opts.insert("Hybrid".into(), Value::String("true".into()));
    let mut a2 = Vec::new(); a2.push(args.get(0).cloned().unwrap_or(Value::String("mem".into()))); a2.push(args.get(1).cloned().unwrap_or(Value::String(String::new()))); a2.push(Value::Assoc(opts));
    super::vector::vs_query(ev, a2)
}

fn cite(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Cite[matches_or_answer, <|Style->"markdown", WithScores->False, Max->n|>]
    let c = citations(_ev, vec![args.get(0).cloned().unwrap_or(Value::List(vec![]))]);
    let mut style = String::from("markdown");
    let mut with_scores = false;
    let mut max: Option<usize> = None;
    if let Some(Value::Assoc(m)) = args.get(1) {
        if let Some(Value::String(s)) = m.get("Style") { style = s.clone(); }
        if let Some(Value::Boolean(b)) = m.get("WithScores") { with_scores = *b; }
        if let Some(Value::String(s)) = m.get("WithScores") { let ls = s.to_lowercase(); if ls=="true"||ls=="on"||ls=="1" { with_scores = true; } }
        if let Some(Value::Integer(n)) = m.get("Max") { if *n>0 { max = Some(*n as usize); } }
    }
    let list = match &c { Value::List(vs)=>vs.clone(), _=>vec![] };
    let iter = list.into_iter().take(max.unwrap_or(usize::MAX));
    match style.as_str() {
        "markdown" => {
            let mut out = String::new();
            for v in iter { if let Value::Assoc(m) = v {
                let id = m.get("id").and_then(|v| if let Value::String(s)|Value::Symbol(s)=v { Some(s.clone()) } else { None }).unwrap_or_else(|| String::from("doc"));
                let text = m.get("text").and_then(|v| if let Value::String(s)=v { Some(s.clone()) } else { None }).unwrap_or_default();
                let score_s = if with_scores { match m.get("score") { Some(Value::Real(r))=> format!(" (score: {:.3})", r), Some(Value::Integer(i))=> format!(" (score: {})", i), _=> String::new() } } else { String::new() };
                out.push_str(&format!("- [{}] {}{}
", id, text, score_s));
            }}
            Value::String(out)
        }
        "json" => c,
        _ => {
            let mut out = String::new();
            for v in iter { if let Value::Assoc(m) = v {
                let id = m.get("id").and_then(|v| if let Value::String(s)|Value::Symbol(s)=v { Some(s.clone()) } else { None }).unwrap_or_else(|| String::from("doc"));
                let text = m.get("text").and_then(|v| if let Value::String(s)=v { Some(s.clone()) } else { None }).unwrap_or_default();
                let score_s = if with_scores { match m.get("score") { Some(Value::Real(r))=> format!("	{:.3}", r), Some(Value::Integer(i))=> format!("	{}", i), _=> String::new() } } else { String::new() };
                out.push_str(&format!("{}	{}{}
", id, text, score_s));
            }}
            Value::String(out)
        }
    }
}

fn citations(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Citations[matches_or_answer] -> normalized list of <|id,text,score?,url?|>
    let mut matches = Vec::new();
    if let Some(v) = args.get(0) {
        match v {
            Value::Assoc(m) => {
                if let Some(Value::List(ms)) = m.get("matches") { matches = ms.clone(); }
                else if let Some(Value::List(cs)) = m.get("citations") { matches = cs.clone(); }
            }
            Value::List(ms) => matches = ms.clone(),
            _ => {}
        }
    }
    let mut out: Vec<Value> = Vec::new();
    for m in matches.into_iter() {
        if let Value::Assoc(am) = m {
            let id = am.get("id").and_then(|v| if let Value::String(s)|Value::Symbol(s)=v { Some(s.clone()) } else { None }).unwrap_or_else(|| String::from("doc"));
            let mut text = String::new();
            let mut score: Option<Value> = None;
            let mut url: Option<String> = None;
            if let Some(Value::Real(r)) = am.get("score") { score = Some(Value::Real(*r)); }
            if let Some(Value::Integer(i)) = am.get("score") { score = Some(Value::Integer(*i)); }
            match am.get("meta") {
                Some(Value::Assoc(mm)) => {
                    if let Some(Value::String(s)) = mm.get("text") { text = s.clone(); }
                    if let Some(Value::String(u)) = mm.get("url") { url = Some(u.clone()); }
                }
                Some(Value::String(s)) => { text = s.clone(); }
                _ => {}
            }
            let mut m2 = HashMap::new();
            m2.insert("id".into(), Value::String(id));
            m2.insert("text".into(), Value::String(text));
            if let Some(sc) = score { m2.insert("score".into(), sc); }
            if let Some(u) = url { m2.insert("url".into(), Value::String(u)); }
            out.push(Value::Assoc(m2));
        }
    }
    Value::List(out)
}

