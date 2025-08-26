use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_runtime::attrs::Attributes;
use std::collections::HashMap;
#[cfg(feature = "net_https")] use reqwest::blocking::Client;
#[cfg(feature = "net_https")] use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

fn str_of(v: &Value) -> Option<String> { match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None } }

pub fn register_model(ev: &mut Evaluator) {
    ev.register("Model", model as NativeFn, Attributes::empty());
    ev.register("ModelsList", models_list as NativeFn, Attributes::empty());
    ev.register("Chat", chat as NativeFn, Attributes::HOLD_ALL);
    ev.register("Complete", complete as NativeFn, Attributes::HOLD_ALL);
    ev.register("Embed", embed as NativeFn, Attributes::empty());
}

pub fn register_model_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str)->bool) {
    super::register_if(ev, pred, "Model", model as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "ModelsList", models_list as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "Chat", chat as NativeFn, Attributes::HOLD_ALL);
    super::register_if(ev, pred, "Complete", complete as NativeFn, Attributes::HOLD_ALL);
    super::register_if(ev, pred, "Embed", embed as NativeFn, Attributes::empty());
}

fn model(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Model["mock:echo"] or Model[<| "Id"->..., "Provider"->... |>]
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Model".into())), args } }
    match &args[0] {
        Value::String(s)|Value::Symbol(s) => {
            let mut m = HashMap::new();
            m.insert("Id".into(), Value::String(s.clone()));
            let provider = if let Some((p,_)) = s.split_once(":") { p.to_string() } else { "mock".to_string() };
            m.insert("Provider".into(), Value::String(provider));
            Value::Assoc(m)
        }
        Value::Assoc(am) => Value::Assoc(am.clone()),
        _ => Value::Assoc(HashMap::new()),
    }
}

fn models_list(_ev: &mut Evaluator, _args: Vec<Value>) -> Value {
    Value::List(vec![
        Value::Assoc(HashMap::from([
            ("Id".into(), Value::String("mock:echo".into())),
            ("Provider".into(), Value::String("mock".into())),
            ("Modality".into(), Value::String("text".into())),
        ]))
    ])
}

fn extract_messages(m: &Value) -> Vec<(String,String)> {
    match m {
        Value::List(items) => items.iter().filter_map(|it| {
            if let Value::Assoc(a) = it { let role = a.get("role").and_then(str_of)?; let content = a.get("content").and_then(str_of)?; Some((role, content)) } else { None }
        }).collect(),
        _ => vec![],
    }
}

pub fn chat(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Chat[Model["mock:echo"], <|"Messages"->[{role,content}...], "Tools"->..., "Stream"->True|>]
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Chat".into())), args } }
    let mut model_id: String = "mock:echo".into();
    let mut opts = HashMap::new();
    for a in &args {
        let av = ev.eval(a.clone());
        match av {
            Value::Assoc(m) => { for (k,v) in m { opts.insert(k.clone(), v.clone()); } }
            Value::String(s)|Value::Symbol(s) => { model_id = s; }
            _ => {}
        }
    }
    // Evaluate Messages if it's a symbol/expression
    let msgs_val = match opts.get("Messages") { Some(v) => ev.eval(v.clone()), None => Value::List(vec![]) };
    let messages = extract_messages(&msgs_val);
    // Provider routing: if ModelsMode=="auto" and model is openai:* and OPENAI_API_KEY set, try provider; else mock
    let mode = ev.get_env("ModelsMode").and_then(|v| if let Value::String(s)|Value::Symbol(s)=v { Some(s) } else { None }).unwrap_or_else(|| "mock".into());
    if mode=="auto" && model_id.starts_with("openai:") {
        if let Some(res) = chat_openai(&model_id, &messages) { return res; }
    }
    // Mock behavior: echo last user content or compose a trivial response
    let mut response = String::new();
    if let Some((_, last)) = messages.iter().rev().find(|(r,_)| r=="user") { response = format!("Echo: {}", last); }
    if response.is_empty() { response = "(ok)".into(); }

    // Optionally simulate a tool call if Tools->... and content looks like command (stub)
    // Record a minimal span for observability if trace module present: Span["model:Chat", Attrs->...]
    let span_start = Value::Expr { head: Box::new(Value::Symbol("Span".into())), args: vec![
        Value::String("model:Chat".into()),
        Value::Assoc(HashMap::from([
            ("Model".into(), Value::String(model_id.clone())),
            ("TokensIn".into(), Value::Integer(messages.iter().map(|(_,c)| c.len() as i64).sum())),
        ])),
    ]};
    let _ = ev.eval(span_start);

    // Stream tokens if requested
    let stream = match opts.get("Stream") {
        Some(Value::Boolean(b)) => *b,
        Some(Value::String(s)) | Some(Value::Symbol(s)) => {
            let ls = s.to_lowercase(); ls=="true" || ls=="on" || ls=="1"
        }
        _ => false
    };
    if stream {
        let tokens: Vec<&str> = response.split_whitespace().collect();
        for t in tokens {
            let _ = ev.eval(Value::Expr { head: Box::new(Value::Symbol("Puts".into())), args: vec![ Value::String(t.to_string()) ] });
        }
    }

    let out = Value::Assoc(HashMap::from([
        ("role".into(), Value::String("assistant".into())),
        ("content".into(), Value::String(response.clone())),
        ("model".into(), Value::String(model_id.clone())),
    ]));

    // Update metrics if available
    let _ = ev.eval(Value::Expr { head: Box::new(Value::Symbol("CostAdd".into())), args: vec![Value::Real(0.0)] });
    let _ = ev.eval(Value::Expr { head: Box::new(Value::Symbol("SpanEnd".into())), args: vec![] });
    out
}

fn complete(ev: &mut Evaluator, mut args: Vec<Value>) -> Value {
    // Complete[Model[...], Prompt->"..."] or Complete[Prompt]
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Complete".into())), args } }
    let mut prompt = String::new();
    // Accept Prompt in assoc or as direct string
    if args.len()==1 { if let Some(s) = str_of(&args[0]) { prompt = s; } }
    for a in &args { if let Value::Assoc(m) = a { if let Some(Value::String(s)) = m.get("Prompt") { prompt = s.clone(); } } }
    // Simple mock: return the prompt with suffix
    let _ = ev.eval(Value::Expr { head: Box::new(Value::Symbol("Span".into())), args: vec![ Value::String("model:Complete".into()) ] });
    let out = Value::String(format!("{}", prompt));
    let _ = ev.eval(Value::Expr { head: Box::new(Value::Symbol("SpanEnd".into())), args: vec![] });
    out
}

pub fn embed(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Embed[Text->list|string] (auto mode may call providers)
    let mut texts: Vec<String> = Vec::new();
    for a in &args {
        match a {
            Value::Assoc(m) => {
                if let Some(v) = m.get("Text") {
                    match v {
                        Value::String(s) => texts.push(s.clone()),
                        Value::List(items) => {
                            for it in items { if let Value::String(s) = it { texts.push(s.clone()); } }
                        }
                        _ => {}
                    }
                }
            }
            Value::String(s) => texts.push(s.clone()),
            _ => {}
        }
    }
    // Provider routing (auto): use OpenAI embeddings if key present
    let mode = _ev.get_env("ModelsMode").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s), _=>None }).unwrap_or_else(|| "mock".into());
    #[cfg(feature = "net_https")]
    if mode=="auto" { if let Some(v) = openai_embed(&texts) { return v; } }

    // Fallback mock: simple numeric vectors: length and hash-like indicator
    Value::List(texts.into_iter().map(|t| {
        let n = t.len() as i64;
        Value::List(vec![ Value::Real(n as f64), Value::Real((n % 7) as f64), Value::Real(((n*3) % 11) as f64) ])
    }).collect())
}

#[cfg(feature = "net_https")]
fn chat_openai(model_id: &str, messages: &Vec<(String,String)>) -> Option<Value> {
    let api_key = std::env::var("OPENAI_API_KEY").ok()?;
    let model = model_id.split(':').nth(1).unwrap_or("gpt-4o");
    let client = Client::new();
    let ms: Vec<serde_json::Value> = messages.iter().map(|(r,c)| serde_json::json!({"role": r, "content": c})).collect();
    let body = serde_json::json!({"model": model, "messages": ms});
    let resp = client.post("https://api.openai.com/v1/chat/completions")
        .header(AUTHORIZATION, format!("Bearer {}", api_key))
        .header(CONTENT_TYPE, "application/json")
        .json(&body)
        .send().ok()?;
    let js: serde_json::Value = resp.json().ok()?;
    let content = js["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string();
    Some(Value::Assoc(HashMap::from([
        ("role".into(), Value::String("assistant".into())),
        ("content".into(), Value::String(content)),
        ("model".into(), Value::String(model_id.to_string())),
    ])))
}

#[cfg(feature = "net_https")]
fn openai_embed(texts: &Vec<String>) -> Option<Value> {
    if texts.is_empty() { return Some(Value::List(vec![])); }
    let api_key = std::env::var("OPENAI_API_KEY").ok()?;
    let client = Client::new();
    let model = "text-embedding-3-small";
    let body = serde_json::json!({"model": model, "input": texts});
    let resp = client.post("https://api.openai.com/v1/embeddings")
        .header(AUTHORIZATION, format!("Bearer {}", api_key))
        .header(CONTENT_TYPE, "application/json")
        .json(&body)
        .send().ok()?;
    let js: serde_json::Value = resp.json().ok()?;
    let arr = js["data"].as_array()?;
    let out: Vec<Value> = arr.iter().map(|row| {
        let v = row["embedding"].as_array().cloned().unwrap_or_else(|| Vec::new());
        Value::List(v.into_iter().filter_map(|x| x.as_f64()).map(Value::Real).collect())
    }).collect();
    Some(Value::List(out))
}
