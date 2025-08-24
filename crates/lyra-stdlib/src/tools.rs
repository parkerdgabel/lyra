use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_runtime::attrs::Attributes;
use std::collections::HashMap;
use std::sync::{OnceLock, Mutex};

// Minimal in-memory tool registry storing self-describing specs as Assoc Values
static TOOL_REG: OnceLock<Mutex<HashMap<String, Value>>> = OnceLock::new();
static CAPABILITIES: OnceLock<Mutex<Option<std::collections::HashSet<String>>>> = OnceLock::new();

fn tool_reg() -> &'static Mutex<HashMap<String, Value>> {
    TOOL_REG.get_or_init(|| Mutex::new(HashMap::new()))
}

// Inline registration helper for stdlib/native code to attach specs at definition time
#[allow(dead_code)]
pub fn add_specs(specs: Vec<Value>) {
    let mut reg = tool_reg().lock().unwrap();
    for s in specs.into_iter() {
        if let Value::Assoc(m) = &s {
            if let Some(id) = get_str(m, "id").or_else(|| get_str(m, "name")) {
                reg.insert(id, s);
            }
        }
    }
}

// Macro to concisely build a tool spec Value::Assoc
// Forms:
//  tool_spec!("Name", summary: "...", params: [..], tags: [..], input_schema: VALUE, output_schema: VALUE, examples: [VALUE,...], effects: [..])
//  tool_spec!("Name", impl: "ImplName", summary: "...", ...)
#[macro_export]
macro_rules! tool_spec {
    ($name:expr, impl: $impln:expr, summary: $summary:expr $(, params: [$($param:expr),* $(,)?])? $(, tags: [$($tag:expr),* $(,)?])? $(, input_schema: $in_schema:expr)? $(, output_schema: $out_schema:expr)? $(, examples: [$($ex:expr),* $(,)?])? $(, effects: [$($eff:expr),* $(,)?])? ) => {{
        let mut m: std::collections::HashMap<String, lyra_core::value::Value> = std::collections::HashMap::new();
        m.insert("id".into(), lyra_core::value::Value::String($name.into()));
        m.insert("name".into(), lyra_core::value::Value::String($name.into()));
        m.insert("impl".into(), lyra_core::value::Value::String($impln.into()));
        m.insert("summary".into(), lyra_core::value::Value::String($summary.into()));
        $( m.insert("params".into(), lyra_core::value::Value::List(vec![ $( lyra_core::value::Value::String($param.into()) ),* ])); )?
        $( m.insert("tags".into(), lyra_core::value::Value::List(vec![ $( lyra_core::value::Value::String($tag.into()) ),* ])); )?
        $( m.insert("input_schema".into(), $in_schema ); )?
        $( m.insert("output_schema".into(), $out_schema ); )?
        $( m.insert("examples".into(), lyra_core::value::Value::List(vec![ $( $ex ),* ])); )?
        $( m.insert("effects".into(), lyra_core::value::Value::List(vec![ $( lyra_core::value::Value::String($eff.into()) ),* ])); )?
        lyra_core::value::Value::Assoc(m)
    }};
    ($name:expr, summary: $summary:expr $(, params: [$($param:expr),* $(,)?])? $(, tags: [$($tag:expr),* $(,)?])? $(, input_schema: $in_schema:expr)? $(, output_schema: $out_schema:expr)? $(, examples: [$($ex:expr),* $(,)?])? $(, effects: [$($eff:expr),* $(,)?])? ) => {{
        tool_spec!($name, impl: $name, summary: $summary $(, params: [$($param),*])? $(, tags: [$($tag),*])? $(, input_schema: $in_schema)? $(, output_schema: $out_schema)? $(, examples: [$($ex),*])? $(, effects: [$($eff),*])? )
    }};
}

// -------- Schema helpers (functions + macros)
#[allow(dead_code)]
pub fn schema_type_value(t: &str) -> Value {
    Value::Assoc(HashMap::from([(String::from("type"), Value::String(t.to_string()))]))
}

#[allow(dead_code)]
pub fn schema_array_value(items: Value) -> Value {
    Value::Assoc(HashMap::from([
        (String::from("type"), Value::String(String::from("array"))),
        (String::from("items"), items),
    ]))
}

#[allow(dead_code)]
pub fn schema_object_value(props: Vec<(String, Value)>, required: Vec<String>) -> Value {
    let mut props_map: HashMap<String, Value> = HashMap::new();
    for (k, v) in props { props_map.insert(k, v); }
    let required_vals: Vec<Value> = required.into_iter().map(Value::String).collect();
    Value::Assoc(HashMap::from([
        (String::from("type"), Value::String(String::from("object"))),
        (String::from("properties"), Value::Assoc(props_map)),
        (String::from("required"), Value::List(required_vals)),
    ]))
}

#[macro_export]
macro_rules! schema_str { () => { $crate::tools::schema_type_value("string") } }
#[macro_export]
macro_rules! schema_int { () => { $crate::tools::schema_type_value("integer") } }
#[macro_export]
macro_rules! schema_num { () => { $crate::tools::schema_type_value("number") } }
#[macro_export]
macro_rules! schema_bool { () => { $crate::tools::schema_type_value("boolean") } }
#[macro_export]
macro_rules! schema_arr { ($items:expr) => { $crate::tools::schema_array_value($items) } }

pub fn register_tools(ev: &mut Evaluator) {
    ev.register("ToolsRegister", tools_register as NativeFn, Attributes::LISTABLE);
    ev.register("ToolsUnregister", tools_unregister as NativeFn, Attributes::empty());
    ev.register("ToolsList", tools_list as NativeFn, Attributes::empty());
    ev.register("ToolsCards", tools_cards as NativeFn, Attributes::empty());
    ev.register("ToolsDescribe", tools_describe as NativeFn, Attributes::empty());
    ev.register("ToolsSearch", tools_search as NativeFn, Attributes::empty());
    ev.register("ToolsResolve", tools_resolve as NativeFn, Attributes::empty());
    ev.register("ToolsInvoke", tools_invoke as NativeFn, Attributes::HOLD_ALL);
    ev.register("ToolsDryRun", tools_dry_run as NativeFn, Attributes::HOLD_ALL);
    ev.register("ToolsExportOpenAI", tools_export_openai as NativeFn, Attributes::empty());
    ev.register("ToolsExportBundle", tools_export_bundle as NativeFn, Attributes::empty());
    ev.register("ToolsSetCapabilities", tools_set_capabilities as NativeFn, Attributes::LISTABLE);
    ev.register("ToolsGetCapabilities", tools_get_capabilities as NativeFn, Attributes::empty());
}

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

fn get_str(m: &HashMap<String, Value>, k: &str) -> Option<String> {
    m.get(k).and_then(|v| match v { Value::String(s)=>Some(s.clone()), Value::Symbol(s)=>Some(s.clone()), _=>None })
}

fn value_to_string(v: &Value) -> Option<String> {
    match v { Value::String(s)=>Some(s.clone()), Value::Symbol(s)=>Some(s.clone()), _=>None }
}

fn type_name_of(v: &Value) -> &'static str {
    match v {
        Value::Integer(_) => "integer",
        Value::Real(_) => "number",
        Value::String(_) => "string",
        Value::Symbol(_) => "string",
        Value::Boolean(_) => "boolean",
        Value::List(_) => "array",
        Value::Assoc(_) => "object",
        _ => "unknown",
    }
}

// Minimal JSON Schema validator: supports object properties/required and primitive types; arrays with items.type
fn validate_against_schema(schema: &Value, provided: &Value) -> (bool, Vec<Value>) {
    let mut ok = true;
    let mut errors: Vec<Value> = Vec::new();
    let mut check_type = |expected: &str, v: &Value, path: &str, errors: &mut Vec<Value>| -> bool {
        let actual = type_name_of(v);
        if expected != actual {
            errors.push(Value::Assoc(HashMap::from([
                ("path".to_string(), Value::String(path.to_string())),
                ("expected".to_string(), Value::String(expected.to_string())),
                ("actual".to_string(), Value::String(actual.to_string())),
            ])));
            false
        } else { true }
    };
    fn get_field<'a>(m: &'a HashMap<String, Value>, k: &str) -> Option<&'a Value> { m.get(k) }
    match (schema, provided) {
        (Value::Assoc(s), Value::Assoc(p)) => {
            // type: object
            if let Some(Value::String(t)) = get_field(s, "type") { if t != "object" { /* ignore other root types */ } }
            // required
            if let Some(Value::List(reqs)) = get_field(s, "required") {
                for r in reqs.iter().filter_map(value_to_string) {
                    if !p.contains_key(&r) {
                        ok = false;
                        errors.push(Value::Assoc(HashMap::from([
                            ("path".to_string(), Value::String(format!(".{}", r))),
                            ("missing".to_string(), Value::Boolean(true)),
                        ])));
                    }
                }
            }
            // properties
            if let Some(Value::Assoc(props)) = get_field(s, "properties") {
                for (k, prop_schema) in props.iter() {
                    if let Some(v) = p.get(k) {
                        match prop_schema {
                            Value::Assoc(pm) => {
                                if let Some(Value::String(t)) = pm.get("type") {
                                    match t.as_str() {
                                        "array" => {
                                            if !check_type("array", v, &format!(".{}", k), &mut errors) { ok = false; }
                                            if let Value::List(items) = v {
                                                if let Some(Value::Assoc(item_schema)) = pm.get("items") {
                                                    if let Some(Value::String(it_t)) = item_schema.get("type") {
                                                        for (idx, it) in items.iter().enumerate() {
                                                            let actual = type_name_of(it);
                                                            if &actual != it_t {
                                                                ok = false;
                                                                errors.push(Value::Assoc(HashMap::from([
                                                                    ("path".to_string(), Value::String(format!(".{}[{}]", k, idx))),
                                                                    ("expected".to_string(), Value::String(it_t.clone())),
                                                                    ("actual".to_string(), Value::String(actual.to_string())),
                                                                ])));
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        other => {
                                            if !check_type(other, v, &format!(".{}", k), &mut errors) { ok = false; }
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        (Value::Assoc(s), other) => {
            if let Some(Value::String(t)) = s.get("type") { if !check_type(t, other, "$", &mut errors) { ok = false; } }
        }
        _ => {}
    }
    (ok, errors)
}

// ToolsRegister[spec] or ToolsRegister[{spec1, spec2, ...}]
fn tools_register(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("ToolsRegister".into())), args } }
    let mut reg = tool_reg().lock().unwrap();
    for a in args.into_iter() {
        match a {
            Value::Assoc(m) => {
                if let Some(id) = get_str(&m, "id").or_else(|| get_str(&m, "name")) {
                    reg.insert(id, Value::Assoc(m));
                }
            }
            Value::List(items) => {
                for it in items.into_iter() {
                    if let Value::Assoc(m) = it {
                        if let Some(id) = get_str(&m, "id").or_else(|| get_str(&m, "name")) {
                            reg.insert(id, Value::Assoc(m));
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Value::Boolean(true)
}

// ToolsUnregister[id]
fn tools_unregister(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("ToolsUnregister".into())), args } }
    let id = match &args[0] { Value::String(s)|Value::Symbol(s)=>s.clone(), _=>return Value::Boolean(false) };
    let mut reg = tool_reg().lock().unwrap();
    Value::Boolean(reg.remove(&id).is_some())
}

fn builtin_cards(ev: &mut Evaluator) -> Vec<Value> {
    // Fallback discovery using only builtin names and attributes
    // We cannot access builtins directly; expose minimal card for known exported functions via DescribeBuiltins (if present)
    // If DescribeBuiltins is not present, return empty list.
    let head = Value::Symbol("DescribeBuiltins".to_string());
    let expr = Value::Expr { head: Box::new(head), args: vec![] };
    let desc = ev.eval(expr);
    match desc {
        Value::List(items) => items,
        _ => vec![],
    }
}

fn stdlib_default_specs(ev: &mut Evaluator) -> Vec<Value> {
    // Provide richer specs for common stdlib functions when present.
    // Only include if function exists in current instance (post tree-shake) by checking DescribeBuiltins names.
    let names: std::collections::HashSet<String> = builtin_cards(ev).into_iter().filter_map(|v| if let Value::Assoc(m)=v { get_str(&m, "name") } else { None }).collect();
    let mut specs: Vec<Value> = Vec::new();
    let mut add = |name: &str, summary: &str, params: Vec<&str>, tags: Vec<&str>| {
        if names.contains(name) {
            let pvals: Vec<Value> = params.iter().map(|s| Value::String((*s).into())).collect();
            let tvals: Vec<Value> = tags.iter().map(|s| Value::String((*s).into())).collect();
            let mut props: HashMap<String, Value> = HashMap::new();
            for p in &params { props.insert((*p).into(), Value::Assoc(HashMap::from([("type".to_string(), Value::String("string".into()))]))); }
            let schema = Value::Assoc(HashMap::from([
                ("type".to_string(), Value::String("object".into())),
                ("properties".to_string(), Value::Assoc(props)),
            ]));
            let m = Value::Assoc(HashMap::from([
                ("id".to_string(), Value::String(name.to_string())),
                ("name".to_string(), Value::String(name.to_string())),
                ("impl".to_string(), Value::String(name.to_string())),
                ("summary".to_string(), Value::String(summary.into())),
                ("tags".to_string(), Value::List(tvals)),
                ("effects".to_string(), Value::List(vec![])),
                ("params".to_string(), Value::List(pvals)),
                ("input_schema".to_string(), schema),
            ]));
            specs.push(m);
        }
    };
    add("HtmlEscape", "Escape HTML special characters.", vec!["s"], vec!["string","html","escape"]);
    add("HtmlUnescape", "Unescape HTML entities.", vec!["s"], vec!["string","html","unescape"]);
    add("UrlEncode", "Percent-encode string for URLs.", vec!["s"], vec!["string","url","encode"]);
    add("UrlDecode", "Decode percent-encoded URL string.", vec!["s"], vec!["string","url","decode"]);
    add("UrlFormEncode", "Form-url-encode string.", vec!["s"], vec!["string","form","encode"]);
    add("UrlFormDecode", "Form-url-decode string.", vec!["s"], vec!["string","form","decode"]);
    add("JsonEscape", "Escape string for JSON.", vec!["s"], vec!["string","json","escape"]);
    add("JsonUnescape", "Unescape JSON string.", vec!["s"], vec!["string","json","unescape"]);
    add("Slugify", "Convert text to lowercase URL slug.", vec!["s"], vec!["string","slug"]);
    add("StringTruncate", "Truncate to max length with suffix.", vec!["s","max","suffix"], vec!["string","truncate"]);
    add("CamelCase", "Convert to lower camelCase.", vec!["s"], vec!["string","case"]);
    add("SnakeCase", "Convert to snake_case.", vec!["s"], vec!["string","case"]);
    add("KebabCase", "Convert to kebab-case.", vec!["s"], vec!["string","case"]);
    add("StringFormat", "Positional format using {0},{1}.", vec!["template","args"], vec!["string","format"]);
    add("StringFormatMap", "Named format using {key}.", vec!["template","map"], vec!["string","format"]);
    add("StringInterpolate", "Interpolate expressions inside braces.", vec!["s"], vec!["string","template"]);
    add("StringInterpolateWith", "Interpolate using provided variables.", vec!["s","vars"], vec!["string","template"]);
    add("TemplateRender", "Render Mustache-like template.", vec!["template","data","opts"], vec!["string","template"]);
    specs
}

// ToolsList[] -> list of cards (registered specs first, then fallback builtins not overridden)
fn tools_list(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Optional filter: <|"effects"->{..}, "tags"->{..}|>
    if args.len()>1 { return Value::Expr { head: Box::new(Value::Symbol("ToolsList".into())), args } }
    let filter = if args.len()==1 { ev.eval(args[0].clone()) } else { Value::Assoc(HashMap::new()) };
    let reg = tool_reg().lock().unwrap();
    let mut out: Vec<Value> = Vec::new();
    for (id, spec) in reg.iter() {
        let mut m = match spec { Value::Assoc(m)=>m.clone(), _=>HashMap::new() };
        m.entry("id".to_string()).or_insert(Value::String(id.clone()));
        out.push(Value::Assoc(m));
    }
    // Add stdlib default specs for present functions (richer than generic cards)
    let mut seen: std::collections::HashSet<String> = out.iter().filter_map(|v| if let Value::Assoc(m)=v { get_str(m, "id") } else { None }).collect();
    for spec in stdlib_default_specs(ev) {
        if let Value::Assoc(m) = &spec {
            if let Some(id) = get_str(m, "id").or_else(|| get_str(m, "name")) {
                if !seen.contains(&id) { out.push(spec); seen.insert(id); }
            }
        }
    }
    // Add fallback builtin cards if available and not already present
    let mut seen: std::collections::HashSet<String> = out.iter().filter_map(|v| if let Value::Assoc(m)=v { get_str(m, "id") } else { None }).collect();
    for card in builtin_cards(ev) {
        if let Value::Assoc(m) = &card {
            if let Some(id) = get_str(m, "id").or_else(|| get_str(m, "name")) {
                if !seen.contains(&id) { out.push(card); seen.insert(id); }
            }
        }
    }
    // Apply filter if provided
    // Optional global capability gate
    let granted_caps: std::collections::HashSet<String> = CAPABILITIES
        .get_or_init(|| Mutex::new(None))
        .lock().unwrap()
        .as_ref()
        .cloned()
        .unwrap_or_default();
    let filtered = match filter {
        Value::Assoc(f) if !f.is_empty() => {
            let eff_filter: std::collections::HashSet<String> = match f.get("effects") { Some(Value::List(vs))=>vs.iter().filter_map(value_to_string).map(|s| s.to_lowercase()).collect(), _=>Default::default() };
            let caps_filter: std::collections::HashSet<String> = match f.get("capabilities") { Some(Value::List(vs))=>vs.iter().filter_map(value_to_string).map(|s| s.to_lowercase()).collect(), _=>Default::default() };
            let tag_filter: std::collections::HashSet<String> = match f.get("tags") { Some(Value::List(vs))=>vs.iter().filter_map(value_to_string).map(|s| s.to_lowercase()).collect(), _=>Default::default() };
            out.into_iter().filter(|v| {
                if let Value::Assoc(m) = v {
                    let mut ok = true;
                    if !eff_filter.is_empty() {
                        let mut have: std::collections::HashSet<String> = Default::default();
                        if let Some(Value::List(effs)) = m.get("effects") { for e in effs { if let Some(s)=value_to_string(e) { have.insert(s.to_lowercase()); } } }
                        ok = eff_filter.is_subset(&have);
                    }
                    if ok && (!caps_filter.is_empty() || !granted_caps.is_empty()) {
                        // Effective capability set
                        let want = if !caps_filter.is_empty() { &caps_filter } else { &granted_caps };
                        if !want.is_empty() {
                            let mut effs: std::collections::HashSet<String> = Default::default();
                            if let Some(Value::List(effs_v)) = m.get("effects") { for e in effs_v { if let Some(s)=value_to_string(e) { effs.insert(s.to_lowercase()); } } }
                            // Require all effects ⊆ want
                            ok = effs.is_subset(want);
                        }
                    }
                    if ok && !tag_filter.is_empty() {
                        let mut have: std::collections::HashSet<String> = Default::default();
                        if let Some(Value::List(tags)) = m.get("tags") { for t in tags { if let Some(s)=value_to_string(t) { have.insert(s.to_lowercase()); } } }
                        ok = !tag_filter.is_disjoint(&have);
                    }
                    ok
                } else { false }
            }).collect::<Vec<_>>()
        }
        _ => out,
    };
    Value::List(filtered)
}

// ToolsSetCapabilities[List[String]] -> True
fn tools_set_capabilities(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("ToolsSetCapabilities".into())), args } }
    let caps: Option<std::collections::HashSet<String>> = match &args[0] {
        Value::List(vs) => Some(vs.iter().filter_map(|v| value_to_string(v)).map(|s| s.to_lowercase()).collect()),
        _ => None,
    };
    if let Some(set) = caps {
        let mu = CAPABILITIES.get_or_init(|| Mutex::new(None));
        *mu.lock().unwrap() = Some(set);
        Value::Boolean(true)
    } else { Value::Boolean(false) }
}

// ToolsGetCapabilities[] -> List[String]
fn tools_get_capabilities(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if !args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("ToolsGetCapabilities".into())), args } }
    let mu = CAPABILITIES.get_or_init(|| Mutex::new(None));
    if let Some(set) = mu.lock().unwrap().as_ref() {
        Value::List(set.iter().cloned().map(Value::String).collect())
    } else { Value::List(vec![]) }
}

// ToolsCards[cursor?, limit?] -> <|"items"->List, "next_cursor"->String|>
fn tools_cards(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()>2 { return Value::Expr { head: Box::new(Value::Symbol("ToolsCards".into())), args } }
    let cursor = if args.len()>=1 { match ev.eval(args[0].clone()) { Value::Integer(i)=>i as usize, Value::String(s)=>s.parse::<usize>().unwrap_or(0), _=>0 } } else { 0 };
    let limit = if args.len()==2 { match ev.eval(args[1].clone()) { Value::Integer(i)=>i.max(1) as usize, _=>20 } } else { 20 };
    let all = match tools_list(ev, vec![]) { Value::List(vs)=>vs, _=>vec![] };
    let end = (cursor + limit).min(all.len());
    let slice = all[cursor..end].to_vec();
    let next = if end < all.len() { Value::String(end.to_string()) } else { Value::Symbol("Null".into()) };
    Value::Assoc(vec![
        ("items".to_string(), Value::List(slice)),
        ("next_cursor".to_string(), next),
    ].into_iter().collect())
}

// ToolsDescribe[id] -> spec or fallback card
fn tools_describe(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("ToolsDescribe".into())), args } }
    let key = match &args[0] { Value::String(s)|Value::Symbol(s)=>s.clone(), _=>return Value::Assoc(HashMap::new()) };
    let reg = tool_reg().lock().unwrap();
    if let Some(spec) = reg.get(&key) { return spec.clone(); }
    // try fallback cards by name/id
    for card in builtin_cards(ev) {
        if let Value::Assoc(m) = &card {
            let id = get_str(m, "id").or_else(|| get_str(m, "name"));
            if id.as_deref()==Some(&key) { return card; }
        }
    }
    Value::Assoc(HashMap::new())
}

// ToolsSearch[query, topK?] -> compact cards with naive scoring
fn tools_search(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("ToolsSearch".into())), args } }
    let q = match &args[0] { Value::String(s)|Value::Symbol(s)=>s.to_lowercase(), _=>String::new() };
    let topk = if args.len()>=2 { if let Value::Integer(k) = args[1].clone() { k.max(1) as usize } else { 10 } } else { 10 };
    let mut pool: Vec<Value> = match tools_list(ev, vec![]) { Value::List(vs)=>vs, _=>vec![] };
    let mut scored: Vec<(i64, Value)> = Vec::new();
    for v in pool.drain(..) {
        let mut score: i64 = 0;
        if let Value::Assoc(m) = &v {
            if let Some(s) = get_str(m, "name").or_else(|| get_str(m, "id")) { if s.to_lowercase().contains(&q) { score += 5; } }
            if let Some(s) = get_str(m, "summary") { if s.to_lowercase().contains(&q) { score += 3; } }
            if let Some(Value::List(tags)) = m.get("tags") {
                for t in tags { if let Some(ts) = value_to_string(t) { if ts.to_lowercase().contains(&q) { score += 2; } } }
            }
        }
        if score>0 || q.is_empty() { scored.push((score, v)); }
    }
    scored.sort_by(|a,b| b.0.cmp(&a.0));
    Value::List(scored.into_iter().take(topk).map(|(_s, v)| v).collect())
}

// ToolsResolve[pattern, topK?] -> [{"id","score"}]
fn tools_resolve(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("ToolsResolve".into())), args } }
    let q = match &args[0] { Value::String(s)|Value::Symbol(s)=>s.to_lowercase(), _=>String::new() };
    let topk = if args.len()>=2 { if let Value::Integer(k) = args[1].clone() { k.max(1) as usize } else { 5 } } else { 5 };
    let mut pool: Vec<(String, i64)> = Vec::new();
    if let Value::List(vs) = tools_list(ev, vec![]) {
        for v in vs {
            if let Value::Assoc(m) = v {
                if let Some(id) = get_str(&m, "id").or_else(|| get_str(&m, "name")) {
                    let hay = format!("{} {} {}", id, get_str(&m, "summary").unwrap_or_default(), get_str(&m, "name").unwrap_or_default()).to_lowercase();
                    let mut score = 0i64;
                    if id.to_lowercase()==q { score += 10; }
                    if hay.contains(&q) { score += 5; }
                    pool.push((id, score));
                }
            }
        }
    }
    pool.sort_by(|a,b| b.1.cmp(&a.1));
    Value::List(pool.into_iter().take(topk).map(|(id,score)| Value::Assoc(vec![("id".to_string(), Value::String(id)), ("score".to_string(), Value::Integer(score))].into_iter().collect())).collect())
}

// ToolsInvoke[idOrName, argsAssoc?] -> evaluate head with positional mapping if params provided in spec
fn tools_invoke(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("ToolsInvoke".into())), args } }
    let key = match &args[0] { Value::String(s)|Value::Symbol(s)=>s.clone(), other=> match ev.eval(other.clone()) { Value::String(s)|Value::Symbol(s)=>s, _=>String::new() } };
    let provided = if args.len()>=2 { ev.eval(args[1].clone()) } else { Value::Assoc(HashMap::new()) };
    let (name, params): (String, Vec<String>) = {
        let reg = tool_reg().lock().unwrap();
        if let Some(Value::Assoc(m)) = reg.get(&key) {
            let nm = get_str(m, "impl").or_else(|| get_str(m, "name")).unwrap_or(key.clone());
            let ps: Vec<String> = match m.get("params") {
                Some(Value::List(vs)) => vs.iter().filter_map(|x| value_to_string(x)).collect(),
                _ => vec![],
            };
            (nm, ps)
        } else { (key.clone(), vec![]) }
    };
    // If provided args is an Assoc and params exist, map by order; else if provided is a List, use as-is.
    let call_args: Vec<Value> = match &provided {
        Value::Assoc(m) if !params.is_empty() => {
            let mut out: Vec<Value> = Vec::new();
            for p in params.iter() {
                out.push(m.get(p).cloned().unwrap_or(Value::Symbol("Null".into())));
            }
            out
        }
        Value::List(vs) => vs.clone(),
        Value::Assoc(m) => {
            // If no params, but single key "args" -> use list under it
            if let Some(Value::List(vs)) = m.get("args") { vs.clone() } else { vec![Value::Assoc(m.clone())] }
        }
        other => vec![other.clone()],
    };
    let expr = Value::Expr { head: Box::new(Value::Symbol(name)), args: call_args };
    ev.eval(expr)
}

// ToolsDryRun[id, argsAssoc] -> validation + normalized args + estimates
fn tools_dry_run(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("ToolsDryRun".into())), args } }
    let key = match &args[0] { Value::String(s)|Value::Symbol(s)=>s.clone(), other=> match ev.eval(other.clone()) { Value::String(s)|Value::Symbol(s)=>s, _=>String::new() } };
    let provided = ev.eval(args[1].clone());
    let reg = tool_reg().lock().unwrap();
    let spec_opt = reg.get(&key).cloned();
    let mut normalized: Vec<Value> = Vec::new();
    let mut missing: Vec<Value> = Vec::new();
    let mut errors: Vec<Value> = Vec::new();
    let mut ok = true;
    let mut estimates = Value::Assoc(HashMap::new());
    if let Some(Value::Assoc(m)) = spec_opt {
        if let Some(Value::Assoc(cost)) = m.get("cost_estimate") { estimates = Value::Assoc(cost.clone()); }
        let params: Vec<String> = match m.get("params") { Some(Value::List(vs))=>vs.iter().filter_map(value_to_string).collect(), _=>vec![] };
        let defaults: HashMap<String, Value> = match m.get("param_defaults") { Some(Value::Assoc(am))=>am.clone(), _=>HashMap::new() };
        if let Value::Assoc(input) = provided.clone() {
            for p in &params { if let Some(v) = input.get(p) { normalized.push(v.clone()); } else if let Some(v) = defaults.get(p) { normalized.push(v.clone()); } else { ok=false; missing.push(Value::String(p.clone())); normalized.push(Value::Symbol("Null".into())); } }
            // Schema validation if provided
            if let Some(Value::Assoc(schema)) = m.get("input_schema") {
                let (_ok, mut errs) = validate_against_schema(&Value::Assoc(schema.clone()), &Value::Assoc(input.clone()));
                if !_ok { ok = false; }
                errors.append(&mut errs);
            }
        } else if let Value::List(vs) = provided { normalized = vs; }
    } else {
        // Unknown spec; pass through
        normalized = match provided { Value::List(vs)=>vs, Value::Assoc(m)=>vec![Value::Assoc(m)], v=>vec![v] };
    }
    Value::Assoc(vec![
        ("ok".to_string(), Value::Boolean(ok)),
        ("normalized_args".to_string(), Value::List(normalized)),
        ("missing".to_string(), Value::List(missing)),
        ("errors".to_string(), Value::List(errors)),
        ("estimates".to_string(), estimates),
    ].into_iter().collect())
}

fn sanitize_name(mut s: String) -> String {
    if s.is_empty() { return "tool".into(); }
    s = s.chars().map(|c| if c.is_ascii_alphanumeric() || c=='_' || c=='-' { c } else { '_' }).collect();
    if s.len()>64 { s.truncate(64); }
    s
}

// ToolsExportOpenAI[] -> list of {type:"function", function:{ name, description, parameters }}
fn tools_export_openai(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if !args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("ToolsExportOpenAI".into())), args } }
    let list = match tools_list(ev, vec![]) { Value::List(vs)=>vs, _=>vec![] };
    let mut out: Vec<Value> = Vec::new();
    for v in list.into_iter() {
        if let Value::Assoc(m) = v {
            let id = get_str(&m, "id").or_else(|| get_str(&m, "name")).unwrap_or_else(|| "tool".into());
            let name = sanitize_name(get_str(&m, "name").unwrap_or_else(|| id.clone()));
            let description = get_str(&m, "summary").unwrap_or_default();
            let params_schema = match m.get("input_schema") { Some(Value::Assoc(s))=>Value::Assoc(s.clone()), _=>{
                // derive minimal schema from params
                let mut props: HashMap<String, Value> = HashMap::new();
                let mut req: Vec<Value> = Vec::new();
                if let Some(Value::List(vs)) = m.get("params") {
                    for p in vs.iter().filter_map(value_to_string) { props.insert(p.clone(), Value::Assoc(HashMap::from([("type".to_string(), Value::String("string".into()))]))); req.push(Value::String(p)); }
                }
                Value::Assoc(HashMap::from([
                    ("type".to_string(), Value::String("object".into())),
                    ("properties".to_string(), Value::Assoc(props)),
                    ("required".to_string(), Value::List(req)),
                ]))
            } };
            let function = Value::Assoc(HashMap::from([
                ("name".to_string(), Value::String(name)),
                ("description".to_string(), Value::String(description)),
                ("parameters".to_string(), params_schema),
            ]));
            out.push(Value::Assoc(HashMap::from([
                ("type".to_string(), Value::String("function".into())),
                ("function".to_string(), function),
                ("id".to_string(), Value::String(id)),
            ])));
        }
    }
    Value::List(out)
}

// ToolsExportBundle[] -> all registered specs as a list (machine-cachable)
fn tools_export_bundle(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if !args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("ToolsExportBundle".into())), args } }
    let reg = tool_reg().lock().unwrap();
    Value::List(reg.values().cloned().collect())
}
