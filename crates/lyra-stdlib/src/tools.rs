use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_runtime::attrs::Attributes;
use std::collections::HashMap;
use std::sync::{OnceLock, Mutex};
use crate::register_if;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

// Minimal in-memory tool registry storing self-describing specs as Assoc Values
static TOOL_REG: OnceLock<Mutex<HashMap<String, Value>>> = OnceLock::new();
static CAPABILITIES: OnceLock<Mutex<Option<std::collections::HashSet<String>>>> = OnceLock::new();
static TOOL_CACHE: OnceLock<Mutex<HashMap<u64, Value>>> = OnceLock::new();
static IDEMP_SEQ: OnceLock<std::sync::atomic::AtomicI64> = OnceLock::new();

fn tool_reg() -> &'static Mutex<HashMap<String, Value>> {
    TOOL_REG.get_or_init(|| Mutex::new(HashMap::new()))
}
fn tool_cache() -> &'static Mutex<HashMap<u64, Value>> {
    TOOL_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}
fn next_idemp() -> i64 { let a = IDEMP_SEQ.get_or_init(|| std::sync::atomic::AtomicI64::new(1)); a.fetch_add(1, std::sync::atomic::Ordering::Relaxed) }

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
    ev.register("ToolsCacheClear", tools_cache_clear as NativeFn, Attributes::empty());
    ev.register("IdempotencyKey", idempotency_key as NativeFn, Attributes::empty());
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
    let check_type = |expected: &str, v: &Value, path: &str, errors: &mut Vec<Value>| -> bool {
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
    let add = |name: &str, summary: &str, params: Vec<&str>, tags: Vec<&str>| {
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
    // (string helpers now have detailed specs later in this function)
    // Core string functions (generic defaults)
    // (moved to detailed specs below)
    // Tightened schemas with examples for core string functions
    // Note: placed after generic adds to avoid borrow conflicts with the `add` closure above.
    if names.contains("StringLength") { specs.push(tool_spec!(
        "StringLength", summary: "Length of string (Unicode scalar count)", params: ["s"], tags: ["string"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()) ], vec!["s".into()]), output_schema: schema_int!(), examples: [
            Value::Assoc(HashMap::from([("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("hé".into())) ]))), ("result".into(), Value::Integer(2))]))
        ]
    )); }
    if names.contains("ToUpper") { specs.push(tool_spec!(
        "ToUpper", summary: "Uppercase string", params: ["s"], tags: ["string","case"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()) ], vec!["s".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("hi".into())) ]))), ("result".into(), Value::String("HI".into()))]))
        ]
    )); }
    if names.contains("ToLower") { specs.push(tool_spec!(
        "ToLower", summary: "Lowercase string", params: ["s"], tags: ["string","case"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()) ], vec!["s".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("Hi".into())) ]))), ("result".into(), Value::String("hi".into()))]))
        ]
    )); }
    if names.contains("StringJoin") { specs.push(tool_spec!(
        "StringJoin", summary: "Concatenate list of parts", params: ["parts"], tags: ["string","join"],
        input_schema: schema_object_value(vec![ ("parts".into(), schema_arr!(schema_str!())) ], vec!["parts".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([("args".into(), Value::Assoc(HashMap::from([( "parts".into(), Value::List(vec![Value::String("a".into()), Value::String("b".into())])) ]))), ("result".into(), Value::String("ab".into()))]))
        ]
    )); }
    if names.contains("StringJoinWith") { specs.push(tool_spec!(
        "StringJoinWith", summary: "Join list with separator", params: ["sep","parts"], tags: ["string","join"],
        input_schema: schema_object_value(vec![ ("sep".into(), schema_str!()), ("parts".into(), schema_arr!(schema_str!())) ], vec!["sep".into(), "parts".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([("args".into(), Value::Assoc(HashMap::from([( "sep".into(), Value::String(",".into())), ( "parts".into(), Value::List(vec![Value::String("a".into()), Value::String("b".into())])) ]))), ("result".into(), Value::String("a,b".into()))]))
        ]
    )); }
    if names.contains("StringTrim") { specs.push(tool_spec!(
        "StringTrim", summary: "Trim whitespace at both ends", params: ["s"], tags: ["string","trim"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()) ], vec!["s".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("  hi  ".into())) ]))), ("result".into(), Value::String("hi".into()))]))
        ]
    )); }
    if names.contains("StringTrimLeft") { specs.push(tool_spec!(
        "StringTrimLeft", summary: "Trim leading whitespace", params: ["s"], tags: ["string","trim"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()) ], vec!["s".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("  hi".into())) ]))), ("result".into(), Value::String("hi".into()))]))
        ]
    )); }
    if names.contains("StringTrimRight") { specs.push(tool_spec!(
        "StringTrimRight", summary: "Trim trailing whitespace", params: ["s"], tags: ["string","trim"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()) ], vec!["s".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("hi  ".into())) ]))), ("result".into(), Value::String("hi".into()))]))
        ]
    )); }
    if names.contains("StringTrimPrefix") { specs.push(tool_spec!(
        "StringTrimPrefix", summary: "Drop prefix if present", params: ["s","prefix"], tags: ["string","trim"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()), ("prefix".into(), schema_str!()) ], vec!["s".into(), "prefix".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("foobar".into())), ("prefix".into(), Value::String("foo".into())) ]))), ("result".into(), Value::String("bar".into()))]))
        ]
    )); }
    if names.contains("StringTrimSuffix") { specs.push(tool_spec!(
        "StringTrimSuffix", summary: "Drop suffix if present", params: ["s","suffix"], tags: ["string","trim"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()), ("suffix".into(), schema_str!()) ], vec!["s".into(), "suffix".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("foobar".into())), ("suffix".into(), Value::String("bar".into())) ]))), ("result".into(), Value::String("foo".into()))]))
        ]
    )); }
    if names.contains("StringContains") { specs.push(tool_spec!(
        "StringContains", summary: "Substring containment test", params: ["s","substr"], tags: ["string","search"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()), ("substr".into(), schema_str!()) ], vec!["s".into(), "substr".into()]), output_schema: schema_bool!(), examples: [
            Value::Assoc(HashMap::from([("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("hello".into())), ("substr".into(), Value::String("ell".into())) ]))), ("result".into(), Value::Boolean(true))]))
        ]
    )); }
    if names.contains("StringSplit") { specs.push(tool_spec!(
        "StringSplit", summary: "Split string by delimiter", params: ["s","delim"], tags: ["string","split"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()), ("delim".into(), schema_str!()) ], vec!["s".into(), "delim".into()]), output_schema: schema_arr!(schema_str!()), examples: [
            Value::Assoc(HashMap::from([("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("a,b".into())), ("delim".into(), Value::String(",".into())) ]))), ("result".into(), Value::List(vec![Value::String("a".into()), Value::String("b".into())]))]))
        ]
    )); }
    if names.contains("SplitLines") { specs.push(tool_spec!(
        "SplitLines", summary: "Split into lines", params: ["s"], tags: ["string","split"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()) ], vec!["s".into()]), output_schema: schema_arr!(schema_str!()), examples: [
            Value::Assoc(HashMap::from([("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("a\nb".into())) ]))), ("result".into(), Value::List(vec![Value::String("a".into()), Value::String("b".into())]))]))
        ]
    )); }
    if names.contains("JoinLines") { specs.push(tool_spec!(
        "JoinLines", summary: "Join lines with newline", params: ["lines"], tags: ["string","join"],
        input_schema: schema_object_value(vec![ ("lines".into(), schema_arr!(schema_str!())) ], vec!["lines".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([("args".into(), Value::Assoc(HashMap::from([( "lines".into(), Value::List(vec![Value::String("a".into()), Value::String("b".into())])) ]))), ("result".into(), Value::String("a\nb".into()))]))
        ]
    )); }
    if names.contains("StartsWith") { specs.push(tool_spec!(
        "StartsWith", summary: "Prefix test", params: ["s","prefix"], tags: ["string","search"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()), ("prefix".into(), schema_str!()) ], vec!["s".into(), "prefix".into()]), output_schema: schema_bool!(), examples: [
            Value::Assoc(HashMap::from([("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("hello".into())), ("prefix".into(), Value::String("he".into())) ]))), ("result".into(), Value::Boolean(true))]))
        ]
    )); }
    if names.contains("EndsWith") { specs.push(tool_spec!(
        "EndsWith", summary: "Suffix test", params: ["s","suffix"], tags: ["string","search"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()), ("suffix".into(), schema_str!()) ], vec!["s".into(), "suffix".into()]), output_schema: schema_bool!(), examples: [
            Value::Assoc(HashMap::from([("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("hello".into())), ("suffix".into(), Value::String("lo".into())) ]))), ("result".into(), Value::Boolean(true))]))
        ]
    )); }
    if names.contains("StringReplace") { specs.push(tool_spec!(
        "StringReplace", summary: "Replace all occurrences", params: ["s","from","to"], tags: ["string","replace"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()), ("from".into(), schema_str!()), ("to".into(), schema_str!()) ], vec!["s".into(), "from".into(), "to".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("a-b-a".into())), ("from".into(), Value::String("a".into())), ("to".into(), Value::String("x".into())) ]))), ("result".into(), Value::String("x-b-x".into()))]))
        ]
    )); }
    if names.contains("StringReplaceFirst") { specs.push(tool_spec!(
        "StringReplaceFirst", summary: "Replace first occurrence", params: ["s","from","to"], tags: ["string","replace"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()), ("from".into(), schema_str!()), ("to".into(), schema_str!()) ], vec!["s".into(), "from".into(), "to".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("a-b-a".into())), ("from".into(), Value::String("a".into())), ("to".into(), Value::String("x".into())) ]))), ("result".into(), Value::String("x-b-a".into()))]))
        ]
    )); }
    if names.contains("StringReverse") { specs.push(tool_spec!(
        "StringReverse", summary: "Reverse string", params: ["s"], tags: ["string"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()) ], vec!["s".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("abc".into())) ]))), ("result".into(), Value::String("cba".into())) ]))
        ]
    )); }
    if names.contains("StringPadLeft") { specs.push(tool_spec!(
        "StringPadLeft", summary: "Pad on the left to width", params: ["s","width","ch?"], tags: ["string","pad"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()), ("width".into(), schema_int!()), ("ch?".into(), schema_str!()) ], vec!["s".into(), "width".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("x".into())), ("width".into(), Value::Integer(3)), ("ch?".into(), Value::String("0".into())) ]))), ("result".into(), Value::String("00x".into())) ]))
        ]
    )); }
    if names.contains("StringPadRight") { specs.push(tool_spec!(
        "StringPadRight", summary: "Pad on the right to width", params: ["s","width","ch?"], tags: ["string","pad"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()), ("width".into(), schema_int!()), ("ch?".into(), schema_str!()) ], vec!["s".into(), "width".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("x".into())), ("width".into(), Value::Integer(3)), ("ch?".into(), Value::String(".".into())) ]))), ("result".into(), Value::String("x..".into())) ]))
        ]
    )); }
    if names.contains("StringSlice") { specs.push(tool_spec!(
        "StringSlice", summary: "Substring by start and optional length (0-based, UTF-8 chars)", params: ["s","start","len?"], tags: ["string","slice"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()), ("start".into(), schema_int!()), ("len?".into(), schema_int!()) ], vec!["s".into(), "start".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("abcdef".into())), ("start".into(), Value::Integer(1)), ("len?".into(), Value::Integer(3)) ]))), ("result".into(), Value::String("bcd".into())) ]))
        ]
    )); }
    if names.contains("IndexOf") { specs.push(tool_spec!(
        "IndexOf", summary: "Index of substring (0-based, -1 if not found)", params: ["s","substr","from?"], tags: ["string","search"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()), ("substr".into(), schema_str!()), ("from?".into(), schema_int!()) ], vec!["s".into(), "substr".into()]), output_schema: schema_int!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("banana".into())), ("substr".into(), Value::String("na".into())) ]))), ("result".into(), Value::Integer(2)) ]))
        ]
    )); }
    if names.contains("LastIndexOf") { specs.push(tool_spec!(
        "LastIndexOf", summary: "Last index of substring (0-based, -1 if not found)", params: ["s","substr","from?"], tags: ["string","search"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()), ("substr".into(), schema_str!()), ("from?".into(), schema_int!()) ], vec!["s".into(), "substr".into()]), output_schema: schema_int!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("banana".into())), ("substr".into(), Value::String("na".into())) ]))), ("result".into(), Value::Integer(4)) ]))
        ]
    )); }
    if names.contains("StringRepeat") { specs.push(tool_spec!(
        "StringRepeat", summary: "Repeat string n times", params: ["s","n"], tags: ["string"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()), ("n".into(), schema_int!()) ], vec!["s".into(), "n".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("ab".into())), ("n".into(), Value::Integer(3)) ]))), ("result".into(), Value::String("ababab".into())) ]))
        ]
    )); }
    if names.contains("IsBlank") { specs.push(tool_spec!(
        "IsBlank", summary: "True if string is empty or whitespace", params: ["s"], tags: ["string","predicate"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()) ], vec!["s".into()]), output_schema: schema_bool!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("  ".into())) ]))), ("result".into(), Value::Boolean(true)) ]))
        ]
    )); }
    if names.contains("Capitalize") { specs.push(tool_spec!(
        "Capitalize", summary: "Capitalize first character; rest lowercased", params: ["s"], tags: ["string","case"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()) ], vec!["s".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("hELLO".into())) ]))), ("result".into(), Value::String("Hello".into())) ]))
        ]
    )); }
    if names.contains("TitleCase") { specs.push(tool_spec!(
        "TitleCase", summary: "Capitalize words; lowercase others", params: ["s"], tags: ["string","case"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()) ], vec!["s".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("hello world".into())) ]))), ("result".into(), Value::String("Hello World".into())) ]))
        ]
    )); }
    if names.contains("EqualsIgnoreCase") { specs.push(tool_spec!(
        "EqualsIgnoreCase", summary: "Case-insensitive string equality", params: ["a","b"], tags: ["string","compare"],
        input_schema: schema_object_value(vec![ ("a".into(), schema_str!()), ("b".into(), schema_str!()) ], vec!["a".into(), "b".into()]), output_schema: schema_bool!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "a".into(), Value::String("Hello".into())), ("b".into(), Value::String("hello".into())) ]))), ("result".into(), Value::Boolean(true)) ]))
        ]
    )); }
    if names.contains("StringChars") { specs.push(tool_spec!(
        "StringChars", summary: "Split into Unicode scalar chars", params: ["s"], tags: ["string"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()) ], vec!["s".into()]), output_schema: schema_arr!(schema_str!()), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("hé".into())) ]))), ("result".into(), Value::List(vec![Value::String("h".into()), Value::String("é".into())])) ]))
        ]
    )); }
    if names.contains("StringFromChars") { specs.push(tool_spec!(
        "StringFromChars", summary: "Join Unicode scalar chars", params: ["chars"], tags: ["string"],
        input_schema: schema_object_value(vec![ ("chars".into(), schema_arr!(schema_str!())) ], vec!["chars".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "chars".into(), Value::List(vec![Value::String("h".into()), Value::String("é".into())])) ]))), ("result".into(), Value::String("hé".into())) ]))
        ]
    )); }
    // Regex helpers
    if names.contains("RegexMatch") { specs.push(tool_spec!(
        "RegexMatch", summary: "Regex match predicate", params: ["s","pattern"], tags: ["string","regex"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()), ("pattern".into(), schema_str!()) ], vec!["s".into(), "pattern".into()]), output_schema: schema_bool!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("abc123".into())), ("pattern".into(), Value::String("[a-z]+\\d+".into())) ]))), ("result".into(), Value::Boolean(true)) ]))
        ]
    )); }
    if names.contains("RegexIsMatch") { specs.push(tool_spec!(
        "RegexIsMatch", summary: "Alias: regex match predicate", params: ["s","pattern"], tags: ["string","regex"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()), ("pattern".into(), schema_str!()) ], vec!["s".into(), "pattern".into()]), output_schema: schema_bool!()
    )); }
    if names.contains("RegexFind") { specs.push(tool_spec!(
        "RegexFind", summary: "Find first regex match", params: ["s","pattern"], tags: ["string","regex"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()), ("pattern".into(), schema_str!()) ], vec!["s".into(), "pattern".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("abc123".into())), ("pattern".into(), Value::String("\\d+".into())) ]))), ("result".into(), Value::String("123".into())) ]))
        ]
    )); }
    if names.contains("RegexFindAll") { specs.push(tool_spec!(
        "RegexFindAll", summary: "Find all regex matches", params: ["s","pattern"], tags: ["string","regex"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()), ("pattern".into(), schema_str!()) ], vec!["s".into(), "pattern".into()]), output_schema: schema_arr!(schema_str!()), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("a1 b22 c333".into())), ("pattern".into(), Value::String("\\d+".into())) ]))), ("result".into(), Value::List(vec![Value::String("1".into()), Value::String("22".into()), Value::String("333".into())])) ]))
        ]
    )); }
    if names.contains("RegexReplace") { specs.push(tool_spec!(
        "RegexReplace", summary: "Replace by regex pattern", params: ["s","pattern","repl"], tags: ["string","regex"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()), ("pattern".into(), schema_str!()), ("repl".into(), schema_str!()) ], vec!["s".into(), "pattern".into(), "repl".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("a1 b2".into())), ("pattern".into(), Value::String("\\d".into())), ("repl".into(), Value::String("#".into())) ]))), ("result".into(), Value::String("a# b#".into())) ]))
        ]
    )); }
    // Formatting and interpolation
    if names.contains("StringFormat") { specs.push(tool_spec!(
        "StringFormat", summary: "Positional format using {0},{1}", params: ["template","args"], tags: ["string","format"],
        input_schema: schema_object_value(vec![ ("template".into(), schema_str!()), ("args".into(), schema_arr!(Value::Assoc(HashMap::new()))) ], vec!["template".into(), "args".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "template".into(), Value::String("Hello {0}".into())), ("args".into(), Value::List(vec![Value::String("Lyra".into())])) ]))), ("result".into(), Value::String("Hello Lyra".into())) ]))
        ]
    )); }
    if names.contains("StringFormatMap") { specs.push(tool_spec!(
        "StringFormatMap", summary: "Named format using {key}", params: ["template","map"], tags: ["string","format"],
        input_schema: schema_object_value(vec![ ("template".into(), schema_str!()), ("map".into(), Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("object")))]))) ], vec!["template".into(), "map".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "template".into(), Value::String("Hello {name}".into())), ("map".into(), Value::Assoc(HashMap::from([(String::from("name"), Value::String(String::from("Lyra")))]))) ]))), ("result".into(), Value::String("Hello Lyra".into())) ]))
        ]
    )); }
    if names.contains("StringInterpolate") { specs.push(tool_spec!(
        "StringInterpolate", summary: "Evaluate expressions inside braces", params: ["s"], tags: ["string","template"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()) ], vec!["s".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("2+2={Plus[2,2]}".into())) ]))), ("result".into(), Value::String("2+2=4".into())) ]))
        ]
    )); }
    if names.contains("StringInterpolateWith") { specs.push(tool_spec!(
        "StringInterpolateWith", summary: "Interpolate using provided variables", params: ["s","vars"], tags: ["string","template"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()), ("vars".into(), Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("object")))]))) ], vec!["s".into(), "vars".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("Hello {name}".into())), ("vars".into(), Value::Assoc(HashMap::from([(String::from("name"), Value::String(String::from("Lyra")))]))) ]))), ("result".into(), Value::String("Hello Lyra".into())) ]))
        ]
    )); }
    if names.contains("TemplateRender") { specs.push(tool_spec!(
        "TemplateRender", summary: "Render Mustache-like template", params: ["template","data","opts?"], tags: ["string","template"],
        input_schema: schema_object_value(vec![ ("template".into(), schema_str!()), ("data".into(), Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("object")))]))), ("opts?".into(), Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("object")))]))) ], vec!["template".into(), "data".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([
                ("template".into(), Value::String("Hello {{name}}".into())), ("data".into(), Value::Assoc(HashMap::from([(String::from("name"), Value::String(String::from("Lyra")))])))
            ]))), ("result".into(), Value::String("Hello Lyra".into())) ]))
        ]
    )); }
    // Encoders/decoders and transforms
    if names.contains("HtmlEscape") { specs.push(tool_spec!(
        "HtmlEscape", summary: "Escape HTML special characters.", params: ["s"], tags: ["string","html","escape"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()) ], vec!["s".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("a<b".into())) ]))), ("result".into(), Value::String("a&lt;b".into())) ]))
        ]
    )); }
    if names.contains("HtmlUnescape") { specs.push(tool_spec!(
        "HtmlUnescape", summary: "Unescape HTML entities.", params: ["s"], tags: ["string","html","unescape"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()) ], vec!["s".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("a&amp;b".into())) ]))), ("result".into(), Value::String("a&b".into())) ]))
        ]
    )); }
    if names.contains("UrlEncode") { specs.push(tool_spec!(
        "UrlEncode", summary: "Percent-encode string for URLs", params: ["s"], tags: ["string","url","encode"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()) ], vec!["s".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("hello world".into())) ]))), ("result".into(), Value::String("hello%20world".into())) ]))
        ]
    )); }
    if names.contains("UrlDecode") { specs.push(tool_spec!(
        "UrlDecode", summary: "Decode percent-encoded URL string", params: ["s"], tags: ["string","url","decode"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()) ], vec!["s".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("hello%20world".into())) ]))), ("result".into(), Value::String("hello world".into())) ]))
        ]
    )); }
    if names.contains("UrlFormEncode") { specs.push(tool_spec!(
        "UrlFormEncode", summary: "Form-url-encode string", params: ["s"], tags: ["string","form","encode"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()) ], vec!["s".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("a b".into())) ]))), ("result".into(), Value::String("a+b".into())) ]))
        ]
    )); }
    if names.contains("UrlFormDecode") { specs.push(tool_spec!(
        "UrlFormDecode", summary: "Form-url-decode string", params: ["s"], tags: ["string","form","decode"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()) ], vec!["s".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("a+b".into())) ]))), ("result".into(), Value::String("a b".into())) ]))
        ]
    )); }
    if names.contains("JsonEscape") { specs.push(tool_spec!(
        "JsonEscape", summary: "Escape string for JSON", params: ["s"], tags: ["string","json","escape"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()) ], vec!["s".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("a\n".into())) ]))), ("result".into(), Value::String("a\\n".into())) ]))
        ]
    )); }
    if names.contains("JsonUnescape") { specs.push(tool_spec!(
        "JsonUnescape", summary: "Unescape JSON string", params: ["s"], tags: ["string","json","unescape"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()) ], vec!["s".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("a\\n".into())) ]))), ("result".into(), Value::String("a\n".into())) ]))
        ]
    )); }
    if names.contains("Slugify") { specs.push(tool_spec!(
        "Slugify", summary: "Convert text to lowercase URL slug", params: ["s"], tags: ["string","slug"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()) ], vec!["s".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("Hello, World!".into())) ]))), ("result".into(), Value::String("hello-world".into())) ]))
        ]
    )); }
    if names.contains("StringTruncate") { specs.push(tool_spec!(
        "StringTruncate", summary: "Truncate to max length with suffix", params: ["s","max","suffix?"], tags: ["string","truncate"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()), ("max".into(), schema_int!()), ("suffix?".into(), schema_str!()) ], vec!["s".into(), "max".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("abcdef".into())), ("max".into(), Value::Integer(4)), ("suffix?".into(), Value::String("…".into())) ]))), ("result".into(), Value::String("abc…".into())) ]))
        ]
    )); }
    if names.contains("CamelCase") { specs.push(tool_spec!(
        "CamelCase", summary: "Convert to lower camelCase", params: ["s"], tags: ["string","case"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()) ], vec!["s".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("Hello world".into())) ]))), ("result".into(), Value::String("helloWorld".into())) ]))
        ]
    )); }
    if names.contains("SnakeCase") { specs.push(tool_spec!(
        "SnakeCase", summary: "Convert to snake_case", params: ["s"], tags: ["string","case"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()) ], vec!["s".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("Hello world".into())) ]))), ("result".into(), Value::String("hello_world".into())) ]))
        ]
    )); }
    if names.contains("KebabCase") { specs.push(tool_spec!(
        "KebabCase", summary: "Convert to kebab-case", params: ["s"], tags: ["string","case"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()) ], vec!["s".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([( "s".into(), Value::String("Hello world".into())) ]))), ("result".into(), Value::String("hello-world".into())) ]))
        ]
    )); }
    // Dates
    if names.contains("ParseDate") { specs.push(tool_spec!(
        "ParseDate", summary: "Parse date/time to Unix seconds", params: ["s","format?"], tags: ["string","date"],
        input_schema: schema_object_value(vec![ ("s".into(), schema_str!()), ("format?".into(), schema_str!()) ], vec!["s".into()]), output_schema: schema_int!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([ ("s".into(), Value::String("2024-08-01 12:00:00".into())) ]))), ("result".into(), Value::Integer(1722513600)) ]))
        ]
    )); }
    if names.contains("FormatDate") { specs.push(tool_spec!(
        "FormatDate", summary: "Format Unix seconds to string", params: ["ts","format?"], tags: ["string","date"],
        input_schema: schema_object_value(vec![ ("ts".into(), schema_int!()), ("format?".into(), schema_str!()) ], vec!["ts".into()]), output_schema: schema_str!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([ ("ts".into(), Value::Integer(0)) ]))), ("result".into(), Value::String("1970-01-01 00:00:00".into())) ]))
        ]
    )); }
    if names.contains("DateDiff") { specs.push(tool_spec!(
        "DateDiff", summary: "Difference between two dates or timestamps", params: ["a","b","unit?"], tags: ["string","date"],
        input_schema: schema_object_value(vec![ ("a".into(), Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("string")))]))), ("b".into(), Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("string")))]))), ("unit?".into(), schema_str!()) ], vec!["a".into(), "b".into()]), output_schema: schema_num!(), examples: [
            Value::Assoc(HashMap::from([ ("args".into(), Value::Assoc(HashMap::from([ ("a".into(), Value::String("2024-08-02".into())), ("b".into(), Value::String("2024-08-01".into())), ("unit?".into(), Value::String("days".into())) ]))), ("result".into(), Value::Integer(1)) ]))
        ]
    )); }
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
    // Optional global capability gate; always apply if set, and allow explicit override via filter.capabilities
    let granted_caps: std::collections::HashSet<String> = CAPABILITIES.get_or_init(|| Mutex::new(None)).lock().unwrap().as_ref().cloned().unwrap_or_default();
    // If filter provided, parse user filters
    let (eff_filter, caps_filter, tag_filter): (
        std::collections::HashSet<String>,
        std::collections::HashSet<String>,
        std::collections::HashSet<String>,
    ) = match &filter {
        Value::Assoc(f) if !f.is_empty() => (
            match f.get("effects") { Some(Value::List(vs))=>vs.iter().filter_map(value_to_string).map(|s| s.to_lowercase()).collect(), _=>Default::default() },
            match f.get("capabilities") { Some(Value::List(vs))=>vs.iter().filter_map(value_to_string).map(|s| s.to_lowercase()).collect(), _=>Default::default() },
            match f.get("tags") { Some(Value::List(vs))=>vs.iter().filter_map(value_to_string).map(|s| s.to_lowercase()).collect(), _=>Default::default() },
        ),
        _ => (Default::default(), Default::default(), Default::default()),
    };
    let want_caps = if !caps_filter.is_empty() { &caps_filter } else { &granted_caps };
    let filtered: Vec<Value> = out.into_iter().filter(|v| {
        if let Value::Assoc(m) = v {
            // capability/effects gate
            if !want_caps.is_empty() {
                let mut effs: std::collections::HashSet<String> = Default::default();
                if let Some(Value::List(effs_v)) = m.get("effects") { for e in effs_v { if let Some(s)=value_to_string(e) { effs.insert(s.to_lowercase()); } } }
                if !effs.is_subset(want_caps) { return false; }
            }
            // explicit effects filter
            if !eff_filter.is_empty() {
                let mut have: std::collections::HashSet<String> = Default::default();
                if let Some(Value::List(effs)) = m.get("effects") { for e in effs { if let Some(s)=value_to_string(e) { have.insert(s.to_lowercase()); } } }
                if !eff_filter.is_subset(&have) { return false; }
            }
            // tags filter
            if !tag_filter.is_empty() {
                let mut have: std::collections::HashSet<String> = Default::default();
                if let Some(Value::List(tags)) = m.get("tags") { for t in tags { if let Some(s)=value_to_string(t) { have.insert(s.to_lowercase()); } } }
                if tag_filter.is_disjoint(&have) { return false; }
            }
            true
        } else { false }
    }).collect();
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
    let opts = if args.len()>=3 { ev.eval(args[2].clone()) } else { Value::Assoc(HashMap::new()) };
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
    let expr = Value::Expr { head: Box::new(Value::Symbol(name.clone())), args: call_args.clone() };

    // Cache policy: options may include <|"Cache"->"session", "IdempotencyKey"->"..."|>
    let mut cache_enabled = false;
    let mut id_key: Option<String> = None;
    if let Value::Assoc(m) = &opts {
        if let Some(Value::String(s)) = m.get("Cache") { if s=="session" { cache_enabled = true; } }
        if let Some(Value::String(s)) = m.get("IdempotencyKey") { id_key = Some(s.clone()); }
    }
    if cache_enabled || id_key.is_some() {
        let mut hasher = DefaultHasher::new();
        if let Some(idk) = id_key.clone() { idk.hash(&mut hasher); } else {
            name.hash(&mut hasher);
            // Include provided assoc/list as part of key by pretty formatting (session-only stability acceptable)
            let key_v = if let Value::Assoc(_) = provided { provided.clone() } else { Value::List(call_args.clone()) };
            let s = lyra_core::pretty::format_value(&key_v);
            s.hash(&mut hasher);
        }
        let h = hasher.finish();
        if let Some(v) = tool_cache().lock().unwrap().get(&h) { return v.clone(); }
        let out = ev.eval(expr);
        tool_cache().lock().unwrap().insert(h, out.clone());
        return out;
    }
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
    // Merge registered specs, stdlib defaults, and builtin fallback cards (like ToolsList)
    let mut out: Vec<Value> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    // Explicitly registered specs
    {
        let reg = tool_reg().lock().unwrap();
        for (id, spec) in reg.iter() {
            out.push(spec.clone());
            seen.insert(id.clone());
        }
    }
    // Stdlib default specs for present functions
    for spec in stdlib_default_specs(ev) {
        if let Value::Assoc(m) = &spec {
            if let Some(id) = get_str(m, "id").or_else(|| get_str(m, "name")) {
                if !seen.contains(&id) { out.push(spec); seen.insert(id); }
            }
        }
    }
    // Fallback builtin cards
    for card in builtin_cards(ev) {
        if let Value::Assoc(m) = &card {
            if let Some(id) = get_str(m, "id").or_else(|| get_str(m, "name")) {
                if !seen.contains(&id) { out.push(card); seen.insert(id); }
            }
        }
    }
    Value::List(out)
}

fn tools_cache_clear(_ev: &mut Evaluator, _args: Vec<Value>) -> Value {
    tool_cache().lock().unwrap().clear();
    Value::Boolean(true)
}

fn idempotency_key(_ev: &mut Evaluator, _args: Vec<Value>) -> Value {
    Value::String(format!("idemp-{}", next_idemp()))
}


pub fn register_tools_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str)->bool) {
    register_if(ev, pred, "ToolsRegister", tools_register as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "ToolsUnregister", tools_unregister as NativeFn, Attributes::empty());
    register_if(ev, pred, "ToolsList", tools_list as NativeFn, Attributes::empty());
    register_if(ev, pred, "ToolsCards", tools_cards as NativeFn, Attributes::empty());
    register_if(ev, pred, "ToolsDescribe", tools_describe as NativeFn, Attributes::empty());
    register_if(ev, pred, "ToolsSearch", tools_search as NativeFn, Attributes::empty());
    register_if(ev, pred, "ToolsResolve", tools_resolve as NativeFn, Attributes::empty());
    register_if(ev, pred, "ToolsInvoke", tools_invoke as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "ToolsDryRun", tools_dry_run as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "ToolsExportOpenAI", tools_export_openai as NativeFn, Attributes::empty());
    register_if(ev, pred, "ToolsExportBundle", tools_export_bundle as NativeFn, Attributes::empty());
    register_if(ev, pred, "ToolsSetCapabilities", tools_set_capabilities as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "ToolsGetCapabilities", tools_get_capabilities as NativeFn, Attributes::empty());
}
