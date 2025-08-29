use crate::attrs::Attributes;
use crate::eval::Evaluator;
use lyra_core::value::Value;

/// Register introspection helpers: `DescribeBuiltins` and `Documentation`.
pub fn register_introspection(ev: &mut Evaluator) {
    fn describe_builtins_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
        if !args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("DescribeBuiltins".into())), args }; }
        let mut out: Vec<Value> = Vec::new();
        let snapshot: Vec<(String, Attributes)> = ev.builtins_snapshot();
        for (name, attrs) in snapshot.into_iter() {
            let mut attr_list: Vec<Value> = Vec::new();
            if attrs.contains(Attributes::LISTABLE) { attr_list.push(Value::String("LISTABLE".into())); }
            if attrs.contains(Attributes::FLAT) { attr_list.push(Value::String("FLAT".into())); }
            if attrs.contains(Attributes::ORDERLESS) { attr_list.push(Value::String("ORDERLESS".into())); }
            if attrs.contains(Attributes::HOLD_ALL) { attr_list.push(Value::String("HOLD_ALL".into())); }
            if attrs.contains(Attributes::HOLD_FIRST) { attr_list.push(Value::String("HOLD_FIRST".into())); }
            if attrs.contains(Attributes::HOLD_REST) { attr_list.push(Value::String("HOLD_REST".into())); }
            if attrs.contains(Attributes::ONE_IDENTITY) { attr_list.push(Value::String("ONE_IDENTITY".into())); }
            let (summary, params): (String, Vec<String>) = ev.get_doc(&name).unwrap_or((String::new(), Vec::new()));
            let card = Value::Assoc(vec![
                ("id".to_string(), Value::String(name.clone())),
                ("name".to_string(), Value::String(name.clone())),
                ("summary".to_string(), Value::String(summary)),
                ("tags".to_string(), Value::List(vec![])),
                ("params".to_string(), Value::List(params.into_iter().map(Value::String).collect())),
                ("attributes".to_string(), Value::List(attr_list)),
                ("examples".to_string(), Value::List(match ev.get_doc_full(&name) { Some(ent) => ent.examples().into_iter().map(Value::String).collect::<Vec<_>>(), None => vec![] })),
            ].into_iter().collect());
            out.push(card);
        }
        Value::List(out)
    }
    ev.register("DescribeBuiltins", describe_builtins_fn as crate::eval::NativeFn, Attributes::empty());

    fn documentation_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
        if args.is_empty() { let head = Value::Symbol("DescribeBuiltins".to_string()); let expr = Value::Expr { head: Box::new(head), args: vec![] }; return ev.eval(expr); }
        if args.len() == 1 {
            let name = match &args[0] { Value::String(s) | Value::Symbol(s) => s.clone(), other => match ev.eval(other.clone()) { Value::String(s) | Value::Symbol(s) => s, _ => String::new(), }, };
            let head = Value::Symbol("DescribeBuiltins".to_string());
            let desc = ev.eval(Value::Expr { head: Box::new(head), args: vec![] });
            if let Value::List(items) = desc {
                for it in items {
                    if let Value::Assoc(mut m) = it {
                        if let Some(Value::String(n)) = m.get("name").cloned() { if n == name {
                            if let Some((sum, params)) = ev.get_doc(&n) {
                                m.insert("summary".into(), Value::String(sum));
                                m.insert("params".into(), Value::List(params.into_iter().map(Value::String).collect()));
                                if let Some(ent) = ev.get_doc_full(&n) { m.insert("examples".into(), Value::List(ent.examples().into_iter().map(Value::String).collect())); }
                            }
                            return Value::Assoc(m);
                        }}
                    }
                }
            }
            if let Some((sum, params)) = ev.get_doc(&name) {
                let examples = ev.get_doc_full(&name).map(|e| e.examples()).unwrap_or_default();
                return Value::Assoc(vec![
                    ("id".to_string(), Value::String(name.clone())),
                    ("name".to_string(), Value::String(name.clone())),
                    ("summary".to_string(), Value::String(sum)),
                    ("params".to_string(), Value::List(params.into_iter().map(Value::String).collect())),
                    ("attributes".to_string(), Value::List(vec![])),
                    ("examples".to_string(), Value::List(examples.into_iter().map(Value::String).collect())),
                ].into_iter().collect());
            }
            return Value::Assoc(std::collections::HashMap::new());
        }
        Value::Expr { head: Box::new(Value::Symbol("Documentation".into())), args }
    }
    ev.register("Documentation", documentation_fn as crate::eval::NativeFn, Attributes::empty());
}
