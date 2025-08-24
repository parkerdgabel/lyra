//! Core language utilities: construct and evaluate Expr values
use crate::vm::{Value, VmError, VmResult};
use crate::stdlib::StandardLibrary;

/// Expr[head, args...] → Value::Expr
pub fn expr_constructor(args: &[Value]) -> VmResult<Value> {
    if args.is_empty() {
        return Err(VmError::TypeError { expected: "Expr[head, args...]".into(), actual: "0 args".into() });
    }
    let head = args[0].clone();
    let rest = if args.len() > 1 { args[1..].to_vec() } else { vec![] };
    Ok(Value::Expr { head: Box::new(head), args: rest })
}

/// Eval[value] → evaluates Value::Expr to a stdlib call if possible, recursively
pub fn eval_function(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError { expected: "Eval[value]".into(), actual: format!("{} args", args.len()) });
    }
    fn eval_once(v: Value, stdlib: &StandardLibrary) -> VmResult<Value> {
        match v {
            Value::Expr { head, args } => {
                // Evaluate arguments first
                let mut eval_args = Vec::with_capacity(args.len());
                for a in args { eval_args.push(eval_once(a, stdlib)?); }
                // Resolve head name
                let name_opt = match *head {
                    Value::Symbol(s) | Value::Function(s) => Some(s),
                    _ => None,
                };
                if let Some(name) = name_opt {
                    if let Some(f) = stdlib.get_function(&name) {
                        return f(&eval_args);
                    }
                }
                Ok(Value::Expr { head, args: eval_args })
            }
            // Optionally, walk lists/objects; keep minimal for now
            other => Ok(other),
        }
    }
    let stdlib = StandardLibrary::new();
    eval_once(args[0].clone(), &stdlib)
}

