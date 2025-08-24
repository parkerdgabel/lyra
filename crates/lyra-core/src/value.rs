use std::collections::HashMap;
use serde::{Serialize, Deserialize};

pub type AssocMap = HashMap<String, Value>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Integer(i64),
    Real(f64),
    BigReal(String),
    Rational { num: i64, den: i64 },
    Complex { re: Box<Value>, im: Box<Value> },
    PackedArray { shape: Vec<usize>, data: Vec<f64> },
    String(String),
    Symbol(String),
    Boolean(bool),
    List(Vec<Value>),
    Assoc(AssocMap),
    Expr { head: Box<Value>, args: Vec<Value> },
    Slot(Option<usize>),
    PureFunction { params: Option<Vec<String>>, body: Box<Value> },
}

impl Value {
    pub fn symbol<S: Into<String>>(s: S) -> Self { Value::Symbol(s.into()) }
    pub fn list(items: Vec<Value>) -> Self { Value::List(items) }
    pub fn assoc(pairs: Vec<(impl Into<String>, Value)>) -> Self {
        let mut m = AssocMap::with_capacity(pairs.len());
        for (k, v) in pairs { m.insert(k.into(), v); }
        Value::Assoc(m)
    }
    pub fn expr(head: Value, args: Vec<Value>) -> Self { Value::Expr { head: Box::new(head), args } }
    pub fn slot(n: Option<usize>) -> Self { Value::Slot(n) }
    pub fn pure_function(params: Option<Vec<String>>, body: Value) -> Self { Value::PureFunction { params, body: Box::new(body) } }
}
