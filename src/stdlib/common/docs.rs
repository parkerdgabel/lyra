//! Function documentation primitives surfaced via Describe/Help

use std::collections::HashMap;
use crate::vm::Value;
use super::result::assoc;

#[derive(Debug, Clone)]
pub struct FunctionDoc {
    pub name: String,
    pub summary: String,
    pub signature: String,
    pub attributes: Vec<String>,
    pub options: HashMap<String, String>, // name -> description
    pub examples: Vec<String>,
}

impl FunctionDoc {
    pub fn to_association(&self) -> Value {
        let opts = self.options.iter()
            .map(|(k,v)| (k.clone(), Value::String(v.clone())))
            .collect::<HashMap<_,_>>();
        assoc(vec![
            ("name", Value::String(self.name.clone())),
            ("summary", Value::String(self.summary.clone())),
            ("signature", Value::String(self.signature.clone())),
            ("attributes", Value::List(self.attributes.iter().cloned().map(Value::String).collect())),
            ("options", Value::Object(opts)),
            ("examples", Value::List(self.examples.iter().cloned().map(Value::String).collect())),
        ])
    }
}

