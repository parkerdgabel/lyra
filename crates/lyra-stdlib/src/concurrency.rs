use lyra_runtime::Evaluator;
#[cfg(feature = "tools")] use crate::tools::{add_specs, schema_object_value};
#[cfg(feature = "tools")] use crate::{tool_spec, schema_int, schema_bool};
#[cfg(feature = "tools")] use lyra_core::value::Value;
#[cfg(feature = "tools")] use std::collections::HashMap;

pub fn register_concurrency(ev: &mut Evaluator) {
    lyra_runtime::eval::register_concurrency(ev);
    #[cfg(feature = "tools")]
    add_specs(vec![
        tool_spec!("Future", summary: "Run expression asynchronously", params: ["expr","opts"], tags: ["concurrency","async"], input_schema: schema_object_value(vec![
            (String::from("opts"), Value::Assoc(HashMap::from([
                (String::from("type"), Value::String(String::from("object"))),
                (String::from("properties"), Value::Assoc(HashMap::from([
                    (String::from("MaxThreads"), schema_int!()),
                    (String::from("TimeBudgetMs"), schema_int!()),
                ]))),
            ]))),
        ], vec![])),
        tool_spec!("Await", summary: "Wait for a Future to complete", params: ["future"], tags: ["concurrency","async"]),
        tool_spec!("ParallelMap", summary: "Map function over list in parallel", params: ["fn","list","opts"], tags: ["concurrency","parallel"], input_schema: schema_object_value(vec![
            (String::from("opts"), Value::Assoc(HashMap::from([
                (String::from("type"), Value::String(String::from("object"))),
                (String::from("properties"), Value::Assoc(HashMap::from([
                    (String::from("MaxThreads"), schema_int!()),
                    (String::from("TimeBudgetMs"), schema_int!()),
                ]))),
            ]))),
        ], vec![])),
        tool_spec!("ParallelTable", summary: "Build table by mapping function in parallel", params: ["fn","list","opts"], tags: ["concurrency","parallel"], input_schema: schema_object_value(vec![
            (String::from("opts"), Value::Assoc(HashMap::from([
                (String::from("type"), Value::String(String::from("object"))),
                (String::from("properties"), Value::Assoc(HashMap::from([
                    (String::from("MaxThreads"), schema_int!()),
                    (String::from("TimeBudgetMs"), schema_int!()),
                ]))),
            ]))),
        ], vec![])),
        tool_spec!("MapAsync", summary: "Launch async map, returns list of futures", params: ["fn","list"], tags: ["concurrency","async"], output_schema: Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("array")))]))),
        tool_spec!("Gather", summary: "Wait for list of futures, return results", params: ["futures"], tags: ["concurrency","async"], output_schema: Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("array")))]))),
        tool_spec!("Scope", summary: "Run expression in bounded scope (threads/deadline)", params: ["opts","expr"], tags: ["concurrency","scope"], input_schema: schema_object_value(vec![
            (String::from("opts"), Value::Assoc(HashMap::from([
                (String::from("type"), Value::String(String::from("object"))),
                (String::from("properties"), Value::Assoc(HashMap::from([
                    (String::from("MaxThreads"), schema_int!()),
                    (String::from("TimeBudgetMs"), schema_int!()),
                ]))),
            ]))),
        ], vec![])),
        tool_spec!("StartScope", summary: "Start a scope and return id", params: ["opts","expr"], tags: ["concurrency","scope"], input_schema: schema_object_value(vec![
            (String::from("opts"), Value::Assoc(HashMap::from([
                (String::from("type"), Value::String(String::from("object"))),
                (String::from("properties"), Value::Assoc(HashMap::from([
                    (String::from("MaxThreads"), schema_int!()),
                    (String::from("TimeBudgetMs"), schema_int!()),
                ]))),
            ]))),
        ], vec![])),
        tool_spec!("InScope", summary: "Run expression inside an existing scope", params: ["scopeId","expr"], tags: ["concurrency","scope"]),
        tool_spec!("CancelScope", summary: "Cancel a running scope", params: ["scopeId"], tags: ["concurrency","scope"]),
        tool_spec!("EndScope", summary: "End a running scope", params: ["scopeId"], tags: ["concurrency","scope"]),
        tool_spec!("BoundedChannel", summary: "Create a bounded channel", params: ["cap"], tags: ["concurrency","channel"]),
        tool_spec!("Send", summary: "Send a value to channel (blocking)", params: ["channel","value"], tags: ["concurrency","channel"]),
        tool_spec!("Receive", summary: "Receive a value from channel (blocking)", params: ["channel"], tags: ["concurrency","channel"]),
        tool_spec!("CloseChannel", summary: "Close the channel", params: ["channel"], tags: ["concurrency","channel"]),
        tool_spec!("TrySend", summary: "Try to send with optional timeout", params: ["channel","value","opts"], tags: ["concurrency","channel"], input_schema: schema_object_value(vec![
            (String::from("opts"), Value::Assoc(HashMap::from([
                (String::from("type"), Value::String(String::from("object"))),
                (String::from("properties"), Value::Assoc(HashMap::from([
                    (String::from("TimeoutMs"), schema_int!()),
                    (String::from("NonBlocking"), schema_bool!()),
                ]))),
            ]))),
        ], vec![]), output_schema: Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("boolean")))]))),
        tool_spec!("TryReceive", summary: "Try to receive with optional timeout", params: ["channel","opts"], tags: ["concurrency","channel"], input_schema: schema_object_value(vec![
            (String::from("opts"), Value::Assoc(HashMap::from([
                (String::from("type"), Value::String(String::from("object"))),
                (String::from("properties"), Value::Assoc(HashMap::from([
                    (String::from("TimeoutMs"), schema_int!()),
                    (String::from("NonBlocking"), schema_bool!()),
                ]))),
            ]))),
        ], vec![])),
        tool_spec!("Actor", summary: "Start an actor from handler function", params: ["fn","opts"], tags: ["concurrency","actor"]),
        tool_spec!("Tell", summary: "Send message to actor (fire-and-forget)", params: ["actor","msg"], tags: ["concurrency","actor"]),
        tool_spec!("Ask", summary: "Send message to actor and await response", params: ["actor","msg"], tags: ["concurrency","actor"]),
        tool_spec!("StopActor", summary: "Stop the actor", params: ["actor"], tags: ["concurrency","actor"]),
    ]);
}

pub fn register_concurrency_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str)->bool) {
    lyra_runtime::eval::register_concurrency_filtered(ev, pred);
}
