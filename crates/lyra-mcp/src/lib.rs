use anyhow::Result;
use lyra_core::value::Value as LVal;
use lyra_runtime::{set_default_registrar, Evaluator};
use serde::{Deserialize, Serialize};
use serde_json as sj;
use std::io::{Read, Write};

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
enum Id {
    Num(i64),
    Str(String),
}

#[derive(Debug, Serialize, Deserialize)]
struct Request {
    #[serde(rename = "jsonrpc")]
    jsonrpc: String,
    id: Option<Id>,
    method: String,
    #[serde(default)]
    params: sj::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct Response {
    #[serde(rename = "jsonrpc")]
    jsonrpc: String,
    id: Id,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<sj::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<RpcError>,
}

#[derive(Debug, Serialize, Deserialize)]
struct RpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<sj::Value>,
}

fn write_message<W: Write>(out: &mut W, val: &sj::Value) -> Result<()> {
    let s = sj::to_string(val)?;
    write!(out, "Content-Length: {}\r\n\r\n{}", s.len(), s)?;
    out.flush()?;
    Ok(())
}

fn read_message<R: Read>(inp: &mut R) -> Result<Option<sj::Value>> {
    // Read headers until blank line
    let mut headers = String::new();
    let mut buf = [0u8; 1];
    let mut last4 = [0u8; 4];
    loop {
        let n = inp.read(&mut buf)?;
        if n == 0 {
            // EOF
            if headers.is_empty() {
                return Ok(None);
            } else {
                break;
            }
        }
        headers.push(buf[0] as char);
        last4.rotate_left(1);
        last4[3] = buf[0];
        if &last4 == b"\r\n\r\n" {
            break;
        }
    }
    let mut content_length: Option<usize> = None;
    for line in headers.split("\r\n") {
        let l = line.trim();
        if l.is_empty() {
            continue;
        }
        if let Some(rest) = l.strip_prefix("Content-Length:") {
            content_length = rest.trim().parse::<usize>().ok();
        }
    }
    let len = match content_length {
        Some(n) => n,
        None => return Ok(None),
    };
    let mut body = vec![0u8; len];
    let mut readn = 0usize;
    while readn < len {
        let n = inp.read(&mut body[readn..])?;
        if n == 0 {
            break;
        }
        readn += n;
    }
    if readn < len {
        return Ok(None);
    }
    let v: sj::Value = sj::from_slice(&body)?;
    Ok(Some(v))
}

#[derive(Debug, Serialize, Deserialize)]
struct InitializeParams {
    #[serde(rename = "protocolVersion")]
    protocol_version: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct InitializeResult {
    #[serde(rename = "protocolVersion")]
    protocol_version: String,
    #[serde(rename = "serverInfo")]
    server_info: sj::Value,
    capabilities: sj::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct McpTool {
    name: String,
    description: String,
    #[serde(rename = "inputSchema")]
    #[serde(skip_serializing_if = "Option::is_none")]
    input_schema: Option<sj::Value>,
    #[serde(rename = "x-lyra-id")]
    #[serde(skip_serializing_if = "Option::is_none")]
    lyra_id: Option<String>,
    #[serde(rename = "x-lyra-tags")]
    #[serde(skip_serializing_if = "Option::is_none")]
    tags: Option<Vec<String>>,
    #[serde(rename = "x-lyra-effects")]
    #[serde(skip_serializing_if = "Option::is_none")]
    effects: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ToolsListResult {
    tools: Vec<McpTool>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ToolsCallParams {
    name: String,
    #[serde(default)]
    arguments: sj::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct ToolsCallResult {
    content: Vec<sj::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ToolsSearchParams {
    query: String,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    tags: Option<Vec<String>>,
    #[serde(default)]
    effects: Option<Vec<String>>,
    #[serde(default)]
    capabilities: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ToolsSearchResult {
    tools: Vec<McpTool>,
}

fn sanitize_name(s: &str) -> String {
    let mut out: String = s
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect();
    if out.is_empty() {
        out.push_str("tool");
    }
    if out.len() > 64 {
        out.truncate(64);
    }
    out
}

fn lyra_to_mcp_tool(v: &LVal) -> Option<McpTool> {
    if let LVal::Assoc(m) = v {
        let name = m.get("name").or_else(|| m.get("id")).and_then(|vv| match vv {
            LVal::String(s) | LVal::Symbol(s) => Some(s.clone()),
            _ => None,
        })?;
        let desc = m
            .get("summary")
            .and_then(|vv| match vv {
                LVal::String(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_default();
        let input_schema = m.get("input_schema").and_then(|vv| serde_json::to_value(vv).ok());
        let tags = m.get("tags").and_then(|vv| match vv {
            LVal::List(vs) => Some(
                vs.iter()
                    .filter_map(|x| match x {
                        LVal::String(s) | LVal::Symbol(s) => Some(s.clone()),
                        _ => None,
                    })
                    .collect(),
            ),
            _ => None,
        });
        let effects = m.get("effects").and_then(|vv| match vv {
            LVal::List(vs) => Some(
                vs.iter()
                    .filter_map(|x| match x {
                        LVal::String(s) | LVal::Symbol(s) => Some(s.clone()),
                        _ => None,
                    })
                    .collect(),
            ),
            _ => None,
        });
        let lyra_id = m.get("id").and_then(|vv| match vv {
            LVal::String(s) | LVal::Symbol(s) => Some(s.clone()),
            _ => None,
        });
        Some(McpTool {
            name: sanitize_name(&name),
            description: desc,
            input_schema,
            lyra_id,
            tags,
            effects,
        })
    } else {
        None
    }
}

fn list_all_tools(ev: &mut Evaluator) -> Vec<McpTool> {
    let expr =
        LVal::Expr { head: Box::new(LVal::Symbol("ToolsExportBundle".into())), args: vec![] };
    let v = ev.eval(expr);
    match v {
        LVal::List(items) => items.iter().filter_map(lyra_to_mcp_tool).collect(),
        _ => vec![],
    }
}

fn search_tools(ev: &mut Evaluator, params: &ToolsSearchParams) -> Vec<McpTool> {
    let mut args = vec![LVal::String(params.query.clone())];
    if let Some(k) = params.limit {
        args.push(LVal::Integer(k as i64));
    }
    let expr = LVal::Expr { head: Box::new(LVal::Symbol("ToolsSearch".into())), args };
    let v = ev.eval(expr);
    let mut tools: Vec<McpTool> = match v {
        LVal::List(items) => items.into_iter().filter_map(|it| lyra_to_mcp_tool(&it)).collect(),
        _ => vec![],
    };
    if let Some(tags) = &params.tags {
        let set: std::collections::HashSet<String> =
            tags.iter().map(|s| s.to_lowercase()).collect();
        tools.retain(|t| {
            t.tags
                .as_ref()
                .map(|ts| ts.iter().any(|x| set.contains(&x.to_lowercase())))
                .unwrap_or(false)
        });
    }
    if let Some(effs) = &params.effects {
        let set: std::collections::HashSet<String> =
            effs.iter().map(|s| s.to_lowercase()).collect();
        tools.retain(|t| {
            t.effects
                .as_ref()
                .map(|es| es.iter().any(|x| set.contains(&x.to_lowercase())))
                .unwrap_or(false)
        });
    }
    tools
}

fn value_from_json(j: &sj::Value) -> LVal {
    match sj::from_value::<LVal>(j.clone()) {
        Ok(v) => v,
        Err(_) => LVal::Symbol("Null".into()),
    }
}
fn json_from_value(v: &LVal) -> sj::Value {
    sj::to_value(v).unwrap_or(sj::Value::Null)
}

pub fn run_stdio_server() -> Result<()> {
    set_default_registrar(lyra_stdlib::register_all);
    let mut ev = Evaluator::new();
    ev.set_env("ModelsMode", LVal::String("mock".into()));

    let stdin = std::io::stdin();
    let mut reader = stdin.lock();
    let stdout = std::io::stdout();
    let mut writer = stdout.lock();

    while let Some(msg) = read_message(&mut reader)? {
        let req: Request = match sj::from_value(msg) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("parse error: {e}");
                continue;
            }
        };
        let id_opt = req.id.clone();
        let method = req.method.as_str().to_string();
        match method.as_str() {
            "initialize" => {
                let _params: InitializeParams = sj::from_value(req.params)
                    .unwrap_or(InitializeParams { protocol_version: None });
                let res = InitializeResult {
                    protocol_version: "2024-05-24".into(),
                    server_info: sj::json!({"name":"lyra","version": env!("CARGO_PKG_VERSION")}),
                    capabilities: sj::json!({ "tools": { "list": true, "call": true }, "x-lyra": { "tools.search": true } }),
                };
                if let Some(id) = id_opt {
                    write_message(
                        &mut writer,
                        &sj::json!({"jsonrpc":"2.0","id":id,"result": res}),
                    )?;
                }
            }
            "tools/list" => {
                let tools = list_all_tools(&mut ev);
                if let Some(id) = id_opt {
                    write_message(
                        &mut writer,
                        &sj::json!({"jsonrpc":"2.0","id":id,"result": ToolsListResult{ tools }}),
                    )?;
                }
            }
            "tools/call" => {
                let params: ToolsCallParams = match sj::from_value(req.params) {
                    Ok(p) => p,
                    Err(e) => {
                        if let Some(id) = id_opt {
                            let err = RpcError {
                                code: -32602,
                                message: format!("invalid params: {e}"),
                                data: None,
                            };
                            write_message(
                                &mut writer,
                                &sj::json!({"jsonrpc":"2.0","id":id,"error": err}),
                            )?;
                        }
                        continue;
                    }
                };
                let args_assoc = match &params.arguments {
                    sj::Value::Null => LVal::Assoc(std::collections::HashMap::new()),
                    sj::Value::Object(_) => value_from_json(&params.arguments),
                    _ => LVal::Assoc(std::collections::HashMap::new()),
                };
                let expr = LVal::Expr {
                    head: Box::new(LVal::Symbol("ToolsInvoke".into())),
                    args: vec![LVal::String(params.name.clone()), args_assoc],
                };
                let out = ev.eval(expr);
                let content = vec![
                    sj::json!({"type":"json","json": json_from_value(&out)}),
                    sj::json!({"type":"text","text": lyra_core::pretty::format_value(&out) }),
                ];
                if let Some(id) = id_opt {
                    write_message(
                        &mut writer,
                        &sj::json!({"jsonrpc":"2.0","id":id,"result": ToolsCallResult{ content }}),
                    )?;
                }
            }
            "lyra/tools.search" => {
                let params: ToolsSearchParams = match sj::from_value(req.params) {
                    Ok(p) => p,
                    Err(e) => {
                        if let Some(id) = id_opt {
                            let err = RpcError {
                                code: -32602,
                                message: format!("invalid params: {e}"),
                                data: None,
                            };
                            write_message(
                                &mut writer,
                                &sj::json!({"jsonrpc":"2.0","id":id,"error": err}),
                            )?;
                        }
                        continue;
                    }
                };
                let tools = search_tools(&mut ev, &params);
                if let Some(id) = id_opt {
                    write_message(
                        &mut writer,
                        &sj::json!({"jsonrpc":"2.0","id":id,"result": ToolsSearchResult{ tools }}),
                    )?;
                }
            }
            "shutdown" => {
                if let Some(id) = id_opt {
                    write_message(
                        &mut writer,
                        &sj::json!({"jsonrpc":"2.0","id":id,"result": sj::Value::Null}),
                    )?;
                }
            }
            "exit" => {
                break;
            }
            _ => {
                if let Some(id) = id_opt {
                    let err = RpcError {
                        code: -32601,
                        message: format!("method not found: {}", method),
                        data: None,
                    };
                    write_message(&mut writer, &sj::json!({"jsonrpc":"2.0","id":id,"error": err}))?;
                }
            }
        }
    }
    Ok(())
}
