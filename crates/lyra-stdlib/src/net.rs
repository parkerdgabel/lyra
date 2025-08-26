use lyra_core::value::Value;
use lyra_runtime::{Evaluator};
use lyra_runtime::attrs::Attributes;
use serde_json as sj;
use crate::register_if;
#[cfg(feature = "tools")] use crate::tools::add_specs;
#[cfg(feature = "tools")] use crate::tool_spec;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

pub fn register_net(ev: &mut Evaluator) {
    ev.register("HttpGet", http_get as NativeFn, Attributes::empty());
    ev.register("HttpPost", http_post as NativeFn, Attributes::empty());
    ev.register("HttpPut", http_put as NativeFn, Attributes::empty());
    ev.register("HttpPatch", http_patch as NativeFn, Attributes::empty());
    ev.register("HttpDelete", http_delete as NativeFn, Attributes::empty());
    ev.register("HttpHead", http_head as NativeFn, Attributes::empty());
    ev.register("HttpOptions", http_options as NativeFn, Attributes::empty());
    ev.register("Download", download as NativeFn, Attributes::empty());
    ev.register("DownloadStream", download_stream as NativeFn, Attributes::empty());
    ev.register("HttpRequest", http_request_generic as NativeFn, Attributes::empty());
    ev.register("HttpStreamRequest", http_stream_request as NativeFn, Attributes::empty());
    ev.register("HttpStreamRead", http_stream_read as NativeFn, Attributes::empty());
    ev.register("HttpStreamClose", http_stream_close as NativeFn, Attributes::empty());
    ev.register("HttpDownloadCached", http_download_cached as NativeFn, Attributes::empty());
    ev.register("HttpRetry", http_retry as NativeFn, Attributes::empty());
    ev.register("HttpServe", http_serve as NativeFn, Attributes::HOLD_ALL);
    ev.register("HttpServerStop", http_server_stop as NativeFn, Attributes::empty());
    ev.register("HttpServerAddr", http_server_addr as NativeFn, Attributes::empty());
    ev.register("HttpServeRoutes", http_serve_routes as NativeFn, Attributes::HOLD_ALL);
    ev.register("PathMatch", path_match_builtin as NativeFn, Attributes::empty());
    ev.register("RespondFile", respond_file as NativeFn, Attributes::empty());
    ev.register("RespondText", respond_text as NativeFn, Attributes::empty());
    ev.register("RespondJson", respond_json as NativeFn, Attributes::empty());
    ev.register("RespondBytes", respond_bytes as NativeFn, Attributes::empty());
    ev.register("RespondHtml", respond_html as NativeFn, Attributes::empty());
    ev.register("RespondRedirect", respond_redirect as NativeFn, Attributes::empty());
    ev.register("RespondNoContent", respond_no_content as NativeFn, Attributes::empty());
    ev.register("CookiesHeader", cookies_header as NativeFn, Attributes::empty());
    ev.register("GetResponseCookies", get_response_cookies as NativeFn, Attributes::empty());
    #[cfg(feature = "net_https")]
    ev.register("HttpServeTls", http_serve_tls as NativeFn, Attributes::HOLD_ALL);

    #[cfg(feature = "tools")]
    add_specs(vec![
        tool_spec!("HttpGet", summary: "HTTP GET request (http/https)", params: ["url","opts"], tags: ["net","http"], effects: ["net.http"]),
        tool_spec!("HttpPost", summary: "HTTP POST request (http/https)", params: ["url","body","opts"], tags: ["net","http"], effects: ["net.http"]),
        tool_spec!("HttpPut", summary: "HTTP PUT request (http/https)", params: ["url","body","opts"], tags: ["net","http"], effects: ["net.http"]),
        tool_spec!("HttpPatch", summary: "HTTP PATCH request (http/https)", params: ["url","body","opts"], tags: ["net","http"], effects: ["net.http"]),
        tool_spec!("HttpDelete", summary: "HTTP DELETE request (http/https)", params: ["url","opts"], tags: ["net","http"], effects: ["net.http"]),
        tool_spec!("HttpHead", summary: "HTTP HEAD request (http/https)", params: ["url","opts"], tags: ["net","http"], effects: ["net.http"]),
        tool_spec!("HttpOptions", summary: "HTTP OPTIONS request (http/https)", params: ["url","opts"], tags: ["net","http"], effects: ["net.http"]),
        tool_spec!("Download", summary: "Download URL to file (http/https)", params: ["url","path","opts"], tags: ["net","http","fs"], effects: ["net.http","fs.write"]),
        tool_spec!("DownloadStream", summary: "Stream download URL directly to file", params: ["url","path","opts"], tags: ["net","http","fs"], effects: ["net.http","fs.write"]),
        tool_spec!("HttpServe", summary: "Start an HTTP server and handle requests with a function", params: ["handler","opts"], tags: ["net","http","server"], effects: ["net.listen"]),
        tool_spec!("HttpServeRoutes", summary: "Start an HTTP server with a routes table", params: ["routes","opts"], tags: ["net","http","server"], effects: ["net.listen"]),
        tool_spec!("HttpServerStop", summary: "Stop a running HTTP server by id", params: ["server"], tags: ["net","http","server"], effects: ["net.listen"]),
        tool_spec!("HttpServerAddr", summary: "Get bound address for a server id", params: ["server"], tags: ["net","http","server"], effects: []),
        tool_spec!("HttpRequest", summary: "Generic HTTP request via options object", params: ["options"], tags: ["net","http"], effects: ["net.http"]),
        tool_spec!("PathMatch", summary: "Match a path pattern like /users/:id against a path", params: ["pattern","path"], tags: ["http","routing"], effects: []),
        tool_spec!("RespondFile", summary: "Build a file response for HttpServe", params: ["path","opts"], tags: ["http","server"], effects: []),
        tool_spec!("RespondText", summary: "Build a text response for HttpServe", params: ["text","opts"], tags: ["http","server"], effects: []),
        tool_spec!("RespondJson", summary: "Build a JSON response for HttpServe", params: ["value","opts"], tags: ["http","server","json"], effects: []),
        tool_spec!("RespondBytes", summary: "Build a binary response for HttpServe", params: ["bytes","opts"], tags: ["http","server","binary"], effects: []),
        tool_spec!("RespondHtml", summary: "Build an HTML response for HttpServe", params: ["html","opts"], tags: ["http","server","html"], effects: []),
        tool_spec!("RespondRedirect", summary: "Build a redirect response (Location header)", params: ["location","opts"], tags: ["http","server","redirect"], effects: []),
        tool_spec!("RespondNoContent", summary: "Build an empty 204/205/304 response", params: ["opts"], tags: ["http","server"], effects: []),
        tool_spec!("CookiesHeader", summary: "Build a Cookie header string from an assoc", params: ["cookies"], tags: ["http","cookies"], effects: []),
        tool_spec!("GetResponseCookies", summary: "Parse Set-Cookie headers from a response map", params: ["response"], tags: ["http","cookies"], effects: []),
        #[cfg(feature = "net_https")]
        tool_spec!("HttpServeTls", summary: "Start an HTTPS server with TLS cert/key", params: ["handler","opts"], tags: ["net","http","server","tls"], effects: ["net.listen"]),
    ]);
}

pub fn register_net_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str)->bool) {
    register_if(ev, pred, "HttpGet", http_get as NativeFn, Attributes::empty());
    register_if(ev, pred, "HttpPost", http_post as NativeFn, Attributes::empty());
    register_if(ev, pred, "HttpPut", http_put as NativeFn, Attributes::empty());
    register_if(ev, pred, "HttpPatch", http_patch as NativeFn, Attributes::empty());
    register_if(ev, pred, "HttpDelete", http_delete as NativeFn, Attributes::empty());
    register_if(ev, pred, "HttpHead", http_head as NativeFn, Attributes::empty());
    register_if(ev, pred, "HttpOptions", http_options as NativeFn, Attributes::empty());
    register_if(ev, pred, "Download", download as NativeFn, Attributes::empty());
    register_if(ev, pred, "DownloadStream", download_stream as NativeFn, Attributes::empty());
    register_if(ev, pred, "HttpRequest", http_request_generic as NativeFn, Attributes::empty());
    register_if(ev, pred, "HttpStreamRequest", http_stream_request as NativeFn, Attributes::empty());
    register_if(ev, pred, "HttpStreamRead", http_stream_read as NativeFn, Attributes::empty());
    register_if(ev, pred, "HttpStreamClose", http_stream_close as NativeFn, Attributes::empty());
    register_if(ev, pred, "HttpDownloadCached", http_download_cached as NativeFn, Attributes::empty());
    register_if(ev, pred, "HttpRetry", http_retry as NativeFn, Attributes::empty());
    register_if(ev, pred, "HttpServe", http_serve as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "HttpServerStop", http_server_stop as NativeFn, Attributes::empty());
    register_if(ev, pred, "HttpServerAddr", http_server_addr as NativeFn, Attributes::empty());
    register_if(ev, pred, "HttpServeRoutes", http_serve_routes as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "PathMatch", path_match_builtin as NativeFn, Attributes::empty());
    register_if(ev, pred, "RespondFile", respond_file as NativeFn, Attributes::empty());
    register_if(ev, pred, "RespondText", respond_text as NativeFn, Attributes::empty());
    register_if(ev, pred, "RespondJson", respond_json as NativeFn, Attributes::empty());
    register_if(ev, pred, "RespondBytes", respond_bytes as NativeFn, Attributes::empty());
    register_if(ev, pred, "RespondHtml", respond_html as NativeFn, Attributes::empty());
    register_if(ev, pred, "RespondRedirect", respond_redirect as NativeFn, Attributes::empty());
    register_if(ev, pred, "RespondNoContent", respond_no_content as NativeFn, Attributes::empty());
    register_if(ev, pred, "CookiesHeader", cookies_header as NativeFn, Attributes::empty());
    register_if(ev, pred, "GetResponseCookies", get_response_cookies as NativeFn, Attributes::empty());
    #[cfg(feature = "net_https")]
    register_if(ev, pred, "HttpServeTls", http_serve_tls as NativeFn, Attributes::HOLD_ALL);
}

fn failure(tag: &str, msg: &str) -> Value {
    Value::Assoc(vec![
        ("message".to_string(), Value::String(msg.to_string())),
        ("tag".to_string(), Value::String(tag.to_string())),
    ].into_iter().collect())
}

#[derive(Clone, PartialEq)]
enum AsKind { Auto, Text, Bytes, Json }

#[derive(Clone)]
struct HttpOpts {
    timeout_ms: Option<u64>,
    headers: Vec<(String, String)>,
    query: Vec<(String, String)>,
    json_body: Option<sj::Value>,
    form_body: Option<Vec<(String, String)>>,
    multipart: Option<Vec<MultipartPart>>,
    as_kind: AsKind,
    tls_insecure: bool,
    follow_redirects: Option<usize>, // None=default, Some(0)=disable, Some(n)=limit
}

#[derive(Clone)]
struct MultipartPart { name: String, value: MultipartValue }

#[derive(Clone)]
enum MultipartValue { Text(String), File { path: String, filename: Option<String>, content_type: Option<String> } }

fn http_opts_from(ev: &mut Evaluator, v: Option<Value>) -> HttpOpts {
    let mut o = HttpOpts { timeout_ms: Some(10_000), headers: vec![], query: vec![], json_body: None, form_body: None, multipart: None, as_kind: AsKind::Auto, tls_insecure: false, follow_redirects: None };
    if let Some(Value::Assoc(m)) = v.map(|x| ev.eval(x)) {
        if let Some(Value::Integer(ms)) = m.get("TimeoutMs") { if *ms > 0 { o.timeout_ms = Some(*ms as u64); } else { o.timeout_ms=None; } }
        if let Some(Value::Assoc(hs)) = m.get("Headers") {
            for (k, vv) in hs.iter() { if let Some(s) = match vv { Value::String(s)=>Some(s.clone()), Value::Symbol(s)=>Some(s.clone()), _=>None } { o.headers.push((k.clone(), s)); } }
        }
        if let Some(vq) = m.get("Query") { o.query = collect_kv_pairs(ev, vq.clone()); }
        if let Some(vf) = m.get("Form") { let pairs = collect_kv_pairs(ev, vf.clone()); if !pairs.is_empty() { o.form_body = Some(pairs); } }
        if let Some(vj) = m.get("Json") { let vv = ev.eval(vj.clone()); o.json_body = Some(value_to_json(&vv)); }
        if let Some(vm) = m.get("Multipart") { o.multipart = collect_multipart(ev, vm.clone()); }
        if let Some(Value::String(s))|Some(Value::Symbol(s)) = m.get("As") {
            let sl = s.to_ascii_lowercase();
            o.as_kind = match sl.as_str() { "bytes"=>AsKind::Bytes, "json"=>AsKind::Json, "text"=>AsKind::Text, _=>AsKind::Auto };
        }

        if let Some(Value::Assoc(cookies)) = m.get("Cookies") {
            let cookie = cookies_header_from_assoc(cookies);
            if !cookie.is_empty() { o.headers.push(("Cookie".into(), cookie)); }
        }
        if let Some(Value::Boolean(b)) = m.get("DisableTlsVerify") { if *b { o.tls_insecure = true; } }
        if let Some(Value::Boolean(b)) = m.get("FollowRedirects") { o.follow_redirects = Some(if *b { 10 } else { 0 }); }
        if let Some(Value::Integer(n)) = m.get("MaxRedirects") { o.follow_redirects = Some((*n).max(0) as usize); }
    }
    o
}

fn collect_multipart(ev: &mut Evaluator, v: Value) -> Option<Vec<MultipartPart>> {
    let v = ev.eval(v);
    let mut out: Vec<MultipartPart> = Vec::new();
    match v {
        Value::Assoc(m) => {
            for (k, vv) in m.into_iter() { if let Some(pv) = to_mpart_value(ev, vv) { out.push(MultipartPart { name: k, value: pv }); } }
        }
        Value::List(items) => {
            for it in items { match ev.eval(it) { Value::List(mut pair) if pair.len()==2 => { let val = pair.pop().unwrap(); let key = pair.pop().unwrap(); let kn = value_to_string(&key); if let Some(pv) = to_mpart_value(ev, val) { out.push(MultipartPart { name: kn, value: pv }); } }, _=>{} } }
        }
        _ => {}
    }
    if out.is_empty() { None } else { Some(out) }
}

fn to_mpart_value(ev: &mut Evaluator, v: Value) -> Option<MultipartValue> {
    let vv = ev.eval(v);
    match vv {
        Value::String(s) | Value::Symbol(s) => Some(MultipartValue::Text(s)),
        Value::Expr { head, args } if matches!(&*head, Value::Symbol(s) if s=="File") => {
            let path = args.get(0).and_then(|a| match ev.eval(a.clone()) { Value::String(s)|Value::Symbol(s)=>Some(s), _=>None })?;
            let mut filename: Option<String> = None;
            let mut ctype: Option<String> = None;
            if let Some(Value::Assoc(opts)) = args.get(1).and_then(|a| match ev.eval(a.clone()) { Value::Assoc(m)=>Some(Value::Assoc(m)), _=>None }) {
                if let Some(Value::String(s))|Some(Value::Symbol(s)) = opts.get("Filename") { filename = Some(s.clone()); }
                if let Some(Value::String(s))|Some(Value::Symbol(s)) = opts.get("ContentType") { ctype = Some(s.clone()); }
            }
            Some(MultipartValue::File { path, filename, content_type: ctype })
        }
        _ => None
    }
}

fn http_get(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("HttpGet".into())), args } }
    let url = to_string_arg(ev, args[0].clone());
    let opts = http_opts_from(ev, args.get(1).cloned());
    match http_request("GET", &url, None, &opts) { Ok(resp)=>resp, Err(e)=>failure("HTTP::error", &e) }
}

fn http_post(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("HttpPost".into())), args } }
    let url = to_string_arg(ev, args[0].clone());
    let body = value_to_bytes(ev.eval(args[1].clone()));
    let opts = http_opts_from(ev, args.get(2).cloned());
    match http_request("POST", &url, Some(&body), &opts) { Ok(resp)=>resp, Err(e)=>failure("HTTP::error", &e) }
}

fn http_put(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("HttpPut".into())), args } }
    let url = to_string_arg(ev, args[0].clone());
    let body = value_to_bytes(ev.eval(args[1].clone()));
    let opts = http_opts_from(ev, args.get(2).cloned());
    match http_request("PUT", &url, Some(&body), &opts) { Ok(resp)=>resp, Err(e)=>failure("HTTP::error", &e) }
}

fn http_patch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("HttpPatch".into())), args } }
    let url = to_string_arg(ev, args[0].clone());
    let body = value_to_bytes(ev.eval(args[1].clone()));
    let opts = http_opts_from(ev, args.get(2).cloned());
    match http_request("PATCH", &url, Some(&body), &opts) { Ok(resp)=>resp, Err(e)=>failure("HTTP::error", &e) }
}

fn http_delete(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("HttpDelete".into())), args } }
    let url = to_string_arg(ev, args[0].clone());
    let opts = http_opts_from(ev, args.get(1).cloned());
    match http_request("DELETE", &url, None, &opts) { Ok(resp)=>resp, Err(e)=>failure("HTTP::error", &e) }
}

fn http_head(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("HttpHead".into())), args } }
    let url = to_string_arg(ev, args[0].clone());
    let opts = http_opts_from(ev, args.get(1).cloned());
    match http_request("HEAD", &url, None, &opts) { Ok(resp)=>resp, Err(e)=>failure("HTTP::error", &e) }
}


fn http_request_generic(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("HttpRequest".into())), args } }
    let opts_v = ev.eval(args[0].clone());
    if let Value::Assoc(m) = opts_v {
        let method = m.get("Method").and_then(|v| if let Value::String(s)|Value::Symbol(s)=v { Some(s.clone()) } else { None }).unwrap_or_else(|| "GET".into());
        let url = m.get("Url").and_then(|v| if let Value::String(s)|Value::Symbol(s)=v { Some(s.clone()) } else { None }).unwrap_or_else(|| String::new());
        if url.is_empty() { return failure("HTTP::request", "Url required"); }
        let mut body_bytes: Option<Vec<u8>> = None;
        if let Some(b) = m.get("Body") { body_bytes = Some(value_to_bytes(ev.eval(b.clone()))); }
        let opts = http_opts_from(ev, Some(Value::Assoc(m)));
        match http_request(&method, &url, body_bytes.as_deref(), &opts) { Ok(resp)=>resp, Err(e)=>failure("HTTP::error", &e) }
    } else { failure("HTTP::request", "Options must be an association") }
}
fn http_options(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("HttpOptions".into())), args } }
    let url = to_string_arg(ev, args[0].clone());
    let opts = http_opts_from(ev, args.get(1).cloned());
    match http_request("OPTIONS", &url, None, &opts) { Ok(resp)=>resp, Err(e)=>failure("HTTP::error", &e) }
}

fn download(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("Download".into())), args } }
    let url = to_string_arg(ev, args[0].clone());
    let path = to_string_arg(ev, args[1].clone());
    let opts = http_opts_from(ev, args.get(2).cloned());
    match http_request_raw("GET", &url, None, &opts) {
        Ok((status, _status_text, _headers, body)) => {
            if status >= 200 && status < 300 {
                match std::fs::write(&path, &body) { Ok(_)=>Value::Boolean(true), Err(e)=>failure("HTTP::fs", &format!("Download write: {}", e)) }
            } else { failure("HTTP::status", &format!("Download failed: status {}", status)) }
        }
        Err(e) => failure("HTTP::error", &e)
    }
}

fn download_stream(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("DownloadStream".into())), args } }
    let url = to_string_arg(ev, args[0].clone());
    let path = to_string_arg(ev, args[1].clone());
    let opts = http_opts_from(ev, args.get(2).cloned());
    match download_stream_impl(&url, &path, &opts) { Ok(()) => Value::Boolean(true), Err(e)=>failure("HTTP::download", &e) }
}

fn value_to_bytes(v: Value) -> Vec<u8> {
    match v {
        Value::String(s) => s.into_bytes(),
        Value::Symbol(s) => s.into_bytes(),
        _ => lyra_core::pretty::format_value(&v).into_bytes(),
    }
}

fn to_string_arg(ev: &mut Evaluator, v: Value) -> String {
    match ev.eval(v) { Value::String(s)|Value::Symbol(s)=>s, other=>lyra_core::pretty::format_value(&other) }
}

fn http_request(method: &str, url: &str, body: Option<&[u8]>, opts: &HttpOpts) -> Result<Value, String> {
    #[cfg(feature = "net_https")]
    {
        if url.to_ascii_lowercase().starts_with("https://") || url.to_ascii_lowercase().starts_with("http://") {
            return http_request_reqwest(method, url, body, opts);
        }
    }
    let (status, status_text, headers, body_bytes) = http_request_raw(method, url, body, opts)?;
    Ok(build_response(status, status_text, headers, body_bytes, opts))
}

fn build_response(status: u16, status_text: String, headers: Vec<(String, Value)>, body: Vec<u8>, opts: &HttpOpts) -> Value {
    let headers_val = Value::Assoc(headers.clone().into_iter().collect());
    let headers_list_val = Value::List(headers.iter().map(|(k,v)| Value::Assoc(vec![ ("name".into(), Value::String(k.clone())), ("value".into(), v.clone()) ].into_iter().collect())).collect());
    let text = String::from_utf8_lossy(&body).to_string();
    let mut map: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
    map.insert("status".into(), Value::Integer(status as i64));
    map.insert("statusText".into(), Value::String(status_text));
    map.insert("headers".into(), headers_val);
    map.insert("headersList".into(), headers_list_val);
    let ct = headers.iter().find(|(k,_)| k.eq_ignore_ascii_case("Content-Type")).and_then(|(_,v)| if let Value::String(s)|Value::Symbol(s)=v { Some(s.to_ascii_lowercase()) } else { None }).unwrap_or_default();
    let want_json = match opts.as_kind { AsKind::Json => true, AsKind::Text => false, AsKind::Bytes => false, AsKind::Auto => ct.contains("application/json") || ct.contains("+json") };
    if want_json {
        let parsed = sj::from_slice::<sj::Value>(&body).map(|j| json_to_value(&j)).unwrap_or(Value::Symbol("Null".into()));
        map.insert("json".into(), parsed);
        map.insert("body".into(), Value::String(text));
    } else {
        match opts.as_kind { AsKind::Bytes => { map.insert("bytes".into(), Value::List(body.into_iter().map(|b| Value::Integer(b as i64)).collect())); }, _ => { map.insert("body".into(), Value::String(text)); } }
    }
    Value::Assoc(map)
}

fn http_request_raw(method: &str, url: &str, body: Option<&[u8]>, opts: &HttpOpts) -> Result<(u16, String, Vec<(String, Value)>, Vec<u8>), String> {
    // Append query from opts
    let mut url_final = url.to_string();
    if !opts.query.is_empty() { url_final = append_query(&url_final, &opts.query); }
    let parsed = parse_http_url(&url_final).ok_or_else(|| format!("Unsupported or invalid URL: {}", url_final))?;
    if parsed.scheme != "http" { return Err("Only http scheme supported; https not available".into()); }
    use std::net::{TcpStream, ToSocketAddrs};
    use std::io::{Read, Write};
    let host_port = format!("{}:{}", parsed.host, parsed.port);
    let addr = host_port.to_socket_addrs().map_err(|e| format!("DNS: {}", e))?.next().ok_or_else(|| "No address".to_string())?;
    let stream = if let Some(ms) = opts.timeout_ms { TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(ms)) } else { TcpStream::connect(addr) };
    let mut stream = stream.map_err(|e| format!("Connect: {}", e))?;
    if let Some(ms) = opts.timeout_ms { let d = Some(std::time::Duration::from_millis(ms)); let _ = stream.set_read_timeout(d); let _ = stream.set_write_timeout(d); }
    let path = if parsed.path.is_empty() { "/".to_string() } else { parsed.path.clone() };
    let req_line = format!("{} {} HTTP/1.1\r\n", method, path);
    let mut req = String::new();
    req.push_str(&req_line);
    req.push_str(&format!("Host: {}\r\n", parsed.host));
    req.push_str("User-Agent: Lyra/0.1\r\n");
    req.push_str("Connection: close\r\n");
    for (k, v) in &opts.headers { req.push_str(&format!("{}: {}\r\n", k, v)); }
    let mut body_bytes: Vec<u8> = Vec::new();
    // body resolution precedence: Multipart > Json > Form > explicit body
    if let Some(parts) = &opts.multipart { let (ctype, bytes) = build_multipart_body(parts); req.push_str(&format!("Content-Type: {}\r\n", ctype)); body_bytes = bytes; }
    else if let Some(j) = &opts.json_body { let s = sj::to_string(j).unwrap_or_else(|_| String::new()); body_bytes = s.into_bytes(); req.push_str("Content-Type: application/json\r\n"); }
    else if let Some(form) = &opts.form_body { let s = form_urlencode(form); body_bytes = s.into_bytes(); req.push_str("Content-Type: application/x-www-form-urlencoded\r\n"); }
    else if let Some(b) = body { body_bytes = b.to_vec(); }
    if !body_bytes.is_empty() { req.push_str(&format!("Content-Length: {}\r\n", body_bytes.len())); }
    req.push_str("\r\n");
    stream.write_all(req.as_bytes()).map_err(|e| format!("Write: {}", e))?;
    if !body_bytes.is_empty() { stream.write_all(&body_bytes).map_err(|e| format!("Write body: {}", e))?; }
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).map_err(|e| format!("Read: {}", e))?;
    parse_http_response(&buf)
}

fn build_multipart_body(parts: &Vec<MultipartPart>) -> (String, Vec<u8>) {
    let boundary = format!("----LyraBoundary{:x}", randish());
    let mut body: Vec<u8> = Vec::new();
    for p in parts {
        body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        match &p.value {
            MultipartValue::Text(s) => {
                let disp = format!("Content-Disposition: form-data; name=\"{}\"\r\n\r\n", p.name);
                body.extend_from_slice(disp.as_bytes());
                body.extend_from_slice(s.as_bytes());
                body.extend_from_slice(b"\r\n");
            }
            MultipartValue::File { path, filename, content_type } => {
                let fname = filename.clone().unwrap_or_else(|| std::path::Path::new(path).file_name().and_then(|s| s.to_str()).unwrap_or("file").to_string());
                let disp = format!("Content-Disposition: form-data; name=\"{}\"; filename=\"{}\"\r\n", p.name, fname);
                let ctype = format!("Content-Type: {}\r\n\r\n", content_type.clone().unwrap_or_else(|| "application/octet-stream".into()));
                body.extend_from_slice(disp.as_bytes());
                body.extend_from_slice(ctype.as_bytes());
                if let Ok(bytes) = std::fs::read(path) { body.extend_from_slice(&bytes); }
                body.extend_from_slice(b"\r\n");
            }
        }
    }
    body.extend_from_slice(format!("--{}--\r\n", boundary).as_bytes());
    (format!("multipart/form-data; boundary={}", boundary), body)
}

fn randish() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.subsec_nanos()).unwrap_or(0);
    let pid = std::process::id() as u64;
    ((nanos as u64) << 32) ^ pid
}

fn parse_http_response(buf: &[u8]) -> Result<(u16, String, Vec<(String, Value)>, Vec<u8>), String> {
    // Find header/body split
    let marker = b"\r\n\r\n";
    let pos = buf.windows(marker.len()).position(|w| w==marker).ok_or_else(|| "Invalid HTTP response".to_string())?;
    let (head_b, mut body) = (&buf[..pos], &buf[pos+4..]);
    let head = String::from_utf8_lossy(head_b);
    let mut lines = head.lines();
    let status_line = lines.next().ok_or_else(|| "Missing status".to_string())?;
    let mut parts = status_line.splitn(3, ' ');
    let _httpver = parts.next().unwrap_or("");
    let status_code = parts.next().unwrap_or("0").parse::<u16>().unwrap_or(0);
    let status_text = parts.next().unwrap_or("").to_string();
    let mut headers: Vec<(String, Value)> = Vec::new();
    let mut is_chunked = false;
    let mut content_length: Option<usize> = None;
    for line in lines {
        if let Some(idx) = line.find(':') {
            let (k, v) = (&line[..idx], line[idx+1..].trim());
            let kl = k.to_string();
            if k.eq_ignore_ascii_case("Transfer-Encoding") && v.to_ascii_lowercase().contains("chunked") { is_chunked = true; }
            if k.eq_ignore_ascii_case("Content-Length") { if let Ok(n) = v.parse::<usize>() { content_length = Some(n); } }
            headers.push((kl, Value::String(v.to_string())));
        }
    }
    let body_vec = if is_chunked { decode_chunked(body)? } else if let Some(n) = content_length { body.get(..n).unwrap_or(body).to_vec() } else { body.to_vec() };
    Ok((status_code, status_text, headers, body_vec))
}

fn has_header_case(headers: &[(String, String)], name: &str, value_contains: &str) -> bool {
    headers.iter().any(|(k,v)| k.eq_ignore_ascii_case(name) && v.to_ascii_lowercase().contains(&value_contains.to_ascii_lowercase()))
}

fn read_chunked<R: std::io::Read>(reader: &mut R, initial: &[u8], max: usize) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    let mut out: Vec<u8> = Vec::new();
    let mut buf: Vec<u8> = initial.to_vec();
    let mut tmp = [0u8; 1024];
    let mut pos = 0usize;
    loop {
        // Ensure we have a full line for size
        while !buf[pos..].windows(2).any(|w| w==b"\r\n") {
            let n = reader.read(&mut tmp)?; if n==0 { break; } buf.extend_from_slice(&tmp[..n]);
        }
        // Find CRLF
        if let Some(rel) = buf[pos..].windows(2).position(|w| w==b"\r\n") {
            let line = &buf[pos..pos+rel];
            pos += rel + 2;
            // Parse hex size (ignore chunk extensions after ';')
            let mut parts = line.split(|&b| b==b';');
            let sz_hex = parts.next().unwrap_or(&[]);
            let sz_str = String::from_utf8_lossy(sz_hex);
            let size = usize::from_str_radix(sz_str.trim(), 16).unwrap_or(0);
            if size == 0 {
                // Consume trailing CRLF after last chunk
                // Optionally read trailer headers until CRLF CRLF; we ignore trailers
                break;
            }
            // Ensure we have size bytes + CRLF
            while buf.len() < pos + size + 2 {
                let n = reader.read(&mut tmp)?; if n==0 { break; } buf.extend_from_slice(&tmp[..n]);
            }
            if buf.len() >= pos + size {
                out.extend_from_slice(&buf[pos..pos+size]);
                if out.len() > max { break; }
                pos += size + 2; // skip chunk + CRLF
            } else { break; }
        } else {
            let n = reader.read(&mut tmp)?; if n==0 { break; } buf.extend_from_slice(&tmp[..n]);
        }
    }
    Ok(out)
}

fn parse_request_has_chunked(headers: &[(String, String)]) -> bool { has_header_case(headers, "Transfer-Encoding", "chunked") }

fn decode_chunked(mut body: &[u8]) -> Result<Vec<u8>, String> {
    let mut out: Vec<u8> = Vec::new();
    loop {
        // read hex size line
        if let Some(pos) = body.windows(2).position(|w| w==b"\r\n") {
            let (size_line, rest) = body.split_at(pos);
            let size_str = String::from_utf8_lossy(size_line);
            let size = usize::from_str_radix(size_str.trim(), 16).map_err(|e| format!("chunk size: {}", e))?;
            body = &rest[2..];
            if size == 0 { break; }
            if body.len() < size + 2 { return Err("truncated chunk".into()); }
            out.extend_from_slice(&body[..size]);
            body = &body[size+2..]; // skip data and CRLF
        } else { return Err("invalid chunked body".into()); }
    }
    Ok(out)
}

#[derive(Debug, Clone)]
struct ParsedUrl { scheme: String, host: String, port: u16, path: String }

fn parse_http_url(url: &str) -> Option<ParsedUrl> {
    let lower = url.to_ascii_lowercase();
    let (scheme, rest) = if let Some(pos) = lower.find("://") { (&lower[..pos], &url[pos+3..]) } else { return None };
    let (authority, path) = if let Some(pos) = rest.find('/') { (&rest[..pos], &rest[pos..]) } else { (rest, "/") };
    let (host, port) = if let Some(pos) = authority.rfind(':') { (&authority[..pos], authority[pos+1..].parse::<u16>().ok().unwrap_or(80)) } else { (authority, 80) };
    Some(ParsedUrl { scheme: scheme.into(), host: host.into(), port, path: path.into() })
}

fn percent_encode_component(s: &str, plus_for_space: bool) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            b' ' if plus_for_space => out.push('+'),
            _ => { out.push('%'); out.push_str(&format!("{:02X}", b)); }
        }
    }
    out
}

fn append_query(base: &str, pairs: &Vec<(String, String)>) -> String {
    if pairs.is_empty() { return base.to_string(); }
    let sep = if base.contains('?') { '&' } else { '?' };
    let qs = pairs.iter().map(|(k,v)| format!("{}={}", percent_encode_component(k,false), percent_encode_component(v,true))).collect::<Vec<_>>().join("&");
    format!("{}{}{}", base, sep, qs)
}

fn download_stream_impl(url: &str, path: &str, opts: &HttpOpts) -> Result<(), String> {
    #[cfg(feature = "net_https")]
    {
        if url.to_ascii_lowercase().starts_with("https://") || url.to_ascii_lowercase().starts_with("http://") {
            return download_stream_reqwest(url, path, opts);
        }
    }
    download_stream_raw(url, path, opts)
}

fn download_stream_raw(url: &str, path: &str, opts: &HttpOpts) -> Result<(), String> {
    use std::net::{TcpStream, ToSocketAddrs};
    use std::io::{Read, Write};
    // Build URL (query)
    let mut url_final = url.to_string();
    if !opts.query.is_empty() { url_final = append_query(&url_final, &opts.query); }
    let parsed = parse_http_url(&url_final).ok_or_else(|| format!("Unsupported or invalid URL: {}", url_final))?;
    if parsed.scheme != "http" { return Err("Only http scheme supported; https not available".into()); }
    // Connect
    let host_port = format!("{}:{}", parsed.host, parsed.port);
    let addr = host_port.to_socket_addrs().map_err(|e| format!("DNS: {}", e))?.next().ok_or_else(|| "No address".to_string())?;
    let stream = if let Some(ms) = opts.timeout_ms { TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(ms)) } else { TcpStream::connect(addr) };
    let mut stream = stream.map_err(|e| format!("Connect: {}", e))?;
    if let Some(ms) = opts.timeout_ms { let d = Some(std::time::Duration::from_millis(ms)); let _ = stream.set_read_timeout(d); let _ = stream.set_write_timeout(d); }
    let path_only = if parsed.path.is_empty() { "/".to_string() } else { parsed.path.clone() };
    // Build request
    let mut req = String::new();
    req.push_str(&format!("GET {} HTTP/1.1\r\n", path_only));
    req.push_str(&format!("Host: {}\r\n", parsed.host));
    req.push_str("User-Agent: Lyra/0.1\r\n");
    req.push_str("Connection: close\r\n");
    for (k, v) in &opts.headers { req.push_str(&format!("{}: {}\r\n", k, v)); }
    req.push_str("\r\n");
    stream.write_all(req.as_bytes()).map_err(|e| format!("Write: {}", e))?;
    // Read response header
    let mut buf = Vec::new(); let mut tmp = [0u8; 2048];
    let header_end_pos = loop {
        let n = stream.read(&mut tmp).map_err(|e| format!("Read: {}", e))?; if n==0 { return Err("EOF before headers".into()); }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(p) = find_subsequence(&buf, b"\r\n\r\n") { break p+4; }
    };
    let header_bytes = &buf[..header_end_pos];
    let (status, _status_text, headers, _body0) = parse_http_response(&buf).map_err(|e| format!("parse: {}", e))?;
    if status < 200 || status >= 300 { return Err(format!("HTTP status {}", status)); }
    // Prepare file
    let mut file = std::fs::File::create(path).map_err(|e| format!("open {}: {}", path, e))?;
    // If chunked
    if has_header_case(&headers.iter().map(|(k,v)|(k.clone(), match v { Value::String(s)=>s.clone(), Value::Symbol(s)=>s.clone(), _=>String::new() })).collect::<Vec<_>>(), "Transfer-Encoding", "chunked") {
        let mut body_initial = &buf[header_end_pos..];
        // Decode chunked by streaming
        let mut rest = body_initial.to_vec();
        loop {
            // Read chunk size line
            let mut line_end = rest.windows(2).position(|w| w==b"\r\n");
            while line_end.is_none() {
                let n = stream.read(&mut tmp).map_err(|e| format!("Read: {}", e))?; if n==0 { return Err("EOF in chunked size".into()); }
                rest.extend_from_slice(&tmp[..n]);
                line_end = rest.windows(2).position(|w| w==b"\r\n");
            }
            let pos = line_end.unwrap();
            let size_hex = String::from_utf8_lossy(&rest[..pos]);
            let size = usize::from_str_radix(size_hex.trim().split(';').next().unwrap_or("0"), 16).unwrap_or(0);
            rest.drain(..pos+2);
            if size==0 { break; }
            while rest.len() < size+2 {
                let n = stream.read(&mut tmp).map_err(|e| format!("Read: {}", e))?; if n==0 { return Err("EOF in chunk".into()); }
                rest.extend_from_slice(&tmp[..n]);
            }
            file.write_all(&rest[..size]).map_err(|e| format!("write: {}", e))?;
            rest.drain(..size+2); // remove data + CRLF
        }
    } else if let Some((_, Value::String(cl))) = headers.iter().find(|(k,_)| k.eq_ignore_ascii_case("Content-Length")) {
        let len: usize = cl.parse().unwrap_or(0);
        let mut written = 0usize;
        let mut body_initial = &buf[header_end_pos..];
        if !body_initial.is_empty() { file.write_all(body_initial).map_err(|e| format!("write: {}", e))?; written += body_initial.len(); }
        while written < len {
            let n = stream.read(&mut tmp).map_err(|e| format!("Read: {}", e))?; if n==0 { break; }
            file.write_all(&tmp[..n]).map_err(|e| format!("write: {}", e))?;
            written += n;
        }
    } else {
        // Unknown length: just copy to EOF
        let mut body_initial = &buf[header_end_pos..];
        if !body_initial.is_empty() { file.write_all(body_initial).map_err(|e| format!("write: {}", e))?; }
        loop {
            let n = stream.read(&mut tmp).map_err(|e| format!("Read: {}", e))?; if n==0 { break; }
            file.write_all(&tmp[..n]).map_err(|e| format!("write: {}", e))?;
        }
    }
    Ok(())
}

#[cfg(feature = "net_https")]
fn download_stream_reqwest(url: &str, path: &str, opts: &HttpOpts) -> Result<(), String> {
    use reqwest::blocking::ClientBuilder;
    use reqwest::{Method, redirect};
    let mut cb = ClientBuilder::new();
    if let Some(ms) = opts.timeout_ms { cb = cb.timeout(std::time::Duration::from_millis(ms)); }
    if opts.tls_insecure { cb = cb.danger_accept_invalid_certs(true).danger_accept_invalid_hostnames(true); }
    if let Some(n) = opts.follow_redirects { cb = if n==0 { cb.redirect(redirect::Policy::none()) } else { cb.redirect(redirect::Policy::limited(n)) } }
    let client = cb.build().map_err(|e| format!("Client: {}", e))?;
    let mut url_final = url.to_string();
    if !opts.query.is_empty() { url_final = append_query(&url_final, &opts.query); }
    let mut req = client.request(Method::GET, &url_final);
    for (k, v) in &opts.headers { req = req.header(k, v); }
    let mut resp = req.send().map_err(|e| format!("send: {}", e))?;
    if !resp.status().is_success() { return Err(format!("HTTP status {}", resp.status())); }
    let mut file = std::fs::File::create(path).map_err(|e| format!("open {}: {}", path, e))?;
    let _ = std::io::copy(&mut resp, &mut file).map_err(|e| format!("copy: {}", e))?;
    Ok(())
}

fn collect_kv_pairs(ev: &mut Evaluator, v: Value) -> Vec<(String, String)> {
    match ev.eval(v) {
        Value::Assoc(m) => m.into_iter().map(|(k, vv)| (k, value_to_string(&vv))).collect(),
        Value::List(items) => items.into_iter().filter_map(|it| match ev.eval(it) {
            Value::List(mut pair) if pair.len()==2 => {
                let b = pair.pop().unwrap(); let a = pair.pop().unwrap(); Some((value_to_string(&a), value_to_string(&b)))
            }
            _ => None
        }).collect(),
        other => { let s = value_to_string(&other); if !s.is_empty() { vec![("".into(), s)] } else { vec![] } }
    }
}

fn value_to_string(v: &Value) -> String {
    match v { Value::String(s)|Value::Symbol(s)=>s.clone(), Value::Integer(n)=>n.to_string(), Value::Real(f)=>f.to_string(), Value::Boolean(b)=>b.to_string(), _=> lyra_core::pretty::format_value(v) }
}

fn form_urlencode(pairs: &Vec<(String, String)>) -> String {
    pairs.iter().map(|(k,v)| format!("{}={}", percent_encode_component(k,false), percent_encode_component(v,true))).collect::<Vec<_>>().join("&")
}

// ---- Streaming client registry (requires net_https) ----
use std::sync::{OnceLock as OnceLock2, Mutex as Mutex2};
use std::sync::atomic::AtomicI64 as AtomicI64_2;
static STREAM_REG: OnceLock2<Mutex2<std::collections::HashMap<i64, Box<dyn std::io::Read + Send>>>> = OnceLock2::new();
static STREAM_NEXT: OnceLock2<AtomicI64_2> = OnceLock2::new();
fn sreg() -> &'static Mutex2<std::collections::HashMap<i64, Box<dyn std::io::Read + Send>>> { STREAM_REG.get_or_init(|| Mutex2::new(std::collections::HashMap::new())) }
fn snext() -> i64 { STREAM_NEXT.get_or_init(|| AtomicI64_2::new(1)).fetch_add(1, std::sync::atomic::Ordering::Relaxed) }

fn http_stream_request(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("HttpStreamRequest".into())), args } }
    let method = to_string_arg(ev, args[0].clone());
    let url = to_string_arg(ev, args[1].clone());
    let opts = http_opts_from(ev, args.get(2).cloned());
    #[cfg(feature = "net_https")]
    {
        use reqwest::blocking::ClientBuilder; use reqwest::{Method, redirect};
        let mut cb = ClientBuilder::new();
        if let Some(ms) = opts.timeout_ms { cb = cb.timeout(std::time::Duration::from_millis(ms)); }
        if opts.tls_insecure { cb = cb.danger_accept_invalid_certs(true).danger_accept_invalid_hostnames(true); }
        if let Some(n) = opts.follow_redirects { cb = if n==0 { cb.redirect(redirect::Policy::none()) } else { cb.redirect(redirect::Policy::limited(n)) } }
        let client = match cb.build() { Ok(c)=>c, Err(e)=> return failure("HTTP::stream", &e.to_string()) };
        let m = match Method::from_bytes(method.as_bytes()) { Ok(m)=>m, Err(e)=> return failure("HTTP::stream", &e.to_string()) };
        let mut url_final = url.clone();
        if !opts.query.is_empty() { url_final = append_query(&url_final, &opts.query); }
        let mut req = client.request(m, &url_final);
        for (k, v) in &opts.headers { req = req.header(k, v); }
        let resp = match req.send() { Ok(r)=>r, Err(e)=> return failure("HTTP::stream", &e.to_string()) };
        let status = resp.status();
        let status_code = status.as_u16();
        let mut headers_vec: Vec<(String, Value)> = Vec::new();
        for (k, v) in resp.headers().iter() { headers_vec.push((k.as_str().to_string(), Value::String(v.to_str().unwrap_or("").to_string()))); }
        let reader: Box<dyn std::io::Read + Send> = Box::new(resp);
        let id = snext(); sreg().lock().unwrap().insert(id, reader);
        return Value::Assoc(vec![
            ("status".into(), Value::Integer(status_code as i64)),
            ("url".into(), Value::String(url_final)),
            ("headers".into(), Value::Assoc(headers_vec.into_iter().collect())),
            ("stream".into(), Value::Integer(id)),
        ].into_iter().collect());
    }
    #[allow(unreachable_code)]
    failure("HTTP::stream", "HTTPS client not enabled")
}

fn http_stream_read(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("HttpStreamRead".into())), args } }
    let id = match ev.eval(args[0].clone()) { Value::Integer(n)=>n, _=>0 };
    let max = if args.len()>1 { match ev.eval(args[1].clone()) { Value::Integer(n) if n>0=> n as usize, _=> 65536 } } else { 65536 };
    let mut guard = sreg().lock().unwrap();
    if let Some(reader) = guard.get_mut(&id) {
        let mut buf = vec![0u8; max];
        match reader.read(&mut buf) { Ok(n)=> { buf.truncate(n); Value::Assoc(vec![("chunk".into(), Value::List(buf.into_iter().map(|b| Value::Integer(b as i64)).collect())), ("done".into(), Value::Boolean(n==0))].into_iter().collect()) }, Err(e)=> failure("HTTP::stream", &e.to_string()) }
    } else { failure("HTTP::stream", "Invalid stream handle") }
}

fn http_stream_close(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("HttpStreamClose".into())), args } }
    let id = match ev.eval(args[0].clone()) { Value::Integer(n)=>n, _=>0 };
    sreg().lock().unwrap().remove(&id);
    Value::Boolean(true)
}

fn http_download_cached(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("HttpDownloadCached".into())), args } }
    let url = to_string_arg(ev, args[0].clone());
    let dest = to_string_arg(ev, args[1].clone());
    let opts = http_opts_from(ev, args.get(2).cloned());
    let _cache_dir = match args.get(2).and_then(|v| if let Value::Assoc(m)=ev.eval(v.clone()) { m.get("CacheDir").cloned() } else { None }) { Some(Value::String(s))|Some(Value::Symbol(s)) => Some(s), _ => None };
    let ttl_ms = match args.get(2).and_then(|v| if let Value::Assoc(m)=ev.eval(v.clone()) { m.get("TtlMs").cloned() } else { None }) { Some(Value::Integer(n)) if n>0 => Some(n as i64), _ => None };
    let meta_path = format!("{}.etag", &dest);
    let mut etag_old: Option<String> = None;
    if let (Some(ttl), Ok(md)) = (ttl_ms, std::fs::metadata(&dest)) {
        let age = std::time::SystemTime::now().duration_since(md.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH)).map(|d| d.as_millis() as i64).unwrap_or(i64::MAX);
        if age < ttl { if let Ok(s) = std::fs::read_to_string(&meta_path) { etag_old = Some(s.trim().to_string()); } }
    }
    #[cfg(feature = "net_https")]
    {
        use reqwest::blocking::ClientBuilder; use reqwest::{Method, redirect};
        let mut cb = ClientBuilder::new();
        if let Some(ms) = opts.timeout_ms { cb = cb.timeout(std::time::Duration::from_millis(ms)); }
        if opts.tls_insecure { cb = cb.danger_accept_invalid_certs(true).danger_accept_invalid_hostnames(true); }
        if let Some(n) = opts.follow_redirects { cb = if n==0 { cb.redirect(redirect::Policy::none()) } else { cb.redirect(redirect::Policy::limited(n)) } }
        let client = match cb.build() { Ok(c)=>c, Err(e)=> return failure("HTTP::cache", &e.to_string()) };
        let mut url_final = url.clone();
        if !opts.query.is_empty() { url_final = append_query(&url_final, &opts.query); }
        let mut req = client.request(Method::GET, &url_final);
        for (k, v) in &opts.headers { req = req.header(k, v); }
        if let Some(et) = &etag_old { req = req.header("If-None-Match", et); }
        let mut resp = match req.send() { Ok(r)=>r, Err(e)=> return failure("HTTP::cache", &e.to_string()) };
        if resp.status()==reqwest::StatusCode::NOT_MODIFIED { return Value::Assoc(vec![("path".into(), Value::String(dest)), ("bytes_written".into(), Value::Integer(0)), ("from_cache".into(), Value::Boolean(true)), ("etag".into(), Value::String(etag_old.unwrap_or_default()))].into_iter().collect()); }
        let mut file = match std::fs::File::create(&dest) { Ok(f)=>f, Err(e)=> return failure("HTTP::cache", &e.to_string()) };
        let n = match std::io::copy(&mut resp, &mut file) { Ok(n)=>n, Err(e)=> return failure("HTTP::cache", &e.to_string()) } as i64;
        if let Some(et) = resp.headers().get("ETag").and_then(|v| v.to_str().ok()) { let _ = std::fs::write(&meta_path, et.as_bytes()); }
        return Value::Assoc(vec![("path".into(), Value::String(dest)), ("bytes_written".into(), Value::Integer(n)), ("from_cache".into(), Value::Boolean(false))].into_iter().collect());
    }
    #[allow(unreachable_code)]
    failure("HTTP::cache", "HTTPS client not enabled")
}

fn http_retry(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("HttpRetry".into())), args } }
    let req_map = match ev.eval(args[0].clone()) { Value::Assoc(m)=>m, _ => return failure("HTTP::retry", "Request map required") };
    let opts = if args.len()>1 { if let Value::Assoc(m)=ev.eval(args[1].clone()) { m } else { std::collections::HashMap::new() } } else { std::collections::HashMap::new() };
    let attempts = opts.get("Attempts").and_then(|v| if let Value::Integer(n)=v { Some(*n as i32) } else { None }).unwrap_or(3).max(1);
    let backoff_min = opts.get("BackoffMs").and_then(|v| if let Value::Integer(n)=v { Some(*n as i64) } else { None }).unwrap_or(200);
    let mut last_resp: Option<Value> = None;
    for i in 0..attempts {
        let resp = http_request_generic(ev, vec![Value::Assoc(req_map.clone())]);
        if let Value::Assoc(m) = &resp { if let Some(Value::Integer(code)) = m.get("status") { if *code>=200 && *code<300 { return resp; } } }
        last_resp = Some(resp);
        if i < attempts-1 { std::thread::sleep(std::time::Duration::from_millis((backoff_min * (i as i64 + 1)) as u64)); }
    }
    last_resp.unwrap_or_else(|| failure("HTTP::retry", "No attempts"))
}
// --- HTTPS capable implementation via reqwest (feature net_https) ---
#[cfg(feature = "net_https")]
fn http_request_reqwest(method: &str, url: &str, body: Option<&[u8]>, opts: &HttpOpts) -> Result<Value, String> {
    use reqwest::blocking::ClientBuilder;
    use reqwest::{Method, redirect};
    use reqwest::blocking::multipart as mp;
    let mut cb = ClientBuilder::new();
    if let Some(ms) = opts.timeout_ms { cb = cb.timeout(std::time::Duration::from_millis(ms)); }
    if opts.tls_insecure { cb = cb.danger_accept_invalid_certs(true).danger_accept_invalid_hostnames(true); }
    if let Some(n) = opts.follow_redirects { cb = if n==0 { cb.redirect(redirect::Policy::none()) } else { cb.redirect(redirect::Policy::limited(n)) } }
    // allow default gzip/brotli/deflate via reqwest features
    let client = cb.build().map_err(|e| format!("Client: {}", e))?;
    let m = Method::from_bytes(method.as_bytes()).map_err(|e| format!("method: {}", e))?;
    // Query params
    let mut url_final = url.to_string();
    if !opts.query.is_empty() { url_final = append_query(&url_final, &opts.query); }
    let mut req = client.request(m, &url_final);
    for (k, v) in &opts.headers { req = req.header(k, v); }
    // body resolution: Multipart > Json > Form > explicit body
    if let Some(parts) = &opts.multipart {
        let mut form = mp::Form::new();
        for p in parts {
            match &p.value {
                MultipartValue::Text(s) => { form = form.text(p.name.clone(), s.clone()); }
                MultipartValue::File { path, filename, content_type } => {
                    let mut part = mp::Part::file(path).map_err(|e| format!("file: {}", e))?;
                    if let Some(fname) = filename { part = part.file_name(fname.clone()); }
                    if let Some(ct) = content_type { part = part.mime_str(ct).map_err(|e| format!("mime: {}", e))?; }
                    form = form.part(p.name.clone(), part);
                }
            }
        }
        req = req.multipart(form);
    }
    else if let Some(j) = &opts.json_body { req = req.json(j); }
    else if let Some(form) = &opts.form_body { req = req.form(&form); }
    else if let Some(b) = body { req = req.body(b.to_vec()); }
    let resp = req.send().map_err(|e| format!("send: {}", e))?;
    let status = resp.status();
    let status_code = status.as_u16();
    let status_text = status.canonical_reason().unwrap_or("").to_string();
    let mut headers_vec: Vec<(String, Value)> = Vec::new();
    for (k, v) in resp.headers().iter() { headers_vec.push((k.as_str().to_string(), Value::String(v.to_str().unwrap_or("").to_string()))); }
    // Decode body per As (Auto handled inside build_response via Content-Type)
    let bytes = resp.bytes().map_err(|e| format!("read: {}", e))?.to_vec();
    Ok(build_response(status_code, status_text, headers_vec, bytes, opts))
}

// --- Minimal HTTP server (feature net) ---
use std::sync::{OnceLock, Mutex, Arc, atomic::{AtomicBool, Ordering}};
use std::collections::HashMap;
use std::sync::atomic::AtomicI64;

#[derive(Clone)]
struct ServerOpts { host: String, port: u16, read_timeout_ms: Option<u64>, write_timeout_ms: Option<u64>, max_body_bytes: usize }

fn server_opts_from(ev: &mut Evaluator, v: Option<Value>) -> ServerOpts {
    let mut o = ServerOpts { host: "127.0.0.1".into(), port: 8080, read_timeout_ms: None, write_timeout_ms: None, max_body_bytes: 2*1024*1024 };
    if let Some(Value::Assoc(m)) = v.map(|x| ev.eval(x)) {
        if let Some(Value::Integer(p)) = m.get("Port") { if *p>0 && *p<=65535 { o.port = *p as u16; } }
        if let Some(Value::String(h))|Some(Value::Symbol(h)) = m.get("Host") { o.host = h.clone(); }
        if let Some(Value::Integer(ms)) = m.get("ReadTimeoutMs") { if *ms>0 { o.read_timeout_ms = Some(*ms as u64); } }
        if let Some(Value::Integer(ms)) = m.get("WriteTimeoutMs") { if *ms>0 { o.write_timeout_ms = Some(*ms as u64); } }
        if let Some(Value::Integer(n)) = m.get("MaxBodyBytes") { if *n>0 { o.max_body_bytes = *n as usize; } }
    }
    o
}

#[cfg(feature = "net_https")]
#[derive(Clone)]
struct TlsOpts { cert_path: String, key_path: String }

#[cfg(feature = "net_https")]
fn tls_opts_from(ev: &mut Evaluator, v: Option<Value>) -> Result<(ServerOpts, TlsOpts), String> {
    let srv = server_opts_from(ev, v.clone());
    if let Some(Value::Assoc(m)) = v.map(|x| ev.eval(x)) {
        let cert_path = match m.get("CertPath") { Some(Value::String(s))|Some(Value::Symbol(s)) => s.clone(), _ => return Err("CertPath required".into()) };
        let key_path = match m.get("KeyPath") { Some(Value::String(s))|Some(Value::Symbol(s)) => s.clone(), _ => return Err("KeyPath required".into()) };
        Ok((srv, TlsOpts { cert_path, key_path }))
    } else { Err("TLS options required".into()) }
}

#[derive(Clone)]
struct ServerEntry { stop: Arc<AtomicBool>, addr: String }

static SERVERS: OnceLock<Mutex<HashMap<i64, ServerEntry>>> = OnceLock::new();
static NEXT_SRV_ID: OnceLock<AtomicI64> = OnceLock::new();
fn servers_reg() -> &'static Mutex<HashMap<i64, ServerEntry>> { SERVERS.get_or_init(|| Mutex::new(HashMap::new())) }
fn next_srv_id() -> i64 { let a = NEXT_SRV_ID.get_or_init(|| AtomicI64::new(1)); a.fetch_add(1, Ordering::Relaxed) }

fn http_serve(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // HttpServe[handler, opts]
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("HttpServe".into())), args } }
    let handler = args[0].clone();
    let opts = server_opts_from(ev, args.get(1).cloned());
    // snapshot environment to reuse inside server threads
    let stop = Arc::new(AtomicBool::new(false));
    let stop_cl = stop.clone();
    let host = opts.host.clone();
    let port = opts.port;
    let read_to = opts.read_timeout_ms.map(|ms| std::time::Duration::from_millis(ms));
    let write_to = opts.write_timeout_ms.map(|ms| std::time::Duration::from_millis(ms));
    let max_body = opts.max_body_bytes;
    let addr_str;
    // Bind listener
    let listener = match std::net::TcpListener::bind((host.as_str(), port)) { Ok(l)=>l, Err(e)=> return failure("HTTP::listen", &format!("bind: {}", e)) };
    addr_str = listener.local_addr().map(|a| a.to_string()).unwrap_or(format!("{}:{}", host, port));
    let addr_for_wake = addr_str.clone();
    // Spawn a background thread that accepts connections
    std::thread::spawn(move || {
        if let Some(d) = read_to { let _ = listener.set_nonblocking(false); let _ = listener.set_ttl(d.as_secs() as u32); }
        loop {
            match listener.accept() {
                Ok((mut stream, _peer)) => {
                    if stop_cl.load(Ordering::Relaxed) { let _ = stream.shutdown(std::net::Shutdown::Both); break; }
                    let handler_v = handler.clone();
                    let stop_req = stop_cl.clone();
                    std::thread::spawn(move || {
                        use std::io::{Read, Write};
                        let _ = read_to.map(|d| stream.set_read_timeout(Some(d)));
                        let _ = write_to.map(|d| stream.set_write_timeout(Some(d)));
                        let mut buf = Vec::new();
                        // Read until header end; then read body if Content-Length
                        let mut header = Vec::new();
                        let mut tmp = [0u8; 1024];
                        let mut total = 0usize;
                        let mut body_len_opt: Option<usize> = None;
                        let header_end;
                        loop {
                            match stream.read(&mut tmp) { Ok(0)=>break, Ok(n)=>{ buf.extend_from_slice(&tmp[..n]); total+=n; if total> (max_body + 16_384) { break; } }, Err(_)=>break }
                            if let Some(p) = find_subsequence(&buf, b"\r\n\r\n") { header_end = p+4; header = buf[..header_end].to_vec(); if let Some(cl) = parse_content_length(&header) { body_len_opt = Some(cl); } break; }
                        }
                        let mut body = Vec::new();
                        let mut headers_vec: Vec<(String, String)> = Vec::new();
                        let _ = parse_request_head(&header).map(|(_,_,hs)| { headers_vec = hs; });
                        if let Some(hend) = find_subsequence(&buf, b"\r\n\r\n") {
                            body.extend_from_slice(&buf[hend..]);
                        }
                        if parse_request_has_chunked(&headers_vec) {
                            match read_chunked(&mut stream, &body, max_body) { Ok(data)=> body = data, Err(_)=>{} }
                        } else if let Some(blen) = body_len_opt { while body.len() < blen { match stream.read(&mut tmp) { Ok(0)=>break, Ok(n)=>{ body.extend_from_slice(&tmp[..n]); if body.len()>max_body { break; } }, Err(_)=>break } } }
                        // Parse request
                        let (method, target, headers) = match parse_request_head(&header) { Some(t)=>t, None=>{ let _ = write_simple_resp(&mut stream, 400, "Bad Request", &[], b""); return; } };
                        let (path, query_map) = split_path_query(&target);
                        // Build request map
                        let mut req_map: HashMap<String, Value> = HashMap::new();
                        req_map.insert("method".into(), Value::String(method));
                        req_map.insert("path".into(), Value::String(path));
                        req_map.insert("query".into(), Value::Assoc(query_map));
                        let mut hdr_assoc: HashMap<String, Value> = HashMap::new();
                        for (k,v) in headers { hdr_assoc.insert(k, Value::String(v)); }
                        req_map.insert("headers".into(), Value::Assoc(hdr_assoc));
                        req_map.insert("body".into(), Value::List(body.iter().map(|b| Value::Integer(*b as i64)).collect()));
                        // Evaluate handler
                        let mut evh = lyra_runtime::Evaluator::new();
                        crate::register_all(&mut evh);
                        let req_val = Value::Assoc(req_map);
                        let call = Value::Expr { head: Box::new(handler_v), args: vec![req_val] };
                        let out = evh.eval(call);
                        // Compose response (supports streaming and file)
                        let spec = render_response_from_value(out);
                        let status = spec.status;
                        let mut headers = spec.headers;
                        if let Some(ref body) = spec.body { headers.entry("Content-Length".into()).or_insert(body.len().to_string()); }
                        if let Some(file_path) = spec.body_file.clone() {
                            let mut hdrs: Vec<(String,String)> = headers.into_iter().collect();
                            let _ = write_chunked_resp(&mut stream, status, "OK", hdrs, Box::new(FileChunkIter::new(&file_path)));
                        } else if let Some(chunks) = spec.chunks.clone() {
                            let mut hdrs: Vec<(String,String)> = headers.into_iter().collect();
                            let _ = write_chunked_resp(&mut stream, status, "OK", hdrs, Box::new(chunks.into_iter()));
                        } else if let Some(cid) = spec.chan_id {
                            let mut hdrs: Vec<(String,String)> = headers.into_iter().collect();
                            let _ = write_chunked_resp(&mut stream, status, "OK", hdrs, Box::new(ChannelChunkIter::new(cid)));
                        } else if let Some(body) = spec.body.clone() {
                            let mut hdrs: Vec<(String,String)> = headers.into_iter().collect();
                            let mut has_conn = false; for (k,v) in &hdrs { if k.eq_ignore_ascii_case("Connection") { has_conn = true; } }
                            if !has_conn { hdrs.push(("Connection".into(), "close".into())); }
                            let _ = write_simple_resp(&mut stream, status, "OK", &hdrs, &body);
                        }
                        let _ = stream.flush();
                        let _ = stream.shutdown(std::net::Shutdown::Both);
                        drop(stop_req);
                    });
                }
                Err(e) => {
                    // Wake and exit if stopping
                    if stop_cl.load(Ordering::Relaxed) { break; }
                    // Transient accept error; small sleep to avoid busy loop
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
            }
        }
    });
    // Register server entry and return handle
    let id = next_srv_id();
    servers_reg().lock().unwrap().insert(id, ServerEntry { stop, addr: addr_str.clone() });
    Value::Expr { head: Box::new(Value::Symbol("ServerId".into())), args: vec![Value::Integer(id)] }
}

fn http_server_stop(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("HttpServerStop".into())), args } }
    let id_opt = match &args[0] {
        Value::Expr { head, args } if matches!(&**head, Value::Symbol(s) if s=="ServerId") => args.get(0).and_then(|v| if let Value::Integer(i)=v { Some(*i) } else { None }),
        Value::Integer(i) => Some(*i),
        _ => None,
    };
    if let Some(id) = id_opt {
        if let Some(entry) = servers_reg().lock().unwrap().remove(&id) {
            entry.stop.store(true, Ordering::Relaxed);
            // Wake accept by dialing loopback
            let _ = std::net::TcpStream::connect(entry.addr);
            return Value::Boolean(true);
        }
    }
    failure("HTTP::server", "Unknown server id")
}

fn http_server_addr(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("HttpServerAddr".into())), args } }
    let id_opt = match &args[0] {
        Value::Expr { head, args } if matches!(&**head, Value::Symbol(s) if s=="ServerId") => args.get(0).and_then(|v| if let Value::Integer(i)=v { Some(*i) } else { None }),
        Value::Integer(i) => Some(*i),
        _ => None,
    };
    if let Some(id) = id_opt { if let Some(entry) = servers_reg().lock().unwrap().get(&id) { return Value::String(entry.addr.clone()); } }
    failure("HTTP::server", "Unknown server id")
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w==needle)
}

fn parse_content_length(header: &[u8]) -> Option<usize> {
    let s = String::from_utf8_lossy(header);
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("Content-Length:") { return rest.trim().parse::<usize>().ok(); }
        if let Some(rest) = line.strip_prefix("content-length:") { return rest.trim().parse::<usize>().ok(); }
    }
    None
}

fn parse_request_head(head: &[u8]) -> Option<(String, String, Vec<(String, String)>)> {
    let s = String::from_utf8_lossy(head);
    let mut lines = s.lines();
    let line1 = lines.next()?;
    let mut parts = line1.splitn(3, ' ');
    let method = parts.next()?.to_string();
    let target = parts.next()?.to_string();
    let mut headers = Vec::new();
    for l in lines { if l.is_empty() { break; } if let Some((k,v)) = l.split_once(':') { headers.push((k.trim().to_string(), v.trim().to_string())); } }
    Some((method, target, headers))
}

fn split_path_query(target: &str) -> (String, HashMap<String, Value>) {
    let mut out: HashMap<String, Value> = HashMap::new();
    let (path, qs) = if let Some(pos) = target.find('?') { (&target[..pos], &target[pos+1..]) } else { (target, "") };
    for pair in qs.split('&') { if pair.is_empty() { continue; } if let Some((k,v)) = pair.split_once('=') { out.insert(url_decode(k), Value::String(url_decode(v))); } else { out.insert(url_decode(pair), Value::String(String::new())); } }
    (path.to_string(), out)
}

fn url_decode(s: &str) -> String {
    let mut out = String::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => { out.push(' '); i+=1; }
            b'%' if i+2 < bytes.len() => {
                if let (Some(h), Some(l)) = (hex_val(bytes[i+1]), hex_val(bytes[i+2])) { out.push(((h<<4)|l) as char); i+=3; } else { out.push('%'); i+=1; }
            }
            b => { out.push(b as char); i+=1; }
        }
    }
    out
}

fn hex_val(b: u8) -> Option<u8> {
    match b { b'0'..=b'9' => Some(b - b'0'), b'a'..=b'f' => Some(b - b'a' + 10), b'A'..=b'F' => Some(b - b'A' + 10), _=>None }
}

struct ResponseSpec { status: u16, headers: std::collections::HashMap<String,String>, body: Option<Vec<u8>>, body_file: Option<String>, chunks: Option<Vec<Vec<u8>>>, chan_id: Option<i64> }

fn render_response_from_value(v: Value) -> ResponseSpec {
    match v {
        Value::Assoc(m) => {
            let mut status: u16 = 200;
            let mut headers: std::collections::HashMap<String,String> = std::collections::HashMap::new();
            let mut body: Option<Vec<u8>> = None;
            let mut body_file: Option<String> = None;
            let mut chunks: Option<Vec<Vec<u8>>> = None;
            let mut chan_id: Option<i64> = None;
            if let Some(Value::Integer(s)) = m.get("status") { if *s>=100 && *s<=599 { status = *s as u16; } }
            if let Some(Value::Assoc(h)) = m.get("headers") { for (k, vv) in h { if let Value::String(s)|Value::Symbol(s) = vv { headers.insert(k.clone(), s.clone()); } } }
            if let Some(j) = m.get("json") { let jv = value_to_json(j); if let Ok(s) = serde_json::to_string(&jv) { body = Some(s.into_bytes()); headers.entry("Content-Type".into()).or_insert("application/json".into()); } }
            if body.is_none() { if let Some(bf) = m.get("bodyFile") { if let Value::String(p)|Value::Symbol(p) = bf { body_file = Some(p.clone()); headers.entry("Content-Type".into()).or_insert("application/octet-stream".into()); } } }
            if body.is_none() && body_file.is_none() {
                if let Some(st) = m.get("stream") {
                    match st {
                        Value::List(list) => {
                            let mut vecs: Vec<Vec<u8>> = Vec::new();
                            for it in list {
                                match it {
                                    Value::String(s) | Value::Symbol(s) => vecs.push(s.clone().into_bytes()),
                                    Value::List(bytes) => { let mut out = Vec::with_capacity(bytes.len()); for b in bytes { if let Value::Integer(i)=b { out.push(*i as u8); } } vecs.push(out); }
                                    other => { let j = value_to_json(other); if let Ok(s) = serde_json::to_string(&j) { vecs.push(s.into_bytes()); headers.entry("Content-Type".into()).or_insert("application/json".into()); } }
                                }
                            }
                            chunks = Some(vecs);
                        }
                        Value::Expr { head, args } if matches!(&**head, Value::Symbol(s) if s=="ChannelId") => {
                            if let Some(Value::Integer(i)) = args.get(0) { chan_id = Some(*i); }
                        }
                        _ => {}
                    }
                }
            }
            if body.is_none() && body_file.is_none() && chunks.is_none() {
                if let Some(b) = m.get("body") { match b {
                    Value::String(s) | Value::Symbol(s) => { body = Some(s.clone().into_bytes()); headers.entry("Content-Type".into()).or_insert("text/plain; charset=utf-8".into()); }
                    Value::List(list) => { let mut tmp = Vec::with_capacity(list.len()); for it in list { if let Value::Integer(i)=it { tmp.push(*i as u8); } } body = Some(tmp); headers.entry("Content-Type".into()).or_insert("application/octet-stream".into()); }
                    other => { let j = value_to_json(&other); if let Ok(s) = serde_json::to_string(&j) { body = Some(s.into_bytes()); headers.entry("Content-Type".into()).or_insert("application/json".into()); } }
                } }
            }
            if !headers.contains_key("Content-Type") {
                if let Some(ref body_bytes) = body { if let Ok(s) = std::str::from_utf8(body_bytes) {
                    let s_trim = s.trim_start();
                    if s_trim.starts_with("<!DOCTYPE html") || s_trim.starts_with("<html") { headers.insert("Content-Type".into(), "text/html; charset=utf-8".into()); }
                    else if (s_trim.starts_with('{') || s_trim.starts_with('[')) && serde_json::from_str::<serde_json::Value>(s_trim).is_ok() { headers.insert("Content-Type".into(), "application/json".into()); }
                    else { headers.insert("Content-Type".into(), "text/plain; charset=utf-8".into()); }
                } }
            }
            ResponseSpec { status, headers, body, body_file, chunks, chan_id }
        }
        Value::String(s) | Value::Symbol(s) => ResponseSpec { status: 200, headers: std::collections::HashMap::from([(String::from("Content-Type"), String::from("text/plain; charset=utf-8"))]), body: Some(s.into_bytes()), body_file: None, chunks: None, chan_id: None },
        other => {
            let j = value_to_json(&other);
            let body = serde_json::to_vec(&j).unwrap_or_else(|_| b"null".to_vec());
            ResponseSpec { status: 200, headers: std::collections::HashMap::from([(String::from("Content-Type"), String::from("application/json"))]), body: Some(body), body_file: None, chunks: None, chan_id: None }
        }
    }
}


// Minimal JSON bridge for net
fn value_to_json(v: &Value) -> sj::Value {
    match v {
        Value::Integer(n) => sj::Value::from(*n),
        Value::Real(f) => sj::Value::from(*f),
        Value::Boolean(b) => sj::Value::from(*b),
        Value::String(s) | Value::Symbol(s) => sj::Value::from(s.clone()),
        Value::List(xs) => sj::Value::Array(xs.iter().map(value_to_json).collect()),
        Value::Assoc(m) => {
            let mut map = serde_json::Map::new();
            for (k, vv) in m { map.insert(k.clone(), value_to_json(vv)); }
            sj::Value::Object(map)
        }
        Value::Expr { .. } | Value::Slot(_) | Value::PureFunction { .. } | Value::PackedArray { .. } | Value::Complex { .. } | Value::Rational { .. } | Value::BigReal(_) => {
            sj::Value::Null
        }
    }
}

fn json_to_value(j: &sj::Value) -> Value {
    match j {
        sj::Value::Null => Value::Symbol("Null".into()),
        sj::Value::Bool(b) => Value::Boolean(*b),
        sj::Value::Number(n) => {
            if let Some(i) = n.as_i64() { Value::Integer(i) } else if let Some(f) = n.as_f64() { Value::Real(f) } else { Value::Real(0.0) }
        }
        sj::Value::String(s) => Value::String(s.clone()),
        sj::Value::Array(xs) => Value::List(xs.iter().map(json_to_value).collect()),
        sj::Value::Object(m) => Value::Assoc(m.iter().map(|(k,v)| (k.clone(), json_to_value(v))).collect()),
    }
}

fn write_simple_resp(stream: &mut std::net::TcpStream, status: u16, reason: &str, headers: &[(String,String)], body: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let mut head = format!("HTTP/1.1 {} {}\r\n", status, reason);
    for (k,v) in headers { head.push_str(&format!("{}: {}\r\n", k, v)); }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    Ok(())
}

#[cfg(feature = "net_https")]
fn write_simple_resp_tls<S: std::io::Write>(stream: &mut S, status: u16, reason: &str, headers: &[(String,String)], body: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let mut head = format!("HTTP/1.1 {} {}\r\n", status, reason);
    for (k,v) in headers { head.push_str(&format!("{}: {}\r\n", k, v)); }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    Ok(())
}

#[cfg(feature = "net_https")]
fn http_serve_tls(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // HttpServeTls[handler, <|Port->8443, CertPath->..., KeyPath->...|>]
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("HttpServeTls".into())), args } }
    let handler = args[0].clone();
    let (opts, tls) = match tls_opts_from(ev, args.get(1).cloned()) { Ok(v)=>v, Err(e)=> return failure("HTTP::tls", &e) };
    // Load certs/keys
    let certs = match std::fs::File::open(&tls.cert_path).and_then(|mut f| { let mut reader = std::io::BufReader::new(f); let certs = rustls_pemfile::certs(&mut reader)?; Ok(certs) }) { Ok(v)=> v.into_iter().map(rustls::Certificate).collect::<Vec<_>>(), Err(e)=> return failure("HTTP::tls", &format!("certs: {}", e)) };
    let key = match std::fs::File::open(&tls.key_path).and_then(|mut f| {
        let mut reader = std::io::BufReader::new(f);
        let mut keys = rustls_pemfile::pkcs8_private_keys(&mut reader)?;
        if keys.is_empty() {
            let mut reader = std::io::BufReader::new(std::fs::File::open(&tls.key_path)?);
            keys = rustls_pemfile::rsa_private_keys(&mut reader)?;
        }
        keys.into_iter().next().ok_or(std::io::Error::new(std::io::ErrorKind::InvalidData, "no private key"))
    }) { Ok(k)=> rustls::PrivateKey(k), Err(e)=> return failure("HTTP::tls", &format!("key: {}", e)) };
    let mut cfg = match rustls::ServerConfig::builder().with_safe_defaults().with_no_client_auth().with_single_cert(certs, key) { Ok(c)=>c, Err(e)=> return failure("HTTP::tls", &format!("config: {}", e)) };
    cfg.alpn_protocols = vec![b"http/1.1".to_vec()];
    let acceptor = std::sync::Arc::new(cfg);
    // Bind
    let stop = Arc::new(AtomicBool::new(false));
    let stop_cl = stop.clone();
    let host = opts.host.clone(); let port = opts.port; let read_to = opts.read_timeout_ms.map(|ms| std::time::Duration::from_millis(ms)); let write_to = opts.write_timeout_ms.map(|ms| std::time::Duration::from_millis(ms)); let max_body = opts.max_body_bytes;
    let listener = match std::net::TcpListener::bind((host.as_str(), port)) { Ok(l)=>l, Err(e)=> return failure("HTTP::listen", &format!("bind: {}", e)) };
    let addr_str = listener.local_addr().map(|a| a.to_string()).unwrap_or(format!("{}:{}", host, port));
    let addr_for_stop = addr_str.clone();
    std::thread::spawn(move || {
        loop {
            match listener.accept() {
                Ok((stream, _peer)) => {
                    if stop_cl.load(Ordering::Relaxed) { let _ = stream.shutdown(std::net::Shutdown::Both); break; }
                    let acceptor2 = acceptor.clone();
                    let handler_v = handler.clone();
                    std::thread::spawn(move || {
                        let mut conn = match rustls::ServerConnection::new(acceptor2) { Ok(c)=>c, Err(_)=> return };
                        let mut tls_stream = rustls::StreamOwned::new(conn, stream);
                        use std::io::{Read, Write};
                        if let Some(d) = read_to { let _ = tls_stream.sock.set_read_timeout(Some(d)); }
                        if let Some(d) = write_to { let _ = tls_stream.sock.set_write_timeout(Some(d)); }
                        let mut buf = Vec::new(); let mut tmp = [0u8; 1024]; let mut total = 0usize; let mut header = Vec::new(); let mut body = Vec::new(); let mut body_len_opt=None;
                        loop { match tls_stream.read(&mut tmp) { Ok(0)=>break, Ok(n)=>{ buf.extend_from_slice(&tmp[..n]); total+=n; if total>(max_body+16_384) { break; } }, Err(_)=>break } if let Some(p)=find_subsequence(&buf,b"\r\n\r\n"){ header = buf[..p+4].to_vec(); if let Some(cl)=parse_content_length(&header){ body_len_opt=Some(cl);} break; } }
                        let mut headers_vec: Vec<(String, String)> = Vec::new();
                        let _ = parse_request_head(&header).map(|(_,_,hs)| { headers_vec = hs; });
                        if let Some(hend) = find_subsequence(&buf, b"\r\n\r\n") { body.extend_from_slice(&buf[hend..]); }
                        if parse_request_has_chunked(&headers_vec) {
                            match read_chunked(&mut tls_stream, &body, max_body) { Ok(data)=> body = data, Err(_)=>{} }
                        } else if let Some(blen) = body_len_opt { while body.len()<blen { match tls_stream.read(&mut tmp) { Ok(0)=>break, Ok(n)=>{ body.extend_from_slice(&tmp[..n]); if body.len()>max_body { break; } }, Err(_)=>break } } }
                        let (method, target, headers) = match parse_request_head(&header) { Some(t)=>t, None=>{ let _ = write_simple_resp_tls(&mut tls_stream, 400, "Bad Request", &[], b""); return; } };
                        let (path, query_map) = split_path_query(&target);
                        let mut req_map: HashMap<String, Value> = HashMap::new();
                        req_map.insert("method".into(), Value::String(method));
                        req_map.insert("path".into(), Value::String(path));
                        req_map.insert("query".into(), Value::Assoc(query_map));
                        let mut hdr_assoc: HashMap<String, Value> = HashMap::new(); for (k,v) in headers { hdr_assoc.insert(k, Value::String(v)); }
                        req_map.insert("headers".into(), Value::Assoc(hdr_assoc));
                        req_map.insert("body".into(), Value::List(body.iter().map(|b| Value::Integer(*b as i64)).collect()));
                        let req_val = Value::Assoc(req_map);
                        let mut evh = lyra_runtime::Evaluator::new();
                        crate::register_all(&mut evh);
                        let call = Value::Expr { head: Box::new(handler_v), args: vec![req_val] };
                        let out = evh.eval(call);
                        let spec = render_response_from_value(out);
                        let status = spec.status;
                        let mut headers = spec.headers;
                        if let Some(ref body) = spec.body { headers.entry("Content-Length".into()).or_insert(body.len().to_string()); }
                        if let Some(file_path) = spec.body_file.clone() {
                            let mut hdrs: Vec<(String,String)> = headers.into_iter().collect();
                            let _ = write_chunked_resp_tls(&mut tls_stream, status, "OK", hdrs, Box::new(FileChunkIter::new(&file_path)));
                        } else if let Some(chunks) = spec.chunks.clone() {
                            let mut hdrs: Vec<(String,String)> = headers.into_iter().collect();
                            let _ = write_chunked_resp_tls(&mut tls_stream, status, "OK", hdrs, Box::new(chunks.into_iter()));
                        } else if let Some(cid) = spec.chan_id {
                            let mut hdrs: Vec<(String,String)> = headers.into_iter().collect();
                            let _ = write_chunked_resp_tls(&mut tls_stream, status, "OK", hdrs, Box::new(ChannelChunkIter::new(cid)));
                        } else if let Some(body) = spec.body.clone() {
                            let mut hdrs: Vec<(String,String)> = headers.into_iter().collect();
                            let mut has_conn=false; for (k,v) in &hdrs { if k.eq_ignore_ascii_case("Connection") { has_conn=true; } }
                            if !has_conn { hdrs.push(("Connection".into(), "close".into())); }
                            let _ = write_simple_resp_tls(&mut tls_stream, status, "OK", &hdrs, &body);
                        }
                        let _ = tls_stream.flush();
                    });
                }
                Err(_e) => { if stop_cl.load(Ordering::Relaxed) { break; } std::thread::sleep(std::time::Duration::from_millis(10)); }
            }
        }
    });
    let id = next_srv_id();
    servers_reg().lock().unwrap().insert(id, ServerEntry { stop, addr: addr_for_stop });
    Value::Expr { head: Box::new(Value::Symbol("ServerId".into())), args: vec![Value::Integer(id)] }
}

// --- Routing helpers ---

fn path_match(pattern: &str, path: &str) -> Option<HashMap<String, String>> {
    let pseg: Vec<&str> = pattern.trim_start_matches('/').split('/').collect();
    let aseg: Vec<&str> = path.trim_start_matches('/').split('/').collect();
    let mut params: HashMap<String,String> = HashMap::new();
    let mut i = 0usize;
    let mut j = 0usize;
    while i < pseg.len() && j < aseg.len() {
        let ps = pseg[i];
        if ps == "*" { // splat
            let rest = aseg[j..].join("/");
            params.insert("splat".into(), url_decode(&rest));
            return Some(params);
        } else if let Some(name) = ps.strip_prefix(':') { params.insert(name.to_string(), url_decode(aseg[j])); i+=1; j+=1; }
        else if ps == aseg[j] { i+=1; j+=1; } else { return None; }
    }
    if i == pseg.len() && j == aseg.len() { Some(params) } else if i+1==pseg.len() && pseg[i]=="*" { params.insert("splat".into(), String::new()); Some(params) } else { None }
}

fn path_match_builtin(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("PathMatch".into())), args } }
    let pat = to_string_arg(ev, args[0].clone());
    let path = to_string_arg(ev, args[1].clone());
    match path_match(&pat, &path) {
        Some(params) => Value::Assoc(HashMap::from([(String::from("ok"), Value::Boolean(true)), (String::from("params"), Value::Assoc(params.into_iter().map(|(k,v)| (k, Value::String(v))).collect()))])),
        None => Value::Assoc(HashMap::from([(String::from("ok"), Value::Boolean(false))])),
    }
}

fn respond_file(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("RespondFile".into())), args } }
    let path = to_string_arg(ev, args[0].clone());
    let mut headers: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
    let mut attach = false; let mut filename: Option<String> = None; let mut ctype: Option<String> = None; let mut status: i64 = 200;
    if let Some(Value::Assoc(m)) = args.get(1).and_then(|a| Some(ev.eval(a.clone()))).and_then(|v| if let Value::Assoc(m)=v { Some(Value::Assoc(m)) } else { None }) {
        if let Some(Value::Boolean(b)) = m.get("Attachment") { attach = *b; }
        if let Some(Value::String(s))|Some(Value::Symbol(s)) = m.get("Filename") { filename = Some(s.clone()); }
        if let Some(Value::String(s))|Some(Value::Symbol(s)) = m.get("ContentType") { ctype = Some(s.clone()); }
        if let Some(Value::Integer(s)) = m.get("Status") { status = *s; }
        if let Some(Value::Assoc(h)) = m.get("Headers") { for (k, v) in h { headers.insert(k.clone(), v.clone()); } }
    }
    // Content-Type default
    let ct_str = ctype.unwrap_or_else(|| "application/octet-stream".into());
    headers.entry("Content-Type".into()).or_insert(Value::String(ct_str));
    if attach || filename.is_some() {
        let fname = filename.unwrap_or_else(|| std::path::Path::new(&path).file_name().and_then(|s| s.to_str()).unwrap_or("file").to_string());
        let cd = format!("attachment; filename=\"{}\"", fname);
        headers.insert("Content-Disposition".into(), Value::String(cd));
    }
    Value::Assoc(vec![
        ("status".into(), Value::Integer(status)),
        ("headers".into(), Value::Assoc(headers)),
        ("bodyFile".into(), Value::String(path)),
    ].into_iter().collect())
}

fn respond_text(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("RespondText".into())), args } }
    let text = to_string_arg(ev, args[0].clone());
    let mut headers: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
    let mut status: i64 = 200;
    let mut ctype: Option<String> = None;
    if let Some(Value::Assoc(m)) = args.get(1).and_then(|a| Some(ev.eval(a.clone()))).and_then(|v| if let Value::Assoc(m)=v { Some(Value::Assoc(m)) } else { None }) {
        if let Some(Value::Integer(s)) = m.get("Status") { status = *s; }
        if let Some(Value::Assoc(h)) = m.get("Headers") { for (k, v) in h { headers.insert(k.clone(), v.clone()); } }
        if let Some(Value::String(s))|Some(Value::Symbol(s)) = m.get("ContentType") { ctype = Some(s.clone()); }
    }
    headers.entry("Content-Type".into()).or_insert(Value::String(ctype.unwrap_or_else(|| "text/plain; charset=utf-8".into())));
    Value::Assoc(vec![
        ("status".into(), Value::Integer(status)),
        ("headers".into(), Value::Assoc(headers)),
        ("body".into(), Value::String(text)),
    ].into_iter().collect())
}

fn respond_json(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("RespondJson".into())), args } }
    let val = ev.eval(args[0].clone());
    let mut headers: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
    let mut status: i64 = 200;
    let mut pretty = false;
    if let Some(Value::Assoc(m)) = args.get(1).and_then(|a| Some(ev.eval(a.clone()))).and_then(|v| if let Value::Assoc(m)=v { Some(Value::Assoc(m)) } else { None }) {
        if let Some(Value::Integer(s)) = m.get("Status") { status = *s; }
        if let Some(Value::Assoc(h)) = m.get("Headers") { for (k, v) in h { headers.insert(k.clone(), v.clone()); } }
        if let Some(Value::Boolean(b)) = m.get("Pretty") { pretty = *b; }
    }
    headers.entry("Content-Type".into()).or_insert(Value::String("application/json".into()));
    let j = value_to_json(&val);
    let body = if pretty { serde_json::to_string_pretty(&j).unwrap_or_else(|_| String::from("null")) } else { serde_json::to_string(&j).unwrap_or_else(|_| String::from("null")) };
    Value::Assoc(vec![
        ("status".into(), Value::Integer(status)),
        ("headers".into(), Value::Assoc(headers)),
        ("body".into(), Value::String(body)),
    ].into_iter().collect())
}

fn http_serve_routes(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // HttpServeRoutes[<|"GET /"->handler, ...|>, opts]
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("HttpServeRoutes".into())), args } }
    let routes_val = ev.eval(args[0].clone());
    let opts = server_opts_from(ev, args.get(1).cloned());
    let mut routes: Vec<(String, String, Value)> = Vec::new();
    if let Value::Assoc(m) = routes_val {
        for (k, v) in m { // k like "GET /path"
            let ks = k.trim();
            let (method, pat) = if let Some(space) = ks.find(' ') { (ks[..space].to_string(), ks[space+1..].to_string()) } else { ("GET".into(), ks.to_string()) };
            routes.push((method, pat, v));
        }
    } else { return failure("HTTP::routes", "Routes must be an association mapping 'METHOD path' -> handler"); }
    // Build a handler that dispatches
    let handler = Value::Expr { head: Box::new(Value::Symbol("RoutesHandler".into())), args: vec![] };
    // We will capture routes/env inside the server closure
    let stop = Arc::new(AtomicBool::new(false));
    let stop_cl = stop.clone();
    let host = opts.host.clone(); let port = opts.port; let read_to = opts.read_timeout_ms.map(|ms| std::time::Duration::from_millis(ms)); let write_to = opts.write_timeout_ms.map(|ms| std::time::Duration::from_millis(ms)); let max_body = opts.max_body_bytes;
    let listener = match std::net::TcpListener::bind((host.as_str(), port)) { Ok(l)=>l, Err(e)=> return failure("HTTP::listen", &format!("bind: {}", e)) };
    let addr_str = listener.local_addr().map(|a| a.to_string()).unwrap_or(format!("{}:{}", host, port));
    let addr_clone_for_stop = addr_str.clone();
    std::thread::spawn(move || {
        loop {
            match listener.accept() {
                Ok((mut stream, _peer)) => {
                    if stop_cl.load(Ordering::Relaxed) { let _ = stream.shutdown(std::net::Shutdown::Both); break; }
                    let routes2 = routes.clone();
                    std::thread::spawn(move || {
                        use std::io::{Read, Write};
                        let _ = read_to.map(|d| stream.set_read_timeout(Some(d)));
                        let _ = write_to.map(|d| stream.set_write_timeout(Some(d)));
                        let mut buf = Vec::new(); let mut tmp = [0u8; 1024]; let mut total = 0usize; let mut header = Vec::new(); let mut body = Vec::new(); let mut body_len_opt=None; let header_end;
                        loop { match stream.read(&mut tmp) { Ok(0)=>break, Ok(n)=>{ buf.extend_from_slice(&tmp[..n]); total+=n; if total>(max_body+16_384) { break; } }, Err(_)=>break } if let Some(p) = find_subsequence(&buf, b"\r\n\r\n") { header_end=p+4; header = buf[..header_end].to_vec(); if let Some(cl)=parse_content_length(&header) { body_len_opt=Some(cl); } break; } }
                        let mut headers_vec: Vec<(String, String)> = Vec::new();
                        let _ = parse_request_head(&header).map(|(_,_,hs)| { headers_vec = hs; });
                        if let Some(hend) = find_subsequence(&buf, b"\r\n\r\n") { body.extend_from_slice(&buf[hend..]); }
                        if parse_request_has_chunked(&headers_vec) {
                            match read_chunked(&mut stream, &body, max_body) { Ok(data)=> body = data, Err(_)=>{} }
                        } else if let Some(blen) = body_len_opt { while body.len()<blen { match stream.read(&mut tmp) { Ok(0)=>break, Ok(n)=>{ body.extend_from_slice(&tmp[..n]); if body.len()>max_body { break; } }, Err(_)=>break } } }
                        let (method, target, headers) = match parse_request_head(&header) { Some(t)=>t, None=>{ let _ = write_simple_resp(&mut stream, 400, "Bad Request", &[], b""); return; } };
                        let (path, query_map) = split_path_query(&target);
                        // Route match
                        let mut chosen: Option<(Value, HashMap<String, String>)> = None;
                        for (m, pat, hv) in routes2.iter() {
                            if m.eq_ignore_ascii_case(&method) { if let Some(params) = path_match(pat, &path) { chosen = Some((hv.clone(), params)); break; } }
                        }
                        let mut req_map: HashMap<String, Value> = HashMap::new();
                        req_map.insert("method".into(), Value::String(method));
                        req_map.insert("path".into(), Value::String(path));
                        req_map.insert("query".into(), Value::Assoc(query_map));
                        let mut hdr_assoc: HashMap<String, Value> = HashMap::new(); for (k,v) in headers { hdr_assoc.insert(k, Value::String(v)); }
                        req_map.insert("headers".into(), Value::Assoc(hdr_assoc));
                        req_map.insert("body".into(), Value::List(body.iter().map(|b| Value::Integer(*b as i64)).collect()));
                        if let Some((_hv, params)) = &chosen { req_map.insert("params".into(), Value::Assoc(params.iter().map(|(k,v)| (k.clone(), Value::String(v.clone()))).collect())); }
                        let req_val = Value::Assoc(req_map);
                        let out = if let Some((hv, _)) = chosen { let mut evh = lyra_runtime::Evaluator::new(); crate::register_all(&mut evh); let call = Value::Expr { head: Box::new(hv), args: vec![req_val] }; evh.eval(call) } else { Value::assoc(vec![("status", Value::Integer(404)), ("body", Value::String(String::from("Not Found")))]) };
                        let spec = render_response_from_value(out);
                        let status = spec.status;
                        let mut headers = spec.headers;
                        if let Some(ref body) = spec.body { headers.entry("Content-Length".into()).or_insert(body.len().to_string()); }
                        if let Some(file_path) = spec.body_file.clone() {
                            let mut hdrs: Vec<(String,String)> = headers.into_iter().collect();
                            let _ = write_chunked_resp(&mut stream, status, "OK", hdrs, Box::new(FileChunkIter::new(&file_path)));
                        } else if let Some(chunks) = spec.chunks.clone() {
                            let mut hdrs: Vec<(String,String)> = headers.into_iter().collect();
                            let _ = write_chunked_resp(&mut stream, status, "OK", hdrs, Box::new(chunks.into_iter()));
                        } else if let Some(cid) = spec.chan_id {
                            let mut hdrs: Vec<(String,String)> = headers.into_iter().collect();
                            let _ = write_chunked_resp(&mut stream, status, "OK", hdrs, Box::new(ChannelChunkIter::new(cid)));
                        } else if let Some(body) = spec.body.clone() {
                            let mut hdrs: Vec<(String,String)> = headers.into_iter().collect();
                            let mut has_conn=false; for (k,v) in &hdrs { if k.eq_ignore_ascii_case("Connection") { has_conn=true; } }
                            if !has_conn { hdrs.push(("Connection".into(), "close".into())); }
                            let _ = write_simple_resp(&mut stream, status, "OK", &hdrs, &body);
                        }
                        let _ = stream.flush(); let _ = stream.shutdown(std::net::Shutdown::Both);
                    });
                }
                Err(_e) => { if stop_cl.load(Ordering::Relaxed) { break; } std::thread::sleep(std::time::Duration::from_millis(10)); }
            }
        }
    });
    let id = next_srv_id();
    servers_reg().lock().unwrap().insert(id, ServerEntry { stop, addr: addr_clone_for_stop });
    Value::Expr { head: Box::new(Value::Symbol("ServerId".into())), args: vec![Value::Integer(id)] }
}


fn write_chunked_resp<W: std::io::Write>(mut stream: W, status: u16, reason: &str, mut headers: Vec<(String,String)>, mut chunks: Box<dyn Iterator<Item=Vec<u8>>>) -> std::io::Result<()> {
    use std::io::Write;
    let mut head = format!("HTTP/1.1 {} {}
", status, reason);
    let mut has_te = false; let mut has_conn = false;
    for (k,v) in &headers { if k.eq_ignore_ascii_case("Transfer-Encoding") { has_te = true; } if k.eq_ignore_ascii_case("Connection") { has_conn = true; } }
    if !has_te { headers.push(("Transfer-Encoding".into(), "chunked".into())); }
    if !has_conn { headers.push(("Connection".into(), "close".into())); }
    for (k,v) in headers { head.push_str(&format!("{}: {}
", k, v)); }
    head.push_str("
");
    stream.write_all(head.as_bytes())?;
    while let Some(chunk) = chunks.next() {
        if chunk.is_empty() { continue; }
        write!(stream, "{:X}
", chunk.len())?;
        stream.write_all(&chunk)?;
        stream.write_all(b"
")?;
    }
    stream.write_all(b"0

")?;
    Ok(())
}

struct FileChunkIter { file: std::fs::File, bufsize: usize }
impl FileChunkIter { fn new(path: &str) -> Self { let f = std::fs::File::open(path).unwrap(); Self { file: f, bufsize: 32*1024 } } }
impl Iterator for FileChunkIter { type Item = Vec<u8>; fn next(&mut self) -> Option<Self::Item> { use std::io::Read; let mut buf = vec![0u8; self.bufsize]; match self.file.read(&mut buf) { Ok(0)=>None, Ok(n)=>{ buf.truncate(n); Some(buf) }, Err(_)=>None } } }


#[cfg(feature = "net_https")]
fn write_chunked_resp_tls<W: std::io::Write>(mut stream: W, status: u16, reason: &str, mut headers: Vec<(String,String)>, mut chunks: Box<dyn Iterator<Item=Vec<u8>>>) -> std::io::Result<()> {
    use std::io::Write;
    let mut head = format!("HTTP/1.1 {} {}
", status, reason);
    let mut has_te = false; let mut has_conn = false;
    for (k,v) in &headers { if k.eq_ignore_ascii_case("Transfer-Encoding") { has_te = true; } if k.eq_ignore_ascii_case("Connection") { has_conn = true; } }
    if !has_te { headers.push(("Transfer-Encoding".into(), "chunked".into())); }
    if !has_conn { headers.push(("Connection".into(), "close".into())); }
    for (k,v) in headers { head.push_str(&format!("{}: {}
", k, v)); }
    head.push_str("
");
    stream.write_all(head.as_bytes())?;
    while let Some(chunk) = chunks.next() {
        if chunk.is_empty() { continue; }
        write!(stream, "{:X}
", chunk.len())?;
        stream.write_all(&chunk)?;
        stream.write_all(b"
")?;
    }
    stream.write_all(b"0

")?;
    Ok(())
}


struct ChannelChunkIter { ev: lyra_runtime::Evaluator, chan: Value }
impl ChannelChunkIter { fn new(id: i64) -> Self { let mut ev = lyra_runtime::Evaluator::new(); crate::register_all(&mut ev); let chan = Value::Expr { head: Box::new(Value::Symbol("ChannelId".into())), args: vec![Value::Integer(id)] }; Self { ev, chan } } }
impl Iterator for ChannelChunkIter { type Item = Vec<u8>; fn next(&mut self) -> Option<Self::Item> { let recv = Value::Expr { head: Box::new(Value::Symbol("Receive".into())), args: vec![self.chan.clone()] }; let v = self.ev.eval(recv); match v { Value::Symbol(s) if s=="Null" => None, Value::String(s)|Value::Symbol(s) => Some(s.into_bytes()), Value::List(bytes) => { let mut out = Vec::with_capacity(bytes.len()); for b in bytes { if let Value::Integer(i)=b { out.push(i as u8); } } Some(out) }, other => { let j = value_to_json(&other); serde_json::to_vec(&j).ok() } } } }


fn respond_bytes(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("RespondBytes".into())), args } }
    let data_v = ev.eval(args[0].clone());
    let mut bytes: Vec<u8> = Vec::new();
    match data_v {
        Value::List(xs) => { for b in xs { if let Value::Integer(i)=b { bytes.push((i as i64) as u8); } } }
        Value::String(s) | Value::Symbol(s) => { bytes = s.into_bytes(); }
        other => { let s = lyra_core::pretty::format_value(&other); bytes = s.into_bytes(); }
    }
    let mut headers: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
    let mut status: i64 = 200;
    let mut ctype: Option<String> = None;
    if let Some(Value::Assoc(m)) = args.get(1).and_then(|a| Some(ev.eval(a.clone()))).and_then(|v| if let Value::Assoc(m)=v { Some(Value::Assoc(m)) } else { None }) {
        if let Some(Value::Integer(s)) = m.get("Status") { status = *s; }
        if let Some(Value::Assoc(h)) = m.get("Headers") { for (k, v) in h { headers.insert(k.clone(), v.clone()); } }
        if let Some(Value::String(s))|Some(Value::Symbol(s)) = m.get("ContentType") { ctype = Some(s.clone()); }
    }
    headers.entry("Content-Type".into()).or_insert(Value::String(ctype.unwrap_or_else(|| "application/octet-stream".into())));
    Value::Assoc(vec![
        ("status".into(), Value::Integer(status)),
        ("headers".into(), Value::Assoc(headers)),
        ("body".into(), Value::List(bytes.into_iter().map(|b| Value::Integer(b as i64)).collect())),
    ].into_iter().collect())
}

fn respond_html(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("RespondHtml".into())), args } }
    let html = to_string_arg(ev, args[0].clone());
    let mut headers: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
    let mut status: i64 = 200;
    let mut ctype: Option<String> = None;
    if let Some(Value::Assoc(m)) = args.get(1).and_then(|a| Some(ev.eval(a.clone()))).and_then(|v| if let Value::Assoc(m)=v { Some(Value::Assoc(m)) } else { None }) {
        if let Some(Value::Integer(s)) = m.get("Status") { status = *s; }
        if let Some(Value::Assoc(h)) = m.get("Headers") { for (k, v) in h { headers.insert(k.clone(), v.clone()); } }
        if let Some(Value::String(s))|Some(Value::Symbol(s)) = m.get("ContentType") { ctype = Some(s.clone()); }
    }
    headers.entry("Content-Type".into()).or_insert(Value::String(ctype.unwrap_or_else(|| "text/html; charset=utf-8".into())));
    Value::Assoc(vec![
        ("status".into(), Value::Integer(status)),
        ("headers".into(), Value::Assoc(headers)),
        ("body".into(), Value::String(html)),
    ].into_iter().collect())
}

fn respond_redirect(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("RespondRedirect".into())), args } }
    let location = to_string_arg(ev, args[0].clone());
    let mut status: i64 = 302;
    let mut headers: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
    if let Some(Value::Assoc(m)) = args.get(1).and_then(|a| Some(ev.eval(a.clone()))).and_then(|v| if let Value::Assoc(m)=v { Some(Value::Assoc(m)) } else { None }) {
        if let Some(Value::Integer(s)) = m.get("Status") { status = *s; }
        if let Some(Value::Assoc(h)) = m.get("Headers") { for (k, v) in h { headers.insert(k.clone(), v.clone()); } }
    }
    headers.insert("Location".into(), Value::String(location));
    Value::Assoc(vec![
        ("status".into(), Value::Integer(status)),
        ("headers".into(), Value::Assoc(headers)),
    ].into_iter().collect())
}

fn respond_no_content(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let mut status: i64 = 204;
    let mut headers: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
    if let Some(Value::Assoc(m)) = args.get(0).and_then(|a| Some(ev.eval(a.clone()))).and_then(|v| if let Value::Assoc(m)=v { Some(Value::Assoc(m)) } else { None }) {
        if let Some(Value::Integer(s)) = m.get("Status") { status = *s; }
        if let Some(Value::Assoc(h)) = m.get("Headers") { for (k, v) in h { headers.insert(k.clone(), v.clone()); } }
    }
    Value::Assoc(vec![
        ("status".into(), Value::Integer(status)),
        ("headers".into(), Value::Assoc(headers)),
    ].into_iter().collect())
}

fn cookies_header_from_assoc(m: &std::collections::HashMap<String, Value>) -> String {
    let mut parts: Vec<String> = Vec::new();
    for (k, v) in m.iter() { let val = match v { Value::String(s)|Value::Symbol(s)=>s.clone(), Value::Integer(i)=>i.to_string(), Value::Real(f)=>f.to_string(), Value::Boolean(b)=>b.to_string(), other=>lyra_core::pretty::format_value(other) }; parts.push(format!("{}={}", k, val)); }
    parts.join("; ")
}


fn cookies_header(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("CookiesHeader".into())), args } }
    match ev.eval(args[0].clone()) {
        Value::Assoc(m) => Value::String(cookies_header_from_assoc(&m)),
        _ => Value::String(String::new())
    }
}

fn get_response_cookies(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("GetResponseCookies".into())), args } }
    let v = ev.eval(args[0].clone());
    let mut out: Vec<Value> = Vec::new();
    if let Value::Assoc(m) = v {
        if let Some(Value::List(items)) = m.get("headersList") {
            for it in items {
                if let Value::Assoc(h) = it {
                    let name_ok = h.get("name").and_then(|x| if let Value::String(s)|Value::Symbol(s)=x { Some(s.clone()) } else { None }).map(|s| s.to_ascii_lowercase()=="set-cookie").unwrap_or(false);
                    if name_ok { if let Some(Value::String(line))|Some(Value::Symbol(line)) = h.get("value") { out.push(parse_set_cookie_line(&line)); } }
                }
            }
        } else if let Some(Value::Assoc(hs)) = m.get("headers") {
            for (k, v) in hs { if k.eq_ignore_ascii_case("set-cookie") { if let Value::String(line)|Value::Symbol(line) = v { out.push(parse_set_cookie_line(&line)); } }
            }
        }
    }
    Value::List(out)
}

fn parse_set_cookie_line(line: &str) -> Value {
    let mut parts = line.split(';');
    let nv = parts.next().unwrap_or("");
    let (name, value) = if let Some((n,v)) = nv.split_once('=') { (n.trim().to_string(), v.trim().to_string()) } else { (nv.trim().to_string(), String::new()) };
    let mut attrs: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
    for a in parts { let a=a.trim(); if a.is_empty() { continue; } if let Some((k,v)) = a.split_once('=') { attrs.insert(k.trim().to_string(), Value::String(v.trim().to_string())); } else { attrs.insert(a.to_string(), Value::Boolean(true)); } }
    Value::Assoc(vec![ ("name".into(), Value::String(name)), ("value".into(), Value::String(value)), ("attrs".into(), Value::Assoc(attrs)) ].into_iter().collect())
}
