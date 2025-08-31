use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
use std::collections::HashMap;
use std::net::TcpStream;
use std::sync::{Mutex, OnceLock};

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

fn failure(tag: &str, msg: &str) -> Value {
    Value::Assoc(HashMap::from([
        ("message".into(), Value::String(msg.into())),
        ("tag".into(), Value::String(tag.into())),
    ]))
}

fn as_str(v: &Value) -> Option<String> { match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None } }

#[derive(Clone)]
struct SessInfo { host: String, port: u16, user: String }

static SESS_REG: OnceLock<Mutex<HashMap<i64, SessInfo>>> = OnceLock::new();
fn sess_reg() -> &'static Mutex<HashMap<i64, SessInfo>> { SESS_REG.get_or_init(|| Mutex::new(HashMap::new())) }
static NEXT_ID: OnceLock<std::sync::atomic::AtomicI64> = OnceLock::new();
fn next_id() -> i64 { let a = NEXT_ID.get_or_init(|| std::sync::atomic::AtomicI64::new(1)); a.fetch_add(1, std::sync::atomic::Ordering::Relaxed) }

fn sess_handle(id: i64) -> Value { Value::Assoc(HashMap::from([(String::from("__type"), Value::String(String::from("SshSession"))),(String::from("id"), Value::Integer(id))])) }
fn get_id(v: &Value) -> Option<i64> { if let Value::Assoc(m)=v { if matches!(m.get("__type"), Some(Value::String(s)) if s=="SshSession") { if let Some(Value::Integer(id)) = m.get("id") { return Some(*id); } } } None }

fn expand_tilde(path: &str) -> String {
    if path.starts_with("~/") {
        if let Some(home) = std::env::var_os("HOME") { return format!("{}/{}", home.to_string_lossy(), &path[2..]); }
    }
    path.into()
}

// In non-ssh builds, provide stubs returning Failure
#[cfg(not(feature = "ssh"))]
fn ssh_unavailable(_ev: &mut Evaluator, args: Vec<Value>, head: &str) -> Value {
    Value::Assoc(HashMap::from([
        (String::from("message"), Value::String(String::from("SSH operations require the 'ssh' feature"))),
        (String::from("tag"), Value::String(String::from("SSH::unavailable"))),
        (String::from("head"), Value::String(head.into())),
        (String::from("args"), Value::List(args)),
    ]))
}

#[cfg(not(feature = "ssh"))]
fn ssh_connect(ev: &mut Evaluator, args: Vec<Value>) -> Value { ssh_unavailable(ev, args, "SshConnect") }
#[cfg(not(feature = "ssh"))]
fn ssh_disconnect(ev: &mut Evaluator, args: Vec<Value>) -> Value { ssh_unavailable(ev, args, "SshDisconnect") }
#[cfg(not(feature = "ssh"))]
fn ssh_session_info(ev: &mut Evaluator, args: Vec<Value>) -> Value { ssh_unavailable(ev, args, "SshSessionInfo") }
#[cfg(not(feature = "ssh"))]
fn ssh_exec(ev: &mut Evaluator, args: Vec<Value>) -> Value { ssh_unavailable(ev, args, "SshExec") }
#[cfg(not(feature = "ssh"))]
fn ssh_upload(ev: &mut Evaluator, args: Vec<Value>) -> Value { ssh_unavailable(ev, args, "SshUpload") }
#[cfg(not(feature = "ssh"))]
fn ssh_download(ev: &mut Evaluator, args: Vec<Value>) -> Value { ssh_unavailable(ev, args, "SshDownload") }

#[cfg(feature = "ssh")]
fn ssh_connect(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let opts = match args.as_slice() { [v] => match ev.eval(v.clone()) { Value::Assoc(m)=>m, _=>HashMap::new() }, _=> HashMap::new() };
    let host = opts.get("host").or_else(|| opts.get("Host")).and_then(|v| as_str(v)).unwrap_or_else(|| "127.0.0.1".into());
    let port = opts.get("port").or_else(|| opts.get("Port")).and_then(|v| if let Value::Integer(n)=v { Some(*n as u16) } else { None }).unwrap_or(22);
    let user = opts.get("user").or_else(|| opts.get("User")).and_then(|v| as_str(v)).unwrap_or_else(|| whoami::username());
    let timeout_ms = opts.get("timeoutMs").and_then(|v| if let Value::Integer(n)=v { Some(*n as u32) } else { None }).unwrap_or(10000);
    let tcp = match TcpStream::connect((host.as_str(), port)) { Ok(s)=>s, Err(e)=> return failure("SSH::connect", &format!("tcp: {}", e)) };
    tcp.set_nodelay(true).ok();
    let mut sess = ssh2::Session::new().map_err(|e| e.to_string()).unwrap();
    sess.set_tcp_stream(tcp);
    sess.set_timeout(timeout_ms);
    if let Err(e) = sess.handshake() { return failure("SSH::handshake", &e.to_string()); }
    // Host key check
    let strict_opt = opts.get("strictHostKeyChecking").or_else(|| opts.get("StrictHostKeyChecking"));
    let strict = match strict_opt { Some(Value::Boolean(b)) => if *b { Some("yes") } else { Some("no") }, Some(Value::String(s))|Some(Value::Symbol(s)) => Some(s.as_str()), _ => None };
    if let Some(mode) = strict {
        if let Some((hostkey, _keytype)) = sess.host_key() {
            let mut kh = ssh2::KnownHosts::new().unwrap();
            let path = opts.get("knownHostsPath").or_else(|| opts.get("KnownHostsPath")).and_then(|v| as_str(v)).unwrap_or_else(|| expand_tilde("~/.ssh/known_hosts"));
            let _ = kh.read_file(std::path::Path::new(&path), ssh2::KnownHostFileKind::OpenSSH);
            match kh.check_port(&host, port, hostkey) {
                Ok(ssh2::CheckResult::Match) => { /* ok */ }
                Ok(ssh2::CheckResult::NotFound) => {
                    if mode.eq_ignore_ascii_case("accept-new") {
                        let key = base64::engine::general_purpose::STANDARD.encode(hostkey);
                        let _ = kh.add(&host, None, &host, key.as_bytes(), None);
                        let parent = std::path::Path::new(&path).parent().map(|p| p.to_path_buf()).unwrap_or(std::path::PathBuf::from("."));
                        let _ = std::fs::create_dir_all(&parent);
                        let _ = kh.write_file(std::path::Path::new(&path), ssh2::KnownHostFileKind::OpenSSH);
                    } else if mode.eq_ignore_ascii_case("no") {
                        // skip check
                    } else {
                        return failure("SSH::hostkey", &format!("unknown host; add to known_hosts or use strictHostKeyChecking->\"accept-new\""));
                    }
                }
                Ok(ssh2::CheckResult::Mismatch) => { return failure("SSH::hostkey", "host key mismatch"); }
                Err(e) => { return failure("SSH::hostkey", &format!("check: {}", e)); }
            }
        }
    }
    // Auth priority: agent, key, password
    if matches!(opts.get("agent"), Some(Value::Boolean(true))) {
        if let Ok(mut agent) = sess.agent() { if agent.connect().is_ok() && agent.list_identities().is_ok() { if let Ok(ids) = agent.identities() { let mut ok=false; for id in ids { if agent.userauth(user.as_str(), &id).is_ok() { ok=true; break; } } if !ok { return failure("SSH::auth", "agent auth failed"); } } else { return failure("SSH::auth", "agent list failed"); } } else { return failure("SSH::auth", "agent connect failed"); }
    } else if let Some(pk_path) = opts.get("privateKeyPath").and_then(|v| as_str(v)) {
        let passphrase = opts.get("passphrase").and_then(|v| as_str(v));
        if let Err(e) = sess.userauth_pubkey_file(user.as_str(), None, std::path::Path::new(&pk_path), passphrase.as_deref()) { return failure("SSH::auth", &e.to_string()); }
    } else if let Some(pw) = opts.get("password").and_then(|v| as_str(v)) {
        if let Err(e) = sess.userauth_password(user.as_str(), &pw) { return failure("SSH::auth", &e.to_string()); }
    } else {
        return failure("SSH::auth", "no auth method provided (password/privateKeyPath/agent)");
    }
    if !sess.authenticated() { return failure("SSH::auth", "authentication failed"); }
    // Host key info
    let mut algos = Vec::new();
    if let Ok(m) = sess.methods(ssh2::MethodType::HostKey) { algos.push(format!("hostkey:{}", m)); }
    let id = next_id();
    sess_reg().lock().unwrap().insert(id, SessInfo { host: host.clone(), port, user: user.clone() });
    // Leak session into static registry for now (minimal viable)
    SESS_STORE::put(id, sess);
    Value::Assoc(HashMap::from([
        ("id".into(), Value::Integer(id)),
        ("user".into(), Value::String(user)),
        ("host".into(), Value::String(host)),
        ("port".into(), Value::Integer(port as i64)),
        ("algorithms".into(), Value::List(algos.into_iter().map(Value::String).collect())),
        ("__type".into(), Value::String("SshSession".into())),
    ]))
}

#[cfg(feature = "ssh")]
fn ssh_disconnect(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("SshDisconnect".into())), args }; }
    let id = match get_id(&args[0]) { Some(id)=>id, None=> return Value::Expr { head: Box::new(Value::Symbol("SshDisconnect".into())), args } };
    SESS_STORE::remove(id);
    sess_reg().lock().unwrap().remove(&id);
    Value::Boolean(true)
}

#[cfg(feature = "ssh")]
fn ssh_session_info(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("SshSessionInfo".into())), args }; }
    let id = match get_id(&args[0]) { Some(id)=>id, None=> return Value::Expr { head: Box::new(Value::Symbol("SshSessionInfo".into())), args } };
    if let Some(info) = sess_reg().lock().unwrap().get(&id).cloned() {
        Value::Assoc(HashMap::from([
            ("id".into(), Value::Integer(id)),
            ("user".into(), Value::String(info.user)),
            ("host".into(), Value::String(info.host)),
            ("port".into(), Value::Integer(info.port as i64)),
        ]))
    } else { failure("SSH::session", "unknown id") }
}

#[cfg(feature = "ssh")]
fn ssh_exec(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("SshExec".into())), args }; }
    let id = match get_id(&args[0]) { Some(id)=>id, None=> return Value::Expr { head: Box::new(Value::Symbol("SshExec".into())), args } };
    let cmd = match ev.eval(args[1].clone()) { Value::String(s)|Value::Symbol(s)=>s, other=> lyra_core::pretty::format_value(&other) };
    let mut sess = match SESS_STORE::get(id) { Some(s)=>s, None=> return failure("SSH::session", "unknown id") };
    let mut channel = match sess.channel_session() { Ok(c)=>c, Err(e)=> return failure("SSH::exec", &e.to_string()) };
    if let Err(e) = channel.exec(&cmd) { return failure("SSH::exec", &e.to_string()) }
    let mut stdout = String::new();
    let mut stderr = String::new();
    use std::io::Read;
    if channel.read_to_string(&mut stdout).is_err() { /* ignore */ }
    if channel.stderr().read_to_string(&mut stderr).is_err() { /* ignore */ }
    let _ = channel.wait_close();
    let code = channel.exit_status().unwrap_or(0);
    Value::Assoc(HashMap::from([
        ("code".into(), Value::Integer(code as i64)),
        ("stdout".into(), Value::String(stdout)),
        ("stderr".into(), Value::String(stderr)),
    ]))
}

#[cfg(feature = "ssh")]
fn ssh_upload(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<3 { return Value::Expr { head: Box::new(Value::Symbol("SshUpload".into())), args }; }
    let id = match get_id(&args[0]) { Some(id)=>id, None=> return Value::Expr { head: Box::new(Value::Symbol("SshUpload".into())), args } };
    let source_path = match ev.eval(args[1].clone()) { Value::String(s)|Value::Symbol(s)=>s, _=> return failure("SSH::upload", "source must be path string (for now)") };
    let dest_path = match ev.eval(args[2].clone()) { Value::String(s)|Value::Symbol(s)=>s, _=> return failure("SSH::upload", "destPath must be string") };
    let data = match std::fs::read(&source_path) { Ok(b)=>b, Err(e)=> return failure("SSH::upload", &format!("read: {}", e)) };
    let mut sess = match SESS_STORE::get(id) { Some(s)=>s, None=> return failure("SSH::session", "unknown id") };
    let mut ch = match sess.scp_send(std::path::Path::new(&dest_path), 0o644, data.len() as u64, None) { Ok(c)=>c, Err(e)=> return failure("SSH::scp", &e.to_string()) };
    use std::io::Write; if let Err(e) = ch.write_all(&data) { return failure("SSH::scp", &e.to_string()) }
    let _ = ch.send_eof(); let _ = ch.wait_eof(); let _ = ch.wait_close();
    Value::Assoc(HashMap::from([(String::from("bytes"), Value::Integer(data.len() as i64)), (String::from("path"), Value::String(dest_path))]))
}

#[cfg(feature = "ssh")]
fn ssh_download(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("SshDownload".into())), args }; }
    let id = match get_id(&args[0]) { Some(id)=>id, None=> return Value::Expr { head: Box::new(Value::Symbol("SshDownload".into())), args } };
    let path = match ev.eval(args[1].clone()) { Value::String(s)|Value::Symbol(s)=>s, _=> return failure("SSH::download", "path must be string") };
    let mut sess = match SESS_STORE::get(id) { Some(s)=>s, None=> return failure("SSH::session", "unknown id") };
    let (mut ch, _stat) = match sess.scp_recv(std::path::Path::new(&path)) { Ok(t)=>t, Err(e)=> return failure("SSH::scp", &e.to_string()) };
    let mut buf = Vec::new(); use std::io::Read; if let Err(e) = ch.read_to_end(&mut buf) { return failure("SSH::scp", &e.to_string()) }
    let _ = ch.send_eof(); let _ = ch.wait_eof(); let _ = ch.wait_close();
    Value::Assoc(HashMap::from([(String::from("bytes"), Value::Integer(buf.len() as i64)), (String::from("data"), Value::List(buf.into_iter().map(|b| Value::Integer(b as i64)).collect()))]))
}

#[cfg(not(feature = "ssh"))]
fn ssh_host_key(ev: &mut Evaluator, args: Vec<Value>) -> Value { ssh_unavailable(ev, args, "SshHostKey") }

#[cfg(feature = "ssh")]
fn ssh_host_key(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("SshHostKey".into())), args }; }
    let id = match get_id(&args[0]) { Some(id)=>id, None=> return Value::Expr { head: Box::new(Value::Symbol("SshHostKey".into())), args } };
    let sess = match SESS_STORE::get(id) { Some(s)=>s, None=> return failure("SSH::session", "unknown id") };
    match sess.host_key() {
        Some((blob, t)) => {
            let key_type = match t { ssh2::HostKeyType::Rsa=>"ssh-rsa", ssh2::HostKeyType::Dss=>"ssh-dss", ssh2::HostKeyType::Unknown=>"unknown", ssh2::HostKeyType::Ecdsa=>"ecdsa", ssh2::HostKeyType::Ed25519=>"ssh-ed25519" };
            use sha2::{Digest,Sha256};
            let fp = base64::engine::general_purpose::STANDARD.encode(Sha256::digest(blob));
            let b64 = base64::engine::general_purpose::STANDARD.encode(blob);
            Value::Assoc(HashMap::from([
                ("keyType".into(), Value::String(key_type.into())),
                ("fingerprint".into(), Value::String(format!("SHA256:{}", fp))),
                ("base64".into(), Value::String(b64)),
            ]))
        }
        None => failure("SSH::hostkey", "unavailable"),
    }
}

#[cfg(not(feature = "ssh"))]
fn ssh_copy_id(ev: &mut Evaluator, args: Vec<Value>) -> Value { ssh_unavailable(ev, args, "SshCopyId") }

#[cfg(feature = "ssh")]
fn ssh_copy_id(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("SshCopyId".into())), args }; }
    let id = match get_id(&args[0]) { Some(id)=>id, None=> return Value::Expr { head: Box::new(Value::Symbol("SshCopyId".into())), args } };
    let pubkey = match ev.eval(args[1].clone()) { Value::String(s)|Value::Symbol(s)=>s, _=> return failure("SSH::copyid", "publicKeyOpenSsh must be string") };
    let mkdir = matches!(args.get(2).map(|v| ev.eval(v.clone())), Some(Value::Assoc(m)) if m.get("mkdir")==Some(&Value::Boolean(true)));
    let mut sess = match SESS_STORE::get(id) { Some(s)=>s, None=> return failure("SSH::session", "unknown id") };
    let shell = format!("{}; printf '%s\\n' '{}' >> ~/.ssh/authorized_keys; chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys",
        if mkdir {"mkdir -p ~/.ssh"} else {":"},
        pubkey.replace("'","'\\''")
    );
    let mut channel = match sess.channel_session() { Ok(c)=>c, Err(e)=> return failure("SSH::exec", &e.to_string()) };
    if let Err(e) = channel.exec(&shell) { return failure("SSH::exec", &e.to_string()) }
    let _ = channel.wait_close();
    let code = channel.exit_status().unwrap_or(0);
    Value::Assoc(HashMap::from([("code".into(), Value::Integer(code as i64))]))
}

// Simple process-wide store for ssh2::Session (feature-gated)
#[cfg(feature = "ssh")]
mod SESS_STORE {
    use super::*;
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static STORE: OnceLock<Mutex<HashMap<i64, ssh2::Session>>> = OnceLock::new();
    fn inner() -> &'static Mutex<HashMap<i64, ssh2::Session>> { STORE.get_or_init(|| Mutex::new(HashMap::new())) }
    pub fn put(id: i64, sess: ssh2::Session) { inner().lock().unwrap().insert(id, sess); }
    pub fn get(id: i64) -> Option<ssh2::Session> { inner().lock().unwrap().get(&id).cloned() }
    pub fn remove(id: i64) { inner().lock().unwrap().remove(&id); }
}

pub fn register_ssh(ev: &mut Evaluator) {
    ev.register("SshConnect", ssh_connect as NativeFn, Attributes::empty());
    ev.register("SshDisconnect", ssh_disconnect as NativeFn, Attributes::empty());
    ev.register("SshSessionInfo", ssh_session_info as NativeFn, Attributes::empty());
    ev.register("SshExec", ssh_exec as NativeFn, Attributes::empty());
    ev.register("SshUpload", ssh_upload as NativeFn, Attributes::empty());
    ev.register("SshDownload", ssh_download as NativeFn, Attributes::empty());
    ev.register("SshHostKey", ssh_host_key as NativeFn, Attributes::empty());
    ev.register("SshCopyId", ssh_copy_id as NativeFn, Attributes::empty());
    ev.register("SshKeyGen", ssh_key_gen as NativeFn, Attributes::empty());
}

#[cfg(not(feature = "ssh"))]
fn ssh_key_gen(ev: &mut Evaluator, args: Vec<Value>) -> Value { ssh_unavailable(ev, args, "SshKeyGen") }

#[cfg(feature = "ssh")]
fn ssh_key_gen(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let opts = match args.as_slice() { [v] => match ev.eval(v.clone()) { Value::Assoc(m)=>m, _=> HashMap::new() }, _=> HashMap::new() };
    let alg = opts.get("type").or_else(|| opts.get("alg")).and_then(|v| as_str(v)).unwrap_or_else(|| "ed25519".into()).to_ascii_lowercase();
    let comment = opts.get("comment").and_then(|v| as_str(v)).unwrap_or_else(|| "lyra".into());
    let passphrase = opts.get("passphrase").and_then(|v| as_str(v));
    match alg.as_str() {
        "ed25519" => {
            use ed25519_dalek::Signer;
            use rand_core::OsRng;
            let sk = ed25519_dalek::SigningKey::generate(&mut OsRng);
            let pk = sk.verifying_key();
            let pk_bytes = pk.to_bytes();
            // Build OpenSSH public blob
            let mut blob: Vec<u8> = Vec::new();
            fn put_str(buf: &mut Vec<u8>, s: &[u8]) { let len = s.len() as u32; buf.extend_from_slice(&len.to_be_bytes()); buf.extend_from_slice(s); }
            put_str(&mut blob, b"ssh-ed25519");
            put_str(&mut blob, &pk_bytes);
            let b64 = base64::engine::general_purpose::STANDARD.encode(&blob);
            let public_openssh = format!("ssh-ed25519 {} {}", b64, comment);
            use sha2::{Digest,Sha256};
            let fp = base64::engine::general_purpose::STANDARD.encode(Sha256::digest(&blob));
            let secret_raw = lyra_core::value::Value::String(crate::crypto::base64url_encode(&sk.to_bytes()));
            // Try to emit OpenSSH private key (encrypted if passphrase provided) when ssh_openssh feature is enabled
            #[cfg(feature = "ssh_openssh")]
            let private_pem = {
                use ssh_key::{Algorithm, LineEnding, PrivateKey as OsPrivateKey};
                use rand_core::OsRng as Rng2;
                let kp = OsPrivateKey::random(&mut Rng2, Algorithm::Ed25519).map_err(|e| e.to_string());
                match kp {
                    Ok(mut key) => {
                        if let Some(pw) = passphrase.clone() {
                            // Encrypt with OpenSSH-native format
                            match key.encrypt(pw) {
                                Ok(_) => key.to_openssh(LineEnding::LF).unwrap_or_default(),
                                Err(_) => String::new(),
                            }
                        } else {
                            key.to_openssh(LineEnding::LF).unwrap_or_default()
                        }
                    }
                    Err(_) => String::new(),
                }
            };
            #[cfg(not(feature = "ssh_openssh"))]
            let private_pem = String::new();
            let mut map = HashMap::from([
                ("type".into(), Value::String("ed25519".into())),
                ("publicKeyOpenSsh".into(), Value::String(public_openssh)),
                ("fingerprint".into(), Value::String(format!("SHA256:{}", fp))),
                ("secretKey".into(), secret_raw),
            ]);
            if !private_pem.is_empty() {
                map.insert("privateKeyPem".into(), Value::String(private_pem));
            } else {
                map.insert("privateKeyPem".into(), Value::Symbol("Null".into()));
                if passphrase.is_some() {
                    map.insert("warning".into(), Value::String("Enable feature ssh_openssh to produce encrypted OpenSSH private key".into()));
                }
            }
            Value::Assoc(map)
        }
        a if a.starts_with("rsa") => {
            // RSA support via ssh-key when ssh_openssh is enabled
            #[cfg(feature = "ssh_openssh")]
            {
                use rand_core::OsRng as Rng2;
                use ssh_key::{Algorithm, LineEnding, PrivateKey as OsPrivateKey};
                // Pick key size
                let bits_from_ty = if a == "rsa" { Some(2048) } else { a.strip_prefix("rsa").and_then(|s| s.parse::<u16>().ok()).map(|n| n as usize) };
                let bits = opts.get("bits").and_then(|v| if let Value::Integer(n)=v { Some(*n as usize) } else { None }).or(bits_from_ty).unwrap_or(2048);
                let alg = Algorithm::rsa(bits).map_err(|e| e.to_string());
                let pem = match alg {
                    Ok(alg) => {
                        let mut key = OsPrivateKey::random(&mut Rng2, alg).map_err(|e| e.to_string()).unwrap();
                        if let Some(pw) = passphrase.clone() { let _ = key.encrypt(pw); }
                        key.to_openssh(LineEnding::LF).unwrap_or_default()
                    }
                    Err(_) => String::new(),
                };
                // Derive public authorized_keys line from private PEM
                let public_openssh = {
                    // Parse back to get PublicKey
                    match ssh_key::PrivateKey::from_openssh(&pem) {
                        Ok(k) => {
                            let pubkey = k.public_key();
                            let mut out = Vec::new();
                            use core::fmt::Write as _;
                            let _ = write!(&mut out, "{} {} {}", pubkey.algorithm().name(), base64::engine::general_purpose::STANDARD.encode(pubkey.key_data().to_bytes()), comment);
                            String::from_utf8_lossy(&out).to_string()
                        }
                        Err(_) => String::new(),
                    }
                };
                // Fingerprint from authorized_keys blob
                let fp = if !public_openssh.is_empty() {
                    let parts: Vec<&str> = public_openssh.split_whitespace().collect();
                    if parts.len() >= 2 { use sha2::{Digest,Sha256}; let blob_b64 = parts[1]; if let Ok(blob) = base64::engine::general_purpose::STANDARD.decode(blob_b64) { let d = Sha256::digest(&blob); format!("SHA256:{}", base64::engine::general_purpose::STANDARD.encode(d)) } else { String::new() } } else { String::new() }
                } else { String::new() };
                let mut map = HashMap::from([
                    ("type".into(), Value::String(format!("rsa{}", bits))),
                    ("publicKeyOpenSsh".into(), Value::String(public_openssh)),
                    ("fingerprint".into(), Value::String(fp)),
                ]);
                if !pem.is_empty() { map.insert("privateKeyPem".into(), Value::String(pem)); }
                Value::Assoc(map)
            }
            #[cfg(not(feature = "ssh_openssh"))]
            {
                failure("SSH::keygen", "RSA key generation requires feature 'ssh_openssh'")
            }
        }
        other => failure("SSH::keygen", &format!("unsupported type: {} (supported: ed25519)", other)),
    }
}
