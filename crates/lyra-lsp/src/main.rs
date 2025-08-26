use anyhow::Result;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use tower_lsp::jsonrpc::Result as LspResult;
use tower_lsp::lsp_types::{self as lsp, CodeAction, CodeActionKind, CodeActionOrCommand, CodeActionParams, CodeActionResponse, Command, CompletionItem, CompletionItemKind, CompletionParams, CompletionResponse, Diagnostic, DiagnosticSeverity, DocumentChanges, DocumentSymbolParams, DocumentSymbolResponse, ExecuteCommandOptions, ExecuteCommandParams, GotoDefinitionParams, GotoDefinitionResponse, Hover, HoverContents, HoverParams, HoverProviderCapability, InitializeParams, InitializeResult, InitializedParams, Location, OneOf, OptionalVersionedTextDocumentIdentifier, ParameterInformation, Position, Range, ServerCapabilities, SignatureHelp, SignatureHelpOptions, SignatureInformation, SymbolInformation, SymbolKind, TextDocumentContentChangeEvent, TextDocumentEdit, TextDocumentSyncCapability, TextDocumentSyncKind, TextEdit, Url, WorkspaceSymbolParams, CreateFile, CreateFileOptions, ResourceOp, DocumentChangeOperation};
use tower_lsp::{Client, LanguageServer, LspService, Server};

use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use regex::Regex;
use std::path::{Path, PathBuf};
use url::Url as StdUrl;

#[derive(Clone)]
struct DocIndex { map: Arc<Mutex<HashMap<String, (String, Vec<String>)>>> } // name -> (summary, params)

struct Backend {
    client: Client,
    docs: Arc<Mutex<HashMap<Url, String>>>,
    evaluator: Arc<Mutex<Evaluator>>,
    doc_index: DocIndex,
    builtins: Arc<Vec<String>>,
    // per-document simple symbol index (name -> definition position)
    sym_index: Arc<Mutex<HashMap<Url, HashMap<String, Position>>>>,
    // Project root path -> Modules mapping (name -> abs path)
    projects: Arc<Mutex<HashMap<String, HashMap<String, String>>>>,
    // Project symbol index: project-root -> (symbol -> Location)
    project_sym_index: Arc<Mutex<HashMap<String, HashMap<String, Location>>>>,
    // Reverse mapping: file path -> project key
    file_to_project: Arc<Mutex<HashMap<String, String>>>,
    // Global default config and per-project overrides
    default_config: Arc<Mutex<ServerConfig>>,
    project_configs: Arc<Mutex<HashMap<String, ServerConfig>>>,
}

#[derive(Clone, Debug)]
struct ServerConfig { module_skeleton: String, organize_on_save: bool, organize_on_save_globs: Vec<String> }

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig { module_skeleton: String::from("Exported[{}];\n\n"), organize_on_save: false, organize_on_save_globs: Vec::new() }
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, _params: InitializeParams) -> LspResult<InitializeResult> {
        // Load global defaults from initialize options if provided
        // Accepts: { "moduleSkeleton": "...", "organizeOnSave": true, "organizeOnSaveGlobs": ["src/**/*.lyra"] }
        if let Some(opts) = _params.initialization_options {
            if let Some(skel) = opts.get("moduleSkeleton").and_then(|v| v.as_str()) {
                self.default_config.lock().module_skeleton = skel.to_string();
            }
            if let Some(b) = opts.get("organizeOnSave").and_then(|v| v.as_bool()) {
                self.default_config.lock().organize_on_save = b;
            }
            if let Some(arr) = opts.get("organizeOnSaveGlobs").and_then(|v| v.as_array()) {
                let mut pat = Vec::new();
                for x in arr { if let Some(s)=x.as_str() { pat.push(s.to_string()); } }
                self.default_config.lock().organize_on_save_globs = pat;
            }
        }
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(TextDocumentSyncKind::INCREMENTAL)),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                definition_provider: Some(OneOf::Left(true)),
                signature_help_provider: Some(SignatureHelpOptions { trigger_characters: Some(vec!["[".into(), ",".into()]), retrigger_characters: None, work_done_progress_options: Default::default() }),
                completion_provider: Some(lsp::CompletionOptions {
                    resolve_provider: Some(false),
                    trigger_characters: Some(vec!["[".into(), ".".into()]),
                    ..Default::default()
                }),
                document_symbol_provider: Some(OneOf::Left(true)),
                workspace_symbol_provider: Some(OneOf::Left(true)),
                code_action_provider: Some(lsp::CodeActionProviderCapability::Simple(true)),
                execute_command_provider: Some(ExecuteCommandOptions { commands: vec!["lyra.createModule".into(), "lyra.organizeImports".into(), "lyra.reloadProject".into()], work_done_progress_options: Default::default() }),
                ..Default::default()
            },
            server_info: Some(lsp::ServerInfo { name: "lyra-lsp".into(), version: Some(env!("CARGO_PKG_VERSION").into()) }),
        })
    }

    async fn initialized(&self, _params: InitializedParams) {
        let _ = self.client.log_message(lsp::MessageType::INFO, "lyra-lsp initialized").await;
    }

    async fn shutdown(&self) -> LspResult<()> { Ok(()) }

    async fn did_open(&self, params: lsp::DidOpenTextDocumentParams) {
        let uri = params.text_document.uri;
        self.docs.lock().insert(uri.clone(), params.text_document.text.clone());
        self.update_symbol_index(&uri);
        self.ensure_project_for_uri(&uri).await;
        self.publish_diagnostics(&uri).await;
    }

    async fn did_change(&self, params: lsp::DidChangeTextDocumentParams) {
        if let Some(text) = apply_changes(self.docs.lock().get(&params.text_document.uri).cloned(), &params.content_changes) {
            self.docs.lock().insert(params.text_document.uri.clone(), text);
        }
        self.update_symbol_index(&params.text_document.uri);
        self.refresh_project_index_for_uri(&params.text_document.uri);
        self.ensure_project_for_uri(&params.text_document.uri).await;
        self.publish_diagnostics(&params.text_document.uri).await;
    }

    async fn did_save(&self, params: lsp::DidSaveTextDocumentParams) {
        let uri = params.text_document.uri;
        self.update_symbol_index(&uri);
        self.refresh_project_index_for_uri(&uri);
        // Optionally organize imports on save based on config
        let should_org = {
            let mut flag = self.default_config.lock().organize_on_save;
            let mut globs: Vec<String> = self.default_config.lock().organize_on_save_globs.clone();
            if let Ok(path) = uri.to_file_path() {
                if let Some(root) = discover_project_root(&path) {
                    let key = root.to_string_lossy().to_string();
                    if let Some(cfg) = self.project_configs.lock().get(&key) { flag = cfg.organize_on_save; globs = cfg.organize_on_save_globs.clone(); }
                }
                // If globs present, require a match
                if !globs.is_empty() {
                    let pstr = path.to_string_lossy().to_string();
                    if !matches_any_glob(&globs, &pstr) { flag = false; }
                }
            }
            flag
        };
        if should_org {
            let text_opt = { self.docs.lock().get(&uri).cloned() };
            if let Some(text) = text_opt {
                if let Some(edits) = build_organize_imports_edits(&text, &uri) {
                    let we = lsp::WorkspaceEdit { changes: Some(std::iter::once((uri.clone(), edits)).collect()), document_changes: None, ..Default::default() };
                    let _ = self.client.apply_edit(we).await;
                }
            }
        }
    }

    async fn hover(&self, params: HoverParams) -> LspResult<Option<Hover>> {
        let (uri, pos) = (params.text_document_position_params.text_document.uri, params.text_document_position_params.position);
        let text = match self.docs.lock().get(&uri) { Some(t)=>t.clone(), None=> return Ok(None) };
        let (word, _range) = word_at(&text, pos);
        if word.is_empty() { return Ok(None); }
        if let Some((summary, params)) = self.doc_index.map.lock().get(&word).cloned() {
            let mut parts: Vec<String> = Vec::new();
            if !params.is_empty() { parts.push(format!("{}[{}]", word, params.join(", "))); }
            if !summary.is_empty() { parts.push(summary); }
            let md = parts.join("\n\n");
            return Ok(Some(Hover { contents: HoverContents::Markup(lsp::MarkupContent { kind: lsp::MarkupKind::Markdown, value: md }), range: None }));
        }
        Ok(None)
    }

    async fn completion(&self, params: CompletionParams) -> LspResult<Option<CompletionResponse>> {
        let (uri, pos) = (params.text_document_position.text_document.uri, params.text_document_position.position);
        let text = match self.docs.lock().get(&uri) { Some(t)=>t.clone(), None=> return Ok(None) };
        let (prefix, _range) = word_at(&text, pos);
        if prefix.is_empty() { return Ok(None); }
        let mut items: Vec<CompletionItem> = Vec::new();
        for name in self.builtins.iter() {
            if score(name, &prefix).is_some() {
                let mut it = CompletionItem::new_simple(name.clone(), String::new());
                it.kind = Some(CompletionItemKind::FUNCTION);
                if let Some((summary, params)) = self.doc_index.map.lock().get(name).cloned() {
                    it.detail = Some(if params.is_empty() { summary.clone() } else { format!("{}[{}] — {}", name, params.join(", "), summary) });
                    it.documentation = Some(lsp::Documentation::MarkupContent(lsp::MarkupContent { kind: lsp::MarkupKind::Markdown, value: summary }));
                    it.insert_text = Some(format!("{}[", name));
                }
                items.push(it);
            }
        }
        // Add project module names (if available)
        if let Some(root) = uri.to_file_path().ok().and_then(|p| discover_project_root(&p)) {
            if let Some(mods) = self.projects.lock().get(&root.to_string_lossy().to_string()) {
                for (mname, _path) in mods.iter() {
                    if score(mname, &prefix).is_some() {
                        let mut it = CompletionItem::new_simple(mname.clone(), String::from("(module)"));
                        it.kind = Some(CompletionItemKind::MODULE);
                        items.push(it);
                    }
                }
            }
        }
        items.sort_by(|a,b| a.label.cmp(&b.label));
        Ok(Some(CompletionResponse::Array(items)))
    }

    async fn goto_definition(&self, params: GotoDefinitionParams) -> LspResult<Option<GotoDefinitionResponse>> {
        let uri = params.text_document_position_params.text_document.uri;
        let pos = params.text_document_position_params.position;
        let text = match self.docs.lock().get(&uri) { Some(t)=>t.clone(), None=> return Ok(None) };
        let (word, _range) = word_at(&text, pos);
        if word.is_empty() { return Ok(None); }
        if let Some(map) = self.sym_index.lock().get(&uri) {
            if let Some(defpos) = map.get(&word) {
                let loc = Location { uri: uri.clone(), range: Range { start: *defpos, end: Position::new(defpos.line, defpos.character+word.len() as u32) } };
                return Ok(Some(GotoDefinitionResponse::Scalar(loc)));
            }
        }
        // Cross-file lookup: project symbol index
        if let Some(root) = uri.to_file_path().ok().and_then(|p| discover_project_root(&p)) {
            let key = root.to_string_lossy().to_string();
            // Module name go-to-file
            if let Some(mods) = self.projects.lock().get(&key) {
                if let Some(path) = mods.get(&word) {
                    if let Ok(u) = StdUrl::from_file_path(path) {
                        let loc = Location { uri: Url::from(u), range: Range { start: Position::new(0,0), end: Position::new(0,0) } };
                        return Ok(Some(GotoDefinitionResponse::Scalar(loc)));
                    }
                }
            }
            if let Some(pmap) = self.project_sym_index.lock().get(&key) {
                if let Some(loc) = pmap.get(&word) { return Ok(Some(GotoDefinitionResponse::Scalar(loc.clone()))); }
            }
        }
        Ok(None)
    }

    async fn signature_help(&self, params: lsp::SignatureHelpParams) -> LspResult<Option<SignatureHelp>> {
        let uri = params.text_document_position_params.text_document.uri;
        let pos = params.text_document_position_params.position;
        let text = match self.docs.lock().get(&uri) { Some(t)=>t.clone(), None=> return Ok(None) };
        if let Some((head, arg_idx)) = current_call_context_at(&text, pos) {
            if let Some((summary, params)) = self.doc_index.map.lock().get(&head).cloned() {
                let label = if params.is_empty() { format!("{}[]", head) } else { format!("{}[{}]", head, params.join(", ")) };
                let parameters: Vec<ParameterInformation> = params.iter().map(|p| ParameterInformation { label: lsp::ParameterLabel::Simple(p.clone()), documentation: None }).collect();
                let sig = SignatureInformation { label, documentation: Some(lsp::Documentation::MarkupContent(lsp::MarkupContent { kind: lsp::MarkupKind::Markdown, value: summary })), parameters: Some(parameters), active_parameter: None }; 
                let help = SignatureHelp { signatures: vec![sig], active_signature: Some(0), active_parameter: Some(arg_idx as u32) };
                return Ok(Some(help));
            }
        }
        Ok(None)
    }

    async fn document_symbol(&self, params: DocumentSymbolParams) -> LspResult<Option<DocumentSymbolResponse>> {
        let uri = params.text_document.uri;
        let text = match self.docs.lock().get(&uri) { Some(t)=>t.clone(), None=> return Ok(None) };
        let re = Regex::new(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?\s*(:=|=)").unwrap();
        let mut syms: Vec<SymbolInformation> = Vec::new();
        for (i, line) in text.lines().enumerate() {
            if let Some(caps) = re.captures(line) { if let Some(m) = caps.get(1) {
                let name = m.as_str().to_string();
                let start = Position::new(i as u32, m.start() as u32);
                let end = Position::new(i as u32, m.end() as u32);
                syms.push(SymbolInformation { name, kind: SymbolKind::FUNCTION, tags: None, deprecated: None, location: Location { uri: uri.clone(), range: Range { start, end } }, container_name: None });
            } }
        }
        Ok(Some(DocumentSymbolResponse::Flat(syms)))
    }

    async fn symbol(&self, params: WorkspaceSymbolParams) -> LspResult<Option<Vec<SymbolInformation>>> {
        let query = params.query;
        let mut out: Vec<SymbolInformation> = Vec::new();
        for (_proj, map) in self.project_sym_index.lock().iter() {
            for (name, loc) in map.iter() {
                if name.contains(&query) {
                    out.push(SymbolInformation { name: name.clone(), kind: SymbolKind::FUNCTION, tags: None, deprecated: None, location: loc.clone(), container_name: None });
                    if out.len() > 200 { break; }
                }
            }
            if out.len() > 200 { break; }
        }
        Ok(Some(out))
    }

    async fn code_action(&self, params: CodeActionParams) -> LspResult<Option<CodeActionResponse>> {
        let uri = params.text_document.uri;
        let range = params.range;
        let text = match self.docs.lock().get(&uri) { Some(t)=>t.clone(), None=> return Ok(None) };
        let mut sel = extract_word_in_range(&text, range);
        if sel.is_none() { sel = Some(word_at(&text, range.start).0); }
        let mut actions: Vec<CodeActionOrCommand> = Vec::new();
        if let Some(root) = uri.to_file_path().ok().and_then(|p| discover_project_root(&p)) {
            let key = root.to_string_lossy().to_string();
            if let Some(name) = sel.clone().filter(|s| !s.is_empty()) {
                if let Some(mods) = self.projects.lock().get(&key) {
                    if mods.contains_key(&name) {
                        // Insert Using if not present
                        if !text.contains(&format!("Using[\"{}\"", name)) {
                            let edit = lsp::WorkspaceEdit { changes: Some(std::iter::once((uri.clone(), vec![TextEdit { range: Range { start: Position::new(0,0), end: Position::new(0,0) }, new_text: format!("Using[\"{}\", <|Import->All|>];\n", name) }])).collect()), document_changes: None, ..Default::default() };
                            actions.push(CodeActionOrCommand::CodeAction(CodeAction { title: format!("Insert Using[\"{}\"]", name), kind: Some(CodeActionKind::QUICKFIX), diagnostics: None, edit: Some(edit), command: None, is_preferred: Some(true), disabled: None, data: None }));
                        }
                        // Create module file if missing
                        if let Some(path) = mods.get(&name) {
                            if !std::path::Path::new(path).exists() {
                                let cmd = Command::new(format!("Create module {}", name), "lyra.createModule".into(), Some(vec![serde_json::json!({"path": path, "name": name})]));
                                actions.push(CodeActionOrCommand::Command(cmd));
                            }
                        }
                    }
                }
            }
            // Organize Imports (source action): always present as command; attach edit if available
            let edits_opt = build_organize_imports_edits(&text, &uri);
            let ca = CodeAction {
                title: String::from("Organize Imports"),
                kind: Some(CodeActionKind::SOURCE_ORGANIZE_IMPORTS),
                diagnostics: None,
                edit: edits_opt.map(|edits| lsp::WorkspaceEdit { changes: Some(std::iter::once((uri.clone(), edits)).collect()), document_changes: None, ..Default::default() }),
                command: Some(Command::new(String::from("Organize Imports"), "lyra.organizeImports".into(), Some(vec![serde_json::json!({"uri": uri.to_string()})]))),
                is_preferred: Some(true),
                disabled: None,
                data: None,
            };
            actions.push(CodeActionOrCommand::CodeAction(ca));
        }
        Ok(Some(actions))
    }

    async fn execute_command(&self, params: ExecuteCommandParams) -> LspResult<Option<serde_json::Value>> {
        match params.command.as_str() {
            "lyra.createModule" => {
                // Expect arguments: first element object { path: string, name?: string }
                let args = params.arguments;
                if let Some(first) = args.into_iter().next() {
                    if let (Some(path), name_opt) = (first.get("path").and_then(|v| v.as_str()).map(|s| s.to_string()), first.get("name").and_then(|v| v.as_str()).map(|s| s.to_string())) {
                        let p = std::path::Path::new(&path);
                        if let Some(parent) = p.parent() { let _ = std::fs::create_dir_all(parent); }
                        // Determine project-specific skeleton
                        let skel = {
                            let mut applied = None;
                            if let Some(root) = discover_project_root(p) { 
                                let key = root.to_string_lossy().to_string();
                                if let Some(cfg) = self.project_configs.lock().get(&key) { applied = Some(cfg.module_skeleton.clone()); }
                            }
                            applied.unwrap_or_else(|| self.default_config.lock().module_skeleton.clone())
                        };
                        let nm = name_opt.unwrap_or_else(|| p.file_stem().and_then(|s| s.to_str()).unwrap_or("module").to_string());
                        let content = format!("(* New module: {} *)\n{}", nm, skel);
                        // Prepare WorkspaceEdit with CreateFile + initial content
                        let uri = match StdUrl::from_file_path(&p) { Ok(u) => Url::from(u), Err(_) => return Ok(Some(serde_json::json!({"ok": false, "error": "invalid path"}))) };
                        let create = DocumentChangeOperation::Op(ResourceOp::Create(CreateFile { uri: uri.clone(), options: Some(CreateFileOptions { overwrite: Some(false), ignore_if_exists: Some(true) }), annotation_id: None }));
                        let text_edit = TextEdit { range: Range { start: Position::new(0,0), end: Position::new(0,0) }, new_text: content };
                        let tde = TextDocumentEdit { text_document: OptionalVersionedTextDocumentIdentifier { uri: uri.clone(), version: None }, edits: vec![OneOf::Left(text_edit)] };
                        let op_edit = DocumentChangeOperation::Edit(tde);
                        let we = lsp::WorkspaceEdit { changes: None, document_changes: Some(DocumentChanges::Operations(vec![create, op_edit])), ..Default::default() };
                        let _ = self.client.apply_edit(we).await;
                        return Ok(Some(serde_json::json!({"ok": true, "path": path})));
                    }
                }
                Ok(Some(serde_json::json!({"ok": false, "error": "invalid arguments"})))
            }
            "lyra.reloadProject" => {
                // arguments: optional { path: string }
                let key_opt = params.arguments.into_iter().next().and_then(|v| v.get("path").and_then(|s| s.as_str()).map(|s| s.to_string()));
                if let Some(path) = key_opt {
                    let pb = std::path::Path::new(&path).to_path_buf();
                    if let Some(root) = discover_project_root(&pb) {
                        let fake_uri = match StdUrl::from_file_path(root.join("project.lyra")) { Ok(u) => Url::from(u), Err(_) => return Ok(Some(serde_json::json!({"ok": false, "error": "invalid path"}))) };
                        self.ensure_project_for_uri(&fake_uri).await;
                        return Ok(Some(serde_json::json!({"ok": true})));
                    }
                }
                Ok(Some(serde_json::json!({"ok": false, "error": "no project"})))
            }
            "lyra.organizeImports" => {
                // args: { uri: string }
                let uri = params.arguments.into_iter().next().and_then(|v| v.get("uri").and_then(|s| s.as_str()).and_then(|s| Url::parse(s).ok()));
                if let Some(u) = uri {
                    let text_opt = { self.docs.lock().get(&u).cloned() };
                    if let Some(text) = text_opt {
                        if let Some(edits) = build_organize_imports_edits(&text, &u) {
                            let we = lsp::WorkspaceEdit { changes: Some(std::iter::once((u.clone(), edits)).collect()), document_changes: None, ..Default::default() };
                            let _ = self.client.apply_edit(we).await;
                            return Ok(Some(serde_json::json!({"ok": true})));
                        }
                    }
                }
                Ok(Some(serde_json::json!({"ok": false, "error": "no edit"})))
            }
            _ => Ok(Some(serde_json::json!({"ok": false, "error": "unknown command"}))),
        }
    }
}

impl Backend {
    fn update_symbol_index(&self, uri: &Url) {
        if let Some(text) = self.docs.lock().get(uri).cloned() {
            let idx = index_defs(&text);
            self.sym_index.lock().insert(uri.clone(), idx);
        }
    }

    async fn publish_diagnostics(&self, uri: &Url) {
        let text = match self.docs.lock().get(uri) { Some(t)=>t.clone(), None=> return };
        let mut diags: Vec<Diagnostic> = Vec::new();
        let mut p = lyra_parser::Parser::from_source(&text);
        match p.parse_all() {
            Ok(_) => { /* no diagnostics */ }
            Err(e) => {
                let msg = format!("parse error: {:?}", e);
                diags.push(Diagnostic {
                    range: Range { start: Position::new(0,0), end: Position::new(0,1) },
                    severity: Some(DiagnosticSeverity::ERROR),
                    source: Some("lyra".into()),
                    message: msg,
                    ..Default::default()
                });
            }
        }
        let _ = self.client.publish_diagnostics(uri.clone(), diags, None).await;
    }

    async fn ensure_project_for_uri(&self, uri: &Url) {
        let Some(path) = uri.to_file_path().ok() else { return };
        let Some(root) = discover_project_root(&path) else { return };
        let key = root.to_string_lossy().to_string();
        if self.projects.lock().contains_key(&key) { return; }
        // Load via stdlib ProjectInfo["<root>"]
        let mut ev = self.evaluator.lock();
        let q = Value::Expr { head: Box::new(Value::Symbol("ProjectInfo".into())), args: vec![Value::String(key.clone())] };
        let val = ev.eval(q);
        // Also attempt to load project manifest for tool config overrides
        let cfg = {
            let mut out = None;
            let v = ev.eval(Value::Expr { head: Box::new(Value::Symbol("ProjectLoad".into())), args: vec![Value::String(key.clone())] });
            if let Value::Assoc(m) = v {
                // Accept Tool or tool
                let tool = m.get("Tool").or_else(|| m.get("tool"));
                if let Some(Value::Assoc(t)) = tool {
                    if let Some(Value::Assoc(ls)) = t.get("lyra.lsp").or_else(|| t.get("Lyra.LSP")) {
                        let mut cfg = ServerConfig::default();
                        if let Some(Value::String(s)) | Some(Value::Symbol(s)) = ls.get("moduleSkeleton").or_else(|| ls.get("ModuleSkeleton")) {
                            cfg.module_skeleton = s.clone();
                        }
                        if let Some(Value::Boolean(b)) = ls.get("organizeOnSave").or_else(|| ls.get("OrganizeOnSave")) { cfg.organize_on_save = *b; }
                        if let Some(Value::List(vs)) = ls.get("organizeOnSaveGlobs").or_else(|| ls.get("OrganizeOnSaveGlobs")) {
                            let mut pat: Vec<String> = Vec::new();
                            for v in vs { if let Value::String(s)=v { pat.push(s.clone()); } else if let Value::Symbol(s)=v { pat.push(s.clone()); } }
                            cfg.organize_on_save_globs = pat;
                        }
                        out = Some(cfg);
                    }
                }
            }
            out
        };
        drop(ev);
        let mut modules: HashMap<String, String> = HashMap::new();
        if let Value::Assoc(m) = val {
            if let Some(Value::Assoc(mods)) = m.get("modules") {
                for (k, v) in mods.iter() {
                    if let Value::String(p) | Value::Symbol(p) = v { modules.insert(k.clone(), p.clone()); }
                }
            }
        }
        // Index module files for cross-file definitions
        let mut sym_map: HashMap<String, Location> = HashMap::new();
        for (mname, path) in modules.iter() {
            // open file and index definitions
            if let Ok(src) = std::fs::read_to_string(path) {
                let defs = index_defs(&src);
                for (sym, pos) in defs {
                    if let Ok(u) = StdUrl::from_file_path(path) {
                        let loc = Location { uri: Url::from(u), range: Range { start: pos, end: Position::new(pos.line, pos.character + sym.len() as u32) } };
                        // later definitions can override earlier ones; that's fine for now
                        sym_map.insert(sym, loc);
                    }
                }
            }
            // reverse map file -> project key
            self.file_to_project.lock().insert(path.clone(), key.clone());
        }
        self.projects.lock().insert(key.clone(), modules);
        self.project_sym_index.lock().insert(key.clone(), sym_map);
        if let Some(c) = cfg { self.project_configs.lock().insert(key.clone(), c); }
    }

    fn refresh_project_index_for_uri(&self, uri: &Url) {
        if let Ok(path) = uri.to_file_path() {
            let pstr = path.to_string_lossy().to_string();
            if let Some(proj) = self.file_to_project.lock().get(&pstr).cloned() {
                if let Some(text) = self.docs.lock().get(uri).cloned() {
                    let defs = index_defs(&text);
                    let mut map_lock = self.project_sym_index.lock();
                    let entry = map_lock.entry(proj.clone()).or_insert_with(HashMap::new);
                    // remove any existing symbols pointing to this file
                    entry.retain(|_k, v| v.uri != *uri);
                    for (sym, pos) in defs {
                        let loc = Location { uri: uri.clone(), range: Range { start: pos, end: Position::new(pos.line, pos.character + sym.len() as u32) } };
                        entry.insert(sym, loc);
                    }
                }
            }
        }
    }
}

fn apply_changes(prev: Option<String>, changes: &[TextDocumentContentChangeEvent]) -> Option<String> {
    // For simplicity, handle full sync or last change if incremental; many clients send full text
    if changes.is_empty() { return prev; }
    if changes.len()==1 && changes[0].range.is_none() { return Some(changes[0].text.clone()); }
    // Fallback: replace with last text
    Some(changes.last().unwrap().text.clone())
}

fn index_defs(text: &str) -> HashMap<String, Position> {
    // Very simple: match lines like: Name[ ... ] := or Name := or Name =
    let re = Regex::new(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?\s*(:=|=)").unwrap();
    let mut out: HashMap<String, Position> = HashMap::new();
    for (i, line) in text.lines().enumerate() {
        if let Some(caps) = re.captures(line) {
            if let Some(m) = caps.get(1) {
                let name = m.as_str().to_string();
                let col = m.start() as u32;
                out.entry(name).or_insert(Position::new(i as u32, col));
            }
        }
    }
    out
}

fn discover_project_root(start: &Path) -> Option<PathBuf> {
    let mut cur = Some(start);
    while let Some(p) = cur {
        let cand = p.join("project.lyra");
        if cand.exists() { return Some(p.to_path_buf()); }
        cur = p.parent();
    }
    None
}

fn word_at(text: &str, pos: Position) -> (String, Range) {
    // Naive UTF-8/ASCII: split lines; treat letters, digits, underscore as part of symbol
    let line = text.lines().nth(pos.line as usize).unwrap_or("");
    let mut col = pos.character as usize;
    if col > line.len() { col = line.len(); }
    let bytes = line.as_bytes();
    let is_ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let mut start = col;
    while start>0 && is_ident(bytes[start-1]) { start -= 1; }
    let mut end = col;
    while end < bytes.len() && is_ident(bytes[end]) { end += 1; }
    let word = &line[start..end];
    (word.to_string(), Range { start: Position::new(pos.line, start as u32), end: Position::new(pos.line, end as u32) })
}

fn score(candidate: &str, pat: &str) -> Option<i64> {
    if candidate.starts_with(pat) { return Some(1000 - (candidate.len() as i64 - pat.len() as i64).max(0)); }
    if candidate.contains(pat) { return Some(100 - (candidate.len() as i64)); }
    None
}

fn current_call_context_at(text: &str, pos: Position) -> Option<(String, usize)> {
    // Find nearest head[ ... ] that includes pos. We'll scan backward from pos to find '[' and the head name, then compute argument index by counting commas at depth 0 until pos.
    let mut line_idx = pos.line as usize;
    let lines: Vec<&str> = text.lines().collect();
    if line_idx >= lines.len() { return None; }
    // Build a single string up to pos as a simpler approach
    let mut up_to: String = String::new();
    for i in 0..=line_idx {
        if i == line_idx {
            let l = lines[i];
            let col = pos.character as usize;
            let col = col.min(l.len());
            up_to.push_str(&l[..col]);
        } else {
            up_to.push_str(lines[i]);
        }
        if i < line_idx { up_to.push('\n'); }
    }
    // Walk backward to find the last '[' and then extract head identifier before it.
    let bytes = up_to.as_bytes();
    let mut i: isize = (bytes.len() as isize) - 1;
    let mut bracket_depth = 0i32;
    let mut target_lbracket: Option<usize> = None;
    while i >= 0 {
        let c = bytes[i as usize] as char;
        if c == '[' { if bracket_depth == 0 { target_lbracket = Some(i as usize); break; } else { bracket_depth -= 1; } }
        else if c == ']' { bracket_depth += 1; }
        i -= 1;
    }
    let lb = target_lbracket?;
    // Extract head identifier before '['
    let mut j: isize = lb as isize - 1;
    while j >= 0 && bytes[j as usize].is_ascii_whitespace() { j -= 1; }
    let end = (j + 1) as usize;
    while j >= 0 {
        let ch = bytes[j as usize] as char;
        if ch.is_ascii_alphanumeric() || ch == '_' { j -= 1; } else { break; }
    }
    let start = (j + 1) as usize;
    if end <= start { return None; }
    let head = up_to[start..end].to_string();
    if head.is_empty() { return None; }
    // Compute argument index from inside the brackets until pos: count commas at depth 0.
    let mut idx = 0usize;
    let mut depth = 0i32;
    // slice after '[' in the global text, but since we only have 'up_to', we count within the substring between lb+1..end_of_up_to
    for &b in &bytes[lb+1..] {
        let ch = b as char;
        if ch == '[' { depth += 1; }
        else if ch == ']' { if depth == 0 { break; } else { depth -= 1; } }
        else if ch == ',' && depth == 0 { idx += 1; }
    }
    Some((head, idx))
}

fn extract_word_in_range(text: &str, range: Range) -> Option<String> {
    if range.start.line != range.end.line { return None; }
    let line_idx = range.start.line as usize;
    let line = text.lines().nth(line_idx)?;
    let s = range.start.character as usize;
    let e = range.end.character as usize;
    if s>=e || e>line.len() { return None; }
    let w = &line[s..e];
    let ok = w.chars().all(|c| c.is_ascii_alphanumeric() || c=='_');
    if ok { Some(w.to_string()) } else { None }
}

fn build_organize_imports_edits(text: &str, _uri: &Url) -> Option<Vec<TextEdit>> {
    let re = Regex::new(r#"^\s*Using\[\s*"([^"]+)"[\s\S]*?\];\s*$"#).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    let mut first_code_line: usize = 0;
    while first_code_line < lines.len() && lines[first_code_line].trim().is_empty() { first_code_line += 1; }
    // Collect all Using occurrences and their line indices
    let mut mods: Vec<String> = Vec::new();
    let mut using_lines: Vec<usize> = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if let Some(cap) = re.captures(line) {
            if let Some(m) = cap.get(1) { mods.push(m.as_str().to_string()); using_lines.push(i); }
        }
    }
    if mods.is_empty() { return None; }
    // Unique + sorted
    use std::collections::BTreeSet;
    let set: BTreeSet<String> = mods.into_iter().collect();
    let uniq: Vec<String> = set.into_iter().collect();
    // Build standard block
    let new_block = uniq.into_iter().map(|m| format!("Using[\"{}\", <|Import->All|>];", m)).collect::<Vec<_>>().join("\n");
    // Prepare edits:
    // 1) Insert/replace at top (at first_code_line): ensure block + trailing newline
    let insert_edit = TextEdit { range: Range { start: Position::new(first_code_line as u32, 0), end: Position::new(first_code_line as u32, 0) }, new_text: format!("{}\n", new_block) };
    // 2) Remove all existing Using lines (including ones at top). Replace each line with empty string + newline removal: we will delete the line by replacing the line range.
    let mut remove_edits: Vec<TextEdit> = Vec::new();
    for &i in using_lines.iter().rev() {
        let start = Position::new(i as u32, 0);
        let end = Position::new((i as u32)+1, 0);
        remove_edits.push(TextEdit { range: Range { start, end }, new_text: String::new() });
    }
    let mut edits = remove_edits;
    edits.push(insert_edit);
    Some(edits)
}

fn matches_any_glob(patterns: &Vec<String>, path: &str) -> bool {
    for pat in patterns {
        // Convert simple glob to regex: * -> .*, ? -> .
        let mut regex = String::from("^");
        for ch in pat.chars() {
            match ch {
                '*' => regex.push_str(".*"),
                '?' => regex.push('.'),
                '.' => regex.push_str("\\."),
                '\\' => regex.push_str("\\\\"),
                '+' | '(' | ')' | '|' | '{' | '}' | '^' | '$' | '[' | ']' => { regex.push('\\'); regex.push(ch); }
                _ => regex.push(ch),
            }
        }
        regex.push('$');
        if Regex::new(&regex).map(|re| re.is_match(path)).unwrap_or(false) { return true; }
    }
    false
}

fn build_docs_and_builtins(ev: &mut Evaluator) -> (Arc<Vec<String>>, DocIndex) {
    // Discover builtins via DescribeBuiltins, then enrich via ToolsDescribe
    let mut names: Vec<String> = Vec::new();
    let resp = ev.eval(Value::Expr { head: Box::new(Value::Symbol("DescribeBuiltins".into())), args: vec![] });
    if let Value::List(items) = resp {
        for it in items {
            if let Value::Assoc(m) = it { if let Some(Value::String(n)) = m.get("name") { names.push(n.clone()); } }
        }
    }
    names.sort(); names.dedup();
    let mut map: HashMap<String, (String, Vec<String>)> = HashMap::new();
    for n in names.iter() {
        // ToolsDescribe returns a card with summary/params if available
        let q = Value::Expr { head: Box::new(Value::Symbol("ToolsDescribe".into())), args: vec![Value::String(n.clone())] };
        match ev.eval(q) {
            Value::Assoc(m) => {
                let summary = m.get("summary").and_then(|v| match v { Value::String(s)=>Some(s.clone()), _=>None }).unwrap_or_default();
                let params = m.get("params").and_then(|v| match v { Value::List(vs)=>Some(vs.iter().filter_map(|x| match x { Value::String(s)=>Some(s.clone()), _=>None }).collect::<Vec<_>>()), _=>None }).unwrap_or_default();
                map.insert(n.clone(), (summary, params));
            }
            _ => { map.insert(n.clone(), (String::new(), Vec::new())); }
        }
    }
    (Arc::new(names), DocIndex { map: Arc::new(Mutex::new(map)) })
}

#[tokio::main]
async fn main() -> Result<()> {
    // Build evaluator and docs once, then clone into backend per client
    let mut ev0 = Evaluator::new();
    lyra_stdlib::register_all(&mut ev0);
    let (builtins0, doc_index0) = build_docs_and_builtins(&mut ev0);

    let (service, socket) = LspService::new(move |client: Client| {
        // Fresh evaluator for the server instance
        let mut ev = Evaluator::new();
        lyra_stdlib::register_all(&mut ev);
        let backend = Backend {
            client,
            docs: Arc::new(Mutex::new(HashMap::new())),
            evaluator: Arc::new(Mutex::new(ev)),
            doc_index: DocIndex { map: Arc::new(Mutex::new(doc_index0.map.lock().clone())) },
            builtins: builtins0.clone(),
            sym_index: Arc::new(Mutex::new(HashMap::new())),
            projects: Arc::new(Mutex::new(HashMap::new())),
            project_sym_index: Arc::new(Mutex::new(HashMap::new())),
            file_to_project: Arc::new(Mutex::new(HashMap::new())),
            default_config: Arc::new(Mutex::new(ServerConfig::default())),
            project_configs: Arc::new(Mutex::new(HashMap::new())),
        };
        backend
    });
    Server::new(tokio::io::stdin(), tokio::io::stdout(), socket).serve(service).await;
    Ok(())
}
