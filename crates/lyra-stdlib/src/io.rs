use crate::register_if;
use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
use serde::Serialize;
use serde_json as sj;
use serde_json::ser::{PrettyFormatter, Serializer};
use std::io::Read;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Mutex, OnceLock};

#[cfg(feature = "tools")]
use crate::tool_spec;
#[cfg(feature = "tools")]
use crate::tools::add_specs;
#[cfg(feature = "tools")]
use crate::{schema_bool, schema_str};
#[cfg(feature = "tools")]
use std::collections::HashMap;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

// ---------------- UI backend trait + registry ----------------
trait UiBackend: Send + Sync {
    fn prompt_string(&self, msg: &str, opts: &std::collections::HashMap<String, Value>) -> Value;
    fn prompt_password(&self, msg: &str, opts: &std::collections::HashMap<String, Value>) -> Value;
    fn confirm(&self, msg: &str, opts: &std::collections::HashMap<String, Value>) -> Value;
    fn select_one(
        &self,
        msg: &str,
        items: &[(String, String)],
        opts: &std::collections::HashMap<String, Value>,
    ) -> Value;
    fn progress_create(&self, total: i64, opts: &std::collections::HashMap<String, Value>) -> Value;
    fn progress_advance(&self, id: i64, n: i64) -> Value;
    fn progress_finish(&self, id: i64) -> Value;
    fn render_table(
        &self,
        headers: &[String],
        rows: &[Vec<String>],
        opts: &std::collections::HashMap<String, Value>,
    ) -> Value;
    fn render_box(&self, text: &str, opts: &std::collections::HashMap<String, Value>) -> Value;
    fn select_many(
        &self,
        msg: &str,
        items: &[(String, String)],
        opts: &std::collections::HashMap<String, Value>,
    ) -> Value;
    fn notify(&self, msg: &str, opts: &std::collections::HashMap<String, Value>) -> Value;
    fn spinner_start(&self, msg: &str, opts: &std::collections::HashMap<String, Value>) -> Value;
    fn spinner_stop(&self, id: i64) -> Value;
}

struct TerminalBackend;
impl UiBackend for TerminalBackend {
    fn prompt_string(&self, msg: &str, opts: &std::collections::HashMap<String, Value>) -> Value {
        print!("{} ", msg);
        let _ = std::io::Write::flush(&mut std::io::stdout());
        let mut s = String::new();
        match std::io::stdin().read_line(&mut s) {
            Ok(_) => {
                let t = s.trim_end_matches(['\n', '\r']).to_string();
                if t.is_empty() {
                    if let Some(v) = opts.get("Default") { return v.clone(); }
                }
                Value::String(t)
            }
            Err(e) => failure("UI::prompt", &e.to_string()),
        }
    }
    fn prompt_password(&self, msg: &str, opts: &std::collections::HashMap<String, Value>) -> Value {
        // Plaintext fallback (no masking)
        print!("{} ", msg);
        let _ = std::io::Write::flush(&mut std::io::stdout());
        let mut s = String::new();
        match std::io::stdin().read_line(&mut s) {
            Ok(_) => {
                let t = s.trim_end_matches(['\n', '\r']).to_string();
                if t.is_empty() {
                    if let Some(v) = opts.get("Default") { return v.clone(); }
                }
                Value::String(t)
            }
            Err(e) => failure("UI::prompt", &e.to_string()),
        }
    }
    fn confirm(&self, msg: &str, opts: &std::collections::HashMap<String, Value>) -> Value {
        let default = matches!(opts.get("Default"), Some(Value::Boolean(true)));
        print!("{} {} ", msg, if default {"[Y/n]"} else {"[y/N]"});
        let _ = std::io::Write::flush(&mut std::io::stdout());
        let mut s = String::new();
        match std::io::stdin().read_line(&mut s) {
            Ok(_) => {
                let t = s.trim().to_ascii_lowercase();
                if t.is_empty() { return Value::Boolean(default); }
                Value::Boolean(t == "y" || t == "yes")
            }
            Err(e) => failure("UI::prompt", &e.to_string()),
        }
    }
    fn select_one(
        &self,
        msg: &str,
        items: &[(String, String)],
        _opts: &std::collections::HashMap<String, Value>,
    ) -> Value {
        println!("{}", msg);
        for (i, (name, _)) in items.iter().enumerate() {
            println!("  {}. {}", i + 1, name);
        }
        print!("Select (1-{}): ", items.len());
        let _ = std::io::Write::flush(&mut std::io::stdout());
        let mut s = String::new();
        let _ = std::io::stdin().read_line(&mut s);
        if let Ok(n) = s.trim().parse::<usize>() {
            if n >= 1 && n <= items.len() {
                return Value::String(items[n - 1].1.clone());
            }
        }
        Value::Symbol("Null".into())
    }
    fn progress_create(&self, total: i64, _opts: &std::collections::HashMap<String, Value>) -> Value {
        let id = pb_next();
        pb_reg().lock().unwrap().insert(id, PState { total, cur: 0 });
        Value::Integer(id)
    }
    fn progress_advance(&self, id: i64, n: i64) -> Value {
        if let Some(st) = pb_reg().lock().unwrap().get_mut(&id) {
            st.cur += n;
            Value::Boolean(true)
        } else {
            Value::Boolean(false)
        }
    }
    fn progress_finish(&self, id: i64) -> Value {
        pb_reg().lock().unwrap().remove(&id);
        Value::Boolean(true)
    }
    fn render_table(
        &self,
        headers: &[String],
        rows: &[Vec<String>],
        opts: &std::collections::HashMap<String, Value>,
    ) -> Value {
        Value::String(term_render_table(headers, rows, opts))
    }
    fn render_box(&self, text: &str, opts: &std::collections::HashMap<String, Value>) -> Value {
        Value::String(term_render_box(text, opts))
    }
    fn select_many(
        &self,
        msg: &str,
        items: &[(String, String)],
        _opts: &std::collections::HashMap<String, Value>,
    ) -> Value {
        println!("{}", msg);
        for (i, (name, _)) in items.iter().enumerate() {
            println!("  {}. {}", i + 1, name);
        }
        print!("Select (e.g., 1,3,4): ");
        let _ = std::io::Write::flush(&mut std::io::stdout());
        let mut s = String::new();
        let _ = std::io::stdin().read_line(&mut s);
        let mut out: Vec<Value> = Vec::new();
        for tok in s.split(|c: char| c == ',' || c.is_whitespace()) {
            if let Ok(n) = tok.trim().parse::<usize>() {
                if n >= 1 && n <= items.len() {
                    out.push(Value::String(items[n - 1].1.clone()));
                }
            }
        }
        Value::List(out)
    }
    fn notify(&self, msg: &str, opts: &std::collections::HashMap<String, Value>) -> Value {
        let level = opts.get("Level").and_then(|v| match v { Value::String(s) | Value::Symbol(s) => Some(s.to_ascii_lowercase()), _ => None });
        let mut theme = ui_term_theme().lock().unwrap().clone();
        // Per-call override
        if let Some(Value::String(s)) | Some(Value::Symbol(s)) = opts.get("AccentColor") { theme.accent = Some(s.clone()); }
        let mut col_name: Option<String> = None;
        if let Some(lv) = level.as_deref() {
            let pick = |s: &Option<String>| s.clone();
            col_name = match lv {
                "success" => pick(&theme.palette.success),
                "warning" => pick(&theme.palette.warning),
                "error" => pick(&theme.palette.error),
                "info" => pick(&theme.palette.info).or_else(|| theme.palette.primary.clone()),
                _ => None,
            };
        }
        if col_name.is_none() { col_name = theme.accent.clone().or_else(|| theme.palette.primary.clone()); }
        let (start, end) = term_accent_codes(col_name.as_deref());
        if let Some(lv) = level.as_deref() {
            let tag_owned = match lv { "success" => "SUCCESS".to_string(), "warning" => "WARNING".to_string(), "error" => "ERROR".to_string(), "info" => "INFO".to_string(), _ => lv.to_ascii_uppercase() };
            if !start.is_empty() { eprint!("{}", start); }
            eprint!("[{}]", tag_owned);
            if !end.is_empty() { eprint!("{}", end); }
            eprintln!(" {}", msg);
        } else {
            if !start.is_empty() { eprint!("{}", start); }
            eprint!("•");
            if !end.is_empty() { eprint!("{}", end); }
            eprintln!(" {}", msg);
        }
        Value::Boolean(true)
    }
    fn spinner_start(&self, msg: &str, _opts: &std::collections::HashMap<String, Value>) -> Value {
        spinner_start_thread(msg)
    }
    fn spinner_stop(&self, id: i64) -> Value { spinner_stop_thread(id) }
}

struct NullBackend;
impl UiBackend for NullBackend {
    fn prompt_string(&self, _msg: &str, opts: &std::collections::HashMap<String, Value>) -> Value {
        if let Some(v) = opts.get("Default") { return v.clone(); }
        Value::Symbol("Null".into())
    }
    fn prompt_password(&self, _msg: &str, opts: &std::collections::HashMap<String, Value>) -> Value {
        if let Some(v) = opts.get("Default") { return v.clone(); }
        Value::Symbol("Null".into())
    }
    fn confirm(&self, _msg: &str, opts: &std::collections::HashMap<String, Value>) -> Value {
        if let Some(Value::Boolean(b)) = opts.get("Default") { return Value::Boolean(*b); }
        Value::Boolean(false)
    }
    fn select_one(
        &self,
        _msg: &str,
        items: &[(String, String)],
        opts: &std::collections::HashMap<String, Value>,
    ) -> Value {
        if let Some(v) = opts.get("Default") { return v.clone(); }
        if let Some(Value::Integer(i)) = opts.get("DefaultIndex") {
            let idx = (*i as isize - 1).max(0) as usize;
            if idx < items.len() { return Value::String(items[idx].1.clone()); }
        }
        Value::Symbol("Null".into())
    }
    fn progress_create(&self, _total: i64, _opts: &std::collections::HashMap<String, Value>) -> Value {
        Value::Symbol("Null".into())
    }
    fn progress_advance(&self, _id: i64, _n: i64) -> Value { Value::Boolean(true) }
    fn progress_finish(&self, _id: i64) -> Value { Value::Boolean(true) }
    fn render_table(
        &self,
        _headers: &[String],
        _rows: &[Vec<String>],
        _opts: &std::collections::HashMap<String, Value>,
    ) -> Value { Value::String(String::new()) }
    fn render_box(&self, text: &str, _opts: &std::collections::HashMap<String, Value>) -> Value { Value::String(text.to_string()) }
    fn select_many(
        &self,
        _msg: &str,
        items: &[(String, String)],
        opts: &std::collections::HashMap<String, Value>,
    ) -> Value {
        if let Some(Value::List(vs)) = opts.get("Default") { return Value::List(vs.clone()); }
        if let Some(Value::List(vs)) = opts.get("DefaultIndex") {
            let mut out: Vec<Value> = Vec::new();
            for v in vs {
                if let Value::Integer(i) = v {
                    let idx = (*i as isize - 1).max(0) as usize;
                    if idx < items.len() { out.push(Value::String(items[idx].1.clone())); }
                }
            }
            return Value::List(out);
        }
        Value::Symbol("Null".into())
    }
    fn notify(&self, _msg: &str, _opts: &std::collections::HashMap<String, Value>) -> Value { Value::Boolean(true) }
    fn spinner_start(&self, _msg: &str, _opts: &std::collections::HashMap<String, Value>) -> Value { Value::Symbol("Null".into()) }
    fn spinner_stop(&self, _id: i64) -> Value { Value::Boolean(true) }
}

static UI_BACKEND: OnceLock<Mutex<Box<dyn UiBackend + Send + Sync>>> = OnceLock::new();
static UI_BACKEND_NAME: OnceLock<Mutex<String>> = OnceLock::new();
static UI_THEME_NAME: OnceLock<Mutex<String>> = OnceLock::new();
#[derive(Clone, Default)]
struct UiPalette { primary: Option<String>, success: Option<String>, warning: Option<String>, error: Option<String>, info: Option<String>, background: Option<String>, surface: Option<String>, text: Option<String> }
#[cfg(feature = "ui_egui")]
#[derive(Clone, Default)]
struct UiThemeOpts { accent: Option<String>, rounding: Option<f32>, font_size: Option<f32>, compact: bool, spacing_scale: Option<f32>, palette: UiPalette }
#[cfg(feature = "ui_egui")]
static UI_THEME_OPTS: OnceLock<Mutex<UiThemeOpts>> = OnceLock::new();

#[derive(Clone, Default)]
struct TermTheme { accent: Option<String>, compact: bool, palette: TermPalette }
#[derive(Clone, Default)]
#[allow(dead_code)]
struct TermPalette { primary: Option<String>, success: Option<String>, warning: Option<String>, error: Option<String>, info: Option<String>, background: Option<String>, text: Option<String> }
static UI_TERM_THEME: OnceLock<Mutex<TermTheme>> = OnceLock::new();

fn ui_backend() -> &'static Mutex<Box<dyn UiBackend + Send + Sync>> {
    UI_BACKEND.get_or_init(|| Mutex::new(Box::new(TerminalBackend)))
}
fn ui_backend_name() -> &'static Mutex<String> {
    UI_BACKEND_NAME.get_or_init(|| Mutex::new("terminal".to_string()))
}
fn ui_theme_name() -> &'static Mutex<String> {
    UI_THEME_NAME.get_or_init(|| Mutex::new("system".to_string()))
}
#[cfg(feature = "ui_egui")]
fn ui_theme_opts() -> &'static Mutex<UiThemeOpts> { UI_THEME_OPTS.get_or_init(|| Mutex::new(UiThemeOpts::default())) }
fn ui_term_theme() -> &'static Mutex<TermTheme> { UI_TERM_THEME.get_or_init(|| Mutex::new(TermTheme::default())) }

fn set_ui_backend(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("SetUiBackend".into())), args };
    }
    let mode_s = match ev.eval(args[0].clone()) {
        Value::String(s) | Value::Symbol(s) => s.to_ascii_lowercase(),
        other => return Value::Expr { head: Box::new(Value::Symbol("SetUiBackend".into())), args: vec![other] },
    };
    let (name, boxed): (String, Box<dyn UiBackend + Send + Sync>) = match mode_s.as_str() {
        "terminal" | "auto" => ("terminal".into(), Box::new(TerminalBackend)),
        "null" => ("null".into(), Box::new(NullBackend)),
        "gui" | "egui" => match mk_egui_backend() {
            Some(b) => ("gui".into(), b),
            None => ("terminal".into(), Box::new(TerminalBackend)),
        },
        _ => ("terminal".into(), Box::new(TerminalBackend)),
    };
    *ui_backend().lock().unwrap() = boxed;
    *ui_backend_name().lock().unwrap() = name;
    Value::Boolean(true)
}

fn get_ui_backend(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if !args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("GetUiBackend".into())), args };
    }
    Value::String(ui_backend_name().lock().unwrap().clone())
}

fn set_ui_theme(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() || args.len() > 2 {
        return Value::Expr { head: Box::new(Value::Symbol("SetUiTheme".into())), args };
    }
    let mode = match ev.eval(args[0].clone()) {
        Value::String(s) | Value::Symbol(s) => s.to_ascii_lowercase(),
        other => return Value::Expr { head: Box::new(Value::Symbol("SetUiTheme".into())), args: vec![other] },
    };
    let mode = match mode.as_str() { "system" | "light" | "dark" => mode, _ => "system".into() };
    *ui_theme_name().lock().unwrap() = mode;
    // Parse optional options
    let mut accent: Option<String> = None;
    let mut _rounding: Option<f32> = None;
    let mut _font_size: Option<f32> = None;
    let mut compact: bool = false;
    let mut _spacing_scale: Option<f32> = None;
    // Palette
    let mut pal = UiPalette::default();
    if args.len() == 2 {
        if let Value::Assoc(m) = ev.eval(args[1].clone()) {
            if let Some(Value::String(s)) | Some(Value::Symbol(s)) = m.get("AccentColor") { accent = Some(s.clone()); }
            if let Some(Value::Integer(n)) = m.get("Rounding") { _rounding = Some(*n as f32); }
            if let Some(Value::Integer(n)) = m.get("FontSize") { _font_size = Some(*n as f32); }
            if let Some(Value::Boolean(b)) = m.get("Compact") { compact = *b; }
            if let Some(Value::Integer(n)) = m.get("SpacingScale") { _spacing_scale = Some(*n as f32); }
            if let Some(Value::Assoc(pm)) = m.get("Palette") {
                let get = |k: &str| pm.get(k).and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None });
                pal.primary = get("Primary"); pal.success = get("Success"); pal.warning = get("Warning"); pal.error = get("Error"); pal.info = get("Info"); pal.background = get("Background"); pal.surface = get("Surface"); pal.text = get("Text");
                if accent.is_none() { accent = pal.primary.clone(); }
            }
        }
    }
    // Store opts
    #[cfg(feature = "ui_egui")]
    {
        let mut o = ui_theme_opts().lock().unwrap();
        o.accent = accent.clone();
        o.rounding = _rounding;
        o.font_size = _font_size;
        o.compact = compact;
        o.spacing_scale = _spacing_scale;
        o.palette = pal.clone();
    }
    {
        let mut t = ui_term_theme().lock().unwrap();
        t.accent = accent.or_else(|| pal.primary.clone());
        t.compact = compact;
        t.palette = TermPalette { primary: pal.primary.clone(), success: pal.success.clone(), warning: pal.warning.clone(), error: pal.error.clone(), info: pal.info.clone(), background: pal.background.clone(), text: pal.text.clone() };
    }
    Value::Boolean(true)
}

fn get_ui_theme(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if !args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("GetUiTheme".into())), args };
    }
    Value::String(ui_theme_name().lock().unwrap().clone())
}

#[cfg(feature = "ui_egui")]
fn mk_egui_backend() -> Option<Box<dyn UiBackend + Send + Sync>> {
    Some(Box::new(EguiBackend {}))
}
#[cfg(not(feature = "ui_egui"))]
fn mk_egui_backend() -> Option<Box<dyn UiBackend + Send + Sync>> { None }

#[cfg(feature = "ui_egui")]
struct EguiBackend {}
#[cfg(feature = "ui_egui")]
impl UiBackend for EguiBackend {
    fn prompt_string(&self, msg: &str, opts: &std::collections::HashMap<String, Value>) -> Value {
        if let Some(v) = egui_modal_prompt_string(msg, opts, false) { v } else { Value::Symbol("Null".into()) }
    }
    fn prompt_password(&self, msg: &str, opts: &std::collections::HashMap<String, Value>) -> Value {
        if let Some(v) = egui_modal_prompt_string(msg, opts, true) { v } else { Value::Symbol("Null".into()) }
    }
    fn confirm(&self, msg: &str, opts: &std::collections::HashMap<String, Value>) -> Value {
        Value::Boolean(egui_modal_confirm(msg, opts))
    }
    fn select_one(&self, msg: &str, items: &[(String, String)], opts: &std::collections::HashMap<String, Value>) -> Value {
        if let Some(v) = egui_modal_select_one(msg, items, opts) { v } else { Value::Symbol("Null".into()) }
    }
    fn progress_create(&self, total: i64, opts: &std::collections::HashMap<String, Value>) -> Value {
        let msg = opts.get("Message").and_then(|v| match v { Value::String(s) | Value::Symbol(s) => Some(s.clone()), _ => None }).unwrap_or_default();
        let accent = parse_theme_overrides(opts).accent.and_then(|s| egui_parse_color32(&s));
        let id = pb_next();
        egui_progress_send(ProgressCmd::Create { id, total, msg, accent });
        Value::Integer(id)
    }
    fn progress_advance(&self, id: i64, n: i64) -> Value {
        egui_progress_send(ProgressCmd::Advance { id, n });
        Value::Boolean(true)
    }
    fn progress_finish(&self, id: i64) -> Value {
        egui_progress_send(ProgressCmd::Finish { id });
        Value::Boolean(true)
    }
    fn render_table(&self, headers: &[String], rows: &[Vec<String>], opts: &std::collections::HashMap<String, Value>) -> Value {
        let overrides = parse_theme_overrides(opts);
        egui_show_table("Table", headers.to_vec(), rows.to_vec(), Some(overrides));
        Value::String(String::new())
    }
    fn render_box(&self, text: &str, opts: &std::collections::HashMap<String, Value>) -> Value {
        let overrides = parse_theme_overrides(opts);
        egui_show_box("Text", text.to_string(), Some(overrides));
        Value::String(String::new())
    }
    fn select_many(&self, msg: &str, items: &[(String, String)], opts: &std::collections::HashMap<String, Value>) -> Value {
        if let Some(vs) = egui_modal_select_many(msg, items, opts) { Value::List(vs) } else { Value::Symbol("Null".into()) }
    }
    fn notify(&self, msg: &str, opts: &std::collections::HashMap<String, Value>) -> Value { egui_show_notify(msg, opts) }
    fn spinner_start(&self, msg: &str, opts: &std::collections::HashMap<String, Value>) -> Value {
        let id = pb_next();
        let accent = parse_theme_overrides(opts).accent.and_then(|s| egui_parse_color32(&s));
        egui_progress_send(ProgressCmd::Create { id, total: -1, msg: msg.to_string(), accent });
        Value::Integer(id)
    }
    fn spinner_stop(&self, id: i64) -> Value { egui_progress_send(ProgressCmd::Finish { id }); Value::Boolean(true) }
}

#[cfg(feature = "ui_egui")]
#[derive(Clone, Default)]
struct ThemeOverrides { accent: Option<String>, rounding: Option<f32>, font_size: Option<f32>, compact: Option<bool>, spacing_scale: Option<f32> }

#[cfg(feature = "ui_egui")]
fn parse_theme_overrides(opts: &std::collections::HashMap<String, Value>) -> ThemeOverrides {
    let mut o = ThemeOverrides::default();
    if let Some(Value::String(s)) | Some(Value::Symbol(s)) = opts.get("AccentColor") { o.accent = Some(s.clone()); }
    if let Some(Value::Integer(n)) = opts.get("Rounding") { o.rounding = Some(*n as f32); }
    if let Some(Value::Integer(n)) = opts.get("FontSize") { o.font_size = Some(*n as f32); }
    if let Some(Value::Boolean(b)) = opts.get("Compact") { o.compact = Some(*b); }
    if let Some(Value::Integer(n)) = opts.get("SpacingScale") { o.spacing_scale = Some(*n as f32); }
    o
}

#[cfg(feature = "ui_egui")]
fn egui_apply_theme(cc: &eframe::CreationContext<'_>) {
    let name = ui_theme_name().lock().unwrap().clone();
    match name.as_str() {
        "light" => cc.egui_ctx.set_visuals(egui::Visuals::light()),
        "dark" => cc.egui_ctx.set_visuals(egui::Visuals::dark()),
        _ => { /* system: leave defaults */ }
    }
    // Extended theme options
    let opts = ui_theme_opts().lock().unwrap().clone();
    let mut style = (*cc.egui_ctx.style()).clone();
    // Accent color: selection + hyperlink
    if let Some(ac) = opts.accent.as_ref().or_else(|| opts.palette.primary.as_ref()) {
        if let Some(col) = egui_parse_color32(ac) {
            style.visuals.selection.bg_fill = col;
            style.visuals.hyperlink_color = col;
            // Buttons & interactive widgets
            style.visuals.widgets.active.bg_fill = col;
            style.visuals.widgets.hovered.bg_fill = egui_color_tint(col, 1.1);
            style.visuals.widgets.inactive.bg_stroke.color = col;
        }
    }
    // Background / Surface / Text
    if let Some(bg) = opts.palette.background.as_ref().and_then(|s| egui_parse_color32(s)) { style.visuals.window_fill = bg; style.visuals.extreme_bg_color = bg; }
    if let Some(sf) = opts.palette.surface.as_ref().and_then(|s| egui_parse_color32(s)) { style.visuals.panel_fill = sf; }
    if let Some(tx) = opts.palette.text.as_ref().and_then(|s| egui_parse_color32(s)) { style.visuals.override_text_color = Some(tx); }
    // Rounding
    if let Some(r) = opts.rounding {
        let rr = egui::Rounding::same(r);
        style.visuals.widgets.noninteractive.rounding = rr;
        style.visuals.widgets.inactive.rounding = rr;
        style.visuals.widgets.active.rounding = rr;
        style.visuals.widgets.hovered.rounding = rr;
        style.visuals.window_rounding = rr;
        style.visuals.menu_rounding = rr;
    }
    // Font size
    if let Some(sz) = opts.font_size {
        use egui::{FontId, TextStyle};
        for ts in [TextStyle::Body, TextStyle::Button, TextStyle::Small, TextStyle::Monospace, TextStyle::Heading] {
            style.text_styles.insert(ts.clone(), FontId::proportional(match ts { TextStyle::Heading => sz * 1.2, TextStyle::Small => sz * 0.9, _ => sz }));
        }
    }
    // Spacing / compact
    if opts.compact || opts.spacing_scale.is_some() {
        let scale = opts.spacing_scale.unwrap_or(if opts.compact { 0.8 } else { 1.0 });
        style.spacing.item_spacing *= scale;
        style.spacing.window_margin *= scale;
        style.spacing.button_padding *= scale;
        style.spacing.menu_margin *= scale;
    }
    cc.egui_ctx.set_style(style);
}

#[cfg(feature = "ui_egui")]
fn egui_apply_theme_with_overrides(cc: &eframe::CreationContext<'_>, overrides: Option<ThemeOverrides>) {
    egui_apply_theme(cc);
    if overrides.is_none() { return; }
    let ov = overrides.unwrap();
    let mut style = (*cc.egui_ctx.style()).clone();
    if let Some(ac) = ov.accent.as_ref() {
        if let Some(col) = egui_parse_color32(ac) {
            style.visuals.selection.bg_fill = col;
            style.visuals.hyperlink_color = col;
            style.visuals.widgets.active.bg_fill = col;
            style.visuals.widgets.hovered.bg_fill = egui_color_tint(col, 1.1);
            style.visuals.widgets.inactive.bg_stroke.color = col;
        }
    }
    if let Some(r) = ov.rounding { let rr = egui::Rounding::same(r); style.visuals.widgets.noninteractive.rounding = rr; style.visuals.widgets.inactive.rounding = rr; style.visuals.widgets.active.rounding = rr; style.visuals.widgets.hovered.rounding = rr; style.visuals.window_rounding = rr; style.visuals.menu_rounding = rr; }
    if let Some(sz) = ov.font_size {
        use egui::{FontId, TextStyle};
        for ts in [TextStyle::Body, TextStyle::Button, TextStyle::Small, TextStyle::Monospace, TextStyle::Heading] {
            style.text_styles.insert(ts.clone(), FontId::proportional(match ts { TextStyle::Heading => sz * 1.2, TextStyle::Small => sz * 0.9, _ => sz }));
        }
    }
    if ov.compact.unwrap_or(false) || ov.spacing_scale.is_some() {
        let scale = ov.spacing_scale.unwrap_or(if ov.compact.unwrap_or(false) { 0.8 } else { 1.0 });
        style.spacing.item_spacing *= scale; style.spacing.window_margin *= scale; style.spacing.button_padding *= scale; style.spacing.menu_margin *= scale;
    }
    cc.egui_ctx.set_style(style);
}

#[cfg(feature = "ui_egui")]
fn egui_theme_accent() -> Option<egui::Color32> {
    let opts = ui_theme_opts().lock().unwrap().clone();
    opts.accent.as_ref().and_then(|s| egui_parse_color32(s))
}

#[cfg(feature = "ui_egui")]
fn egui_color_tint(c: egui::Color32, f: f32) -> egui::Color32 {
    let [r,g,b,a] = c.to_array();
    let rf = ((r as f32 * f).round() as i32).clamp(0,255) as u8;
    let gf = ((g as f32 * f).round() as i32).clamp(0,255) as u8;
    let bf = ((b as f32 * f).round() as i32).clamp(0,255) as u8;
    egui::Color32::from_rgba_premultiplied(rf, gf, bf, a)
}

#[cfg(feature = "ui_egui")]
fn egui_modal_prompt_string(
    msg: &str,
    opts: &std::collections::HashMap<String, Value>,
    is_password: bool,
) -> Option<Value> {
    use std::sync::mpsc::{channel, Sender};
    let (tx, rx) = channel::<Value>();
    let default = opts.get("Default").and_then(|v| match v { Value::String(s) | Value::Symbol(s) => Some(s.clone()), _ => None }).unwrap_or_default();
    let title = if is_password { "Password" } else { "Prompt" };
    let title_owned = title.to_string();
    let msg_owned = msg.to_string();
    let default_owned = default.clone();
    let is_pw = is_password;
    let overrides = parse_theme_overrides(opts);
    let res = eframe::run_native(
        &title_owned,
        eframe::NativeOptions::default(),
        Box::new(move |cc| { egui_apply_theme_with_overrides(cc, Some(overrides)); Box::new(ModalApp::new_prompt(msg_owned, default_owned, is_pw, tx)) }),
    );
    match res { Ok(_) => rx.recv().ok(), Err(_) => None }
}

#[cfg(feature = "ui_egui")]
fn egui_modal_confirm(
    msg: &str,
    opts: &std::collections::HashMap<String, Value>,
) -> bool {
    use std::sync::mpsc::channel;
    let (tx, rx) = channel::<Value>();
    let default_yes = matches!(opts.get("Default"), Some(Value::Boolean(true)));
    let msg_owned = msg.to_string();
    let overrides = parse_theme_overrides(opts);
    let _ = eframe::run_native(
        "Confirm",
        eframe::NativeOptions::default(),
        Box::new(move |cc| { egui_apply_theme_with_overrides(cc, Some(overrides)); Box::new(ModalApp::new_confirm(msg_owned, default_yes, tx)) }),
    );
    rx.recv().ok().and_then(|v| if let Value::Boolean(b) = v { Some(b) } else { None }).unwrap_or(default_yes)
}

#[cfg(feature = "ui_egui")]
fn egui_modal_select_one(
    msg: &str,
    items: &[(String, String)],
    opts: &std::collections::HashMap<String, Value>,
) -> Option<Value> {
    use std::sync::mpsc::channel;
    let (tx, rx) = channel::<Value>();
    let title = "Select";
    let title_owned = title.to_string();
    let msg_owned = msg.to_string();
    let items_vec = items.to_vec();
    let overrides = parse_theme_overrides(opts);
    let res = eframe::run_native(
        &title_owned,
        eframe::NativeOptions::default(),
        Box::new(move |cc| { egui_apply_theme_with_overrides(cc, Some(overrides)); Box::new(ModalApp::new_select(msg_owned, items_vec, tx)) }),
    );
    match res { Ok(_) => rx.recv().ok(), Err(_) => None }
}

#[cfg(feature = "ui_egui")]
enum ModalKind {
    Prompt { is_password: bool },
    Confirm { default_yes: bool },
    Select,
    SelectMany,
}

#[cfg(feature = "ui_egui")]
struct ModalApp {
    kind: ModalKind,
    msg: String,
    input: String,
    items: Vec<(String, String)>,
    selected: usize,
    selected_flags: Vec<bool>,
    tx: std::sync::mpsc::Sender<Value>,
}

#[cfg(feature = "ui_egui")]
impl ModalApp {
    fn new_prompt(msg: String, default: String, is_password: bool, tx: std::sync::mpsc::Sender<Value>) -> Self {
        ModalApp { kind: ModalKind::Prompt { is_password }, msg, input: default, items: vec![], selected: 0, selected_flags: vec![], tx }
    }
    fn new_confirm(msg: String, default_yes: bool, tx: std::sync::mpsc::Sender<Value>) -> Self {
        ModalApp { kind: ModalKind::Confirm { default_yes }, msg, input: String::new(), items: vec![], selected: 0, selected_flags: vec![], tx }
    }
    fn new_select(msg: String, items: Vec<(String,String)>, tx: std::sync::mpsc::Sender<Value>) -> Self {
        ModalApp { kind: ModalKind::Select, msg, input: String::new(), items, selected: 0, selected_flags: vec![], tx }
    }
    fn new_select_many(msg: String, items: Vec<(String,String)>, defaults: Vec<bool>, tx: std::sync::mpsc::Sender<Value>) -> Self {
        let mut flags = defaults;
        if flags.len() != items.len() { flags = vec![false; items.len()]; }
        ModalApp { kind: ModalKind::SelectMany, msg, input: String::new(), items, selected: 0, selected_flags: flags, tx }
    }
}

#[cfg(feature = "ui_egui")]
impl eframe::App for ModalApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        use egui::{CentralPanel, Label, RichText, TextEdit};
        CentralPanel::default().show(ctx, |ui| {
            ui.heading(&self.msg);
            ui.separator();
            match self.kind {
                ModalKind::Prompt { is_password } => {
                    let te = TextEdit::singleline(&mut self.input);
                    if is_password { ui.add(te.password(true)); } else { ui.add(te); }
                    ui.horizontal(|ui| {
                        if ui.button("OK").clicked() {
                            let _ = self.tx.send(Value::String(self.input.clone()));
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                        if ui.button("Cancel").clicked() { ctx.send_viewport_cmd(egui::ViewportCommand::Close); }
                    });
                }
                ModalKind::Confirm { default_yes } => {
                    ui.horizontal(|ui| {
                        if ui.button("Yes").clicked() { let _ = self.tx.send(Value::Boolean(true)); ctx.send_viewport_cmd(egui::ViewportCommand::Close); }
                        if ui.button("No").clicked() { let _ = self.tx.send(Value::Boolean(false)); ctx.send_viewport_cmd(egui::ViewportCommand::Close); }
                    });
                    ui.label(RichText::new(if default_yes {"Default: Yes"} else {"Default: No"}).italics());
                }
                ModalKind::Select => {
                    egui::ScrollArea::vertical().max_height(300.0).show(ui, |ui| {
                        for (i, (name, _)) in self.items.iter().enumerate() {
                            if ui.selectable_label(self.selected == i, name).clicked() {
                                self.selected = i;
                            }
                        }
                    });
                    ui.horizontal(|ui| {
                        if ui.button("OK").clicked() {
                            if let Some((_n, v)) = self.items.get(self.selected) {
                                let _ = self.tx.send(Value::String(v.clone()));
                            }
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                        if ui.button("Cancel").clicked() { ctx.send_viewport_cmd(egui::ViewportCommand::Close); }
                    });
                }
                ModalKind::SelectMany => {
                    egui::ScrollArea::vertical().max_height(300.0).show(ui, |ui| {
                        for (i, (name, _)) in self.items.iter().enumerate() {
                            if i >= self.selected_flags.len() { self.selected_flags.resize(self.items.len(), false); }
                            ui.checkbox(&mut self.selected_flags[i], name);
                        }
                    });
                    ui.horizontal(|ui| {
                        if ui.button("OK").clicked() {
                            let mut out: Vec<Value> = Vec::new();
                            for (i, (_n, v)) in self.items.iter().enumerate() {
                                if self.selected_flags.get(i).copied().unwrap_or(false) {
                                    out.push(Value::String(v.clone()));
                                }
                            }
                            let _ = self.tx.send(Value::List(out));
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                        if ui.button("Cancel").clicked() { ctx.send_viewport_cmd(egui::ViewportCommand::Close); }
                    });
                }
            }
        });
    }
}

#[cfg(feature = "ui_egui")]
fn egui_modal_select_many(
    msg: &str,
    items: &[(String, String)],
    opts: &std::collections::HashMap<String, Value>,
) -> Option<Vec<Value>> {
    use std::sync::mpsc::channel;
    let (tx, rx) = channel::<Value>();
    // Build default flags from opts
    let mut flags: Vec<bool> = vec![false; items.len()];
    if let Some(Value::List(vals)) = opts.get("Default") {
        for (i, (_name, v)) in items.iter().enumerate() {
            if vals.iter().any(|x| matches!(x, Value::String(s) | Value::Symbol(s) if s == v)) {
                flags[i] = true;
            }
        }
    } else if let Some(Value::List(idxs)) = opts.get("DefaultIndex") {
        for idxv in idxs {
            if let Value::Integer(n) = idxv { let j = (*n as isize - 1).max(0) as usize; if j < flags.len() { flags[j] = true; } }
        }
    }
    let title_owned = "Select Many".to_string();
    let msg_owned = msg.to_string();
    let items_vec = items.to_vec();
    let overrides = parse_theme_overrides(opts);
    let res = eframe::run_native(
        &title_owned,
        eframe::NativeOptions::default(),
        Box::new(move |cc| { egui_apply_theme_with_overrides(cc, Some(overrides)); Box::new(ModalApp::new_select_many(msg_owned, items_vec, flags, tx)) }),
    );
    match res { Ok(_) => rx.recv().ok().and_then(|v| if let Value::List(vs) = v { Some(vs) } else { None }), Err(_) => None }
}

// ---------- Egui progress window ----------
#[cfg(feature = "ui_egui")]
enum ProgressCmd {
    Create { id: i64, total: i64, msg: String, accent: Option<egui::Color32> },
    Advance { id: i64, n: i64 },
    Finish { id: i64 },
}

#[cfg(feature = "ui_egui")]
static PROG_TX: OnceLock<Mutex<Option<std::sync::mpsc::Sender<ProgressCmd>>>> = OnceLock::new();

#[cfg(feature = "ui_egui")]
fn egui_progress_send(cmd: ProgressCmd) {
    use std::sync::mpsc::channel;
    let tx_opt = PROG_TX.get_or_init(|| Mutex::new(None));
    {
        let mut guard = tx_opt.lock().unwrap();
        if guard.is_none() {
            // Spawn progress app
            let (tx, rx) = channel::<ProgressCmd>();
            *guard = Some(tx.clone());
            std::thread::spawn(move || {
                let _ = eframe::run_native(
                    "Progress",
                    eframe::NativeOptions::default(),
                    Box::new(move |cc| { egui_apply_theme(cc); Box::new(ProgressApp::new(rx)) }),
                );
            });
        }
    }
    // Send after ensured
    if let Some(ref tx) = *tx_opt.lock().unwrap() { let _ = tx.send(cmd); }
}

#[cfg(feature = "ui_egui")]
struct ProgressState { total: i64, cur: i64, msg: String, accent: Option<egui::Color32> }

#[cfg(feature = "ui_egui")]
struct ProgressApp {
    rx: std::sync::mpsc::Receiver<ProgressCmd>,
    bars: std::collections::HashMap<i64, ProgressState>,
    tick: usize,
}

#[cfg(feature = "ui_egui")]
impl ProgressApp {
    fn new(rx: std::sync::mpsc::Receiver<ProgressCmd>) -> Self { Self { rx, bars: std::collections::HashMap::new(), tick: 0 } }
}

#[cfg(feature = "ui_egui")]
impl eframe::App for ProgressApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Drain channel
        while let Ok(cmd) = self.rx.try_recv() {
            match cmd {
                ProgressCmd::Create { id, total, msg, accent } => { self.bars.insert(id, ProgressState { total, cur: 0, msg, accent }); },
                ProgressCmd::Advance { id, n } => { if let Some(st) = self.bars.get_mut(&id) { st.cur += n; } },
                ProgressCmd::Finish { id } => { self.bars.remove(&id); },
            }
        }
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Progress");
            ui.separator();
            let mut ids: Vec<i64> = self.bars.keys().copied().collect();
            ids.sort();
            for id in ids {
                if let Some(st) = self.bars.get(&id) {
                    if st.total > 0 {
                        let frac = (st.cur as f32 / st.total as f32).clamp(0.0, 1.0);
                        let label = if !st.msg.is_empty() { format!("#{id} {} ({}/{})", st.msg, st.cur, st.total) } else { format!("#{id} ({}/{})", st.cur, st.total) };
                        let mut pb = egui::widgets::ProgressBar::new(frac).text(label);
                        if let Some(ac) = st.accent.or_else(|| egui_theme_accent()) { pb = pb.fill(ac); }
                        ui.add(pb);
                    } else {
                        let frames = ["|","/","-","\\"];
                        let f = frames[self.tick % frames.len()];
                        let label = if !st.msg.is_empty() { format!("#{id} {} {}", f, st.msg) } else { format!("#{id} {}", f) };
                        if let Some(ac) = st.accent.or_else(|| egui_theme_accent()) { ui.label(egui::RichText::new(label).color(ac)); } else { ui.label(label); }
                    }
                }
            }
            if self.bars.is_empty() {
                ui.label("No active tasks.");
            }
        });
        self.tick = self.tick.wrapping_add(1);
        ctx.request_repaint_after(std::time::Duration::from_millis(100));
    }
}

// ---------- Egui helpers for notify/table/box ----------
#[cfg(feature = "ui_egui")]
fn egui_show_notify(msg: &str, opts: &std::collections::HashMap<String, Value>) -> Value {
    let title = opts.get("Title").and_then(|v| match v { Value::String(s) | Value::Symbol(s) => Some(s.clone()), _ => None }).unwrap_or_else(|| "Notification".into());
    let timeout_ms = opts
        .get("TimeoutMs").or_else(|| opts.get("timeoutMs"))
        .and_then(|v| if let Value::Integer(n) = v { Some(*n as u64) } else { None })
        .unwrap_or(2000);
    let close_on_click = opts.get("CloseOnClick").and_then(|v| if let Value::Boolean(b) = v { Some(*b) } else { None }).unwrap_or(true);
    let close_on_esc = opts.get("CloseOnEsc").and_then(|v| if let Value::Boolean(b) = v { Some(*b) } else { None }).unwrap_or(true);
    let show_dismiss = opts.get("ShowDismiss").and_then(|v| if let Value::Boolean(b) = v { Some(*b) } else { None }).unwrap_or(true);
    let msg_owned = msg.to_string();
    // Determine accent from Level if not explicitly provided
    let mut overrides = parse_theme_overrides(opts);
    if overrides.accent.is_none() {
        let level = opts.get("Level").and_then(|v| match v { Value::String(s) | Value::Symbol(s) => Some(s.to_ascii_lowercase()), _ => None });
        if let Some(lv) = level.as_deref() {
            let pal = ui_theme_opts().lock().unwrap().palette.clone();
            overrides.accent = match lv {
                "success" => pal.success,
                "warning" => pal.warning,
                "error" => pal.error,
                "info" => pal.info.or(pal.primary),
                _ => pal.primary,
            };
        }
    }
    let level = opts.get("Level").and_then(|v| match v { Value::String(s) | Value::Symbol(s) => Some(s.to_ascii_lowercase()), _ => None });
    std::thread::spawn(move || {
        let _ = eframe::run_native(
            &title,
            eframe::NativeOptions::default(),
            Box::new(move |cc| { egui_apply_theme_with_overrides(cc, Some(overrides)); Box::new(NotifyApp::new(msg_owned.clone(), timeout_ms).with_level(level).with_behavior(close_on_click, close_on_esc, show_dismiss)) }),
        );
    });
    Value::Boolean(true)
}

#[cfg(feature = "ui_egui")]
struct NotifyApp { msg: String, left: u64, start: std::time::Instant, level: Option<String>, close_on_click: bool, close_on_esc: bool, show_dismiss: bool }
#[cfg(feature = "ui_egui")]
impl NotifyApp {
    fn new(msg: String, timeout_ms: u64) -> Self { Self { msg, left: timeout_ms, start: std::time::Instant::now(), level: None, close_on_click: true, close_on_esc: true, show_dismiss: true } }
    fn with_level(mut self, level: Option<String>) -> Self { self.level = level; self }
    fn with_behavior(mut self, close_on_click: bool, close_on_esc: bool, show_dismiss: bool) -> Self { self.close_on_click = close_on_click; self.close_on_esc = close_on_esc; self.show_dismiss = show_dismiss; self }
}
#[cfg(feature = "ui_egui")]
impl eframe::App for NotifyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let icon = match self.level.as_deref() { Some("success") => "✓", Some("warning") => "⚠", Some("error") => "✕", _ => "ℹ" };
            let color = match self.level.as_deref() {
                Some("success") => ui_theme_opts().lock().unwrap().palette.success.clone().and_then(|s| egui_parse_color32(&s)),
                Some("warning") => ui_theme_opts().lock().unwrap().palette.warning.clone().and_then(|s| egui_parse_color32(&s)),
                Some("error") => ui_theme_opts().lock().unwrap().palette.error.clone().and_then(|s| egui_parse_color32(&s)),
                Some("info") => ui_theme_opts().lock().unwrap().palette.info.clone().and_then(|s| egui_parse_color32(&s)).or_else(|| ui_theme_opts().lock().unwrap().palette.primary.clone().and_then(|s| egui_parse_color32(&s))),
                _ => ui_theme_opts().lock().unwrap().palette.primary.clone().and_then(|s| egui_parse_color32(&s)),
            };
            ui.horizontal(|ui| {
                if let Some(c) = color { ui.label(egui::RichText::new(icon).color(c).strong()); } else { ui.label(icon); }
                if let Some(c) = color { ui.label(egui::RichText::new(&self.msg).color(c)); } else { ui.label(&self.msg); }
            });
            if self.show_dismiss && ui.button("Dismiss").clicked() { ctx.send_viewport_cmd(egui::ViewportCommand::Close); }
        });
        if self.close_on_click {
            let clicked = ctx.input(|i| i.pointer.any_click());
            if clicked { ctx.send_viewport_cmd(egui::ViewportCommand::Close); }
        }
        if self.close_on_esc {
            let esc = ctx.input(|i| i.key_pressed(egui::Key::Escape));
            if esc { ctx.send_viewport_cmd(egui::ViewportCommand::Close); }
        }
        let elapsed = self.start.elapsed();
        if elapsed.as_millis() as u64 >= self.left { ctx.send_viewport_cmd(egui::ViewportCommand::Close); }
        ctx.request_repaint_after(std::time::Duration::from_millis(100));
    }
}

#[cfg(feature = "ui_egui")]
fn egui_show_table(title: &str, headers: Vec<String>, rows: Vec<Vec<String>>, overrides: Option<ThemeOverrides>) {
    let title_owned = title.to_string();
    std::thread::spawn(move || {
        let _ = eframe::run_native(
            &title_owned,
            eframe::NativeOptions::default(),
            Box::new(move |cc| { egui_apply_theme_with_overrides(cc, overrides.clone()); Box::new(TableApp::new(headers, rows)) }),
        );
    });
}

#[cfg(feature = "ui_egui")]
struct TableApp { headers: Vec<String>, rows: Vec<Vec<String>> }
#[cfg(feature = "ui_egui")]
impl TableApp { fn new(headers: Vec<String>, rows: Vec<Vec<String>>) -> Self { Self { headers, rows } } }
#[cfg(feature = "ui_egui")]
impl eframe::App for TableApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| {
                egui::Grid::new("table_grid").striped(true).show(ui, |ui| {
                    for h in &self.headers { ui.label(egui::RichText::new(h).strong()); }
                    ui.end_row();
                    for row in &self.rows {
                        for (i, cell) in row.iter().enumerate() {
                            let text = if i < self.headers.len() { cell } else { cell };
                            ui.label(text);
                        }
                        ui.end_row();
                    }
                });
            });
        });
    }
}

#[cfg(feature = "ui_egui")]
fn egui_show_box(title: &str, text: String, overrides: Option<ThemeOverrides>) {
    let title_owned = title.to_string();
    std::thread::spawn(move || {
        let t = text.clone();
        let _ = eframe::run_native(
            &title_owned,
            eframe::NativeOptions::default(),
            Box::new(move |cc| { egui_apply_theme_with_overrides(cc, overrides.clone()); Box::new(TextApp::new(t.clone())) }),
        );
    });
}
#[cfg(feature = "ui_egui")]
struct TextApp { text: String }
#[cfg(feature = "ui_egui")]
impl TextApp { fn new(text: String) -> Self { Self { text } } }
#[cfg(feature = "ui_egui")]
impl eframe::App for TextApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let mut te = egui::TextEdit::multiline(&mut self.text);
            te = te.desired_width(f32::INFINITY);
            ui.add(te);
        });
    }
}

/// Register file/FS/path/env IO, CLI/UI prompts, progress/spinner, and
/// JSON/YAML/TOML/CSV encode/decode utilities.
pub fn register_io(ev: &mut Evaluator) {
    ev.register("ReadFile", read_file as NativeFn, Attributes::empty());
    ev.register("WriteFile", write_file as NativeFn, Attributes::empty());
    ev.register("Puts", puts as NativeFn, Attributes::empty());
    ev.register("Gets", gets as NativeFn, Attributes::empty());
    ev.register("PutsAppend", puts_append as NativeFn, Attributes::empty());
    ev.register("ReadLines", read_lines as NativeFn, Attributes::empty());
    ev.register("FileExistsQ", file_exists_q as NativeFn, Attributes::empty());
    ev.register("ListDirectory", list_directory as NativeFn, Attributes::empty());
    ev.register("Stat", stat_fn as NativeFn, Attributes::empty());
    ev.register("PathJoin", path_join as NativeFn, Attributes::empty());
    ev.register("PathSplit", path_split as NativeFn, Attributes::empty());
    ev.register("CanonicalPath", canonical_path as NativeFn, Attributes::empty());
    ev.register("ExpandPath", expand_path as NativeFn, Attributes::empty());
    ev.register("PathNormalize", canonical_path as NativeFn, Attributes::empty());
    ev.register("PathRelative", path_relative as NativeFn, Attributes::empty());
    ev.register("PathResolve", path_resolve as NativeFn, Attributes::empty());
    ev.register("CurrentDirectory", current_directory as NativeFn, Attributes::empty());
    ev.register("SetDirectory", set_directory as NativeFn, Attributes::empty());
    ev.register("Basename", basename_fn as NativeFn, Attributes::empty());
    ev.register("Dirname", dirname_fn as NativeFn, Attributes::empty());
    ev.register("FileStem", file_stem_fn as NativeFn, Attributes::empty());
    ev.register("FileExtension", file_extension_fn as NativeFn, Attributes::empty());
    ev.register("PathExtname", file_extension_fn as NativeFn, Attributes::empty());
    ev.register("GetEnv", get_env as NativeFn, Attributes::empty());
    ev.register("SetEnv", set_env as NativeFn, Attributes::empty());
    ev.register("DotenvLoad", dotenv_load as NativeFn, Attributes::empty());
    ev.register("ConfigFind", config_find as NativeFn, Attributes::empty());
    ev.register("XdgDirs", xdg_dirs as NativeFn, Attributes::empty());
    ev.register("EnvExpand", env_expand as NativeFn, Attributes::empty());
    ev.register("ReadStdin", read_stdin as NativeFn, Attributes::empty());
    // CLI ergonomics / UI
    ev.register("ArgsParse", args_parse as NativeFn, Attributes::empty());
    ev.register("SetUiBackend", set_ui_backend as NativeFn, Attributes::empty());
    ev.register("GetUiBackend", get_ui_backend as NativeFn, Attributes::empty());
    ev.register("SetUiTheme", set_ui_theme as NativeFn, Attributes::empty());
    ev.register("GetUiTheme", get_ui_theme as NativeFn, Attributes::empty());
    ev.register("Prompt", prompt as NativeFn, Attributes::empty());
    ev.register("Confirm", confirm as NativeFn, Attributes::empty());
    ev.register("PasswordPrompt", password_prompt as NativeFn, Attributes::empty());
    ev.register("PromptSelect", prompt_select_fn as NativeFn, Attributes::empty());
    ev.register("PromptSelectMany", prompt_select_many_fn as NativeFn, Attributes::empty());
    ev.register("Notify", notify_fn as NativeFn, Attributes::empty());
    ev.register("SpinnerStart", spinner_start_fn as NativeFn, Attributes::empty());
    ev.register("SpinnerStop", spinner_stop_fn as NativeFn, Attributes::empty());
    ev.register("ProgressBar", progress_bar as NativeFn, Attributes::empty());
    ev.register("ProgressAdvance", progress_advance as NativeFn, Attributes::empty());
    ev.register("ProgressFinish", progress_finish as NativeFn, Attributes::empty());
    ev.register("ToJson", to_json as NativeFn, Attributes::empty());
    ev.register("FromJson", from_json as NativeFn, Attributes::empty());
    // Aliases for common JSON ops
    ev.register("JsonStringify", to_json as NativeFn, Attributes::empty());
    ev.register("JsonParse", from_json as NativeFn, Attributes::empty());
    // YAML/TOML parse/stringify
    ev.register("YamlParse", yaml_parse as NativeFn, Attributes::empty());
    ev.register("YamlStringify", yaml_stringify as NativeFn, Attributes::empty());
    ev.register("TomlParse", toml_parse as NativeFn, Attributes::empty());
    ev.register("TomlStringify", toml_stringify as NativeFn, Attributes::empty());
    ev.register("ParseCSV", parse_csv as NativeFn, Attributes::empty());
    ev.register("ReadCSV", read_csv as NativeFn, Attributes::empty());
    ev.register("RenderCSV", render_csv as NativeFn, Attributes::empty());
    ev.register("WriteCSV", write_csv as NativeFn, Attributes::empty());
    // Aliases for CSV
    ev.register("CsvRead", parse_csv as NativeFn, Attributes::empty());
    ev.register("CsvWrite", render_csv as NativeFn, Attributes::empty());
    // Bytes / encoding utilities
    ev.register("Base64Encode", base64_encode as NativeFn, Attributes::empty());
    ev.register("Base64Decode", base64_decode as NativeFn, Attributes::empty());
    ev.register("HexEncode", hex_encode_fn as NativeFn, Attributes::empty());
    ev.register("HexDecode", hex_decode_fn as NativeFn, Attributes::empty());
    ev.register("TextEncode", text_encode as NativeFn, Attributes::empty());
    ev.register("TextDecode", text_decode as NativeFn, Attributes::empty());
    ev.register("BytesConcat", bytes_concat as NativeFn, Attributes::empty());
    ev.register("BytesSlice", bytes_slice as NativeFn, Attributes::empty());
    ev.register("BytesLength", bytes_length as NativeFn, Attributes::empty());

    // Terminal formatting helpers
    ev.register("TermSize", term_size as NativeFn, Attributes::empty());
    ev.register("AnsiStyle", ansi_style as NativeFn, Attributes::empty());
    ev.register("AnsiEnabled", ansi_enabled as NativeFn, Attributes::empty());
    ev.register("StripAnsi", strip_ansi as NativeFn, Attributes::empty());
    ev.register("AlignLeft", align_left as NativeFn, Attributes::empty());
    ev.register("AlignRight", align_right as NativeFn, Attributes::empty());
    ev.register("AlignCenter", align_center as NativeFn, Attributes::empty());
    ev.register("Truncate", truncate_fn as NativeFn, Attributes::empty());
    ev.register("Wrap", wrap_fn as NativeFn, Attributes::empty());
    ev.register("TableSimple", table_simple as NativeFn, Attributes::empty());
    ev.register("Rule", rule_fn as NativeFn, Attributes::empty());
    ev.register("BoxText", box_text as NativeFn, Attributes::empty());
    ev.register("Columnize", columns_fn as NativeFn, Attributes::empty());

    #[cfg(feature = "tools")]
    add_specs(vec![
        tool_spec!("ReadFile", summary: "Read entire file as string", params: ["path"], tags: ["io","fs"], input_schema: schema_str!(), output_schema: schema_str!(), effects: ["fs.read"]),
        tool_spec!("WriteFile", summary: "Write stringified content to file", params: ["path","content"], tags: ["io","fs"], input_schema: lyra_core::value::Value::Assoc(HashMap::new()), effects: ["fs.write"]),
        tool_spec!("ReadLines", summary: "Read file and split into lines", params: ["path"], tags: ["io","fs"], input_schema: schema_str!(), effects: ["fs.read"]),
        tool_spec!("FileExistsQ", summary: "Check if file or directory exists", params: ["path"], tags: ["io","fs"], input_schema: schema_str!(), output_schema: schema_bool!()),
        tool_spec!("ListDirectory", summary: "List directory entries (names only)", params: ["path"], tags: ["io","fs"], input_schema: schema_str!(), output_schema: lyra_core::value::Value::Assoc(HashMap::from([(String::from("type"), lyra_core::value::Value::String(String::from("array")))])), effects: ["fs.read"]),
        tool_spec!("Stat", summary: "Get basic file metadata", params: ["path"], tags: ["io","fs"], input_schema: schema_str!(), output_schema: lyra_core::value::Value::Assoc(HashMap::from([(String::from("type"), lyra_core::value::Value::String(String::from("object")))])), effects: ["fs.read"]),
        tool_spec!("PathJoin", summary: "Join path segments", params: ["parts"], tags: ["io","path"]),
        tool_spec!("PathSplit", summary: "Split path into segments", params: ["path"], tags: ["io","path"], output_schema: lyra_core::value::Value::Assoc(HashMap::from([(String::from("type"), lyra_core::value::Value::String(String::from("array")))]))),
        tool_spec!("CanonicalPath", summary: "Resolve symlinks and norm path", params: ["path"], tags: ["io","path"], effects: ["fs.read"]),
        tool_spec!("ExpandPath", summary: "Expand ~ to home", params: ["path"], tags: ["io","path"]),
        tool_spec!("CurrentDirectory", summary: "Get current working directory", params: [], tags: ["io","path"]),
        tool_spec!("SetDirectory", summary: "Change current directory", params: ["path"], tags: ["io","path"], effects: ["fs.chdir"]),
        tool_spec!("Basename", summary: "Last path component", params: ["path"], tags: ["io","path"]),
        tool_spec!("Dirname", summary: "Parent directory path", params: ["path"], tags: ["io","path"]),
        tool_spec!("FileStem", summary: "Filename without extension", params: ["path"], tags: ["io","path"]),
        tool_spec!("FileExtension", summary: "File extension (no dot)", params: ["path"], tags: ["io","path"]),
        tool_spec!("GetEnv", summary: "Read environment variable", params: ["name"], tags: ["io","env"], effects: ["env.read"]),
        tool_spec!("SetEnv", summary: "Set environment variable", params: ["name","value"], tags: ["io","env"], effects: ["env.write"]),
        tool_spec!("ReadStdin", summary: "Read all text from stdin", params: [], tags: ["io","stdio"], effects: ["stdio.read"]),
        tool_spec!("ToJson", summary: "Serialize value to JSON string", params: ["value","opts"], tags: ["io","json"]),
        tool_spec!("FromJson", summary: "Parse JSON string to value", params: ["json"], tags: ["io","json"]),
        tool_spec!("JsonStringify", summary: "Alias of ToJson", params: ["value","opts"], tags: ["io","json"]),
        tool_spec!("JsonParse", summary: "Alias of FromJson", params: ["json"], tags: ["io","json"]),
        tool_spec!("YamlParse", summary: "Parse YAML string to value", params: ["yaml"], tags: ["io","yaml"]),
        tool_spec!("YamlStringify", summary: "Render value as YAML", params: ["value","opts"], tags: ["io","yaml"]),
        tool_spec!("TomlParse", summary: "Parse TOML string to value", params: ["toml"], tags: ["io","toml"]),
        tool_spec!("TomlStringify", summary: "Render value as TOML", params: ["value"], tags: ["io","toml"]),
        tool_spec!("ParseCSV", summary: "Parse CSV string to rows", params: ["csv","opts"], tags: ["io","csv"]),
        tool_spec!("ReadCSV", summary: "Read and parse CSV file", params: ["path","opts"], tags: ["io","csv"], effects: ["fs.read"]),
        tool_spec!("RenderCSV", summary: "Render rows to CSV string", params: ["rows","opts"], tags: ["io","csv"]),
        tool_spec!("WriteCSV", summary: "Write rows to CSV file", params: ["path","rows","opts"], tags: ["io","csv"], effects: ["fs.write"]),
        tool_spec!("CsvRead", summary: "Alias of ParseCSV", params: ["csv","opts"], tags: ["io","csv"]),
        tool_spec!("CsvWrite", summary: "Alias of RenderCSV", params: ["rows","opts"], tags: ["io","csv"]),
        tool_spec!("Base64Encode", summary: "Encode bytes to base64", params: ["bytes","opts"], tags: ["io","bytes","encoding"]),
        tool_spec!("Base64Decode", summary: "Decode base64 to bytes", params: ["text","opts"], tags: ["io","bytes","encoding"]),
        tool_spec!("HexEncode", summary: "Encode bytes to hex", params: ["bytes"], tags: ["io","bytes","encoding"]),
        tool_spec!("HexDecode", summary: "Decode hex to bytes", params: ["text"], tags: ["io","bytes","encoding"]),
        tool_spec!("TextEncode", summary: "Encode text to bytes (utf-8)", params: ["text","opts"], tags: ["io","bytes","encoding"]),
        tool_spec!("TextDecode", summary: "Decode bytes to text (utf-8)", params: ["bytes","opts"], tags: ["io","bytes","encoding"]),
        tool_spec!("BytesConcat", summary: "Concatenate byte arrays", params: ["chunks"], tags: ["io","bytes"], output_schema: lyra_core::value::Value::Assoc(HashMap::from([(String::from("type"), lyra_core::value::Value::String(String::from("array")))]))),
        tool_spec!("BytesSlice", summary: "Slice a byte array", params: ["bytes","start","end"], tags: ["io","bytes"], output_schema: lyra_core::value::Value::Assoc(HashMap::from([(String::from("type"), lyra_core::value::Value::String(String::from("array")))]))),
        tool_spec!("BytesLength", summary: "Length of byte array", params: ["bytes"], tags: ["io","bytes"]),
    ]);
}

/// Conditionally register IO functions based on `pred`.
pub fn register_io_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str) -> bool) {
    register_if(ev, pred, "ReadFile", read_file as NativeFn, Attributes::empty());
    register_if(ev, pred, "WriteFile", write_file as NativeFn, Attributes::empty());
    register_if(ev, pred, "ReadLines", read_lines as NativeFn, Attributes::empty());
    register_if(ev, pred, "FileExistsQ", file_exists_q as NativeFn, Attributes::empty());
    register_if(ev, pred, "ListDirectory", list_directory as NativeFn, Attributes::empty());
    register_if(ev, pred, "Stat", stat_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "PathJoin", path_join as NativeFn, Attributes::empty());
    register_if(ev, pred, "PathSplit", path_split as NativeFn, Attributes::empty());
    register_if(ev, pred, "CanonicalPath", canonical_path as NativeFn, Attributes::empty());
    register_if(ev, pred, "ExpandPath", expand_path as NativeFn, Attributes::empty());
    register_if(ev, pred, "CurrentDirectory", current_directory as NativeFn, Attributes::empty());
    register_if(ev, pred, "SetDirectory", set_directory as NativeFn, Attributes::empty());
    register_if(ev, pred, "Basename", basename_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Dirname", dirname_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "FileStem", file_stem_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "FileExtension", file_extension_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "GetEnv", get_env as NativeFn, Attributes::empty());
    register_if(ev, pred, "SetEnv", set_env as NativeFn, Attributes::empty());
    register_if(ev, pred, "ReadStdin", read_stdin as NativeFn, Attributes::empty());
    register_if(ev, pred, "ArgsParse", args_parse as NativeFn, Attributes::empty());
    register_if(ev, pred, "SetUiBackend", set_ui_backend as NativeFn, Attributes::empty());
    register_if(ev, pred, "GetUiBackend", get_ui_backend as NativeFn, Attributes::empty());
    register_if(ev, pred, "SetUiTheme", set_ui_theme as NativeFn, Attributes::empty());
    register_if(ev, pred, "GetUiTheme", get_ui_theme as NativeFn, Attributes::empty());
    register_if(ev, pred, "Prompt", prompt as NativeFn, Attributes::empty());
    register_if(ev, pred, "Confirm", confirm as NativeFn, Attributes::empty());
    register_if(ev, pred, "PasswordPrompt", password_prompt as NativeFn, Attributes::empty());
    register_if(ev, pred, "PromptSelect", prompt_select_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "PromptSelectMany", prompt_select_many_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Notify", notify_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "SpinnerStart", spinner_start_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "SpinnerStop", spinner_stop_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "ProgressBar", progress_bar as NativeFn, Attributes::empty());
    register_if(ev, pred, "ProgressAdvance", progress_advance as NativeFn, Attributes::empty());
    register_if(ev, pred, "ProgressFinish", progress_finish as NativeFn, Attributes::empty());
    register_if(ev, pred, "ToJson", to_json as NativeFn, Attributes::empty());
    register_if(ev, pred, "FromJson", from_json as NativeFn, Attributes::empty());
    register_if(ev, pred, "JsonStringify", to_json as NativeFn, Attributes::empty());
    register_if(ev, pred, "JsonParse", from_json as NativeFn, Attributes::empty());
    register_if(ev, pred, "YamlParse", yaml_parse as NativeFn, Attributes::empty());
    register_if(ev, pred, "YamlStringify", yaml_stringify as NativeFn, Attributes::empty());
    register_if(ev, pred, "TomlParse", toml_parse as NativeFn, Attributes::empty());
    register_if(ev, pred, "TomlStringify", toml_stringify as NativeFn, Attributes::empty());
    register_if(ev, pred, "ParseCSV", parse_csv as NativeFn, Attributes::empty());
    register_if(ev, pred, "ReadCSV", read_csv as NativeFn, Attributes::empty());
    register_if(ev, pred, "RenderCSV", render_csv as NativeFn, Attributes::empty());
    register_if(ev, pred, "WriteCSV", write_csv as NativeFn, Attributes::empty());
    register_if(ev, pred, "CsvRead", parse_csv as NativeFn, Attributes::empty());
    register_if(ev, pred, "CsvWrite", render_csv as NativeFn, Attributes::empty());
    register_if(ev, pred, "Base64Encode", base64_encode as NativeFn, Attributes::empty());
    register_if(ev, pred, "Base64Decode", base64_decode as NativeFn, Attributes::empty());
    register_if(ev, pred, "HexEncode", hex_encode_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "HexDecode", hex_decode_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "TextEncode", text_encode as NativeFn, Attributes::empty());
    register_if(ev, pred, "TextDecode", text_decode as NativeFn, Attributes::empty());
    register_if(ev, pred, "BytesConcat", bytes_concat as NativeFn, Attributes::empty());
    register_if(ev, pred, "BytesSlice", bytes_slice as NativeFn, Attributes::empty());
    register_if(ev, pred, "BytesLength", bytes_length as NativeFn, Attributes::empty());
}

fn read_file(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("ReadFile".into())), args };
    }
    let path = match &args[0] {
        Value::String(s) | Value::Symbol(s) => s.clone(),
        other => {
            return Value::Expr {
                head: Box::new(Value::Symbol("ReadFile".into())),
                args: vec![other.clone()],
            }
        }
    };
    match std::fs::read_to_string(&path) {
        Ok(s) => Value::String(s),
        Err(e) => failure("IO::read", &format!("ReadFile: {}", e)),
    }
}

fn write_file(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("WriteFile".into())), args };
    }
    let path = match &args[0] {
        Value::String(s) | Value::Symbol(s) => s.clone(),
        other => {
            return Value::Expr {
                head: Box::new(Value::Symbol("WriteFile".into())),
                args: vec![other.clone(), args[1].clone()],
            }
        }
    };
    let content_v = ev.eval(args[1].clone());
    let content = match &content_v {
        Value::String(s) => s.clone(),
        _ => lyra_core::pretty::format_value(&content_v),
    };
    match std::fs::write(&path, content) {
        Ok(_) => Value::Boolean(true),
        Err(e) => failure("IO::write", &format!("WriteFile: {}", e)),
    }
}

// Puts[value] -> prints to stdout with newline
// Puts[value, path] -> writes stringified value to file (overwrites)
fn puts(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v] => {
            let s = to_string_arg(ev, v.clone());
            println!("{}", s);
            Value::Boolean(true)
        }
        [v, p] => {
            let s = to_string_arg(ev, v.clone());
            let path = to_string_arg(ev, p.clone());
            match std::fs::write(&path, s.as_bytes()) {
                Ok(_) => Value::Boolean(true),
                Err(e) => failure("IO::puts", &e.to_string()),
            }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("Puts".into())), args },
    }
}

// Gets[path] -> reads entire file as string
// Future: Gets[] could read stdin
fn gets(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        let mut s = String::new();
        match std::io::stdin().read_to_string(&mut s) {
            Ok(_) => Value::String(s),
            Err(e) => failure("IO::gets", &e.to_string()),
        }
    } else if args.len() == 1 {
        let path = to_string_arg(ev, args[0].clone());
        match std::fs::read_to_string(&path) {
            Ok(s) => Value::String(s),
            Err(e) => failure("IO::gets", &e.to_string()),
        }
    } else {
        Value::Expr { head: Box::new(Value::Symbol("Gets".into())), args }
    }
}

fn puts_append(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("PutsAppend".into())), args };
    }
    let s = to_string_arg(ev, args[0].clone());
    let path = to_string_arg(ev, args[1].clone());
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut f| std::io::Write::write_all(&mut f, s.as_bytes()))
    {
        Ok(_) => Value::Boolean(true),
        Err(e) => failure("IO::puts", &e.to_string()),
    }
}

// -------- Terminal formatting helpers --------
fn term_size(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if !args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("TermSize".into())), args };
    }
    if let Some((w, h)) = terminal_size::terminal_size() {
        Value::Assoc(std::collections::HashMap::from([
            ("width".into(), Value::Integer(w.0 as i64)),
            ("height".into(), Value::Integer(h.0 as i64)),
        ]))
    } else {
        Value::Assoc(std::collections::HashMap::from([
            ("width".into(), Value::Integer(80)),
            ("height".into(), Value::Integer(24)),
        ]))
    }
}

fn strip_ansi_regex() -> &'static regex::Regex {
    use std::sync::OnceLock;
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| regex::Regex::new(r"\x1B\[[0-9;?]*[ -/]*[@-~]").unwrap())
}

fn strip_ansi(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("StripAnsi".into())), args };
    }
    match &args[0] {
        v => {
            let s = match v {
                Value::String(s) | Value::Symbol(s) => s.clone(),
                _ => lyra_core::pretty::format_value(v),
            };
            Value::String(strip_ansi_regex().replace_all(&s, "").to_string())
        }
    }
}

fn visible_width(s: &str) -> usize {
    strip_ansi_regex().replace_all(s, "").chars().count()
}

fn ansi_style(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("AnsiStyle".into())), args };
    }
    let text = to_string_arg(ev, args[0].clone());
    let opts = if args.len() > 1 {
        if let Value::Assoc(m) = ev.eval(args[1].clone()) {
            m
        } else {
            std::collections::HashMap::new()
        }
    } else {
        std::collections::HashMap::new()
    };
    let mut codes: Vec<&str> = Vec::new();
    let color_map = |name: &str| -> Option<&'static str> {
        match name.to_ascii_lowercase().as_str() {
            "black" => Some("30"),
            "red" => Some("31"),
            "green" => Some("32"),
            "yellow" => Some("33"),
            "blue" => Some("34"),
            "magenta" => Some("35"),
            "cyan" => Some("36"),
            "white" => Some("37"),
            "gray" | "grey" => Some("90"),
            _ => None,
        }
    };
    if let Some(Value::String(s)) | Some(Value::Symbol(s)) = opts.get("Color") {
        if let Some(c) = color_map(s) {
            codes.push(c);
        }
    }
    if let Some(Value::String(s)) | Some(Value::Symbol(s)) = opts.get("BgColor") {
        if let Some(c) = color_map(s) {
            match c {
                "30" => codes.push("40"),
                "31" => codes.push("41"),
                "32" => codes.push("42"),
                "33" => codes.push("43"),
                "34" => codes.push("44"),
                "35" => codes.push("45"),
                "36" => codes.push("46"),
                "37" => codes.push("47"),
                "90" => codes.push("100"),
                _ => {}
            }
        }
    }
    if matches!(opts.get("Bold"), Some(Value::Boolean(true))) {
        codes.push("1");
    }
    if matches!(opts.get("Dim"), Some(Value::Boolean(true))) {
        codes.push("2");
    }
    if matches!(opts.get("Italic"), Some(Value::Boolean(true))) {
        codes.push("3");
    }
    if matches!(opts.get("Underline"), Some(Value::Boolean(true))) {
        codes.push("4");
    }
    if matches!(opts.get("Invert"), Some(Value::Boolean(true))) {
        codes.push("7");
    }
    if codes.is_empty() {
        return Value::String(text);
    }
    let start = format!("\x1b[{}m", codes.join(";"));
    let end = "\x1b[0m";
    Value::String(format!("{}{}{}", start, text, end))
}

fn ansi_enabled(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if !args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("AnsiEnabled".into())), args };
    }
    let no_color = std::env::var("NO_COLOR").ok();
    Value::Boolean(no_color.is_none())
}

fn align_left(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    align_generic(ev, args, 0)
}
fn align_right(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    align_generic(ev, args, 2)
}
fn align_center(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    align_generic(ev, args, 1)
}

fn align_generic(ev: &mut Evaluator, args: Vec<Value>, mode: i32) -> Value {
    if args.len() < 2 {
        return Value::Expr {
            head: Box::new(Value::Symbol(
                match mode {
                    0 => "AlignLeft",
                    1 => "AlignCenter",
                    _ => "AlignRight",
                }
                .into(),
            )),
            args,
        };
    }
    let s = to_string_arg(ev, args[0].clone());
    let w = match ev.eval(args[1].clone()) {
        Value::Integer(n) if n >= 0 => n as usize,
        other => {
            return Value::Expr {
                head: Box::new(Value::Symbol(
                    match mode {
                        0 => "AlignLeft",
                        1 => "AlignCenter",
                        _ => "AlignRight",
                    }
                    .into(),
                )),
                args: vec![Value::String(s), other],
            }
        }
    };
    let pad = if args.len() > 2 { to_string_arg(ev, args[2].clone()) } else { " ".to_string() };
    let pad_ch = pad.chars().next().unwrap_or(' ');
    let width = visible_width(&s);
    if width >= w {
        return Value::String(s);
    }
    let spaces = w - width;
    let out = match mode {
        0 => format!("{}{}", s, pad_ch.to_string().repeat(spaces)),
        2 => format!("{}{}", pad_ch.to_string().repeat(spaces), s),
        _ => {
            let l = spaces / 2;
            let r = spaces - l;
            format!("{}{}{}", pad_ch.to_string().repeat(l), s, pad_ch.to_string().repeat(r))
        }
    };
    Value::String(out)
}

fn truncate_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("Truncate".into())), args };
    }
    let s = to_string_arg(ev, args[0].clone());
    let w = match ev.eval(args[1].clone()) {
        Value::Integer(n) if n >= 0 => n as usize,
        other => {
            return Value::Expr {
                head: Box::new(Value::Symbol("Truncate".into())),
                args: vec![Value::String(s), other],
            }
        }
    };
    let ell = if args.len() > 2 { to_string_arg(ev, args[2].clone()) } else { "…".to_string() };
    let width = visible_width(&s);
    if width <= w {
        return Value::String(s);
    }
    let target = w.saturating_sub(visible_width(&ell));
    let mut out = String::new();
    let mut count = 0;
    for ch in strip_ansi_regex().replace_all(&s, "").chars() {
        if count >= target {
            break;
        }
        out.push(ch);
        count += 1;
    }
    Value::String(out + &ell)
}

fn wrap_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("Wrap".into())), args };
    }
    let s = to_string_arg(ev, args[0].clone());
    let w = match ev.eval(args[1].clone()) {
        Value::Integer(n) if n > 0 => n as usize,
        other => {
            return Value::Expr {
                head: Box::new(Value::Symbol("Wrap".into())),
                args: vec![Value::String(s), other],
            }
        }
    };
    let mut out_lines: Vec<String> = Vec::new();
    for line in s.split('\n') {
        let mut cur = String::new();
        for word in line.split_whitespace() {
            let sep = if cur.is_empty() { 0 } else { 1 };
            let next_w = visible_width(&cur) + sep + word.chars().count();
            if next_w > w && !cur.is_empty() {
                out_lines.push(cur);
                cur = word.to_string();
            } else {
                if sep == 1 {
                    cur.push(' ');
                }
                cur.push_str(word);
            }
        }
        out_lines.push(cur);
    }
    Value::String(out_lines.join("\n"))
}

fn table_simple(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("TableSimple".into())), args };
    }
    let rows = match ev.eval(args[0].clone()) {
        Value::List(vs) => vs,
        other => {
            return Value::Expr {
                head: Box::new(Value::Symbol("TableSimple".into())),
                args: vec![other],
            }
        }
    };
    let opts = if args.len() > 1 {
        if let Value::Assoc(m) = ev.eval(args[1].clone()) {
            m
        } else {
            std::collections::HashMap::new()
        }
    } else {
        std::collections::HashMap::new()
    };
    let headers: Vec<String> = if let Some(Value::List(cols)) = opts.get("Columns") {
        cols.iter()
            .filter_map(|v| match v {
                Value::String(s) | Value::Symbol(s) => Some(s.clone()),
                _ => None,
            })
            .collect()
    } else {
        // Infer from first assoc row
        rows.iter()
            .find_map(|r| {
                if let Value::Assoc(m) = r {
                    Some(m.keys().cloned().collect())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| vec![])
    };
    let mut data: Vec<Vec<String>> = Vec::new();
    for r in &rows {
        if let Value::Assoc(m) = r {
            data.push(
                headers
                    .iter()
                    .map(|k| match m.get(k) {
                        Some(Value::String(s)) => s.clone(),
                        Some(Value::Symbol(s)) => s.clone(),
                        Some(v) => lyra_core::pretty::format_value(v),
                        None => "".into(),
                    })
                    .collect(),
            );
        } else if let Value::List(items) = r {
            data.push(items.iter().map(|v| lyra_core::pretty::format_value(v)).collect());
        }
    }
    ui_backend().lock().unwrap().render_table(&headers, &data, &opts)
}

fn rule_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let ch = if !args.is_empty() {
        to_string_arg(ev, args[0].clone()).chars().next().unwrap_or('-')
    } else {
        '-'
    };
    let width = if args.len() > 1 {
        match ev.eval(args[1].clone()) {
            Value::Integer(n) if n > 0 => n as usize,
            _ => 0,
        }
    } else {
        0
    };
    let w = if width > 0 {
        width
    } else {
        if let Some((tw, _)) = terminal_size::terminal_size() {
            tw.0 as usize
        } else {
            80
        }
    };
    Value::String(ch.to_string().repeat(w))
}

fn box_text(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("BoxText".into())), args };
    }
    let text = to_string_arg(ev, args[0].clone());
    let opts = if args.len() > 1 {
        if let Value::Assoc(m) = ev.eval(args[1].clone()) {
            m
        } else {
            std::collections::HashMap::new()
        }
    } else {
        std::collections::HashMap::new()
    };
    ui_backend().lock().unwrap().render_box(&text, &opts)
}

// Terminal render helpers for backend
fn term_render_table(
    headers: &[String],
    data: &[Vec<String>],
    opts: &std::collections::HashMap<String, Value>,
) -> String {
    let mut theme = ui_term_theme().lock().unwrap().clone();
    if let Some(Value::String(s)) | Some(Value::Symbol(s)) = opts.get("AccentColor") { theme.accent = Some(s.clone()); }
    if let Some(Value::Boolean(b)) = opts.get("Compact") { theme.compact = *b; }
    let mut widths: Vec<usize> = headers.iter().map(|h| h.chars().count()).collect();
    for row in data.iter() {
        for (i, cell) in row.iter().enumerate() {
            if i < widths.len() {
                widths[i] = widths[i].max(cell.chars().count());
            } else {
                widths.push(cell.chars().count());
            }
        }
    }
    let pad = if theme.compact { 0usize } else { 1usize };
    let mut out = String::new();
    if !headers.is_empty() {
        let (start, end) = term_accent_codes(theme.accent.as_deref());
        for (i, h) in headers.iter().enumerate() {
            let w = widths.get(i).cloned().unwrap_or(0) + pad;
            if !start.is_empty() { out.push_str(start); }
            out.push_str(&format!("{:width$}", h, width = w));
            if !end.is_empty() { out.push_str(end); }
        }
        out.push('\n');
        for (i, _) in headers.iter().enumerate() {
            let w = widths.get(i).cloned().unwrap_or(0) + pad;
            if !start.is_empty() { out.push_str(start); }
            out.push_str(&"-".repeat(w));
            if !end.is_empty() { out.push_str(end); }
        }
        out.push('\n');
    }
    for row in data.iter() {
        for (i, cell) in row.iter().enumerate() {
            let w = widths.get(i).cloned().unwrap_or(0) + pad;
            out.push_str(&format!("{:width$}", cell, width = w));
        }
        out.push('\n');
    }
    out
}

fn term_render_box(
    text: &str,
    opts: &std::collections::HashMap<String, Value>,
) -> String {
    let mut theme = ui_term_theme().lock().unwrap().clone();
    if let Some(Value::String(s)) | Some(Value::Symbol(s)) = opts.get("AccentColor") { theme.accent = Some(s.clone()); }
    if let Some(Value::Boolean(b)) = opts.get("Compact") { theme.compact = *b; }
    let padding = opts
        .get("Padding")
        .and_then(|v| if let Value::Integer(n) = v { Some(*n as usize) } else { None })
        .unwrap_or_else(|| if theme.compact { 0 } else { 1 });
    let border = opts
        .get("Border")
        .and_then(|v| if let Value::String(s) | Value::Symbol(s) = v { Some(s.clone()) } else { None })
        .unwrap_or("+-|".into());
    let mut chars = border.chars();
    let tl = chars.next().unwrap_or('+');
    let tr = chars.next().unwrap_or('-');
    let vbar = chars.next().unwrap_or('|');
    let lines: Vec<&str> = text.split('\n').collect();
    let maxw = lines.iter().map(|l| visible_width(l)).max().unwrap_or(0);
    let inner = maxw + 2 * padding;
    let mut out = String::new();
    let (start, end) = term_accent_codes(theme.accent.as_deref());
    if !start.is_empty() { out.push_str(start); }
    out.push(tl);
    out.push_str(&tr.to_string().repeat(inner));
    out.push(tl);
    if !end.is_empty() { out.push_str(end); }
    out.push('\n');
    for ln in lines {
        if !start.is_empty() { out.push_str(start); }
        out.push(vbar);
        if !end.is_empty() { out.push_str(end); }
        out.push_str(&" ".repeat(padding));
        out.push_str(ln);
        out.push_str(&" ".repeat(inner - visible_width(ln)));
        if !start.is_empty() { out.push_str(start); }
        out.push(vbar);
        if !end.is_empty() { out.push_str(end); }
        out.push('\n');
    }
    if !start.is_empty() { out.push_str(start); }
    out.push(tl);
    out.push_str(&tr.to_string().repeat(inner));
    out.push(tl);
    if !end.is_empty() { out.push_str(end); }
    out
}

fn term_accent_codes(name: Option<&str>) -> (&'static str, &'static str) {
    if let Some(n) = name { if let Some(code) = ansi_code_for_color(n) { return (code, "\x1b[0m"); } }
    ("", "")
}

fn ansi_code_for_color(name: &str) -> Option<&'static str> {
    match name.to_ascii_lowercase().as_str() {
        "black" => Some("\x1b[30m"),
        "red" => Some("\x1b[31m"),
        "green" => Some("\x1b[32m"),
        "yellow" => Some("\x1b[33m"),
        "blue" => Some("\x1b[34m"),
        "magenta" => Some("\x1b[35m"),
        "cyan" => Some("\x1b[36m"),
        "white" => Some("\x1b[37m"),
        "gray" | "grey" => Some("\x1b[90m"),
        _ => None,
    }
}

#[cfg(feature = "ui_egui")]
fn egui_parse_color32(s: &str) -> Option<egui::Color32> {
    let lower = s.trim().to_ascii_lowercase();
    match lower.as_str() {
        "black" => Some(egui::Color32::BLACK),
        "white" => Some(egui::Color32::WHITE),
        "gray" | "grey" => Some(egui::Color32::GRAY),
        "red" => Some(egui::Color32::from_rgb(255,0,0)),
        "green" => Some(egui::Color32::from_rgb(0,255,0)),
        "blue" => Some(egui::Color32::from_rgb(0,128,255)),
        "yellow" => Some(egui::Color32::from_rgb(255,255,0)),
        "magenta" | "purple" => Some(egui::Color32::from_rgb(255,0,255)),
        "cyan" | "teal" => Some(egui::Color32::from_rgb(0,255,255)),
        _ => {
            // hex #RRGGBB
            let h = lower.trim_start_matches('#');
            if h.len() == 6 { if let (Ok(r), Ok(g), Ok(b)) = (u8::from_str_radix(&h[0..2],16), u8::from_str_radix(&h[2..4],16), u8::from_str_radix(&h[4..6],16)) { return Some(egui::Color32::from_rgb(r,g,b)); } }
            None
        }
    }
}

fn columns_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("Columns".into())), args };
    }
    let items = match ev.eval(args[0].clone()) {
        Value::List(vs) => {
            vs.into_iter().map(|v| lyra_core::pretty::format_value(&v)).collect::<Vec<_>>()
        }
        other => {
            return Value::Expr {
                head: Box::new(Value::Symbol("Columns".into())),
                args: vec![other, args[1].clone()],
            }
        }
    };
    let cols = match ev.eval(args[1].clone()) {
        Value::Integer(n) if n > 0 => n as usize,
        _ => 2,
    };
    let termw = if args.len() > 2 {
        match ev.eval(args[2].clone()) {
            Value::Integer(n) if n > 0 => n as usize,
            _ => 0,
        }
    } else {
        0
    };
    let width = if termw > 0 {
        termw
    } else {
        if let Some((tw, _)) = terminal_size::terminal_size() {
            tw.0 as usize
        } else {
            80
        }
    };
    let colw = (width.saturating_sub(cols - 1)) / cols; // 1 space between columns
    let mut out = String::new();
    let mut i = 0usize;
    while i < items.len() {
        let mut line = String::new();
        for c in 0..cols {
            if i + c < items.len() {
                let cell = truncate_visible(&items[i + c], colw);
                line.push_str(&format!("{:<width$}", cell, width = colw));
                if c + 1 < cols {
                    line.push(' ');
                }
            }
        }
        out.push_str(&line);
        out.push('\n');
        i += cols;
    }
    Value::String(out)
}

fn truncate_visible(s: &str, w: usize) -> String {
    let width = visible_width(s);
    if width <= w {
        return s.to_string();
    }
    let mut out = String::new();
    let mut count = 0;
    for ch in strip_ansi_regex().replace_all(s, "").chars() {
        if count >= w {
            break;
        }
        out.push(ch);
        count += 1;
    }
    out
}

fn read_lines(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("ReadLines".into())), args };
    }
    let path = match &args[0] {
        Value::String(s) | Value::Symbol(s) => s.clone(),
        other => {
            return Value::Expr {
                head: Box::new(Value::Symbol("ReadLines".into())),
                args: vec![other.clone()],
            }
        }
    };
    match std::fs::read_to_string(&path) {
        Ok(s) => Value::List(s.lines().map(|x| Value::String(x.to_string())).collect()),
        Err(e) => failure("IO::read", &format!("ReadLines: {}", e)),
    }
}

fn file_exists_q(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("FileExistsQ".into())), args };
    }
    let path = match &args[0] {
        Value::String(s) | Value::Symbol(s) => s.clone(),
        other => {
            return Value::Expr {
                head: Box::new(Value::Symbol("FileExistsQ".into())),
                args: vec![other.clone()],
            }
        }
    };
    Value::Boolean(std::path::Path::new(&path).exists())
}

fn list_directory(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("ListDirectory".into())), args };
    }
    let path = match &args[0] {
        Value::String(s) | Value::Symbol(s) => s.clone(),
        other => {
            return Value::Expr {
                head: Box::new(Value::Symbol("ListDirectory".into())),
                args: vec![other.clone()],
            }
        }
    };
    let mut out: Vec<Value> = Vec::new();
    match std::fs::read_dir(&path) {
        Ok(read_dir) => {
            for entry in read_dir.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    out.push(Value::String(name.to_string()));
                }
            }
            Value::List(out)
        }
        Err(e) => failure("IO::list", &format!("ListDirectory: {}", e)),
    }
}

fn stat_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("Stat".into())), args };
    }
    let path = match &args[0] {
        Value::String(s) | Value::Symbol(s) => s.clone(),
        other => {
            return Value::Expr {
                head: Box::new(Value::Symbol("Stat".into())),
                args: vec![other.clone()],
            }
        }
    };
    match std::fs::metadata(&path) {
        Ok(meta) => {
            let is_dir = meta.is_dir();
            let is_file = meta.is_file();
            let size = meta.len() as i64;
            let modified = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            let mut m: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
            m.insert("isDir".into(), Value::Boolean(is_dir));
            m.insert("isFile".into(), Value::Boolean(is_file));
            m.insert("size".into(), Value::Integer(size));
            m.insert("modified".into(), Value::Integer(modified));
            Value::Assoc(m)
        }
        Err(e) => failure("IO::stat", &format!("Stat: {}", e)),
    }
}

fn failure(tag: &str, msg: &str) -> Value {
    Value::Assoc(
        vec![
            ("message".to_string(), Value::String(msg.to_string())),
            ("tag".to_string(), Value::String(tag.to_string())),
        ]
        .into_iter()
        .collect(),
    )
}

fn to_string_arg(ev: &mut Evaluator, v: Value) -> String {
    match ev.eval(v) {
        Value::String(s) | Value::Symbol(s) => s,
        other => lyra_core::pretty::format_value(&other),
    }
}

fn path_join(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    use std::path::PathBuf;
    if args.is_empty() {
        return Value::String(String::new());
    }
    let parts: Vec<String> = if args.len() == 1 {
        match ev.eval(args[0].clone()) {
            Value::List(items) => items.into_iter().map(|it| to_string_arg(ev, it)).collect(),
            other => vec![to_string_arg(ev, other)],
        }
    } else {
        args.into_iter().map(|a| to_string_arg(ev, a)).collect()
    };
    let mut pb = PathBuf::new();
    for p in parts {
        pb.push(p);
    }
    Value::String(pb.to_string_lossy().to_string())
}

fn path_split(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("PathSplit".into())), args };
    }
    let path = to_string_arg(ev, args[0].clone());
    let p = std::path::Path::new(&path);
    let comps: Vec<Value> = p
        .components()
        .map(|c| Value::String(c.as_os_str().to_string_lossy().to_string()))
        .collect();
    Value::List(comps)
}

fn canonical_path(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("CanonicalPath".into())), args };
    }
    let path = to_string_arg(ev, args[0].clone());
    match std::fs::canonicalize(&path) {
        Ok(pb) => Value::String(pb.to_string_lossy().to_string()),
        Err(e) => failure("IO::canonical", &format!("CanonicalPath: {}", e)),
    }
}

fn path_relative(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("PathRelative".into())), args };
    }
    let base = to_string_arg(ev, args[0].clone());
    let path = to_string_arg(ev, args[1].clone());
    let bp = std::path::Path::new(&base);
    let pp = std::path::Path::new(&path);
    match pathdiff::diff_paths(pp, bp) {
        Some(p) => Value::String(p.to_string_lossy().to_string()),
        None => Value::String(path),
    }
}

fn path_resolve(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("PathResolve".into())), args };
    }
    let base = to_string_arg(ev, args[0].clone());
    let rel = to_string_arg(ev, args[1].clone());
    let p = std::path::Path::new(&base).join(rel);
    Value::String(p.to_string_lossy().to_string())
}

fn expand_path(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("ExpandPath".into())), args };
    }
    let path = to_string_arg(ev, args[0].clone());
    if path == "~" || path.starts_with("~/") {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| String::from(""));
        let rest = if path == "~" { String::new() } else { path[2..].to_string() };
        let mut pb = std::path::PathBuf::from(home);
        if !rest.is_empty() {
            pb.push(rest);
        }
        Value::String(pb.to_string_lossy().to_string())
    } else {
        Value::String(path)
    }
}

fn current_directory(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if !args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("CurrentDirectory".into())), args };
    }
    match std::env::current_dir() {
        Ok(p) => Value::String(p.to_string_lossy().to_string()),
        Err(e) => failure("IO::cwd", &format!("CurrentDirectory: {}", e)),
    }
}

fn set_directory(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("SetDirectory".into())), args };
    }
    let path = to_string_arg(ev, args[0].clone());
    match std::env::set_current_dir(&path) {
        Ok(_) => Value::Boolean(true),
        Err(e) => failure("IO::chdir", &format!("SetDirectory: {}", e)),
    }
}

fn basename_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("Basename".into())), args };
    }
    let path = to_string_arg(ev, args[0].clone());
    let p = std::path::Path::new(&path);
    Value::String(p.file_name().unwrap_or_default().to_string_lossy().to_string())
}

fn dirname_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("Dirname".into())), args };
    }
    let path = to_string_arg(ev, args[0].clone());
    let p = std::path::Path::new(&path);
    Value::String(
        p.parent().unwrap_or_else(|| std::path::Path::new("")).to_string_lossy().to_string(),
    )
}

fn file_stem_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("FileStem".into())), args };
    }
    let path = to_string_arg(ev, args[0].clone());
    let p = std::path::Path::new(&path);
    Value::String(p.file_stem().unwrap_or_default().to_string_lossy().to_string())
}

fn file_extension_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("FileExtension".into())), args };
    }
    let path = to_string_arg(ev, args[0].clone());
    let p = std::path::Path::new(&path);
    Value::String(p.extension().unwrap_or_default().to_string_lossy().to_string())
}

// ---------------- Config / Env helpers ----------------
fn dotenv_load(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let (path_opt, override_vars) = match args.as_slice() {
        [] => (None, false),
        [p] => {
            let s = to_string_arg(ev, p.clone());
            (Some(s), false)
        }
        [p, o] => {
            let s = to_string_arg(ev, p.clone());
            let ov = matches!(ev.eval(o.clone()), Value::Assoc(m) if m.get("Override").cloned()==Some(Value::Boolean(true)));
            (Some(s), ov)
        }
        _ => (None, false),
    };
    let path = path_opt.unwrap_or_else(|| {
        let cwd = std::env::current_dir().unwrap_or_default();
        cwd.join(".env").to_string_lossy().to_string()
    });
    let content = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => return failure("IO::env", &format!("DotenvLoad: {}", e)),
    };
    let mut count = 0;
    for line in content.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let mut parts = t.splitn(2, '=');
        if let (Some(k), Some(v)) = (parts.next(), parts.next()) {
            let key = k.trim().to_string();
            let val = v.trim().trim_matches('"').trim_matches('\'').to_string();
            if override_vars || std::env::var(&key).is_err() {
                std::env::set_var(&key, &val);
                count += 1;
            }
        }
    }
    Value::Assoc(std::collections::HashMap::from([
        ("path".into(), Value::String(path)),
        ("loaded".into(), Value::Integer(count)),
    ]))
}

fn config_find(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let (names, start_dir) = match args.as_slice() {
        [] => (
            vec![".env".to_string(), "lyra.toml".to_string(), "config.yaml".to_string()],
            std::env::current_dir().unwrap_or_default(),
        ),
        [n] => {
            let mut v = Vec::new();
            match ev.eval(n.clone()) {
                Value::List(xs) => {
                    for x in xs {
                        if let Value::String(s) | Value::Symbol(s) = x {
                            v.push(s);
                        }
                    }
                }
                Value::String(s) | Value::Symbol(s) => v.push(s),
                _ => {}
            }
            (
                if v.is_empty() { vec![".env".into()] } else { v },
                std::env::current_dir().unwrap_or_default(),
            )
        }
        [n, sdir] => {
            let mut v = Vec::new();
            match ev.eval(n.clone()) {
                Value::List(xs) => {
                    for x in xs {
                        if let Value::String(s) | Value::Symbol(s) = x {
                            v.push(s);
                        }
                    }
                }
                Value::String(s) | Value::Symbol(s) => v.push(s),
                _ => {}
            }
            let sd = std::path::PathBuf::from(to_string_arg(ev, sdir.clone()));
            (v, sd)
        }
        _ => (vec![".env".into()], std::env::current_dir().unwrap_or_default()),
    };
    let mut dir = start_dir.as_path();
    loop {
        for name in &names {
            let p = dir.join(name);
            if p.exists() {
                return Value::Assoc(std::collections::HashMap::from([
                    ("path".into(), Value::String(p.to_string_lossy().to_string())),
                    ("name".into(), Value::String(name.clone())),
                ]));
            }
        }
        if let Some(parent) = dir.parent() {
            dir = parent;
        } else {
            break;
        }
    }
    Value::Symbol("Null".into())
}

fn xdg_dirs(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let app = args.get(0).and_then(|v| match ev.eval(v.clone()) {
        Value::String(s) | Value::Symbol(s) => Some(s),
        _ => None,
    });
    let home = std::env::var("HOME").unwrap_or_else(|_| "".into());
    let cfg = std::env::var("XDG_CONFIG_HOME")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| format!("{}/.config", home));
    let cache = std::env::var("XDG_CACHE_HOME")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| format!("{}/.cache", home));
    let data = std::env::var("XDG_DATA_HOME")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| format!("{}/.local/share", home));
    let (cfg, cache, data) = if let Some(appn) = app {
        (format!("{}/{}", cfg, appn), format!("{}/{}", cache, appn), format!("{}/{}", data, appn))
    } else {
        (cfg, cache, data)
    };
    Value::Assoc(std::collections::HashMap::from([
        ("config_dir".into(), Value::String(cfg)),
        ("cache_dir".into(), Value::String(cache)),
        ("data_dir".into(), Value::String(data)),
    ]))
}

fn env_expand(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("EnvExpand".into())), args };
    }
    let text = to_string_arg(ev, args[0].clone());
    let (vars, style_shell) = if args.len() > 1 {
        if let Value::Assoc(m) = ev.eval(args[1].clone()) {
            let vars = m
                .get("Vars")
                .and_then(|v| if let Value::Assoc(mm) = v { Some(mm.clone()) } else { None })
                .unwrap_or_default();
            let style_shell = match m.get("Style") {
                Some(Value::String(s)) | Some(Value::Symbol(s)) if s == "windows" => false,
                _ => true,
            };
            (vars, style_shell)
        } else {
            (std::collections::HashMap::new(), true)
        }
    } else {
        (std::collections::HashMap::new(), true)
    };
    let mut out = String::new();
    if style_shell {
        let bytes = text.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            if bytes[i] == b'$' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'{' {
                    // ${VAR}
                    let mut j = i + 2;
                    while j < bytes.len() && bytes[j] != b'}' {
                        j += 1;
                    }
                    if j < bytes.len() {
                        let name = String::from_utf8_lossy(&bytes[i + 2..j]).to_string();
                        let val = vars
                            .get(&name)
                            .cloned()
                            .or_else(|| std::env::var(&name).ok().map(Value::String))
                            .unwrap_or(Value::String(String::new()));
                        match val {
                            Value::String(s) | Value::Symbol(s) => out.push_str(&s),
                            other => out.push_str(&lyra_core::pretty::format_value(&other)),
                        }
                        i = j + 1;
                        continue;
                    }
                }
                // $VAR
                let mut j = i + 1;
                while j < bytes.len() {
                    let c = bytes[j] as char;
                    if c.is_alphanumeric() || c == '_' {
                        j += 1;
                    } else {
                        break;
                    }
                }
                if j > i + 1 {
                    let name = String::from_utf8_lossy(&bytes[i + 1..j]).to_string();
                    let val = vars
                        .get(&name)
                        .cloned()
                        .or_else(|| std::env::var(&name).ok().map(Value::String))
                        .unwrap_or(Value::String(String::new()));
                    match val {
                        Value::String(s) | Value::Symbol(s) => out.push_str(&s),
                        other => out.push_str(&lyra_core::pretty::format_value(&other)),
                    }
                    i = j;
                    continue;
                }
            }
            out.push(bytes[i] as char);
            i += 1;
        }
    } else {
        // Windows style %VAR%
        let mut i = 0;
        let b = text.as_bytes();
        while i < b.len() {
            if b[i] == b'%' {
                if let Some(jrel) = b[i + 1..].iter().position(|&ch| ch == b'%') {
                    let j = i + 1 + jrel;
                    let name = String::from_utf8_lossy(&b[i + 1..j]).to_string();
                    let val = vars
                        .get(&name)
                        .cloned()
                        .or_else(|| std::env::var(&name).ok().map(Value::String))
                        .unwrap_or(Value::String(String::new()));
                    match val {
                        Value::String(s) | Value::Symbol(s) => out.push_str(&s),
                        other => out.push_str(&lyra_core::pretty::format_value(&other)),
                    };
                    i = j + 1;
                    continue;
                }
            }
            out.push(b[i] as char);
            i += 1;
        }
    }
    Value::String(out)
}

fn get_env(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("GetEnv".into())), args };
    }
    let name = to_string_arg(ev, args[0].clone());
    match std::env::var(&name) {
        Ok(v) => Value::String(v),
        Err(_) => Value::Symbol("Null".into()),
    }
}

fn set_env(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("SetEnv".into())), args };
    }
    let name = to_string_arg(ev, args[0].clone());
    let val = to_string_arg(ev, args[1].clone());
    std::env::set_var(name, val);
    Value::Boolean(true)
}

fn read_stdin(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if !args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("ReadStdin".into())), args };
    }
    use std::io::Read;
    let mut s = String::new();
    match std::io::stdin().read_to_string(&mut s) {
        Ok(_) => Value::String(s),
        Err(e) => failure("IO::stdin", &format!("ReadStdin: {}", e)),
    }
}

// ------------- CLI Ergonomics -------------
fn args_parse(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // ArgsParse[spec?, argv?]
    let (spec, argv) = match args.as_slice() {
        [] => (
            Value::Assoc(std::collections::HashMap::new()),
            std::env::args().skip(1).collect::<Vec<_>>(),
        ),
        [s] => (ev.eval(s.clone()), std::env::args().skip(1).collect::<Vec<_>>()),
        [s, av] => {
            let v = match ev.eval(av.clone()) {
                Value::List(xs) => xs
                    .into_iter()
                    .filter_map(|v| {
                        if let Value::String(s) | Value::Symbol(s) = v {
                            Some(s)
                        } else {
                            None
                        }
                    })
                    .collect(),
                _ => vec![],
            };
            (ev.eval(s.clone()), v)
        }
        _ => (Value::Assoc(std::collections::HashMap::new()), vec![]),
    };
    let mut opts_spec: std::collections::HashMap<String, (String, String, bool, bool)> =
        std::collections::HashMap::new(); // name -> (short, typ, repeat, is_flag)
    if let Value::Assoc(m) = spec {
        if let Some(Value::List(items)) = m.get("Options") {
            for it in items {
                if let Value::Assoc(o) = it {
                    let name = o
                        .get("Name")
                        .and_then(|v| {
                            if let Value::String(s) | Value::Symbol(s) = v {
                                Some(s.clone())
                            } else {
                                None
                            }
                        })
                        .unwrap_or_default();
                    let short = o
                        .get("Short")
                        .and_then(|v| {
                            if let Value::String(s) | Value::Symbol(s) = v {
                                Some(s.clone())
                            } else {
                                None
                            }
                        })
                        .unwrap_or_default();
                    let typ = o
                        .get("Type")
                        .and_then(|v| {
                            if let Value::String(s) | Value::Symbol(s) = v {
                                Some(s.clone())
                            } else {
                                None
                            }
                        })
                        .unwrap_or_else(|| "string".into());
                    let rpt = matches!(o.get("Repeat"), Some(Value::Boolean(true)));
                    opts_spec.insert(name, (short, typ, rpt, false));
                }
            }
        }
        if let Some(Value::List(items)) = m.get("Flags") {
            for it in items {
                if let Value::Assoc(o) = it {
                    let name = o
                        .get("Name")
                        .and_then(|v| {
                            if let Value::String(s) | Value::Symbol(s) = v {
                                Some(s.clone())
                            } else {
                                None
                            }
                        })
                        .unwrap_or_default();
                    let short = o
                        .get("Short")
                        .and_then(|v| {
                            if let Value::String(s) | Value::Symbol(s) = v {
                                Some(s.clone())
                            } else {
                                None
                            }
                        })
                        .unwrap_or_default();
                    opts_spec.insert(name, (short, "bool".into(), false, true));
                }
            }
        }
    }
    let mut options: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
    let mut flags: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
    let mut rest: Vec<Value> = Vec::new();
    let mut i = 0;
    while i < argv.len() {
        let a = &argv[i];
        if a.starts_with("--") {
            let mut kv = a[2..].splitn(2, '=');
            let k = kv.next().unwrap().to_string();
            let v_opt = kv.next().map(|s| s.to_string());
            if let Some((_, typ, rpt, is_flag)) = opts_spec.get(&k).cloned() {
                if is_flag {
                    flags.insert(k, Value::Boolean(true));
                } else {
                    let val = if let Some(vs) = v_opt {
                        vs
                    } else {
                        i += 1;
                        if i < argv.len() {
                            argv[i].clone()
                        } else {
                            String::new()
                        }
                    };
                    let v = match typ.as_str() {
                        "int" => Value::Integer(val.parse::<i64>().unwrap_or(0)),
                        "float" => Value::Real(val.parse::<f64>().unwrap_or(0.0)),
                        "bool" => Value::Boolean(val == "true"),
                        _ => Value::String(val),
                    };
                    if rpt {
                        let e = options.entry(k).or_insert_with(|| Value::List(vec![]));
                        if let Value::List(xs) = e {
                            xs.push(v);
                        }
                    } else {
                        options.insert(k, v);
                    }
                }
            }
        } else if a.starts_with('-') && a.len() > 1 {
            let s = a[1..].to_string();
            // treat combined like -abc; set flags or take next as value for option type
            for ch in s.chars() {
                let key = opts_spec
                    .iter()
                    .find(|(_, (short, _, _, _))| short == &ch.to_string())
                    .map(|(name, (..))| name.clone());
                if let Some(name) = key {
                    if let Some((_, typ, rpt, is_flag)) = opts_spec.get(&name).cloned() {
                        if is_flag {
                            flags.insert(name, Value::Boolean(true));
                        } else {
                            i += 1;
                            let val = if i < argv.len() { argv[i].clone() } else { String::new() };
                            let v = match typ.as_str() {
                                "int" => Value::Integer(val.parse::<i64>().unwrap_or(0)),
                                "float" => Value::Real(val.parse::<f64>().unwrap_or(0.0)),
                                "bool" => Value::Boolean(val == "true"),
                                _ => Value::String(val),
                            };
                            if rpt {
                                let e = options.entry(name).or_insert_with(|| Value::List(vec![]));
                                if let Value::List(xs) = e {
                                    xs.push(v);
                                }
                            } else {
                                options.insert(name, v);
                            }
                        }
                    }
                }
            }
        } else {
            rest.push(Value::String(a.clone()));
        }
        i += 1;
    }
    Value::Assoc(std::collections::HashMap::from([
        ("Options".into(), Value::Assoc(options)),
        ("Flags".into(), Value::Assoc(flags)),
        ("Rest".into(), Value::List(rest)),
    ]))
}

fn prompt(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("Prompt".into())), args };
    }
    let msg = to_string_arg(ev, args[0].clone());
    let opts = if args.len() > 1 { match ev.eval(args[1].clone()) { Value::Assoc(m) => m, _ => std::collections::HashMap::new() } } else { std::collections::HashMap::new() };
    ui_backend().lock().unwrap().prompt_string(&msg, &opts)
}

fn confirm(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("Confirm".into())), args };
    }
    let msg = to_string_arg(ev, args[0].clone());
    let opts = if args.len() > 1 { match ev.eval(args[1].clone()) { Value::Assoc(m) => m, _ => std::collections::HashMap::new() } } else { std::collections::HashMap::new() };
    ui_backend().lock().unwrap().confirm(&msg, &opts)
}

fn password_prompt(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("PasswordPrompt".into())), args };
    }
    let msg = to_string_arg(ev, args[0].clone());
    let opts = if args.len() > 1 { match ev.eval(args[1].clone()) { Value::Assoc(m) => m, _ => std::collections::HashMap::new() } } else { std::collections::HashMap::new() };
    ui_backend().lock().unwrap().prompt_password(&msg, &opts)
}

fn prompt_select_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("PromptSelect".into())), args };
    }
    let msg = to_string_arg(ev, args[0].clone());
    let choices = match ev.eval(args[1].clone()) {
        Value::List(xs) => xs
            .into_iter()
            .filter_map(|v| match v {
                Value::String(s) | Value::Symbol(s) => Some((s.clone(), s)),
                Value::Assoc(m) => m
                    .get("name")
                    .and_then(|n| {
                        if let Value::String(ns) | Value::Symbol(ns) = n {
                            Some(ns.clone())
                        } else {
                            None
                        }
                    })
                    .zip(m.get("value").and_then(|n| {
                        if let Value::String(ns) | Value::Symbol(ns) = n {
                            Some(ns.clone())
                        } else {
                            None
                        }
                    })),
                _ => None,
            })
            .collect::<Vec<_>>(),
        _ => vec![],
    };
    let opts = if args.len() > 2 { match ev.eval(args[2].clone()) { Value::Assoc(m) => m, _ => std::collections::HashMap::new() } } else { std::collections::HashMap::new() };
    ui_backend().lock().unwrap().select_one(&msg, &choices, &opts)
}

fn prompt_select_many_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("PromptSelectMany".into())), args };
    }
    let msg = to_string_arg(ev, args[0].clone());
    let choices = match ev.eval(args[1].clone()) {
        Value::List(xs) => xs
            .into_iter()
            .filter_map(|v| match v {
                Value::String(s) | Value::Symbol(s) => Some((s.clone(), s)),
                Value::Assoc(m) => m
                    .get("name")
                    .and_then(|n| match n { Value::String(ns) | Value::Symbol(ns) => Some(ns.clone()), _ => None })
                    .zip(m.get("value").and_then(|n| match n { Value::String(ns) | Value::Symbol(ns) => Some(ns.clone()), _ => None })),
                _ => None,
            })
            .collect::<Vec<_>>(),
        _ => vec![],
    };
    let opts = if args.len() > 2 { match ev.eval(args[2].clone()) { Value::Assoc(m) => m, _ => std::collections::HashMap::new() } } else { std::collections::HashMap::new() };
    ui_backend().lock().unwrap().select_many(&msg, &choices, &opts)
}

fn notify_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("Notify".into())), args };
    }
    let msg = to_string_arg(ev, args[0].clone());
    let opts = if args.len() > 1 { match ev.eval(args[1].clone()) { Value::Assoc(m) => m, _ => std::collections::HashMap::new() } } else { std::collections::HashMap::new() };
    ui_backend().lock().unwrap().notify(&msg, &opts)
}

fn spinner_start_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let msg = if args.is_empty() { String::from("") } else { to_string_arg(ev, args[0].clone()) };
    let opts = if args.len() > 1 { match ev.eval(args[1].clone()) { Value::Assoc(m) => m, _ => std::collections::HashMap::new() } } else { std::collections::HashMap::new() };
    ui_backend().lock().unwrap().spinner_start(&msg, &opts)
}

fn spinner_stop_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::Expr { head: Box::new(Value::Symbol("SpinnerStop".into())), args } }
    let id = match ev.eval(args[0].clone()) { Value::Integer(n) => n, _ => 0 };
    ui_backend().lock().unwrap().spinner_stop(id)
}

#[derive(Clone)]
struct PState {
    #[allow(dead_code)]
    total: i64,
    cur: i64,
}
static PB_REG: OnceLock<Mutex<std::collections::HashMap<i64, PState>>> = OnceLock::new();
static PB_NEXT: OnceLock<AtomicI64> = OnceLock::new();
fn pb_reg() -> &'static Mutex<std::collections::HashMap<i64, PState>> {
    PB_REG.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}
fn pb_next() -> i64 {
    PB_NEXT.get_or_init(|| AtomicI64::new(1)).fetch_add(1, Ordering::Relaxed)
}

// Simple terminal spinner implementation
struct SpinState { stop: std::sync::Arc<std::sync::atomic::AtomicBool> }
static SPIN_REG: OnceLock<Mutex<std::collections::HashMap<i64, SpinState>>> = OnceLock::new();
static SPIN_NEXT: OnceLock<AtomicI64> = OnceLock::new();
fn spin_reg() -> &'static Mutex<std::collections::HashMap<i64, SpinState>> {
    SPIN_REG.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}
fn spin_next() -> i64 { SPIN_NEXT.get_or_init(|| AtomicI64::new(1)).fetch_add(1, Ordering::Relaxed) }
fn spinner_start_thread(msg: &str) -> Value {
    use std::thread;
    use std::time::Duration;
    let id = spin_next();
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop2 = stop.clone();
    let msg_owned = msg.to_string();
    thread::spawn(move || {
        let frames = ["|","/","-","\\"];
        let mut i = 0usize;
        while !stop2.load(std::sync::atomic::Ordering::Relaxed) {
            eprint!("\r{} {}", frames[i % frames.len()], msg_owned);
            let _ = std::io::Write::flush(&mut std::io::stderr());
            i += 1;
            thread::sleep(Duration::from_millis(100));
        }
        eprint!("\r{}\r", " ".repeat(msg_owned.len() + 2));
        let _ = std::io::Write::flush(&mut std::io::stderr());
    });
    spin_reg().lock().unwrap().insert(id, SpinState { stop });
    Value::Integer(id)
}
fn spinner_stop_thread(id: i64) -> Value {
    if let Some(st) = spin_reg().lock().unwrap().remove(&id) {
        st.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        Value::Boolean(true)
    } else {
        Value::Boolean(false)
    }
}

fn progress_bar(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("ProgressBar".into())), args };
    }
    let total = match ev.eval(args[0].clone()) {
        Value::Integer(n) if n > 0 => n,
        _ => 0,
    };
    let opts = if args.len() > 1 { match ev.eval(args[1].clone()) { Value::Assoc(m) => m, _ => std::collections::HashMap::new() } } else { std::collections::HashMap::new() };
    ui_backend().lock().unwrap().progress_create(total, &opts)
}

fn progress_advance(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("ProgressAdvance".into())), args };
    }
    let id = match ev.eval(args[0].clone()) { Value::Integer(n) => n, _ => 0 };
    let n = if args.len() > 1 { match ev.eval(args[1].clone()) { Value::Integer(n) => n, _ => 1 } } else { 1 };
    ui_backend().lock().unwrap().progress_advance(id, n)
}

fn progress_finish(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("ProgressFinish".into())), args };
    }
    let id = match ev.eval(args[0].clone()) { Value::Integer(n) => n, _ => 0 };
    ui_backend().lock().unwrap().progress_finish(id)
}

// ---------------- JSON ----------------
struct JsonOpts {
    pretty: bool,
    sort_keys: bool,
    ensure_ascii: bool,
    trailing_newline: bool,
    indent: Option<Vec<u8>>,
}

fn json_opts_from(ev: &mut Evaluator, v: Option<Value>) -> JsonOpts {
    let mut o = JsonOpts {
        pretty: false,
        sort_keys: false,
        ensure_ascii: false,
        trailing_newline: false,
        indent: None,
    };
    if let Some(Value::Assoc(m)) = v.map(|x| ev.eval(x)) {
        if let Some(Value::Boolean(b)) = m.get("Pretty") {
            o.pretty = *b;
        }
        if let Some(Value::Boolean(b)) = m.get("SortKeys") {
            o.sort_keys = *b;
        }
        if let Some(Value::Boolean(b)) = m.get("EnsureAscii") {
            o.ensure_ascii = *b;
        }
        if let Some(Value::Boolean(b)) = m.get("TrailingNewline") {
            o.trailing_newline = *b;
        }
        if let Some(Value::Integer(n)) = m.get("Indent") {
            let n = (*n).clamp(0, 8) as usize;
            if n > 0 {
                o.indent = Some(vec![b' '; n]);
            }
        }
        if let Some(Value::String(s)) = m.get("Indent") {
            let bytes = s.as_bytes().to_vec();
            if !bytes.is_empty() {
                o.indent = Some(bytes);
            }
        }
    }
    o
}

fn to_json(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("ToJson".into())), args };
    }
    let v = ev.eval(args[0].clone());
    let opts = json_opts_from(ev, args.get(1).cloned());
    let j = value_to_json_opts(&v, opts.sort_keys);
    let mut out: Vec<u8> = Vec::new();
    if opts.pretty {
        let indent = opts.indent.unwrap_or_else(|| vec![b' '; 2]);
        let fmt = PrettyFormatter::with_indent(&indent);
        let mut ser = Serializer::with_formatter(&mut out, fmt);
        if let Err(e) = j.serialize(&mut ser) {
            return failure("IO::json", &format!("ToJson: {}", e));
        }
    } else {
        let mut ser = Serializer::new(&mut out);
        if let Err(e) = j.serialize(&mut ser) {
            return failure("IO::json", &format!("ToJson: {}", e));
        }
    }
    let mut s = String::from_utf8_lossy(&out).to_string();
    if opts.ensure_ascii {
        s = ensure_ascii_json(&s);
    }
    if opts.trailing_newline {
        s.push('\n');
    }
    Value::String(s)
}

fn from_json(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("FromJson".into())), args };
    }
    let s = to_string_arg(ev, args[0].clone());
    match sj::from_str::<sj::Value>(&s) {
        Ok(j) => json_to_value(&j),
        Err(e) => failure("IO::json", &format!("FromJson: {}", e)),
    }
}

// ---------------- YAML ----------------
fn yaml_parse(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("YamlParse".into())), args };
    }
    let s = to_string_arg(ev, args[0].clone());
    match serde_yaml::from_str::<serde_yaml::Value>(&s) {
        Ok(y) => yaml_to_value(&y),
        Err(e) => failure("IO::yaml", &format!("YamlParse: {}", e)),
    }
}

fn yaml_to_value(y: &serde_yaml::Value) -> Value {
    match y {
        serde_yaml::Value::Null => Value::Symbol("Null".into()),
        serde_yaml::Value::Bool(b) => Value::Boolean(*b),
        serde_yaml::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Integer(i)
            } else if let Some(f) = n.as_f64() {
                Value::Real(f)
            } else {
                Value::String(n.to_string())
            }
        }
        serde_yaml::Value::String(s) => Value::String(s.clone()),
        serde_yaml::Value::Sequence(arr) => Value::List(arr.iter().map(yaml_to_value).collect()),
        serde_yaml::Value::Mapping(map) => {
            let mut out = std::collections::HashMap::new();
            for (k, v) in map.iter() {
                let kk = match k {
                    serde_yaml::Value::String(s) => s.clone(),
                    other => format!("{:?}", other),
                };
                out.insert(kk, yaml_to_value(v));
            }
            Value::Assoc(out)
        }
        _ => Value::Symbol("Null".into()),
    }
}

fn yaml_stringify(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("YamlStringify".into())), args };
    }
    let v = ev.eval(args[0].clone());
    let j = value_to_json_opts(&v, false);
    match serde_yaml::to_string(&j) {
        Ok(s) => Value::String(s),
        Err(e) => failure("IO::yaml", &format!("YamlStringify: {}", e)),
    }
}

// ---------------- TOML ----------------
fn toml_parse(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("TomlParse".into())), args };
    }
    let s = to_string_arg(ev, args[0].clone());
    match s.parse::<toml::Value>() {
        Ok(t) => toml_to_value(&t),
        Err(e) => failure("IO::toml", &format!("TomlParse: {}", e)),
    }
}

fn toml_to_value(t: &toml::Value) -> Value {
    match t {
        toml::Value::String(s) => Value::String(s.clone()),
        toml::Value::Integer(i) => Value::Integer(*i),
        toml::Value::Float(f) => Value::Real(*f),
        toml::Value::Boolean(b) => Value::Boolean(*b),
        toml::Value::Datetime(dt) => Value::String(dt.to_string()),
        toml::Value::Array(arr) => Value::List(arr.iter().map(toml_to_value).collect()),
        toml::Value::Table(map) => {
            Value::Assoc(map.iter().map(|(k, v)| (k.clone(), toml_to_value(v))).collect())
        }
    }
}

fn toml_stringify(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("TomlStringify".into())), args };
    }
    let v = ev.eval(args[0].clone());
    // Route via JSON mapping for broad coverage
    let j = value_to_json_opts(&v, true);
    // Convert serde_json::Value to toml::Value recursively
    fn json_to_toml(j: &serde_json::Value) -> toml::Value {
        match j {
            serde_json::Value::Null => toml::Value::String("".into()),
            serde_json::Value::Bool(b) => toml::Value::Boolean(*b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    toml::Value::Integer(i)
                } else if let Some(f) = n.as_f64() {
                    toml::Value::Float(f)
                } else {
                    toml::Value::String(n.to_string())
                }
            }
            serde_json::Value::String(s) => toml::Value::String(s.clone()),
            serde_json::Value::Array(arr) => {
                toml::Value::Array(arr.iter().map(json_to_toml).collect())
            }
            serde_json::Value::Object(map) => {
                let mut tbl = toml::map::Map::new();
                for (k, v) in map.iter() {
                    tbl.insert(k.clone(), json_to_toml(v));
                }
                toml::Value::Table(tbl)
            }
        }
    }
    let t = json_to_toml(&j);
    match toml::to_string_pretty(&t) {
        Ok(s) => Value::String(s),
        Err(e) => failure("IO::toml", &format!("TomlStringify: {}", e)),
    }
}

// ---------------- Bytes / Encoding ----------------
fn bytes_from_value(ev: &mut Evaluator, v: Value) -> Result<Vec<u8>, String> {
    match ev.eval(v) {
        Value::List(items) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                match it {
                    Value::Integer(i) if i >= 0 && i <= 255 => out.push(i as u8),
                    _ => return Err("Bytes must be list of 0..255".into()),
                }
            }
            Ok(out)
        }
        // Support Binary[base64url] fallback
        Value::Expr { head, args } => {
            if let Value::Symbol(sym) = *head {
                if sym == "Binary" && !args.is_empty() {
                    if let Value::String(s) = &args[0] {
                        return base64url_to_bytes(s).map_err(|e| format!("{}", e));
                    }
                }
            }
            Err("Expected bytes".into())
        }
        Value::String(s) => Ok(s.into_bytes()),
        _ => Err("Expected bytes".into()),
    }
}

fn bytes_to_value(bytes: &[u8]) -> Value {
    Value::List(bytes.iter().map(|b| Value::Integer(*b as i64)).collect())
}

fn base64_encode(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("Base64Encode".into())), args };
    }
    let data = match bytes_from_value(ev, args[0].clone()) {
        Ok(b) => b,
        Err(e) => return failure("IO::bytes", &format!("Base64Encode: {}", e)),
    };
    use base64::Engine as _;
    let s = base64::engine::general_purpose::STANDARD.encode(data);
    Value::String(s)
}

fn base64_decode(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("Base64Decode".into())), args };
    }
    let s = to_string_arg(ev, args[0].clone());
    use base64::Engine as _;
    match base64::engine::general_purpose::STANDARD.decode(s) {
        Ok(b) => bytes_to_value(&b),
        Err(e) => failure("IO::bytes", &format!("Base64Decode: {}", e)),
    }
}

fn hex_encode_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("HexEncode".into())), args };
    }
    let data = match bytes_from_value(ev, args[0].clone()) {
        Ok(b) => b,
        Err(e) => return failure("IO::bytes", &format!("HexEncode: {}", e)),
    };
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(data.len() * 2);
    for b in data {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    Value::String(out)
}

fn from_hex(c: u8) -> Result<u8, String> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        _ => Err("invalid hex".into()),
    }
}

fn hex_decode_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("HexDecode".into())), args };
    }
    let s = to_string_arg(ev, args[0].clone());
    if s.len() % 2 != 0 {
        return failure("IO::bytes", "HexDecode: hex length must be even");
    }
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len() / 2);
    for i in (0..bytes.len()).step_by(2) {
        let h = match from_hex(bytes[i]) {
            Ok(x) => x,
            Err(e) => return failure("IO::bytes", &format!("HexDecode: {}", e)),
        };
        let l = match from_hex(bytes[i + 1]) {
            Ok(x) => x,
            Err(e) => return failure("IO::bytes", &format!("HexDecode: {}", e)),
        };
        out.push((h << 4) | l);
    }
    bytes_to_value(&out)
}

fn text_encode(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("TextEncode".into())), args };
    }
    let s = to_string_arg(ev, args[0].clone());
    // Currently only utf-8 supported
    bytes_to_value(s.as_bytes())
}

fn text_decode(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("TextDecode".into())), args };
    }
    let b = match bytes_from_value(ev, args[0].clone()) {
        Ok(x) => x,
        Err(e) => return failure("IO::bytes", &format!("TextDecode: {}", e)),
    };
    match String::from_utf8(b) {
        Ok(s) => Value::String(s),
        Err(_) => failure("IO::bytes", "TextDecode: invalid utf-8"),
    }
}

fn bytes_concat(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("BytesConcat".into())), args };
    }
    let list_v = ev.eval(args[0].clone());
    if let Value::List(chunks) = list_v {
        let mut out: Vec<u8> = Vec::new();
        for c in chunks {
            match bytes_from_value(ev, c) {
                Ok(mut b) => out.append(&mut b),
                Err(e) => return failure("IO::bytes", &format!("BytesConcat: {}", e)),
            }
        }
        return bytes_to_value(&out);
    }
    failure("IO::bytes", "BytesConcat: expected list of byte arrays")
}

fn bytes_slice(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("BytesSlice".into())), args };
    }
    let data = match bytes_from_value(ev, args[0].clone()) {
        Ok(b) => b,
        Err(e) => return failure("IO::bytes", &format!("BytesSlice: {}", e)),
    };
    let start = match ev.eval(args[1].clone()) {
        Value::Integer(i) if i >= 0 => i as usize,
        _ => 0,
    };
    let end = if args.len() > 2 {
        match ev.eval(args[2].clone()) {
            Value::Integer(i) if i >= 0 => (i as usize).min(data.len()),
            _ => data.len(),
        }
    } else {
        data.len()
    };
    let s = if start <= end && start <= data.len() { &data[start..end] } else { &[] };
    bytes_to_value(s)
}

fn bytes_length(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("BytesLength".into())), args };
    }
    match bytes_from_value(ev, args[0].clone()) {
        Ok(b) => Value::Integer(b.len() as i64),
        Err(e) => failure("IO::bytes", &format!("BytesLength: {}", e)),
    }
}

fn base64url_to_bytes(s: &str) -> Result<Vec<u8>, String> {
    use base64::Engine as _;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(s).map_err(|e| e.to_string())
}

fn value_to_json_opts(v: &Value, sort_keys: bool) -> sj::Value {
    match v {
        Value::Integer(n) => sj::Value::Number((*n).into()),
        Value::Real(f) => sj::json!(*f),
        Value::String(s) => sj::Value::String(s.clone()),
        Value::Symbol(s) => {
            if s == "Null" {
                sj::Value::Null
            } else {
                sj::Value::String(s.clone())
            }
        }
        Value::Boolean(b) => sj::Value::Bool(*b),
        Value::List(items) => {
            sj::Value::Array(items.iter().map(|x| value_to_json_opts(x, sort_keys)).collect())
        }
        Value::Assoc(m) => {
            let mut obj = serde_json::Map::new();
            if sort_keys {
                let mut keys: Vec<&String> = m.keys().collect();
                keys.sort();
                for k in keys {
                    obj.insert(k.clone(), value_to_json_opts(m.get(k).unwrap(), sort_keys));
                }
            } else {
                for (k, vv) in m.iter() {
                    obj.insert(k.clone(), value_to_json_opts(vv, sort_keys));
                }
            }
            sj::Value::Object(obj)
        }
        _ => sj::Value::String(lyra_core::pretty::format_value(v)),
    }
}

fn json_to_value(j: &sj::Value) -> Value {
    match j {
        sj::Value::Null => Value::Symbol("Null".into()),
        sj::Value::Bool(b) => Value::Boolean(*b),
        sj::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Integer(i)
            } else if let Some(f) = n.as_f64() {
                Value::Real(f)
            } else {
                Value::String(n.to_string())
            }
        }
        sj::Value::String(s) => Value::String(s.clone()),
        sj::Value::Array(arr) => Value::List(arr.iter().map(json_to_value).collect()),
        sj::Value::Object(map) => {
            Value::Assoc(map.iter().map(|(k, v)| (k.clone(), json_to_value(v))).collect())
        }
    }
}

fn ensure_ascii_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        if ch <= '\u{7F}' {
            out.push(ch);
        } else {
            let mut buf = [0u16; 2];
            for unit in ch.encode_utf16(&mut buf) {
                out.push_str(&format!("\\u{:04X}", unit));
            }
        }
    }
    out
}

// ---------------- CSV ----------------
#[derive(Clone)]
struct CsvOpts {
    delim: char,
    quote: char,
    header: bool,
    eol: &'static str,
    columns: Option<Vec<String>>,
    headers: Option<Vec<String>>,
}

fn csv_opts_from(ev: &mut Evaluator, v: Option<Value>) -> CsvOpts {
    let mut o =
        CsvOpts { delim: ',', quote: '"', header: true, eol: "\n", columns: None, headers: None };
    if let Some(Value::Assoc(m)) = v.map(|x| ev.eval(x)) {
        if let Some(Value::String(s)) | Some(Value::Symbol(s)) = m.get("Delimiter").or_else(|| m.get("delimiter")) {
            if let Some(c) = s.chars().next() {
                o.delim = c;
            }
        }
        if let Some(Value::String(s)) | Some(Value::Symbol(s)) = m.get("Quote").or_else(|| m.get("quote")) {
            if let Some(c) = s.chars().next() {
                o.quote = c;
            }
        }
        if let Some(Value::Boolean(b)) = m.get("Header").or_else(|| m.get("header")) {
            o.header = *b;
        }
        if let Some(Value::String(s)) = m.get("Eol").or_else(|| m.get("eol")) {
            if s == "\r\n" {
                o.eol = "\r\n";
            }
        }
        if let Some(Value::List(cols_v)) = m.get("Columns").or_else(|| m.get("columns")) {
            let cols: Vec<String> = cols_v
                .iter()
                .filter_map(|v| match v {
                    Value::String(s) | Value::Symbol(s) => Some(s.clone()),
                    _ => None,
                })
                .collect();
            if !cols.is_empty() {
                o.columns = Some(cols);
            }
        }
        if let Some(Value::List(cols_v)) = m.get("Headers").or_else(|| m.get("headers")) {
            let cols: Vec<String> = cols_v
                .iter()
                .filter_map(|v| match v {
                    Value::String(s) | Value::Symbol(s) => Some(s.clone()),
                    _ => None,
                })
                .collect();
            if !cols.is_empty() {
                o.headers = Some(cols);
            }
        }
    }
    o
}

fn parse_csv_records(s: &str, opts: &CsvOpts) -> Vec<Vec<String>> {
    let mut out: Vec<Vec<String>> = Vec::new();
    let mut rec: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut it = s.chars().peekable();
    let mut in_q = false;
    while let Some(ch) = it.next() {
        if in_q {
            if ch == opts.quote {
                if let Some(nextc) = it.peek() {
                    if *nextc == opts.quote {
                        field.push(opts.quote);
                        it.next();
                    } else {
                        in_q = false;
                    }
                } else {
                    in_q = false;
                }
            } else {
                field.push(ch);
            }
        } else {
            if ch == opts.quote {
                in_q = true;
            } else if ch == opts.delim {
                rec.push(field.clone());
                field.clear();
            } else if ch == '\n' || ch == '\r' {
                // handle CRLF
                if ch == '\r' {
                    if let Some('\n') = it.peek().copied() {
                        it.next();
                    }
                }
                rec.push(field.clone());
                field.clear();
                out.push(rec);
                rec = Vec::new();
            } else {
                field.push(ch);
            }
        }
    }
    // flush last
    if in_q { /* unbalanced quote: treat as text */ }
    rec.push(field);
    if !rec.is_empty() {
        out.push(rec);
    }
    out
}

fn parse_csv(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("ParseCSV".into())), args };
    }
    let s = to_string_arg(ev, args[0].clone());
    let opts = csv_opts_from(ev, args.get(1).cloned());
    let rows = parse_csv_records(&s, &opts);
    csv_rows_to_values(rows, &opts)
}

fn read_csv(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("ReadCSV".into())), args };
    }
    let path = to_string_arg(ev, args[0].clone());
    let s = match std::fs::read_to_string(&path) {
        Ok(x) => x,
        Err(e) => return failure("IO::csv", &format!("ReadCSV: {}", e)),
    };
    let opts = csv_opts_from(ev, args.get(1).cloned());
    let rows = parse_csv_records(&s, &opts);
    csv_rows_to_values(rows, &opts)
}

fn csv_rows_to_values(rows: Vec<Vec<String>>, opts: &CsvOpts) -> Value {
    if rows.is_empty() {
        return Value::List(vec![]);
    }
    if opts.headers.is_some() || opts.header {
        let headers: Vec<String> =
            if let Some(h) = &opts.headers { h.clone() } else { rows[0].clone() };
        let mut out: Vec<Value> = Vec::new();
        let start_idx = if opts.headers.is_some() { 0 } else { 1 };
        for r in rows.into_iter().skip(start_idx) {
            let mut m: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
            for (i, h) in headers.iter().enumerate() {
                if let Some(cell) = r.get(i) {
                    m.insert(h.clone(), Value::String(cell.clone()));
                } else {
                    m.insert(h.clone(), Value::Symbol("Null".into()));
                }
            }
            out.push(Value::Assoc(m));
        }
        Value::List(out)
    } else {
        Value::List(
            rows.into_iter()
                .map(|r| Value::List(r.into_iter().map(Value::String).collect()))
                .collect(),
        )
    }
}

fn render_csv(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("RenderCSV".into())), args };
    }
    let data = ev.eval(args[0].clone());
    let opts = csv_opts_from(ev, args.get(1).cloned());
    match render_csv_string(&data, &opts) {
        Ok(s) => Value::String(s),
        Err(e) => failure("IO::csv", &e),
    }
}

fn write_csv(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("WriteCSV".into())), args };
    }
    let path = to_string_arg(ev, args[0].clone());
    let data = ev.eval(args[1].clone());
    let opts = csv_opts_from(ev, args.get(2).cloned());
    match render_csv_string(&data, &opts) {
        Ok(s) => match std::fs::write(&path, s) {
            Ok(_) => Value::Boolean(true),
            Err(e) => failure("IO::csv", &format!("WriteCSV: {}", e)),
        },
        Err(e) => failure("IO::csv", &e),
    }
}

fn render_csv_string(data: &Value, opts: &CsvOpts) -> Result<String, String> {
    fn esc(cell: &str, delim: char, quote: char) -> String {
        let must_quote = cell.contains(delim)
            || cell.contains(quote)
            || cell.contains('\n')
            || cell.contains('\r');
        if must_quote {
            format!("{}{}{}", quote, cell.replace(quote, &format!("{}{}", quote, quote)), quote)
        } else {
            cell.to_string()
        }
    }
    match data {
        Value::List(rows) => {
            if rows.is_empty() {
                return Ok(String::new());
            }
            // Decide format by inspecting first row
            match &rows[0] {
                Value::Assoc(first) => {
                    // columns from opts or derived from first row (sorted for stability)
                    let cols: Vec<String> = if let Some(cols) = &opts.columns {
                        cols.clone()
                    } else {
                        let mut v: Vec<String> = first.keys().cloned().collect();
                        v.sort();
                        v
                    };
                    let mut out = String::new();
                    if opts.header {
                        out.push_str(
                            &cols
                                .iter()
                                .map(|h| esc(h, opts.delim, opts.quote))
                                .collect::<Vec<_>>()
                                .join(&opts.delim.to_string()),
                        );
                        out.push_str(opts.eol);
                    }
                    for r in rows {
                        if let Value::Assoc(m) = r {
                            let cells: Vec<String> = cols
                                .iter()
                                .map(|k| {
                                    let v =
                                        m.get(k).cloned().unwrap_or(Value::Symbol("Null".into()));
                                    let s = match v {
                                        Value::String(s) => s,
                                        Value::Symbol(ref s) if s == "Null" => String::new(),
                                        _ => lyra_core::pretty::format_value(&v),
                                    };
                                    esc(&s, opts.delim, opts.quote)
                                })
                                .collect();
                            out.push_str(&cells.join(&opts.delim.to_string()));
                            out.push_str(opts.eol);
                        }
                    }
                    Ok(out)
                }
                Value::List(_) => {
                    let mut out = String::new();
                    for r in rows {
                        if let Value::List(cells_v) = r {
                            let cells: Vec<String> = cells_v
                                .iter()
                                .map(|v| match v {
                                    Value::String(s) => esc(s, opts.delim, opts.quote),
                                    _ => esc(
                                        &lyra_core::pretty::format_value(v),
                                        opts.delim,
                                        opts.quote,
                                    ),
                                })
                                .collect();
                            out.push_str(&cells.join(&opts.delim.to_string()));
                            out.push_str(opts.eol);
                        }
                    }
                    Ok(out)
                }
                _ => Err("RenderCSV: expected list of assoc or list of lists".into()),
            }
        }
        _ => Err("RenderCSV: expected list".into()),
    }
}
