use crate::register_if;
#[cfg(feature = "tools")]
use crate::tool_spec;
#[cfg(feature = "tools")]
use crate::tools::add_specs;
use lyra_core::value::Value;
use lyra_parser::Parser;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;

/// Register core string utilities: split/join, trim, case changes,
/// predicates, replace, format/interpolate, and URL/HTML helpers.
pub fn register_string(ev: &mut Evaluator) {
    // Maintain StringLength for compatibility (aliases generic Length on strings)
    ev.register("StringLength", string_length as NativeFn, Attributes::LISTABLE);
    // Core string helpers
    ev.register("ToUpper", to_upper as NativeFn, Attributes::LISTABLE);
    ev.register("ToLower", to_lower as NativeFn, Attributes::LISTABLE);
    ev.register("StringJoin", string_join as NativeFn, Attributes::empty());
    ev.register("StringJoinWith", string_join_with as NativeFn, Attributes::empty());
    ev.register("StringTrim", string_trim as NativeFn, Attributes::LISTABLE);
    ev.register("StringTrimLeft", string_trim_left as NativeFn, Attributes::LISTABLE);
    ev.register("StringTrimRight", string_trim_right as NativeFn, Attributes::LISTABLE);
    ev.register("StringTrimPrefix", string_trim_prefix as NativeFn, Attributes::LISTABLE);
    ev.register("StringTrimSuffix", string_trim_suffix as NativeFn, Attributes::LISTABLE);
    ev.register("StringTrimChars", string_trim_chars as NativeFn, Attributes::LISTABLE);
    ev.register("StringContains", string_contains as NativeFn, Attributes::empty());
    // Canonical Split and legacy alias
    ev.register("Split", string_split as NativeFn, Attributes::empty());
    ev.register("StringSplit", string_split as NativeFn, Attributes::empty());
    ev.register("SplitLines", split_lines as NativeFn, Attributes::LISTABLE);
    ev.register("JoinLines", join_lines as NativeFn, Attributes::empty());
    ev.register("StartsWith", starts_with as NativeFn, Attributes::LISTABLE);
    ev.register("EndsWith", ends_with as NativeFn, Attributes::LISTABLE);
    // Q-suffixed predicate aliases
    ev.register("StartsWithQ", starts_with as NativeFn, Attributes::LISTABLE);
    ev.register("EndsWithQ", ends_with as NativeFn, Attributes::LISTABLE);
    // String-specific replace APIs (do not shadow core Replace/ReplaceFirst)
    ev.register("StringReplace", string_replace as NativeFn, Attributes::empty());
    ev.register("StringReplaceFirst", string_replace_first as NativeFn, Attributes::empty());
    ev.register("StringReverse", string_reverse as NativeFn, Attributes::LISTABLE);
    ev.register("StringPadLeft", string_pad_left as NativeFn, Attributes::LISTABLE);
    ev.register("StringPadRight", string_pad_right as NativeFn, Attributes::LISTABLE);
    ev.register("StringSlice", string_slice as NativeFn, Attributes::LISTABLE);
    ev.register("IndexOf", index_of as NativeFn, Attributes::LISTABLE);
    ev.register("LastIndexOf", last_index_of as NativeFn, Attributes::LISTABLE);
    ev.register("StringRepeat", string_repeat as NativeFn, Attributes::LISTABLE);
    ev.register("IsBlank", is_blank as NativeFn, Attributes::LISTABLE);
    ev.register("BlankQ", is_blank as NativeFn, Attributes::LISTABLE);
    ev.register("Capitalize", capitalize as NativeFn, Attributes::LISTABLE);
    ev.register("TitleCase", title_case as NativeFn, Attributes::LISTABLE);
    ev.register("EqualsIgnoreCase", equals_ignore_case as NativeFn, Attributes::LISTABLE);
    ev.register("StringChars", string_chars as NativeFn, Attributes::LISTABLE);
    ev.register("StringFromChars", string_from_chars as NativeFn, Attributes::empty());
    ev.register("StringInterpolate", string_interpolate as NativeFn, Attributes::LISTABLE);
    ev.register("StringInterpolateWith", string_interpolate_with as NativeFn, Attributes::empty());
    ev.register("StringFormat", string_format as NativeFn, Attributes::empty());
    ev.register("StringFormatMap", string_format_map as NativeFn, Attributes::empty());
    ev.register("TemplateRender", template_render as NativeFn, Attributes::empty());
    // HTML/XML templating helpers
    ev.register("HtmlTemplate", html_template as NativeFn, Attributes::empty());
    ev.register("HtmlTemplateCompile", html_template_compile as NativeFn, Attributes::empty());
    ev.register("HtmlTemplateRender", html_template_render as NativeFn, Attributes::empty());
    ev.register("HtmlAttr", html_attr_fn as NativeFn, Attributes::LISTABLE);
    ev.register("SafeHtml", safe_html_fn as NativeFn, Attributes::LISTABLE);
    ev.register("HtmlEscape", html_escape_fn as NativeFn, Attributes::LISTABLE);
    ev.register("HtmlUnescape", html_unescape_fn as NativeFn, Attributes::LISTABLE);
    ev.register("UrlEncode", url_encode_fn as NativeFn, Attributes::LISTABLE);
    ev.register("UrlDecode", url_decode_fn as NativeFn, Attributes::LISTABLE);
    // (Legacy URLEncode/URLDecode removed pre-release; use UrlEncode/UrlDecode)
    ev.register("JsonEscape", json_escape_fn as NativeFn, Attributes::LISTABLE);
    ev.register("JsonUnescape", json_unescape_fn as NativeFn, Attributes::LISTABLE);
    ev.register("UrlFormEncode", url_form_encode_fn as NativeFn, Attributes::LISTABLE);
    ev.register("UrlFormDecode", url_form_decode_fn as NativeFn, Attributes::LISTABLE);
    ev.register("Slugify", slugify_fn as NativeFn, Attributes::LISTABLE);
    ev.register("StringTruncate", string_truncate_fn as NativeFn, Attributes::LISTABLE);
    ev.register("CamelCase", camel_case_fn as NativeFn, Attributes::LISTABLE);
    ev.register("SnakeCase", snake_case_fn as NativeFn, Attributes::LISTABLE);
    ev.register("KebabCase", kebab_case_fn as NativeFn, Attributes::LISTABLE);
    // Regex helpers
    ev.register("RegexMatch", regex_match_fn as NativeFn, Attributes::LISTABLE);
    // Predicate aliases for clarity/consistency
    ev.register("RegexIsMatch", regex_match_fn as NativeFn, Attributes::LISTABLE);
    ev.register("RegexMatchQ", regex_match_fn as NativeFn, Attributes::LISTABLE);
    ev.register("RegexFind", regex_find_fn as NativeFn, Attributes::empty());
    ev.register("RegexFindAll", regex_find_all_fn as NativeFn, Attributes::empty());
    ev.register("RegexReplace", regex_replace_fn as NativeFn, Attributes::empty());
    ev.register("RegexSplit", regex_split_fn as NativeFn, Attributes::empty());
    ev.register("RegexGroups", regex_groups_fn as NativeFn, Attributes::empty());
    ev.register("RegexCaptureNames", regex_capture_names_fn as NativeFn, Attributes::empty());
    // Date/time helpers
    ev.register("ParseDate", parse_date_fn as NativeFn, Attributes::empty());
    ev.register("FormatDate", format_date_fn as NativeFn, Attributes::empty());
    ev.register("DateDiff", date_diff_fn as NativeFn, Attributes::empty());
    // Tier1 Unicode helpers (stubs)
    ev.register("NormalizeUnicode", normalize_unicode as NativeFn, Attributes::LISTABLE);
    ev.register("CaseFold", case_fold as NativeFn, Attributes::LISTABLE);
    ev.register("Transliterate", transliterate as NativeFn, Attributes::LISTABLE);
    ev.register("RemoveDiacritics", remove_diacritics as NativeFn, Attributes::LISTABLE);

    #[cfg(feature = "tools")]
    add_specs(vec![
        tool_spec!("ToUpper", summary: "Uppercase string", params: ["s"], tags: ["string","text"], examples: [lyra_core::value::Value::String("ToUpper[\"hi\"]  ==> \"HI\"".into())]),
        tool_spec!("ToLower", summary: "Lowercase string", params: ["s"], tags: ["string","text"], examples: [lyra_core::value::Value::String("ToLower[\"Hi\"]  ==> \"hi\"".into())]),
        tool_spec!("StringJoin", summary: "Concatenate list of parts", params: ["parts"], tags: ["string","text"], examples: [lyra_core::value::Value::String("StringJoin[{\"a\",\"b\"}]  ==> \"ab\"".into())]),
        // String helpers
        tool_spec!("Split", summary: "Split string by separator", params: ["s","sep?"], tags: ["string","text"]),
        tool_spec!("StringContains", summary: "Does string contain substring?", params: ["s","substr"], tags: ["string","predicate"]),
        tool_spec!("StartsWith", summary: "Starts with prefix?", params: ["s","prefix"], tags: ["string","predicate"]),
        tool_spec!("EndsWith", summary: "Ends with suffix?", params: ["s","suffix"], tags: ["string","predicate"]),
        tool_spec!("StringReplace", summary: "Replace all occurrences", params: ["s","from","to"], tags: ["string","replace"]),
        tool_spec!("StringReplaceFirst", summary: "Replace first occurrence", params: ["s","from","to"], tags: ["string","replace"]),
        tool_spec!("StringTrim", summary: "Trim whitespace from both ends", params: ["s"], tags: ["string","text"]),
        tool_spec!("UrlEncode", summary: "Percent-encode string", params: ["s"], tags: ["string","url"]),
        tool_spec!("UrlDecode", summary: "Decode percent-encoding", params: ["s"], tags: ["string","url"]),
        // URLEncode/URLDecode legacy alias docs removed (use UrlEncode/UrlDecode)
        tool_spec!("HtmlEscape", summary: "Escape HTML entities", params: ["s"], tags: ["string","html"]),
        tool_spec!("HtmlUnescape", summary: "Unescape HTML entities", params: ["s"], tags: ["string","html"]),
        tool_spec!("JsonEscape", summary: "Escape for JSON string literal", params: ["s"], tags: ["string","json"]),
        tool_spec!("JsonUnescape", summary: "Unescape JSON string literal", params: ["s"], tags: ["string","json"]),
        tool_spec!("RegexMatch", summary: "Does regex match string? (Boolean)", params: ["pattern","s"], tags: ["string","regex","predicate"]),
        tool_spec!("RegexMatchQ", summary: "Alias: regex predicate (Boolean)", params: ["pattern","s"], tags: ["string","regex","predicate"]),
        tool_spec!("RegexFind", summary: "Find first regex match", params: ["pattern","s"], tags: ["string","regex"]),
        tool_spec!("RegexFindAll", summary: "Find all regex matches", params: ["pattern","s"], tags: ["string","regex"]),
        tool_spec!("RegexReplace", summary: "Replace by regex (fn or string)", params: ["pattern","s","repl"], tags: ["string","regex"]),
        tool_spec!("RegexSplit", summary: "Split string by regex pattern", params: ["pattern","s"], tags: ["string","regex"], examples: [Value::String("RegexSplit[\",\", \"a,b,c\"]  ==> {\"a\",\"b\",\"c\"}".into())]),
        tool_spec!("RegexGroups", summary: "Capture groups of first match", params: ["pattern","s"], tags: ["string","regex"], examples: [Value::String("RegexGroups[\"(a)(b)\", \"ab\"]  ==> {\"a\",\"b\"}".into())]),
        tool_spec!("RegexCaptureNames", summary: "Named capture group order", params: ["pattern"], tags: ["string","regex"], examples: [Value::String("RegexCaptureNames[\"(?P<x>a)(?P<y>b)\"]  ==> {\"x\",\"y\"}".into())]),
        tool_spec!("NormalizeUnicode", summary: "Normalize to NFC/NFD/NFKC/NFKD", params: ["s","form?"], tags: ["string","unicode"]),
        tool_spec!("CaseFold", summary: "Unicode case-fold", params: ["s"], tags: ["string","unicode"]),
        tool_spec!("Transliterate", summary: "Transliterate to ASCII (stub)", params: ["s"], tags: ["string","unicode"]),
        tool_spec!("RemoveDiacritics", summary: "Strip diacritics (stub)", params: ["s"], tags: ["string","unicode"]),
        tool_spec!("TemplateRender", summary: "Render Mustache-like template", params: ["template","data","opts?"], tags: ["string","template"], examples: [Value::String("TemplateRender[\"Hello {{name}}!\", <|\"name\"->\"Lyra\"|>]  ==> \"Hello Lyra!\"".into())]),
        tool_spec!("HtmlTemplate", summary: "Render HTML/XML template with data and options", params: ["templateOrPath","data","opts?"], tags: ["string","template","html","xml"], examples: [
            Value::String("HtmlTemplate[\\\"<b>{{name}}</b>\\\", <|name->\\\"X\\\"|>]  ==> \\\"<b>X</b>\\\"".into()),
            Value::String("HtmlTemplate[\\\"{{> Header <|title->\\\\\\\"Docs\\\\\\\"|>}}\\\", <||>, <|Partials-><|\\\"Header\\\"->\\\"<header>{{title}}</header>\\\"|>|>]".into()),
            Value::String("HtmlTemplate[\\\"{{< Button <|label->\\\\\\\"Go\\\\\\\", href->\\\\\\\"/a?b=1&c=2\\\\\\\"|>}}\\\", <||>, <|Components-><|\\\"Button\\\"->\\\"<a href=\\\\\\\"{{href|UrlEncode}}\\\\\\\" class=\\\\\\\"btn\\\\\\\">{{label}}</a>\\\"|>|>]".into()),
            Value::String("HtmlTemplate[\\\"{{#block \\\\\\\"content\\\\\\\"}}<p>Hello</p>{{/block}}\\\", <|title->\\\"Home\\\"|>, <|Layout->\\\"<html><head><title>{{title}}</title></head><body>{{yield \\\\\\\"content\\\\\\\"}}</body></html>\\\"|>]".into()),
        ]),
        tool_spec!("HtmlTemplateCompile", summary: "Precompile template (returns handle)", params: ["templateOrPath","opts?"], tags: ["string","template","html"], examples: [Value::String("t := HtmlTemplateCompile[\"<i>{{msg}}</i>\"]; HtmlTemplateRender[t, <|msg->\"hi\"|>]".into())]),
        tool_spec!("HtmlTemplateRender", summary: "Render compiled template with data", params: ["handle","data","opts?"], tags: ["string","template","html"]),
        tool_spec!("HtmlAttr", summary: "Escape for HTML attribute context", params: ["s"], tags: ["string","html"], examples: [Value::String("HtmlAttr[\\\"a&b\\\"]  ==> \\\"a&amp;b\\\"".into())]),
        tool_spec!("SafeHtml", summary: "Mark string as safe HTML (no escaping)", params: ["s"], tags: ["string","html"], examples: [Value::String("SafeHtml[\\\"<strong>x</strong>\\\"]  ==> <|__type->\\\"SafeHtml\\\"|>".into())]),
        tool_spec!("Slugify", summary: "URL-friendly slug from string", params: ["s"], tags: ["string","url"]),
    ]);
}

/// Conditionally register string utilities based on `pred`.
pub fn register_string_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str) -> bool) {
    register_if(ev, pred, "ToUpper", to_upper as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "ToLower", to_lower as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "StringJoin", string_join as NativeFn, Attributes::empty());
    register_if(ev, pred, "StringJoinWith", string_join_with as NativeFn, Attributes::empty());
    register_if(ev, pred, "StringTrim", string_trim as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "StringTrimLeft", string_trim_left as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "StringTrimRight", string_trim_right as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "StringTrimPrefix", string_trim_prefix as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "StringTrimSuffix", string_trim_suffix as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "StringTrimChars", string_trim_chars as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "StringContains", string_contains as NativeFn, Attributes::empty());
    // Split already registered as canonical
    register_if(ev, pred, "SplitLines", split_lines as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "JoinLines", join_lines as NativeFn, Attributes::empty());
    register_if(ev, pred, "StartsWith", starts_with as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "EndsWith", ends_with as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "StartsWithQ", starts_with as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "EndsWithQ", ends_with as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "StringReplace", string_replace as NativeFn, Attributes::empty());
    register_if(
        ev,
        pred,
        "StringReplaceFirst",
        string_replace_first as NativeFn,
        Attributes::empty(),
    );
    register_if(ev, pred, "StringReverse", string_reverse as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "StringPadLeft", string_pad_left as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "StringPadRight", string_pad_right as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "StringSlice", string_slice as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "IndexOf", index_of as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "LastIndexOf", last_index_of as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "StringRepeat", string_repeat as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "IsBlank", is_blank as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "BlankQ", is_blank as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Capitalize", capitalize as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "TitleCase", title_case as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "EqualsIgnoreCase", equals_ignore_case as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "StringChars", string_chars as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "StringFromChars", string_from_chars as NativeFn, Attributes::empty());
    register_if(
        ev,
        pred,
        "StringInterpolate",
        string_interpolate as NativeFn,
        Attributes::LISTABLE,
    );
    register_if(
        ev,
        pred,
        "StringInterpolateWith",
        string_interpolate_with as NativeFn,
        Attributes::empty(),
    );
    register_if(ev, pred, "StringFormat", string_format as NativeFn, Attributes::empty());
    register_if(ev, pred, "StringFormatMap", string_format_map as NativeFn, Attributes::empty());
    register_if(ev, pred, "TemplateRender", template_render as NativeFn, Attributes::empty());
    register_if(ev, pred, "HtmlTemplate", html_template as NativeFn, Attributes::empty());
    register_if(ev, pred, "HtmlTemplateCompile", html_template_compile as NativeFn, Attributes::empty());
    register_if(ev, pred, "HtmlTemplateRender", html_template_render as NativeFn, Attributes::empty());
    register_if(ev, pred, "HtmlAttr", html_attr_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "SafeHtml", safe_html_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "HtmlEscape", html_escape_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "HtmlUnescape", html_unescape_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "UrlEncode", url_encode_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "UrlDecode", url_decode_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "JsonEscape", json_escape_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "JsonUnescape", json_unescape_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "UrlFormEncode", url_form_encode_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "UrlFormDecode", url_form_decode_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Slugify", slugify_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "StringTruncate", string_truncate_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "CamelCase", camel_case_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "SnakeCase", snake_case_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "KebabCase", kebab_case_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "RegexMatch", regex_match_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "RegexIsMatch", regex_match_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "RegexMatchQ", regex_match_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "RegexFind", regex_find_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "RegexFindAll", regex_find_all_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "RegexReplace", regex_replace_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "RegexSplit", regex_split_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "RegexGroups", regex_groups_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "RegexCaptureNames", regex_capture_names_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "ParseDate", parse_date_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "FormatDate", format_date_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "DateDiff", date_diff_fn as NativeFn, Attributes::empty());
}

// Bring NativeFn alias into scope by mirroring lyra-runtime's type
type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

fn string_length(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s)] => Value::Integer(s.chars().count() as i64),
        [other] => match ev.eval(other.clone()) {
            Value::String(s) => Value::Integer(s.chars().count() as i64),
            v => {
                Value::Expr { head: Box::new(Value::Symbol("StringLength".into())), args: vec![v] }
            }
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("StringLength".into())), args },
    }
}

fn to_upper(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s)] => Value::String(s.to_uppercase()),
        [other] => match ev.eval(other.clone()) {
            Value::String(s) => Value::String(s.to_uppercase()),
            v => Value::Expr { head: Box::new(Value::Symbol("ToUpper".into())), args: vec![v] },
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("ToUpper".into())), args },
    }
}

fn to_lower(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s)] => Value::String(s.to_lowercase()),
        [other] => match ev.eval(other.clone()) {
            Value::String(s) => Value::String(s.to_lowercase()),
            v => Value::Expr { head: Box::new(Value::Symbol("ToLower".into())), args: vec![v] },
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("ToLower".into())), args },
    }
}

fn string_join(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::List(parts)] => {
            let mut out = String::new();
            for p in parts {
                match ev.eval(p.clone()) {
                    Value::String(s) => out.push_str(&s),
                    v => out.push_str(&lyra_core::pretty::format_value(&v)),
                }
            }
            Value::String(out)
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("StringJoin".into())), args },
    }
}

fn string_join_with(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(sep), Value::List(parts)] => {
            let mut out = String::new();
            let mut first = true;
            for p in parts {
                if !first {
                    out.push_str(sep);
                }
                first = false;
                match ev.eval(p.clone()) {
                    Value::String(s) => out.push_str(&s),
                    v => out.push_str(&lyra_core::pretty::format_value(&v)),
                }
            }
            Value::String(out)
        }
        [a, b] => {
            let aa = ev.eval(a.clone());
            let bb = ev.eval(b.clone());
            match (aa, bb) {
                (Value::String(sep), Value::List(parts)) => {
                    let mut out = String::new();
                    let mut first = true;
                    for p in parts {
                        if !first {
                            out.push_str(&sep);
                        }
                        first = false;
                        match ev.eval(p.clone()) {
                            Value::String(s) => out.push_str(&s),
                            v => out.push_str(&lyra_core::pretty::format_value(&v)),
                        }
                    }
                    Value::String(out)
                }
                (aa, bb) => Value::Expr {
                    head: Box::new(Value::Symbol("StringJoinWith".into())),
                    args: vec![aa, bb],
                },
            }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("StringJoinWith".into())), args },
    }
}

fn string_trim(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s)] => Value::String(s.trim().to_string()),
        [other] => match ev.eval(other.clone()) {
            Value::String(s) => Value::String(s.trim().to_string()),
            v => Value::Expr { head: Box::new(Value::Symbol("StringTrim".into())), args: vec![v] },
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("StringTrim".into())), args },
    }
}

fn string_trim_left(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s)] => Value::String(s.trim_start().to_string()),
        [other] => match ev.eval(other.clone()) {
            Value::String(s) => Value::String(s.trim_start().to_string()),
            v => Value::Expr {
                head: Box::new(Value::Symbol("StringTrimLeft".into())),
                args: vec![v],
            },
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("StringTrimLeft".into())), args },
    }
}

fn string_trim_right(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s)] => Value::String(s.trim_end().to_string()),
        [other] => match ev.eval(other.clone()) {
            Value::String(s) => Value::String(s.trim_end().to_string()),
            v => Value::Expr {
                head: Box::new(Value::Symbol("StringTrimRight".into())),
                args: vec![v],
            },
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("StringTrimRight".into())), args },
    }
}

fn string_trim_prefix(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s), Value::String(prefix)] => {
            if s.starts_with(prefix) {
                Value::String(s[prefix.len()..].to_string())
            } else {
                Value::String(s.clone())
            }
        }
        [a, b] => {
            let aa = ev.eval(a.clone());
            let bb = ev.eval(b.clone());
            string_trim_prefix(ev, vec![aa, bb])
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("StringTrimPrefix".into())), args },
    }
}

fn string_trim_suffix(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s), Value::String(suf)] => {
            if s.ends_with(suf) {
                Value::String(s[..s.len() - suf.len()].to_string())
            } else {
                Value::String(s.clone())
            }
        }
        [a, b] => {
            let aa = ev.eval(a.clone());
            let bb = ev.eval(b.clone());
            string_trim_suffix(ev, vec![aa, bb])
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("StringTrimSuffix".into())), args },
    }
}

// -------------- Regex helpers --------------
fn compile_regex(pat: &str) -> Result<regex::Regex, String> {
    regex::Regex::new(pat).map_err(|e| e.to_string())
}

fn regex_match_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [s, p] => {
            let ss = match ev.eval(s.clone()) {
                Value::String(x) => x,
                v => {
                    return Value::Expr {
                        head: Box::new(Value::Symbol("RegexMatch".into())),
                        args: vec![v, p.clone()],
                    }
                }
            };
            let pp = match ev.eval(p.clone()) {
                Value::String(x) | Value::Symbol(x) => x,
                v => {
                    return Value::Expr {
                        head: Box::new(Value::Symbol("RegexMatch".into())),
                        args: vec![Value::String(ss), v],
                    }
                }
            };
            match compile_regex(&pp) {
                Ok(re) => Value::Boolean(re.is_match(&ss)),
                Err(_) => Value::Expr {
                    head: Box::new(Value::Symbol("RegexMatch".into())),
                    args: vec![Value::String(ss), Value::String(pp)],
                },
            }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("RegexMatch".into())), args },
    }
}

fn regex_find_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [s, p] => {
            let ss = match ev.eval(s.clone()) {
                Value::String(x) => x,
                v => {
                    return Value::Expr {
                        head: Box::new(Value::Symbol("RegexFind".into())),
                        args: vec![v, p.clone()],
                    }
                }
            };
            let pp = match ev.eval(p.clone()) {
                Value::String(x) | Value::Symbol(x) => x,
                v => {
                    return Value::Expr {
                        head: Box::new(Value::Symbol("RegexFind".into())),
                        args: vec![Value::String(ss), v],
                    }
                }
            };
            match compile_regex(&pp) {
                Ok(re) => re
                    .find(&ss)
                    .map(|m| Value::String(m.as_str().to_string()))
                    .unwrap_or(Value::Symbol("Null".into())),
                Err(_) => Value::Expr {
                    head: Box::new(Value::Symbol("RegexFind".into())),
                    args: vec![Value::String(ss), Value::String(pp)],
                },
            }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("RegexFind".into())), args },
    }
}

fn regex_find_all_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [s, p] => {
            let ss = match ev.eval(s.clone()) {
                Value::String(x) => x,
                v => {
                    return Value::Expr {
                        head: Box::new(Value::Symbol("RegexFindAll".into())),
                        args: vec![v, p.clone()],
                    }
                }
            };
            let pp = match ev.eval(p.clone()) {
                Value::String(x) | Value::Symbol(x) => x,
                v => {
                    return Value::Expr {
                        head: Box::new(Value::Symbol("RegexFindAll".into())),
                        args: vec![Value::String(ss), v],
                    }
                }
            };
            match compile_regex(&pp) {
                Ok(re) => Value::List(
                    re.find_iter(&ss).map(|m| Value::String(m.as_str().to_string())).collect(),
                ),
                Err(_) => Value::Expr {
                    head: Box::new(Value::Symbol("RegexFindAll".into())),
                    args: vec![Value::String(ss), Value::String(pp)],
                },
            }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("RegexFindAll".into())), args },
    }
}

fn regex_replace_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [s, p, r] => {
            let ss = match ev.eval(s.clone()) {
                Value::String(x) => x,
                v => {
                    return Value::Expr {
                        head: Box::new(Value::Symbol("RegexReplace".into())),
                        args: vec![v, p.clone(), r.clone()],
                    }
                }
            };
            let pp = match ev.eval(p.clone()) {
                Value::String(x) | Value::Symbol(x) => x,
                v => {
                    return Value::Expr {
                        head: Box::new(Value::Symbol("RegexReplace".into())),
                        args: vec![Value::String(ss), v, r.clone()],
                    }
                }
            };
            let rr = match ev.eval(r.clone()) {
                Value::String(x) => x,
                v => {
                    return Value::Expr {
                        head: Box::new(Value::Symbol("RegexReplace".into())),
                        args: vec![Value::String(ss), Value::String(pp), v],
                    }
                }
            };
            match compile_regex(&pp) {
                Ok(re) => Value::String(re.replace_all(&ss, rr.as_str()).to_string()),
                Err(_) => Value::Expr {
                    head: Box::new(Value::Symbol("RegexReplace".into())),
                    args: vec![Value::String(ss), Value::String(pp), Value::String(rr)],
                },
            }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("RegexReplace".into())), args },
    }
}

fn regex_split_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [s, p] => {
            let ss = match ev.eval(s.clone()) { Value::String(x) => x, v => return Value::Expr { head: Box::new(Value::Symbol("RegexSplit".into())), args: vec![v, p.clone()] } };
            let pp = match ev.eval(p.clone()) { Value::String(x) | Value::Symbol(x) => x, v => return Value::Expr { head: Box::new(Value::Symbol("RegexSplit".into())), args: vec![Value::String(ss), v] } };
            match compile_regex(&pp) {
                Ok(re) => Value::List(re.split(&ss).map(|t| Value::String(t.to_string())).collect()),
                Err(_) => Value::Expr { head: Box::new(Value::Symbol("RegexSplit".into())), args: vec![Value::String(ss), Value::String(pp)] },
            }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("RegexSplit".into())), args },
    }
}

fn regex_groups_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [s, p] => {
            let ss = match ev.eval(s.clone()) { Value::String(x) => x, v => return Value::Expr { head: Box::new(Value::Symbol("RegexGroups".into())), args: vec![v, p.clone()] } };
            let pp = match ev.eval(p.clone()) { Value::String(x) | Value::Symbol(x) => x, v => return Value::Expr { head: Box::new(Value::Symbol("RegexGroups".into())), args: vec![Value::String(ss), v] } };
            match compile_regex(&pp) {
                Ok(re) => {
                    if let Some(caps) = re.captures(&ss) { Value::List(caps.iter().skip(1).map(|m| m.map(|mm| Value::String(mm.as_str().to_string())).unwrap_or(Value::Symbol("Null".into()))).collect()) } else { Value::Symbol("Null".into()) }
                }
                Err(_) => Value::Expr { head: Box::new(Value::Symbol("RegexGroups".into())), args: vec![Value::String(ss), Value::String(pp)] },
            }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("RegexGroups".into())), args },
    }
}

fn regex_capture_names_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::Expr { head: Box::new(Value::Symbol("RegexCaptureNames".into())), args }; }
    let pat = match ev.eval(args[0].clone()) { Value::String(x) | Value::Symbol(x) => x, v => return Value::Expr { head: Box::new(Value::Symbol("RegexCaptureNames".into())), args: vec![v] } };
    match compile_regex(&pat) {
        Ok(re) => Value::List(re.capture_names().filter_map(|n| n.map(|s| Value::String(s.to_string()))).collect()),
        Err(_) => Value::Expr { head: Box::new(Value::Symbol("RegexCaptureNames".into())), args: vec![Value::String(pat)] },
    }
}

// -------------- Date/time helpers --------------
fn parse_date_flexible(s: &str) -> Option<i64> {
    // Try RFC3339
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return Some(dt.timestamp());
    }
    // Try RFC2822
    if let Ok(dt) = chrono::DateTime::parse_from_rfc2822(s) {
        return Some(dt.timestamp());
    }
    // Common formats
    if let Ok(ndt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
        return Some(
            chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(ndt, chrono::Utc)
                .timestamp(),
        );
    }
    if let Ok(nd) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return Some(
            chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(
                nd.and_hms_opt(0, 0, 0)?,
                chrono::Utc,
            )
            .timestamp(),
        );
    }
    None
}

#[allow(deprecated)]
fn parse_date_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [s] => match ev.eval(s.clone()) {
            Value::String(x) => {
                parse_date_flexible(&x).map(Value::Integer).unwrap_or(Value::Symbol("Null".into()))
            }
            v => Value::Expr { head: Box::new(Value::Symbol("ParseDate".into())), args: vec![v] },
        },
        [s, fmt] => {
            let ss = match ev.eval(s.clone()) {
                Value::String(x) => x,
                v => {
                    return Value::Expr {
                        head: Box::new(Value::Symbol("ParseDate".into())),
                        args: vec![v, fmt.clone()],
                    }
                }
            };
            let ff = match ev.eval(fmt.clone()) {
                Value::String(x) => x,
                v => {
                    return Value::Expr {
                        head: Box::new(Value::Symbol("ParseDate".into())),
                        args: vec![Value::String(ss), v],
                    }
                }
            };
            match chrono::NaiveDateTime::parse_from_str(&ss, &ff).ok().map(|ndt| {
                chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(ndt, chrono::Utc)
                    .timestamp()
            }) {
                Some(ts) => Value::Integer(ts),
                None => Value::Symbol("Null".into()),
            }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("ParseDate".into())), args },
    }
}

#[allow(deprecated)]
fn format_date_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [ts] => match ev.eval(ts.clone()) {
            Value::Integer(secs) => Value::String(
                chrono::NaiveDateTime::from_timestamp_opt(secs, 0)
                    .map(|ndt| ndt.format("%Y-%m-%d %H:%M:%S").to_string())
                    .unwrap_or_default(),
            ),
            v => Value::Expr { head: Box::new(Value::Symbol("FormatDate".into())), args: vec![v] },
        },
        [ts, fmt] => {
            let secs = match ev.eval(ts.clone()) {
                Value::Integer(s) => s,
                v => {
                    return Value::Expr {
                        head: Box::new(Value::Symbol("FormatDate".into())),
                        args: vec![v, fmt.clone()],
                    }
                }
            };
            let ff = match ev.eval(fmt.clone()) {
                Value::String(x) => x,
                v => {
                    return Value::Expr {
                        head: Box::new(Value::Symbol("FormatDate".into())),
                        args: vec![Value::Integer(secs), v],
                    }
                }
            };
            Value::String(
                chrono::NaiveDateTime::from_timestamp_opt(secs, 0)
                    .map(|ndt| ndt.format(&ff).to_string())
                    .unwrap_or_default(),
            )
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("FormatDate".into())), args },
    }
}

fn date_diff_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("DateDiff".into())), args };
    }
    let a = match ev.eval(args[0].clone()) {
        Value::Integer(s) => s,
        Value::String(s) => parse_date_flexible(&s).unwrap_or(0),
        v => {
            return Value::Expr {
                head: Box::new(Value::Symbol("DateDiff".into())),
                args: vec![v, args.get(1).cloned().unwrap_or(Value::Symbol("Null".into()))],
            }
        }
    };
    let b = match ev.eval(args[1].clone()) {
        Value::Integer(s) => s,
        Value::String(s) => parse_date_flexible(&s).unwrap_or(0),
        v => {
            return Value::Expr {
                head: Box::new(Value::Symbol("DateDiff".into())),
                args: vec![Value::Integer(a), v],
            }
        }
    };
    let secs = a - b;
    let unit = if args.len() >= 3 {
        match ev.eval(args[2].clone()) {
            Value::String(u) | Value::Symbol(u) => u.to_lowercase(),
            _ => "seconds".into(),
        }
    } else {
        "seconds".into()
    };
    let val = match unit.as_str() {
        "seconds" => secs as f64,
        "minutes" => (secs as f64) / 60.0,
        "hours" => (secs as f64) / 3600.0,
        "days" => (secs as f64) / 86400.0,
        _ => secs as f64,
    };
    if val.fract() == 0.0 {
        Value::Integer(val as i64)
    } else {
        Value::Real(val)
    }
}

// -------- Tier1 Unicode helpers (minimal stubs) --------
fn normalize_unicode(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // NormalizeUnicode[s, form?] -> stub returns s
    match args.as_slice() {
        [s, _form] => s.clone(),
        [s] => s.clone(),
        _ => Value::Expr { head: Box::new(Value::Symbol("NormalizeUnicode".into())), args },
    }
}

fn case_fold(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [s] => match ev.eval(s.clone()) {
            Value::String(t) => Value::String(t.to_lowercase()),
            v => Value::Expr { head: Box::new(Value::Symbol("CaseFold".into())), args: vec![v] },
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("CaseFold".into())), args },
    }
}

fn transliterate(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Minimal stub: fallback to CaseFold then strip non-ASCII
    match args.as_slice() {
        [s] => match ev.eval(s.clone()) {
            Value::String(t) => {
                let folded = t.to_lowercase();
                let ascii: String = folded.chars().map(|c| if c.is_ascii() { c } else { '?' }).collect();
                Value::String(ascii)
            }
            v => Value::Expr { head: Box::new(Value::Symbol("Transliterate".into())), args: vec![v] },
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("Transliterate".into())), args },
    }
}

fn remove_diacritics(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [s] => match ev.eval(s.clone()) {
            Value::String(t) => {
                // crude strip: keep ASCII only, replace others with base char fallback
                let filtered: String = t.chars().filter(|c| !c.is_ascii() || c.is_ascii_graphic() || c.is_ascii_whitespace()).collect();
                Value::String(filtered)
            }
            v => Value::Expr { head: Box::new(Value::Symbol("RemoveDiacritics".into())), args: vec![v] },
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("RemoveDiacritics".into())), args },
    }
}

fn string_trim_chars(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    fn trim_set(s: &str, set: &str) -> String {
        let mut start = 0usize;
        let mut end = s.len();
        let mut it = s.char_indices();
        let set_chars: Vec<char> = set.chars().collect();

        // Trim start
        while let Some((i, ch)) = it.next() {
            if set_chars.contains(&ch) {
                start = i + ch.len_utf8();
            } else {
                start = i;
                break;
            }
        }
        // If entirely trimmed
        if start >= s.len() {
            return String::new();
        }
        // Trim end
        let mut it_rev = s.char_indices().collect::<Vec<_>>();
        while let Some((i, ch)) = it_rev.pop() {
            if i < start {
                break;
            }
            if set_chars.contains(&ch) {
                end = i;
            } else {
                break;
            }
        }
        s[start..end].to_string()
    }
    match args.as_slice() {
        [Value::String(s), Value::String(chars)] => Value::String(trim_set(s, chars)),
        [a, b] => {
            let aa = ev.eval(a.clone());
            let bb = ev.eval(b.clone());
            string_trim_chars(ev, vec![aa, bb])
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("StringTrimChars".into())), args },
    }
}

fn string_contains(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s), Value::String(sub)] => Value::Boolean(s.contains(sub)),
        [a, b] => {
            let aa = ev.eval(a.clone());
            let bb = ev.eval(b.clone());
            match (aa, bb) {
                (Value::String(s), Value::String(sub)) => Value::Boolean(s.contains(&sub)),
                (aa, bb) => Value::Expr {
                    head: Box::new(Value::Symbol("StringContains".into())),
                    args: vec![aa, bb],
                },
            }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("StringContains".into())), args },
    }
}

fn string_split(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s)] => {
            let parts: Vec<Value> =
                s.split_whitespace().map(|p| Value::String(p.to_string())).collect();
            Value::List(parts)
        }
        [Value::String(s), Value::String(d)] => {
            if d.is_empty() {
                Value::List(s.chars().map(|c| Value::String(c.to_string())).collect())
            } else {
                Value::List(s.split(d).map(|p| Value::String(p.to_string())).collect())
            }
        }
        [a] => match ev.eval(a.clone()) {
            Value::String(s) => {
                Value::List(s.split_whitespace().map(|p| Value::String(p.to_string())).collect())
            }
            v => Value::Expr { head: Box::new(Value::Symbol("Split".into())), args: vec![v] },
        },
        [a, b] => {
            let aa = ev.eval(a.clone());
            let bb = ev.eval(b.clone());
            match (aa, bb) {
                (Value::String(s), Value::String(d)) => {
                    if d.is_empty() {
                        Value::List(s.chars().map(|c| Value::String(c.to_string())).collect())
                    } else {
                        Value::List(s.split(&d).map(|p| Value::String(p.to_string())).collect())
                    }
                }
                (aa, bb) => Value::Expr { head: Box::new(Value::Symbol("Split".into())), args: vec![aa, bb] },
            }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("Split".into())), args },
    }
}

fn split_lines(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    fn split_all_lines(s: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut buf = String::new();
        let mut chars = s.chars().peekable();
        while let Some(ch) = chars.next() {
            match ch {
                '\r' => {
                    if let Some('\n') = chars.peek().copied() {
                        chars.next();
                    }
                    out.push(std::mem::take(&mut buf));
                }
                '\n' => {
                    out.push(std::mem::take(&mut buf));
                }
                _ => buf.push(ch),
            }
        }
        out.push(buf);
        out
    }
    match args.as_slice() {
        [Value::String(s)] => {
            Value::List(split_all_lines(s).into_iter().map(Value::String).collect())
        }
        [other] => match ev.eval(other.clone()) {
            Value::String(s) => {
                Value::List(split_all_lines(&s).into_iter().map(Value::String).collect())
            }
            v => Value::Expr { head: Box::new(Value::Symbol("SplitLines".into())), args: vec![v] },
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("SplitLines".into())), args },
    }
}

fn join_lines(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::List(lines)] => {
            let mut out = String::new();
            let mut first = true;
            for p in lines {
                if !first {
                    out.push('\n');
                }
                first = false;
                match ev.eval(p.clone()) {
                    Value::String(s) => out.push_str(&s),
                    v => out.push_str(&lyra_core::pretty::format_value(&v)),
                }
            }
            Value::String(out)
        }
        [other] => match ev.eval(other.clone()) {
            Value::List(_) => join_lines(ev, vec![other.clone()]),
            v => Value::Expr { head: Box::new(Value::Symbol("JoinLines".into())), args: vec![v] },
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("JoinLines".into())), args },
    }
}

fn starts_with(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s), Value::String(p)] => Value::Boolean(s.starts_with(p)),
        [a, b] => {
            let aa = ev.eval(a.clone());
            let bb = ev.eval(b.clone());
            match (aa, bb) {
                (Value::String(s), Value::String(p)) => Value::Boolean(s.starts_with(&p)),
                (aa, bb) => Value::Expr {
                    head: Box::new(Value::Symbol("StartsWith".into())),
                    args: vec![aa, bb],
                },
            }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("StartsWith".into())), args },
    }
}

fn ends_with(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s), Value::String(p)] => Value::Boolean(s.ends_with(p)),
        [a, b] => {
            let aa = ev.eval(a.clone());
            let bb = ev.eval(b.clone());
            match (aa, bb) {
                (Value::String(s), Value::String(p)) => Value::Boolean(s.ends_with(&p)),
                (aa, bb) => Value::Expr {
                    head: Box::new(Value::Symbol("EndsWith".into())),
                    args: vec![aa, bb],
                },
            }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("EndsWith".into())), args },
    }
}

fn string_replace(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s), Value::String(from), Value::String(to)] => {
            Value::String(s.replace(from, to))
        }
        [a, b, c] => {
            let aa = ev.eval(a.clone());
            let bb = ev.eval(b.clone());
            let cc = ev.eval(c.clone());
            match (aa, bb, cc) {
                (Value::String(s), Value::String(from), Value::String(to)) => {
                    Value::String(s.replace(&from, &to))
                }
                (aa, bb, cc) => Value::Expr {
                    head: Box::new(Value::Symbol("StringReplace".into())),
                    args: vec![aa, bb, cc],
                },
            }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("StringReplace".into())), args },
    }
}

fn string_replace_first(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s), Value::String(from), Value::String(to)] => {
            Value::String(s.replacen(from, to, 1))
        }
        [a, b, c] => {
            let aa = ev.eval(a.clone());
            let bb = ev.eval(b.clone());
            let cc = ev.eval(c.clone());
            match (aa, bb, cc) {
                (Value::String(s), Value::String(from), Value::String(to)) => {
                    Value::String(s.replacen(&from, &to, 1))
                }
                (aa, bb, cc) => Value::Expr {
                    head: Box::new(Value::Symbol("StringReplaceFirst".into())),
                    args: vec![aa, bb, cc],
                },
            }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("StringReplaceFirst".into())), args },
    }
}

fn string_reverse(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s)] => Value::String(s.chars().rev().collect()),
        [other] => match ev.eval(other.clone()) {
            Value::String(s) => Value::String(s.chars().rev().collect()),
            v => {
                Value::Expr { head: Box::new(Value::Symbol("StringReverse".into())), args: vec![v] }
            }
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("StringReverse".into())), args },
    }
}

fn get_first_char(s: &str) -> char {
    s.chars().next().unwrap_or(' ')
}

fn string_pad_left(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s), Value::Integer(n)] => {
            let width = *n as usize;
            let len = s.chars().count();
            if len >= width {
                return Value::String(s.clone());
            }
            let padc = ' ';
            let pads = std::iter::repeat(padc).take(width - len).collect::<String>();
            Value::String(format!("{}{}", pads, s))
        }
        [Value::String(s), Value::Integer(n), Value::String(p)] => {
            let width = *n as usize;
            let len = s.chars().count();
            if len >= width {
                return Value::String(s.clone());
            }
            let padc = get_first_char(p);
            let pads = std::iter::repeat(padc).take(width - len).collect::<String>();
            Value::String(format!("{}{}", pads, s))
        }
        [a, b] => {
            let aa = ev.eval(a.clone());
            let bb = ev.eval(b.clone());
            string_pad_left(ev, vec![aa, bb])
        }
        [a, b, c] => {
            let aa = ev.eval(a.clone());
            let bb = ev.eval(b.clone());
            let cc = ev.eval(c.clone());
            string_pad_left(ev, vec![aa, bb, cc])
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("StringPadLeft".into())), args },
    }
}

fn string_pad_right(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s), Value::Integer(n)] => {
            let width = *n as usize;
            let len = s.chars().count();
            if len >= width {
                return Value::String(s.clone());
            }
            let padc = ' ';
            let pads = std::iter::repeat(padc).take(width - len).collect::<String>();
            Value::String(format!("{}{}", s, pads))
        }
        [Value::String(s), Value::Integer(n), Value::String(p)] => {
            let width = *n as usize;
            let len = s.chars().count();
            if len >= width {
                return Value::String(s.clone());
            }
            let padc = get_first_char(p);
            let pads = std::iter::repeat(padc).take(width - len).collect::<String>();
            Value::String(format!("{}{}", s, pads))
        }
        [a, b] => {
            let aa = ev.eval(a.clone());
            let bb = ev.eval(b.clone());
            string_pad_right(ev, vec![aa, bb])
        }
        [a, b, c] => {
            let aa = ev.eval(a.clone());
            let bb = ev.eval(b.clone());
            let cc = ev.eval(c.clone());
            string_pad_right(ev, vec![aa, bb, cc])
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("StringPadRight".into())), args },
    }
}

fn clamp_start_len(len: usize, start: i64, opt_len: Option<i64>) -> (usize, usize) {
    let slen = len as i64;
    let mut s = if start < 0 { slen + start } else { start };
    if s < 0 {
        s = 0;
    }
    if s > slen {
        return (len, len);
    }
    let mut e = match opt_len {
        Some(l) if l <= 0 => s,
        Some(l) => (s + l).min(slen),
        None => slen,
    };
    if e < s {
        e = s;
    }
    (s as usize, e as usize)
}

fn string_slice(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    fn slice_chars(s: &str, start: i64, opt_len: Option<i64>) -> String {
        let chars: Vec<char> = s.chars().collect();
        let (sidx, eidx) = clamp_start_len(chars.len(), start, opt_len);
        chars[sidx..eidx].iter().collect()
    }
    match args.as_slice() {
        [Value::String(s), Value::Integer(start)] => Value::String(slice_chars(s, *start, None)),
        [Value::String(s), Value::Integer(start), Value::Integer(len)] => {
            Value::String(slice_chars(s, *start, Some(*len)))
        }
        [a, b] => {
            let aa = ev.eval(a.clone());
            let bb = ev.eval(b.clone());
            string_slice(ev, vec![aa, bb])
        }
        [a, b, c] => {
            let aa = ev.eval(a.clone());
            let bb = ev.eval(b.clone());
            let cc = ev.eval(c.clone());
            string_slice(ev, vec![aa, bb, cc])
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("StringSlice".into())), args },
    }
}

fn index_of(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    fn find_index(hay: &str, needle: &str, from: Option<i64>) -> i64 {
        let chars: Vec<char> = hay.chars().collect();
        let start = from.unwrap_or(0);
        let (sidx, _) = clamp_start_len(chars.len(), start, None);
        let prefix: String = chars[..sidx].iter().collect();
        let rest: String = chars[sidx..].iter().collect();
        match rest.find(needle) {
            Some(byte_pos) => {
                let full = format!("{}{}", prefix, &rest[..byte_pos]);
                full.chars().count() as i64
            }
            None => -1,
        }
    }
    match args.as_slice() {
        [Value::String(s), Value::String(sub)] => Value::Integer(find_index(s, sub, None)),
        [Value::String(s), Value::String(sub), Value::Integer(from)] => {
            Value::Integer(find_index(s, sub, Some(*from)))
        }
        [a, b] => {
            let aa = ev.eval(a.clone());
            let bb = ev.eval(b.clone());
            index_of(ev, vec![aa, bb])
        }
        [a, b, c] => {
            let aa = ev.eval(a.clone());
            let bb = ev.eval(b.clone());
            let cc = ev.eval(c.clone());
            index_of(ev, vec![aa, bb, cc])
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("IndexOf".into())), args },
    }
}

fn last_index_of(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    fn find_last_index(hay: &str, needle: &str, from: Option<i64>) -> i64 {
        let chars: Vec<char> = hay.chars().collect();
        let len = chars.len() as i64;
        let start = from.unwrap_or(len - 1);
        let (sidx, _) = clamp_start_len(chars.len(), start, None);
        let head: String = chars[..=sidx.min(chars.len().saturating_sub(1))].iter().collect();
        match head.rfind(needle) {
            Some(byte_pos) => head[..byte_pos].chars().count() as i64,
            None => -1,
        }
    }
    match args.as_slice() {
        [Value::String(s), Value::String(sub)] => Value::Integer(find_last_index(s, sub, None)),
        [Value::String(s), Value::String(sub), Value::Integer(from)] => {
            Value::Integer(find_last_index(s, sub, Some(*from)))
        }
        [a, b] => {
            let aa = ev.eval(a.clone());
            let bb = ev.eval(b.clone());
            last_index_of(ev, vec![aa, bb])
        }
        [a, b, c] => {
            let aa = ev.eval(a.clone());
            let bb = ev.eval(b.clone());
            let cc = ev.eval(c.clone());
            last_index_of(ev, vec![aa, bb, cc])
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("LastIndexOf".into())), args },
    }
}

fn string_repeat(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s), Value::Integer(n)] => {
            let times = if *n <= 0 { 0 } else { *n as usize };
            Value::String(s.repeat(times))
        }
        [a, b] => {
            let aa = ev.eval(a.clone());
            let bb = ev.eval(b.clone());
            string_repeat(ev, vec![aa, bb])
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("StringRepeat".into())), args },
    }
}

fn is_blank(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s)] => Value::Boolean(s.trim().is_empty()),
        [other] => match ev.eval(other.clone()) {
            Value::String(s) => Value::Boolean(s.trim().is_empty()),
            v => Value::Expr { head: Box::new(Value::Symbol("IsBlank".into())), args: vec![v] },
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("IsBlank".into())), args },
    }
}

fn capitalize(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    fn cap_one(s: &str) -> String {
        let mut iter = s.chars();
        match iter.next() {
            None => String::new(),
            Some(c0) => {
                let mut out = String::new();
                for uc in c0.to_uppercase() {
                    out.push(uc);
                }
                out.push_str(&iter.as_str().to_lowercase());
                out
            }
        }
    }
    match args.as_slice() {
        [Value::String(s)] => Value::String(cap_one(s)),
        [other] => match ev.eval(other.clone()) {
            Value::String(s) => Value::String(cap_one(&s)),
            v => Value::Expr { head: Box::new(Value::Symbol("Capitalize".into())), args: vec![v] },
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("Capitalize".into())), args },
    }
}

fn title_case(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    fn titleize(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        let mut start_word = true;
        for ch in s.chars() {
            if ch.is_whitespace() {
                start_word = true;
                out.push(ch);
            } else if start_word {
                for uc in ch.to_uppercase() {
                    out.push(uc);
                }
                start_word = false;
            } else {
                for lc in ch.to_lowercase() {
                    out.push(lc);
                }
            }
        }
        out
    }
    match args.as_slice() {
        [Value::String(s)] => Value::String(titleize(s)),
        [other] => match ev.eval(other.clone()) {
            Value::String(s) => Value::String(titleize(&s)),
            v => Value::Expr { head: Box::new(Value::Symbol("TitleCase".into())), args: vec![v] },
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("TitleCase".into())), args },
    }
}

fn equals_ignore_case(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(a), Value::String(b)] => {
            Value::Boolean(a.to_lowercase() == b.to_lowercase())
        }
        [a, b] => {
            let aa = ev.eval(a.clone());
            let bb = ev.eval(b.clone());
            equals_ignore_case(ev, vec![aa, bb])
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("EqualsIgnoreCase".into())), args },
    }
}

fn string_chars(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s)] => {
            Value::List(s.chars().map(|c| Value::String(c.to_string())).collect())
        }
        [other] => match ev.eval(other.clone()) {
            Value::String(s) => {
                Value::List(s.chars().map(|c| Value::String(c.to_string())).collect())
            }
            v => Value::Expr { head: Box::new(Value::Symbol("StringChars".into())), args: vec![v] },
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("StringChars".into())), args },
    }
}

fn string_from_chars(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::List(chars)] => {
            let mut out = String::new();
            for p in chars {
                match ev.eval(p.clone()) {
                    Value::String(s) => out.push(get_first_char(&s)),
                    v => {
                        let fs = lyra_core::pretty::format_value(&v);
                        out.push(get_first_char(&fs));
                    }
                }
            }
            Value::String(out)
        }
        [other] => match ev.eval(other.clone()) {
            Value::List(_) => string_from_chars(ev, vec![other.clone()]),
            v => Value::Expr {
                head: Box::new(Value::Symbol("StringFromChars".into())),
                args: vec![v],
            },
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("StringFromChars".into())), args },
    }
}

fn scan_interpolation_parts(src: &str) -> Result<Vec<Value>, String> {
    let mut parts: Vec<Value> = Vec::new();
    let mut buf = String::new();
    let chars: Vec<char> = src.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        let ch = chars[i];
        if ch == '{' {
            if i + 1 < chars.len() && chars[i + 1] == '{' {
                buf.push('{');
                i += 2;
                continue;
            }
            if !buf.is_empty() {
                parts.push(Value::String(std::mem::take(&mut buf)));
            }
            i += 1;
            let start = i;
            let mut depth: i32 = 1;
            let mut in_str = false;
            while i < chars.len() {
                let c = chars[i];
                if in_str {
                    if c == '"' {
                        in_str = false;
                    }
                    i += 1;
                    continue;
                }
                match c {
                    '"' => {
                        in_str = true;
                        i += 1;
                    }
                    '{' => {
                        depth += 1;
                        i += 1;
                    }
                    '}' => {
                        depth -= 1;
                        i += 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    _ => {
                        i += 1;
                    }
                }
            }
            if depth != 0 {
                return Err("unterminated interpolation".into());
            }
            let inner: String = chars[start..i - 1].iter().collect();
            let mut p = Parser::from_source(&inner);
            match p.parse_all() {
                Ok(vals) if !vals.is_empty() => parts.push(vals.last().unwrap().clone()),
                Ok(_) => return Err("empty interpolation".into()),
                Err(e) => return Err(format!("parse error: {e}")),
            }
        } else if ch == '}' {
            if i + 1 < chars.len() && chars[i + 1] == '}' {
                buf.push('}');
                i += 2;
            } else {
                buf.push('}');
                i += 1;
            }
        } else {
            buf.push(ch);
            i += 1;
        }
    }
    if !buf.is_empty() {
        parts.push(Value::String(buf));
    }
    Ok(parts)
}

fn string_interpolate(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s)] => match scan_interpolation_parts(s) {
            Ok(parts) => {
                if parts.is_empty() {
                    return Value::String(String::new());
                }
                if parts.len() == 1 {
                    return ev.eval(parts[0].clone());
                }
                let expr =
                    Value::expr(Value::Symbol("StringJoin".into()), vec![Value::List(parts)]);
                ev.eval(expr)
            }
            Err(_) => {
                Value::Expr { head: Box::new(Value::Symbol("StringInterpolate".into())), args }
            }
        },
        [other] => match ev.eval(other.clone()) {
            Value::String(s) => string_interpolate(ev, vec![Value::String(s)]),
            v => Value::Expr {
                head: Box::new(Value::Symbol("StringInterpolate".into())),
                args: vec![v],
            },
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("StringInterpolate".into())), args },
    }
}

fn string_interpolate_with(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s), assoc] => {
            let assoc_eval = ev.eval(assoc.clone());
            if let Value::Assoc(m) = assoc_eval {
                let mut out = String::new();
                let cs: Vec<char> = s.chars().collect();
                let mut i = 0usize;
                while i < cs.len() {
                    let ch = cs[i];
                    if ch == '{' {
                        if i + 1 < cs.len() && cs[i + 1] == '{' {
                            out.push('{');
                            i += 2;
                            continue;
                        }
                        // read placeholder
                        let mut j = i + 1;
                        let mut key = String::new();
                        while j < cs.len() && cs[j] != '}' {
                            key.push(cs[j]);
                            j += 1;
                        }
                        if j < cs.len() && cs[j] == '}' {
                            if let Some(val) = m.get(key.trim()) {
                                let v = ev.eval(val.clone());
                                match v {
                                    Value::String(s2) => out.push_str(&s2),
                                    other => out.push_str(&lyra_core::pretty::format_value(&other)),
                                }
                                i = j + 1;
                                continue;
                            }
                        }
                        // no match; emit '{' and continue
                        out.push('{');
                        i += 1;
                    } else if ch == '}' {
                        if i + 1 < cs.len() && cs[i + 1] == '}' {
                            out.push('}');
                            i += 2;
                        } else {
                            out.push('}');
                            i += 1;
                        }
                    } else {
                        out.push(ch);
                        i += 1;
                    }
                }
                Value::String(out)
            } else {
                Value::Expr { head: Box::new(Value::Symbol("StringInterpolateWith".into())), args }
            }
        }
        [a, b] => {
            let aa = ev.eval(a.clone());
            let bb = ev.eval(b.clone());
            string_interpolate_with(ev, vec![aa, bb])
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("StringInterpolateWith".into())), args },
    }
}

fn string_format(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    fn render(fmt: &str, args: &[Value], ev: &mut Evaluator) -> String {
        let mut out = String::new();
        let cs: Vec<char> = fmt.chars().collect();
        let mut i = 0usize;
        while i < cs.len() {
            let ch = cs[i];
            if ch == '{' {
                if i + 1 < cs.len() && cs[i + 1] == '{' {
                    out.push('{');
                    i += 2;
                    continue;
                }
                let mut j = i + 1;
                let mut num = String::new();
                while j < cs.len() && cs[j] != '}' {
                    num.push(cs[j]);
                    j += 1;
                }
                if j < cs.len() && cs[j] == '}' {
                    if let Ok(idx) = num.trim().parse::<usize>() {
                        if let Some(val) = args.get(idx) {
                            let v = ev.eval(val.clone());
                            match v {
                                Value::String(s) => out.push_str(&s),
                                other => out.push_str(&lyra_core::pretty::format_value(&other)),
                            }
                        }
                        i = j + 1;
                        continue;
                    }
                }
                out.push('{');
                i += 1;
            } else if ch == '}' {
                if i + 1 < cs.len() && cs[i + 1] == '}' {
                    out.push('}');
                    i += 2;
                } else {
                    out.push('}');
                    i += 1;
                }
            } else {
                out.push(ch);
                i += 1;
            }
        }
        out
    }
    match args.as_slice() {
        [Value::String(fmt), Value::List(xs)] => Value::String(render(fmt, xs, ev)),
        [a, b] => {
            let aa = ev.eval(a.clone());
            let bb = ev.eval(b.clone());
            string_format(ev, vec![aa, bb])
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("StringFormat".into())), args },
    }
}

fn string_format_map(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    fn render(
        fmt: &str,
        map: &std::collections::HashMap<String, Value>,
        ev: &mut Evaluator,
    ) -> String {
        let mut out = String::new();
        let cs: Vec<char> = fmt.chars().collect();
        let mut i = 0usize;
        while i < cs.len() {
            let ch = cs[i];
            if ch == '{' {
                if i + 1 < cs.len() && cs[i + 1] == '{' {
                    out.push('{');
                    i += 2;
                    continue;
                }
                let mut j = i + 1;
                let mut key = String::new();
                while j < cs.len() && cs[j] != '}' {
                    key.push(cs[j]);
                    j += 1;
                }
                if j < cs.len() && cs[j] == '}' {
                    if let Some(val) = map.get(key.trim()) {
                        let v = ev.eval(val.clone());
                        match v {
                            Value::String(s) => out.push_str(&s),
                            other => out.push_str(&lyra_core::pretty::format_value(&other)),
                        }
                        i = j + 1;
                        continue;
                    }
                }
                out.push('{');
                i += 1;
            } else if ch == '}' {
                if i + 1 < cs.len() && cs[i + 1] == '}' {
                    out.push('}');
                    i += 2;
                } else {
                    out.push('}');
                    i += 1;
                }
            } else {
                out.push(ch);
                i += 1;
            }
        }
        out
    }
    match args.as_slice() {
        [Value::String(fmt), Value::Assoc(m)] => Value::String(render(fmt, m, ev)),
        [a, b] => {
            let aa = ev.eval(a.clone());
            let bb = ev.eval(b.clone());
            string_format_map(ev, vec![aa, bb])
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("StringFormatMap".into())), args },
    }
}

fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(ch),
        }
    }
    out
}

fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(ch),
        }
    }
    out
}

fn html_escape_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s)] => Value::String(html_escape(s)),
        [other] => match ev.eval(other.clone()) {
            Value::String(s) => Value::String(html_escape(&s)),
            v => Value::Expr { head: Box::new(Value::Symbol("HtmlEscape".into())), args: vec![v] },
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("HtmlEscape".into())), args },
    }
}

fn html_unescape_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    fn unesc(s: &str) -> String {
        let mut out = String::new();
        let mut i = 0usize;
        let b = s.as_bytes();
        while i < b.len() {
            if b[i] == b'&' {
                if s[i..].starts_with("&amp;") {
                    out.push('&');
                    i += 5;
                    continue;
                }
                if s[i..].starts_with("&lt;") {
                    out.push('<');
                    i += 4;
                    continue;
                }
                if s[i..].starts_with("&gt;") {
                    out.push('>');
                    i += 4;
                    continue;
                }
                if s[i..].starts_with("&quot;") {
                    out.push('"');
                    i += 6;
                    continue;
                }
                if s[i..].starts_with("&#39;") {
                    out.push('\'');
                    i += 5;
                    continue;
                }
            }
            out.push(b[i] as char);
            i += 1;
        }
        out
    }
    match args.as_slice() {
        [Value::String(s)] => Value::String(unesc(s)),
        [other] => match ev.eval(other.clone()) {
            Value::String(s) => Value::String(unesc(&s)),
            v => {
                Value::Expr { head: Box::new(Value::Symbol("HtmlUnescape".into())), args: vec![v] }
            }
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("HtmlUnescape".into())), args },
    }
}

fn url_encode_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    fn encode(s: &str) -> String {
        let mut out = String::new();
        for &b in s.as_bytes() {
            let c = b as char;
            let unreserved =
                c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' || c == '~';
            if unreserved {
                out.push(c);
            } else {
                out.push_str(&format!("%{:02X}", b));
            }
        }
        out
    }
    match args.as_slice() {
        [Value::String(s)] => Value::String(encode(s)),
        [other] => match ev.eval(other.clone()) {
            Value::String(s) => Value::String(encode(&s)),
            v => Value::Expr { head: Box::new(Value::Symbol("UrlEncode".into())), args: vec![v] },
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("UrlEncode".into())), args },
    }
}

fn url_decode_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    fn from_hex(x: u8) -> Option<u8> {
        match x {
            b'0'..=b'9' => Some(x - b'0'),
            b'a'..=b'f' => Some(10 + x - b'a'),
            b'A'..=b'F' => Some(10 + x - b'A'),
            _ => None,
        }
    }
    fn decode(s: &str) -> String {
        let mut out: Vec<u8> = Vec::with_capacity(s.len());
        let b = s.as_bytes();
        let mut i = 0usize;
        while i < b.len() {
            if b[i] == b'%' && i + 2 < b.len() {
                if let (Some(h), Some(l)) = (from_hex(b[i + 1]), from_hex(b[i + 2])) {
                    out.push((h << 4) | l);
                    i += 3;
                    continue;
                }
            }
            out.push(b[i]);
            i += 1;
        }
        String::from_utf8_lossy(&out).to_string()
    }
    match args.as_slice() {
        [Value::String(s)] => Value::String(decode(s)),
        [other] => match ev.eval(other.clone()) {
            Value::String(s) => Value::String(decode(&s)),
            v => Value::Expr { head: Box::new(Value::Symbol("UrlDecode".into())), args: vec![v] },
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("UrlDecode".into())), args },
    }
}

fn json_escape_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    fn esc(s: &str) -> String {
        let mut out = String::new();
        for ch in s.chars() {
            match ch {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04X}", c as u32)),
                c => out.push(c),
            }
        }
        out
    }
    match args.as_slice() {
        [Value::String(s)] => Value::String(esc(s)),
        [other] => match ev.eval(other.clone()) {
            Value::String(s) => Value::String(esc(&s)),
            v => Value::Expr { head: Box::new(Value::Symbol("JsonEscape".into())), args: vec![v] },
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("JsonEscape".into())), args },
    }
}

fn json_unescape_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    fn hex_val(c: u8) -> Option<u8> {
        match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(10 + c - b'a'),
            b'A'..=b'F' => Some(10 + c - b'A'),
            _ => None,
        }
    }
    fn unesc(s: &str) -> String {
        let b = s.as_bytes();
        let mut i = 0usize;
        let mut out = String::new();
        while i < b.len() {
            if b[i] == b'\\' && i + 1 < b.len() {
                match b[i + 1] {
                    b'"' => {
                        out.push('"');
                        i += 2;
                    }
                    b'\\' => {
                        out.push('\\');
                        i += 2;
                    }
                    b'n' => {
                        out.push('\n');
                        i += 2;
                    }
                    b'r' => {
                        out.push('\r');
                        i += 2;
                    }
                    b't' => {
                        out.push('\t');
                        i += 2;
                    }
                    b'u' => {
                        if i + 5 < b.len() {
                            if let (Some(a), Some(c), Some(d), Some(e)) = (
                                hex_val(b[i + 2]),
                                hex_val(b[i + 3]),
                                hex_val(b[i + 4]),
                                hex_val(b[i + 5]),
                            ) {
                                let cp = ((a as u16) << 12)
                                    | ((c as u16) << 8)
                                    | ((d as u16) << 4)
                                    | (e as u16);
                                if let Some(ch) = char::from_u32(cp as u32) {
                                    out.push(ch);
                                }
                                i += 6;
                                continue;
                            }
                        }
                        // fallback
                        out.push('u');
                        i += 2;
                    }
                    _ => {
                        out.push(b[i + 1] as char);
                        i += 2;
                    }
                }
                continue;
            }
            out.push(b[i] as char);
            i += 1;
        }
        out
    }
    match args.as_slice() {
        [Value::String(s)] => Value::String(unesc(s)),
        [other] => match ev.eval(other.clone()) {
            Value::String(s) => Value::String(unesc(&s)),
            v => {
                Value::Expr { head: Box::new(Value::Symbol("JsonUnescape".into())), args: vec![v] }
            }
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("JsonUnescape".into())), args },
    }
}

fn url_form_encode_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    fn encode(s: &str) -> String {
        let mut out = String::new();
        for &b in s.as_bytes() {
            match b as char {
                ' ' => out.push('+'),
                c if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' || c == '*' => {
                    out.push(c)
                }
                _ => out.push_str(&format!("%{:02X}", b)),
            }
        }
        out
    }
    match args.as_slice() {
        [Value::String(s)] => Value::String(encode(s)),
        [other] => match ev.eval(other.clone()) {
            Value::String(s) => Value::String(encode(&s)),
            v => {
                Value::Expr { head: Box::new(Value::Symbol("UrlFormEncode".into())), args: vec![v] }
            }
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("UrlFormEncode".into())), args },
    }
}

fn url_form_decode_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    fn from_hex(x: u8) -> Option<u8> {
        match x {
            b'0'..=b'9' => Some(x - b'0'),
            b'a'..=b'f' => Some(10 + x - b'a'),
            b'A'..=b'F' => Some(10 + x - b'A'),
            _ => None,
        }
    }
    fn decode(s: &str) -> String {
        let mut out: Vec<u8> = Vec::with_capacity(s.len());
        let b = s.as_bytes();
        let mut i = 0usize;
        while i < b.len() {
            if b[i] == b'+' {
                out.push(b' ');
                i += 1;
                continue;
            }
            if b[i] == b'%' && i + 2 < b.len() {
                if let (Some(h), Some(l)) = (from_hex(b[i + 1]), from_hex(b[i + 2])) {
                    out.push((h << 4) | l);
                    i += 3;
                    continue;
                }
            }
            out.push(b[i]);
            i += 1;
        }
        String::from_utf8_lossy(&out).to_string()
    }
    match args.as_slice() {
        [Value::String(s)] => Value::String(decode(s)),
        [other] => match ev.eval(other.clone()) {
            Value::String(s) => Value::String(decode(&s)),
            v => {
                Value::Expr { head: Box::new(Value::Symbol("UrlFormDecode".into())), args: vec![v] }
            }
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("UrlFormDecode".into())), args },
    }
}

fn slugify_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    fn slugify(s: &str) -> String {
        let mut out = String::new();
        let mut prev_dash = false;
        for ch in s.chars() {
            if ch.is_ascii_alphanumeric() {
                out.push(ch.to_ascii_lowercase());
                prev_dash = false;
            } else {
                if !prev_dash && !out.is_empty() {
                    out.push('-');
                    prev_dash = true;
                }
            }
        }
        if out.ends_with('-') {
            out.pop();
        }
        out
    }
    match args.as_slice() {
        [Value::String(s)] => Value::String(slugify(s)),
        [other] => match ev.eval(other.clone()) {
            Value::String(s) => Value::String(slugify(&s)),
            v => Value::Expr { head: Box::new(Value::Symbol("Slugify".into())), args: vec![v] },
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("Slugify".into())), args },
    }
}

fn string_truncate_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    fn trunc(s: &str, max_chars: usize, suffix: &str) -> String {
        let total = s.chars().count();
        if total <= max_chars {
            return s.to_string();
        }
        let suf_len = suffix.chars().count();
        let keep = max_chars.saturating_sub(suf_len);
        let mut it = s.chars();
        let kept: String = it.by_ref().take(keep).collect();
        format!("{}{}", kept, suffix)
    }
    match args.as_slice() {
        [Value::String(s), Value::Integer(n)] => {
            let len = (*n).max(0) as usize;
            Value::String(trunc(s, len, "…"))
        }
        [Value::String(s), Value::Integer(n), Value::String(suf)] => {
            let len = (*n).max(0) as usize;
            Value::String(trunc(s, len, suf))
        }
        [a, b] => {
            let aa = ev.eval(a.clone());
            let bb = ev.eval(b.clone());
            string_truncate_fn(ev, vec![aa, bb])
        }
        [a, b, c] => {
            let aa = ev.eval(a.clone());
            let bb = ev.eval(b.clone());
            let cc = ev.eval(c.clone());
            string_truncate_fn(ev, vec![aa, bb, cc])
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("StringTruncate".into())), args },
    }
}

fn split_words(s: &str) -> Vec<String> {
    let cs: Vec<char> = s.chars().collect();
    let mut words: Vec<String> = Vec::new();
    let mut cur = String::new();
    for i in 0..cs.len() {
        let ch = cs[i];
        if ch.is_ascii_alphanumeric() {
            let prev = if i > 0 { cs[i - 1] } else { '\0' };
            let next = if i + 1 < cs.len() { cs[i + 1] } else { '\0' };
            let boundary =
                ch.is_ascii_uppercase() && prev.is_ascii_lowercase() && next.is_ascii_lowercase();
            if boundary && !cur.is_empty() {
                words.push(cur.clone());
                cur.clear();
            }
            cur.push(ch);
        } else {
            if !cur.is_empty() {
                words.push(cur.clone());
                cur.clear();
            }
        }
    }
    if !cur.is_empty() {
        words.push(cur);
    }
    words
}

fn camel_case_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    fn camel(s: &str) -> String {
        let words = split_words(s);
        if words.is_empty() {
            return String::new();
        }
        let mut out = String::new();
        out.push_str(&words[0].to_ascii_lowercase());
        for w in words.iter().skip(1) {
            let mut cs = w.chars();
            if let Some(c0) = cs.next() {
                for uc in c0.to_uppercase() {
                    out.push(uc);
                }
            }
            out.push_str(&cs.as_str().to_ascii_lowercase());
        }
        out
    }
    match args.as_slice() {
        [Value::String(s)] => Value::String(camel(s)),
        [other] => match ev.eval(other.clone()) {
            Value::String(s) => Value::String(camel(&s)),
            v => Value::Expr { head: Box::new(Value::Symbol("CamelCase".into())), args: vec![v] },
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("CamelCase".into())), args },
    }
}

fn snake_case_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    fn snake(s: &str) -> String {
        let words = split_words(s);
        words.into_iter().map(|w| w.to_ascii_lowercase()).collect::<Vec<_>>().join("_")
    }
    match args.as_slice() {
        [Value::String(s)] => Value::String(snake(s)),
        [other] => match ev.eval(other.clone()) {
            Value::String(s) => Value::String(snake(&s)),
            v => Value::Expr { head: Box::new(Value::Symbol("SnakeCase".into())), args: vec![v] },
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("SnakeCase".into())), args },
    }
}

fn kebab_case_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    fn kebab(s: &str) -> String {
        let words = split_words(s);
        words.into_iter().map(|w| w.to_ascii_lowercase()).collect::<Vec<_>>().join("-")
    }
    match args.as_slice() {
        [Value::String(s)] => Value::String(kebab(s)),
        [other] => match ev.eval(other.clone()) {
            Value::String(s) => Value::String(kebab(&s)),
            v => Value::Expr { head: Box::new(Value::Symbol("KebabCase".into())), args: vec![v] },
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("KebabCase".into())), args },
    }
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Boolean(b) => *b,
        Value::String(s) => !s.is_empty(),
        Value::List(xs) => !xs.is_empty(),
        Value::Assoc(m) => !m.is_empty(),
        Value::Integer(n) => *n != 0,
        Value::Real(f) => *f != 0.0,
        _ => true,
    }
}

fn resolve_path<'a>(stack: &'a [Value], name: &str) -> Option<Value> {
    if name == "." {
        return stack.last().cloned();
    }
    let parts: Vec<&str> = name.split('.').collect();
    for ctx in stack.iter().rev() {
        let mut cur = ctx.clone();
        let mut ok = true;
        for p in &parts {
            match cur {
                Value::Assoc(ref m) => {
                    if let Some(v) = m.get(&p.to_string()) {
                        cur = v.clone();
                    } else {
                        ok = false;
                        break;
                    }
                }
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            return Some(cur);
        }
    }
    None
}

fn apply_filter(ev: &mut Evaluator, filter: &str, val: Value) -> Value {
    // Only allow simple symbol filters without args: Filter[val]
    let expr = Value::expr(Value::Symbol(filter.to_string()), vec![val]);
    ev.eval(expr)
}

fn template_render(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // TemplateRender[template, assoc, opts?] where opts can include <|"Partials"->Assoc, "EscapeHtml"->True|>
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("TemplateRender".into())), args };
    }
    let template = match ev.eval(args[0].clone()) {
        Value::String(s) => s,
        other => {
            return Value::Expr {
                head: Box::new(Value::Symbol("TemplateRender".into())),
                args: vec![other, args[1].clone()],
            }
        }
    };
    let ctx0 = match ev.eval(args[1].clone()) {
        Value::Assoc(m) => Value::Assoc(m),
        other => other,
    };
    let mut partials: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut escape_html = true;
    if args.len() >= 3 {
        if let Value::Assoc(m) = ev.eval(args[2].clone()) {
            if let Some(Value::Assoc(p)) = m.get("Partials") {
                for (k, v) in p.iter() {
                    if let Value::String(s) = v {
                        partials.insert(k.clone(), s.clone());
                    }
                }
            }
            if let Some(Value::Boolean(b)) = m.get("EscapeHtml") {
                escape_html = *b;
            }
        }
    }

    fn render_block(
        ev: &mut Evaluator,
        tpl: &str,
        stack: &mut Vec<Value>,
        partials: &std::collections::HashMap<String, String>,
        escape_html: bool,
    ) -> String {
        let mut out = String::new();
        let chars: Vec<char> = tpl.chars().collect();
        let mut i = 0usize;
        while i < chars.len() {
            if chars[i] == '{' && i + 1 < chars.len() && chars[i + 1] == '{' {
                // detect triple mustache {{{
                let triple = i + 2 < chars.len() && chars[i + 2] == '{';
                let start = i + if triple { 3 } else { 2 };
                // find end
                let mut j = start;
                while j < chars.len() {
                    if !triple && j + 1 < chars.len() && chars[j] == '}' && chars[j + 1] == '}' {
                        break;
                    }
                    if triple
                        && j + 2 < chars.len()
                        && chars[j] == '}'
                        && chars[j + 1] == '}'
                        && chars[j + 2] == '}'
                    {
                        break;
                    }
                    j += 1;
                }
                let content: String = chars[start..j].iter().collect();
                i = j + if triple { 3 } else { 2 };
                let tag = content.trim();
                if tag.starts_with('!') {
                    continue;
                }
                if tag.starts_with('>') {
                    let name = tag[1..].trim();
                    if let Some(ptpl) = partials.get(name) {
                        out.push_str(&render_block(ev, ptpl, stack, partials, escape_html));
                    }
                    continue;
                }
                if tag.starts_with('#') || tag.starts_with('^') {
                    let inverted = tag.starts_with('^');
                    let key = tag[1..].trim();
                    // find matching closing {{/key}}
                    let mut depth = 1i32;
                    let mut k = i;
                    let mut inner_start = i;
                    while k < chars.len() {
                        if chars[k] == '{' && k + 1 < chars.len() && chars[k + 1] == '{' {
                            let triple2 = k + 2 < chars.len() && chars[k + 2] == '{';
                            let start2 = k + if triple2 { 3 } else { 2 };
                            let mut j2 = start2;
                            while j2 < chars.len() {
                                if !triple2
                                    && j2 + 1 < chars.len()
                                    && chars[j2] == '}'
                                    && chars[j2 + 1] == '}'
                                {
                                    break;
                                }
                                if triple2
                                    && j2 + 2 < chars.len()
                                    && chars[j2] == '}'
                                    && chars[j2 + 1] == '}'
                                    && chars[j2 + 2] == '}'
                                {
                                    break;
                                }
                                j2 += 1;
                            }
                            let tag2: String = chars[start2..j2].iter().collect();
                            let tag2t = tag2.trim();
                            if tag2t.starts_with('#') && tag2t[1..].trim() == key {
                                depth += 1;
                            } else if tag2t.starts_with('/') && tag2t[1..].trim() == key {
                                depth -= 1;
                                if depth == 0 {
                                    inner_start = i;
                                    i = j2 + if triple2 { 3 } else { 2 };
                                    break;
                                }
                            }
                            k = j2 + if triple2 { 3 } else { 2 };
                            continue;
                        }
                        k += 1;
                    }
                    let inner: String =
                        chars[inner_start..(i - if triple { 0 } else { 0 })].iter().collect();
                    // Evaluate key
                    let val = resolve_path(stack, key).unwrap_or(Value::Symbol("Null".into()));
                    if inverted {
                        if !truthy(&val) {
                            out.push_str(&render_block(ev, &inner, stack, partials, escape_html));
                        }
                    } else {
                        match val {
                            Value::List(xs) => {
                                for item in xs {
                                    stack.push(item);
                                    out.push_str(&render_block(
                                        ev,
                                        &inner,
                                        stack,
                                        partials,
                                        escape_html,
                                    ));
                                    stack.pop();
                                }
                            }
                            Value::Assoc(_) => {
                                stack.push(val);
                                out.push_str(&render_block(
                                    ev,
                                    &inner,
                                    stack,
                                    partials,
                                    escape_html,
                                ));
                                stack.pop();
                            }
                            other => {
                                if truthy(&other) {
                                    out.push_str(&render_block(
                                        ev,
                                        &inner,
                                        stack,
                                        partials,
                                        escape_html,
                                    ));
                                }
                            }
                        }
                    }
                    continue;
                }
                if tag.starts_with('/') {
                    continue;
                }
                // variable with optional filter: name|Filter
                let mut parts = tag.split('|');
                let name = parts.next().unwrap().trim();
                let filt = parts.next().map(|s| s.trim()).filter(|s| !s.is_empty());
                let mut val = resolve_path(stack, name).unwrap_or(Value::String(String::new()));
                if let Some(f) = filt {
                    val = apply_filter(ev, f, val);
                }
                let s = match val {
                    Value::String(s) => s,
                    other => lyra_core::pretty::format_value(&other),
                };
                if triple || !escape_html {
                    out.push_str(&s);
                } else {
                    out.push_str(&html_escape(&s));
                }
            } else {
                out.push(chars[i]);
                i += 1;
            }
        }
        out
    }

    let mut stack: Vec<Value> = vec![ctx0];
    let rendered = render_block(ev, &template, &mut stack, &partials, escape_html);
    Value::String(rendered)
}

// -------------- HTML templating --------------

fn is_safe_html(v: &Value) -> Option<String> {
    if let Value::Assoc(m) = v {
        if let Some(Value::String(t)) = m.get("__type") {
            if t == "SafeHtml" {
                if let Some(Value::String(s)) = m.get("value") {
                    return Some(s.clone());
                }
            }
        }
    }
    None
}

fn html_attr_escape(s: &str) -> String { html_escape(s) }

fn html_attr_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s)] => Value::String(html_attr_escape(s)),
        [other] => match ev.eval(other.clone()) {
            Value::String(s) => Value::String(html_attr_escape(&s)),
            v => Value::Expr { head: Box::new(Value::Symbol("HtmlAttr".into())), args: vec![v] },
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("HtmlAttr".into())), args },
    }
}

fn safe_html_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s)] => Value::Assoc(std::collections::HashMap::from([
            ("__type".into(), Value::String("SafeHtml".into())),
            ("value".into(), Value::String(s.clone())),
        ])),
        [other] => match ev.eval(other.clone()) {
            Value::String(s) => Value::Assoc(std::collections::HashMap::from([
                ("__type".into(), Value::String("SafeHtml".into())),
                ("value".into(), Value::String(s.clone())),
            ])),
            v => Value::Expr { head: Box::new(Value::Symbol("SafeHtml".into())), args: vec![v] },
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("SafeHtml".into())), args },
    }
}

fn html_render_block(
    ev: &mut Evaluator,
    tpl: &str,
    stack: &mut Vec<Value>,
    partials: &std::collections::HashMap<String, String>,
    components: &std::collections::HashMap<String, String>,
    loader: Option<&Value>,
    blocks_collect: &mut Option<&mut std::collections::HashMap<String, String>>,
    blocks_lookup: Option<&std::collections::HashMap<String, String>>,
    xml_mode: bool,
    escape_html: bool,
    strict: bool,
    whitespace_mode: &str,
) -> String {
    fn trailing_indent(out: &str) -> String {
        let mut idx = out.len();
        let b = out.as_bytes();
        while idx > 0 {
            let c = b[idx - 1] as char;
            if c == ' ' || c == '\t' {
                idx -= 1;
                continue;
            }
            if c == '\n' { break; }
            return String::new();
        }
        out[idx..].to_string()
    }
    fn trim_standalone(out: &mut String, chars: &[char], i: &mut usize) {
        // Remove trailing spaces/tabs up to last newline
        while out.ends_with(' ') || out.ends_with('\t') { out.pop(); }
        // If next chars are spaces/tabs and optional \r then a \n, consume them
        let mut k = *i;
        while k < chars.len() && (chars[k] == ' ' || chars[k] == '\t') { k += 1; }
        if k < chars.len() && chars[k] == '\r' { k += 1; }
        if k < chars.len() && chars[k] == '\n' { k += 1; *i = k; }
    }
    let mut out = String::new();
    let chars: Vec<char> = tpl.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        if chars[i] == '{' && i + 1 < chars.len() && chars[i + 1] == '{' {
            let triple = i + 2 < chars.len() && chars[i + 2] == '{';
            let start = i + if triple { 3 } else { 2 };
            let mut j = start;
            while j < chars.len() {
                if !triple && j + 1 < chars.len() && chars[j] == '}' && chars[j + 1] == '}' { break; }
                if triple && j + 2 < chars.len() && chars[j] == '}' && chars[j + 1] == '}' && chars[j + 2] == '}' { break; }
                j += 1;
            }
            let content: String = chars[start..j].iter().collect();
            i = j + if triple { 3 } else { 2 };
            let tag = content.trim();
            let standalone = |t: &str| t.starts_with('!') || t.starts_with('>') || t.starts_with('#') || t.starts_with('^') || t.starts_with('/') || t.starts_with("block") || t.starts_with("yield");
            let mut did_output = false;
            let mut should_trim = false;
            let _ = should_trim; // avoid unused-assignment lint prior to first use
            if tag.starts_with('!') {
                if whitespace_mode == "trim-tags" || (whitespace_mode == "smart" && standalone(tag)) {
                    trim_standalone(&mut out, &chars, &mut i);
                }
                continue;
            }
            // Partials with optional props: {{> name}} or {{> name propsPath}}
            if tag.starts_with('>') {
                let body = tag[1..].trim();
                let mut it = body.split_whitespace();
                let name = it.next().unwrap_or("").trim();
                let props_path = it.next().map(|s| s.trim());
                let indent = trailing_indent(&out);
                // Resolve partial from map or via loader
                let mut src: Option<String> = partials.get(name).cloned();
                if src.is_none() {
                    if let Some(f) = loader {
                        let res = ev.eval(Value::Expr{ head: Box::new(f.clone()), args: vec![Value::String(name.into()), Value::String("partial".into())] });
                        if let Value::String(s) = res { src = Some(normalize_delimiters(&s, "{{", "}}")); }
                    }
                }
                if let Some(ptpl) = src {
                    let mut pushed = false;
                    if let Some(p) = props_path { if let Some(val) = resolve_path(stack, p) { stack.push(val); pushed = true; } }
                    let mut rendered = html_render_block(ev, &ptpl, stack, partials, components, loader, blocks_collect, blocks_lookup, xml_mode, escape_html, strict, whitespace_mode);
                    if !indent.is_empty() {
                        let mut acc = String::new();
                        for (idx, seg) in rendered.split_inclusive('\n').enumerate() {
                            let _ = idx; // suppress warning
                            acc.push_str(&indent);
                            acc.push_str(seg);
                        }
                        rendered = acc;
                    }
                    out.push_str(&rendered);
                    if pushed { stack.pop(); }
                    did_output = true;
                } else if strict {
                    out.push_str("<!-- Missing partial: "); out.push_str(name); out.push_str(" -->");
                    did_output = true;
                }
                should_trim = whitespace_mode == "trim-tags" || (whitespace_mode == "smart" && standalone(tag));
                if should_trim && did_output {
                    trim_standalone(&mut out, &chars, &mut i);
                }
                continue;
            }
            // Components: {{< Name propsPath?}} ... {{/Name}}
            if tag.starts_with('<') {
                let body = tag[1..].trim();
                let mut it = body.split_whitespace();
                let name = it.next().unwrap_or("").trim();
                let props_path = it.next().map(|s| s.trim());
                // find matching closing {{/Name}}
                let mut depth = 1i32;
                let mut k = i;
                let mut inner_start = i;
                    let _close_tag = format!("/{}", name);
                while k < chars.len() {
                    if chars[k] == '{' && k + 1 < chars.len() && chars[k + 1] == '{' {
                        let triple2 = k + 2 < chars.len() && chars[k + 2] == '{';
                        let start2 = k + if triple2 { 3 } else { 2 };
                        let mut j2 = start2;
                        while j2 < chars.len() {
                            if !triple2 && j2 + 1 < chars.len() && chars[j2] == '}' && chars[j2 + 1] == '}' { break; }
                            if triple2 && j2 + 2 < chars.len() && chars[j2] == '}' && chars[j2 + 1] == '}' && chars[j2 + 2] == '}' { break; }
                            j2 += 1;
                        }
                        let tag2: String = chars[start2..j2].iter().collect();
                        let tag2t = tag2.trim();
                        if tag2t.starts_with('<') && tag2t[1..].trim().starts_with(name) { depth += 1; }
                        else if tag2t.starts_with('/') && tag2t[1..].trim() == name { depth -= 1; if depth == 0 { inner_start = i; i = j2 + if triple2 { 3 } else { 2 }; break; } }
                        k = j2 + if triple2 { 3 } else { 2 };
                        continue;
                    }
                    k += 1;
                }
                let inner: String = chars[inner_start..(i)].iter().collect();
                // Resolve component
                let mut src: Option<String> = components.get(name).cloned();
                if src.is_none() {
                    if let Some(f) = loader {
                        let res = ev.eval(Value::Expr{ head: Box::new(f.clone()), args: vec![Value::String(name.into()), Value::String("component".into())] });
                        if let Value::String(s) = res { src = Some(normalize_delimiters(&s, "{{", "}}")); }
                    }
                }
                if let Some(ctpl) = src {
                    let mut pushed = false;
                    let mut local = std::collections::HashMap::new();
                    if let Some(p) = props_path { if let Some(val) = resolve_path(stack, p) { stack.push(val); pushed = true; } }
                    // Collect named slots: {{#slot "name"}}...{{/slot}} inside inner
                    let (default_slot_html, slots_map) = collect_named_slots(ev, &inner, stack, partials, components, loader, blocks_lookup, xml_mode, escape_html, strict, whitespace_mode);
                    local.insert("slot".to_string(), Value::Assoc(std::collections::HashMap::from([
                        ("__type".into(), Value::String("SafeHtml".into())),
                        ("value".into(), Value::String(default_slot_html)),
                    ])));
                    // Provide Slots assoc for named slots
                    let mut slots_assoc = std::collections::HashMap::new();
                    for (k, v) in slots_map.into_iter() {
                        slots_assoc.insert(k, Value::Assoc(std::collections::HashMap::from([
                            ("__type".into(), Value::String("SafeHtml".into())),
                            ("value".into(), Value::String(v)),
                        ])));
                    }
                    local.insert("Slots".to_string(), Value::Assoc(slots_assoc));
                    stack.push(Value::Assoc(local));
                    out.push_str(&html_render_block(ev, &ctpl, stack, partials, components, loader, blocks_collect, blocks_lookup, xml_mode, escape_html, strict, whitespace_mode));
                    stack.pop();
                    if pushed { stack.pop(); }
                    did_output = true;
                } else if strict {
                    out.push_str("<!-- Missing component: "); out.push_str(name); out.push_str(" -->");
                    did_output = true;
                }
                should_trim = whitespace_mode == "trim-tags" || (whitespace_mode == "smart" && true);
                if should_trim && did_output {
                    trim_standalone(&mut out, &chars, &mut i);
                }
                continue;
            }
            // Named blocks for layout: {{#block "name"}}...{{/block}}
            if tag.starts_with("block") {
                if let Some(mut rest) = tag.strip_prefix("block") {
                    rest = rest.trim();
                    let name = rest.trim_matches('"');
                    // find matching closing {{/block}}
                    let mut depth = 1i32; let mut k = i; let mut inner_start = i;
                    while k < chars.len() {
                        if chars[k]== '{' && k+1 < chars.len() && chars[k+1] == '{' {
                            let triple2 = k+2 < chars.len() && chars[k+2] == '{';
                            let start2 = k + if triple2 {3} else {2};
                            let mut j2 = start2;
                            while j2 < chars.len() { if !triple2 && j2+1 < chars.len() && chars[j2]=='}' && chars[j2+1]=='}' {break;} if triple2 && j2+2 < chars.len() && chars[j2]=='}' && chars[j2+1]=='}' && chars[j2+2]=='}' {break;} j2 += 1; }
                            let tag2: String = chars[start2..j2].iter().collect();
                            let tag2t = tag2.trim();
                            if tag2t.starts_with("block") { depth += 1; }
                            else if tag2t.starts_with('/') && tag2t[1..].trim() == "block" { depth -= 1; if depth==0 { inner_start = i; i = j2 + if triple2 {3} else {2}; break; } }
                            k = j2 + if triple2 {3} else {2};
                            continue;
                        }
                        k += 1;
                    }
                    let inner: String = chars[inner_start..(i)].iter().collect();
                    let rendered = html_render_block(ev, &inner, stack, partials, components, loader, blocks_collect, blocks_lookup, xml_mode, escape_html, strict, whitespace_mode);
                    if let Some(coll) = blocks_collect.as_deref_mut() { coll.insert(name.to_string(), rendered); }
                    // do not write block content inline
                    should_trim = whitespace_mode == "trim-tags" || (whitespace_mode == "smart" && true);
                    if should_trim { trim_standalone(&mut out, &chars, &mut i); }
                    continue;
                }
            }
            // Layout yields: {{yield}} or {{yield "name"}}
            if tag.starts_with("yield") {
                let arg = tag.strip_prefix("yield").map(|s| s.trim()).unwrap_or("");
                if !arg.is_empty() {
                    let name = arg.trim_matches('"');
                    if let Some(blk) = blocks_lookup.and_then(|m| m.get(name)) { out.push_str(blk); }
                } else {
                    // default yield uses PAGE_CONTENT from scope, prefer SafeHtml
                    if let Some(v) = resolve_path(stack, "PAGE_CONTENT") {
                        if let Some(s) = is_safe_html(&v) { out.push_str(&s); } else { let s = match v { Value::String(s) => s, other => lyra_core::pretty::format_value(&other) }; if escape_html { out.push_str(&html_escape(&s)); } else { out.push_str(&s); } }
                    }
                }
                should_trim = whitespace_mode == "trim-tags" || (whitespace_mode == "smart" && true);
                if should_trim { trim_standalone(&mut out, &chars, &mut i); }
                continue;
            }
            if tag.starts_with('#') || tag.starts_with('^') {
                // Reuse section logic from TemplateRender: copy minimal implementation
                let inverted = tag.starts_with('^');
                let key = tag[1..].trim();
                let mut depth = 1i32;
                let mut k = i;
                let mut inner_start = i;
                while k < chars.len() {
                    if chars[k] == '{' && k + 1 < chars.len() && chars[k + 1] == '{' {
                        let triple2 = k + 2 < chars.len() && chars[k + 2] == '{';
                        let start2 = k + if triple2 { 3 } else { 2 };
                        let mut j2 = start2;
                        while j2 < chars.len() {
                            if !triple2 && j2 + 1 < chars.len() && chars[j2] == '}' && chars[j2 + 1] == '}' { break; }
                            if triple2 && j2 + 2 < chars.len() && chars[j2] == '}' && chars[j2 + 1] == '}' && chars[j2 + 2] == '}' { break; }
                            j2 += 1;
                        }
                        let tag2: String = chars[start2..j2].iter().collect();
                        let tag2t = tag2.trim();
                        if tag2t.starts_with('#') && tag2t[1..].trim() == key { depth += 1; }
                        else if tag2t.starts_with('/') && tag2t[1..].trim() == key { depth -= 1; if depth == 0 { inner_start = i; i = j2 + if triple2 { 3 } else { 2 }; break; } }
                        k = j2 + if triple2 { 3 } else { 2 };
                        continue;
                    }
                    k += 1;
                }
                let inner: String = chars[inner_start..(i)].iter().collect();
                let val = resolve_path(stack, key).unwrap_or(Value::Symbol("Null".into()));
                if inverted {
                    if !truthy(&val) { out.push_str(&html_render_block(ev, &inner, stack, partials, components, loader, blocks_collect, blocks_lookup, xml_mode, escape_html, strict, whitespace_mode)); }
                } else {
                    match val {
                        Value::List(xs) => {
                            for item in xs { stack.push(item); out.push_str(&html_render_block(ev, &inner, stack, partials, components, loader, blocks_collect, blocks_lookup, xml_mode, escape_html, strict, whitespace_mode)); stack.pop(); }
                        }
                        Value::Assoc(_) => { stack.push(val); out.push_str(&html_render_block(ev, &inner, stack, partials, components, loader, blocks_collect, blocks_lookup, xml_mode, escape_html, strict, whitespace_mode)); stack.pop(); }
                        other => { if truthy(&other) { out.push_str(&html_render_block(ev, &inner, stack, partials, components, loader, blocks_collect, blocks_lookup, xml_mode, escape_html, strict, whitespace_mode)); } }
                    }
                }
                continue;
            }
            if tag.starts_with('/') { continue; }
            // variable with optional filter: name|Filter
            let mut parts = tag.split('|');
            let name = parts.next().unwrap().trim();
            let filt = parts.next().map(|s| s.trim()).filter(|s| !s.is_empty());
            let lookup = resolve_path(stack, name);
            if lookup.is_none() && strict {
                out.push_str("<!-- Missing var: "); out.push_str(name); out.push_str(" -->");
                continue;
            }
            let mut val = lookup.unwrap_or(Value::String(String::new()));
            // Unescaped with ampersand per Mustache: {{& name}}
            let (unescaped, name) = if name.starts_with('&') { (true, name[1..].trim()) } else { (false, name) };
            if unescaped { val = resolve_path(stack, name).unwrap_or(Value::String(String::new())); }
            if let Some(f) = filt { val = apply_filter(ev, f, val); }
            if let Some(safe) = is_safe_html(&val) {
                out.push_str(&safe);
            } else {
                let s = match val { Value::String(s) => s, other => lyra_core::pretty::format_value(&other), };
                if triple || unescaped || !escape_html { out.push_str(&s); } else { out.push_str(&(if xml_mode { xml_escape(&s) } else { html_escape(&s) })); }
            }
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

fn read_template_source(s: &str) -> Option<String> {
    let p = std::path::Path::new(s);
    if p.exists() && p.is_file() { std::fs::read_to_string(p).ok() } else { None }
}

fn normalize_delimiters(input: &str, initial_open: &str, initial_close: &str) -> String {
    // Convert custom delimiters and in-template delimiter changes into standard {{ }} tags.
    let mut out = String::new();
    let mut i = 0usize;
    let mut open = initial_open.to_string();
    let mut close = initial_close.to_string();
    let s = input;
    while i < s.len() {
        if let Some(pos) = s[i..].find(&open) {
            let start = i + pos;
            out.push_str(&s[i..start]);
            // triple-mustache only when using {{ }}
            let triple = open == "{{" && s[start..].starts_with("{{{");
            let (tag_start, tag_close_str) = if triple { (start + 3, String::from("}}}")) } else { (start + open.len(), close.clone()) };
            if let Some(end_rel) = s[tag_start..].find(tag_close_str.as_str()) {
                let end = tag_start + end_rel;
                let content = &s[tag_start..end];
                let tag = content.trim();
                // Delimiter change: {{= newOpen newClose =}}
                let is_change = !triple && tag.starts_with('=') && tag.ends_with('=');
                if is_change {
                    let inner = tag.trim_matches('=');
                    let mut parts = inner.split_whitespace().filter(|p| !p.is_empty());
                    if let (Some(nopen), Some(nclose)) = (parts.next(), parts.next()) {
                        open = nopen.to_string();
                        close = nclose.to_string();
                    }
                    // do not emit anything for delimiter change
                } else {
                    if triple {
                        out.push_str("{{{");
                        out.push_str(content);
                        out.push_str("}}}");
                    } else {
                        out.push_str("{{");
                        out.push_str(content);
                        out.push_str("}}");
                    }
                }
                i = end + tag_close_str.len();
            } else {
                // No closing found; emit rest and break
                out.push_str(&s[start..]);
                break;
            }
        } else {
            out.push_str(&s[i..]);
            break;
        }
    }
    out
}

fn html_template(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("HtmlTemplate".into())), args } }
    let tpl_in = match ev.eval(args[0].clone()) { Value::String(s) | Value::Symbol(s) => s, v => return Value::Expr { head: Box::new(Value::Symbol("HtmlTemplate".into())), args: vec![v, args[1].clone()] } };
    let data = ev.eval(args[1].clone());
    let opts = if args.len()>=3 { ev.eval(args[2].clone()) } else { Value::Assoc(std::collections::HashMap::new()) };
    let mut partials: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut components: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut layout: Option<String> = None;
    let mut escape_html_opt = true;
    let mut strict_opt = false;
    let mut whitespace_mode = String::from("smart");
    let mut loader_fn: Option<Value> = None;
    let mut xml_mode = false;
    let mut init_open = String::from("{{");
    let mut init_close = String::from("}}");
    if let Value::Assoc(m) = &opts {
        if let Some(Value::Assoc(p)) = m.get("Partials") { for (k,v) in p { if let Value::String(s)=v { if let Some(fs)=read_template_source(s) { partials.insert(k.clone(), normalize_delimiters(&fs, &init_open, &init_close)); } else { partials.insert(k.clone(), normalize_delimiters(s, &init_open, &init_close)); } } } }
        if let Some(Value::Assoc(p)) = m.get("Components") { for (k,v) in p { if let Value::String(s)=v { if let Some(fs)=read_template_source(s) { components.insert(k.clone(), normalize_delimiters(&fs, &init_open, &init_close)); } else { components.insert(k.clone(), normalize_delimiters(s, &init_open, &init_close)); } } } }
        if let Some(Value::String(s)) = m.get("Layout") { layout = Some(normalize_delimiters(&read_template_source(s).unwrap_or_else(|| s.clone()), &init_open, &init_close)); }
        if let Some(Value::Boolean(b)) = m.get("EscapeHtml") { escape_html_opt = *b; }
        if let Some(Value::Boolean(b)) = m.get("Strict") { strict_opt = *b; }
        if let Some(Value::String(s)) = m.get("Whitespace") { let low = s.to_ascii_lowercase(); if low=="preserve"||low=="trim-tags"||low=="smart" { whitespace_mode = low; } }
        if let Some(v) = m.get("Loader") { loader_fn = Some(v.clone()); }
        if let Some(Value::String(mode)) = m.get("Mode") { xml_mode = mode.to_ascii_lowercase() == "xml"; }
        if let Some(Value::Assoc(dm)) = m.get("Delimiters") {
            if let (Some(Value::String(op)), Some(Value::String(cl))) = (dm.get("Open"), dm.get("Close")) {
                init_open = op.clone(); init_close = cl.clone();
            }
        }
    }
    let tpl_raw = read_template_source(&tpl_in).unwrap_or(tpl_in);
    let tpl = normalize_delimiters(&tpl_raw, &init_open, &init_close);
    // Render page/body
    let mut stack: Vec<Value> = vec![data.clone()];
    let mut blocks: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut collect_opt = Some(&mut blocks);
    let body = html_render_block(ev, &tpl, &mut stack, &partials, &components, loader_fn.as_ref(), &mut collect_opt, None, xml_mode, escape_html_opt, strict_opt, &whitespace_mode);
    if let Some(layout_tpl_raw) = layout {
        let layout_tpl = normalize_delimiters(&layout_tpl_raw, &init_open, &init_close);
        // Provide PAGE_CONTENT as SafeHtml
        let mut top = match data { Value::Assoc(m) => m, other => std::collections::HashMap::from([(String::from("."), other)]) };
        top.insert("PAGE_CONTENT".into(), Value::Assoc(std::collections::HashMap::from([(String::from("__type"), Value::String("SafeHtml".into())), (String::from("value"), Value::String(body))])));
        let mut stack2 = vec![Value::Assoc(top)];
        let mut no_collect: Option<&mut std::collections::HashMap<String, String>> = None;
        let out = html_render_block(ev, &layout_tpl, &mut stack2, &partials, &components, loader_fn.as_ref(), &mut no_collect, Some(&blocks), xml_mode, escape_html_opt, strict_opt, &whitespace_mode);
        Value::String(out)
    } else {
        Value::String(body)
    }
}

fn html_template_compile(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 1 { return Value::Expr { head: Box::new(Value::Symbol("HtmlTemplateCompile".into())), args } }
    let tpl_in = match ev.eval(args[0].clone()) { Value::String(s) | Value::Symbol(s) => s, v => return Value::Expr { head: Box::new(Value::Symbol("HtmlTemplateCompile".into())), args: vec![v] } };
    let tpl = read_template_source(&tpl_in).unwrap_or(tpl_in);
    let opts = if args.len()>=2 { ev.eval(args[1].clone()) } else { Value::Assoc(std::collections::HashMap::new()) };
    Value::Assoc(std::collections::HashMap::from([
        ("__type".into(), Value::String("HtmlTemplate".into())),
        ("template".into(), Value::String(tpl)),
        ("opts".into(), opts),
    ]))
}

fn html_template_render(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("HtmlTemplateRender".into())), args } }
    let handle = ev.eval(args[0].clone());
    let data = ev.eval(args[1].clone());
    let extra_opts = if args.len()>=3 { ev.eval(args[2].clone()) } else { Value::Assoc(std::collections::HashMap::new()) };
    if let Value::Assoc(m) = handle {
        if matches!(m.get("__type"), Some(Value::String(t)) if t=="HtmlTemplate") {
            let tpl = match m.get("template") { Some(Value::String(s)) => s.clone(), _ => String::new() };
            let base_opts = match m.get("opts") { Some(Value::Assoc(mm)) => Value::Assoc(mm.clone()), _ => Value::Assoc(std::collections::HashMap::new()) };
            // Merge opts: extra overrides
            let merged = match (base_opts, extra_opts) {
                (Value::Assoc(mut a), Value::Assoc(b)) => { for (k,v) in b { a.insert(k,v); } Value::Assoc(a) },
                (a, _) => a,
            };
            return html_template(ev, vec![Value::String(tpl), data, merged]);
        }
    }
    Value::Expr { head: Box::new(Value::Symbol("HtmlTemplateRender".into())), args }
}

// Helper: collect named slot blocks and render them; return (default_html, slots)
fn collect_named_slots(
    ev: &mut Evaluator,
    src: &str,
    stack: &mut Vec<Value>,
    partials: &std::collections::HashMap<String, String>,
    components: &std::collections::HashMap<String, String>,
    loader: Option<&Value>,
    blocks_lookup: Option<&std::collections::HashMap<String, String>>,
    xml_mode: bool,
    escape_html: bool,
    strict: bool,
    whitespace_mode: &str,
) -> (String, std::collections::HashMap<String, String>) {
    let chars: Vec<char> = src.chars().collect();
    let mut i = 0usize;
    let mut ranges: Vec<(usize, usize, usize, usize, String)> = Vec::new(); // (block_start, block_end, inner_start, inner_end, name)
    while i < chars.len() {
        if chars[i] == '{' && i + 1 < chars.len() && chars[i + 1] == '{' {
            let triple = i + 2 < chars.len() && chars[i + 2] == '{';
            let start = i + if triple { 3 } else { 2 };
            let mut j = start;
            while j < chars.len() {
                if !triple && j + 1 < chars.len() && chars[j] == '}' && chars[j + 1] == '}' { break; }
                if triple && j + 2 < chars.len() && chars[j] == '}' && chars[j + 1] == '}' && chars[j + 2] == '}' { break; }
                j += 1;
            }
            let tag: String = chars[start..j].iter().collect();
            i = j + if triple { 3 } else { 2 };
            let t = tag.trim();
            if t.starts_with("#slot") {
                // parse name within quotes
                let name = t[5..].trim().trim_matches('"').to_string();
                // find closing {{/slot}}
                let mut depth = 1i32; let mut k = i; let inner_start = i; let mut inner_end = i; let mut end_pos = i;
                while k < chars.len() {
                    if chars[k] == '{' && k + 1 < chars.len() && chars[k + 1] == '{' {
                        let triple2 = k + 2 < chars.len() && chars[k + 2] == '{';
                        let start2 = k + if triple2 { 3 } else { 2 };
                        let mut j2 = start2;
                        while j2 < chars.len() { if !triple2 && j2 + 1 < chars.len() && chars[j2] == '}' && chars[j2 + 1] == '}' { break; } if triple2 && j2 + 2 < chars.len() && chars[j2] == '}' && chars[j2 + 1] == '}' && chars[j2 + 2] == '}' { break; } j2 += 1; }
                        let tag2: String = chars[start2..j2].iter().collect();
                        let tt = tag2.trim();
                        if tt.starts_with("#slot") { depth += 1; }
                        else if tt.starts_with("/slot") { depth -= 1; if depth == 0 { inner_end = k; end_pos = j2 + if triple2 { 3 } else { 2 }; i = end_pos; break; } }
                        k = j2 + if triple2 { 3 } else { 2 };
                        continue;
                    }
                    k += 1;
                }
                let block_start = start - if triple { 3 } else { 2 };
                ranges.push((block_start, end_pos, inner_start, inner_end, name));
                continue;
            }
        }
        // proceed
    }
    // Render each slot and build output with slots removed
    let mut slots: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    // Build a string with slot blocks removed
    let mut out = String::new();
    let mut cursor = 0usize;
    for (s, e, is, ie, name) in ranges.iter() {
        if *s > cursor { let seg: String = chars[cursor..*s].iter().collect(); out.push_str(&seg); }
        let inner_block: String = chars[*is..*ie].iter().collect();
        let rendered = html_render_block(ev, &inner_block, stack, partials, components, loader, &mut None, blocks_lookup, xml_mode, escape_html, strict, whitespace_mode);
        slots.insert(name.clone(), rendered);
        cursor = *e;
    }
    if cursor < chars.len() { let tail: String = chars[cursor..].iter().collect(); out.push_str(&tail); }
    // Now out contains template without slot blocks; render default slot
    let default_html = html_render_block(ev, &out, stack, partials, components, loader, &mut None, blocks_lookup, xml_mode, escape_html, strict, whitespace_mode);
    (default_html, slots)
}
