use lyra_core::value::Value;
use lyra_notebook_core as nb;
use lyra_notebook_core::schema::{CellAttrs, CellType};
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
use serde_json as sj;
use uuid::Uuid;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

pub fn register_notebook(ev: &mut Evaluator) {
    ev.register("NotebookCreate", notebook_create as NativeFn, Attributes::empty());
    ev.register("NotebookRead", notebook_read as NativeFn, Attributes::empty());
    ev.register("NotebookWrite", notebook_write as NativeFn, Attributes::empty());
    ev.register("Cells", cells as NativeFn, Attributes::empty());
    ev.register("CellCreate", cell_create as NativeFn, Attributes::empty());
    ev.register("CellInsert", cell_insert as NativeFn, Attributes::empty());
    ev.register("CellDelete", cell_delete as NativeFn, Attributes::empty());
    ev.register("CellMove", cell_move as NativeFn, Attributes::empty());
    ev.register("CellUpdate", cell_update as NativeFn, Attributes::empty());
    ev.register("ClearOutputs", clear_outputs as NativeFn, Attributes::empty());
    ev.register("NotebookMetadata", notebook_metadata as NativeFn, Attributes::empty());
    ev.register("NotebookSetMetadata", notebook_set_metadata as NativeFn, Attributes::empty());
    ev.register("FindByTag", find_by_tag as NativeFn, Attributes::empty());
    ev.register("FindCells", find_cells as NativeFn, Attributes::empty());
    ev.register("NotebookValidate", notebook_validate as NativeFn, Attributes::empty());

    ev.set_doc("NotebookCreate", "Create a new notebook association", &[]);
    ev.set_doc("NotebookRead", "Read a .lynb file into an association", &["path"]);
    ev.set_doc("NotebookWrite", "Write a notebook association to .lynb", &["notebook","path","opts?"]);
    ev.set_doc("FindByTag", "Find cell ids by tag", &["notebook","tag"]);
    ev.set_doc("FindCells", "Find cell ids by spec", &["notebook","spec"]);
    ev.set_doc("NotebookValidate", "Validate notebook structure", &["notebook"]);
}

fn value_to_json(v: &Value) -> Option<sj::Value> {
    Some(match v {
        Value::String(s) => sj::Value::String(s.clone()),
        Value::Symbol(s) => sj::Value::String(s.clone()),
        Value::Boolean(b) => sj::Value::Bool(*b),
        Value::Integer(n) => sj::Value::Number((*n).into()),
        Value::Real(f) => sj::json!(f),
        Value::List(xs) => sj::Value::Array(xs.iter().filter_map(value_to_json).collect()),
        Value::Assoc(m) => {
            let mut obj = sj::Map::new();
            for (k, v) in m.iter() {
                if let Some(j) = value_to_json(v) { obj.insert(k.clone(), j); }
            }
            sj::Value::Object(obj)
        }
        _ => return None,
    })
}

fn json_to_value(j: sj::Value) -> Value {
    match j {
        sj::Value::Null => Value::Symbol("Null".into()),
        sj::Value::Bool(b) => Value::Boolean(b),
        sj::Value::Number(n) => {
            if let Some(i) = n.as_i64() { Value::Integer(i) }
            else if let Some(f) = n.as_f64() { Value::Real(f) }
            else { Value::String(n.to_string()) }
        }
        sj::Value::String(s) => Value::String(s),
        sj::Value::Array(xs) => Value::List(xs.into_iter().map(json_to_value).collect()),
        sj::Value::Object(m) => {
            let mut out = lyra_core::value::AssocMap::new();
            for (k, v) in m {
                out.insert(k, json_to_value(v));
            }
            Value::Assoc(out)
        }
    }
}

fn parse_uuid(s: &str) -> Option<Uuid> { Uuid::try_parse(s).ok() }

fn parse_cell_type(v: &Value) -> Option<CellType> {
    let s = match v { Value::String(s) | Value::Symbol(s) => s.as_str(), _ => return None };
    match s {
        "Code" => Some(CellType::Code),
        "Markdown" => Some(CellType::Markdown),
        "Text" => Some(CellType::Text),
        "Output" => Some(CellType::Output),
        "Graphics" => Some(CellType::Graphics),
        "Table" => Some(CellType::Table),
        "Raw" => Some(CellType::Raw),
        _ => None,
    }
}

fn as_str(v: &Value) -> Option<&str> { if let Value::String(s) = v { Some(s) } else { None } }

fn parse_attrs_value(v: &Value) -> Option<CellAttrs> {
    match v {
        Value::Integer(n) => Some(CellAttrs::from_bits_truncate(*n as u32)),
        Value::List(xs) => {
            let mut flags = CellAttrs::default();
            for x in xs {
                let s = match x { Value::String(s) | Value::Symbol(s) => s.as_str(), _ => return None };
                match s {
                    "Collapsed" => flags |= CellAttrs::COLLAPSED,
                    "Initialization" => flags |= CellAttrs::INITIALIZATION,
                    "Hidden" => flags |= CellAttrs::HIDDEN,
                    "Locked" => flags |= CellAttrs::LOCKED,
                    "NoOutline" => flags |= CellAttrs::NO_OUTLINE,
                    "NoLineNumbers" => flags |= CellAttrs::NO_LINE_NUM,
                    _ => return None,
                }
            }
            Some(flags)
        }
        _ => None,
    }
}

fn notebook_create(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let mut opts = nb::ops::NotebookCreateOpts::default();
    if let Some(Value::Assoc(m)) = args.get(0) {
        if let Some(Value::String(s)) = m.get("Title") { opts.title = Some(s.clone()); }
        if let Some(Value::List(xs)) = m.get("Authors") {
            let mut a = Vec::new();
            for x in xs { if let Value::String(s) = x { a.push(s.clone()); } }
            opts.authors = Some(a);
        }
        if let Some(Value::String(s)) = m.get("Theme") { opts.theme = Some(s.clone()); }
        if let Some(Value::String(s)) = m.get("DefaultLanguage") { opts.default_language = Some(s.clone()); }
    }
    let nb = nb::ops::notebook_create(opts);
    json_to_value(sj::to_value(nb).unwrap())
}

fn notebook_read(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if let Some(Value::String(path)) = args.get(0) {
        match nb::io::read_notebook(path) {
            Ok(nb) => json_to_value(sj::to_value(nb).unwrap()),
            Err(e) => failure(&format!("NotebookRead: {}", e)),
        }
    } else {
        failure("NotebookRead: expected path string")
    }
}

fn notebook_write(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return failure("NotebookWrite: expected (nb, path, opts?)"); }
    let nbv = &args[0];
    let path = match &args[1] { Value::String(s) => s.clone(), _ => return failure("NotebookWrite: path must be String") };
    let nb: nb::schema::Notebook = match value_to_json(nbv) {
        Some(j) => match sj::from_value(j) { Ok(n) => n, Err(e) => return failure(&format!("NotebookWrite: invalid notebook: {}", e)) },
        None => return failure("NotebookWrite: nb must be an Association"),
    };
    let mut wo = nb::io::WriteOpts::default();
    if let Some(Value::Assoc(m)) = args.get(2) {
        if let Some(Value::Boolean(b)) = m.get("IncludeOutputs") { wo.include_outputs = *b; }
        if let Some(Value::Boolean(b)) = m.get("Pretty") { wo.pretty = *b; }
    }
    match nb::io::write_notebook(&path, &nb, wo) { Ok(_) => Value::String(path), Err(e) => failure(&format!("NotebookWrite: {}", e)) }
}

fn cells(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if let Some(nbv) = args.get(0) {
        let nb: nb::schema::Notebook = match value_to_json(nbv) {
            Some(j) => match sj::from_value(j) { Ok(n) => n, Err(e) => return failure(&format!("Cells: invalid notebook: {}", e)) },
            None => return failure("Cells: expected Notebook association"),
        };
        let j = sj::to_value(nb.cells).unwrap();
        json_to_value(j)
    } else {
        failure("Cells: expected notebook")
    }
}

fn cell_create(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return failure("CellCreate: expected (type, input, opts?)"); }
    let ctyp = match parse_cell_type(&args[0]) { Some(t) => t, None => return failure("CellCreate: invalid type") };
    let input = match &args[1] { Value::String(s) => s.clone(), _ => return failure("CellCreate: input must be String") };
    let mut copts = nb::ops::CellCreateOpts::default();
    if let Some(Value::Assoc(m)) = args.get(2) {
        if let Some(Value::String(s)) = m.get("Language") { copts.language = Some(s.clone()); }
        if let Some(v) = m.get("Attrs") { if let Some(f) = parse_attrs_value(v) { copts.attrs = Some(f); } }
        if let Some(Value::List(xs)) = m.get("Labels") { copts.labels = Some(xs.iter().filter_map(as_str).map(|s| s.to_string()).collect()); }
        if let Some(Value::List(xs)) = m.get("Tags") { copts.tags = Some(xs.iter().filter_map(as_str).map(|s| s.to_string()).collect()); }
    }
    let cell = nb::ops::cell_create(ctyp, input, copts);
    json_to_value(sj::to_value(cell).unwrap())
}

fn cell_insert(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 3 { return failure("CellInsert: expected (nb, cell, pos)"); }
    let nb: nb::schema::Notebook = match value_to_json(&args[0]) { Some(j) => match sj::from_value(j) { Ok(n) => n, Err(e) => return failure(&format!("CellInsert: invalid notebook: {}", e)) }, None => return failure("CellInsert: invalid nb") };
    let cell: nb::schema::Cell = match value_to_json(&args[1]) { Some(j) => match sj::from_value(j) { Ok(c) => c, Err(e) => return failure(&format!("CellInsert: invalid cell: {}", e)) }, None => return failure("CellInsert: invalid cell") };
    let pos = match &args[2] {
        Value::Integer(i) => nb::ops::InsertPos::Index(*i as usize),
        Value::Assoc(m) => {
            if let Some(Value::String(s)) = m.get("Before") { if let Some(id) = parse_uuid(s) { nb::ops::InsertPos::Before(id) } else { return failure("CellInsert: Before must be UUID"); } }
            else if let Some(Value::String(s)) = m.get("After") { if let Some(id) = parse_uuid(s) { nb::ops::InsertPos::After(id) } else { return failure("CellInsert: After must be UUID"); } }
            else { return failure("CellInsert: pos must be Integer or <|Before|After->uuid|>"); }
        }
        _ => return failure("CellInsert: pos must be Integer or Assoc"),
    };
    let nb2 = nb::ops::cell_insert(&nb, cell, pos);
    json_to_value(sj::to_value(nb2).unwrap())
}

fn cell_delete(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return failure("CellDelete: expected (nb, id)"); }
    let nb: nb::schema::Notebook = match value_to_json(&args[0]) { Some(j) => match sj::from_value(j) { Ok(n) => n, Err(e) => return failure(&format!("CellDelete: invalid notebook: {}", e)) }, None => return failure("CellDelete: invalid nb") };
    let id = match &args[1] { Value::String(s) => parse_uuid(s), _ => None };
    let id = match id { Some(u) => u, None => return failure("CellDelete: id must be UUID string") };
    let nb2 = nb::ops::cell_delete(&nb, id);
    json_to_value(sj::to_value(nb2).unwrap())
}

fn cell_move(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 3 { return failure("CellMove: expected (nb, id, toIndex)"); }
    let nb: nb::schema::Notebook = match value_to_json(&args[0]) { Some(j) => match sj::from_value(j) { Ok(n) => n, Err(e) => return failure(&format!("CellMove: invalid notebook: {}", e)) }, None => return failure("CellMove: invalid nb") };
    let id = match &args[1] { Value::String(s) => parse_uuid(s), _ => None };
    let id = match id { Some(u) => u, None => return failure("CellMove: id must be UUID string") };
    let to_index = match &args[2] { Value::Integer(i) => *i as usize, _ => return failure("CellMove: toIndex must be Integer") };
    let nb2 = nb::ops::cell_move(&nb, id, to_index);
    json_to_value(sj::to_value(nb2).unwrap())
}

fn cell_update(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 3 { return failure("CellUpdate: expected (nb, id, updates)"); }
    let nb: nb::schema::Notebook = match value_to_json(&args[0]) { Some(j) => match sj::from_value(j) { Ok(n) => n, Err(e) => return failure(&format!("CellUpdate: invalid notebook: {}", e)) }, None => return failure("CellUpdate: invalid nb") };
    let id = match &args[1] { Value::String(s) => parse_uuid(s), _ => None };
    let id = match id { Some(u) => u, None => return failure("CellUpdate: id must be UUID string") };
    let mut patch = nb::ops::CellPatch::default();
    match args.get(2) {
        Some(Value::Assoc(m)) => {
            if let Some(Value::String(s)) = m.get("Language") { patch.language = Some(s.clone()); }
            if let Some(v) = m.get("Attrs") { if let Some(f) = parse_attrs_value(v) { patch.attrs = Some(f); } }
            if let Some(Value::List(xs)) = m.get("Labels") { patch.labels = Some(xs.iter().filter_map(as_str).map(|s| s.to_string()).collect()); }
            if let Some(Value::List(xs)) = m.get("Tags") { patch.tags = Some(xs.iter().filter_map(as_str).map(|s| s.to_string()).collect()); }
            if let Some(Value::String(s)) = m.get("Input") { patch.input = Some(s.clone()); }
        }
        _ => return failure("CellUpdate: updates must be Association"),
    }
    let nb2 = nb::ops::cell_update(&nb, id, patch);
    json_to_value(sj::to_value(nb2).unwrap())
}

fn clear_outputs(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return failure("ClearOutputs: expected (nb, ids?)"); }
    let nb: nb::schema::Notebook = match value_to_json(&args[0]) { Some(j) => match sj::from_value(j) { Ok(n) => n, Err(e) => return failure(&format!("ClearOutputs: invalid notebook: {}", e)) }, None => return failure("ClearOutputs: invalid nb") };
    let nb2 = if let Some(Value::List(xs)) = args.get(1) {
        let ids: Vec<Uuid> = xs.iter().filter_map(|v| as_str(v).and_then(|s| parse_uuid(s))).collect();
        nb::ops::clear_outputs(&nb, Some(&ids))
    } else {
        nb::ops::clear_outputs(&nb, None)
    };
    json_to_value(sj::to_value(nb2).unwrap())
}

fn notebook_metadata(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if let Some(nbv) = args.get(0) {
        let nb: nb::schema::Notebook = match value_to_json(nbv) { Some(j) => match sj::from_value(j) { Ok(n) => n, Err(e) => return failure(&format!("NotebookMetadata: invalid notebook: {}", e)) }, None => return failure("NotebookMetadata: invalid nb") };
        json_to_value(sj::Value::Object(nb.metadata))
    } else { failure("NotebookMetadata: expected notebook") }
}

fn notebook_set_metadata(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return failure("NotebookSetMetadata: expected (nb, updates)"); }
    let mut nb: nb::schema::Notebook = match value_to_json(&args[0]) { Some(j) => match sj::from_value(j) { Ok(n) => n, Err(e) => return failure(&format!("NotebookSetMetadata: invalid notebook: {}", e)) }, None => return failure("NotebookSetMetadata: invalid nb") };
    let up = match value_to_json(&args[1]) { Some(sj::Value::Object(m)) => m, _ => return failure("NotebookSetMetadata: updates must be Assoc") };
    for (k, v) in up { nb.metadata.insert(k, v); }
    json_to_value(sj::to_value(nb).unwrap())
}

fn failure(msg: &str) -> Value {
    let mut m = lyra_core::value::AssocMap::new();
    m.insert("error".into(), Value::Boolean(true));
    m.insert("message".into(), Value::String(msg.to_string()));
    m.insert("tag".into(), Value::String("Notebook".into()));
    Value::Assoc(m)
}

// --- Find helpers and validate ---

fn find_by_tag(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return failure("FindByTag: expected (nb, tag)"); }
    let nb: nb::schema::Notebook = match value_to_json(&args[0]) { Some(j) => match sj::from_value(j) { Ok(n) => n, Err(e) => return failure(&format!("FindByTag: invalid notebook: {}", e)) }, None => return failure("FindByTag: invalid nb") };
    let tag = match &args[1] { Value::String(s) => s, Value::Symbol(s) => s, _ => return failure("FindByTag: tag must be String or Symbol") };
    let ids: Vec<Value> = nb.cells.iter().filter(|c| c.tags.iter().any(|t| t == tag)).map(|c| Value::String(c.id.to_string())).collect();
    Value::List(ids)
}

fn find_cells(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return failure("FindCells: expected (nb, spec)"); }
    let nb: nb::schema::Notebook = match value_to_json(&args[0]) { Some(j) => match sj::from_value(j) { Ok(n) => n, Err(e) => return failure(&format!("FindCells: invalid notebook: {}", e)) }, None => return failure("FindCells: invalid nb") };
    let spec = match &args[1] { Value::Assoc(m) => m, _ => return failure("FindCells: spec must be Association") };
    let type_ok = |c: &nb::schema::Cell| -> bool {
        if let Some(v) = spec.get("Type") {
            if let Some(t) = parse_cell_type(v) { return c.r#type == t; } else { return false; }
        }
        true
    };
    let lang_ok = |c: &nb::schema::Cell| -> bool {
        if let Some(Value::String(s)) = spec.get("Language") { return &c.language == s; }
        true
    };
    let has_attr_ok = |c: &nb::schema::Cell| -> bool {
        if let Some(v) = spec.get("HasAttr") { if let Some(f) = parse_attrs_value(v) { return c.attrs.intersects(f); } else { return false; } }
        true
    };
    let text_contains_ok = |c: &nb::schema::Cell| -> bool {
        if let Some(Value::String(substr)) = spec.get("TextContains") { return c.input.contains(substr); }
        true
    };
    let label_contains_ok = |c: &nb::schema::Cell| -> bool {
        if let Some(Value::String(substr)) = spec.get("LabelContains") { return c.labels.iter().any(|l| l.contains(substr)); }
        true
    };
    let ids: Vec<Value> = nb
        .cells
        .iter()
        .filter(|c| type_ok(c) && lang_ok(c) && has_attr_ok(c) && text_contains_ok(c) && label_contains_ok(c))
        .map(|c| Value::String(c.id.to_string()))
        .collect();
    Value::List(ids)
}

fn notebook_validate(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return failure("NotebookValidate: expected notebook"); }
    let nb: nb::schema::Notebook = match value_to_json(&args[0]) { Some(j) => match sj::from_value(j) { Ok(n) => n, Err(e) => return failure(&format!("NotebookValidate: invalid notebook: {}", e)) }, None => return failure("NotebookValidate: invalid nb") };
    let rep = nb::validate::validate_notebook(&nb);
    let mut m = lyra_core::value::AssocMap::new();
    m.insert("valid".into(), Value::Boolean(rep.valid));
    m.insert("errors".into(), Value::List(rep.errors.into_iter().map(Value::String).collect()));
    Value::Assoc(m)
}
