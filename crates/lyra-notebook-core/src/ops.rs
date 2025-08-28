use crate::ids::new_id_v7;
use crate::schema::{Assoc, Cell, CellAttrs, CellType, DisplayData, Notebook};
use uuid::Uuid;

#[derive(Debug, Clone, Default)]
pub struct NotebookCreateOpts {
    pub title: Option<String>,
    pub authors: Option<Vec<String>>,
    pub theme: Option<String>,
    pub default_language: Option<String>,
}

pub fn notebook_create(opts: NotebookCreateOpts) -> Notebook {
    let mut nb = Notebook::new(new_id_v7());
    // Minimal metadata defaults
    if let Some(title) = opts.title {
        nb.metadata.insert("title".into(), title.into());
    }
    if let Some(authors) = opts.authors {
        nb.metadata
            .insert("authors".into(), serde_json::Value::Array(authors.into_iter().map(Into::into).collect()));
    }
    if let Some(theme) = opts.theme {
        nb.metadata.insert("theme".into(), theme.into());
    }
    nb.metadata.insert(
        "defaultLanguage".into(),
        opts.default_language.unwrap_or_else(|| "Lyra".to_string()).into(),
    );
    nb
}

#[derive(Debug, Clone, Default)]
pub struct CellCreateOpts {
    pub language: Option<String>,
    pub attrs: Option<CellAttrs>,
    pub labels: Option<Vec<String>>,
    pub tags: Option<Vec<String>>,
    pub meta: Option<Assoc>,
}

pub fn cell_create(cell_type: CellType, input: impl Into<String>, opts: CellCreateOpts) -> Cell {
    let lang = opts.language.unwrap_or_else(|| match cell_type {
        CellType::Markdown => "Markdown".into(),
        CellType::Text => "Text".into(),
        _ => "Lyra".into(),
    });
    Cell {
        id: new_id_v7(),
        r#type: cell_type,
        language: lang,
        attrs: opts.attrs.unwrap_or_default(),
        labels: opts.labels.unwrap_or_default(),
        tags: opts.tags.unwrap_or_default(),
        input: input.into(),
        output: Vec::new(),
        meta: opts.meta.unwrap_or_default(),
    }
}

pub enum InsertPos {
    Index(usize),
    Before(Uuid),
    After(Uuid),
}

pub fn cell_insert(nb: &Notebook, cell: Cell, pos: InsertPos) -> Notebook {
    let mut out = nb.clone();
    match pos {
        InsertPos::Index(i) => {
            let idx = i.min(out.cells.len());
            out.cells.insert(idx, cell);
        }
        InsertPos::Before(id) => {
            let idx = out
                .cells
                .iter()
                .position(|c| c.id == id)
                .unwrap_or(out.cells.len());
            out.cells.insert(idx, cell);
        }
        InsertPos::After(id) => {
            let idx = out
                .cells
                .iter()
                .position(|c| c.id == id)
                .map(|i| i + 1)
                .unwrap_or(out.cells.len());
            out.cells.insert(idx, cell);
        }
    }
    out
}

pub fn cell_delete(nb: &Notebook, id: Uuid) -> Notebook {
    let mut out = nb.clone();
    if let Some(i) = out.cells.iter().position(|c| c.id == id) {
        out.cells.remove(i);
    }
    out
}

pub fn cell_move(nb: &Notebook, id: Uuid, to_index: usize) -> Notebook {
    let mut out = nb.clone();
    if let Some(i) = out.cells.iter().position(|c| c.id == id) {
        let cell = out.cells.remove(i);
        let idx = to_index.min(out.cells.len());
        out.cells.insert(idx, cell);
    }
    out
}

#[derive(Debug, Clone, Default)]
pub struct CellPatch {
    pub language: Option<String>,
    pub attrs: Option<CellAttrs>,
    pub labels: Option<Vec<String>>,
    pub tags: Option<Vec<String>>,
    pub input: Option<String>,
    pub meta: Option<Assoc>,
}

pub fn cell_update(nb: &Notebook, id: Uuid, patch: CellPatch) -> Notebook {
    let mut out = nb.clone();
    if let Some(c) = out.cells.iter_mut().find(|c| c.id == id) {
        if let Some(lang) = patch.language { c.language = lang; }
        if let Some(attrs) = patch.attrs { c.attrs = attrs; }
        if let Some(labels) = patch.labels { c.labels = labels; }
        if let Some(tags) = patch.tags { c.tags = tags; }
        if let Some(input) = patch.input { c.input = input; }
        if let Some(meta) = patch.meta { c.meta = meta; }
    }
    out
}

pub fn cells(nb: &Notebook) -> &Vec<Cell> { &nb.cells }

pub fn cell_by_id<'a>(nb: &'a Notebook, id: Uuid) -> Option<&'a Cell> {
    nb.cells.iter().find(|c| c.id == id)
}

pub fn cell_position(nb: &Notebook, id: Uuid) -> Option<usize> {
    nb.cells.iter().position(|c| c.id == id)
}

pub fn clear_outputs(nb: &Notebook, ids: Option<&[Uuid]>) -> Notebook {
    let mut out = nb.clone();
    match ids {
        None => {
            for c in &mut out.cells { c.output.clear(); }
        }
        Some(list) => {
            for c in &mut out.cells {
                if list.iter().any(|id| *id == c.id) { c.output.clear(); }
            }
        }
    }
    out
}

pub fn cell_attrs(nb: &Notebook, id: Uuid) -> Option<CellAttrs> {
    cell_by_id(nb, id).map(|c| c.attrs)
}

pub fn cell_set_attrs(nb: &Notebook, id: Uuid, attrs: CellAttrs) -> Notebook {
    cell_update(nb, id, CellPatch { attrs: Some(attrs), ..Default::default() })
}

pub fn append_output(nb: &Notebook, id: Uuid, item: DisplayData) -> Notebook {
    let mut out = nb.clone();
    if let Some(c) = out.cells.iter_mut().find(|c| c.id == id) {
        c.output.push(item);
    }
    out
}
