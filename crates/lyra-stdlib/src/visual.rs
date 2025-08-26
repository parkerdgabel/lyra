use lyra_core::value::Value;
use image::GenericImageView;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

#[cfg(feature = "tools")] use crate::tools::add_specs;
#[cfg(feature = "tools")] use crate::tool_spec;
#[cfg(feature = "tools")] use crate::schema_str;

pub fn register_visual(ev: &mut Evaluator) {
    ev.register("Chart", chart as NativeFn, Attributes::empty());
    ev.register("LinePlot", line_plot as NativeFn, Attributes::empty());
    ev.register("ScatterPlot", scatter_plot as NativeFn, Attributes::empty());
    ev.register("BarChart", bar_chart as NativeFn, Attributes::empty());
    ev.register("Histogram", histogram as NativeFn, Attributes::empty());
    ev.register("Figure", figure as NativeFn, Attributes::empty());

    #[cfg(feature = "tools")]
    add_specs(vec![
        tool_spec!("Chart", summary: "Render a chart from a spec", params: ["spec","opts"], tags: ["viz","chart"], output_schema: schema_str!()),
        tool_spec!("LinePlot", summary: "Render a line plot", params: ["data","opts"], tags: ["viz","plot"], output_schema: schema_str!()),
        tool_spec!("ScatterPlot", summary: "Render a scatter plot", params: ["data","opts"], tags: ["viz","plot"], output_schema: schema_str!()),
        tool_spec!("BarChart", summary: "Render a bar chart", params: ["data","opts"], tags: ["viz","chart"], output_schema: schema_str!()),
        tool_spec!("Histogram", summary: "Render a histogram", params: ["data","opts"], tags: ["viz","chart"], output_schema: schema_str!()),
        tool_spec!("Figure", summary: "Compose multiple charts in a grid", params: ["items","opts"], tags: ["viz","layout"], output_schema: schema_str!()),
    ]);
}

pub fn register_visual_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str)->bool) {
    crate::register_if(ev, pred, "Chart", chart as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "LinePlot", line_plot as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "ScatterPlot", scatter_plot as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "BarChart", bar_chart as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "Histogram", histogram as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "Figure", figure as NativeFn, Attributes::empty());
}

fn failure(tag: &str, msg: &str) -> Value {
    Value::Assoc(vec![
        ("message".to_string(), Value::String(msg.to_string())),
        ("tag".to_string(), Value::String(tag.to_string())),
    ].into_iter().collect())
}

fn as_f64(v: &Value) -> Option<f64> {
    match v { Value::Integer(i)=>Some(*i as f64), Value::Real(x)=>Some(*x), _=>None }
}

fn base64url_encode(data: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(data)
}

fn parse_color(v: Option<&Value>, default: [u8;4]) -> image::Rgba<u8> {
    if let Some(Value::String(s)) = v {
        let cs = s.trim();
        if let Some(hex) = cs.strip_prefix('#') {
            let bytes = hex.as_bytes();
            if bytes.len() == 6 || bytes.len() == 8 {
                let h = |i| -> u8 {
                    fn val(c: u8) -> u8 { match c { b'0'..=b'9' => c-b'0', b'a'..=b'f' => c-b'a'+10, b'A'..=b'F'=>c-b'A'+10, _=>0 } }
                    (val(bytes[i])<<4) | val(bytes[i+1])
                };
                let r = h(0); let g = h(2); let b = h(4); let a = if bytes.len()==8 { h(6) } else { 255 };
                return image::Rgba([r,g,b,a]);
            }
        }
    }
    image::Rgba(default)
}

fn encode_image(img: &image::DynamicImage, fmt: &str, quality: Option<u8>) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    match fmt.to_lowercase().as_str() {
        "png" => img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageOutputFormat::Png).map_err(|e| e.to_string())?,
        "jpeg" | "jpg" => img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageOutputFormat::Jpeg(quality.unwrap_or(85))).map_err(|e| e.to_string())?,
        other => return Err(format!("Unsupported format: {}", other)),
    }
    Ok(buf)
}

fn wrap_bytes_output(bytes: Vec<u8>, opts: Option<&std::collections::HashMap<String, Value>>) -> Value {
    // Currently return base64url string by default
    let enc = opts.and_then(|m| m.get("Output")).and_then(|v| if let Value::Assoc(mm)=v { mm.get("Encoding").cloned() } else { None })
                  .and_then(|v| if let Value::String(s)=v { Some(s) } else { None })
                  .unwrap_or_else(|| "base64url".into());
    match enc.as_str() { _ => Value::String(base64url_encode(&bytes)) }
}

fn pick_palette(n: usize) -> Vec<image::Rgba<u8>> {
    let base = vec![
        image::Rgba([31,119,180,255]), // blue
        image::Rgba([255,127,14,255]), // orange
        image::Rgba([44,160,44,255]),  // green
        image::Rgba([214,39,40,255]),  // red
        image::Rgba([148,103,189,255]),// purple
        image::Rgba([140,86,75,255]),  // brown
        image::Rgba([227,119,194,255]),// pink
        image::Rgba([127,127,127,255]),// gray
        image::Rgba([188,189,34,255]), // olive
        image::Rgba([23,190,207,255]), // teal
    ];
    let mut out = Vec::with_capacity(n);
    for i in 0..n { out.push(base[i % base.len()]); }
    out
}

struct Canvas {
    img: image::RgbaImage,
    w: u32,
    h: u32,
    left: u32,
    right: u32,
    top: u32,
    bottom: u32,
    fg: image::Rgba<u8>,
    bg: image::Rgba<u8>,
}

impl Canvas {
    fn new(width: u32, height: u32, bg: image::Rgba<u8>) -> Self {
        let mut img = image::RgbaImage::from_pixel(width, height, bg);
        Canvas { img, w: width, h: height, left: 40, right: 20, top: 20, bottom: 40, fg: image::Rgba([0,0,0,255]), bg }
    }
    fn plot_area(&self) -> (u32,u32,u32,u32) { (self.left, self.top, self.w-self.right, self.h-self.bottom) }
    fn put_px(&mut self, x: i32, y: i32, c: image::Rgba<u8>) {
        if x>=0 && y>=0 && (x as u32) < self.w && (y as u32) < self.h { self.img.put_pixel(x as u32, y as u32, c); }
    }
    fn draw_line(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, c: image::Rgba<u8>) {
        let (mut x0, mut y0, x1, y1) = (x0,y0,x1,y1);
        let dx = (x1-x0).abs(); let sx = if x0<x1 {1} else {-1};
        let dy = -(y1-y0).abs(); let sy = if y0<y1 {1} else {-1};
        let mut err = dx + dy;
        loop {
            self.put_px(x0,y0,c);
            if x0==x1 && y0==y1 { break; }
            let e2 = 2*err;
            if e2 >= dy { err += dy; x0 += sx; }
            if e2 <= dx { err += dx; y0 += sy; }
        }
    }
    fn fill_rect(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, c: image::Rgba<u8>) {
        let (mut x0, mut y0, mut x1, mut y1) = (x0,y0,x1,y1);
        if x0>x1 { std::mem::swap(&mut x0,&mut x1); }
        if y0>y1 { std::mem::swap(&mut y0,&mut y1); }
        let x0 = x0.max(0); let y0 = y0.max(0);
        let x1 = x1.min(self.w as i32 - 1); let y1 = y1.min(self.h as i32 - 1);
        for y in y0..=y1 { for x in x0..=x1 { self.put_px(x,y,c); } }
    }
    fn draw_marker_square(&mut self, xc: i32, yc: i32, size: i32, c: image::Rgba<u8>) {
        let r = size.max(1);
        self.fill_rect(xc-r, yc-r, xc+r, yc+r, c);
    }
    fn draw_axes(&mut self) {
        // Simple axes: bottom and left lines
        let (l,t,r,b) = self.plot_area();
        let x0 = l as i32; let x1 = r as i32; let y0 = b as i32; let y1 = t as i32;
        self.draw_line(x0, y0, x1, y0, self.fg);
        self.draw_line(x0, y0, x0, y1, self.fg);
    }
    fn draw_ticks(&mut self, xr: (f64,f64), yr: (f64,f64)) {
        let (l,t,r,b) = self.plot_area();
        // 5 ticks on each axis
        for i in 0..=4 {
            let tx = l + ((r - l) * i / 4);
            self.draw_line(tx as i32, b as i32, tx as i32, b as i32 - 4, self.fg);
            let ty = t + ((b - t) * i / 4);
            self.draw_line(l as i32, ty as i32, l as i32 + 4, ty as i32, self.fg);
        }
        let _ = (xr, yr); // reserved for future labeled ticks
    }
}

fn compute_range(vals: &[(f64,f64)]) -> (f64,f64,f64,f64) {
    let mut xmin = f64::INFINITY; let mut xmax = f64::NEG_INFINITY;
    let mut ymin = f64::INFINITY; let mut ymax = f64::NEG_INFINITY;
    for (x,y) in vals { if *x<xmin { xmin=*x; } if *x>xmax { xmax=*x; } if *y<ymin { ymin=*y; } if *y>ymax { ymax=*y; } }
    if !xmin.is_finite() || !xmax.is_finite() { xmin=0.0; xmax=1.0; }
    if !ymin.is_finite() || !ymax.is_finite() { ymin=0.0; ymax=1.0; }
    if xmax==xmin { xmax = xmin + 1.0; }
    if ymax==ymin { ymax = ymin + 1.0; }
    (xmin,xmax,ymin,ymax)
}

fn to_pixel(x: f64, y: f64, xr: (f64,f64), yr: (f64,f64), area: (u32,u32,u32,u32)) -> (i32,i32) {
    let (xmin,xmax) = xr; let (ymin,ymax) = yr; let (l,t,r,b) = area;
    let px = if xmax!=xmin { l as f64 + (x - xmin) / (xmax - xmin) * (r - l) as f64 } else { l as f64 };
    let py = if ymax!=ymin { b as f64 - (y - ymin) / (ymax - ymin) * (b - t) as f64 } else { b as f64 };
    (px.round() as i32, py.round() as i32)
}

// -------- Data normalization helpers --------

fn normalize_series(data: &Value) -> Option<Vec<(f64,f64)>> {
    match data {
        Value::List(xs) => {
            if xs.is_empty() { return Some(vec![]); }
            // Case 1: list of numbers => y; x = 1..n
            if xs.iter().all(|v| matches!(v, Value::Integer(_) | Value::Real(_))) {
                let mut out = Vec::with_capacity(xs.len());
                for (i,v) in xs.iter().enumerate() { out.push(((i+1) as f64, as_f64(v).unwrap())); }
                Some(out)
            } else if xs.iter().all(|v| matches!(v, Value::List(_))) {
                None // multi-series; caller should handle
            } else {
                // list of points: {x,y} or {x,y,z}
                let mut out: Vec<(f64,f64)> = Vec::with_capacity(xs.len());
                for v in xs {
                    match v {
                        Value::List(pair) if pair.len()>=2 => {
                            if let (Some(x), Some(y)) = (as_f64(&pair[0]), as_f64(&pair[1])) { out.push((x,y)); }
                        }
                        Value::Assoc(m) => {
                            let x = m.get("x").or_else(|| m.get("X")).and_then(|v| as_f64(v));
                            let y = m.get("y").or_else(|| m.get("Y")).and_then(|v| as_f64(v));
                            if let (Some(x), Some(y)) = (x,y) { out.push((x,y)); }
                        }
                        _ => {}
                    }
                }
                Some(out)
            }
        }
        _ => None
    }
}

fn normalize_multi_series(data: &Value) -> Option<Vec<Vec<(f64,f64)>>> {
    match data {
        Value::List(xs) if xs.iter().all(|v| matches!(v, Value::List(_))) => {
            let mut out: Vec<Vec<(f64,f64)>> = Vec::new();
            for s in xs { if let Some(series) = normalize_series(s) { out.push(series); } }
            Some(out)
        }
        Value::Assoc(m) => {
            let mut out: Vec<Vec<(f64,f64)>> = Vec::new();
            for (_k,v) in m { if let Some(series) = normalize_series(v) { out.push(series); } }
            Some(out)
        }
        _ => None
    }
}

fn is_dataset(v: &Value) -> bool {
    if let Value::Assoc(m) = v { matches!(m.get("__type"), Some(Value::String(s)) if s=="Dataset") } else { false }
}

fn head_rows(ev: &mut Evaluator, data: Value, n: usize) -> Vec<Value> {
    let call = Value::Expr { head: Box::new(Value::Symbol("Head".into())), args: vec![data, Value::Integer(n as i64)] };
    match ev.eval(call) { Value::List(rows) => rows, _ => Vec::new() }
}

fn collect_rows_from_data(ev: &mut Evaluator, data: &Value, sample: usize) -> Vec<Value> {
    match data {
        Value::List(rows) => rows.clone(),
        v if is_dataset(v) => head_rows(ev, v.clone(), sample),
        _ => Vec::new(),
    }
}

fn enc_str(m: &std::collections::HashMap<String, Value>, key: &str) -> Option<String> {
    m.get(key)
        .or_else(|| m.get(&key.to_string().to_lowercase()))
        .and_then(|v| match v { Value::String(s)|Value::Symbol(s) => Some(s.clone()), _=>None })
}

fn parse_encoded_line_or_scatter(
    ev: &mut Evaluator,
    data: &Value,
    enc: &std::collections::HashMap<String, Value>,
    sample: usize,
) -> Option<Vec<Vec<(f64,f64)>>> {
    let rows = collect_rows_from_data(ev, data, sample);
    if rows.is_empty() { return None; }
    let x_key = enc_str(enc, "X");
    let y_key = enc_str(enc, "Y");
    let color_key = enc_str(enc, "Color");
    let mut groups: std::collections::BTreeMap<String, Vec<(f64,f64)>> = std::collections::BTreeMap::new();
    let mut idx: usize = 1;
    for r in rows {
        if let Value::Assoc(m) = r {
            let x = if let Some(k) = &x_key { m.get(k).and_then(|v| as_f64(v)).unwrap_or(idx as f64) } else { idx as f64 };
            let y = if let Some(k) = &y_key { m.get(k).and_then(|v| as_f64(v)) } else { None };
            if let Some(y) = y {
                let g = if let Some(ck) = &color_key { match m.get(ck) { Some(Value::String(s))|Some(Value::Symbol(s)) => s.clone(), Some(v) => lyra_core::pretty::format_value(v), None => "".into() } } else { "".into() };
                groups.entry(g).or_default().push((x,y));
            }
            idx += 1;
        }
    }
    if groups.is_empty() { None } else { Some(groups.into_iter().map(|(_g, s)| s).collect()) }
}

fn parse_encoded_histogram(
    ev: &mut Evaluator,
    data: &Value,
    enc: &std::collections::HashMap<String, Value>,
    sample: usize,
) -> Option<Vec<f64>> {
    let rows = collect_rows_from_data(ev, data, sample);
    if rows.is_empty() { return None; }
    let val_key = enc_str(enc, "Values").or_else(|| enc_str(enc, "Y")).or_else(|| enc_str(enc, "Value"));
    let mut out: Vec<f64> = Vec::new();
    for r in rows {
        if let Value::Assoc(m) = r {
            if let Some(k) = &val_key { if let Some(v) = m.get(k).and_then(|v| as_f64(v)) { out.push(v); } }
        } else if let Some(v)=as_f64(&r) { out.push(v); }
    }
    if out.is_empty() { None } else { Some(out) }
}

fn parse_encoded_bar(
    ev: &mut Evaluator,
    data: &Value,
    enc: &std::collections::HashMap<String, Value>,
    sample: usize,
) -> Option<std::collections::HashMap<String, f64>> {
    let rows = collect_rows_from_data(ev, data, sample);
    if rows.is_empty() { return None; }
    let x_key = enc_str(enc, "X").or_else(|| enc_str(enc, "Label"));
    let y_key = enc_str(enc, "Y").or_else(|| enc_str(enc, "Value"));
    let mut agg: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for r in rows {
        if let Value::Assoc(m) = r {
            let label = x_key.as_ref().and_then(|k| match m.get(k) { Some(Value::String(s))|Some(Value::Symbol(s)) => Some(s.clone()), Some(v)=>Some(lyra_core::pretty::format_value(v)), None=>None });
            let y = y_key.as_ref().and_then(|k| m.get(k)).and_then(|v| as_f64(v));
            if let (Some(lbl), Some(val)) = (label, y) { *agg.entry(lbl).or_insert(0.0) += val; }
        }
    }
    if agg.is_empty() { None } else { Some(agg) }
}

fn parse_size_from_opts(opts: &std::collections::HashMap<String, Value>) -> (u32,u32) {
    let w = opts.get("Width").or_else(|| opts.get("width")).and_then(|v| if let Value::Integer(i)=v { Some(*i as u32) } else { None }).unwrap_or(640);
    let h = opts.get("Height").or_else(|| opts.get("height")).and_then(|v| if let Value::Integer(i)=v { Some(*i as u32) } else { None }).unwrap_or(400);
    (w,h)
}

fn encoding_format(opts: &std::collections::HashMap<String, Value>) -> (String, Option<u8>) {
    let enc = opts.get("Encoding").and_then(|v| if let Value::Assoc(m)=v { Some(m) } else { None });
    let fmt = enc.and_then(|m| m.get("format")).or_else(|| enc.and_then(|m| m.get("Format"))).and_then(|v| if let Value::String(s)=v { Some(s.clone()) } else { None }).unwrap_or_else(|| "png".into());
    let qual = enc.and_then(|m| m.get("quality")).or_else(|| enc.and_then(|m| m.get("Quality"))).and_then(|v| if let Value::Integer(i)=v { Some(*i as u8) } else { None });
    (fmt, qual)
}

// -------- Renderers --------

fn render_line_or_scatter(series: Vec<Vec<(f64,f64)>>, opts: &std::collections::HashMap<String, Value>, scatter_only: bool) -> Value {
    let (w,h) = parse_size_from_opts(opts);
    let bg = parse_color(opts.get("Background"), [255,255,255,255]);
    let mut c = Canvas::new(w,h,bg);
    c.draw_axes();
    // Combined ranges
    let mut all: Vec<(f64,f64)> = Vec::new();
    for s in &series { for p in s { all.push(*p); } }
    let (xmin,xmax,ymin,ymax) = compute_range(&all);
    let pal = pick_palette(series.len());
    let area = c.plot_area();
    for (si, s) in series.iter().enumerate() {
        let col = pal[si];
        let mut last: Option<(i32,i32)> = None;
        for (x,y) in s {
            let (px,py) = to_pixel(*x,*y,(xmin,xmax),(ymin,ymax),area);
            if scatter_only { c.draw_marker_square(px, py, 2, col); }
            else {
                if let Some((lx,ly)) = last { c.draw_line(lx,ly,px,py,col); }
                else { c.draw_marker_square(px,py,1,col); }
                last = Some((px,py));
            }
        }
    }
    c.draw_ticks((xmin,xmax),(ymin,ymax));
    let dynimg = image::DynamicImage::ImageRgba8(c.img);
    let (fmt,q) = encoding_format(opts);
    match encode_image(&dynimg, &fmt, q) { Ok(bytes)=>wrap_bytes_output(bytes, Some(opts)), Err(e)=>failure("Visual::encode", &e) }
}

fn render_bar(values: &std::collections::HashMap<String, f64>, opts: &std::collections::HashMap<String, Value>) -> Value {
    let (w,h) = parse_size_from_opts(opts);
    let bg = parse_color(opts.get("Background"), [255,255,255,255]);
    let mut c = Canvas::new(w,h,bg);
    c.draw_axes();
    if values.is_empty() { let dynimg = image::DynamicImage::ImageRgba8(c.img); let (fmt,q)=encoding_format(opts); return match encode_image(&dynimg,&fmt,q){Ok(b)=>wrap_bytes_output(b,Some(opts)), Err(e)=>failure("Visual::encode", &e)} }
    let mut items: Vec<(&String,f64)> = values.iter().map(|(k,v)| (k, *v)).collect();
    // Optional sort by Y
    if let Some(Value::String(s)) = opts.get("Sort").or_else(|| opts.get("sort")) { if s.eq_ignore_ascii_case("y") { items.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap()); } }
    let (xmin,xmax) = (0.5, (items.len() as f64) + 0.5);
    let (ymin,ymax) = {
        let mut ymin = 0.0; let mut ymax = f64::NEG_INFINITY; for (_k,v) in &items { if *v>ymax { ymax=*v; } }
        if ymax<=0.0 { ymax = 1.0; }
        (ymin,ymax)
    };
    let area = c.plot_area();
    let pal = pick_palette(1);
    let col = pal[0];
    let bar_w = 0.8; // relative width in x units
    for (i,(_k,v)) in items.iter().enumerate() {
        let x_center = (i as f64) + 1.0; // 1..n
        let x0 = x_center - bar_w/2.0; let x1 = x_center + bar_w/2.0;
        let (px0,py0) = to_pixel(x0, 0.0, (xmin,xmax), (ymin,ymax), area);
        let (px1,py1) = to_pixel(x1, *v, (xmin,xmax), (ymin,ymax), area);
        c.fill_rect(px0, py1, px1, py0, col);
    }
    let dynimg = image::DynamicImage::ImageRgba8(c.img);
    let (fmt,q) = encoding_format(opts);
    match encode_image(&dynimg, &fmt, q) { Ok(bytes)=>wrap_bytes_output(bytes, Some(opts)), Err(e)=>failure("Visual::encode", &e) }
}

fn render_histogram(vals: &[f64], opts: &std::collections::HashMap<String, Value>) -> Value {
    let bins = opts.get("Bins").or_else(|| opts.get("bins")).and_then(|v| if let Value::Integer(i)=v { Some(*i as usize) } else { None }).unwrap_or(10).max(1);
    if vals.is_empty() { return failure("Visual::histogram", "Empty data"); }
    let mut vmin = f64::INFINITY; let mut vmax = f64::NEG_INFINITY; for v in vals { if *v<vmin { vmin=*v; } if *v>vmax { vmax=*v; } }
    if vmin==vmax { vmax = vmin + 1.0; }
    let width = (vmax - vmin) / (bins as f64);
    let mut counts = vec![0usize; bins];
    for v in vals { let mut idx = ((v - vmin) / width).floor() as isize; if idx<0 { idx=0; } if idx as usize >= bins { idx = (bins-1) as isize; } counts[idx as usize]+=1; }
    let maxc = counts.iter().cloned().max().unwrap_or(1) as f64;
    // Render as bar chart on numeric x
    let (w,h) = parse_size_from_opts(opts);
    let bg = parse_color(opts.get("Background"), [255,255,255,255]);
    let mut c = Canvas::new(w,h,bg);
    c.draw_axes();
    let area = c.plot_area();
    let pal = pick_palette(1);
    let col = pal[0];
    for i in 0..bins {
        let x0 = vmin + (i as f64)*width; let x1 = x0 + width;
        let y = counts[i] as f64;
        let (px0,py0) = to_pixel(x0, 0.0, (vmin,vmax), (0.0,maxc), area);
        let (px1,py1) = to_pixel(x1, y, (vmin,vmax), (0.0,maxc), area);
        c.fill_rect(px0, py1, px1, py0, col);
    }
    c.draw_ticks((vmin,vmax),(0.0,maxc));
    let dynimg = image::DynamicImage::ImageRgba8(c.img);
    let (fmt,q) = encoding_format(opts);
    match encode_image(&dynimg, &fmt, q) { Ok(bytes)=>wrap_bytes_output(bytes, Some(opts)), Err(e)=>failure("Visual::encode", &e) }
}

// -------- API functions --------

fn chart(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Chart".into())), args } }
    let spec = match ev.eval(args[0].clone()) { Value::Assoc(m)=>m, other => return Value::Expr { head: Box::new(Value::Symbol("Chart".into())), args: vec![other] } };
    let mut opts: std::collections::HashMap<String, Value> = if args.len()>1 { match ev.eval(args[1].clone()) { Value::Assoc(m)=>m, _=>std::collections::HashMap::new() } } else { std::collections::HashMap::new() };
    // Merge spec.style/opts into opts (spec takes precedence)
    if let Some(Value::Assoc(style)) = spec.get("style").or_else(|| spec.get("Style")) { for (k,v) in style { opts.insert(k.clone(), v.clone()); } }
    if let Some(Value::Assoc(o2)) = spec.get("opts").or_else(|| spec.get("Opts")) { for (k,v) in o2 { opts.insert(k.clone(), v.clone()); } }
    let enc_map = spec.get("encoding").or_else(|| spec.get("Encoding")).and_then(|v| if let Value::Assoc(m)=v { Some(m.clone()) } else { None });
    let data_val = spec.get("data").or_else(|| spec.get("Data")).cloned();
    let sample = opts.get("Sample").or_else(|| opts.get("MaxPoints")).and_then(|v| if let Value::Integer(i)=v { Some((*i).max(1) as usize) } else { None }).unwrap_or(1000);
    let typ = spec.get("type").or_else(|| spec.get("Type")).and_then(|v| if let Value::String(s)=v { Some(s.to_lowercase()) } else { None }).unwrap_or_else(|| "line".into());
    match typ.as_str() {
        "line" => {
            if let (Some(data), Some(enc)) = (data_val.clone(), enc_map.clone()) {
                if let Some(ms) = parse_encoded_line_or_scatter(ev, &data, &enc, sample) { return render_line_or_scatter(ms, &opts, false); }
            }
            if let Some(ms) = spec.get("data").or_else(|| spec.get("Data")).and_then(|d| normalize_multi_series(d)) {
                return render_line_or_scatter(ms, &opts, false);
            }
            if let Some(s) = spec.get("data").or_else(|| spec.get("Data")).and_then(|d| normalize_series(d)) { return render_line_or_scatter(vec![s], &opts, false); }
            failure("Visual::data", "Unsupported data shape for line plot")
        }
        "scatter" => {
            if let (Some(data), Some(enc)) = (data_val.clone(), enc_map.clone()) {
                if let Some(ms) = parse_encoded_line_or_scatter(ev, &data, &enc, sample) { return render_line_or_scatter(ms, &opts, true); }
            }
            if let Some(ms) = spec.get("data").or_else(|| spec.get("Data")).and_then(|d| normalize_multi_series(d)) {
                return render_line_or_scatter(ms, &opts, true);
            }
            if let Some(s) = spec.get("data").or_else(|| spec.get("Data")).and_then(|d| normalize_series(d)) { return render_line_or_scatter(vec![s], &opts, true); }
            failure("Visual::data", "Unsupported data shape for scatter plot")
        }
        "bar" => {
            // Accept assoc label->value or list of {label,value}
            if let (Some(data), Some(enc)) = (data_val.clone(), enc_map.clone()) {
                if let Some(vals) = parse_encoded_bar(ev, &data, &enc, sample) { return render_bar(&vals, &opts); }
            }
            if let Some(Value::Assoc(m)) = spec.get("data").or_else(|| spec.get("Data")) {
                let mut vals: std::collections::HashMap<String,f64> = std::collections::HashMap::new();
                for (k,v) in m { if let Some(y) = as_f64(v) { vals.insert(k.clone(), y); } }
                return render_bar(&vals, &opts);
            }
            if let Some(Value::List(ls)) = spec.get("data").or_else(|| spec.get("Data")) {
                let mut vals: std::collections::HashMap<String,f64> = std::collections::HashMap::new();
                for it in ls { match it { Value::Assoc(m)=>{
                        if let (Some(Value::String(label)), Some(y)) = (m.get("label").or_else(|| m.get("Label")).cloned(), m.get("value").or_else(|| m.get("Value")).and_then(|v| as_f64(v))) {
                            vals.insert(label, y);
                        }
                    }, Value::List(pair) if pair.len()>=2 => {
                        if let (Some(label), Some(y)) = (match &pair[0] { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None }, as_f64(&pair[1])) { vals.insert(label, y); }
                    }, _=>{} }
                }
                return render_bar(&vals, &opts);
            }
            failure("Visual::data", "Unsupported data shape for bar chart")
        }
        "histogram" => {
            if let (Some(data), Some(enc)) = (data_val.clone(), enc_map.clone()) { if let Some(vals) = parse_encoded_histogram(ev, &data, &enc, sample) { return render_histogram(&vals, &opts); } }
            if let Some(Value::List(ls)) = spec.get("data").or_else(|| spec.get("Data")) {
                let mut vals: Vec<f64> = Vec::new(); for v in ls { if let Some(x)=as_f64(v) { vals.push(x); } }
                return render_histogram(&vals, &opts);
            }
            failure("Visual::data", "Unsupported data shape for histogram")
        }
        other => failure("Visual::type", &format!("Unsupported chart type: {}", other)),
    }
}

fn line_plot(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("LinePlot".into())), args } }
    let data = ev.eval(args[0].clone());
    let opts = if args.len()>1 { match ev.eval(args[1].clone()) { Value::Assoc(m)=>m, _=>std::collections::HashMap::new() } } else { std::collections::HashMap::new() };
    if let Some(ms) = normalize_multi_series(&data) { return render_line_or_scatter(ms, &opts, false); }
    if let Some(s) = normalize_series(&data) { return render_line_or_scatter(vec![s], &opts, false); }
    Value::Expr { head: Box::new(Value::Symbol("LinePlot".into())), args: vec![data] }
}

fn scatter_plot(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("ScatterPlot".into())), args } }
    let data = ev.eval(args[0].clone());
    let opts = if args.len()>1 { match ev.eval(args[1].clone()) { Value::Assoc(m)=>m, _=>std::collections::HashMap::new() } } else { std::collections::HashMap::new() };
    if let Some(ms) = normalize_multi_series(&data) { return render_line_or_scatter(ms, &opts, true); }
    if let Some(s) = normalize_series(&data) { return render_line_or_scatter(vec![s], &opts, true); }
    Value::Expr { head: Box::new(Value::Symbol("ScatterPlot".into())), args: vec![data] }
}

fn bar_chart(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("BarChart".into())), args } }
    let data = ev.eval(args[0].clone());
    let opts = if args.len()>1 { match ev.eval(args[1].clone()) { Value::Assoc(m)=>m, _=>std::collections::HashMap::new() } } else { std::collections::HashMap::new() };
    match data {
        Value::Assoc(m) => {
            let mut vals: std::collections::HashMap<String,f64> = std::collections::HashMap::new();
            for (k,v) in m { if let Some(y) = as_f64(&v) { vals.insert(k, y); } }
            render_bar(&vals, &opts)
        }
        Value::List(ls) => {
            let mut vals: std::collections::HashMap<String,f64> = std::collections::HashMap::new();
            for it in ls { match it { Value::Assoc(m)=>{
                    if let (Some(Value::String(label)), Some(y)) = (m.get("label").or_else(|| m.get("Label")).cloned(), m.get("value").or_else(|| m.get("Value")).and_then(|v| as_f64(v))) {
                        vals.insert(label, y);
                    }
                }, Value::List(pair) if pair.len()>=2 => {
                    if let (Some(label), Some(y)) = (match &pair[0] { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None }, as_f64(&pair[1])) { vals.insert(label, y); }
                }, _=>{} }
            }
            render_bar(&vals, &opts)
        }
        other => Value::Expr { head: Box::new(Value::Symbol("BarChart".into())), args: vec![other] }
    }
}

fn histogram(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Histogram".into())), args } }
    let data = ev.eval(args[0].clone());
    let opts = if args.len()>1 { match ev.eval(args[1].clone()) { Value::Assoc(m)=>m, _=>std::collections::HashMap::new() } } else { std::collections::HashMap::new() };
    match data {
        Value::List(ls) => { let mut vals: Vec<f64> = Vec::new(); for v in ls { if let Some(x)=as_f64(&v) { vals.push(x); } } render_histogram(&vals, &opts) }
        other => Value::Expr { head: Box::new(Value::Symbol("Histogram".into())), args: vec![other] }
    }
}

fn base64url_decode(s: &str) -> Result<Vec<u8>, String> {
    use base64::Engine as _;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(s).map_err(|e| e.to_string())
}

fn decode_image_value(ev: &mut Evaluator, v: Value) -> Result<image::DynamicImage, String> {
    match ev.eval(v) {
        Value::String(s) => {
            let bytes = base64url_decode(&s)?;
            image::load_from_memory(&bytes).map_err(|e| e.to_string())
        }
        Value::Assoc(m) => {
            if let Some(Value::String(b)) = m.get("bytes").or_else(|| m.get("Bytes")) {
                let bytes = base64url_decode(b)?;
                image::load_from_memory(&bytes).map_err(|e| e.to_string())
            } else { Err("Figure: unsupported image object".into()) }
        }
        other => Err(format!("Figure: unsupported item {:?}", other))
    }
}

fn figure(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Figure".into())), args } }
    let items_v = ev.eval(args[0].clone());
    let opts = if args.len()>1 { match ev.eval(args[1].clone()) { Value::Assoc(m)=>m, _=>std::collections::HashMap::new() } } else { std::collections::HashMap::new() };
    let cols_opt = opts.get("Cols").or_else(|| opts.get("cols")).and_then(|v| if let Value::Integer(i)=v { Some((*i).max(1) as usize) } else { None });
    let rows_opt = opts.get("Rows").or_else(|| opts.get("rows")).and_then(|v| if let Value::Integer(i)=v { Some((*i).max(1) as usize) } else { None });
    let spacing = opts.get("Spacing").or_else(|| opts.get("spacing")).and_then(|v| if let Value::Integer(i)=v { Some((*i).max(0) as u32) } else { None }).unwrap_or(8);
    let bg = parse_color(opts.get("Background"), [255,255,255,255]);
    let list = match items_v { Value::List(ls)=>ls, other=> return Value::Expr { head: Box::new(Value::Symbol("Figure".into())), args: vec![other] } };
    if list.is_empty() { return failure("Visual::figure", "No items"); }
    // Materialize each item: if Assoc spec, call Chart[spec]; if String, decode image
    let mut imgs: Vec<image::DynamicImage> = Vec::new();
    for it in list.into_iter() {
        let img = match &it {
            Value::Assoc(m) if m.get("type").is_some() || m.get("Type").is_some() => {
                let out = chart(ev, vec![Value::Assoc(m.clone())]);
                match decode_image_value(ev, out) { Ok(i)=>i, Err(e)=> return failure("Visual::figure", &e) }
            }
            _ => match decode_image_value(ev, it) { Ok(i)=>i, Err(e)=> return failure("Visual::figure", &e) }
        };
        imgs.push(img);
    }
    let n = imgs.len();
    let cols = cols_opt.unwrap_or_else(|| if let Some(r)=rows_opt { ((n + r - 1)/r).max(1) } else { (n as f64).sqrt().ceil() as usize });
    let rows = rows_opt.unwrap_or_else(|| ((n + cols - 1)/cols).max(1));
    // Compute max width per col and max height per row
    let mut colw = vec![0u32; cols];
    let mut rowh = vec![0u32; rows];
    for i in 0..n { let (w,h) = imgs[i].dimensions(); let r = i/cols; let c = i%cols; if w>colw[c] { colw[c]=w; } if h>rowh[r] { rowh[r]=h; } }
    let total_w: u32 = colw.iter().sum::<u32>() + spacing.saturating_mul(cols.saturating_sub(1) as u32) + 40; // padding for axes-like margin
    let total_h: u32 = rowh.iter().sum::<u32>() + spacing.saturating_mul(rows.saturating_sub(1) as u32) + 40;
    let mut canvas = image::RgbaImage::from_pixel(total_w, total_h, bg);
    // Paste
    let mut y = 20u32; // top padding
    for r in 0..rows {
        let mut x = 20u32; // left padding
        for c in 0..cols {
            let idx = r*cols + c; if idx>=n { break; }
            let (w,h) = imgs[idx].dimensions();
            image::imageops::overlay(&mut canvas, &imgs[idx].to_rgba8(), x as i64, y as i64);
            x += colw[c] + spacing;
        }
        y += rowh[r] + spacing;
    }
    let dynimg = image::DynamicImage::ImageRgba8(canvas);
    let (fmt,q) = encoding_format(&opts);
    match encode_image(&dynimg, &fmt, q) { Ok(bytes)=>wrap_bytes_output(bytes, Some(&opts)), Err(e)=>failure("Visual::encode", &e) }
}
