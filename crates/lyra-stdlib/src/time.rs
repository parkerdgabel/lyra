use lyra_core::value::Value;
#[cfg(feature = "tools")]
use crate::tool_spec;
#[cfg(feature = "tools")]
use crate::tools::add_specs;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration as StdDuration, Instant, SystemTime, UNIX_EPOCH};
// Import per-need within functions to avoid unused warnings

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

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

pub fn register_time(ev: &mut Evaluator) {
    ev.register("NowMs", now_ms as NativeFn, Attributes::empty());
    ev.register("MonotonicNow", monotonic_now as NativeFn, Attributes::empty());
    ev.register("Sleep", sleep_fn as NativeFn, Attributes::empty());
    ev.register("DateTime", date_time as NativeFn, Attributes::empty());
    ev.register("DateParse", date_parse as NativeFn, Attributes::empty());
    ev.register("DateFormat", date_format as NativeFn, Attributes::empty());
    ev.register("Duration", duration_fn as NativeFn, Attributes::empty());
    ev.register("DurationParse", duration_parse as NativeFn, Attributes::empty());
    ev.register("AddDuration", add_duration as NativeFn, Attributes::empty());
    ev.register("DiffDuration", diff_duration as NativeFn, Attributes::empty());
    ev.register("StartOf", start_of as NativeFn, Attributes::empty());
    ev.register("EndOf", end_of as NativeFn, Attributes::empty());
    ev.register("TimeZoneConvert", tz_convert as NativeFn, Attributes::empty());
    ev.register("ScheduleEvery", schedule_every as NativeFn, Attributes::HOLD_ALL);
    ev.register("Cron", cron_schedule as NativeFn, Attributes::HOLD_ALL);
    ev.register("CancelSchedule", cancel_schedule as NativeFn, Attributes::empty());

    #[cfg(feature = "tools")]
    add_specs(vec![
        tool_spec!("NowMs", summary: "Current UNIX time in milliseconds", params: [], tags: ["time","clock"], examples: [Value::String("NowMs[]  ==> 1690000000000".into())]),
        tool_spec!("MonotonicNow", summary: "Monotonic milliseconds since start", params: [], tags: ["time","clock"]),
        tool_spec!("Sleep", summary: "Sleep for N milliseconds", params: ["ms"], tags: ["time","sleep"]),
        tool_spec!("DateTime", summary: "Build/parse DateTime assoc (UTC)", params: ["spec"], tags: ["time","datetime"]),
        tool_spec!("DateParse", summary: "Parse date/time string to epochMs", params: ["s"], tags: ["time","datetime"]),
        tool_spec!("DateFormat", summary: "Format DateTime or epochMs to string", params: ["dt","fmt?"], tags: ["time","datetime"]),
        tool_spec!("Duration", summary: "Build Duration assoc from ms or fields", params: ["spec"], tags: ["time","duration"]),
        tool_spec!("DurationParse", summary: "Parse human duration (e.g., 1h30m)", params: ["s"], tags: ["time","duration"]),
        tool_spec!("AddDuration", summary: "Add duration to DateTime/epochMs", params: ["dt","dur"], tags: ["time","duration"]),
        tool_spec!("DiffDuration", summary: "Difference between DateTimes", params: ["a","b"], tags: ["time","duration"]),
        tool_spec!("StartOf", summary: "Start of unit (day/week/month)", params: ["dt","unit"], tags: ["time","calendar"]),
        tool_spec!("EndOf", summary: "End of unit (day/week/month)", params: ["dt","unit"], tags: ["time","calendar"]),
        tool_spec!("TimeZoneConvert", summary: "Convert DateTime to another timezone", params: ["dt","tz"], tags: ["time","tz"]),
        tool_spec!("ScheduleEvery", summary: "Schedule recurring task (held)", params: ["ms","body"], tags: ["time","schedule"], effects: ["schedule"]),
        tool_spec!("Cron", summary: "Schedule with cron expression (held)", params: ["expr","body"], tags: ["time","schedule","cron"], effects: ["schedule"]),
        tool_spec!("CancelSchedule", summary: "Cancel scheduled task", params: ["token"], tags: ["time","schedule"]),
    ]);
}

pub fn register_time_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str) -> bool) {
    crate::register_if(ev, pred, "NowMs", now_ms as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "MonotonicNow", monotonic_now as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "Sleep", sleep_fn as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "DateTime", date_time as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "DateParse", date_parse as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "DateFormat", date_format as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "Duration", duration_fn as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "DurationParse", duration_parse as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "AddDuration", add_duration as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "DiffDuration", diff_duration as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "StartOf", start_of as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "EndOf", end_of as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "TimeZoneConvert", tz_convert as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "ScheduleEvery", schedule_every as NativeFn, Attributes::HOLD_ALL);
    crate::register_if(ev, pred, "Cron", cron_schedule as NativeFn, Attributes::HOLD_ALL);
    crate::register_if(
        ev,
        pred,
        "CancelSchedule",
        cancel_schedule as NativeFn,
        Attributes::empty(),
    );
}

fn now_ms(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if !args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("NowMs".into())), args };
    }
    let ms =
        SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0);
    Value::Integer(ms)
}

static MONO_START: OnceLock<Instant> = OnceLock::new();
fn monotonic_now(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if !args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("MonotonicNow".into())), args };
    }
    let start = MONO_START.get_or_init(Instant::now);
    Value::Integer(start.elapsed().as_millis() as i64)
}

fn sleep_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("Sleep".into())), args };
    }
    let ms = match &args[0] {
        Value::Integer(n) if *n >= 0 => *n as u64,
        _ => 0,
    };
    std::thread::sleep(StdDuration::from_millis(ms));
    Value::Symbol("Null".into())
}

fn dt_assoc(epoch_ms: i64, tz: &str) -> Value {
    Value::Assoc(HashMap::from([
        ("__type".into(), Value::String("DateTime".into())),
        ("epochMs".into(), Value::Integer(epoch_ms)),
        ("timeZone".into(), Value::String(tz.to_string())),
    ]))
}

fn as_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) | Value::Symbol(s) => Some(s.clone()),
        _ => None,
    }
}

fn date_time(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("DateTime".into())), args };
    }
    let v = ev.eval(args[0].clone());
    // Accept assoc, ISO string, or list spec
    if let Value::Assoc(m) = &v {
        let tz = m.get("TimeZone").and_then(as_string).unwrap_or_else(|| "UTC".into());
        let epoch_ms = match m.get("epochMs") {
            Some(Value::Integer(ms)) => Some(*ms),
            _ => None,
        };
        if let Some(ms) = epoch_ms {
            return dt_assoc(ms, &tz);
        }
        // Build from fields in UTC
        let y = match m.get("Year") {
            Some(Value::Integer(n)) => *n as i32,
            _ => 1970,
        };
        let mon = match m.get("Month") {
            Some(Value::Integer(n)) => *n as u32,
            _ => 1,
        };
        let d = match m.get("Day") {
            Some(Value::Integer(n)) => *n as u32,
            _ => 1,
        };
        let hh = match m.get("Hour") {
            Some(Value::Integer(n)) => *n as u32,
            _ => 0,
        };
        let mm = match m.get("Minute") {
            Some(Value::Integer(n)) => *n as u32,
            _ => 0,
        };
        let ss = match m.get("Second") {
            Some(Value::Integer(n)) => *n as u32,
            _ => 0,
        };
        let ms = match m.get("Millisecond") {
            Some(Value::Integer(n)) => *n as u32,
            _ => 0,
        };
        let nd = chrono::NaiveDate::from_ymd_opt(y, mon.max(1).min(12), d.max(1).min(31));
        if let Some(nd) = nd {
            if let Some(ndt) = nd.and_hms_milli_opt(hh.min(23), mm.min(59), ss.min(59), ms.min(999))
            {
                let dt =
                    chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(ndt, chrono::Utc);
                return dt_assoc(dt.timestamp_millis(), &tz);
            }
        }
        return failure("Time::parse", "Invalid DateTime fields");
    }
    if let Some(s) = as_string(&v) {
        if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(&s) {
            return dt_assoc(dt.timestamp_millis(), dt.offset().to_string().as_str());
        }
        if let Ok(dt) = chrono::DateTime::parse_from_rfc2822(&s) {
            return dt_assoc(dt.timestamp_millis(), dt.offset().to_string().as_str());
        }
        if let Ok(ndt) = chrono::NaiveDateTime::parse_from_str(&s, "%Y-%m-%d %H:%M:%S%.3f") {
            let dt = chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(ndt, chrono::Utc);
            return dt_assoc(dt.timestamp_millis(), "UTC");
        }
        if let Ok(nd) = chrono::NaiveDate::parse_from_str(&s, "%Y-%m-%d") {
            if let Some(ndt) = nd.and_hms_opt(0, 0, 0) {
                let dt =
                    chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(ndt, chrono::Utc);
                return dt_assoc(dt.timestamp_millis(), "UTC");
            }
        }
        return failure("Time::parse", "Unsupported DateTime string");
    }
    if let Value::List(vs) = &v {
        // {y,m,d,hh,mm,ss,ms?}
        let g = |i: usize, def: i64| {
            vs.get(i)
                .and_then(|x| if let Value::Integer(n) = x { Some(*n) } else { None })
                .unwrap_or(def)
        };
        let y = g(0, 1970) as i32;
        let m = g(1, 1) as u32;
        let d = g(2, 1) as u32;
        let hh = g(3, 0) as u32;
        let mm = g(4, 0) as u32;
        let ss = g(5, 0) as u32;
        let ms = g(6, 0) as u32;
        if let Some(nd) = chrono::NaiveDate::from_ymd_opt(y, m.max(1).min(12), d.max(1).min(31)) {
            if let Some(ndt) = nd.and_hms_milli_opt(hh.min(23), mm.min(59), ss.min(59), ms.min(999))
            {
                let dt =
                    chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(ndt, chrono::Utc);
                return dt_assoc(dt.timestamp_millis(), "UTC");
            }
        }
        return failure("Time::parse", "Invalid DateTime list");
    }
    Value::Expr { head: Box::new(Value::Symbol("DateTime".into())), args }
}

fn date_parse(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    date_time(ev, args)
}

fn date_format(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("DateFormat".into())), args };
    }
    let dt = ev.eval(args[0].clone());
    let fmt = ev.eval(args[1].clone());
    let fmt_str = match fmt {
        Value::String(s) | Value::Symbol(s) => s,
        _ => "%+".into(),
    };
    let tz = match &dt {
        Value::Assoc(m) => m.get("timeZone").and_then(as_string).unwrap_or_else(|| "UTC".into()),
        _ => "UTC".into(),
    };
    let ms = match &dt {
        Value::Assoc(m) => {
            m.get("epochMs").and_then(|v| if let Value::Integer(n) = v { Some(*n) } else { None })
        }
        _ => None,
    };
    if let Some(ms) = ms {
        let secs = ms / 1000;
        let subms = (ms % 1000) as u32;
        if let Some(dt_utc) =
            chrono::DateTime::<chrono::Utc>::from_timestamp(secs, subms * 1_000_000)
        {
            // timezone conversion (basic: only UTC or offset like "+HH:MM")
            let out = if tz == "UTC" {
                dt_utc.format(&fmt_str).to_string()
            } else {
                if let Ok(ofs) = tz.parse::<chrono::FixedOffset>() {
                    dt_utc.with_timezone(&ofs).format(&fmt_str).to_string()
                } else {
                    dt_utc.format(&fmt_str).to_string()
                }
            };
            return Value::String(out);
        }
    }
    failure("Time::format", "Invalid DateTime value")
}

fn duration_assoc(ms: i64) -> Value {
    Value::Assoc(HashMap::from([
        ("__type".into(), Value::String("Duration".into())),
        ("ms".into(), Value::Integer(ms)),
    ]))
}

fn duration_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("Duration".into())), args };
    }
    let v = ev.eval(args[0].clone());
    if let Value::Assoc(m) = v {
        let mut ms: i64 = 0;
        if let Some(Value::Integer(d)) = m.get("Days") {
            ms += *d * 86_400_000;
        }
        if let Some(Value::Integer(h)) = m.get("Hours") {
            ms += *h * 3_600_000;
        }
        if let Some(Value::Integer(mn)) = m.get("Minutes") {
            ms += *mn * 60_000;
        }
        if let Some(Value::Integer(s)) = m.get("Seconds") {
            ms += *s * 1_000;
        }
        if let Some(Value::Integer(msec)) = m.get("Milliseconds") {
            ms += *msec;
        }
        return duration_assoc(ms);
    }
    if let Some(s) = as_string(&v) {
        // very small ISO-8601 subset: PnDTnHnMnS
        let p = s.trim();
        if p.starts_with('P') {
            let rest = &p[1..];
            let mut ms: i64 = 0;
            let mut in_time = false;
            let mut num = String::new();
            for ch in rest.chars() {
                match ch {
                    'T' => {
                        in_time = true;
                    }
                    '0'..='9' => num.push(ch),
                    'D' => {
                        if let Ok(n) = num.parse::<i64>() {
                            ms += n * 86_400_000;
                        }
                        num.clear();
                    }
                    'H' => {
                        if let Ok(n) = num.parse::<i64>() {
                            ms += n * 3_600_000;
                        }
                        num.clear();
                    }
                    'M' => {
                        if let Ok(n) = num.parse::<i64>() {
                            ms += if in_time { n * 60_000 } else { n * 30 * 86_400_000 };
                        }
                        num.clear();
                    }
                    'S' => {
                        if let Ok(n) = num.parse::<i64>() {
                            ms += n * 1_000;
                        }
                        num.clear();
                    }
                    _ => {}
                }
            }
            return duration_assoc(ms);
        }
        return failure("Time::parse", "Unsupported Duration string");
    }
    Value::Expr { head: Box::new(Value::Symbol("Duration".into())), args }
}

fn duration_parse(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    duration_fn(ev, args)
}

fn add_duration(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("AddDuration".into())), args };
    }
    let dt = ev.eval(args[0].clone());
    let dur = ev.eval(args[1].clone());
    let (ms, tz) = match dt {
        Value::Assoc(m) => (
            m.get("epochMs").and_then(|v| if let Value::Integer(n) = v { Some(*n) } else { None }),
            m.get("timeZone").and_then(as_string).unwrap_or_else(|| "UTC".into()),
        ),
        _ => (None, "UTC".into()),
    };
    let dms = match dur {
        Value::Assoc(m) => {
            m.get("ms").and_then(|v| if let Value::Integer(n) = v { Some(*n) } else { None })
        }
        _ => None,
    };
    match (ms, dms) {
        (Some(a), Some(b)) => dt_assoc(a + b, &tz),
        _ => failure("Time::format", "Invalid inputs"),
    }
}

fn diff_duration(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("DiffDuration".into())), args };
    }
    let a = ev.eval(args[0].clone());
    let b = ev.eval(args[1].clone());
    let unit = args.get(2).and_then(|v| as_string(v)).unwrap_or_else(|| "ms".into());
    let ams = match a {
        Value::Assoc(m) => {
            m.get("epochMs").and_then(|v| if let Value::Integer(n) = v { Some(*n) } else { None })
        }
        _ => None,
    };
    let bms = match b {
        Value::Assoc(m) => {
            m.get("epochMs").and_then(|v| if let Value::Integer(n) = v { Some(*n) } else { None })
        }
        _ => None,
    };
    if let (Some(x), Some(y)) = (ams, bms) {
        let d = x - y;
        return match unit.as_str() {
            "ms" => Value::Integer(d),
            "s" => Value::Real(d as f64 / 1000.0),
            "m" => Value::Real(d as f64 / 60_000.0),
            "h" => Value::Real(d as f64 / 3_600_000.0),
            _ => duration_assoc(d),
        };
    }
    failure("Time::format", "Invalid DateTime inputs")
}

fn start_end_internal(ev: &mut Evaluator, args: Vec<Value>, end: bool) -> Value {
    if args.len() < 2 {
        return Value::Expr {
            head: Box::new(Value::Symbol(if end { "EndOf" } else { "StartOf" }.into())),
            args,
        };
    }
    let dt = ev.eval(args[0].clone());
    let unit = as_string(&ev.eval(args[1].clone())).unwrap_or_else(|| "day".into());
    let (ms, tz) = match dt {
        Value::Assoc(m) => (
            m.get("epochMs").and_then(|v| if let Value::Integer(n) = v { Some(*n) } else { None }),
            m.get("timeZone").and_then(as_string).unwrap_or_else(|| "UTC".into()),
        ),
        _ => (None, "UTC".into()),
    };
    if let Some(ms) = ms {
        let secs = ms / 1000;
        let subms = (ms % 1000) as u32;
        if let Some(dt_utc) =
            chrono::DateTime::<chrono::Utc>::from_timestamp(secs, subms * 1_000_000)
        {
            let ndt = dt_utc.naive_utc();
            use chrono::{Datelike, Timelike};
            let mut y = ndt.year();
            let mut mo = ndt.month();
            let mut d = ndt.day();
            let mut hh = ndt.hour();
            let mut mm = ndt.minute();
            let mut ss = ndt.second();
            let mut msec = subms;
            match unit.as_str() {
                "year" => {
                    if end {
                        y += 1;
                        mo = 1;
                        d = 1;
                        hh = 0;
                        mm = 0;
                        ss = 0;
                        msec = 0;
                    } else {
                        mo = 1;
                        d = 1;
                        hh = 0;
                        mm = 0;
                        ss = 0;
                        msec = 0;
                    }
                }
                "month" => {
                    if end {
                        mo += 1;
                        d = 1;
                        hh = 0;
                        mm = 0;
                        ss = 0;
                        msec = 0;
                    } else {
                        d = 1;
                        hh = 0;
                        mm = 0;
                        ss = 0;
                        msec = 0;
                    }
                }
                "week" => {
                    let wd = ndt.weekday().num_days_from_monday() as i64;
                    let delta = if end { 6 - wd } else { -wd };
                    let base = ndt + chrono::Duration::days(delta);
                    let base = base.date().and_hms_milli_opt(0, 0, 0, 0).unwrap();
                    return dt_assoc(
                        chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(
                            base,
                            chrono::Utc,
                        )
                        .timestamp_millis(),
                        &tz,
                    );
                }
                "day" => {
                    if !end {
                        hh = 0;
                        mm = 0;
                        ss = 0;
                        msec = 0;
                    } else {
                        hh = 23;
                        mm = 59;
                        ss = 59;
                        msec = 999;
                    }
                }
                "hour" => {
                    if !end {
                        mm = 0;
                        ss = 0;
                        msec = 0;
                    } else {
                        mm = 59;
                        ss = 59;
                        msec = 999;
                    }
                }
                "minute" => {
                    if !end {
                        ss = 0;
                        msec = 0;
                    } else {
                        ss = 59;
                        msec = 999;
                    }
                }
                "second" => {
                    if !end {
                        msec = 0;
                    } else {
                        msec = 999;
                    }
                }
                _ => {}
            }
            let nd = chrono::NaiveDate::from_ymd_opt(y, mo, d)
                .unwrap_or(chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap());
            let ndt2 = nd.and_hms_milli_opt(hh, mm, ss, msec).unwrap();
            let dt = chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(ndt2, chrono::Utc);
            return dt_assoc(dt.timestamp_millis(), &tz);
        }
    }
    failure("Time::format", "Invalid DateTime value")
}

fn start_of(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    start_end_internal(ev, args, false)
}
fn end_of(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    start_end_internal(ev, args, true)
}

fn tz_convert(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("TimeZoneConvert".into())), args };
    }
    let dt = ev.eval(args[0].clone());
    let tz = as_string(&ev.eval(args[1].clone())).unwrap_or_else(|| "UTC".into());
    let ms = match &dt {
        Value::Assoc(m) => {
            m.get("epochMs").and_then(|v| if let Value::Integer(n) = v { Some(*n) } else { None })
        }
        _ => None,
    };
    match ms {
        Some(ms) => dt_assoc(ms, &tz),
        None => failure("Time::format", "Invalid DateTime value"),
    }
}

// ---------------- Scheduling ----------------

#[derive(Clone)]
struct Sched {
    #[allow(dead_code)]
    id: i64,
    cancelled: bool,
}
static SCHED_REG: OnceLock<Mutex<HashMap<i64, Sched>>> = OnceLock::new();
static NEXT_SCHED_ID: OnceLock<std::sync::atomic::AtomicI64> = OnceLock::new();
fn reg() -> &'static Mutex<HashMap<i64, Sched>> {
    SCHED_REG.get_or_init(|| Mutex::new(HashMap::new()))
}
fn next_id() -> i64 {
    let a = NEXT_SCHED_ID.get_or_init(|| std::sync::atomic::AtomicI64::new(1));
    a.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

fn schedule_handle(id: i64) -> Value {
    Value::Assoc(HashMap::from([
        ("__type".into(), Value::String("Schedule".into())),
        ("id".into(), Value::Integer(id)),
    ]))
}

fn get_handle_id(v: &Value) -> Option<i64> {
    if let Value::Assoc(m) = v {
        if m.get("__type") == Some(&Value::String("Schedule".into())) {
            if let Some(Value::Integer(id)) = m.get("id") {
                return Some(*id);
            }
        }
    }
    None
}

fn schedule_every(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("ScheduleEvery".into())), args };
    }
    let callable = args[0].clone();
    let dur_v = ev.eval(args[1].clone());
    let every_ms = match dur_v {
        Value::Assoc(m) => {
            m.get("ms").and_then(|v| if let Value::Integer(n) = v { Some(*n) } else { None })
        }
        Value::Integer(n) => Some(n),
        _ => None,
    };
    if let Some(ms) = every_ms {
        let id = next_id();
        {
            reg().lock().unwrap().insert(id, Sched { id, cancelled: false });
        }
        let opts =
            if args.len() >= 3 { ev.eval(args[2].clone()) } else { Value::Assoc(HashMap::new()) };
        let immediate = match &opts {
            Value::Assoc(m) => matches!(m.get("Immediate"), Some(Value::Boolean(true))),
            _ => false,
        };
        let max_runs = match &opts {
            Value::Assoc(m) => match m.get("MaxRuns") {
                Some(Value::Integer(n)) if *n > 0 => *n as i64,
                _ => i64::MAX,
            },
            _ => i64::MAX,
        };
        std::thread::spawn(move || {
            let mut runs = 0i64;
            if immediate {
                let mut ev2 = Evaluator::new();
                crate::register_all(&mut ev2);
                let _ = ev2.eval(Value::Expr { head: Box::new(callable.clone()), args: vec![] });
                runs += 1;
            }
            loop {
                std::thread::sleep(StdDuration::from_millis(ms as u64));
                // check cancellation
                let cancelled = {
                    if let Some(s) = reg().lock().unwrap().get(&id) {
                        s.cancelled
                    } else {
                        true
                    }
                };
                if cancelled || runs >= max_runs {
                    break;
                }
                let mut ev2 = Evaluator::new();
                crate::register_all(&mut ev2);
                let _ = ev2.eval(Value::Expr { head: Box::new(callable.clone()), args: vec![] });
                runs += 1;
            }
            reg().lock().unwrap().remove(&id);
        });
        return schedule_handle(id);
    }
    failure("Time::schedule", "Invalid duration for ScheduleEvery")
}

fn cron_schedule(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("Cron".into())), args };
    }
    let callable = args[0].clone();
    let expr = match ev.eval(args[1].clone()) {
        Value::String(s) | Value::Symbol(s) => s,
        _ => return failure("Time::schedule", "Cron: expected expression string"),
    };
    let opts =
        if args.len() >= 3 { ev.eval(args[2].clone()) } else { Value::Assoc(HashMap::new()) };
    let immediate = match &opts {
        Value::Assoc(m) => matches!(m.get("Immediate"), Some(Value::Boolean(true))),
        _ => false,
    };
    // Parse cron expression
    #[cfg(feature = "time_cron")]
    {
        let schedule = match expr.parse::<cron::Schedule>() {
            Ok(s) => s,
            Err(e) => return failure("Time::schedule", &format!("Cron parse: {}", e)),
        };
        let id = next_id();
        reg().lock().unwrap().insert(id, Sched { id, cancelled: false });
        std::thread::spawn(move || {
            if immediate {
                let mut ev2 = Evaluator::new();
                crate::register_all(&mut ev2);
                let _ = ev2.eval(Value::Expr { head: Box::new(callable.clone()), args: vec![] });
            }
            let mut upcoming = schedule.upcoming(chrono::Utc);
            loop {
                // Check cancellation before computing next
                if if let Some(s) = reg().lock().unwrap().get(&id) { s.cancelled } else { true } {
                    break;
                }
                let next_time = match upcoming.next() {
                    Some(dt) => dt,
                    None => break,
                };
                let now = chrono::Utc::now();
                let dur_ms = (next_time - now).num_milliseconds();
                if dur_ms > 0 {
                    std::thread::sleep(StdDuration::from_millis(dur_ms as u64));
                }
                if if let Some(s) = reg().lock().unwrap().get(&id) { s.cancelled } else { true } {
                    break;
                }
                let mut ev2 = Evaluator::new();
                crate::register_all(&mut ev2);
                let _ = ev2.eval(Value::Expr { head: Box::new(callable.clone()), args: vec![] });
            }
            reg().lock().unwrap().remove(&id);
        });
        return schedule_handle(id);
    }
    #[cfg(not(feature = "time_cron"))]
    {
        let _ = (callable, expr);
        failure("Time::schedule", "Cron feature disabled")
    }
}

fn cancel_schedule(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("CancelSchedule".into())), args };
    }
    if let Some(id) = get_handle_id(&args[0]) {
        let mut m = reg().lock().unwrap();
        if let Some(s) = m.get_mut(&id) {
            s.cancelled = true;
            return Value::Boolean(true);
        }
    }
    Value::Boolean(false)
}
