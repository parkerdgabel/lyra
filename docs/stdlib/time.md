Time and Scheduling

Overview
- Wall and monotonic time, DateTime/Duration construction, formatting/parsing.
- Periodic and cron-based scheduling with cancellable handles.

Functions
- NowMs[]: Integer epoch milliseconds (UTC wall clock).
- MonotonicNow[]: Integer milliseconds since process start (monotonic).
- Sleep[ms]: Block current thread for ms; returns Null.
- DateTime[x]: Construct or parse DateTime.
  - Accepts: ISO/RFC3339 string; list {y,m,d,hh,mm,ss,ms?}; assoc <|Year, Month, Day, Hour, Minute, Second, Millisecond, TimeZone|>.
  - Returns: <|__type:"DateTime", epochMs, timeZone|>.
- DateParse[str, opts]: Alias of DateTime for parse convenience.
- DateFormat[dt, fmt, opts]: Format dt to string.
  - fmt: strftime pattern (e.g., "%Y-%m-%dT%H:%M:%S.%3f%:z").
  - opts: <|TimeZone->"UTC"|> (basic fixed-offset support).
- Duration[spec]: Build durations.
  - Spec: <|Days, Hours, Minutes, Seconds, Milliseconds|> or ISO-8601-like (subset) string.
  - Returns: <|__type:"Duration", ms|>.
- DurationParse[str]: Parse string form; same result as Duration.
- AddDuration[dt, dur]: Shift dt by duration.
- DiffDuration[a, b, unit?]: Difference of two DateTimes.
  - unit: "ms" (default) | "s" | "m" | "h". Otherwise returns Duration.
- StartOf[dt, unit] / EndOf[dt, unit]: Floor/ceil dt to unit: second|minute|hour|day|week|month|year.
- TimeZoneConvert[dt, tz]: Return DateTime with same epochMs and provided timeZone string.

Scheduling
- ScheduleEvery[fnOrExpr, duration, opts] -> <|__type:"Schedule", id|>
  - duration: Duration[…] or integer ms.
  - opts: <|Immediate->True|False, MaxRuns->Integer, JitterMs->Integer|>.
  - Runs callable in a background thread using a fresh evaluator.
- Cron[fnOrExpr, cronExpr, opts] -> handle
  - cronExpr: standard 5/6 field spec (UTC). Enabled with feature time_cron (default).
  - opts: <|Immediate->True|False|>.
- CancelSchedule[handle]: Cancels either kind; returns True on success.

Failures
- Time::parse: invalid DateTime/Duration input.
- Time::format: invalid DateTime value or formatting.
- Time::schedule: invalid cron expression or scheduling error.

Examples
- NumberQ[NowMs[]] -> True
- DateTime[{2020,1,1}] -> <|__type:"DateTime", epochMs->1577836800000, timeZone->"UTC"|>
- DateFormat[DateTime["2020-01-01T00:00:00Z"], "%Y-%m-%d"] -> "2020-01-01"
- AddDuration[DateTime["2020-01-01"], Duration[<|Seconds->10|>]]
- s = ScheduleEvery[Function[Log["info","tick"]], Duration[<|Seconds->1|>], <|Immediate->True|>]
- c = Cron[Function[Log["info","hourly"]], "0 * * * *"]
- CancelSchedule[s]

