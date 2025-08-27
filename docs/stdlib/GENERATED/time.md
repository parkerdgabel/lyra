# TIME

| Function | Usage | Summary |
|---|---|---|
| `AddDuration` | `AddDuration[dt, dur]` | Add duration to DateTime/epochMs |
| `CancelSchedule` | `CancelSchedule[token]` | Cancel scheduled task |
| `Cron` | `Cron[expr, body]` | Schedule with cron expression (held) |
| `DateFormat` | `DateFormat[dt, fmt?]` | Format DateTime or epochMs to string |
| `DateParse` | `DateParse[s]` | Parse date/time string to epochMs |
| `DateTime` | `DateTime[spec]` | Build/parse DateTime assoc (UTC) |
| `DiffDuration` | `DiffDuration[a, b]` | Difference between DateTimes |
| `Duration` | `Duration[spec]` | Build Duration assoc from ms or fields |
| `DurationParse` | `DurationParse[s]` | Parse human duration (e.g., 1h30m) |
| `EndOf` | `EndOf[dt, unit]` | End of unit (day/week/month) |
| `MonotonicNow` | `MonotonicNow[]` | Monotonic clock milliseconds since start |
| `NowMs` | `NowMs[]` | Current UNIX time in milliseconds |
| `ScheduleEvery` | `ScheduleEvery[ms, body]` | Schedule recurring task (held) |
| `Sleep` | `Sleep[ms]` | Sleep for N milliseconds |
| `StartOf` | `StartOf[dt, unit]` | Start of unit (day/week/month) |
| `TimeZoneConvert` | `TimeZoneConvert[dt, tz]` | Convert DateTime to another timezone |

## `DateFormat`

- Usage: `DateFormat[dt, fmt?]`
- Summary: Format DateTime or epochMs to string
- Tags: time, datetime
- Examples:
  - `DateFormat[DateTime[<|"Year"->2024,"Month"->8,"Day"->1|>], "%Y-%m-%d"]  ==> "2024-08-01"`

## `DateTime`

- Usage: `DateTime[spec]`
- Summary: Build/parse DateTime assoc (UTC)
- Tags: time, datetime
- Examples:
  - `DateTime["2024-08-01T00:00:00Z"]  ==> <|"epochMs"->...|>`

## `DurationParse`

- Usage: `DurationParse[s]`
- Summary: Parse human duration (e.g., 1h30m)
- Tags: time, duration
- Examples:
  - `DurationParse["2h30m"]  ==> <|...|>`

## `NowMs`

- Usage: `NowMs[]`
- Summary: Current UNIX time in milliseconds
- Tags: time, clock
- Examples:
  - `NowMs[]  ==> 1710000000000`

## `Sleep`

- Usage: `Sleep[ms]`
- Summary: Sleep for N milliseconds
- Tags: time, sleep
- Examples:
  - `Sleep[100]  ==> Null`
