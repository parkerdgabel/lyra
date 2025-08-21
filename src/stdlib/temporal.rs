//! Date/Time and Temporal Operations for Lyra Standard Library
//!
//! This module provides comprehensive date/time manipulation, timezone operations,
//! calendar calculations, and duration arithmetic. All temporal types are implemented
//! as Foreign objects to maintain VM simplicity while providing rich functionality.
//!
//! # Features
//! - DateTime objects with timezone support
//! - Duration objects for time arithmetic  
//! - TimeZone objects for timezone conversions
//! - DateRange objects for iterating date sequences
//! - Business day calculations
//! - Date parsing and formatting
//! - Calendar operations (leap years, quarters, etc.)

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmError, VmResult};
use chrono::{DateTime, Datelike, Duration, NaiveDate, NaiveDateTime, TimeZone, Timelike, Utc, Weekday};
use chrono_tz::{Tz, TZ_VARIANTS};
use std::any::Any;

// ================================================================================================
// Foreign Object Definitions
// ================================================================================================

/// Foreign object representing a date/time value with timezone information
#[derive(Debug, Clone)]
pub struct DateTimeObject {
    /// The internal chrono DateTime<Tz> representation
    datetime: DateTime<Tz>,
}

impl DateTimeObject {
    /// Create a new DateTimeObject from a chrono DateTime
    pub fn new(datetime: DateTime<Tz>) -> Self {
        Self { datetime }
    }
    
    /// Create a DateTimeObject from components (year, month, day, hour, min, sec, timezone)
    pub fn from_components(
        year: i32,
        month: u32,
        day: u32,
        hour: u32,
        min: u32,
        sec: u32,
        timezone: Tz,
    ) -> Result<Self, ForeignError> {
        let naive = NaiveDate::from_ymd_opt(year, month, day)
            .and_then(|d| d.and_hms_opt(hour, min, sec))
            .ok_or_else(|| ForeignError::InvalidArgument(
                format!("Invalid date/time components: {}-{:02}-{:02} {:02}:{:02}:{:02}", 
                       year, month, day, hour, min, sec)
            ))?;
            
        let datetime = timezone
            .from_local_datetime(&naive)
            .single()
            .ok_or_else(|| ForeignError::RuntimeError {
                message: "Ambiguous or invalid local time".to_string(),
            })?;
            
        Ok(Self::new(datetime))
    }
    
    /// Create a DateTimeObject for just a date (midnight in given timezone)
    pub fn from_date_components(year: i32, month: u32, day: u32, timezone: Tz) -> Result<Self, ForeignError> {
        Self::from_components(year, month, day, 0, 0, 0, timezone)
    }
    
    /// Get the underlying DateTime value
    pub fn datetime(&self) -> &DateTime<Tz> {
        &self.datetime
    }
    
    /// Convert to different timezone
    pub fn with_timezone(&self, tz: &Tz) -> Self {
        Self::new(self.datetime.with_timezone(tz))
    }
    
    /// Add a duration
    pub fn add_duration(&self, duration: &DurationObject) -> Result<Self, ForeignError> {
        let new_dt = self.datetime + duration.duration();
        Ok(Self::new(new_dt))
    }
    
    /// Subtract a duration
    pub fn subtract_duration(&self, duration: &DurationObject) -> Result<Self, ForeignError> {
        let new_dt = self.datetime - duration.duration();
        Ok(Self::new(new_dt))
    }
    
    /// Calculate difference to another datetime
    pub fn difference_to(&self, other: &DateTimeObject) -> DurationObject {
        let diff = self.datetime.signed_duration_since(other.datetime);
        DurationObject::new(diff)
    }
}

impl Foreign for DateTimeObject {
    fn type_name(&self) -> &'static str {
        "DateTime"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "year" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity { 
                        method: method.to_string(), 
                        expected: 0, 
                        actual: args.len() 
                    });
                }
                Ok(Value::Integer(self.datetime.year() as i64))
            }
            "month" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity { 
                        method: method.to_string(), 
                        expected: 0, 
                        actual: args.len() 
                    });
                }
                Ok(Value::Integer(self.datetime.month() as i64))
            }
            "day" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity { 
                        method: method.to_string(), 
                        expected: 0, 
                        actual: args.len() 
                    });
                }
                Ok(Value::Integer(self.datetime.day() as i64))
            }
            "hour" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity { 
                        method: method.to_string(), 
                        expected: 0, 
                        actual: args.len() 
                    });
                }
                Ok(Value::Integer(self.datetime.hour() as i64))
            }
            "minute" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity { 
                        method: method.to_string(), 
                        expected: 0, 
                        actual: args.len() 
                    });
                }
                Ok(Value::Integer(self.datetime.minute() as i64))
            }
            "second" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity { 
                        method: method.to_string(), 
                        expected: 0, 
                        actual: args.len() 
                    });
                }
                Ok(Value::Integer(self.datetime.second() as i64))
            }
            "dayOfWeek" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity { 
                        method: method.to_string(), 
                        expected: 0, 
                        actual: args.len() 
                    });
                }
                // Convert Weekday to 1-7 where 1 = Monday, 7 = Sunday
                let day_num = match self.datetime.weekday() {
                    Weekday::Mon => 1,
                    Weekday::Tue => 2,
                    Weekday::Wed => 3,
                    Weekday::Thu => 4,
                    Weekday::Fri => 5,
                    Weekday::Sat => 6,
                    Weekday::Sun => 7,
                };
                Ok(Value::Integer(day_num))
            }
            "dayOfYear" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity { 
                        method: method.to_string(), 
                        expected: 0, 
                        actual: args.len() 
                    });
                }
                Ok(Value::Integer(self.datetime.ordinal() as i64))
            }
            "weekOfYear" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity { 
                        method: method.to_string(), 
                        expected: 0, 
                        actual: args.len() 
                    });
                }
                // ISO week number
                Ok(Value::Integer(self.datetime.iso_week().week() as i64))
            }
            "quarter" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity { 
                        method: method.to_string(), 
                        expected: 0, 
                        actual: args.len() 
                    });
                }
                let quarter = (self.datetime.month() - 1) / 3 + 1;
                Ok(Value::Integer(quarter as i64))
            }
            "timezone" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity { 
                        method: method.to_string(), 
                        expected: 0, 
                        actual: args.len() 
                    });
                }
                Ok(Value::LyObj(LyObj::new(Box::new(TimeZoneObject::new(self.datetime.timezone())))))
            }
            "unixTimestamp" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity { 
                        method: method.to_string(), 
                        expected: 0, 
                        actual: args.len() 
                    });
                }
                Ok(Value::Integer(self.datetime.timestamp()))
            }
            "formatISO" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity { 
                        method: method.to_string(), 
                        expected: 0, 
                        actual: args.len() 
                    });
                }
                Ok(Value::String(self.datetime.to_rfc3339()))
            }
            "format" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity { 
                        method: method.to_string(), 
                        expected: 1, 
                        actual: args.len() 
                    });
                }
                match &args[0] {
                    Value::String(format_str) => {
                        let formatted = self.datetime.format(format_str).to_string();
                        Ok(Value::String(formatted))
                    }
                    _ => Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "String".to_string(),
                        actual: format!("{:?}", args[0]),
                    })
                }
            }
            "isBusinessDay" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity { 
                        method: method.to_string(), 
                        expected: 0, 
                        actual: args.len() 
                    });
                }
                let is_business = !matches!(self.datetime.weekday(), Weekday::Sat | Weekday::Sun);
                Ok(Value::Boolean(is_business))
            }
            "isWeekend" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity { 
                        method: method.to_string(), 
                        expected: 0, 
                        actual: args.len() 
                    });
                }
                let is_weekend = matches!(self.datetime.weekday(), Weekday::Sat | Weekday::Sun);
                Ok(Value::Boolean(is_weekend))
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            })
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Foreign object representing a time duration
#[derive(Debug, Clone)]
pub struct DurationObject {
    /// The internal chrono Duration representation
    duration: Duration,
}

impl DurationObject {
    /// Create a new DurationObject from a chrono Duration
    pub fn new(duration: Duration) -> Self {
        Self { duration }
    }
    
    /// Create a duration from amount and unit
    pub fn from_amount_unit(amount: i64, unit: &str) -> Result<Self, ForeignError> {
        let duration = match unit.to_lowercase().as_str() {
            "second" | "seconds" | "sec" | "s" => Duration::seconds(amount),
            "minute" | "minutes" | "min" | "m" => Duration::minutes(amount),
            "hour" | "hours" | "h" => Duration::hours(amount),
            "day" | "days" | "d" => Duration::days(amount),
            "week" | "weeks" | "w" => Duration::weeks(amount),
            _ => return Err(ForeignError::InvalidArgument(
                format!("Unknown duration unit: {}", unit)
            ))
        };
        
        Ok(Self::new(duration))
    }
    
    /// Get the underlying Duration value
    pub fn duration(&self) -> Duration {
        self.duration
    }
    
    /// Add two durations
    pub fn add(&self, other: &DurationObject) -> Self {
        Self::new(self.duration + other.duration)
    }
    
    /// Subtract a duration from this one
    pub fn subtract(&self, other: &DurationObject) -> Self {
        Self::new(self.duration - other.duration)
    }
}

impl Foreign for DurationObject {
    fn type_name(&self) -> &'static str {
        "Duration"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "toSeconds" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity { 
                        method: method.to_string(), 
                        expected: 0, 
                        actual: args.len() 
                    });
                }
                Ok(Value::Integer(self.duration.num_seconds()))
            }
            "toMinutes" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity { 
                        method: method.to_string(), 
                        expected: 0, 
                        actual: args.len() 
                    });
                }
                Ok(Value::Integer(self.duration.num_minutes()))
            }
            "toHours" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity { 
                        method: method.to_string(), 
                        expected: 0, 
                        actual: args.len() 
                    });
                }
                Ok(Value::Integer(self.duration.num_hours()))
            }
            "toDays" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity { 
                        method: method.to_string(), 
                        expected: 0, 
                        actual: args.len() 
                    });
                }
                Ok(Value::Integer(self.duration.num_days()))
            }
            "toWeeks" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity { 
                        method: method.to_string(), 
                        expected: 0, 
                        actual: args.len() 
                    });
                }
                Ok(Value::Integer(self.duration.num_weeks()))
            }
            "abs" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity { 
                        method: method.to_string(), 
                        expected: 0, 
                        actual: args.len() 
                    });
                }
                let abs_duration = if self.duration < Duration::zero() {
                    Duration::zero() - self.duration
                } else {
                    self.duration
                };
                Ok(Value::LyObj(LyObj::new(Box::new(DurationObject::new(abs_duration)))))
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            })
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Foreign object representing a timezone
#[derive(Debug, Clone)]
pub struct TimeZoneObject {
    /// The internal chrono_tz::Tz representation
    timezone: Tz,
}

impl TimeZoneObject {
    /// Create a new TimeZoneObject
    pub fn new(timezone: Tz) -> Self {
        Self { timezone }
    }
    
    /// Create from timezone string
    pub fn from_string(tz_str: &str) -> Result<Self, ForeignError> {
        let timezone = tz_str.parse::<Tz>()
            .map_err(|_| ForeignError::InvalidArgument(
                format!("Invalid timezone: {}", tz_str)
            ))?;
        Ok(Self::new(timezone))
    }
    
    /// Get the underlying timezone
    pub fn timezone(&self) -> Tz {
        self.timezone
    }
}

impl Foreign for TimeZoneObject {
    fn type_name(&self) -> &'static str {
        "TimeZone"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "name" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity { 
                        method: method.to_string(), 
                        expected: 0, 
                        actual: args.len() 
                    });
                }
                Ok(Value::String(self.timezone.name().to_string()))
            }
            "isDST" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity { 
                        method: method.to_string(), 
                        expected: 1, 
                        actual: args.len() 
                    });
                }
                
                match &args[0] {
                    Value::LyObj(obj) => {
                        if let Some(dt_obj) = obj.downcast_ref::<DateTimeObject>() {
                            let dt_in_tz = dt_obj.datetime.with_timezone(&self.timezone);
                            // Check if DST is active by comparing offset names
                            let is_dst = dt_in_tz.offset().to_string().contains("DT") || dt_in_tz.offset().to_string().contains("DST");
                            Ok(Value::Boolean(is_dst))
                        } else {
                            Err(ForeignError::InvalidArgumentType {
                                method: method.to_string(),
                                expected: "DateTime".to_string(),
                                actual: obj.type_name().to_string(),
                            })
                        }
                    }
                    _ => Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "DateTime".to_string(),
                        actual: format!("{:?}", args[0]),
                    })
                }
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            })
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Foreign object for date ranges (iteration support)
#[derive(Debug, Clone)]
pub struct DateRangeObject {
    start: DateTime<Tz>,
    end: DateTime<Tz>,
    step: Duration,
}

impl DateRangeObject {
    /// Create a new DateRangeObject
    pub fn new(start: DateTime<Tz>, end: DateTime<Tz>, step: Duration) -> Self {
        Self { start, end, step }
    }
    
    /// Collect all dates in range
    pub fn collect(&self) -> Vec<DateTimeObject> {
        let mut dates = Vec::new();
        let mut current = self.start;
        
        if self.step > Duration::zero() {
            while current <= self.end {
                dates.push(DateTimeObject::new(current));
                current = current + self.step;
            }
        } else if self.step < Duration::zero() {
            while current >= self.end {
                dates.push(DateTimeObject::new(current));
                current = current + self.step;
            }
        }
        
        dates
    }
    
    /// Count the number of dates in range
    pub fn count(&self) -> i64 {
        if self.step == Duration::zero() {
            return 0;
        }
        
        let total_duration = if self.step > Duration::zero() {
            self.end.signed_duration_since(self.start)
        } else {
            self.start.signed_duration_since(self.end)
        };
        
        if total_duration < Duration::zero() {
            return 0;
        }
        
        let step_abs = if self.step > Duration::zero() {
            self.step
        } else {
            Duration::zero() - self.step
        };
        
        total_duration.num_seconds() / step_abs.num_seconds() + 1
    }
}

impl Foreign for DateRangeObject {
    fn type_name(&self) -> &'static str {
        "DateRange"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "length" | "count" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity { 
                        method: method.to_string(), 
                        expected: 0, 
                        actual: args.len() 
                    });
                }
                Ok(Value::Integer(self.count()))
            }
            "toList" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity { 
                        method: method.to_string(), 
                        expected: 0, 
                        actual: args.len() 
                    });
                }
                let dates = self.collect();
                let values = dates.into_iter()
                    .map(|dt| Value::LyObj(LyObj::new(Box::new(dt))))
                    .collect::<Vec<_>>();
                Ok(Value::List(values))
            }
            "start" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity { 
                        method: method.to_string(), 
                        expected: 0, 
                        actual: args.len() 
                    });
                }
                Ok(Value::LyObj(LyObj::new(Box::new(DateTimeObject::new(self.start)))))
            }
            "end" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity { 
                        method: method.to_string(), 
                        expected: 0, 
                        actual: args.len() 
                    });
                }
                Ok(Value::LyObj(LyObj::new(Box::new(DateTimeObject::new(self.end)))))
            }
            "step" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity { 
                        method: method.to_string(), 
                        expected: 0, 
                        actual: args.len() 
                    });
                }
                Ok(Value::LyObj(LyObj::new(Box::new(DurationObject::new(self.step)))))
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            })
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ================================================================================================
// Utility Functions
// ================================================================================================

/// Helper to extract DateTimeObject from Value
pub fn extract_datetime(value: &Value) -> Result<&DateTimeObject, VmError> {
    match value {
        Value::LyObj(obj) => {
            obj.downcast_ref::<DateTimeObject>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "DateTime".to_string(),
                    actual: obj.type_name().to_string(),
                })
        }
        _ => Err(VmError::TypeError {
            expected: "DateTime".to_string(),
            actual: format!("{:?}", value),
        })
    }
}

/// Helper to extract DurationObject from Value
pub fn extract_duration(value: &Value) -> Result<&DurationObject, VmError> {
    match value {
        Value::LyObj(obj) => {
            obj.downcast_ref::<DurationObject>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "Duration".to_string(),
                    actual: obj.type_name().to_string(),
                })
        }
        _ => Err(VmError::TypeError {
            expected: "Duration".to_string(),
            actual: format!("{:?}", value),
        })
    }
}

/// Helper to extract TimeZoneObject from Value
pub fn extract_timezone(value: &Value) -> Result<&TimeZoneObject, VmError> {
    match value {
        Value::LyObj(obj) => {
            obj.downcast_ref::<TimeZoneObject>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "TimeZone".to_string(),
                    actual: obj.type_name().to_string(),
                })
        }
        _ => Err(VmError::TypeError {
            expected: "TimeZone".to_string(),
            actual: format!("{:?}", value),
        })
    }
}

/// Helper to get system local timezone
pub fn get_local_timezone() -> Tz {
    // This is a simplified approach - in a real implementation you'd want to
    // detect the system timezone more robustly
    chrono_tz::UTC
}

/// Parse a date/time string with optional format
pub fn parse_datetime_string(date_str: &str, format_str: Option<&str>) -> Result<NaiveDateTime, VmError> {
    if let Some(format) = format_str {
        NaiveDateTime::parse_from_str(date_str, format)
            .map_err(|e| VmError::Runtime(
                format!("Failed to parse date '{}' with format '{}': {}", date_str, format, e)
            ))
    } else {
        // Try common formats
        let formats = &[
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S%.f%z",
        ];
        
        for format in formats {
            if let Ok(dt) = NaiveDateTime::parse_from_str(date_str, format) {
                return Ok(dt);
            }
        }
        
        // Try RFC 3339 parsing
        if let Ok(dt) = DateTime::parse_from_rfc3339(date_str) {
            return Ok(dt.naive_utc());
        }
        
        Err(VmError::Runtime(
            format!("Unable to parse date string: {}", date_str)
        ))
    }
}

/// Calculate business days between two dates
pub fn business_days_between(start: &DateTimeObject, end: &DateTimeObject) -> i64 {
    let start_date = start.datetime.naive_local().date();
    let end_date = end.datetime.naive_local().date();
    
    let mut count = 0;
    let mut current = start_date;
    
    while current <= end_date {
        if !matches!(current.weekday(), Weekday::Sat | Weekday::Sun) {
            count += 1;
        }
        current = current + chrono::Duration::days(1);
    }
    
    count
}

/// Add business days to a date (skipping weekends)
pub fn add_business_days_util(datetime: &DateTimeObject, days: i64) -> Result<DateTimeObject, VmError> {
    let mut current = datetime.datetime;
    let mut remaining = days.abs();
    let step = if days >= 0 { 1 } else { -1 };
    
    while remaining > 0 {
        current = current + Duration::days(step);
        if !matches!(current.weekday(), Weekday::Sat | Weekday::Sun) {
            remaining -= 1;
        }
    }
    
    Ok(DateTimeObject::new(current))
}

/// Check if a year is a leap year
pub fn is_leap_year(year: i32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

// ================================================================================================
// Core Date/Time Creation Functions
// ================================================================================================

/// Date[year, month, day] - Create date objects (midnight in local timezone)
pub fn date(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "exactly 3 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let year = match &args[0] {
        Value::Integer(y) => *y as i32,
        _ => return Err(VmError::TypeError {
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let month = match &args[1] {
        Value::Integer(m) => *m as u32,
        _ => return Err(VmError::TypeError {
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let day = match &args[2] {
        Value::Integer(d) => *d as u32,
        _ => return Err(VmError::TypeError {
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };
    
    let dt = DateTimeObject::from_date_components(year, month, day, get_local_timezone())
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    
    Ok(Value::LyObj(LyObj::new(Box::new(dt))))
}

/// DateTime[year, month, day, hour, minute, second] - Create datetime objects
pub fn datetime(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 || args.len() > 6 {
        return Err(VmError::TypeError {
            expected: "3-6 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let year = match &args[0] {
        Value::Integer(y) => *y as i32,
        _ => return Err(VmError::TypeError {
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let month = match &args[1] {
        Value::Integer(m) => *m as u32,
        _ => return Err(VmError::TypeError {
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let day = match &args[2] {
        Value::Integer(d) => *d as u32,
        _ => return Err(VmError::TypeError {
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };
    
    let hour = if args.len() > 3 {
        match &args[3] {
            Value::Integer(h) => *h as u32,
            _ => return Err(VmError::TypeError {
                expected: "Integer".to_string(),
                actual: format!("{:?}", args[3]),
            }),
        }
    } else { 0 };
    
    let minute = if args.len() > 4 {
        match &args[4] {
            Value::Integer(m) => *m as u32,
            _ => return Err(VmError::TypeError {
                expected: "Integer".to_string(),
                actual: format!("{:?}", args[4]),
            }),
        }
    } else { 0 };
    
    let second = if args.len() > 5 {
        match &args[5] {
            Value::Integer(s) => *s as u32,
            _ => return Err(VmError::TypeError {
                expected: "Integer".to_string(),
                actual: format!("{:?}", args[5]),
            }),
        }
    } else { 0 };
    
    let dt = DateTimeObject::from_components(year, month, day, hour, minute, second, get_local_timezone())
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    
    Ok(Value::LyObj(LyObj::new(Box::new(dt))))
}

/// Duration[amount, unit] - Create duration objects
pub fn duration(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let amount = match &args[0] {
        Value::Integer(a) => *a,
        _ => return Err(VmError::TypeError {
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let unit = match &args[1] {
        Value::String(u) => u,
        _ => return Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let dur = DurationObject::from_amount_unit(amount, unit)
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    
    Ok(Value::LyObj(LyObj::new(Box::new(dur))))
}

/// TimeZone[zone_id] - Create timezone objects
pub fn timezone(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let zone_id = match &args[0] {
        Value::String(z) => z,
        _ => return Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let tz = TimeZoneObject::from_string(zone_id)
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    
    Ok(Value::LyObj(LyObj::new(Box::new(tz))))
}

// ================================================================================================
// Current Time Functions
// ================================================================================================

/// Now[] - Current datetime with system timezone
pub fn now(args: &[Value]) -> VmResult<Value> {
    if !args.is_empty() {
        return Err(VmError::TypeError {
            expected: "no arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let now = Utc::now().with_timezone(&get_local_timezone());
    let dt = DateTimeObject::new(now);
    
    Ok(Value::LyObj(LyObj::new(Box::new(dt))))
}

/// Today[] - Current date (midnight)
pub fn today(args: &[Value]) -> VmResult<Value> {
    if !args.is_empty() {
        return Err(VmError::TypeError {
            expected: "no arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let now = Utc::now().with_timezone(&get_local_timezone());
    let today = now.date_naive().and_hms_opt(0, 0, 0).unwrap();
    let today_tz = get_local_timezone().from_local_datetime(&today).single()
        .ok_or(VmError::Runtime("Failed to create today's date".to_string()))?;
    let dt = DateTimeObject::new(today_tz);
    
    Ok(Value::LyObj(LyObj::new(Box::new(dt))))
}

/// UTCNow[] - Current UTC datetime
pub fn utc_now(args: &[Value]) -> VmResult<Value> {
    if !args.is_empty() {
        return Err(VmError::TypeError {
            expected: "no arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let now = Utc::now().with_timezone(&chrono_tz::UTC);
    let dt = DateTimeObject::new(now);
    
    Ok(Value::LyObj(LyObj::new(Box::new(dt))))
}

/// UnixTimestamp[] - Unix timestamp (seconds since epoch)
pub fn unix_timestamp(args: &[Value]) -> VmResult<Value> {
    if !args.is_empty() {
        return Err(VmError::TypeError {
            expected: "no arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let timestamp = Utc::now().timestamp();
    Ok(Value::Integer(timestamp))
}

// ================================================================================================
// Date Construction Functions
// ================================================================================================

/// DateParse[string, format] - Parse string to date/datetime
pub fn date_parse(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let date_str = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let format_str = if args.len() > 1 {
        match &args[1] {
            Value::String(f) => Some(f.as_str()),
            _ => return Err(VmError::TypeError {
                expected: "String".to_string(),
                actual: format!("{:?}", args[1]),
            }),
        }
    } else {
        None
    };
    
    let naive_dt = parse_datetime_string(date_str, format_str)?;
    let tz_dt = get_local_timezone().from_local_datetime(&naive_dt).single()
        .ok_or(VmError::Runtime(
            format!("Ambiguous or invalid local time: {}", naive_dt)
        ))?;
    let dt = DateTimeObject::new(tz_dt);
    
    Ok(Value::LyObj(LyObj::new(Box::new(dt))))
}

/// DateFromUnix[timestamp] - Create from Unix timestamp
pub fn date_from_unix(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let timestamp = match &args[0] {
        Value::Integer(ts) => *ts,
        _ => return Err(VmError::TypeError {
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let dt = DateTime::from_timestamp(timestamp, 0)
        .ok_or(VmError::Runtime( format!("Invalid Unix timestamp: {}", timestamp) 
        ))?
        .with_timezone(&get_local_timezone());
    
    Ok(Value::LyObj(LyObj::new(Box::new(DateTimeObject::new(dt)))))
}

/// DateFromDays[days] - Days since epoch
pub fn date_from_days(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let days = match &args[0] {
        Value::Integer(d) => *d,
        _ => return Err(VmError::TypeError {
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    // Unix epoch is 1970-01-01
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1)
        .ok_or(VmError::Runtime("Failed to create epoch date".to_string()))?;
    let target_date = epoch + chrono::Duration::days(days);
    let target_datetime = target_date.and_hms_opt(0, 0, 0)
        .ok_or(VmError::Runtime("Failed to create target datetime".to_string()))?;
    
    let tz_dt = get_local_timezone().from_local_datetime(&target_datetime).single()
        .ok_or(VmError::Runtime( "Failed to apply timezone to date".to_string() 
        ))?;
    
    Ok(Value::LyObj(LyObj::new(Box::new(DateTimeObject::new(tz_dt)))))
}

/// DateFromISOWeek[year, week, day] - ISO week date
pub fn date_from_iso_week(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "exactly 3 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let year = match &args[0] {
        Value::Integer(y) => *y as i32,
        _ => return Err(VmError::TypeError {
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let week = match &args[1] {
        Value::Integer(w) => *w as u32,
        _ => return Err(VmError::TypeError {
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let day = match &args[2] {
        Value::Integer(d) => *d as u32,
        _ => return Err(VmError::TypeError {
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };
    
    // Create from ISO week
    let target_date = NaiveDate::from_isoywd_opt(year, week, Weekday::try_from((day % 7) as u8).unwrap_or(Weekday::Mon))
        .ok_or(VmError::Runtime( format!("Invalid ISO week date: year={}, week={}, day={}", year, week, day) 
        ))?;
    
    let target_datetime = target_date.and_hms_opt(0, 0, 0)
        .ok_or(VmError::Runtime("Failed to create datetime from ISO week".to_string()))?;
    
    let tz_dt = get_local_timezone().from_local_datetime(&target_datetime).single()
        .ok_or(VmError::Runtime( "Failed to apply timezone to ISO week date".to_string() 
        ))?;
    
    Ok(Value::LyObj(LyObj::new(Box::new(DateTimeObject::new(tz_dt)))))
}

// ================================================================================================
// Date Manipulation Functions
// ================================================================================================

/// DateAdd[date, amount, unit] - Add time periods
pub fn date_add(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "exactly 3 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let datetime = extract_datetime(&args[0])?;
    
    let amount = match &args[1] {
        Value::Integer(a) => *a,
        _ => return Err(VmError::TypeError {
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let unit = match &args[2] {
        Value::String(u) => u,
        _ => return Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };
    
    let duration = DurationObject::from_amount_unit(amount, unit)
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let result = datetime.add_duration(&duration)
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    
    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}

/// DateSubtract[date, amount, unit] - Subtract time periods
pub fn date_subtract(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "exactly 3 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let datetime = extract_datetime(&args[0])?;
    
    let amount = match &args[1] {
        Value::Integer(a) => *a,
        _ => return Err(VmError::TypeError {
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let unit = match &args[2] {
        Value::String(u) => u,
        _ => return Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };
    
    let duration = DurationObject::from_amount_unit(amount, unit)
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let result = datetime.subtract_duration(&duration)
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    
    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}

/// DateDifference[date1, date2, unit] - Calculate difference
pub fn date_difference(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "exactly 3 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let datetime1 = extract_datetime(&args[0])?;
    let datetime2 = extract_datetime(&args[1])?;
    
    let unit = match &args[2] {
        Value::String(u) => u,
        _ => return Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };
    
    let diff = datetime1.difference_to(datetime2);
    
    let result = match unit.to_lowercase().as_str() {
        "second" | "seconds" | "sec" | "s" => diff.duration().num_seconds(),
        "minute" | "minutes" | "min" | "m" => diff.duration().num_minutes(),
        "hour" | "hours" | "h" => diff.duration().num_hours(),
        "day" | "days" | "d" => diff.duration().num_days(),
        "week" | "weeks" | "w" => diff.duration().num_weeks(),
        "year" | "years" | "y" => {
            // Approximate year calculation
            diff.duration().num_days() / 365
        }
        "month" | "months" => {
            // Approximate month calculation
            diff.duration().num_days() / 30
        }
        _ => return Err(VmError::Runtime(
            format!("Unknown duration unit: {}", unit)
        ))
    };
    
    Ok(Value::Integer(result))
}

/// DateTruncate[date, unit] - Truncate to unit
pub fn date_truncate(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let datetime = extract_datetime(&args[0])?;
    
    let unit = match &args[1] {
        Value::String(u) => u,
        _ => return Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let dt = datetime.datetime();
    let truncated = match unit.to_lowercase().as_str() {
        "second" | "seconds" | "sec" | "s" => {
            dt.with_nanosecond(0).unwrap()
        }
        "minute" | "minutes" | "min" | "m" => {
            dt.with_second(0).unwrap().with_nanosecond(0).unwrap()
        }
        "hour" | "hours" | "h" => {
            dt.with_minute(0).unwrap().with_second(0).unwrap().with_nanosecond(0).unwrap()
        }
        "day" | "days" | "d" => {
            dt.date_naive().and_hms_opt(0, 0, 0).unwrap().and_local_timezone(dt.timezone()).single().unwrap()
        }
        "month" | "months" => {
            let date = NaiveDate::from_ymd_opt(dt.year(), dt.month(), 1).unwrap();
            date.and_hms_opt(0, 0, 0).unwrap().and_local_timezone(dt.timezone()).single().unwrap()
        }
        "year" | "years" | "y" => {
            let date = NaiveDate::from_ymd_opt(dt.year(), 1, 1).unwrap();
            date.and_hms_opt(0, 0, 0).unwrap().and_local_timezone(dt.timezone()).single().unwrap()
        }
        _ => return Err(VmError::Runtime(
            format!("Unknown truncation unit: {}", unit)
        ))
    };
    
    Ok(Value::LyObj(LyObj::new(Box::new(DateTimeObject::new(truncated)))))
}

/// DateRound[date, unit] - Round to nearest unit
pub fn date_round(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let datetime = extract_datetime(&args[0])?;
    
    let unit = match &args[1] {
        Value::String(u) => u,
        _ => return Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let dt = datetime.datetime();
    let rounded = match unit.to_lowercase().as_str() {
        "second" | "seconds" | "sec" | "s" => {
            // Round nanoseconds to nearest second
            if dt.nanosecond() >= 500_000_000 {
                (*dt + Duration::seconds(1)).with_nanosecond(0).unwrap()
            } else {
                dt.with_nanosecond(0).unwrap()
            }
        }
        "minute" | "minutes" | "min" | "m" => {
            if dt.second() >= 30 {
                (*dt + Duration::minutes(1)).with_second(0).unwrap().with_nanosecond(0).unwrap()
            } else {
                dt.with_second(0).unwrap().with_nanosecond(0).unwrap()
            }
        }
        "hour" | "hours" | "h" => {
            if dt.minute() >= 30 {
                (*dt + Duration::hours(1)).with_minute(0).unwrap().with_second(0).unwrap().with_nanosecond(0).unwrap()
            } else {
                dt.with_minute(0).unwrap().with_second(0).unwrap().with_nanosecond(0).unwrap()
            }
        }
        "day" | "days" | "d" => {
            if dt.hour() >= 12 {
                let next_day = dt.date_naive() + chrono::Duration::days(1);
                next_day.and_hms_opt(0, 0, 0).unwrap().and_local_timezone(dt.timezone()).single().unwrap()
            } else {
                dt.date_naive().and_hms_opt(0, 0, 0).unwrap().and_local_timezone(dt.timezone()).single().unwrap()
            }
        }
        _ => return Err(VmError::Runtime(
            format!("Rounding not supported for unit: {}", unit)
        ))
    };
    
    Ok(Value::LyObj(LyObj::new(Box::new(DateTimeObject::new(rounded)))))
}

/// DateRange[start, end, step] - Generate date ranges
pub fn date_range(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "exactly 3 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let start = extract_datetime(&args[0])?;
    let end = extract_datetime(&args[1])?;
    let step_dur = extract_duration(&args[2])?;
    
    let range = DateRangeObject::new(
        *start.datetime(),
        *end.datetime(),
        step_dur.duration(),
    );
    
    Ok(Value::LyObj(LyObj::new(Box::new(range))))
}

// ================================================================================================
// Date Formatting Functions
// ================================================================================================

/// DateFormat[date, format] - Format date/time to string
pub fn date_format(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let datetime = extract_datetime(&args[0])?;
    
    let format_str = match &args[1] {
        Value::String(f) => f,
        _ => return Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let formatted = datetime.datetime().format(format_str).to_string();
    Ok(Value::String(formatted))
}

/// DateFormatISO[date] - ISO 8601 format
pub fn date_format_iso(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let datetime = extract_datetime(&args[0])?;
    let formatted = datetime.datetime().to_rfc3339();
    Ok(Value::String(formatted))
}

/// DateFormatLocal[date, locale] - Locale-specific formatting
pub fn date_format_local(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let datetime = extract_datetime(&args[0])?;
    
    let _locale = match &args[1] {
        Value::String(l) => l,
        _ => return Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    // For now, just return a standard format
    // In a real implementation, you'd use locale-aware formatting
    let formatted = datetime.datetime().format("%A, %B %e, %Y at %l:%M %p").to_string();
    Ok(Value::String(formatted))
}

/// DateToString[date] - Default string representation
pub fn date_to_string(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let datetime = extract_datetime(&args[0])?;
    let formatted = datetime.datetime().format("%Y-%m-%d %H:%M:%S %Z").to_string();
    Ok(Value::String(formatted))
}

// ================================================================================================
// Date Component Extraction Functions
// ================================================================================================

/// Year[date] - Extract year
pub fn year(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let datetime = extract_datetime(&args[0])?;
    Ok(Value::Integer(datetime.datetime().year() as i64))
}

/// Month[date] - Extract month
pub fn month(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let datetime = extract_datetime(&args[0])?;
    Ok(Value::Integer(datetime.datetime().month() as i64))
}

/// Day[date] - Extract day
pub fn day(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let datetime = extract_datetime(&args[0])?;
    Ok(Value::Integer(datetime.datetime().day() as i64))
}

/// Hour[date] - Extract hour
pub fn hour(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let datetime = extract_datetime(&args[0])?;
    Ok(Value::Integer(datetime.datetime().hour() as i64))
}

/// Minute[date] - Extract minute
pub fn minute(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let datetime = extract_datetime(&args[0])?;
    Ok(Value::Integer(datetime.datetime().minute() as i64))
}

/// Second[date] - Extract second
pub fn second(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let datetime = extract_datetime(&args[0])?;
    Ok(Value::Integer(datetime.datetime().second() as i64))
}

/// DayOfWeek[date] - Extract day of week (1=Monday, 7=Sunday)
pub fn day_of_week(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let datetime = extract_datetime(&args[0])?;
    let day_num = match datetime.datetime().weekday() {
        Weekday::Mon => 1,
        Weekday::Tue => 2,
        Weekday::Wed => 3,
        Weekday::Thu => 4,
        Weekday::Fri => 5,
        Weekday::Sat => 6,
        Weekday::Sun => 7,
    };
    Ok(Value::Integer(day_num))
}

/// DayOfYear[date] - Extract day of year
pub fn day_of_year(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let datetime = extract_datetime(&args[0])?;
    Ok(Value::Integer(datetime.datetime().ordinal() as i64))
}

/// WeekOfYear[date] - Extract ISO week number
pub fn week_of_year(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let datetime = extract_datetime(&args[0])?;
    Ok(Value::Integer(datetime.datetime().iso_week().week() as i64))
}

/// Quarter[date] - Extract quarter (1-4)
pub fn quarter(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let datetime = extract_datetime(&args[0])?;
    let quarter = (datetime.datetime().month() - 1) / 3 + 1;
    Ok(Value::Integer(quarter as i64))
}

// ================================================================================================
// Timezone Operation Functions
// ================================================================================================

/// TimeZoneConvert[datetime, from_tz, to_tz] - Convert between timezones
pub fn timezone_convert(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "exactly 3 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let datetime = extract_datetime(&args[0])?;
    let _from_tz = extract_timezone(&args[1])?; // Could be used for validation
    let to_tz = extract_timezone(&args[2])?;
    
    let converted = datetime.with_timezone(&to_tz.timezone());
    Ok(Value::LyObj(LyObj::new(Box::new(converted))))
}

/// TimeZoneList[] - List available timezones
pub fn timezone_list(args: &[Value]) -> VmResult<Value> {
    if !args.is_empty() {
        return Err(VmError::TypeError {
            expected: "no arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let timezone_names: Vec<Value> = TZ_VARIANTS
        .iter()
        .map(|tz| Value::String(tz.name().to_string()))
        .collect();
    
    Ok(Value::List(timezone_names))
}

/// LocalTimeZone[] - Get system timezone
pub fn local_timezone(args: &[Value]) -> VmResult<Value> {
    if !args.is_empty() {
        return Err(VmError::TypeError {
            expected: "no arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let local_tz = get_local_timezone();
    Ok(Value::LyObj(LyObj::new(Box::new(TimeZoneObject::new(local_tz)))))
}

/// IsDST[datetime, timezone] - Check daylight saving time
pub fn is_dst(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let datetime = extract_datetime(&args[0])?;
    let timezone = extract_timezone(&args[1])?;
    
    let dt_in_tz = datetime.datetime().with_timezone(&timezone.timezone());
    // Check if DST is active by comparing offset names or other means
    // This is a simplified check - in practice you'd need more sophisticated DST detection
    let is_dst = dt_in_tz.offset().to_string().contains("DT") || dt_in_tz.offset().to_string().contains("DST");
    Ok(Value::Boolean(is_dst))
}

// ================================================================================================
// Calendar Operation Functions
// ================================================================================================

/// BusinessDays[start_date, end_date] - Count business days
pub fn business_days(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let start_date = extract_datetime(&args[0])?;
    let end_date = extract_datetime(&args[1])?;
    
    let count = business_days_between(start_date, end_date);
    Ok(Value::Integer(count))
}

/// AddBusinessDays[date, days] - Add business days only
pub fn add_business_days(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let datetime = extract_datetime(&args[0])?;
    
    let days = match &args[1] {
        Value::Integer(d) => *d,
        _ => return Err(VmError::TypeError {
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let result = add_business_days_util(datetime, days)?;
    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}

/// IsBusinessDay[date] - Check if business day
pub fn is_business_day(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let datetime = extract_datetime(&args[0])?;
    let is_business = !matches!(datetime.datetime().weekday(), Weekday::Sat | Weekday::Sun);
    Ok(Value::Boolean(is_business))
}

/// IsWeekend[date] - Check if weekend
pub fn is_weekend(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let datetime = extract_datetime(&args[0])?;
    let is_weekend = matches!(datetime.datetime().weekday(), Weekday::Sat | Weekday::Sun);
    Ok(Value::Boolean(is_weekend))
}

/// IsLeapYear[year] - Check leap year
pub fn is_leap_year_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let year = match &args[0] {
        Value::Integer(y) => *y as i32,
        _ => return Err(VmError::TypeError {
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    Ok(Value::Boolean(is_leap_year(year)))
}

// ================================================================================================
// Duration Operation Functions
// ================================================================================================

/// DurationToSeconds[duration] - Convert to seconds
pub fn duration_to_seconds(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let duration = extract_duration(&args[0])?;
    Ok(Value::Integer(duration.duration().num_seconds()))
}

/// DurationToMinutes[duration] - Convert to minutes
pub fn duration_to_minutes(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let duration = extract_duration(&args[0])?;
    Ok(Value::Integer(duration.duration().num_minutes()))
}

/// DurationToHours[duration] - Convert to hours
pub fn duration_to_hours(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let duration = extract_duration(&args[0])?;
    Ok(Value::Integer(duration.duration().num_hours()))
}

/// DurationToDays[duration] - Convert to days
pub fn duration_to_days(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let duration = extract_duration(&args[0])?;
    Ok(Value::Integer(duration.duration().num_days()))
}

/// DurationAdd[dur1, dur2] - Add durations
pub fn duration_add(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let dur1 = extract_duration(&args[0])?;
    let dur2 = extract_duration(&args[1])?;
    
    let result = dur1.add(dur2);
    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}

/// DurationSubtract[dur1, dur2] - Subtract durations
pub fn duration_subtract(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let dur1 = extract_duration(&args[0])?;
    let dur2 = extract_duration(&args[1])?;
    
    let result = dur1.subtract(dur2);
    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}