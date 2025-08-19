//! Serializable implementations for Lyra core types

use super::{Serializable, SerializationResult, SerializationError, BinaryWriter, BinaryReader};
use crate::ast::{Expr, Number, Pattern, Symbol, InterpolationPart};
use crate::vm::Value;
use std::io::{Read, Write};

// Type tags for discriminated unions
const EXPR_SYMBOL: u8 = 0;
const EXPR_NUMBER: u8 = 1;
const EXPR_STRING: u8 = 2;
const EXPR_LIST: u8 = 3;
const EXPR_FUNCTION: u8 = 4;
const EXPR_PATTERN: u8 = 5;
const EXPR_RULE: u8 = 6;
const EXPR_ASSIGNMENT: u8 = 7;
const EXPR_REPLACE: u8 = 8;
const EXPR_ASSOCIATION: u8 = 9;
const EXPR_PIPELINE: u8 = 10;
const EXPR_DOT_CALL: u8 = 11;
const EXPR_RANGE: u8 = 12;
const EXPR_ARROW_FUNCTION: u8 = 13;
const EXPR_INTERPOLATED_STRING: u8 = 14;

const NUMBER_INTEGER: u8 = 0;
const NUMBER_REAL: u8 = 1;

const PATTERN_BLANK: u8 = 0;
const PATTERN_BLANK_SEQUENCE: u8 = 1;
const PATTERN_BLANK_NULL_SEQUENCE: u8 = 2;
const PATTERN_NAMED: u8 = 3;
const PATTERN_FUNCTION: u8 = 4;
const PATTERN_TYPED: u8 = 5;
const PATTERN_PREDICATE: u8 = 6;
const PATTERN_ALTERNATIVE: u8 = 7;
const PATTERN_CONDITIONAL: u8 = 8;

const VALUE_INTEGER: u8 = 0;
const VALUE_REAL: u8 = 1;
const VALUE_STRING: u8 = 2;
const VALUE_SYMBOL: u8 = 3;
const VALUE_LIST: u8 = 4;
const VALUE_FUNCTION: u8 = 5;
const VALUE_BOOLEAN: u8 = 6;
const VALUE_MISSING: u8 = 7;
const VALUE_QUOTE: u8 = 8;
const VALUE_PATTERN: u8 = 9;

// Helper functions
fn write_string<W: Write>(writer: &mut BinaryWriter<W>, s: &str) -> SerializationResult<()> {
    let bytes = s.as_bytes();
    writer.write_bytes(&(bytes.len() as u32).to_le_bytes())?;
    writer.write_bytes(bytes)?;
    Ok(())
}

fn read_string<R: Read>(reader: &mut BinaryReader<R>) -> SerializationResult<String> {
    let mut len_bytes = [0u8; 4];
    reader.read_bytes(&mut len_bytes)?;
    let len = u32::from_le_bytes(len_bytes) as usize;
    
    let mut bytes = vec![0u8; len];
    reader.read_bytes(&mut bytes)?;
    
    String::from_utf8(bytes).map_err(|_| SerializationError::CorruptData)
}

fn write_option_string<W: Write>(writer: &mut BinaryWriter<W>, opt: &Option<String>) -> SerializationResult<()> {
    match opt {
        Some(s) => {
            writer.write_bytes(&[1u8])?;
            write_string(writer, s)?;
        }
        None => {
            writer.write_bytes(&[0u8])?;
        }
    }
    Ok(())
}

fn read_option_string<R: Read>(reader: &mut BinaryReader<R>) -> SerializationResult<Option<String>> {
    let mut has_value = [0u8; 1];
    reader.read_bytes(&mut has_value)?;
    
    if has_value[0] == 1 {
        Ok(Some(read_string(reader)?))
    } else {
        Ok(None)
    }
}

impl Serializable for Symbol {
    fn serialize<W: Write>(&self, writer: &mut W) -> SerializationResult<()> {
        let mut writer = BinaryWriter::new(writer);
        write_string(&mut writer, &self.name)?;
        writer.flush()?;
        Ok(())
    }
    
    fn deserialize<R: Read>(reader: &mut R) -> SerializationResult<Self> {
        let mut reader = BinaryReader::new(reader);
        let name = read_string(&mut reader)?;
        Ok(Symbol { name })
    }
    
    fn serialized_size(&self) -> usize {
        4 + self.name.len() // u32 length + string bytes
    }
}

impl Serializable for Number {
    fn serialize<W: Write>(&self, writer: &mut W) -> SerializationResult<()> {
        let mut writer = BinaryWriter::new(writer);
        match self {
            Number::Integer(i) => {
                writer.write_bytes(&[NUMBER_INTEGER])?;
                writer.write_bytes(&i.to_le_bytes())?;
            }
            Number::Real(f) => {
                writer.write_bytes(&[NUMBER_REAL])?;
                writer.write_bytes(&f.to_le_bytes())?;
            }
        }
        writer.flush()?;
        Ok(())
    }
    
    fn deserialize<R: Read>(reader: &mut R) -> SerializationResult<Self> {
        let mut reader = BinaryReader::new(reader);
        let mut tag = [0u8; 1];
        reader.read_bytes(&mut tag)?;
        
        match tag[0] {
            NUMBER_INTEGER => {
                let mut bytes = [0u8; 8];
                reader.read_bytes(&mut bytes)?;
                Ok(Number::Integer(i64::from_le_bytes(bytes)))
            }
            NUMBER_REAL => {
                let mut bytes = [0u8; 8];
                reader.read_bytes(&mut bytes)?;
                Ok(Number::Real(f64::from_le_bytes(bytes)))
            }
            _ => Err(SerializationError::CorruptData),
        }
    }
    
    fn serialized_size(&self) -> usize {
        1 + 8 // tag + 8 bytes for i64/f64
    }
}

impl Serializable for Pattern {
    fn serialize<W: Write>(&self, writer: &mut W) -> SerializationResult<()> {
        let mut writer = BinaryWriter::new(writer);
        match self {
            Pattern::Blank { head } => {
                writer.write_bytes(&[PATTERN_BLANK])?;
                write_option_string(&mut writer, head)?;
            }
            Pattern::BlankSequence { head } => {
                writer.write_bytes(&[PATTERN_BLANK_SEQUENCE])?;
                write_option_string(&mut writer, head)?;
            }
            Pattern::BlankNullSequence { head } => {
                writer.write_bytes(&[PATTERN_BLANK_NULL_SEQUENCE])?;
                write_option_string(&mut writer, head)?;
            }
            Pattern::Named { name, pattern } => {
                writer.write_bytes(&[PATTERN_NAMED])?;
                write_string(&mut writer, name)?;
                pattern.serialize(&mut writer)?;
            }
            Pattern::Function { head, args } => {
                writer.write_bytes(&[PATTERN_FUNCTION])?;
                head.serialize(&mut writer)?;
                writer.write_bytes(&(args.len() as u32).to_le_bytes())?;
                for arg in args {
                    arg.serialize(&mut writer)?;
                }
            }
            Pattern::Typed { name, type_pattern } => {
                writer.write_bytes(&[PATTERN_TYPED])?;
                write_string(&mut writer, name)?;
                type_pattern.serialize(&mut writer)?;
            }
            Pattern::Predicate { pattern, test } => {
                writer.write_bytes(&[PATTERN_PREDICATE])?;
                pattern.serialize(&mut writer)?;
                test.serialize(&mut writer)?;
            }
            Pattern::Alternative { patterns } => {
                writer.write_bytes(&[PATTERN_ALTERNATIVE])?;
                writer.write_bytes(&(patterns.len() as u32).to_le_bytes())?;
                for pattern in patterns {
                    pattern.serialize(&mut writer)?;
                }
            }
            Pattern::Conditional { pattern, condition } => {
                writer.write_bytes(&[PATTERN_CONDITIONAL])?;
                pattern.serialize(&mut writer)?;
                condition.serialize(&mut writer)?;
            }
        }
        writer.flush()?;
        Ok(())
    }
    
    fn deserialize<R: Read>(reader: &mut R) -> SerializationResult<Self> {
        let mut reader = BinaryReader::new(reader);
        let mut tag = [0u8; 1];
        reader.read_bytes(&mut tag)?;
        
        match tag[0] {
            PATTERN_BLANK => {
                let head = read_option_string(&mut reader)?;
                Ok(Pattern::Blank { head })
            }
            PATTERN_BLANK_SEQUENCE => {
                let head = read_option_string(&mut reader)?;
                Ok(Pattern::BlankSequence { head })
            }
            PATTERN_BLANK_NULL_SEQUENCE => {
                let head = read_option_string(&mut reader)?;
                Ok(Pattern::BlankNullSequence { head })
            }
            PATTERN_NAMED => {
                let name = read_string(&mut reader)?;
                let pattern = Box::new(Pattern::deserialize(&mut reader)?);
                Ok(Pattern::Named { name, pattern })
            }
            PATTERN_FUNCTION => {
                let head = Box::new(Pattern::deserialize(&mut reader)?);
                let mut len_bytes = [0u8; 4];
                reader.read_bytes(&mut len_bytes)?;
                let len = u32::from_le_bytes(len_bytes) as usize;
                
                let mut args = Vec::with_capacity(len);
                for _ in 0..len {
                    args.push(Pattern::deserialize(&mut reader)?);
                }
                Ok(Pattern::Function { head, args })
            }
            PATTERN_TYPED => {
                let name = read_string(&mut reader)?;
                let type_pattern = Box::new(Expr::deserialize(&mut reader)?);
                Ok(Pattern::Typed { name, type_pattern })
            }
            PATTERN_PREDICATE => {
                let pattern = Box::new(Pattern::deserialize(&mut reader)?);
                let test = Box::new(Expr::deserialize(&mut reader)?);
                Ok(Pattern::Predicate { pattern, test })
            }
            PATTERN_ALTERNATIVE => {
                let mut len_bytes = [0u8; 4];
                reader.read_bytes(&mut len_bytes)?;
                let len = u32::from_le_bytes(len_bytes) as usize;
                
                let mut patterns = Vec::with_capacity(len);
                for _ in 0..len {
                    patterns.push(Pattern::deserialize(&mut reader)?);
                }
                Ok(Pattern::Alternative { patterns })
            }
            PATTERN_CONDITIONAL => {
                let pattern = Box::new(Pattern::deserialize(&mut reader)?);
                let condition = Box::new(Expr::deserialize(&mut reader)?);
                Ok(Pattern::Conditional { pattern, condition })
            }
            _ => Err(SerializationError::CorruptData),
        }
    }
    
    fn serialized_size(&self) -> usize {
        // Rough estimate - actual implementation would be more precise
        match self {
            Pattern::Blank { head } => 1 + head.as_ref().map_or(1, |s| 5 + s.len()),
            Pattern::BlankSequence { head } => 1 + head.as_ref().map_or(1, |s| 5 + s.len()),
            Pattern::BlankNullSequence { head } => 1 + head.as_ref().map_or(1, |s| 5 + s.len()),
            Pattern::Named { name, pattern } => 1 + 4 + name.len() + pattern.serialized_size(),
            Pattern::Function { head, args } => {
                1 + head.serialized_size() + 4 + args.iter().map(|a| a.serialized_size()).sum::<usize>()
            }
            Pattern::Typed { name, type_pattern } => 1 + 4 + name.len() + type_pattern.serialized_size(),
            Pattern::Predicate { pattern, test } => 1 + pattern.serialized_size() + test.serialized_size(),
            Pattern::Alternative { patterns } => {
                1 + 4 + patterns.iter().map(|p| p.serialized_size()).sum::<usize>()
            }
            Pattern::Conditional { pattern, condition } => 1 + pattern.serialized_size() + condition.serialized_size(),
        }
    }
}

impl Serializable for InterpolationPart {
    fn serialize<W: Write>(&self, writer: &mut W) -> SerializationResult<()> {
        let mut writer = BinaryWriter::new(writer);
        match self {
            InterpolationPart::Text(text) => {
                writer.write_bytes(&[0u8])?;
                write_string(&mut writer, text)?;
            }
            InterpolationPart::Expression(expr) => {
                writer.write_bytes(&[1u8])?;
                expr.serialize(&mut writer)?;
            }
        }
        writer.flush()?;
        Ok(())
    }
    
    fn deserialize<R: Read>(reader: &mut R) -> SerializationResult<Self> {
        let mut reader = BinaryReader::new(reader);
        let mut tag = [0u8; 1];
        reader.read_bytes(&mut tag)?;
        
        match tag[0] {
            0 => {
                let text = read_string(&mut reader)?;
                Ok(InterpolationPart::Text(text))
            }
            1 => {
                let expr = Box::new(Expr::deserialize(&mut reader)?);
                Ok(InterpolationPart::Expression(expr))
            }
            _ => Err(SerializationError::CorruptData),
        }
    }
    
    fn serialized_size(&self) -> usize {
        match self {
            InterpolationPart::Text(text) => 1 + 4 + text.len(),
            InterpolationPart::Expression(expr) => 1 + expr.serialized_size(),
        }
    }
}

impl Serializable for Expr {
    fn serialize<W: Write>(&self, writer: &mut W) -> SerializationResult<()> {
        let mut writer = BinaryWriter::new(writer);
        match self {
            Expr::Symbol(symbol) => {
                writer.write_bytes(&[EXPR_SYMBOL])?;
                symbol.serialize(&mut writer)?;
            }
            Expr::Number(number) => {
                writer.write_bytes(&[EXPR_NUMBER])?;
                number.serialize(&mut writer)?;
            }
            Expr::String(s) => {
                writer.write_bytes(&[EXPR_STRING])?;
                write_string(&mut writer, s)?;
            }
            Expr::List(items) => {
                writer.write_bytes(&[EXPR_LIST])?;
                writer.write_bytes(&(items.len() as u32).to_le_bytes())?;
                for item in items {
                    item.serialize(&mut writer)?;
                }
            }
            Expr::Function { head, args } => {
                writer.write_bytes(&[EXPR_FUNCTION])?;
                head.serialize(&mut writer)?;
                writer.write_bytes(&(args.len() as u32).to_le_bytes())?;
                for arg in args {
                    arg.serialize(&mut writer)?;
                }
            }
            Expr::Pattern(pattern) => {
                writer.write_bytes(&[EXPR_PATTERN])?;
                pattern.serialize(&mut writer)?;
            }
            Expr::Rule { lhs, rhs, delayed } => {
                writer.write_bytes(&[EXPR_RULE])?;
                lhs.serialize(&mut writer)?;
                rhs.serialize(&mut writer)?;
                writer.write_bytes(&[if *delayed { 1u8 } else { 0u8 }])?;
            }
            Expr::Assignment { lhs, rhs, delayed } => {
                writer.write_bytes(&[EXPR_ASSIGNMENT])?;
                lhs.serialize(&mut writer)?;
                rhs.serialize(&mut writer)?;
                writer.write_bytes(&[if *delayed { 1u8 } else { 0u8 }])?;
            }
            Expr::Replace { expr, rules, repeated } => {
                writer.write_bytes(&[EXPR_REPLACE])?;
                expr.serialize(&mut writer)?;
                rules.serialize(&mut writer)?;
                writer.write_bytes(&[if *repeated { 1u8 } else { 0u8 }])?;
            }
            Expr::Association(pairs) => {
                writer.write_bytes(&[EXPR_ASSOCIATION])?;
                writer.write_bytes(&(pairs.len() as u32).to_le_bytes())?;
                for (key, value) in pairs {
                    key.serialize(&mut writer)?;
                    value.serialize(&mut writer)?;
                }
            }
            Expr::Pipeline { stages } => {
                writer.write_bytes(&[EXPR_PIPELINE])?;
                writer.write_bytes(&(stages.len() as u32).to_le_bytes())?;
                for stage in stages {
                    stage.serialize(&mut writer)?;
                }
            }
            Expr::DotCall { object, method, args } => {
                writer.write_bytes(&[EXPR_DOT_CALL])?;
                object.serialize(&mut writer)?;
                write_string(&mut writer, method)?;
                writer.write_bytes(&(args.len() as u32).to_le_bytes())?;
                for arg in args {
                    arg.serialize(&mut writer)?;
                }
            }
            Expr::Range { start, end, step } => {
                writer.write_bytes(&[EXPR_RANGE])?;
                start.serialize(&mut writer)?;
                end.serialize(&mut writer)?;
                match step {
                    Some(s) => {
                        writer.write_bytes(&[1u8])?;
                        s.serialize(&mut writer)?;
                    }
                    None => {
                        writer.write_bytes(&[0u8])?;
                    }
                }
            }
            Expr::ArrowFunction { params, body } => {
                writer.write_bytes(&[EXPR_ARROW_FUNCTION])?;
                writer.write_bytes(&(params.len() as u32).to_le_bytes())?;
                for param in params {
                    write_string(&mut writer, param)?;
                }
                body.serialize(&mut writer)?;
            }
            Expr::InterpolatedString(parts) => {
                writer.write_bytes(&[EXPR_INTERPOLATED_STRING])?;
                writer.write_bytes(&(parts.len() as u32).to_le_bytes())?;
                for part in parts {
                    part.serialize(&mut writer)?;
                }
            }
        }
        writer.flush()?;
        Ok(())
    }
    
    fn deserialize<R: Read>(reader: &mut R) -> SerializationResult<Self> {
        let mut reader = BinaryReader::new(reader);
        let mut tag = [0u8; 1];
        reader.read_bytes(&mut tag)?;
        
        match tag[0] {
            EXPR_SYMBOL => {
                let symbol = Symbol::deserialize(&mut reader)?;
                Ok(Expr::Symbol(symbol))
            }
            EXPR_NUMBER => {
                let number = Number::deserialize(&mut reader)?;
                Ok(Expr::Number(number))
            }
            EXPR_STRING => {
                let s = read_string(&mut reader)?;
                Ok(Expr::String(s))
            }
            EXPR_LIST => {
                let mut len_bytes = [0u8; 4];
                reader.read_bytes(&mut len_bytes)?;
                let len = u32::from_le_bytes(len_bytes) as usize;
                
                let mut items = Vec::with_capacity(len);
                for _ in 0..len {
                    items.push(Expr::deserialize(&mut reader)?);
                }
                Ok(Expr::List(items))
            }
            EXPR_FUNCTION => {
                let head = Box::new(Expr::deserialize(&mut reader)?);
                let mut len_bytes = [0u8; 4];
                reader.read_bytes(&mut len_bytes)?;
                let len = u32::from_le_bytes(len_bytes) as usize;
                
                let mut args = Vec::with_capacity(len);
                for _ in 0..len {
                    args.push(Expr::deserialize(&mut reader)?);
                }
                Ok(Expr::Function { head, args })
            }
            EXPR_PATTERN => {
                let pattern = Pattern::deserialize(&mut reader)?;
                Ok(Expr::Pattern(pattern))
            }
            EXPR_RULE => {
                let lhs = Box::new(Expr::deserialize(&mut reader)?);
                let rhs = Box::new(Expr::deserialize(&mut reader)?);
                let mut delayed_byte = [0u8; 1];
                reader.read_bytes(&mut delayed_byte)?;
                let delayed = delayed_byte[0] == 1;
                Ok(Expr::Rule { lhs, rhs, delayed })
            }
            EXPR_ASSIGNMENT => {
                let lhs = Box::new(Expr::deserialize(&mut reader)?);
                let rhs = Box::new(Expr::deserialize(&mut reader)?);
                let mut delayed_byte = [0u8; 1];
                reader.read_bytes(&mut delayed_byte)?;
                let delayed = delayed_byte[0] == 1;
                Ok(Expr::Assignment { lhs, rhs, delayed })
            }
            EXPR_REPLACE => {
                let expr = Box::new(Expr::deserialize(&mut reader)?);
                let rules = Box::new(Expr::deserialize(&mut reader)?);
                let mut repeated_byte = [0u8; 1];
                reader.read_bytes(&mut repeated_byte)?;
                let repeated = repeated_byte[0] == 1;
                Ok(Expr::Replace { expr, rules, repeated })
            }
            EXPR_ASSOCIATION => {
                let mut len_bytes = [0u8; 4];
                reader.read_bytes(&mut len_bytes)?;
                let len = u32::from_le_bytes(len_bytes) as usize;
                
                let mut pairs = Vec::with_capacity(len);
                for _ in 0..len {
                    let key = Expr::deserialize(&mut reader)?;
                    let value = Expr::deserialize(&mut reader)?;
                    pairs.push((key, value));
                }
                Ok(Expr::Association(pairs))
            }
            EXPR_PIPELINE => {
                let mut len_bytes = [0u8; 4];
                reader.read_bytes(&mut len_bytes)?;
                let len = u32::from_le_bytes(len_bytes) as usize;
                
                let mut stages = Vec::with_capacity(len);
                for _ in 0..len {
                    stages.push(Expr::deserialize(&mut reader)?);
                }
                Ok(Expr::Pipeline { stages })
            }
            EXPR_DOT_CALL => {
                let object = Box::new(Expr::deserialize(&mut reader)?);
                let method = read_string(&mut reader)?;
                let mut len_bytes = [0u8; 4];
                reader.read_bytes(&mut len_bytes)?;
                let len = u32::from_le_bytes(len_bytes) as usize;
                
                let mut args = Vec::with_capacity(len);
                for _ in 0..len {
                    args.push(Expr::deserialize(&mut reader)?);
                }
                Ok(Expr::DotCall { object, method, args })
            }
            EXPR_RANGE => {
                let start = Box::new(Expr::deserialize(&mut reader)?);
                let end = Box::new(Expr::deserialize(&mut reader)?);
                let mut has_step = [0u8; 1];
                reader.read_bytes(&mut has_step)?;
                
                let step = if has_step[0] == 1 {
                    Some(Box::new(Expr::deserialize(&mut reader)?))
                } else {
                    None
                };
                Ok(Expr::Range { start, end, step })
            }
            EXPR_ARROW_FUNCTION => {
                let mut len_bytes = [0u8; 4];
                reader.read_bytes(&mut len_bytes)?;
                let len = u32::from_le_bytes(len_bytes) as usize;
                
                let mut params = Vec::with_capacity(len);
                for _ in 0..len {
                    params.push(read_string(&mut reader)?);
                }
                let body = Box::new(Expr::deserialize(&mut reader)?);
                Ok(Expr::ArrowFunction { params, body })
            }
            EXPR_INTERPOLATED_STRING => {
                let mut len_bytes = [0u8; 4];
                reader.read_bytes(&mut len_bytes)?;
                let len = u32::from_le_bytes(len_bytes) as usize;
                
                let mut parts = Vec::with_capacity(len);
                for _ in 0..len {
                    parts.push(InterpolationPart::deserialize(&mut reader)?);
                }
                Ok(Expr::InterpolatedString(parts))
            }
            _ => Err(SerializationError::CorruptData),
        }
    }
    
    fn serialized_size(&self) -> usize {
        // Rough estimate - actual implementation would be more precise
        match self {
            Expr::Symbol(s) => 1 + s.serialized_size(),
            Expr::Number(n) => 1 + n.serialized_size(),
            Expr::String(s) => 1 + 4 + s.len(),
            Expr::List(items) => 1 + 4 + items.iter().map(|i| i.serialized_size()).sum::<usize>(),
            Expr::Function { head, args } => {
                1 + head.serialized_size() + 4 + args.iter().map(|a| a.serialized_size()).sum::<usize>()
            }
            Expr::Pattern(p) => 1 + p.serialized_size(),
            Expr::Rule { lhs, rhs, .. } => 1 + lhs.serialized_size() + rhs.serialized_size() + 1,
            Expr::Assignment { lhs, rhs, .. } => 1 + lhs.serialized_size() + rhs.serialized_size() + 1,
            Expr::Replace { expr, rules, .. } => 1 + expr.serialized_size() + rules.serialized_size() + 1,
            Expr::Association(pairs) => {
                1 + 4 + pairs.iter().map(|(k, v)| k.serialized_size() + v.serialized_size()).sum::<usize>()
            }
            Expr::Pipeline { stages } => 1 + 4 + stages.iter().map(|s| s.serialized_size()).sum::<usize>(),
            Expr::DotCall { object, method, args } => {
                1 + object.serialized_size() + 4 + method.len() + 4 + args.iter().map(|a| a.serialized_size()).sum::<usize>()
            }
            Expr::Range { start, end, step } => {
                1 + start.serialized_size() + end.serialized_size() + 1 + step.as_ref().map_or(0, |s| s.serialized_size())
            }
            Expr::ArrowFunction { params, body } => {
                1 + 4 + params.iter().map(|p| 4 + p.len()).sum::<usize>() + body.serialized_size()
            }
            Expr::InterpolatedString(parts) => {
                1 + 4 + parts.iter().map(|p| p.serialized_size()).sum::<usize>()
            }
        }
    }
}

impl Serializable for Value {
    fn serialize<W: Write>(&self, writer: &mut W) -> SerializationResult<()> {
        let mut writer = BinaryWriter::new(writer);
        match self {
            Value::Integer(i) => {
                writer.write_bytes(&[VALUE_INTEGER])?;
                writer.write_bytes(&i.to_le_bytes())?;
            }
            Value::Real(f) => {
                writer.write_bytes(&[VALUE_REAL])?;
                writer.write_bytes(&f.to_le_bytes())?;
            }
            Value::String(s) => {
                writer.write_bytes(&[VALUE_STRING])?;
                write_string(&mut writer, s)?;
            }
            Value::Symbol(s) => {
                writer.write_bytes(&[VALUE_SYMBOL])?;
                write_string(&mut writer, s)?;
            }
            Value::List(items) => {
                writer.write_bytes(&[VALUE_LIST])?;
                writer.write_bytes(&(items.len() as u32).to_le_bytes())?;
                for item in items {
                    item.serialize(&mut writer)?;
                }
            }
            Value::Function(name) => {
                writer.write_bytes(&[VALUE_FUNCTION])?;
                write_string(&mut writer, name)?;
            }
            Value::Boolean(b) => {
                writer.write_bytes(&[VALUE_BOOLEAN])?;
                writer.write_bytes(&[if *b { 1u8 } else { 0u8 }])?;
            }
            Value::Missing => {
                writer.write_bytes(&[VALUE_MISSING])?;
            }
            Value::Quote(expr) => {
                writer.write_bytes(&[VALUE_QUOTE])?;
                expr.serialize(&mut writer)?;
            }
            Value::Pattern(pattern) => {
                writer.write_bytes(&[VALUE_PATTERN])?;
                pattern.serialize(&mut writer)?;
            }
            Value::LyObj(_) => {
                // For now, we'll skip LyObj serialization as it requires foreign object support
                return Err(SerializationError::UnsupportedFormat("LyObj serialization not yet implemented".to_string()));
            }
        }
        writer.flush()?;
        Ok(())
    }
    
    fn deserialize<R: Read>(reader: &mut R) -> SerializationResult<Self> {
        let mut reader = BinaryReader::new(reader);
        let mut tag = [0u8; 1];
        reader.read_bytes(&mut tag)?;
        
        match tag[0] {
            VALUE_INTEGER => {
                let mut bytes = [0u8; 8];
                reader.read_bytes(&mut bytes)?;
                Ok(Value::Integer(i64::from_le_bytes(bytes)))
            }
            VALUE_REAL => {
                let mut bytes = [0u8; 8];
                reader.read_bytes(&mut bytes)?;
                Ok(Value::Real(f64::from_le_bytes(bytes)))
            }
            VALUE_STRING => {
                let s = read_string(&mut reader)?;
                Ok(Value::String(s))
            }
            VALUE_SYMBOL => {
                let s = read_string(&mut reader)?;
                Ok(Value::Symbol(s))
            }
            VALUE_LIST => {
                let mut len_bytes = [0u8; 4];
                reader.read_bytes(&mut len_bytes)?;
                let len = u32::from_le_bytes(len_bytes) as usize;
                
                let mut items = Vec::with_capacity(len);
                for _ in 0..len {
                    items.push(Value::deserialize(&mut reader)?);
                }
                Ok(Value::List(items))
            }
            VALUE_FUNCTION => {
                let name = read_string(&mut reader)?;
                Ok(Value::Function(name))
            }
            VALUE_BOOLEAN => {
                let mut byte = [0u8; 1];
                reader.read_bytes(&mut byte)?;
                Ok(Value::Boolean(byte[0] == 1))
            }
            VALUE_MISSING => {
                Ok(Value::Missing)
            }
            VALUE_QUOTE => {
                let expr = Box::new(Expr::deserialize(&mut reader)?);
                Ok(Value::Quote(expr))
            }
            VALUE_PATTERN => {
                let pattern = Pattern::deserialize(&mut reader)?;
                Ok(Value::Pattern(pattern))
            }
            _ => Err(SerializationError::CorruptData),
        }
    }
    
    fn serialized_size(&self) -> usize {
        match self {
            Value::Integer(_) => 1 + 8,
            Value::Real(_) => 1 + 8,
            Value::String(s) => 1 + 4 + s.len(),
            Value::Symbol(s) => 1 + 4 + s.len(),
            Value::List(items) => 1 + 4 + items.iter().map(|i| i.serialized_size()).sum::<usize>(),
            Value::Function(name) => 1 + 4 + name.len(),
            Value::Boolean(_) => 1 + 1,
            Value::Missing => 1,
            Value::Quote(expr) => 1 + expr.serialized_size(),
            Value::Pattern(pattern) => 1 + pattern.serialized_size(),
            Value::LyObj(_) => 0, // Not implemented yet
        }
    }
}