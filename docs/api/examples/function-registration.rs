// Function Registration Example
//
// This example shows how to register custom functions with Lyra's standard library
// so they can be called from Lyra code. This is the primary way to extend Lyra
// with new functionality.

use crate::vm::{Value, VmResult, VmError};
use crate::stdlib::{StandardLibrary, StdlibFunction};
use crate::foreign::LyObj;

/// Example 1: Simple mathematical function
/// This function calculates the factorial of a number
pub fn factorial_function(args: &[Value]) -> VmResult<Value> {
    // Validate argument count
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    // Extract and validate the argument
    let n = match &args[0] {
        Value::Integer(n) if *n >= 0 => *n as u64,
        Value::Integer(n) => {
            return Err(VmError::Runtime(format!("Factorial requires non-negative integer, got {}", n)));
        }
        _ => {
            return Err(VmError::TypeError {
                expected: "non-negative Integer".to_string(),
                actual: format!("{:?}", args[0]),
            });
        }
    };

    // Calculate factorial
    let result = (1..=n).product::<u64>();
    
    // Handle overflow by returning Real for large numbers
    if result > i64::MAX as u64 {
        Ok(Value::Real(result as f64))
    } else {
        Ok(Value::Integer(result as i64))
    }
}

/// Example 2: Function that works with lists
/// This function calculates the sum of squares of a list of numbers
pub fn sum_of_squares(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (List)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let items = match &args[0] {
        Value::List(items) => items,
        _ => {
            return Err(VmError::TypeError {
                expected: "List".to_string(),
                actual: format!("{:?}", args[0]),
            });
        }
    };

    let mut sum = 0.0;
    for item in items {
        let num = match item {
            Value::Integer(n) => *n as f64,
            Value::Real(f) => *f,
            _ => {
                return Err(VmError::TypeError {
                    expected: "List of numbers".to_string(),
                    actual: format!("List containing {:?}", item),
                });
            }
        };
        sum += num * num;
    }

    // Return Integer if the result is a whole number, otherwise Real
    if sum.fract() == 0.0 && sum <= i64::MAX as f64 {
        Ok(Value::Integer(sum as i64))
    } else {
        Ok(Value::Real(sum))
    }
}

/// Example 3: Function with multiple arguments and optional parameters
/// This function creates a range of numbers: Range[start, end] or Range[start, end, step]
pub fn range_function(args: &[Value]) -> VmResult<Value> {
    match args.len() {
        2 => {
            // Range[start, end] with step = 1
            let start = extract_integer(&args[0], "start")?;
            let end = extract_integer(&args[1], "end")?;
            create_range(start, end, 1)
        }
        3 => {
            // Range[start, end, step]
            let start = extract_integer(&args[0], "start")?;
            let end = extract_integer(&args[1], "end")?;
            let step = extract_integer(&args[2], "step")?;
            
            if step == 0 {
                return Err(VmError::Runtime("Step cannot be zero".to_string()));
            }
            
            create_range(start, end, step)
        }
        _ => Err(VmError::TypeError {
            expected: "2 or 3 arguments (start, end[, step])".to_string(),
            actual: format!("{} arguments", args.len()),
        }),
    }
}

/// Helper function to extract integer from Value
fn extract_integer(value: &Value, name: &str) -> VmResult<i64> {
    match value {
        Value::Integer(n) => Ok(*n),
        Value::Real(f) if f.fract() == 0.0 => Ok(*f as i64),
        _ => Err(VmError::TypeError {
            expected: format!("Integer for {}", name),
            actual: format!("{:?}", value),
        }),
    }
}

/// Helper function to create a range
fn create_range(start: i64, end: i64, step: i64) -> VmResult<Value> {
    let mut result = Vec::new();
    
    if step > 0 {
        let mut current = start;
        while current <= end {
            result.push(Value::Integer(current));
            current += step;
        }
    } else {
        let mut current = start;
        while current >= end {
            result.push(Value::Integer(current));
            current += step;
        }
    }
    
    Ok(Value::List(result))
}

/// Example 4: Function that creates Foreign objects
/// This creates a simple Point object as a Foreign type
use std::any::Any;
use crate::foreign::{Foreign, ForeignError};

#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    x: f64,
    y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }

    pub fn distance_to(&self, other: &Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

impl Foreign for Point {
    fn type_name(&self) -> &'static str {
        "Point"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "x" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Real(self.x))
            }
            "y" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Real(self.y))
            }
            "distanceTo" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                match &args[0] {
                    Value::LyObj(obj) => {
                        if let Some(other_point) = obj.downcast_ref::<Point>() {
                            let distance = self.distance_to(other_point);
                            Ok(Value::Real(distance))
                        } else {
                            Err(ForeignError::TypeError {
                                expected: "Point".to_string(),
                                actual: obj.type_name().to_string(),
                            })
                        }
                    }
                    _ => Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Point".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                }
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

unsafe impl Send for Point {}
unsafe impl Sync for Point {}

/// Function to create Point objects from Lyra
pub fn create_point(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (x, y)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let x = match &args[0] {
        Value::Integer(n) => *n as f64,
        Value::Real(f) => *f,
        _ => {
            return Err(VmError::TypeError {
                expected: "number for x coordinate".to_string(),
                actual: format!("{:?}", args[0]),
            });
        }
    };

    let y = match &args[1] {
        Value::Integer(n) => *n as f64,
        Value::Real(f) => *f,
        _ => {
            return Err(VmError::TypeError {
                expected: "number for y coordinate".to_string(),
                actual: format!("{:?}", args[1]),
            });
        }
    };

    let point = Point::new(x, y);
    Ok(Value::LyObj(LyObj::new(Box::new(point))))
}

/// Example 5: Function with complex validation and error handling
/// This function performs polynomial evaluation: Polynomial[coefficients, x]
pub fn polynomial_eval(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (coefficients, x)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    // Extract coefficients
    let coefficients = match &args[0] {
        Value::List(items) => {
            let mut coeffs = Vec::new();
            for (i, item) in items.iter().enumerate() {
                let coeff = match item {
                    Value::Integer(n) => *n as f64,
                    Value::Real(f) => *f,
                    _ => {
                        return Err(VmError::TypeError {
                            expected: "List of numbers for coefficients".to_string(),
                            actual: format!("coefficient {} is {:?}", i, item),
                        });
                    }
                };
                coeffs.push(coeff);
            }
            coeffs
        }
        _ => {
            return Err(VmError::TypeError {
                expected: "List for coefficients".to_string(),
                actual: format!("{:?}", args[0]),
            });
        }
    };

    if coefficients.is_empty() {
        return Err(VmError::Runtime("Coefficients list cannot be empty".to_string()));
    }

    // Extract x value
    let x = match &args[1] {
        Value::Integer(n) => *n as f64,
        Value::Real(f) => *f,
        _ => {
            return Err(VmError::TypeError {
                expected: "number for x value".to_string(),
                actual: format!("{:?}", args[1]),
            });
        }
    };

    // Evaluate polynomial using Horner's method
    let mut result = coefficients[coefficients.len() - 1];
    for i in (0..coefficients.len() - 1).rev() {
        result = result * x + coefficients[i];
    }

    // Return appropriate type
    if result.fract() == 0.0 && result.is_finite() && result <= i64::MAX as f64 && result >= i64::MIN as f64 {
        Ok(Value::Integer(result as i64))
    } else {
        Ok(Value::Real(result))
    }
}

/// Complete example: How to register all functions with the standard library
pub fn register_example_functions(stdlib: &mut StandardLibrary) {
    // Method 1: Register individual functions
    stdlib.register("Factorial", factorial_function);
    stdlib.register("SumOfSquares", sum_of_squares);
    stdlib.register("Range", range_function);
    stdlib.register("Point", create_point);
    stdlib.register("Polynomial", polynomial_eval);
}

/// Alternative registration pattern using a registration function
pub fn get_math_functions() -> Vec<(&'static str, StdlibFunction)> {
    vec![
        ("Factorial", factorial_function),
        ("SumOfSquares", sum_of_squares),
        ("Range", range_function),
        ("Polynomial", polynomial_eval),
    ]
}

pub fn get_geometry_functions() -> Vec<(&'static str, StdlibFunction)> {
    vec![
        ("Point", create_point),
    ]
}

/// Example of a function registry for organized registration
pub struct FunctionRegistry {
    pub math_functions: Vec<(&'static str, StdlibFunction)>,
    pub geometry_functions: Vec<(&'static str, StdlibFunction)>,
    pub utility_functions: Vec<(&'static str, StdlibFunction)>,
}

impl FunctionRegistry {
    pub fn new() -> Self {
        FunctionRegistry {
            math_functions: get_math_functions(),
            geometry_functions: get_geometry_functions(),
            utility_functions: vec![],
        }
    }

    pub fn register_all(&self, stdlib: &mut StandardLibrary) {
        // Register all function categories
        for (name, func) in &self.math_functions {
            stdlib.register(*name, *func);
        }
        for (name, func) in &self.geometry_functions {
            stdlib.register(*name, *func);
        }
        for (name, func) in &self.utility_functions {
            stdlib.register(*name, *func);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorial_function() {
        // Test normal cases
        assert_eq!(factorial_function(&[Value::Integer(0)]).unwrap(), Value::Integer(1));
        assert_eq!(factorial_function(&[Value::Integer(5)]).unwrap(), Value::Integer(120));
        
        // Test error cases
        assert!(factorial_function(&[Value::Integer(-1)]).is_err());
        assert!(factorial_function(&[Value::String("not a number".to_string())]).is_err());
        assert!(factorial_function(&[]).is_err());
    }

    #[test]
    fn test_sum_of_squares() {
        let list = Value::List(vec![
            Value::Integer(1),
            Value::Integer(2),
            Value::Integer(3),
        ]);
        let result = sum_of_squares(&[list]).unwrap();
        assert_eq!(result, Value::Integer(14)); // 1² + 2² + 3² = 14
    }

    #[test]
    fn test_range_function() {
        // Test Range[1, 5]
        let result = range_function(&[Value::Integer(1), Value::Integer(5)]).unwrap();
        if let Value::List(items) = result {
            assert_eq!(items.len(), 5);
            assert_eq!(items[0], Value::Integer(1));
            assert_eq!(items[4], Value::Integer(5));
        }

        // Test Range[0, 10, 2]
        let result = range_function(&[Value::Integer(0), Value::Integer(10), Value::Integer(2)]).unwrap();
        if let Value::List(items) = result {
            assert_eq!(items.len(), 6);
            assert_eq!(items[0], Value::Integer(0));
            assert_eq!(items[5], Value::Integer(10));
        }
    }

    #[test]
    fn test_point_creation() {
        let result = create_point(&[Value::Real(1.0), Value::Real(2.0)]).unwrap();
        if let Value::LyObj(obj) = result {
            assert_eq!(obj.type_name(), "Point");
            if let Some(point) = obj.downcast_ref::<Point>() {
                assert_eq!(point.x, 1.0);
                assert_eq!(point.y, 2.0);
            }
        }
    }

    #[test]
    fn test_polynomial_eval() {
        // Test polynomial: 2x² + 3x + 1 at x = 2
        // Coefficients: [1, 3, 2] (constant term first)
        // Result: 2(4) + 3(2) + 1 = 8 + 6 + 1 = 15
        let coeffs = Value::List(vec![
            Value::Integer(1), // constant term
            Value::Integer(3), // x term
            Value::Integer(2), // x² term
        ]);
        let result = polynomial_eval(&[coeffs, Value::Integer(2)]).unwrap();
        assert_eq!(result, Value::Integer(15));
    }

    #[test]
    fn test_function_registration() {
        let mut stdlib = StandardLibrary::new();
        register_example_functions(&mut stdlib);
        
        // Verify functions are registered
        assert!(stdlib.get_function("Factorial").is_some());
        assert!(stdlib.get_function("Range").is_some());
        assert!(stdlib.get_function("Point").is_some());
    }
}

// Usage Examples:
//
// After registering these functions, you can use them in Lyra code:
//
// (* Mathematical functions *)
// Factorial[5]          (* Returns 120 *)
// SumOfSquares[{1,2,3}] (* Returns 14 *)
// Range[1, 10]          (* Returns {1,2,3,4,5,6,7,8,9,10} *)
// Range[0, 10, 2]       (* Returns {0,2,4,6,8,10} *)
//
// (* Geometry functions *)
// p1 = Point[0, 0]
// p2 = Point[3, 4]
// p1.distanceTo[p2]     (* Returns 5.0 *)
//
// (* Polynomial evaluation *)
// Polynomial[{1, 2, 1}, 3]  (* Evaluates x² + 2x + 1 at x=3, returns 16 *)
//
// Key Registration Patterns:
//
// 1. Simple function: fn my_func(args: &[Value]) -> VmResult<Value>
// 2. Validate argument count and types
// 3. Extract arguments with proper error handling
// 4. Perform computation
// 5. Return appropriate Value type
// 6. Register with stdlib.register("FunctionName", my_func)
// 7. Use organized registration for multiple functions
// 8. Test thoroughly with various input types and edge cases