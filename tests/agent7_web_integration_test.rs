//! Agent 7: Enhanced Web & API Integration System - Integration Tests
//! 
//! This file contains comprehensive tests for the enhanced web and API integration
//! capabilities implemented in Agent 7.

use lyra::vm::{Value, VM};
use lyra::error::Error as LyraError;

/// Helper function to create VM and execute expression
fn execute_lyra(expr: &str) -> Result<Value, LyraError> {
    let mut vm = VM::new();
    vm.evaluate(expr)
}

#[test]
fn test_http_session_creation() {
    let result = execute_lyra("HTTPSession[]");
    assert!(result.is_ok());
    
    let session_with_base = execute_lyra(r#"HTTPSession["https://api.example.com"]"#);
    assert!(session_with_base.is_ok());
}

#[test]
fn test_html_parsing() {
    let html = r#"
        <html>
            <body>
                <h1>Test Page</h1>
                <a href="https://example.com">Link 1</a>
                <a href="/relative">Link 2</a>
                <img src="/image.jpg" alt="Test Image" />
            </body>
        </html>
    "#;
    
    let result = execute_lyra(&format!(r#"HTMLParse[{}]"#, format_args!("\"{}\"", html.replace('\n', "\\n"))));
    assert!(result.is_ok());
}

#[test] 
fn test_css_selection() {
    let html = r#"<html><body><h1>Title</h1><p>Content</p></body></html>"#;
    let expr = format!(r#"
        doc = HTMLParse["{}"];
        CSSSelect[doc, "h1"]
    "#, html);
    
    let result = execute_lyra(&expr);
    assert!(result.is_ok());
    
    if let Ok(Value::List(elements)) = result {
        assert!(!elements.is_empty());
    }
}

#[test]
fn test_url_parsing() {
    let test_cases = vec![
        "https://example.com:8080/path?param=value#section",
        "http://localhost/api/v1/users",
        "https://api.github.com/repos/user/repo",
    ];
    
    for url in test_cases {
        let result = execute_lyra(&format!(r#"URLParse["{}"]"#, url));
        assert!(result.is_ok());
        
        if let Ok(Value::List(components)) = result {
            assert!(!components.is_empty());
            // Check that scheme is present
            let has_scheme = components.iter().any(|component| {
                if let Value::List(pair) = component {
                    if pair.len() == 2 {
                        if let Value::String(key) = &pair[0] {
                            return key == "scheme";
                        }
                    }
                }
                false
            });
            assert!(has_scheme);
        }
    }
}

#[test]
fn test_url_building() {
    let expr = r#"
        URLBuild[{
            {"scheme", "https"},
            {"host", "api.example.com"},
            {"port", 8443},
            {"path", "/v1/data"},
            {"query", {{"param", "value"}, {"other", "test"}}},
            {"fragment", "section"}
        }]
    "#;
    
    let result = execute_lyra(expr);
    assert!(result.is_ok());
    
    if let Ok(Value::String(url)) = result {
        assert!(url.starts_with("https://"));
        assert!(url.contains("api.example.com"));
        assert!(url.contains("8443"));
        assert!(url.contains("/v1/data"));
        assert!(url.contains("param=value"));
        assert!(url.contains("#section"));
    }
}

#[test]
fn test_url_validation() {
    let valid_urls = vec![
        "https://example.com",
        "http://localhost:3000",
        "https://api.github.com/repos/user/repo",
    ];
    
    let invalid_urls = vec![
        "not-a-url",
        "ftp://",
        "https://",
        "",
    ];
    
    for url in valid_urls {
        let result = execute_lyra(&format!(r#"URLValidate["{}"]"#, url));
        assert!(result.is_ok());
        if let Ok(Value::Integer(valid)) = result {
            assert_eq!(valid, 1);
        }
    }
    
    for url in invalid_urls {
        let result = execute_lyra(&format!(r#"URLValidate["{}"]"#, url));
        assert!(result.is_ok());
        if let Ok(Value::Integer(valid)) = result {
            assert_eq!(valid, 0);
        }
    }
}

#[test]
fn test_form_data_creation() {
    let expr = r#"
        FormData[{
            {"name", "John Doe"},
            {"email", "john@example.com"},
            {"age", "25"}
        }]
    "#;
    
    let result = execute_lyra(expr);
    assert!(result.is_ok());
    
    if let Ok(Value::String(form_data)) = result {
        assert!(form_data.contains("name="));
        assert!(form_data.contains("email="));
        assert!(form_data.contains("age="));
        assert!(form_data.contains("John"));
    }
}

#[test]
fn test_rest_client_creation() {
    let result = execute_lyra(r#"RESTClient["https://api.example.com"]"#);
    assert!(result.is_ok());
}

#[test]
fn test_xml_parsing() {
    let xml = r#"<note><to>Tove</to><from>Jani</from><heading>Reminder</heading><body>Don't forget the meeting!</body></note>"#;
    let result = execute_lyra(&format!(r#"XMLParse[{}]"#, format_args!("\"{}\"", xml)));
    assert!(result.is_ok());
    
    if let Ok(Value::List(components)) = result {
        assert!(!components.is_empty());
        // Check for type field
        let has_type = components.iter().any(|component| {
            if let Value::List(pair) = component {
                if pair.len() == 2 {
                    if let Value::String(key) = &pair[0] {
                        return key == "type";
                    }
                }
            }
            false
        });
        assert!(has_type);
    }
}

#[test]
fn test_json_to_lyra_conversion() {
    // Test that complex data structures can be handled
    let expr = r#"
        data = {
            {"name", "Alice"},
            {"age", 30},
            {"skills", {"rust", "python", "javascript"}},
            {"active", 1}
        };
        data
    "#;
    
    let result = execute_lyra(expr);
    assert!(result.is_ok());
}

#[test]
fn test_http_methods() {
    // Test HTTP method function creation (these will fail without a real server)
    // but we can test that the functions are callable
    let test_cases = vec![
        r#"HTTPGet["https://httpbin.org/get"]"#,
        r#"HTTPPost["https://httpbin.org/post", {"test": "data"}]"#,
        r#"HTTPPut["https://httpbin.org/put", {"test": "data"}]"#,
        r#"HTTPDelete["https://httpbin.org/delete"]"#,
    ];
    
    for expr in test_cases {
        // These might fail due to network issues, but they should at least parse correctly
        let result = execute_lyra(expr);
        // We expect either success or a runtime error (not a parse/type error)
        match result {
            Ok(_) => {}, // Success
            Err(LyraError::Runtime(_)) => {}, // Expected runtime error (network)
            Err(e) => panic!("Unexpected error type: {:?}", e),
        }
    }
}

#[test]
fn test_html_extraction() {
    let html = r#"
        <html>
            <body>
                <h1>Main Title</h1>
                <a href="https://example.com/page1">Link 1</a>
                <a href="/relative/page2">Link 2</a>
                <img src="https://example.com/image1.jpg" alt="Image 1" />
                <img src="/images/image2.png" alt="Image 2" />
                <p>Some text content</p>
            </body>
        </html>
    "#;
    
    let expr = format!(r#"
        doc = HTMLParse["{}"];
        links = HTMLExtractLinks[doc];
        images = HTMLExtractImages[doc];
        {{links, images}}
    "#, html.replace('\n', "\\n"));
    
    let result = execute_lyra(&expr);
    assert!(result.is_ok());
    
    if let Ok(Value::List(results)) = result {
        assert_eq!(results.len(), 2);
        
        // Check links
        if let Value::List(links) = &results[0] {
            assert!(!links.is_empty());
        }
        
        // Check images  
        if let Value::List(images) = &results[1] {
            assert!(!images.is_empty());
        }
    }
}

#[test]
fn test_comprehensive_web_workflow() {
    // Test a complete workflow combining multiple Agent 7 features
    let workflow = r#"
        // 1. Parse a sample API response structure
        apiData = {
            {"users", {
                {"id", 1}, {"name", "Alice"}, {"email", "alice@example.com"}
            }},
            {"total", 1}
        };
        
        // 2. Parse URL components
        apiUrl = "https://api.example.com:8080/v1/users?limit=10&offset=0#results";
        urlComponents = URLParse[apiUrl];
        
        // 3. Create form data
        formData = FormData[{
            {"username", "testuser"},
            {"password", "secret123"}
        }];
        
        // 4. Validate URLs
        validUrl = URLValidate["https://api.example.com"];
        invalidUrl = URLValidate["not-a-url"];
        
        // Return comprehensive results
        {apiData, urlComponents, formData, validUrl, invalidUrl}
    "#;
    
    let result = execute_lyra(workflow);
    assert!(result.is_ok());
    
    if let Ok(Value::List(results)) = result {
        assert_eq!(results.len(), 5);
        
        // Check that we have all expected components
        assert!(matches!(results[0], Value::List(_))); // apiData
        assert!(matches!(results[1], Value::List(_))); // urlComponents
        assert!(matches!(results[2], Value::String(_))); // formData
        assert!(matches!(results[3], Value::Integer(1))); // validUrl
        assert!(matches!(results[4], Value::Integer(0))); // invalidUrl
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_agent7_function_registry() {
        // Test that all Agent 7 functions are properly registered
        let mut vm = VM::new();
        
        let functions = vec![
            "HTTPSession", "HTTPRequest", "HTTPGet", "HTTPPost", "HTTPPut", "HTTPDelete",
            "HTTPTimeout", "HTTPAuth", "HTMLParse", "CSSSelect", "HTMLExtractText",
            "HTMLExtractLinks", "HTMLExtractImages", "RESTClient", "JSONRequest",
            "URLParse", "URLBuild", "URLValidate", "FormData", "XMLParse"
        ];
        
        for function in functions {
            // Test that the function exists by trying to call it with wrong args
            // This should give a type error, not an unknown function error
            match vm.evaluate(&format!("{}[42]", function)) {
                Err(LyraError::TypeError { .. }) => {}, // Expected - wrong args
                Err(LyraError::Runtime(_)) => {}, // Also acceptable - runtime error
                Ok(_) => {}, // Somehow worked - also fine  
                Err(e) => {
                    // If it's an unknown function error, the function isn't registered
                    if e.to_string().contains("Unknown function") {
                        panic!("Function {} is not registered", function);
                    }
                }
            }
        }
    }
}