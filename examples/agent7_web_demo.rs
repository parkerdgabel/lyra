//! Agent 7: Enhanced Web & API Integration System - Demo
//!
//! This example demonstrates the comprehensive web and API integration capabilities
//! implemented in Agent 7, including:
//!
//! 1. Enhanced REST Client operations
//! 2. HTML parsing and web scraping
//! 3. Authentication and security features
//! 4. URL manipulation and validation
//! 5. Form data handling and API integration

use lyra::vm::{Value, VM};
use lyra::error::Error as LyraError;

fn main() -> Result<(), LyraError> {
    println!("🌐 Agent 7: Enhanced Web & API Integration System Demo");
    println!("======================================================\n");

    let mut vm = VM::new();

    // Demo 1: URL Operations
    println!("🔗 Demo 1: URL Operations");
    println!("--------------------------");

    let url_demo = r#"
        (* Parse a complex URL *)
        url = "https://api.github.com:8443/repos/user/repo/issues?state=open&labels=bug,enhancement&page=2#comment-123";
        components = URLParse[url];
        Print["Original URL: ", url];
        Print["Parsed components: ", components];
        
        (* Build a new URL from components *)
        newUrl = URLBuild[{
            {"scheme", "https"},
            {"host", "api.example.com"},
            {"port", 443},
            {"path", "/v1/users"},
            {"query", {{"limit", "50"}, {"offset", "100"}}},
            {"fragment", "results"}
        }];
        Print["Built URL: ", newUrl];
        
        (* Validate URLs *)
        validUrls = {"https://example.com", "http://localhost:3000", "https://api.github.com"};
        invalidUrls = {"not-a-url", "ftp://", "https://", ""};
        
        For[i = 1, i <= Length[validUrls], i++,
            url = validUrls[[i]];
            valid = URLValidate[url];
            Print["URL '", url, "' is valid: ", valid == 1]
        ];
        
        For[i = 1, i <= Length[invalidUrls], i++,
            url = invalidUrls[[i]];
            valid = URLValidate[url];
            Print["URL '", url, "' is valid: ", valid == 1]
        ];
    "#;

    println!("Executing URL operations demo...\n");
    match vm.evaluate(url_demo) {
        Ok(_) => println!("✅ URL operations completed successfully\n"),
        Err(e) => println!("❌ URL operations failed: {}\n", e),
    }

    // Demo 2: HTML Parsing and Web Scraping
    println!("📄 Demo 2: HTML Parsing and Web Scraping");
    println!("------------------------------------------");

    let html_demo = r#"
        (* Sample HTML content *)
        html = "<html><head><title>Sample Page</title></head><body><h1 class='title'>Main Heading</h1><p>This is a paragraph with <a href='https://example.com'>a link</a> and <a href='/relative'>another link</a>.</p><div class='content'><img src='https://example.com/image1.jpg' alt='Image 1'/><img src='/images/local.png' alt='Local Image'/></div><ul><li>Item 1</li><li>Item 2</li><li>Item 3</li></ul></body></html>";
        
        (* Parse the HTML *)
        doc = HTMLParse[html, "https://example.com/page"];
        Print["HTML document parsed successfully"];
        
        (* Extract all links *)
        links = HTMLExtractLinks[doc];
        Print["Found ", Length[links], " links:"];
        For[i = 1, i <= Length[links], i++,
            Print["  - ", links[[i]]]
        ];
        
        (* Extract all images *)
        images = HTMLExtractImages[doc];
        Print["Found ", Length[images], " images:"];
        For[i = 1, i <= Length[images], i++,
            Print["  - ", images[[i]]]
        ];
        
        (* Use CSS selectors *)
        headings = CSSSelect[doc, "h1"];
        Print["Found ", Length[headings], " h1 elements"];
        
        paragraphs = CSSSelect[doc, "p"];
        Print["Found ", Length[paragraphs], " paragraph elements"];
        
        listItems = CSSSelect[doc, "li"];
        Print["Found ", Length[listItems], " list items"];
        
        (* Extract text from first heading *)
        If[Length[headings] > 0,
            headingText = HTMLExtractText[headings[[1]]];
            Print["First heading text: '", headingText, "'"]
        ];
    "#;

    println!("Executing HTML parsing demo...\n");
    match vm.evaluate(html_demo) {
        Ok(_) => println!("✅ HTML parsing completed successfully\n"),
        Err(e) => println!("❌ HTML parsing failed: {}\n", e),
    }

    // Demo 3: Form Data and API Integration
    println!("📋 Demo 3: Form Data and API Integration");
    println!("------------------------------------------");

    let form_demo = r#"
        (* Create form data *)
        loginForm = FormData[{
            {"username", "john.doe@example.com"},
            {"password", "secretpassword123"},
            {"remember_me", "true"}
        }];
        Print["Login form data: ", loginForm];
        
        (* Create registration data *)
        registrationData = {
            {"name", "John Doe"},
            {"email", "john.doe@example.com"},
            {"age", 28},
            {"skills", {"JavaScript", "Python", "Rust"}},
            {"active", 1}
        };
        Print["Registration data structure created"];
        
        (* Create API request form data *)
        apiForm = FormData[{
            {"api_key", "sk-1234567890abcdef"},
            {"model", "gpt-4"},
            {"temperature", "0.7"},
            {"max_tokens", "150"}
        }];
        Print["API form data: ", apiForm];
    "#;

    println!("Executing form data demo...\n");
    match vm.evaluate(form_demo) {
        Ok(_) => println!("✅ Form data operations completed successfully\n"),
        Err(e) => println!("❌ Form data operations failed: {}\n", e),
    }

    // Demo 4: XML Processing
    println!("🏷️  Demo 4: XML Processing");
    println!("---------------------------");

    let xml_demo = r#"
        (* Sample XML documents *)
        simpleXml = "<note><to>Alice</to><from>Bob</from><heading>Meeting</heading><body>Don't forget our meeting tomorrow!</body></note>";
        configXml = "<config><database><host>localhost</host><port>5432</port><name>myapp</name></database><cache><enabled>true</enabled><ttl>3600</ttl></cache></config>";
        
        (* Parse XML documents *)
        note = XMLParse[simpleXml];
        Print["Simple XML parsed: ", note];
        
        config = XMLParse[configXml];
        Print["Configuration XML parsed: ", config];
        
        (* Display XML structure information *)
        Print["Note XML length: ", Length[ToString[simpleXml]], " characters"];
        Print["Config XML length: ", Length[ToString[configXml]], " characters"];
    "#;

    println!("Executing XML processing demo...\n");
    match vm.evaluate(xml_demo) {
        Ok(_) => println!("✅ XML processing completed successfully\n"),
        Err(e) => println!("❌ XML processing failed: {}\n", e),
    }

    // Demo 5: HTTP Session and Client Management
    println!("🔐 Demo 5: HTTP Session and Client Management");
    println!("-----------------------------------------------");

    let session_demo = r#"
        (* Create basic HTTP session *)
        session1 = HTTPSession[];
        Print["Basic HTTP session created"];
        
        (* Create session with base URL *)
        session2 = HTTPSession["https://api.github.com"];
        Print["Session with base URL created"];
        
        (* Create REST clients for different APIs *)
        githubClient = RESTClient["https://api.github.com"];
        Print["GitHub API client created"];
        
        jsonApiClient = RESTClient["https://jsonplaceholder.typicode.com"];
        Print["JSON API client created"];
        
        (* Note: Actual HTTP requests would require network connectivity *)
        Print["HTTP sessions and clients ready for use"];
    "#;

    println!("Executing session management demo...\n");
    match vm.evaluate(session_demo) {
        Ok(_) => println!("✅ Session management completed successfully\n"),
        Err(e) => println!("❌ Session management failed: {}\n", e),
    }

    // Demo 6: Comprehensive Web Workflow
    println!("🚀 Demo 6: Comprehensive Web Workflow");
    println!("--------------------------------------");

    let workflow_demo = r#"
        (* Simulate a complete web application workflow *)
        
        (* 1. User authentication data *)
        userCredentials = FormData[{
            {"email", "developer@example.com"},
            {"password", "securepassword123"}
        }];
        
        (* 2. API endpoint configuration *)
        apiBaseUrl = "https://api.myapp.com";
        apiClient = RESTClient[apiBaseUrl];
        
        (* 3. Parse configuration from URL *)
        configUrl = "https://config.myapp.com/app.json?version=1.2&env=production";
        configComponents = URLParse[configUrl];
        
        (* 4. Prepare data for API submission *)
        userData = {
            {"profile", {
                {"name", "Jane Developer"},
                {"title", "Senior Engineer"},
                {"department", "Engineering"}
            }},
            {"preferences", {
                {"theme", "dark"},
                {"notifications", 1},
                {"language", "en-US"}
            }},
            {"metadata", {
                {"last_login", "2024-01-15T10:30:00Z"},
                {"session_timeout", 3600},
                {"features", {"advanced_search", "beta_ui", "api_access"}}
            }}
        };
        
        (* 5. HTML template processing *)
        htmlTemplate = "<html><head><title>{{title}}</title></head><body><h1>{{heading}}</h1><p>Welcome {{username}}!</p><div class='content'>{{content}}</div></body></html>";
        doc = HTMLParse[htmlTemplate];
        
        (* 6. Extract template variables (simplified) *)
        templateLinks = HTMLExtractLinks[doc];
        Print["Template processed, found ", Length[templateLinks], " links"];
        
        (* 7. Validation and summary *)
        endpoints = {
            apiBaseUrl,
            "https://cdn.myapp.com",
            "https://auth.myapp.com",
            "https://metrics.myapp.com"
        };
        
        validEndpoints = 0;
        For[i = 1, i <= Length[endpoints], i++,
            endpoint = endpoints[[i]];
            If[URLValidate[endpoint] == 1,
                validEndpoints = validEndpoints + 1;
                Print["✓ Valid endpoint: ", endpoint],
                Print["✗ Invalid endpoint: ", endpoint]
            ]
        ];
        
        Print["Workflow completed successfully!"];
        Print["- User credentials prepared"];
        Print["- API client configured"];
        Print["- Configuration URL parsed"];
        Print["- User data structured"];
        Print["- HTML template processed"];
        Print["- ", validEndpoints, "/", Length[endpoints], " endpoints validated"];
        
        (* Return summary *)
        {
            {"credentials_ready", 1},
            {"api_configured", 1},
            {"data_structured", 1},
            {"templates_processed", 1},
            {"endpoints_valid", validEndpoints},
            {"workflow_status", "completed"}
        }
    "#;

    println!("Executing comprehensive workflow demo...\n");
    match vm.evaluate(workflow_demo) {
        Ok(result) => {
            println!("✅ Comprehensive workflow completed successfully!");
            if let Value::List(summary) = result {
                println!("\n📊 Workflow Summary:");
                for item in summary {
                    if let Value::List(pair) = item {
                        if pair.len() == 2 {
                            if let (Value::String(key), value) = (&pair[0], &pair[1]) {
                                println!("  {} -> {:?}", key, value);
                            }
                        }
                    }
                }
            }
        },
        Err(e) => println!("❌ Comprehensive workflow failed: {}", e),
    }

    println!("\n🎉 Agent 7 Demo Completed!");
    println!("===========================");
    println!("The Enhanced Web & API Integration System provides:");
    println!("• 🔗 Advanced URL parsing, building, and validation");
    println!("• 📄 HTML parsing with CSS selectors and content extraction");
    println!("• 🔐 HTTP session management with authentication support");
    println!("• 📋 Form data creation and API integration tools");
    println!("• 🏷️  XML parsing and processing capabilities");
    println!("• 🚀 REST client with base configuration support");
    println!("• 🌐 Comprehensive web scraping and data extraction");
    println!();
    println!("All functions integrate seamlessly with Lyra's symbolic computation");
    println!("engine and can be combined with other stdlib modules for powerful");
    println!("web automation and data processing workflows!");

    Ok(())
}