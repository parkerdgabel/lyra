use lyra::stdlib::documentation::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Lyra Documentation System Demo ===\n");
    
    // Create a function registry with metadata
    let registry = FunctionRegistry::new();
    
    // Demonstrate function discovery
    println!("1. Function Discovery:");
    let all_functions = registry.get_all_function_names();
    println!("   Total functions available: {}", all_functions.len());
    
    let string_functions = registry.list_functions_by_pattern("String*");
    println!("   String functions: {:?}", string_functions);
    
    let search_results = registry.search_functions("string");
    println!("   Search results for 'string': {:?}\n", search_results);
    
    // Demonstrate help system
    println!("2. Function Metadata:");
    if let Some(metadata) = registry.get_function("Length") {
        println!("   Function: {}", metadata.name);
        println!("   Category: {}", metadata.category);
        println!("   Description: {}", metadata.description);
        println!("   Signature: {}", metadata.signature);
        println!("   Examples: {} provided\n", metadata.examples.len());
    }
    
    // Demonstrate code generation
    println!("3. Code Generation:");
    let code_gen = CodeGenerator::new();
    let python_code = code_gen.generate_code("Map[Square, {1, 2, 3}]", "Python");
    let js_code = code_gen.generate_code("StringTemplate[\"Hello {name}\"]", "JavaScript");
    
    println!("   Python: {}", python_code);
    println!("   JavaScript: {}\n", js_code);
    
    // Demonstrate documentation generation
    println!("4. Documentation Generation:");
    let doc_gen = DocumentationGenerator::new(registry);
    let markdown_docs = doc_gen.generate_markdown(&["Length".to_string(), "StringJoin".to_string()]);
    println!("   Generated {} characters of Markdown documentation\n", markdown_docs.len());
    
    // Show sample help text
    println!("5. Sample Help Text:");
    let help_text = doc_gen.generate_help_text("StringJoin");
    println!("{}", help_text);
    
    println!("=== Demo Complete ===");
    
    Ok(())
}