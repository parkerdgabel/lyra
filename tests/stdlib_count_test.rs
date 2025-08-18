use lyra::stdlib::StandardLibrary;

#[test]
fn debug_stdlib_function_count() {
    let stdlib = StandardLibrary::new();
    let function_names: Vec<&String> = stdlib.function_names();
    
    println!("\n📊 STDLIB FUNCTION COUNT: {}", function_names.len());
    println!("📋 ALL STDLIB FUNCTIONS:");
    for (i, name) in function_names.iter().enumerate() {
        println!("  {:2}: {}", i, name);
    }
}