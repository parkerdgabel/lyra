use std::env;
use lyra::repl::server::{WebSocketReplServer, ServerConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    // Parse command line arguments for configuration
    let args: Vec<String> = env::args().collect();
    let mut config = ServerConfig::default();

    // Simple argument parsing
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--host" => {
                if i + 1 < args.len() {
                    config.host = args[i + 1].clone();
                    i += 2;
                } else {
                    eprintln!("Error: --host requires a value");
                    std::process::exit(1);
                }
            }
            "--port" => {
                if i + 1 < args.len() {
                    config.port = args[i + 1].parse().unwrap_or_else(|_| {
                        eprintln!("Error: Invalid port number");
                        std::process::exit(1);
                    });
                    i += 2;
                } else {
                    eprintln!("Error: --port requires a value");
                    std::process::exit(1);
                }
            }
            "--max-sessions" => {
                if i + 1 < args.len() {
                    config.max_sessions = args[i + 1].parse().unwrap_or_else(|_| {
                        eprintln!("Error: Invalid max sessions number");
                        std::process::exit(1);
                    });
                    i += 2;
                } else {
                    eprintln!("Error: --max-sessions requires a value");
                    std::process::exit(1);
                }
            }
            "--auth" => {
                config.require_auth = true;
                i += 1;
            }
            "--no-rate-limit" => {
                config.enable_rate_limiting = false;
                i += 1;
            }
            "--help" => {
                print_help();
                std::process::exit(0);
            }
            _ => {
                eprintln!("Error: Unknown argument '{}'", args[i]);
                print_help();
                std::process::exit(1);
            }
        }
    }

    // Create and start the server
    let server = WebSocketReplServer::new(config);
    
    // Set up graceful shutdown
    let shutdown_server = server.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.expect("Failed to listen for ctrl+c");
        println!("\n🛑 Received shutdown signal, gracefully shutting down...");
        if let Err(e) = shutdown_server.shutdown().await {
            eprintln!("Error during shutdown: {}", e);
        }
        std::process::exit(0);
    });

    // Start the server
    server.start().await
}

fn print_help() {
    println!("Lyra WebSocket REPL Server");
    println!("Usage: lyra-server [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --host <HOST>           Server host address (default: 127.0.0.1)");
    println!("  --port <PORT>           Server port (default: 8080)");
    println!("  --max-sessions <N>      Maximum concurrent sessions (default: 100)");
    println!("  --auth                  Enable authentication (default: disabled)");
    println!("  --no-rate-limit         Disable rate limiting (default: enabled)");
    println!("  --help                  Show this help message");
    println!();
    println!("Examples:");
    println!("  lyra-server                         # Start with defaults");
    println!("  lyra-server --port 9000             # Custom port");
    println!("  lyra-server --host 0.0.0.0 --auth  # External access with auth");
    println!();
    println!("WebSocket endpoint: ws://<host>:<port>/ws");
    println!("Health check: http://<host>:<port>/health");
}