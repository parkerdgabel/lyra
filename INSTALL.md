# Installing Lyra

Lyra is a symbolic computation language with Wolfram-inspired syntax and production-ready performance. This guide covers installation on all major platforms.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/parkerdgabel/lyra.git
cd lyra

# Build and install
cargo build --release

# Run the REPL
cargo run --bin lyra -- repl

# Run a script
cargo run --bin lyra -- run examples/01_basic_syntax.lyra
```

## Prerequisites

### All Platforms
- **Rust 1.70 or higher** - [Install Rust](https://rustup.rs/)
- **Git** - For cloning the repository

### Platform-Specific Requirements

#### macOS
```bash
# Install Rust via Homebrew (alternative to rustup)
brew install rust

# Xcode command line tools (if not already installed)
xcode-select --install
```

#### Linux (Ubuntu/Debian)
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install additional dependencies
sudo apt update
sudo apt install build-essential pkg-config libssl-dev
```

#### Linux (RHEL/CentOS/Fedora)
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install additional dependencies
sudo dnf install gcc openssl-devel pkg-config  # Fedora
sudo yum install gcc openssl-devel pkg-config  # CentOS/RHEL
```

#### Windows
```powershell
# Install Rust via rustup
# Download and run: https://win.rustup.rs/x86_64

# Or via Chocolatey
choco install rust

# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

## Installation Methods

### Method 1: Source Installation (Recommended)

```bash
# Clone repository
git clone https://github.com/parkerdgabel/lyra.git
cd lyra

# Build optimized release
cargo build --release

# Add to PATH (optional)
export PATH="$PWD/target/release:$PATH"

# Verify installation
./target/release/lyra --version
```

### Method 2: Cargo Install (Future)

```bash
# Will be available when published to crates.io
cargo install lyra-lang

# Verify installation
lyra --version
```

### Method 3: Package Managers (Future)

#### macOS (Homebrew)
```bash
# Coming soon
brew install lyra-lang
```

#### Linux (Snap)
```bash
# Coming soon  
sudo snap install lyra-lang
```

#### Windows (Chocolatey)
```powershell
# Coming soon
choco install lyra-lang
```

## Verification

After installation, verify Lyra is working correctly:

```bash
# Check version
lyra --version

# Run tests
cargo test

# Start REPL
lyra repl
```

In the REPL, try:
```wolfram
(* Basic arithmetic *)
2 + 3

(* Function definition *)
f[x_] := x^2

(* Pattern matching *)
f[5]

(* List operations *)
{1, 2, 3, 4, 5} |> Map[f] |> Total
```

Expected output:
```
5
25
{1, 4, 9, 16, 25}
55
```

## Configuration

### Environment Variables

```bash
# Set Lyra configuration directory
export LYRA_CONFIG_DIR="$HOME/.config/lyra"

# Set package cache directory  
export LYRA_CACHE_DIR="$HOME/.cache/lyra"

# Enable performance monitoring
export LYRA_PERF=1

# Set log level
export LYRA_LOG=debug
```

### Configuration File

Create `~/.config/lyra/config.toml`:

```toml
[repl]
history_size = 10000
auto_complete = true
syntax_highlighting = true
timing_display = true

[performance]
thread_count = "auto"  # or specific number
numa_aware = true
memory_optimization = true

[packages]
registry = "https://packages.lyra-lang.org"
cache_dir = "~/.cache/lyra"
auto_update = false
```

## Development Setup

For contributing to Lyra development:

```bash
# Clone with development dependencies
git clone https://github.com/parkerdgabel/lyra.git
cd lyra

# Install development tools
cargo install cargo-criterion cargo-tarpaulin

# Run full test suite
cargo test --all-features

# Run benchmarks
cargo bench

# Check code formatting
cargo fmt --check

# Run linter
cargo clippy -- -D warnings
```

## Performance Tuning

### Memory Optimization
```bash
# Enable memory pools for better performance
export LYRA_MEMORY_POOLS=1

# Set pool sizes (advanced)
export LYRA_INTEGER_POOL_SIZE=1000
export LYRA_REAL_POOL_SIZE=1000
```

### NUMA Optimization
```bash
# Check NUMA topology
numactl --hardware

# Run with NUMA awareness
numactl --interleave=all lyra repl

# Or set in configuration
export LYRA_NUMA_POLICY=interleave
```

### Multi-threading
```bash
# Set thread pool size
export LYRA_THREADS=16

# Enable work-stealing optimization
export LYRA_WORK_STEALING=1
```

## Troubleshooting

### Common Issues

#### "command not found: lyra"
- **Cause**: Lyra not in PATH
- **Solution**: Add `target/release` to your PATH or use full path

#### "cargo: command not found" 
- **Cause**: Rust not properly installed
- **Solution**: Install Rust via [rustup.rs](https://rustup.rs/) and restart terminal

#### "linker `cc` not found"
- **Cause**: Missing C compiler
- **Solution**: Install build tools for your platform (see Prerequisites)

#### "failed to compile" on Windows
- **Cause**: Missing Visual Studio Build Tools
- **Solution**: Install Visual Studio Build Tools or Visual Studio Community

#### "permission denied" on Linux/macOS
- **Cause**: Insufficient permissions
- **Solution**: `chmod +x target/release/lyra` or run with `sudo`

#### Slow compilation
- **Cause**: Debug build or low memory
- **Solution**: Use `cargo build --release` and ensure 4GB+ RAM

### Performance Issues

#### High memory usage
```bash
# Enable memory debugging
export LYRA_MEMORY_DEBUG=1

# Check memory pools
lyra repl
%memory status
```

#### Slow execution
```bash
# Enable performance profiling
export LYRA_PERF=1

# Check optimization flags
cargo build --release
```

### Getting Help

- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)
- **Issues**: [GitHub Issues](https://github.com/parkerdgabel/lyra/issues)
- **Discussions**: [GitHub Discussions](https://github.com/parkerdgabel/lyra/discussions)
- **Email**: lyra-lang@example.com

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 2GB | 8GB+ |
| CPU | 2 cores | 4+ cores |
| Storage | 500MB | 2GB+ |
| OS | Linux, macOS, Windows | Latest stable |
| Rust | 1.70+ | Latest stable |

### Feature Support by Platform

| Feature | Linux | macOS | Windows |
|---------|-------|-------|---------|
| Basic Language | ✅ | ✅ | ✅ |
| REPL | ✅ | ✅ | ✅ |
| Concurrency | ✅ | ✅ | ✅ |
| Networking | ✅ | ✅ | ✅ |
| NUMA Optimization | ✅ | ✅ | ⚠️ |
| Package Management | ✅ | ✅ | ✅ |

Legend: ✅ Full Support, ⚠️ Partial Support, ❌ Not Supported

## Next Steps

After successful installation:

1. **Try the tutorial**: `lyra run examples/01_basic_syntax.lyra`
2. **Explore the REPL**: `lyra repl` and try the meta-commands like `%help`
3. **Read the documentation**: [docs/language-reference.md](docs/language-reference.md)
4. **Join the community**: [GitHub Discussions](https://github.com/parkerdgabel/lyra/discussions)
5. **Contribute**: See [CONTRIBUTING.md](CONTRIBUTING.md) (coming soon)

Happy computing with Lyra! 🚀