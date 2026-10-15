# justfile — single command surface for humans and agents.
# Run `just` (no args) to list recipes. Install: https://github.com/casey/just

# Run before pushing. This is the contract that CONTRIBUTING.md points at.
default:
    @just --list

# Format the workspace in place.
fmt:
    cargo fmt --all

# Fail if the workspace is not formatted.
fmt-check:
    cargo fmt --all -- --check

# Run clippy with the same flags CI uses.
clippy:
    cargo clippy --workspace --all-targets --all-features

# Build the workspace and run its tests.
test:
    cargo test --workspace

# Build docs; warnings are fatal, matching CI.
doc:
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features

# Supply-chain check. Requires `cargo install cargo-deny --locked`.
deny:
    cargo deny check

# The one command to run before pushing.
check: fmt-check clippy test doc

# Auto-format and apply safe clippy fixes.
fix:
    cargo fmt --all
    cargo clippy --workspace --all-targets --all-features --fix --allow-dirty --allow-staged
