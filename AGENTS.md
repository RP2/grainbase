# grainbase — Agent notes

This repository holds **only project code and its public documentation** (this README, the
license, and code).

Everything else — specification, plan, architecture decisions, design map, research notes, glossary, monetization, and market analysis — lives in the sibling private repository:

**`../grainbase-private/`**

## Rules

- Read the private repo when you need design context or project history. Never copy its contents
  into this repository.
- Project glossary: `../grainbase-private/CONTEXT.md`.
- Design map + open decisions: `../grainbase-private/docs/wayfinder/grainbase/MAP.md`.
- Never commit secrets or strategy documents here.

## Commands

These are the canonical invocations. Run them through `just` when it is available; the cargo
flags are otherwise identical to what CI uses.

- `just check` — fmt-check + clippy + test + doc. Run before declaring work complete.
- `just fmt` — apply `cargo fmt --all`.
- `just fmt-check` — `cargo fmt --all -- --check`.
- `just clippy` — `cargo clippy --workspace --all-targets --all-features`.
- `just test` — `cargo test --workspace`.
- `just doc` — `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features`.
- `just deny` — `cargo deny check` (requires `cargo install cargo-deny --locked`).

## Lint policy

The workspace sets `[workspace.lints]` and every crate opts in with `[lints] workspace = true`.
The two rules that govern the tree:

- `unsafe_code = "forbid"` — no `unsafe` blocks anywhere; widen only through an ADR.
- `clippy::pedantic = "warn"` — extra lints surface by default. Fix the warnings; do not silence
  them with `#[allow(...)]` unless the lint is genuinely wrong for the code, and add a short
  reason comment when you do.
