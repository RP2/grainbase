# Contributing to grainbase

Thanks for your interest. This document is short on purpose: if `just` is
installed locally, exactly one command covers what CI checks.

## Attribution

External knowledge feeding the code is cited, never absorbed: see
[`docs/ATTRIBUTION.md`](docs/ATTRIBUTION.md) for the rule and the current
ledger. In short: published facts may be used and cited; other projects'
code, constants, and undocumented numbers are not used. Any constant in the
tree whose value has external provenance carries its citation in the doc
comment where it lives.

## Prerequisites

- A recent stable Rust toolchain (`rustup default stable` will do; the
  required components — `rustfmt`, `clippy` — are listed in
  `rust-toolchain.toml`).
- GNU `make`-style task runner [`just`](https://github.com/casey/just)
  (optional but recommended).
- `cargo-deny` for the supply-chain check (`cargo install cargo-deny
  --locked`).

## Build and test

Run everything CI runs locally before pushing:

```sh
just check
```

That recipe wraps the canonical pipeline:

1. `cargo fmt --all -- --check` — formatting.
2. `cargo clippy --workspace --all-targets --all-features` — lint (the
   workspace forbids `unsafe_code` and enables `clippy::pedantic`).
3. `cargo test --workspace` — unit and integration tests.
4. `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
   --all-features` — documentation build, warnings-as-errors.

To run any single step by hand, the underlying cargo invocations are
listed in the `justfile`.

The `fix` recipe (`just fix`) applies `cargo fmt` and the safe subset of
`cargo clippy --fix` automatically. Use it for routine cleanup, then
`just check` to confirm.

## Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/). One
line, imperative mood (`ci: add supply-chain scanning`, not `Added
supply-chain scanning`). The footer block — body, breaking changes,
issue references — is optional.

## Agents

You are welcome here. Read [`AGENTS.md`](AGENTS.md) first — it is the highest-signal
file in this repo for an automated contributor, and it is referenced by
both the project glossary and the private design map (sibling repo). The
build and test commands in `AGENTS.md` are the same ones humans run, so
treat any deviation as suspect.
