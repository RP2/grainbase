# grainbase — Agent notes

**Public repo.** Every commit world-readable, immediately, permanently.

Repo holds **code + public docs only** (this README, license, code).

Everything else — spec, plan, ADRs, design map, research, glossary, monetization, market analysis,
session handoffs
— lives in the sibling private repo:

**`../grainbase-private/`** — handoffs in `docs/handoffs/`, never in a temp directory.

## Rules

- **Never commit private-repo content.** No spec, plan, ADR, strategy, market analysis, glossary
  text. Read it, do not copy it.
- **Never commit a secret.** No keys, tokens, `.env`, credentials.
- **Never commit scans.** Raw captures and lab scans stay in the private `fixtures/`. `.gitignore`
  blocks them; never force-add past it.
- **Never paste a private file's contents into a file here.** Private path = private file. About to
  paste? Stop.

Need design context or project history? Read the private repo:

- Glossary: `../grainbase-private/CONTEXT.md`
- Design map, open decisions: `../grainbase-private/docs/wayfinder/grainbase/MAP.md`

## Commands

Canonical invocations. Use `just` when installed; cargo flags identical to CI either way.

- `just check` — fmt-check + clippy + test + doc. Run before declaring work done.
- `just fmt` — `cargo fmt --all`
- `just fmt-check` — `cargo fmt --all -- --check`
- `just clippy` — `cargo clippy --workspace --all-targets --all-features`
- `just test` — `cargo test --workspace`
- `just doc` — `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features`
- `just deny` — `cargo deny check` (needs `cargo install cargo-deny --locked`)

## Lint policy

Workspace sets `[workspace.lints]`; every crate opts in with `[lints] workspace = true`.

- `unsafe_code = "forbid"` — no `unsafe` anywhere. Widen only via ADR.
- `clippy::pedantic = "warn"` — fix warnings. Do not silence with `#[allow(...)]` unless the lint
  is genuinely wrong; then add a reason comment.
