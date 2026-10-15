# grainbase

A headless command-line tool that converts camera-scanned film negatives (color negative, B&W, and slide) into finished positives.

Status: **pre-alpha.** No implementation yet.

[![ci](https://github.com/RP2/grainbase/actions/workflows/ci.yml/badge.svg)](https://github.com/RP2/grainbase/actions/workflows/ci.yml)
[![codeql](https://github.com/RP2/grainbase/actions/workflows/codeql.yml/badge.svg)](https://github.com/RP2/grainbase/actions/workflows/codeql.yml)

## License

AGPL-3.0. See [`LICENSE`](LICENSE).

## Development

The full `just check` is the one command to run before pushing — it runs formatting, clippy,
tests, and a `-D warnings` doc build. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for prerequisites
and the conventional-commit style.

Specification, plan, architecture decisions, design map, and research live in a sibling private
repository (`../grainbase-private`). See [`AGENTS.md`](AGENTS.md).
