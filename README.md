# grainbase

A headless command-line tool that converts camera-scanned film negatives (color negative, B&W, and slide) into finished positives.

Status: **pre-alpha.** No implementation yet.

[![ci](https://github.com/RP2/grainbase/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/RP2/grainbase/actions/workflows/ci.yml?query=branch%3Amain+event%3Apush)
[![codeql](https://github.com/RP2/grainbase/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/RP2/grainbase/actions/workflows/codeql.yml?query=branch%3Amain+event%3Apush)

## License

AGPL-3.0. See [`LICENSE`](LICENSE).

## Development

The full `just check` is the one command to run before pushing — it runs formatting, clippy,
tests, and a `-D warnings` doc build. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for prerequisites and the conventional-commit style.
