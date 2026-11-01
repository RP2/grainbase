# grainbase

A headless command-line tool that converts camera-scanned film negatives (color negative, B&W, and slide) into finished positives.

You photograph your film strips against a light table with a regular camera, and grainbase turns the RAW files into positives: it finds each frame, measures the film base from the clear film along the strip edges, inverts and prints through a per-channel density model, and keeps your photos' capture order. Calibration profiles can be fitted from a photographed ColorChecker — one frame per film stock supplies both a stock profile and a rig profile for your camera and light table.

Status: **pre-alpha, under active development.** The conversion pipeline works and is exercised by tests, but its output is not yet validated against lab scans of known targets, and every part of the interface is subject to change. It is not published on crates.io yet; build from source.

[![ci](https://github.com/RP2/grainbase/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/RP2/grainbase/actions/workflows/ci.yml?query=branch%3Amain+event%3Apush)
[![codeql](https://github.com/RP2/grainbase/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/RP2/grainbase/actions/workflows/codeql.yml?query=branch%3Amain+event%3Apush)

## Building

Building uses the exact Rust toolchain pinned in [`rust-toolchain.toml`](rust-toolchain.toml) (installed automatically by [rustup](https://rustup.rs)).

```sh
cargo build --release
```

## Usage

```sh
# Convert a folder of camera RAW scans into finished positives.
cargo run --release -- <folder-of-scans> --out <output-folder>

# Fit a stock profile from a photographed ColorChecker capture.
cargo run --release -- fit --film "Portra 400" --camera "<rig name>" --out <profile> <chart-scan.arw>

# Derive a rig profile from the chart's neutral row.
cargo run --release -- calibrate-rig --family c41 --wedge-card colorchecker-neutral-row <chart-scan.arw> --camera "<rig name>" --out <profile>
```

`--help` on the main command and every subcommand documents the full set of knobs, including `--overlay`, which draws an annotated copy of each scan showing exactly where the pipeline looked.

## License

AGPL-3.0. See [`LICENSE`](LICENSE).

## Development

The full `just check` is the one command to run before pushing — it runs formatting, clippy,
tests, and a `-D warnings` doc build. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for prerequisites and the conventional-commit style.
