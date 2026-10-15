//! The grainbase command line.
//!
//! A thin adapter over the engine: argument parsing and reporting only. All
//! conversion work lives in `grainbase::driver`, so callers and the CLI exercise
//! the same code.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use grainbase::driver::{FilmMode, Job, run};

/// Convert camera-scanned film negatives into finished positives.
#[derive(Debug, Parser)]
#[command(name = "grainbase", version, about, long_about = None)]
struct Cli {
    /// Folder of camera RAW scans. Treated as one roll.
    input: PathBuf,

    /// Folder for the finished positives. Mirrors the input tree.
    #[arg(short, long)]
    out: PathBuf,

    /// Which film the scans are.
    #[arg(long, value_enum, default_value_t = Mode::ColorNegative)]
    mode: Mode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Mode {
    /// Color negative film.
    ColorNegative,
}

impl From<Mode> for FilmMode {
    fn from(mode: Mode) -> Self {
        match mode {
            Mode::ColorNegative => FilmMode::ColorNegative,
        }
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let report = run(&Job {
        input: cli.input,
        output: cli.out,
        mode: cli.mode.into(),
    });

    println!(
        "{} converted, {} already current",
        report.converted(),
        report.skipped()
    );

    let failures = report.failures();
    if failures.is_empty() {
        return ExitCode::SUCCESS;
    }

    eprintln!("{} failed:", failures.len());
    for (path, reason) in failures {
        eprintln!("  {}: {reason}", path.display());
    }
    ExitCode::FAILURE
}
