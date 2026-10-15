//! The grainbase command line.
//!
//! A thin adapter over the engine: argument parsing and reporting only. All
//! conversion work lives in `grainbase::driver`, so callers and the CLI exercise
//! the same code.

use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use grainbase::driver::{FilmMode, Job, run};
use grainbase::pipeline::Settings;
use grainbase::profile::Profile;

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

    /// Calibration profile to apply. Without one, neutral defaults are used.
    #[arg(long)]
    profile: Option<PathBuf>,

    /// Do not crop to the film frame.
    ///
    /// Cropping is a judgement, so it can be switched off. Sources are never
    /// modified, so this only changes what the output contains.
    #[arg(long)]
    no_crop: bool,
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

    let settings = match load_profile(cli.profile.as_deref()) {
        Ok(settings) => settings,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::FAILURE;
        }
    };

    let report = run(&Job {
        input: cli.input,
        output: cli.out,
        mode: cli.mode.into(),
        settings,
        crop: !cli.no_crop,
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

/// Reads a profile from disk, or falls back to neutral defaults.
///
/// A profile that cannot be read is a hard error rather than a silent fallback:
/// converting a whole roll with defaults because a path was mistyped would
/// produce a folder of wrong output that looks deliberate.
fn load_profile(path: Option<&std::path::Path>) -> Result<Settings, String> {
    let Some(path) = path else {
        return Ok(Settings::default());
    };

    let text = fs::read_to_string(path)
        .map_err(|err| format!("could not read profile {}: {err}", path.display()))?;

    let profile =
        Profile::parse(&text).map_err(|err| format!("bad profile {}: {err}", path.display()))?;

    eprintln!(
        "profile: {} on {} ({}), fitted from {} frame(s)",
        profile.film, profile.camera, profile.method, profile.frames
    );

    Ok(profile.settings)
}
