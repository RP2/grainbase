//! The grainbase command line.
//!
//! A thin adapter over the engine: argument parsing and reporting only. All
//! conversion work lives in `grainbase::driver`, so callers and the CLI exercise
//! the same code.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use grainbase::decode;
use grainbase::driver::{FilmMode, Flips, Job, run};
use grainbase::geometry;
use grainbase::pipeline::{FilmBase, Settings, film_base};
use grainbase::profile::Profile;

/// Convert camera-scanned film negatives into finished positives.
#[derive(Debug, Parser)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each switch names an independent operator decision: crop, the two \
              mirrors, and the diagnostic overlay"
)]
#[command(name = "grainbase", version, about, long_about = None)]
#[command(subcommand_negates_reqs = true)]
struct Cli {
    /// Fit a calibration profile from a `ColorChecker` capture.
    #[command(subcommand)]
    command: Option<Command>,

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

    /// Crop to the detected film frame.
    ///
    /// Off by default, and deliberately: the frame's edges are the one thing a
    /// camera scan does not state reliably, so the free conversion keeps the
    /// whole scan and lets the operator crop where they want. This exists for
    /// trying the detector out, and for a caller that has checked it on its own
    /// rig. Sources are never modified.
    #[arg(long)]
    crop: bool,

    /// Mirror the scans left to right: the emulsion faced the lens.
    #[arg(long)]
    flip_h: bool,

    /// Mirror the scans top to bottom: the strip was captured inverted.
    #[arg(long)]
    flip_v: bool,

    /// The film base by hand, as per-channel transmittance `r,g,b`.
    ///
    /// The reference tools' own escape hatch for a film border that cannot be
    /// located automatically: sampling the border by hand is their documented
    /// method, and a hand-supplied base disables every detector.
    #[arg(long, value_name = "R,G,B", value_parser = parse_base_rgb)]
    film_base: Option<[f32; 3]>,

    /// Authoritative base sample rectangle, as fractions `x0,y0,x1,y1` of the
    /// uncropped scan. Overrides the automatic rebate-region detector.
    #[arg(long, value_parser = parse_base_sample)]
    base_sample: Option<[f32; 4]>,

    /// Per-channel density white points `r,g,b`, overriding a profile's.
    ///
    /// The one knob that moves a color cast: a channel rendering too bright has
    /// its white point too low. Tune by eye against a known neutral in the
    /// frame — a white wall, a shirt — because a cast is what the numbers
    /// cannot name.
    #[arg(long, value_name = "R,G,B", value_parser = parse_base_rgb)]
    density_max: Option<[f32; 3]>,

    /// Per-channel density black points `r,g,b`, overriding a profile's.
    #[arg(long, value_name = "R,G,B", value_parser = parse_base_rgb)]
    density_floor: Option<[f32; 3]>,

    /// Per-channel print-curve contrast `r,g,b`, overriding a profile's. Above
    /// 1 deepens shadows, below 1 lifts them.
    #[arg(long, value_name = "R,G,B", value_parser = parse_base_rgb)]
    gamma: Option<[f32; 3]>,

    /// Per-channel density a correctly exposed gray measures at `r,g,b`, for
    /// the print curve's pivot. Needs --gamma.
    #[arg(long, value_name = "R,G,B", value_parser = parse_base_rgb)]
    pivot_density: Option<[f32; 3]>,

    /// Write a diagnostic overlay beside each positive: the scan with the
    /// frame, the sprocket bands, and the film-base probe points drawn on it.
    ///
    /// Irreplaceable when a conversion comes out wrong: it shows exactly where
    /// the pipeline looked. Yellow probes are the ones the film base was read
    /// from; magenta probes disagreed with the rest and were ignored.
    #[arg(long)]
    overlay: bool,

    /// Meter each frame's integral color and correct a cast before rendering.
    ///
    /// The frame's large-area integral density is pulled back toward its own
    /// neutral, weighted down when the frame is a scene that is genuinely one
    /// color rather than a film or scanning cast. Off by default while its
    /// effect on the maintainer's fixtures is measured.
    #[arg(long)]
    meter: bool,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Fit a calibration profile from a ColorChecker-class chart capture.
    Fit(FitArgs),
}

#[derive(Debug, Parser)]
struct FitArgs {
    /// Camera RAW capture of a ColorChecker-class chart.
    capture: PathBuf,

    /// Film stock the profile is for.
    #[arg(long)]
    film: String,

    /// Camera or rig the profile was captured on.
    #[arg(long)]
    camera: String,

    /// Mirror the capture left to right.
    #[arg(long)]
    flip_h: bool,

    /// Mirror the capture top to bottom.
    #[arg(long)]
    flip_v: bool,

    /// The film base by hand, as per-channel transmittance `r,g,b`.
    #[arg(long, value_name = "R,G,B", value_parser = parse_base_rgb)]
    film_base: Option<[f32; 3]>,

    /// Output profile path. Defaults to `<capture stem>.profile`.
    #[arg(long)]
    out: Option<PathBuf>,

    /// Authoritative base sample rectangle, as fractions `x0,y0,x1,y1` of the
    /// uncropped scan. Overrides the automatic rebate-region detector.
    #[arg(long, value_parser = parse_base_sample)]
    base_sample: Option<[f32; 4]>,

    /// Write a diagnostic overlay beside each positive: the scan with the
    /// frame, the sprocket bands, and the film-base probe points drawn on it.
    ///
    /// Irreplaceable when a conversion comes out wrong: it shows exactly where
    /// the pipeline looked. Yellow probes are the ones the film base was read
    /// from; magenta probes disagreed with the rest and were ignored.
    #[arg(long)]
    overlay: bool,
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

    if let Some(Command::Fit(args)) = cli.command {
        return run_fit(args);
    }

    let settings = match load_profile(cli.profile.as_deref()).map(|settings| Settings {
        density_max: cli.density_max.or(settings.density_max),
        density_floor: cli.density_floor.or(settings.density_floor),
        gamma: cli.gamma.or(settings.gamma),
        pivot_density: cli.pivot_density.or(settings.pivot_density),
        ..settings
    }) {
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
        crop: cli.crop,
        flips: Flips {
            horizontal: cli.flip_h,
            vertical: cli.flip_v,
        },
        base_sample: cli.base_sample,
        film_base: cli.film_base,
        overlay: cli.overlay,
        meter: cli.meter,
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

/// Fits a profile from a chart capture and writes it to disk.
fn run_fit(args: FitArgs) -> ExitCode {
    let scan = match decode::decode(&args.capture) {
        Ok(scan) => scan,
        Err(err) => {
            eprintln!("could not decode {}: {err}", args.capture.display());
            return ExitCode::FAILURE;
        }
    };

    println!(
        "decoded {}x{} from {} {} (EXIF orientation {:?})",
        scan.linear.width,
        scan.linear.height,
        scan.meta.make,
        scan.meta.model,
        scan.meta.orientation
    );

    // Orientation first (spec step 1): the flips are the operator's decision
    // about which side of the film faced the lens, applied before anything
    // is detected.
    let uncropped = scan.linear.flipped(Flips {
        horizontal: args.flip_h,
        vertical: args.flip_v,
    });

    // Find the frame, then measure the base on the still-uncropped scan, by
    // the documented source precedence (spec step 3).
    let frame = geometry::find(&uncropped).unwrap_or_else(|| geometry::Frame::whole(&uncropped));
    println!(
        "frame: centre {:.0},{:.0}  size {:.0}x{:.0}  skew {:.4} rad ({:.2} deg)",
        frame.center[0],
        frame.center[1],
        frame.size[0],
        frame.size[1],
        frame.skew,
        frame.skew.to_degrees()
    );

    let base = if let Some(values) = args.film_base {
        FilmBase {
            transmittance: values.map(|value| value.max(f32::MIN_POSITIVE)),
        }
    } else if let Some(rect) = args.base_sample {
        FilmBase::detect_in_rect(&uncropped, rect, film_base::DEFAULT_PERCENTILE)
    } else if let Some(base) = measured_between_sprockets(&uncropped, &frame) {
        base
    } else {
        eprintln!(
            "no film base could be measured: the scan shows no sprocket bands to measure \
             between. Pass --film-base r,g,b or --base-sample x0,y0,x1,y1."
        );
        return ExitCode::FAILURE;
    };

    let image = geometry::apply(&uncropped, &frame);

    let measurements = match grainbase::pipeline::chart::measure(&image, &base) {
        Ok(measurements) => measurements,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::FAILURE;
        }
    };

    let profile =
        match grainbase::pipeline::chart::fit_capture(&measurements, &args.film, &args.camera) {
            Ok(profile) => profile,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::FAILURE;
            }
        };

    let text = profile.to_string();
    if let Err(err) = Profile::parse(&text) {
        eprintln!("generated profile does not round-trip: {err}");
        return ExitCode::FAILURE;
    }

    let output = args
        .out
        .unwrap_or_else(|| default_profile_path(&args.capture));
    if let Some(parent) = output.parent()
        && let Err(err) = fs::create_dir_all(parent)
    {
        eprintln!(
            "could not create output directory {}: {err}",
            parent.display()
        );
        return ExitCode::FAILURE;
    }
    if let Err(err) = fs::write(&output, text) {
        eprintln!("could not write {}: {err}", output.display());
        return ExitCode::FAILURE;
    }

    print_fit_summary(&profile);
    println!("wrote {}", output.display());
    ExitCode::SUCCESS
}

/// The base measured from the film between the sprocket holes, or `None` when
/// there is nothing to measure.
///
/// The identity base means the sampler found nothing to read — a band with no
/// lit punch, or no gap between two holes. That is not a measurement, and
/// converting with it would render every frame wrong in silence.
fn measured_between_sprockets(
    image: &grainbase::LinearImage,
    frame: &geometry::Frame,
) -> Option<FilmBase> {
    let bands: Vec<film_base::FilmRegion> = frame.holes.iter().flatten().copied().collect();
    if bands.is_empty() {
        return None;
    }
    let base = FilmBase::detect_between_sprockets(image, &bands);
    (base != FilmBase::IDENTITY).then_some(base)
}

/// The default profile path is a sibling of the capture with `.profile` extension.
fn default_profile_path(capture: &Path) -> PathBuf {
    let mut path = capture.to_path_buf();
    path.set_extension("profile");
    path
}

/// Prints the fitted anchors and crosstalk matrix in human-readable form.
fn print_fit_summary(profile: &Profile) {
    let floors = profile.settings.density_floor.unwrap_or([0.0; 3]);
    let ceilings = profile
        .settings
        .density_max
        .expect("a fitted profile emits ceilings");
    let matrix = profile.settings.crosstalk;

    println!("fitted profile for {} on {}", profile.film, profile.camera);
    println!(
        "  black point (density)  R {:.4}  G {:.4}  B {:.4}",
        floors[0], floors[1], floors[2]
    );
    println!(
        "  white point (density)  R {:.4}  G {:.4}  B {:.4}",
        ceilings[0], ceilings[1], ceilings[2]
    );
    println!("  crosstalk matrix (rows = output channels):");
    for row in matrix {
        println!("    {:.4} {:.4} {:.4}", row[0], row[1], row[2]);
    }
}

/// Parses `--base-sample x0,y0,x1,y1` as fractions of the image.
fn parse_base_sample(text: &str) -> Result<[f32; 4], String> {
    let parts: Vec<&str> = text.split(',').collect();
    if parts.len() != 4 {
        return Err("expected four comma-separated fractions".to_owned());
    }
    let mut out = [0.0f32; 4];
    for (i, part) in parts.iter().enumerate() {
        out[i] = part
            .trim()
            .parse()
            .map_err(|_| format!("`{part}` is not a number"))?;
    }
    Ok(out)
}

/// Parses `--film-base r,g,b` as per-channel transmittance.
fn parse_base_rgb(text: &str) -> Result<[f32; 3], String> {
    let parts: Vec<&str> = text.split(',').collect();
    if parts.len() != 3 {
        return Err("expected three comma-separated channel values".to_owned());
    }
    let mut out = [0.0f32; 3];
    for (i, part) in parts.iter().enumerate() {
        out[i] = part
            .trim()
            .parse()
            .map_err(|_| format!("`{part}` is not a number"))?;
    }
    Ok(out)
}

/// Reads a profile from disk, or falls back to neutral defaults.
///
/// A profile that cannot be read is a hard error rather than a silent fallback:
/// converting a whole roll with defaults because a path was mistyped would
/// produce a folder of wrong output that looks deliberate.
fn load_profile(path: Option<&Path>) -> Result<Settings, String> {
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
