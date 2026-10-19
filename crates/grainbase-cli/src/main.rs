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
use grainbase::pipeline::{FilmBase, Settings, film_base, wedge};
use grainbase::profile::Profile;

/// Convert camera-scanned film negatives into finished positives.
#[derive(Debug, Parser)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each switch names an independent operator decision: crop, the two \
              mirrors, and the diagnostic overlay"
)]
#[command(name = "grainbase", version, about, long_about = None)]
struct Cli {
    /// A calibration subcommand. When absent, the conversion below runs.
    #[command(subcommand)]
    command: Option<Command>,

    /// Folder of camera RAW scans. Treated as one roll.
    input: Option<PathBuf>,

    /// Folder for the finished positives. Mirrors the input tree.
    #[arg(short, long)]
    out: Option<PathBuf>,

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

    /// Normalize a slide scan by its clear film base.
    ///
    /// Only applies to slide mode. Off by default: the scan is the positive.
    /// When on, each channel is divided by the measured base transmittance as a
    /// per-channel white balance of the film's Dmin.
    #[arg(long)]
    normalize: bool,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Fit a calibration profile from a ColorChecker-class chart capture.
    Fit(FitArgs),
    /// Fit a rig calibration profile from a neutral gray-wedge capture.
    CalibrateRig(CalibrateRigArgs),
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
}

#[derive(Debug, Parser)]
#[command(group = clap::ArgGroup::new("wedge").required(true).args(["wedge_card", "wedge_steps", "neutral_rect"]))]
struct CalibrateRigArgs {
    /// Process family this rig calibration is for.
    #[arg(long, value_enum)]
    family: Family,

    /// Registered wedge card to use.
    #[arg(long)]
    wedge_card: Option<String>,

    /// Wedge step reflection densities, lightest to darkest, comma-separated.
    #[arg(long, value_name = "D,D,..", value_parser = parse_wedge_steps)]
    wedge_steps: Option<Vec<f32>>,

    /// Camera RAW capture of the gray wedge.
    capture: PathBuf,

    /// Camera or rig the profile was captured on.
    ///
    /// Defaults to the decoded camera make and model.
    #[arg(long)]
    camera: Option<String>,

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

    /// A neutral patch in the frame, as fractions `x0,y0,x1,y1` **of the cropped
    /// frame** — the rectangle is read from the image the frame detector cuts to,
    /// which is what a preview shows.
    ///
    /// The manual tier for a scene that has a known grey but no chart. The
    /// profile takes the per-channel ratios from the patch and the level from the
    /// pipeline's default cone, so it removes a cast without becoming exposure
    /// compensation. The patch has to be lit by a **neutral** light: under
    /// tungsten the patch records the lamp and the profile would neutralise it.
    #[arg(long, value_name = "x0,y0,x1,y1", value_parser = parse_base_sample)]
    neutral_rect: Option<[f32; 4]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Mode {
    /// Color negative film.
    ColorNegative,
    /// Black-and-white negative film.
    BlackAndWhite,
    /// Slide (E-6) film.
    Slide,
}

impl From<Mode> for FilmMode {
    fn from(mode: Mode) -> Self {
        match mode {
            Mode::ColorNegative => FilmMode::ColorNegative,
            Mode::BlackAndWhite => FilmMode::BlackAndWhite,
            Mode::Slide => FilmMode::Slide,
        }
    }
}

/// Process family a rig calibration is scoped to.
///
/// Spelled as the profile's own `family` keyword (`c41`, `bw`, `slide`), with
/// the `--mode` spelling accepted as an alias so a caller does not have to
/// remember two vocabularies for the same thing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Family {
    /// Color negative film.
    #[value(name = "c41", alias = "color-negative")]
    C41,
    /// Black-and-white negative film.
    #[value(name = "bw", alias = "black-and-white")]
    Bw,
    /// Slide (E-6) film.
    #[value(name = "slide")]
    Slide,
}

impl From<Family> for FilmMode {
    fn from(family: Family) -> Self {
        match family {
            Family::C41 => FilmMode::ColorNegative,
            Family::Bw => FilmMode::BlackAndWhite,
            Family::Slide => FilmMode::Slide,
        }
    }
}

/// Human-readable name for a film family, used in warnings.
fn family_name(family: FilmMode) -> &'static str {
    match family {
        FilmMode::ColorNegative => "color-negative",
        FilmMode::BlackAndWhite => "black-and-white",
        FilmMode::Slide => "slide",
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match cli.command {
        Some(Command::Fit(args)) => return run_fit(args),
        Some(Command::CalibrateRig(args)) => return run_calibrate_rig(args),
        None => {}
    }

    // clap does not exempt the parent's positionals when a subcommand runs
    // (`subcommand_negates_reqs` did not take effect on 4.6.7), so the
    // conversion inputs are optional at the type level and required here.
    let (Some(input), Some(output)) = (cli.input, cli.out) else {
        eprintln!(
            "a conversion needs an input folder and --out\n\n\
             usage: grainbase <input> --out <out> [options]\n       \
             grainbase <command> --help"
        );
        return ExitCode::FAILURE;
    };

    let settings =
        match load_profile(cli.profile.as_deref(), cli.mode.into()).map(|settings| Settings {
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
        input,
        output,
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
        slide_normalize: cli.normalize,
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

    let frame = calibration_frame(&uncropped);
    let base = match calibration_base(&uncropped, &frame, args.film_base, args.base_sample) {
        Ok(base) => base,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::FAILURE;
        }
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

    let output = match write_profile(&profile, args.out, &args.capture) {
        Ok(output) => output,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::FAILURE;
        }
    };

    print_fit_summary(&profile);
    println!("wrote {}", output.display());
    ExitCode::SUCCESS
}

/// Where the rig calibration's neutral steps come from.
enum WedgeSource {
    /// A registered card, which knows its own step values and how to find itself.
    Card(wedge::WedgeCard),
    /// Step densities given on the command line, found as a 1-D strip.
    Steps(Vec<f32>),
    /// A neutral patch in the frame, stated by the operator as fractions.
    NeutralRect([f32; 4]),
}

/// Fits a rig-scoped profile from a gray-wedge capture and writes it to disk.
fn run_calibrate_rig(args: CalibrateRigArgs) -> ExitCode {
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

    let frame = calibration_frame(&uncropped);
    let base = match calibration_base(&uncropped, &frame, args.film_base, args.base_sample) {
        Ok(base) => base,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::FAILURE;
        }
    };

    let image = geometry::apply(&uncropped, &frame);

    // The card decides both the step values and how the card is found, so a
    // registered card is carried as a card rather than flattened to its steps.
    let source = if let Some(card_name) = &args.wedge_card {
        if let Some(card) = wedge::card(card_name) {
            WedgeSource::Card(card)
        } else {
            eprintln!(
                "unknown wedge card `{card_name}`; known cards: {}",
                wedge::card_names().join(", ")
            );
            return ExitCode::FAILURE;
        }
    } else if let Some(steps) = args.wedge_steps {
        WedgeSource::Steps(steps)
    } else if let Some(rect) = args.neutral_rect {
        WedgeSource::NeutralRect(rect)
    } else {
        // The clap group guarantees one of the three is present.
        unreachable!("--wedge-card, --wedge-steps or --neutral-rect is required")
    };

    let rig = rig_name(&scan.meta, args.camera);
    let family = args.family.into();

    // A neutral rectangle is measured straight from the frame; a card is located
    // and fitted. They differ in what they can say: one patch gives the ratios at
    // its density, a card gives a line.
    let (profile, measurement) = if let WedgeSource::NeutralRect(rect) = &source {
        let profile = match wedge::profile_from_neutral(&image, &base, *rect, family, &rig) {
            Ok(profile) => profile,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::FAILURE;
            }
        };
        (profile, None)
    } else {
        let measurement = match &source {
            WedgeSource::Card(card) => wedge::measure_for_card(&image, &base, card),
            WedgeSource::Steps(steps) => wedge::measure(&image, &base, steps),
            WedgeSource::NeutralRect(_) => unreachable!("handled above"),
        };
        let measurement = match measurement {
            Ok(measurement) => measurement,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::FAILURE;
            }
        };
        let profile = match wedge::profile(&measurement, family, &rig) {
            Ok(profile) => profile,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::FAILURE;
            }
        };
        (profile, Some(measurement))
    };

    let output = match write_profile(&profile, args.out, &args.capture) {
        Ok(output) => output,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::FAILURE;
        }
    };

    match &measurement {
        Some(measurement) => print_calibrate_rig_summary(&profile, measurement),
        None => print_neutral_summary(&profile),
    }
    println!("wrote {}", output.display());
    ExitCode::SUCCESS
}

/// The rig's name: the operator's, or the camera's make and model.
///
/// A rig is a camera plus a light table plus a process, so the camera's model
/// alone does not identify it. This is only the default; the operator can name it.
fn rig_name(meta: &grainbase::ScanMeta, camera: Option<String>) -> String {
    camera.unwrap_or_else(|| {
        let make = meta.make.trim();
        let model = meta.model.trim();
        if make.is_empty() && model.is_empty() {
            "unknown-rig".to_owned()
        } else if make.is_empty() {
            model.to_owned()
        } else if model.is_empty() {
            make.to_owned()
        } else {
            format!("{make} {model}")
        }
    })
}

/// Finds the frame and reports it, identically for every calibration capture.
///
/// Orientation happens before this: the flips are the operator's decision about
/// which side of the film faced the lens, and are applied before anything is
/// detected.
fn calibration_frame(image: &grainbase::LinearImage) -> geometry::Frame {
    let frame = geometry::find(image).unwrap_or_else(|| geometry::Frame::whole(image));
    println!(
        "frame: center {:.0},{:.0}  size {:.0}x{:.0}  skew {:.4} rad ({:.2} deg)",
        frame.center[0],
        frame.center[1],
        frame.size[0],
        frame.size[1],
        frame.skew,
        frame.skew.to_degrees()
    );
    frame
}

/// Measures the calibration base on the still-uncropped scan.
///
/// The documented source precedence (spec step 3): hand-supplied values, then a
/// hand-placed rectangle, then the film between the punch bands. A scan with
/// none of those is refused — a calibration fitted on an invented base would be
/// wrong in silence.
fn calibration_base(
    image: &grainbase::LinearImage,
    frame: &geometry::Frame,
    film_base: Option<[f32; 3]>,
    base_sample: Option<[f32; 4]>,
) -> Result<FilmBase, String> {
    if let Some(values) = film_base {
        return Ok(FilmBase {
            transmittance: values.map(|value| value.max(f32::MIN_POSITIVE)),
        });
    }
    if let Some(rect) = base_sample {
        return Ok(FilmBase::detect_in_rect(
            image,
            rect,
            film_base::DEFAULT_PERCENTILE,
        ));
    }
    if let Some(base) = measured_between_sprockets(image, frame) {
        return Ok(base);
    }
    Err(
        "no film base could be measured: the scan shows no sprocket bands to measure \
         between. Pass --film-base r,g,b or --base-sample x0,y0,x1,y1."
            .to_owned(),
    )
}

/// Verifies that a profile round-trips and writes it, reporting where it went.
///
/// The round trip is not ceremony: a profile the reader cannot parse is worse
/// than no profile, because the write would look like success.
fn write_profile(
    profile: &Profile,
    out: Option<PathBuf>,
    capture: &Path,
) -> Result<PathBuf, String> {
    let text = profile.to_string();
    if let Err(err) = Profile::parse(&text) {
        return Err(format!("generated profile does not round-trip: {err}"));
    }

    let output = out.unwrap_or_else(|| default_profile_path(capture));
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "could not create output directory {}: {err}",
                parent.display()
            )
        })?;
    }
    fs::write(&output, text)
        .map_err(|err| format!("could not write {}: {err}", output.display()))?;
    Ok(output)
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

/// Prints the summary for a profile built from one measured neutral patch.
fn print_neutral_summary(profile: &Profile) {
    let ceilings = profile
        .settings
        .density_max
        .expect("a neutral fit emits ceilings");
    let family = profile.family.expect("a rig profile carries a family");

    println!(
        "rig profile for {} on {}",
        family_name(family),
        profile.camera
    );
    println!("  from a measured neutral patch");
    println!(
        "  white point (density)  R {:.4}  G {:.4}  B {:.4}",
        ceilings[0], ceilings[1], ceilings[2]
    );
    println!(
        "  ceiling ratios         G/R {:.3}  B/R {:.3}",
        ceilings[1] / ceilings[0].max(f32::MIN_POSITIVE),
        ceilings[2] / ceilings[0].max(f32::MIN_POSITIVE)
    );
}

/// Prints the rig-calibration summary: family, rig, axis, steps, and the
/// per-channel fit so a non-straight response is visible.
fn print_calibrate_rig_summary(profile: &Profile, measurement: &wedge::Measurement) {
    let floors = profile
        .settings
        .density_floor
        .expect("a wedge fit emits floors");
    let ceilings = profile
        .settings
        .density_max
        .expect("a wedge fit emits ceilings");
    let family = profile.family.expect("a rig profile carries a family");
    let fit = wedge::fit(measurement).expect("measurement already succeeded");

    println!(
        "rig profile for {} on {}",
        family_name(family),
        profile.camera
    );
    println!(
        "  axis: {}  steps: {}",
        match measurement.axis {
            wedge::Axis::Vertical => "vertical",
            wedge::Axis::Horizontal => "horizontal",
        },
        measurement.steps.len()
    );
    println!(
        "  black point (density)  R {:.4}  G {:.4}  B {:.4}",
        floors[0], floors[1], floors[2]
    );
    println!(
        "  white point (density)  R {:.4}  G {:.4}  B {:.4}",
        ceilings[0], ceilings[1], ceilings[2]
    );
    println!(
        "  slope                  R {:.4}  G {:.4}  B {:.4}",
        fit.slope[0], fit.slope[1], fit.slope[2]
    );
    println!(
        "  straightness ratio     R {:.4}  G {:.4}  B {:.4}",
        wedge::straightness_ratio(&fit, 0),
        wedge::straightness_ratio(&fit, 1),
        wedge::straightness_ratio(&fit, 2)
    );
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

/// Parses `--wedge-steps D,D,...` as reflection densities.
fn parse_wedge_steps(text: &str) -> Result<Vec<f32>, String> {
    let parts: Vec<&str> = text.split(',').collect();
    if parts.len() < 2 {
        return Err("expected at least two comma-separated densities".to_owned());
    }
    parts
        .iter()
        .map(|part| {
            part.trim()
                .parse()
                .map_err(|_| format!("`{part}` is not a number"))
        })
        .collect()
}

/// Reads a profile from disk, or falls back to neutral defaults.
///
/// A profile that cannot be read is a hard error rather than a silent fallback:
/// converting a whole roll with defaults because a path was mistyped would
/// produce a folder of wrong output that looks deliberate.
fn load_profile(path: Option<&Path>, expected_family: FilmMode) -> Result<Settings, String> {
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

    if let Some(family) = profile.family
        && family != expected_family
    {
        eprintln!(
            "warning: profile family {} does not match job family {}",
            family_name(family),
            family_name(expected_family)
        );
    }

    Ok(profile.settings)
}
