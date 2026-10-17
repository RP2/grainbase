//! Batch conversion: a folder of scans in, finished positives out.
//!
//! The folder is treated as one roll. That matters for the film base: the mask,
//! the light source and the camera's channel gains are all fixed by the setup,
//! while the scene changes from frame to frame, so the base is a roll property.
//! Measuring it per frame would let scan noise move the cast around between
//! frames of the same roll.

use std::fmt;
use std::path::{Path, PathBuf};
use std::{fs, io};

use rayon::prelude::*;

use crate::geometry;
pub use crate::image::Flips;
use crate::image::LinearImage;
use crate::overlay;
use crate::pipeline::{self, FilmBase, Settings, film_base};
use crate::{decode, encode};

/// Extensions treated as camera RAW input, matched case-insensitively.
const RAW_EXTENSIONS: &[&str] = &[
    "arw", "dng", "cr2", "cr3", "nef", "nrw", "rw2", "raf", "orf", "pef", "srw",
];

/// Sample points drawn in the diagnostic overlay.
///
/// A drawing of every point the base sample reads is a filled rectangle; this
/// is enough marks to show the shape of the sample and which side of the strip
/// it came from.
const OVERLAY_SAMPLE_MARKS: usize = 400;

/// Frames sampled to measure the roll's film base.
///
/// Decoding is the expensive part, and these frames are decoded twice: once to
/// sample the base, once to convert. A handful is enough, and they are spread
/// across the folder so one unusual frame cannot set the base for the roll.
const BASE_SAMPLE: usize = 3;

/// Which kind of film is being converted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilmMode {
    /// Color negative. The only mode implemented so far.
    ColorNegative,
}

/// What to convert, and where to put it.
#[derive(Debug, Clone, PartialEq)]
pub struct Job {
    /// Folder of scans. Treated as one roll.
    pub input: PathBuf,
    /// Folder for the finished positives. Mirrors the input tree.
    pub output: PathBuf,
    /// Which film the scans are.
    pub mode: FilmMode,
    /// Conversion settings. Neutral unless a calibration profile was loaded.
    pub settings: Settings,
    /// Whether to locate the film frame and crop to it.
    ///
    /// Off in the free CLI: the frame's edges are what a camera scan states
    /// least reliably — the film's aspect varies with the crop the lab or the
    /// rig applied, and content edges inside the picture imitate the film's own
    /// — so the conversion keeps the whole scan and the operator crops. The
    /// base measurement does not depend on this: it is taken from the film the
    /// punch bands bound, which is found without a frame.
    pub crop: bool,
    /// The operator's orientation decision (spec step 1), carried by the whole
    /// roll: a roll is scanned one way.
    pub flips: Flips,
    /// Whether to write a diagnostic overlay beside each positive.
    ///
    /// The overlay is the oriented scan with the pipeline's own marks on it:
    /// the frame it found, the punch bands, and the film the base was measured
    /// from. Off by default: it is a debugging surface, not an output.
    pub overlay: bool,
    /// Whether to meter each frame's integral color before rendering.
    ///
    /// The meter pulls the frame's integral density back toward its own
    /// neutral, so a film or scanning cast is corrected without the scene's own
    /// light being neutralised; the pull is weighted down when the frame is a
    /// scene that is genuinely one color rather than a cast. Per frame, because
    /// the cast moves while the unexposed stock does not.
    pub meter: bool,
    /// Optional authoritative base sample rectangle, as fractions `[x0, y0, x1,
    /// y1]` of the uncropped scan.
    ///
    /// When supplied, the film base is measured from exactly this rectangle and
    /// the band machinery is ignored. For glare-heavy rigs a hand-placed
    /// rebate rectangle beats every heuristic.
    pub base_sample: Option<[f32; 4]>,
    /// The film base supplied by hand, as per-channel transmittance.
    ///
    /// The reference tools' own escape hatch for a border they cannot locate:
    /// NLP and `VueScan` both document sampling the film border by hand rather
    /// than inferring it from frame statistics. When set, it is authoritative
    /// and no detector runs.
    pub film_base: Option<[f32; 3]>,
}

/// What happened to one frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// Converted to this output path.
    Converted(PathBuf),
    /// Output was already up to date.
    Skipped(PathBuf),
    /// Failed, with a human-readable reason.
    Failed(PathBuf, String),
}

/// The result of a whole run.
#[derive(Debug, Default, Clone)]
pub struct Report {
    /// One entry per input frame, in input order.
    pub outcomes: Vec<Outcome>,
}

impl Report {
    /// How many frames were converted.
    #[must_use]
    pub fn converted(&self) -> usize {
        self.count_matching(|outcome| matches!(outcome, Outcome::Converted(_)))
    }

    /// How many frames were skipped as already up to date.
    #[must_use]
    pub fn skipped(&self) -> usize {
        self.count_matching(|outcome| matches!(outcome, Outcome::Skipped(_)))
    }

    /// The failures, as `(path, reason)`.
    #[must_use]
    pub fn failures(&self) -> Vec<(&Path, &str)> {
        self.outcomes
            .iter()
            .filter_map(|outcome| match outcome {
                Outcome::Failed(path, reason) => Some((path.as_path(), reason.as_str())),
                Outcome::Converted(_) | Outcome::Skipped(_) => None,
            })
            .collect()
    }

    fn count_matching(&self, predicate: impl Fn(&Outcome) -> bool) -> usize {
        self.outcomes.iter().filter(|o| predicate(o)).count()
    }
}

/// Why a frame could not be converted.
#[derive(Debug)]
pub enum Error {
    /// The scan could not be decoded.
    Decode(decode::Error),
    /// The positive could not be written.
    Encode(encode::Error),
    /// The output directory could not be created.
    Io(io::Error),
    /// The film base could not be measured from the scan.
    ///
    /// The punch bands are what a base measurement is built on: they mark where
    /// the film is, and the film between them holds the unexposed base. A scan
    /// whose bands are not visible leaves nothing to measure, and inventing a
    /// value there is worse than refusing — the operator knows their film and
    /// can hand the base over.
    NoFilmBase,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decode(err) => write!(f, "{err}"),
            Self::Encode(err) => write!(f, "{err}"),
            Self::Io(err) => write!(f, "{err}"),
            Self::NoFilmBase => write!(
                f,
                "no film base could be measured: the scan shows no sprocket bands to measure \
                 between. Measure the base by hand and pass --film-base r,g,b, or a rebate \
                 rectangle as --base-sample x0,y0,x1,y1"
            ),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Decode(err) => Some(err),
            Self::Encode(err) => Some(err),
            Self::Io(err) => Some(err),
            Self::NoFilmBase => None,
        }
    }
}

impl From<decode::Error> for Error {
    fn from(err: decode::Error) -> Self {
        Self::Decode(err)
    }
}

impl From<encode::Error> for Error {
    fn from(err: encode::Error) -> Self {
        Self::Encode(err)
    }
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

/// Converts every scan in the job's folder.
///
/// Frames are processed in parallel. A frame that fails does not stop the run:
/// one unreadable scan should not cost a whole roll, so failures are collected
/// in the report. Order is preserved regardless of thread scheduling.
#[must_use]
pub fn run(job: &Job) -> Report {
    let mut inputs = Vec::new();
    if let Err(err) = collect_inputs(&job.input, &mut inputs) {
        return Report {
            outcomes: vec![Outcome::Failed(job.input.clone(), err.to_string())],
        };
    }
    // Deterministic order, so the sampling below is reproducible.
    inputs.sort();

    let base = roll_base(&inputs, job.flips, job.base_sample);

    let outcomes = inputs
        .par_iter()
        .map(|input| convert_one(job, input, base.as_ref()))
        .collect();

    Report { outcomes }
}

/// Converts one frame, or reports why not.
fn convert_one(job: &Job, input: &Path, base: Option<&FilmBase>) -> Outcome {
    let Some(output) = output_path_for(&job.input, &job.output, input) else {
        return Outcome::Failed(
            input.to_path_buf(),
            "input is not inside the job's input folder".to_owned(),
        );
    };

    if is_up_to_date(input, &output) {
        return Outcome::Skipped(output);
    }

    match convert_frame(
        input,
        &output,
        &job.settings,
        base,
        job.flips,
        job.crop,
        job.base_sample,
        job.film_base,
        job.overlay,
        job.meter,
    ) {
        Ok(()) => Outcome::Converted(output),
        Err(err) => Outcome::Failed(input.to_path_buf(), err.to_string()),
    }
}

/// Crops to the film frame, when asked and when one can be found.
///
/// Anything ambiguous leaves the image alone. A bad crop is worse than no crop,
/// and an underexposed frame has weaker edges.
fn cropped(image: &LinearImage, frame: &geometry::Frame) -> LinearImage {
    geometry::apply(image, frame)
}

/// Decodes one frame, locates it, measures its base, crops it, inverts it, and
/// writes the positive.
///
/// The order is fixed by the documented lab pipeline (see the private repo's
/// `docs/protocol/negative-scan-order-of-operations.md`; numbered steps below
/// cite it):
///
/// 1. *Orient first*: the operator's flips — the emulsion-side mirror and any
///    180° — are applied to the raw scan before anything is detected, one
///    mirror decision per roll, never derived from EXIF.
/// 2. *Find the frame* on the oriented scan. The frame carries the mechanical
///    knowledge of where the film and its rebate are.
/// 3. *Measure the film base on the still-uncropped scan*, by source
///    precedence. A caller-supplied roll-level base wins (the roll's base is
///    one measurement, per the lab tools' border-lock practice); then a
///    manually supplied per-channel base; then a manually supplied sample
///    rectangle; then the clear film just outside the frame's own boundary,
///    which is the film's mechanical answer and needs neither a rebate region
///    nor a punch band to be recognised; then the whole-frame statistics the
///    scanner tools document as their default base estimate —
///    whose known limit is light-table glare on copy-stand scans, for which
///    the manual inputs are the documented answer, not a further heuristic.
/// 4. *Crop* to the exposed frame, so the inversion only sees picture pixels.
/// 5. *Invert*, with the settings (a profile when loaded).
///
/// # Errors
///
/// Propagates decode, encode, and output-directory failures.
#[expect(
    clippy::too_many_arguments,
    reason = "the documented order has five measured inputs plus the operator's \
              switches; collapsing them into a struct would hide which \
              parameters the caller did not supply"
)]
pub fn convert_frame(
    input: &Path,
    output: &Path,
    settings: &Settings,
    base: Option<&FilmBase>,
    flips: Flips,
    crop: bool,
    base_sample: Option<[f32; 4]>,
    film_base: Option<[f32; 3]>,
    overlay: bool,
    meter: bool,
) -> Result<(), Error> {
    let scan = decode::decode(input)?;

    // Orient before anything is measured: the flips are the operator's
    // decision about which side of the film faced the lens.
    let uncropped = scan.linear.flipped(flips);

    // Find the frame once. Its punch bands and the film they bound are what the
    // base is measured from; its rectangle is what a crop would use, when the
    // caller asked for one.
    let frame = geometry::find(&uncropped).unwrap_or_else(|| geometry::Frame::whole(&uncropped));
    let cropped = if crop {
        let c = cropped(&uncropped, &frame);
        if c.width == 0 || c.height == 0 {
            None
        } else {
            Some(c)
        }
    } else {
        None
    };

    // The base is always measured on the uncropped scan: neutralizing the
    // mask precedes the crop in the documented order.
    let measured;
    let base = if let Some(base) = base {
        base
    } else if let Some(values) = film_base {
        measured = FilmBase {
            transmittance: values.map(|value| value.max(f32::MIN_POSITIVE)),
        };
        &measured
    } else if let Some(rect) = base_sample {
        measured = FilmBase::detect_in_rect(&uncropped, rect, film_base::DEFAULT_PERCENTILE);
        &measured
    } else if let Some(base) = measured_between_sprockets(&uncropped, &frame) {
        // The base is the unexposed film. The punch rows between the
        // perforations are where it is guaranteed to be: outside the exposure,
        // and clear of the light table. Nothing here reads the picture.
        measured = base;
        &measured
    } else {
        // Nothing to measure from: refuse rather than invent a base.
        return Err(Error::NoFilmBase);
    };

    // The meter pulls the frame's integral back to neutral. It runs here, after
    // the base is measured and the frame is located, so the region it reads is the
    // exposure rather than the rebate or the light table.
    let mut frame_settings = *settings;
    if meter {
        frame_settings = metered_settings(&frame_settings, &uncropped, base, &frame);
    }

    let positive = pipeline::invert(
        cropped.as_ref().unwrap_or(&uncropped),
        base,
        &frame_settings,
    );

    // With the crop off, the output *is* the scan the decisions were made on, so
    // the marks belong on it: one image shows the final frame and where the
    // pipeline looked, in the same coordinates. With the crop on, the output is
    // a sub-rectangle the marks would not line up with, so they go to a separate
    // annotated scan instead.
    let positive = if overlay {
        let bands: Vec<film_base::FilmRegion> = frame.holes.iter().flatten().copied().collect();
        let samples = FilmBase::sprocket_sample_points(&uncropped, &bands, OVERLAY_SAMPLE_MARKS);
        let sprockets = FilmBase::sprocket_holes(&uncropped, &bands);
        let marks = overlay::Marks {
            holes: &frame.holes,
            sprockets: &sprockets,
            film: frame.film.as_ref(),
            samples: &samples,
            color_region: Some(&frame.sample_region().exposure_region(&uncropped, base)),
            crop: if crop { Some(&frame) } else { None },
        };
        if cropped.is_some() {
            encode::encode(&overlay::annotate(&uncropped, &marks), overlay_path(output))?;
            positive
        } else {
            overlay::annotate(&positive, &marks)
        }
    } else {
        positive
    };

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    encode::encode(&positive, output)?;
    Ok(())
}

/// Measures the roll's film base from a spread of frames.
///
/// Frames are sampled evenly across the folder and the per-channel median is
/// taken, so one odd frame — a blank, a very dense night shot — cannot set the
/// base for the whole roll. Returns `None` when nothing could be sampled, and
/// callers then fall back to measuring per frame.
fn roll_base(inputs: &[PathBuf], flips: Flips, base_sample: Option<[f32; 4]>) -> Option<FilmBase> {
    if inputs.len() <= 1 {
        // Nothing to pool, and one frame is its own best estimate.
        return None;
    }

    let step = (inputs.len() / BASE_SAMPLE).max(1);
    let mut sampled: Vec<FilmBase> = inputs
        .iter()
        .step_by(step)
        .take(BASE_SAMPLE)
        .filter_map(|input| {
            let scan = decode::decode(input).ok()?;
            let oriented = scan.linear.flipped(flips);
            let frame =
                geometry::find(&oriented).unwrap_or_else(|| geometry::Frame::whole(&oriented));
            // A frame with nothing to measure contributes nothing; the
            // conversion refuses rather than guessing if none of them does.
            roll_frame_base(&oriented, &frame, base_sample)
        })
        .collect();

    if sampled.is_empty() {
        return None;
    }

    let mut transmittance = [0.0f32; 3];
    for (channel, slot) in transmittance.iter_mut().enumerate() {
        let mut values: Vec<f32> = sampled
            .iter()
            .map(|base| base.transmittance[channel])
            .collect();
        let middle = values.len() / 2;
        values.select_nth_unstable_by(middle, f32::total_cmp);
        *slot = values[middle];
    }
    sampled.clear();

    Some(FilmBase {
        transmittance: transmittance.map(|value| value.max(f32::MIN_POSITIVE)),
    })
}

/// One sampled frame's base for the roll measurement.
///
/// The same source ordering as [`convert_frame`], minus the supplied-base
/// case: a manual rectangle, else the film the punch bands bound.
fn roll_frame_base(
    uncropped: &LinearImage,
    frame: &geometry::Frame,
    base_sample: Option<[f32; 4]>,
) -> Option<FilmBase> {
    if let Some(rect) = base_sample {
        return Some(FilmBase::detect_in_rect(
            uncropped,
            rect,
            film_base::DEFAULT_PERCENTILE,
        ));
    }
    measured_between_sprockets(uncropped, frame)
}

/// The base measured from the film between the sprocket holes, when the scan
/// shows any punch at all.
///
/// The punch bands are the one structure a camera scan states reliably, and the
/// film between their perforations is unexposed by construction. A frame with no
/// band has no base to measure, and the caller has to ask the operator.
fn measured_between_sprockets(image: &LinearImage, frame: &geometry::Frame) -> Option<FilmBase> {
    let bands: Vec<film_base::FilmRegion> = frame.holes.iter().flatten().copied().collect();
    if bands.is_empty() {
        return None;
    }
    let base = FilmBase::detect_between_sprockets(image, &bands);
    // The identity base means the sampler found nothing to read — a band with no
    // lit punch, or no gap between two holes. That is not a measurement, and
    // converting with it would render every frame wrong in silence.
    (base != FilmBase::IDENTITY).then_some(base)
}

/// The settings with the frame's metered printer point folded in, or unchanged
/// when the frame cannot be metered.
///
/// The meter's correction is a shift of the density axis, which is exactly a
/// moved black point: the render already reads a per-channel `density_floor`,
/// so the printer point is written there and the ceiling — and so the contrast —
/// is left alone. The analysis region is the frame's own exposure region, the
/// same region every picture-reading operation samples, so the integral never
/// counts the rebate, the holes, or the light table.
fn metered_settings(
    settings: &Settings,
    image: &LinearImage,
    base: &FilmBase,
    frame: &geometry::Frame,
) -> Settings {
    let region = frame.sample_region().exposure_region(image, base);
    let Some(metering) = pipeline::meter::meter(image, base, region.rect_fractions(image)) else {
        return *settings;
    };
    let floor = settings.density_floor.unwrap_or([0.0; 3]);
    Settings {
        density_floor: Some([0, 1, 2].map(|channel| floor[channel] - metering.correction[channel])),
        ..*settings
    }
}

/// Where the diagnostic overlay for an output goes: beside it, marked.
///
/// The overlay is the annotated scan a conversion measured from — the frame it
/// found, the punch bands it kept clear of, and the film it read the base
/// from. It exists
/// so that a wrong result can be told apart from a wrong report, and so a bug
/// report can carry the pipeline's own view of the scan.
fn overlay_path(output: &Path) -> PathBuf {
    let mut path = output.to_path_buf();
    let stem = output.file_stem().map_or_else(
        || "overlay".to_owned(),
        |stem| format!("{}-overlay", stem.to_string_lossy()),
    );
    path.set_file_name(stem);
    path.set_extension("png");
    path
}

/// Where the positive for `file` goes: the input tree mirrored, extension `.png`.
fn output_path_for(input_root: &Path, output_root: &Path, file: &Path) -> Option<PathBuf> {
    let relative = file.strip_prefix(input_root).ok()?;
    let mut output = output_root.join(relative);
    output.set_extension("png");
    Some(output)
}

/// Whether an existing output can stand in for converting the input again.
///
/// A plain comparison rather than a recorded hash: an output at least as new as
/// its input is treated as current. Content-hash sidecars would survive a move
/// that changes timestamps, and are the refinement to make here.
fn is_up_to_date(input: &Path, output: &Path) -> bool {
    let (Ok(input_meta), Ok(output_meta)) = (input.metadata(), output.metadata()) else {
        return false;
    };
    if output_meta.len() == 0 {
        return false;
    }
    let (Ok(input_time), Ok(output_time)) = (input_meta.modified(), output_meta.modified()) else {
        return false;
    };
    output_time >= input_time
}

/// Whether a path looks like a camera RAW we can decode.
fn is_raw(path: &Path) -> bool {
    path.extension()
        .and_then(std::ffi::OsStr::to_str)
        .is_some_and(|extension| {
            RAW_EXTENSIONS
                .iter()
                .any(|known| extension.eq_ignore_ascii_case(known))
        })
}

/// Collects RAW files under `root`, recursively.
fn collect_inputs(root: &Path, found: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_inputs(&path, found)?;
        } else if is_raw(&path) {
            found.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("grainbase-driver-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create temp dir");
        path
    }

    #[test]
    fn output_mirrors_the_input_tree() {
        let root = Path::new("/scans");
        let out = Path::new("/out");

        assert_eq!(
            output_path_for(root, out, Path::new("/scans/roll1/DSC01.arw")),
            Some(PathBuf::from("/out/roll1/DSC01.png"))
        );
        assert_eq!(
            output_path_for(root, out, Path::new("/scans/DSC02.ARW")),
            Some(PathBuf::from("/out/DSC02.png"))
        );
    }

    #[test]
    fn a_file_outside_the_input_root_has_no_output() {
        assert_eq!(
            output_path_for(
                Path::new("/scans"),
                Path::new("/out"),
                Path::new("/other/a.arw")
            ),
            None
        );
    }

    #[test]
    fn raw_extensions_are_matched_case_insensitively() {
        assert!(is_raw(Path::new("a.arw")));
        assert!(is_raw(Path::new("a.ARW")));
        assert!(is_raw(Path::new("a.DNG")));
        assert!(!is_raw(Path::new("a.png")));
        assert!(!is_raw(Path::new("a.txt")));
        assert!(!is_raw(Path::new("noextension")));
    }

    #[test]
    fn a_missing_output_is_never_up_to_date() {
        let dir = temp_dir("missing-output");
        let input = dir.join("a.arw");
        fs::write(&input, b"x").expect("write input");

        assert!(!is_up_to_date(&input, &dir.join("a.png")));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_empty_output_is_not_up_to_date() {
        let dir = temp_dir("empty-output");
        let input = dir.join("a.arw");
        let output = dir.join("a.png");
        fs::write(&input, b"x").expect("write input");
        fs::write(&output, b"").expect("write output");

        assert!(
            !is_up_to_date(&input, &output),
            "a truncated output is not a result"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_output_at_least_as_new_as_its_input_is_up_to_date() {
        let dir = temp_dir("fresh-output");
        let input = dir.join("a.arw");
        let output = dir.join("a.png");
        fs::write(&input, b"x").expect("write input");
        fs::write(&output, b"result").expect("write output");

        assert!(is_up_to_date(&input, &output));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_folder_with_no_raws_converts_nothing() {
        let dir = temp_dir("no-raws");
        fs::write(dir.join("readme.txt"), b"nothing here").expect("write");

        let report = run(&Job {
            input: dir.clone(),
            output: dir.join("out"),
            mode: FilmMode::ColorNegative,
            settings: Settings::default(),
            crop: true,
            flips: Flips::default(),
            overlay: false,
            meter: false,
            base_sample: None,
            film_base: None,
        });

        assert!(report.outcomes.is_empty(), "got {:?}", report.outcomes);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn nested_raws_are_found() {
        let dir = temp_dir("nested");
        let nested = dir.join("roll1/inner");
        fs::create_dir_all(&nested).expect("create nested");
        fs::write(nested.join("a.arw"), b"x").expect("write");

        let mut found = Vec::new();
        collect_inputs(&dir, &mut found).expect("walk");

        assert_eq!(found, vec![nested.join("a.arw")]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_input_folder_is_reported_not_panicked() {
        let report = run(&Job {
            input: PathBuf::from("/nonexistent/grainbase-roll"),
            output: PathBuf::from("/tmp/grainbase-out"),
            mode: FilmMode::ColorNegative,
            settings: Settings::default(),
            crop: true,
            flips: Flips::default(),
            overlay: false,
            meter: false,
            base_sample: None,
            film_base: None,
        });

        assert_eq!(report.failures().len(), 1);
        assert_eq!(report.converted(), 0);
    }
}
