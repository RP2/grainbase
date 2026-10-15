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

use crate::pipeline::{self, FilmBase, Settings, film_base};
use crate::{decode, encode};

/// Extensions treated as camera RAW input, matched case-insensitively.
const RAW_EXTENSIONS: &[&str] = &[
    "arw", "dng", "cr2", "cr3", "nef", "nrw", "rw2", "raf", "orf", "pef", "srw",
];

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
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decode(err) => write!(f, "{err}"),
            Self::Encode(err) => write!(f, "{err}"),
            Self::Io(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Decode(err) => Some(err),
            Self::Encode(err) => Some(err),
            Self::Io(err) => Some(err),
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

    let base = roll_base(&inputs);

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

    match convert_frame(input, &output, &job.settings, base) {
        Ok(()) => Outcome::Converted(output),
        Err(err) => Outcome::Failed(input.to_path_buf(), err.to_string()),
    }
}

/// Decodes one frame, inverts it, and writes the positive.
///
/// When `base` is `None` the frame measures its own, which is what a single
/// frame run does.
///
/// # Errors
///
/// Propagates decode, encode, and output-directory failures.
pub fn convert_frame(
    input: &Path,
    output: &Path,
    settings: &Settings,
    base: Option<&FilmBase>,
) -> Result<(), Error> {
    let scan = decode::decode(input)?;

    // Measured only when the caller has no roll-level base to supply.
    let measured;
    let base = if let Some(base) = base {
        base
    } else {
        measured = FilmBase::detect(&scan.linear, film_base::DEFAULT_PERCENTILE);
        &measured
    };

    let positive = pipeline::invert(&scan.linear, base, settings);

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
fn roll_base(inputs: &[PathBuf]) -> Option<FilmBase> {
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
            Some(FilmBase::detect(
                &scan.linear,
                film_base::DEFAULT_PERCENTILE,
            ))
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
        });

        assert_eq!(report.failures().len(), 1);
        assert_eq!(report.converted(), 0);
    }
}
