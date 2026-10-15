//! Decoding a scan file into linear-light pixels.

use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

use rawler::imgop::develop::{Intermediate, ProcessingStep, RawDevelop};

use crate::image::{LinearImage, SampleCountMismatch};

/// Development steps handed to rawler.
///
/// `WhiteBalance`, `Calibrate`, and `SRgb` are deliberately excluded. The first
/// two apply the camera's white balance and its camera-to-output colour matrix;
/// the third applies sRGB gamma. Film-base neutralisation needs the orange mask
/// to stay a per-channel scale, which only holds *before* any rendering is baked
/// in, so the decoder stops at demosaicing and cropping and hands back
/// camera-space linear light.
const DEVELOP_STEPS: &[ProcessingStep] = &[
    ProcessingStep::Rescale,
    ProcessingStep::Demosaic,
    ProcessingStep::FujiRotate,
    ProcessingStep::CropActiveArea,
    ProcessingStep::CropDefault,
];

/// A decoded scan: pixels plus the capture metadata that travels with them.
#[derive(Debug, Clone)]
pub struct Scan {
    /// Linear-light RGB in the camera's own colour space.
    pub linear: LinearImage,
    /// Capture metadata read from the file.
    pub meta: ScanMeta,
}

/// Flat capture metadata.
///
/// Deliberately film-agnostic: film stock and film mode are supplied by the
/// caller, not read out of the scan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanMeta {
    /// Camera manufacturer, as recorded in the file.
    pub make: String,
    /// Camera model, as recorded in the file.
    pub model: String,
}

/// Why a scan could not be decoded.
#[derive(Debug)]
pub enum Error {
    /// The decoder does not understand this camera, or this capture mode.
    Unsupported {
        /// Camera manufacturer.
        make: String,
        /// Camera model.
        model: String,
        /// What specifically was not understood.
        detail: String,
    },
    /// The file could not be read or decoded: missing, truncated, or corrupt.
    Failed(String),
    /// The decoder panicked. Contained per file, so one bad scan cannot abort a batch.
    Panicked,
    /// The decoder produced an image whose sample count contradicted its dimensions.
    Malformed(SampleCountMismatch),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported {
                make,
                model,
                detail,
            } => {
                write!(f, "unsupported camera '{make} {model}': {detail}")
            }
            Self::Failed(msg) => write!(f, "decode failed: {msg}"),
            Self::Panicked => f.write_str("decoder panicked"),
            Self::Malformed(mismatch) => write!(f, "malformed image: {mismatch}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Malformed(mismatch) => Some(mismatch),
            Self::Unsupported { .. } | Self::Failed(_) | Self::Panicked => None,
        }
    }
}

/// Decodes one camera RAW scan into linear-light RGB.
///
/// No white balance, colour matrix, or gamma is applied: the development steps
/// stop at demosaicing and cropping, so the result is camera-space linear
/// light, ready for film-base neutralisation.
///
/// # Errors
///
/// - [`Error::Unsupported`] — the camera or capture mode is not handled.
/// - [`Error::Failed`] — the file is missing, truncated, or corrupt.
/// - [`Error::Panicked`] — the underlying decoder panicked. `rawler` panics by
///   design on malformed input rather than saturating, so the panic is caught
///   here: one bad frame must not take down a whole batch.
/// - [`Error::Malformed`] — the decoder returned an internally inconsistent image.
pub fn decode(path: impl AsRef<Path>) -> Result<Scan, Error> {
    let path = path.as_ref();

    let raw = catch_unwind(AssertUnwindSafe(|| rawler::decode_file(path)))
        .map_err(|_| Error::Panicked)?;

    let raw = match raw {
        Ok(image) => image,
        Err(rawler::RawlerError::Unsupported {
            what, make, model, ..
        }) => {
            return Err(Error::Unsupported {
                make,
                model,
                detail: what,
            });
        }
        Err(err) => return Err(Error::Failed(err.to_string())),
    };

    let meta = ScanMeta {
        make: raw.make.clone(),
        model: raw.model.clone(),
    };

    let intermediate = catch_unwind(AssertUnwindSafe(|| {
        RawDevelop::new_with(DEVELOP_STEPS).develop_intermediate(&raw)
    }))
    .map_err(|_| Error::Panicked)?
    .map_err(|err| Error::Failed(err.to_string()))?;

    let linear = to_linear(intermediate, &meta)?;

    Ok(Scan { linear, meta })
}

/// Converts rawler's develop output into the engine's image type.
fn to_linear(intermediate: Intermediate, meta: &ScanMeta) -> Result<LinearImage, Error> {
    match intermediate {
        Intermediate::ThreeColor(image) => {
            let width = dimension(image.width, "width")?;
            let height = dimension(image.height, "height")?;
            LinearImage::new(width, height, image.into_flatten()).map_err(Error::Malformed)
        }
        // Reached only for a monochrome sensor, or a four-colour CFA. Neither is
        // a colour-negative scan; say so rather than guessing at a conversion.
        Intermediate::Monochrome(_) => Err(Error::Unsupported {
            make: meta.make.clone(),
            model: meta.model.clone(),
            detail: "monochrome capture".to_owned(),
        }),
        Intermediate::FourColor(_) => Err(Error::Unsupported {
            make: meta.make.clone(),
            model: meta.model.clone(),
            detail: "four-colour sensor pattern".to_owned(),
        }),
    }
}

/// Narrows a decoded dimension to `u32`.
///
/// A real frame cannot exceed this; the check exists so the conversion is not
/// a silent truncation.
fn dimension(value: usize, what: &str) -> Result<u32, Error> {
    u32::try_from(value)
        .map_err(|_| Error::Failed(format!("decoded {what} {value} does not fit in u32")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    /// Writes a uniquely named file under the system temp directory.
    fn temp_file(name: &str, contents: &[u8]) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("grainbase-decode-{}-{name}", std::process::id()));
        fs::write(&path, contents).expect("write temp file");
        path
    }

    #[test]
    fn a_missing_file_is_an_error_not_a_panic() {
        let result = decode("/nonexistent/grainbase-does-not-exist.arw");
        assert!(result.is_err(), "expected an error for a missing file");
    }

    #[test]
    fn garbage_is_an_error_not_a_panic() {
        let path = temp_file("garbage.arw", b"this is definitely not a raw file");
        let result = decode(&path);
        assert!(result.is_err(), "expected an error for garbage input");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn an_empty_file_is_an_error_not_a_panic() {
        let path = temp_file("empty.arw", b"");
        let result = decode(&path);
        assert!(result.is_err(), "expected an error for an empty file");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_directory_is_an_error_not_a_panic() {
        let result = decode(std::env::temp_dir());
        assert!(result.is_err(), "expected an error for a directory");
    }
}
