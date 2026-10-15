//! Decoding a scan file into linear-light pixels.

use std::fmt;
use std::fs::File;
use std::io::BufReader;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

use rawler::decoders::Orientation;
use rawler::imgop::develop::{Intermediate, ProcessingStep, RawDevelop};

use crate::image::{LinearImage, SampleCountMismatch};

/// Development steps handed to rawler.
///
/// `SRgb` is the only default step excluded: it applies sRGB gamma, and the
/// pipeline needs linear light.
///
/// `WhiteBalance` and `Calibrate` are *included*. The camera matrix describes the
/// sensor's response to scene light, and a negative's per-channel transmittance is
/// linear in that light, so the matrix belongs before inversion — this is what
/// darktable's negadoctor and the abpy method both do. Measured on the a6000
/// fixture it also produces the physically correct orange-mask signature (base
/// red highest, blue lowest), which camera space does not.
///
/// Including it does not weaken film-base neutralisation: a linear transform maps
/// the mask, which is constant across the frame, to another per-channel constant,
/// so dividing it out still works.
const DEVELOP_STEPS: &[ProcessingStep] = &[
    ProcessingStep::Rescale,
    ProcessingStep::Demosaic,
    ProcessingStep::FujiRotate,
    ProcessingStep::CropActiveArea,
    ProcessingStep::WhiteBalance,
    ProcessingStep::Calibrate,
    ProcessingStep::CropDefault,
];

/// A decoded scan: pixels plus the capture metadata that travels with them.
#[derive(Debug, Clone)]
pub struct Scan {
    /// Linear-light RGB in the camera's white-balanced color space.
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
    /// The EXIF orientation value (1-8) that was applied to the decoded pixels,
    /// or `None` when the file carried no valid orientation tag.
    ///
    /// Recorded rather than swallowed because the tag describes how the *camera*
    /// was held, which is not necessarily how the *film* should end up. A copy
    /// rig can be inverted or rotated relative to the film, so a caller may need
    /// to override this. See the orientation ticket.
    pub orientation: Option<u16>,
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

/// Decodes one camera RAW scan into linear-light sRGB.
///
/// The camera's white balance and color matrix are applied, so the result is in a
/// defined color space and can be tagged honestly on output. No gamma is applied:
/// the result is linear light, ready for film-base neutralisation.
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

    let orientation = exif_orientation(path);

    let meta = ScanMeta {
        make: raw.make.clone(),
        model: raw.model.clone(),
        orientation,
    };

    let intermediate = catch_unwind(AssertUnwindSafe(|| {
        RawDevelop::new_with(DEVELOP_STEPS).develop_intermediate(&raw)
    }))
    .map_err(|_| Error::Panicked)?
    .map_err(|err| Error::Failed(err.to_string()))?;

    let linear = to_linear(intermediate, &meta)?;

    Ok(Scan { linear, meta })
}

/// Applies an EXIF orientation value to a decoded image.
///
/// Deliberately **not** applied by [`decode`]. The tag describes how the *camera*
/// was held, which is not the same as how the *film* should end up: a copy-rig
/// camera points down at the film, and an orientation sensor has no reliable "up"
/// in that position, so it reports a meaningless value. On the maintainer's rig
/// the tag says "rotate 270" while the frame is correctly landscape.
///
/// Kept here, tested, so a caller that knows its own rig can apply it
/// deliberately rather than having it applied silently. Frame-derived orientation
/// is the real answer; see the orientation ticket.
///
/// Unrecognized values leave the image alone.
#[must_use]
pub fn apply_exif_orientation(image: LinearImage, orientation: Option<u16>) -> LinearImage {
    let Some(value) = orientation else {
        return image;
    };
    let (transpose, flip_horizontal, flip_vertical) = Orientation::from_u16(value).to_flips();

    let mut oriented = image;
    if transpose {
        oriented = oriented.transposed();
    }
    if flip_horizontal {
        oriented = oriented.flipped_horizontal();
    }
    if flip_vertical {
        oriented = oriented.flipped_vertical();
    }
    oriented
}

/// The EXIF orientation value from the file, if it carries a valid one.
///
/// Read separately rather than taken from the decoded image: `rawler` parses EXIF
/// but hardcodes `RawImage::orientation` to `Normal` with a `TODO fixme`, so that
/// field is always `Normal` and cannot be trusted.
fn exif_orientation(path: &Path) -> Option<u16> {
    let file = File::open(path).ok()?;
    let exif = exif::Reader::new()
        .read_from_container(&mut BufReader::new(file))
        .ok()?;
    let value = exif
        .get_field(exif::Tag::Orientation, exif::In::PRIMARY)?
        .value
        .get_uint(0)?;

    // 1..=8 are the defined values; anything else is treated as absent rather
    // than guessed at.
    u16::try_from(value).ok().filter(|v| (1..=8).contains(v))
}

/// Converts rawler's develop output into the engine's image type.
fn to_linear(intermediate: Intermediate, meta: &ScanMeta) -> Result<LinearImage, Error> {
    match intermediate {
        Intermediate::ThreeColor(image) => {
            let width = dimension(image.width, "width")?;
            let height = dimension(image.height, "height")?;
            LinearImage::new(width, height, image.into_flatten()).map_err(Error::Malformed)
        }
        // Reached only for a monochrome sensor, or a four-color CFA. Neither is
        // a color-negative scan; say so rather than guessing at a conversion.
        Intermediate::Monochrome(_) => Err(Error::Unsupported {
            make: meta.make.clone(),
            model: meta.model.clone(),
            detail: "monochrome capture".to_owned(),
        }),
        Intermediate::FourColor(_) => Err(Error::Unsupported {
            make: meta.make.clone(),
            model: meta.model.clone(),
            detail: "four-color sensor pattern".to_owned(),
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

    /// A 3x2 image whose red channel is `1..=6`, for checking orientation.
    fn numbered() -> LinearImage {
        let data = (1..=6u16).flat_map(|v| [f32::from(v), 0.0, 0.0]).collect();
        LinearImage::new(3, 2, data).expect("well-formed")
    }

    fn reds(image: &LinearImage) -> Vec<f32> {
        image.data.as_chunks::<3>().0.iter().map(|p| p[0]).collect()
    }

    #[test]
    fn every_exif_orientation_value_is_handled() {
        // Pinned so the transform order cannot drift. Each case is the
        // composition of rawler's flags: transpose, then flips.
        let cases = [
            (1u16, (3, 2), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            (2, (3, 2), [3.0, 2.0, 1.0, 6.0, 5.0, 4.0]),
            (3, (3, 2), [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
            (4, (3, 2), [4.0, 5.0, 6.0, 1.0, 2.0, 3.0]),
        ];

        for (value, (width, height), expected) in cases {
            let oriented = apply_exif_orientation(numbered(), Some(value));
            assert_eq!(
                (oriented.width, oriented.height),
                (width, height),
                "EXIF {value} changed the dimensions"
            );
            assert_eq!(reds(&oriented).as_slice(), expected, "EXIF {value}");
        }
    }

    #[test]
    fn orientations_that_swap_the_axes_do_swap_them() {
        for value in [5u16, 6, 7, 8] {
            let oriented = apply_exif_orientation(numbered(), Some(value));
            assert_eq!(
                (oriented.width, oriented.height),
                (2, 3),
                "EXIF {value} should swap the axes"
            );
            assert_eq!(oriented.data.len(), 18, "EXIF {value}");
        }
    }

    #[test]
    fn applying_no_orientation_leaves_the_image_alone() {
        let image = numbered();
        let untouched = apply_exif_orientation(image.clone(), None);
        assert_eq!((untouched.width, untouched.height), (3, 2));
        assert_eq!(reds(&untouched), reds(&image));
    }

    /// A minimal little-endian TIFF whose only tag is EXIF Orientation.
    fn tiff_with_orientation(value: u16) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"II");
        bytes.extend_from_slice(&42u16.to_le_bytes());
        bytes.extend_from_slice(&8u32.to_le_bytes()); // offset of the first IFD
        bytes.extend_from_slice(&1u16.to_le_bytes()); // one entry
        bytes.extend_from_slice(&0x0112u16.to_le_bytes()); // Orientation
        bytes.extend_from_slice(&3u16.to_le_bytes()); // type SHORT
        bytes.extend_from_slice(&1u32.to_le_bytes()); // count
        bytes.extend_from_slice(&value.to_le_bytes()); // value
        bytes.extend_from_slice(&0u16.to_le_bytes()); // padding
        bytes.extend_from_slice(&0u32.to_le_bytes()); // no next IFD
        bytes
    }

    #[test]
    fn reads_every_orientation_value_from_the_file() {
        for value in 1..=8u16 {
            let path = temp_file("orientation.tif", &tiff_with_orientation(value));
            assert_eq!(
                exif_orientation(&path),
                Some(value),
                "EXIF value {value} did not survive the round trip"
            );
            let _ = fs::remove_file(&path);
        }
    }

    #[test]
    fn orientation_values_map_to_the_expected_transform() {
        use rawler::decoders::Orientation;

        let expected = [
            (1u16, Orientation::Normal),
            (2, Orientation::HorizontalFlip),
            (3, Orientation::Rotate180),
            (4, Orientation::VerticalFlip),
            (5, Orientation::Transpose),
            (6, Orientation::Rotate90),
            (7, Orientation::Transverse),
            (8, Orientation::Rotate270),
        ];

        for (value, orientation) in expected {
            assert_eq!(
                Orientation::from_u16(value),
                orientation,
                "EXIF value {value} mapped wrongly"
            );
        }
    }

    #[test]
    fn an_out_of_range_orientation_is_absent_rather_than_guessed() {
        let path = temp_file("bad-orientation.tif", &tiff_with_orientation(99));
        assert_eq!(exif_orientation(&path), None);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_file_without_the_tag_has_no_orientation() {
        let path = temp_file("no-orientation.tif", b"II*\0\x08\0\0\0\0\0\0\0\0\0\0\0");
        assert_eq!(exif_orientation(&path), None);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_missing_or_garbage_file_has_no_orientation_rather_than_an_error() {
        assert_eq!(
            exif_orientation(Path::new("/nonexistent/grainbase.arw")),
            None
        );

        let path = temp_file("garbage-orientation.tif", b"not a tiff at all");
        assert_eq!(exif_orientation(&path), None);
        let _ = fs::remove_file(&path);
    }
}
