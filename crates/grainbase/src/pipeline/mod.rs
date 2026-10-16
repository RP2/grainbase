//! The pure conversion core: film-base neutralisation and inversion.
//!
//! No I/O, no per-frame state: the same input always produces the same output,
//! so results can be golden-tested. This is the primary seam for conversion
//! quality.
//!
//! A color negative is a positive printed through a mask. Its film base is the
//! unexposed, clear part, and it is the *brightest* thing in a scan. In the
//! finished positive that same area is black, because unexposed film means no
//! exposure. So the model is:
//!
//! ```text
//! transmittance = value / base          // removes the orange mask, per channel
//! density       = -log10(transmittance) // optical density; base becomes 0
//! positive      = density / density_max // base -> black, densest -> white
//! ```
//!
//! Working in density rather than transmittance is what gives correct tonality:
//! transmittance is multiplicative in the film but we perceive it
//! logarithmically, so a linear inversion washes out the highlights.

pub mod film_base;
pub mod percentile;

pub use film_base::FilmBase;

use crate::image::LinearImage;

/// Density mapped to the positive's white point by default, in log10 units.
///
/// A real negative's density range depends on the film stock, the exposure, and
/// the capture. Fixed, existing, deterministic: the conversion does not
/// compensate for exposure, so a thin scan stays thin — the maintainer's
/// accuracy rule. A fitted calibration profile replaces the ceiling with the
/// stock's measured value.
pub const DEFAULT_DENSITY_MAX: f32 = 2.0;

/// The Kodak LAD aims for a mid-scale neutral, per channel, in Status-M
/// densities above base.
///
/// From Kodak's *Film Users' Guide and Digital Recorder Calibration and
/// Aims* (H-387), Table VII: the Digital LAD aims are what Kodak publishes as
/// the reference gray *on the film* — a correctly exposed neutral patch of
/// C-41 negative measures these densities. The channels are unequal by
/// design: the three dye layers respond differently, and these numbers are
/// Kodak's own published statement of how much. They are the generic
/// per-channel gray correction: the per-stock measured equivalent comes from
/// the maintainer's capture protocol.
const LAD_GRAY_STATUS_M: [f32; 3] = [1.0, 1.45, 1.55];

/// The positive value the Kodak reference gray occupies: CV 470 of the Cineon
/// print-density encoding, PD = 0.002·CV → 0.940 above base, between the
/// H-387 reference black (CV 95 → 0.190) and white (CV 685 → 1.370).
///
/// Cited from Kodak H-387 and the Cineon printing-density convention. Derived
/// in this repo's arithmetic: ceiling = LAD gray ÷ this value, so a correct
/// gray lands here and the white point follows from the film's own range.
const REFERENCE_GRAY_POSITIVE: f32 = 0.636;

/// Yielding the default per-channel ceilings: LAD aims as density white
/// points, per channel, so content denser than a Kodak gray renders above the
/// gray and the film's densest regions approach white.
///
/// [1.5723, 2.2800, 2.4371] — H-387-derived, this repo's arithmetic.
#[must_use]
fn default_density_max() -> [f32; 3] {
    LAD_GRAY_STATUS_M.map(|aim| aim / REFERENCE_GRAY_POSITIVE)
}

/// Tuning for the conversion core.
///
/// This is the shape a fitted calibration profile fills in. Every field here is
/// something measurable from paired scans, and every default is a neutral
/// starting point rather than a tuned value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    /// Density at which each channel reaches the positive's white point, or
    /// `None` to use [`DEFAULT_DENSITY_MAX`].
    ///
    /// Per channel, and that is the point: the negative's three dye layers
    /// develop to slightly different contrasts. Any per-channel error grows
    /// linearly with density, so it is invisible in the midtones you balanced
    /// with and worst in the shadows — the classic residual shadow cast, and in
    /// the maintainer's scans a blue cast: one shared ceiling leaves a channel
    /// whose content runs denser with its whites unfulfilled.
    ///
    /// There is deliberately no per-frame *measurement* here: percentiles of
    /// the frame's own content would brighten thin scans into exposure
    /// compensation, and the conversion's job is accuracy, not rescue. The
    /// honest sources for per-channel values are the film itself: a fitted
    /// profile's measured values — one per stock, foreseen by the capture
    /// protocol's two neutral exposures — or the neutral default.
    pub density_max: Option<[f32; 3]>,

    /// The density each channel's black point sits at, matching the white
    /// point in the same way a print maps its own range.
    ///
    /// A real print does not stretch from density zero: the paper's own
    /// density sits below the film's thinnest content, and the *difference*
    /// carries the print's contrast. Per-channel floors are the same shape of
    /// correction as the ceilings — a channel whose thinnest content is much
    /// denser than zero is remapped to black there instead of to a washed
    /// value. Default is three zeros: no black-point lift, straight inversion.
    pub density_floor: Option<[f32; 3]>,

    /// Dye crosstalk correction, applied to density before it is scaled.
    ///
    /// Film dyes are not spectrally pure and a Bayer sensor's channels overlap,
    /// so each channel sees a mix of all three dye densities. Correcting that is
    /// a linear operation *in density*, not in linear light, because secondary
    /// absorptions are approximately linear in the other layers' densities.
    /// Identity means no correction — which is deliberately the default: no
    /// literature-derived generic matrix ships unmeasured. A row-normalized
    /// prior provably preserves neutrality, and this scan's real cast is a
    /// per-channel level imbalance the prior cannot touch; the honest sources
    /// are the film itself, fitted by the maintainer's capture protocol.
    pub crosstalk: [[f32; 3]; 3],
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            density_max: Some(default_density_max()),
            density_floor: None,
            crosstalk: IDENTITY,
        }
    }
}

/// The identity matrix: leaves density untouched.
pub const IDENTITY: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Neutralises the film base and inverts a negative into a positive.
///
/// Each channel is divided by the base, converted to optical density, corrected
/// for dye crosstalk, and scaled so the base becomes black and
/// [`Settings::density_max`] becomes white. Because the base is divided out per
/// channel, an orange mask becomes a neutral black instead of an orange cast.
///
/// Output samples are in `0.0..=1.0`. The transform is pure and deterministic.
#[must_use]
pub fn invert(image: &LinearImage, base: &FilmBase, settings: &Settings) -> LinearImage {
    // A zero or negative setting would divide by zero or invert the contrast.
    let density_max = settings
        .density_max
        .unwrap_or_else(default_density_max)
        .map(|value| value.max(f32::MIN_POSITIVE));
    let crosstalk = settings.crosstalk;
    let floor = settings.density_floor.unwrap_or([0.0; 3]);

    image.map_rgb(|[red, green, blue]| {
        let density = [
            density_of(red, base.transmittance[0]),
            density_of(green, base.transmittance[1]),
            density_of(blue, base.transmittance[2]),
        ];
        let corrected = mix(crosstalk, density);
        [
            ((corrected[0] - floor[0]) / density_max[0]).clamp(0.0, 1.0),
            ((corrected[1] - floor[1]) / density_max[1]).clamp(0.0, 1.0),
            ((corrected[2] - floor[2]) / density_max[2]).clamp(0.0, 1.0),
        ]
    })
}

/// Density beyond which a sample is simply opaque.
///
/// A real negative tops out near 3 log10; 10 is safely past anything physical
/// while keeping the arithmetic finite. Without a bound, a fully opaque sample
/// (`transmittance == 0`) yields infinity, and the crosstalk step then computes
/// `0.0 * infinity`, which is NaN and would poison the whole frame.
const MAX_DENSITY: f32 = 10.0;

/// Optical density of one sample relative to the film base.
fn density_of(value: f32, base: f32) -> f32 {
    // `decode` rejects non-finite samples, so anything else arriving here is a bug
    // in the caller rather than bad data. This is worth stating because the
    // arithmetic below is only accidentally safe against NaN: `f32::min` returns
    // the non-NaN operand, so a NaN would come out of here as MAX_DENSITY, which
    // renders as a white pixel with nothing reporting a problem.
    debug_assert!(
        value.is_finite() && base.is_finite(),
        "decode guarantees finite samples"
    );

    // Guarded rather than trusted: a zero base would make the whole frame NaN,
    // and one bad frame must not poison a batch.
    let base = base.max(f32::MIN_POSITIVE);

    // Clamped at 1.0 so that a pixel brighter than the base (dust, a light leak,
    // a specular highlight) reads as zero density instead of turning into a
    // negative density and then a negative positive.
    let transmittance = (value / base).clamp(0.0, 1.0);

    // Zero transmittance is infinite density, hence the bound.
    (-transmittance.log10()).min(MAX_DENSITY)
}

/// Applies a 3x3 matrix to a vector: `matrix * vector`, rows dotted with it.
fn mix(matrix: [[f32; 3]; 3], vector: [f32; 3]) -> [f32; 3] {
    matrix.map(|row| row[0] * vector[0] + row[1] * vector[1] + row[2] * vector[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(pixels: &[[f32; 3]]) -> LinearImage {
        let data = pixels.iter().flatten().copied().collect();
        let count = u32::try_from(pixels.len()).expect("test image is small");
        LinearImage::new(count, 1, data).expect("test image is well-formed")
    }

    fn base(transmittance: [f32; 3]) -> FilmBase {
        FilmBase { transmittance }
    }

    /// Explicit ceilings, so math tests stay pinned to known values; the
    /// automatic measurement is tested on its own.
    fn settings() -> Settings {
        Settings {
            density_max: Some([DEFAULT_DENSITY_MAX; 3]),
            ..Settings::default()
        }
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn the_film_base_becomes_black() {
        // Unexposed film is black in the positive.
        let source = image(&[[0.8, 0.8, 0.8]]);
        let result = invert(&source, &base([0.8, 0.8, 0.8]), &settings());
        for channel in result.data {
            assert_close(channel, 0.0);
        }
    }

    #[test]
    fn a_level_one_density_below_the_base_becomes_mid_gray() {
        // density = -log10(0.1) = 1.0, and density_max = 2.0.
        let source = image(&[[0.1, 0.1, 0.1]]);
        let result = invert(&source, &base([1.0, 1.0, 1.0]), &settings());
        for channel in result.data {
            assert_close(channel, 0.5);
        }
    }

    #[test]
    fn the_density_ceiling_becomes_white() {
        // density = -log10(0.01) = 2.0 = density_max.
        let source = image(&[[0.01, 0.01, 0.01]]);
        let result = invert(&source, &base([1.0, 1.0, 1.0]), &settings());
        for channel in result.data {
            assert_close(channel, 1.0);
        }
    }

    #[test]
    fn an_orange_mask_becomes_neutral_black() {
        // A strongly masked base, and a patch that is the base scaled by a
        // constant: the same neutral density in every channel, which must come
        // out as neutral gray rather than orange.
        let base_transmittance = [0.90, 0.60, 0.25];
        let scaled = base_transmittance.map(|channel| channel * 0.5);

        let source = image(&[scaled]);
        let result = invert(&source, &base(base_transmittance), &settings());

        let [red, green, blue] = [result.data[0], result.data[1], result.data[2]];
        assert_close(red, green);
        assert_close(green, blue);

        // And it is a real midtone, not a degenerate black or white:
        // density = -log10(0.5) = 0.30103, halved by density_max = 2.0.
        assert_close(red, 0.150_515);
    }

    #[test]
    fn denser_input_is_brighter() {
        let source = image(&[[0.5, 0.5, 0.5], [0.2, 0.2, 0.2], [0.05, 0.05, 0.05]]);
        let result = invert(&source, &base([1.0, 1.0, 1.0]), &settings());
        assert!(
            result.data[0] < result.data[3] && result.data[3] < result.data[6],
            "inversion should be monotonic: {:?}",
            result.data
        );
    }

    #[test]
    fn output_always_stays_in_range() {
        let source = image(&[
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0],
            [-1.0, -1.0, -1.0],
            [1e30, 1e30, 1e30],
        ]);
        let result = invert(&source, &base([0.5, 0.5, 0.5]), &settings());
        for channel in &result.data {
            assert!(
                (0.0..=1.0).contains(channel),
                "out of range: {channel} in {:?}",
                result.data
            );
        }
    }

    #[test]
    fn a_zero_base_does_not_produce_nan() {
        let source = image(&[[0.5, 0.5, 0.5]]);
        let result = invert(&source, &base([0.0, 0.0, 0.0]), &settings());
        assert!(result.data.iter().all(|channel| channel.is_finite()));
    }

    #[test]
    fn a_zero_density_ceiling_does_not_produce_nan() {
        let source = image(&[[0.5, 0.5, 0.5]]);
        let settings = Settings {
            density_max: Some([0.0; 3]),
            ..Settings::default()
        };
        let result = invert(&source, &base([1.0, 1.0, 1.0]), &settings);
        assert!(result.data.iter().all(|channel| channel.is_finite()));
    }

    #[test]
    fn the_default_ceiling_is_the_kodak_anchored_gray() {
        // The default is a fixed physical anchor derived from the published
        // LAD aims, not content percentiles: gray lands at the reference
        // gray, and the film's densest content approaches white.
        assert_eq!(Settings::default().density_max, Some(default_density_max()));
        assert_eq!(Settings::default().density_floor, None);
    }

    #[test]
    fn a_density_floor_lifts_the_channel_black() {
        // A print maps the density range through its own black point; a
        // non-zero floor puts thinnest content at black rather than washed.
        // The pixel is 10^-0.5: density exactly 0.5.
        let source = image(&[[0.316_227_77, 0.316_227_77, 0.316_227_77]]);
        let settings = Settings {
            density_floor: Some([0.5; 3]),
            ..settings()
        };
        let result = invert(&source, &base([1.0, 1.0, 1.0]), &settings);
        // density 0.5 above a floor of 0.5 is exactly black.
        for channel in result.data {
            assert_close(channel, 0.0);
        }
    }

    #[test]
    fn a_density_floor_below_the_content_is_a_contrast_cut() {
        // Floor 0.25, ceiling 2.0: density 0.5 renders at (0.5-0.25)/2.0.
        let source = image(&[[0.316_227_77, 0.316_227_77, 0.316_227_77]]);
        let settings = Settings {
            density_floor: Some([0.25; 3]),
            ..settings()
        };
        let result = invert(&source, &base([1.0, 1.0, 1.0]), &settings);
        assert_close(result.data[0], 0.125);
    }

    #[test]
    fn a_channel_saturates_sooner_with_a_lower_ceiling() {
        // Same density, different ceilings: the lower ceiling reaches white first.
        let source = image(&[[0.1, 0.1, 0.1]]);
        let settings = Settings {
            density_max: Some([2.0, 1.0, 0.5]),
            ..Settings::default()
        };
        let result = invert(&source, &base([1.0, 1.0, 1.0]), &settings);

        // density = -log10(0.1) = 1.0
        assert!((result.data[0] - 0.5).abs() < 1e-6, "got {:?}", result.data);
        assert!((result.data[1] - 1.0).abs() < 1e-6, "got {:?}", result.data);
        assert!((result.data[2] - 1.0).abs() < 1e-6, "got {:?}", result.data);
    }

    #[test]
    fn the_identity_crosstalk_matrix_changes_nothing() {
        let source = image(&[[0.1, 0.2, 0.3]]);
        let neutral = Settings::default();
        assert_eq!(neutral.crosstalk, IDENTITY);

        let with_identity = invert(&source, &base([1.0, 1.0, 1.0]), &neutral);
        let without_any = invert(&source, &base([1.0, 1.0, 1.0]), &Settings::default());
        assert_eq!(with_identity.data, without_any.data);
    }

    #[test]
    fn crosstalk_mixes_density_between_channels() {
        // Red density leaking entirely into green: the red positive falls, the
        // green one rises to where red was.
        let source = image(&[[0.1, 1.0, 1.0]]);
        let settings = Settings {
            crosstalk: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ..settings()
        };
        let result = invert(&source, &base([1.0, 1.0, 1.0]), &settings);

        // red density = -log10(0.1) = 1.0, green and blue are 0.
        assert!((result.data[0] - 0.0).abs() < 1e-6, "got {:?}", result.data);
        assert!((result.data[1] - 0.5).abs() < 1e-6, "got {:?}", result.data);
        assert!((result.data[2] - 0.0).abs() < 1e-6, "got {:?}", result.data);
    }

    #[test]
    fn conversion_is_deterministic() {
        let source = image(&[[0.7, 0.4, 0.2], [0.1, 0.9, 0.3]]);
        let base = base([0.8, 0.5, 0.3]);
        let first = invert(&source, &base, &settings());
        let second = invert(&source, &base, &settings());
        assert_eq!(first.data, second.data);
    }

    #[test]
    fn geometry_is_preserved() {
        let source = LinearImage::new(3, 2, vec![0.5; 18]).expect("well-formed");
        let result = invert(&source, &base([0.5, 0.5, 0.5]), &settings());
        assert_eq!((result.width, result.height), (3, 2));
        assert_eq!(result.data.len(), 18);
    }
}
