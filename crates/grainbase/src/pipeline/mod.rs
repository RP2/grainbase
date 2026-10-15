//! The pure conversion core: film-base neutralisation and inversion.
//!
//! No I/O, no per-frame state: the same input always produces the same output,
//! so results can be golden-tested. This is the primary seam for conversion
//! quality.
//!
//! A colour negative is a positive printed through a mask. Its film base is the
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
/// the capture. This is a fixed, deterministic starting point; the grade and
/// look units are where per-frame and per-stock behaviour belongs.
pub const DEFAULT_DENSITY_MAX: f32 = 2.0;

/// Tuning for the conversion core.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    /// Density treated as the positive's white point, in log10 units.
    ///
    /// Lower values increase contrast and clip highlights sooner.
    pub density_max: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            density_max: DEFAULT_DENSITY_MAX,
        }
    }
}

/// Neutralises the film base and inverts a negative into a positive.
///
/// Each channel is divided by the base, converted to optical density, and
/// scaled so the base becomes black and [`Settings::density_max`] becomes
/// white. Because the base is divided out per channel, an orange mask becomes a
/// neutral black instead of an orange cast.
///
/// Output samples are in `0.0..=1.0`. The transform is pure and deterministic.
#[must_use]
pub fn invert(image: &LinearImage, base: &FilmBase, settings: &Settings) -> LinearImage {
    // A zero or negative setting would divide by zero or invert the contrast.
    let density_max = settings.density_max.max(f32::MIN_POSITIVE);

    image.map_rgb(|[red, green, blue]| {
        [
            invert_channel(red, base.transmittance[0], density_max),
            invert_channel(green, base.transmittance[1], density_max),
            invert_channel(blue, base.transmittance[2], density_max),
        ]
    })
}

/// One channel of [`invert`].
fn invert_channel(value: f32, base: f32, density_max: f32) -> f32 {
    // Guarded rather than trusted: a zero base would make the whole frame
    // NaN, and one bad frame must not poison a batch.
    let base = base.max(f32::MIN_POSITIVE);

    // Clamped at 1.0 so that a pixel brighter than the base (dust, a light
    // leak, a specular highlight) reads as zero density instead of turning into
    // a negative density and then a negative positive.
    let transmittance = (value / base).clamp(0.0, 1.0);

    // Zero transmittance is infinite density, which clamps to white below.
    let density = -transmittance.log10();

    (density / density_max).clamp(0.0, 1.0)
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

    fn settings() -> Settings {
        Settings::default()
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
    fn a_level_one_density_below_the_base_becomes_mid_grey() {
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
        // out as neutral grey rather than orange.
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
        let settings = Settings { density_max: 0.0 };
        let result = invert(&source, &base([1.0, 1.0, 1.0]), &settings);
        assert!(result.data.iter().all(|channel| channel.is_finite()));
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
