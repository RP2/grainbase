//! Finding the film base: the unexposed, clear part of the film.
//!
//! The base carries the orange mask and no image, so measuring it is what makes
//! the conversion neutral instead of orange. In a scan it is the *brightest*
//! thing the film itself shows: clear film transmits light, exposed film does
//! not.
//!
//! The complication is that not everything bright in a scan is film. Sprocket
//! holes are gaps, and the light table or the holder can show around the frame.
//! Those are brighter than the orange-masked base, because no mask is in the
//! way. Measure them and you have measured the light source, which shows up as a
//! color cast no amount of grading will remove, because the mask is still
//! there, unneutralised.

use crate::image::LinearImage;

use super::percentile;

/// Percentile of each channel taken as the base, by default.
///
/// Used when the frame gives no reason to think anything bright is not film.
pub const DEFAULT_PERCENTILE: f32 = 99.5;

/// Ratio between neighboring samples in the sorted upper tail that marks the
/// boundary between two populations.
///
/// A sprocket hole against clear film is a step, not a slope.
const CLIFF_RATIO: f32 = 1.8;

/// Share of the frame, from the top, that is searched for that boundary.
const CLIFF_WINDOW: f32 = 0.05;

/// A bright population covering less of the frame than this is taken to lie
/// outside the film.
///
/// Every filmic reason for a bright area — the rebate, the clear leader, an
/// unexposed frame — is a substantial part of the frame. A sliver that is
/// brighter than everything else is a hole or the light table.
const MIN_FILM_SHARE: f32 = 0.05;

/// Samples per channel the estimate is taken from.
///
/// The base is a whole-frame property, so a strided sample answers it as well as
/// every pixel would, and keeps the cost flat as scans get larger.
const SAMPLE_TARGET: usize = 1 << 18;

/// The unexposed film base, measured per channel in the camera's linear space.
///
/// Values are in the same units as [`LinearImage`] samples: the linear
/// transmittance of the clear film, orange mask included. Dividing an image by
/// this neutralises the mask.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FilmBase {
    /// Per-channel base transmittance, in red, green, blue order.
    pub transmittance: [f32; 3],
}

impl FilmBase {
    /// A base that changes nothing, for an image with no pixels.
    pub const IDENTITY: Self = Self {
        transmittance: [1.0, 1.0, 1.0],
    };

    /// Estimates the base from a single image.
    ///
    /// Each channel is measured independently, so a base with a color cast (an
    /// orange mask) is measured as such rather than averaged into neutrality.
    ///
    /// An image with no pixels has no base to measure; this returns
    /// [`FilmBase::IDENTITY`], which is never consulted because there are no
    /// pixels to neutralise.
    ///
    /// The returned base is never zero, so callers can divide by it safely.
    #[must_use]
    pub fn detect(image: &LinearImage, percentile: f32) -> Self {
        let (pixels, _remainder) = image.data.as_chunks::<3>();
        if pixels.is_empty() {
            return Self::IDENTITY;
        }

        let step = (pixels.len() / SAMPLE_TARGET).max(1);
        let mut transmittance = [1.0f32; 3];

        for (channel, slot) in transmittance.iter_mut().enumerate() {
            let mut samples: Vec<f32> = pixels.iter().step_by(step).map(|p| p[channel]).collect();
            samples.sort_unstable_by(f32::total_cmp);
            *slot = film_level(&samples, percentile).max(f32::MIN_POSITIVE);
        }

        Self { transmittance }
    }
}

/// The brightest value that represents film rather than something outside it.
///
/// `samples` must be sorted ascending.
///
/// ponytail: a heuristic with tuned constants, chosen over edge detection for
/// now. Cropping to the film frame (U7) removes non-film areas outright and is
/// the real fix; this keeps uncropped scans usable until then, and stays useful
/// afterwards as a guard against dust and light leaks.
fn film_level(samples: &[f32], percentile: f32) -> f32 {
    let fallback = percentile::value_at_sorted(samples, percentile).unwrap_or(1.0);
    match bright_cliff(samples) {
        // A brighter population that is only a sliver of the frame is not film.
        Some(cliff) if cliff.above_share < MIN_FILM_SHARE => samples[cliff.below],
        _ => fallback,
    }
}

/// Where a separate, brighter population begins.
struct Cliff {
    /// Index of the last sample below the gap: the top of the lower population.
    below: usize,
    /// Share of all samples at or above the gap.
    above_share: f32,
}

/// Finds the steepest segment of the sorted upper tail.
///
/// Compares values a fixed *rank distance* apart rather than neighbors. A sharp
/// transition squeezed into a narrow slice of the distribution is spread across
/// many sorted samples, so each neighboring step is tiny and a neighbor test
/// cannot see it. On a real scan the film/non-film step covered 0.05% of samples:
/// a 3.3x change in value, a 1.009x change between neighbors.
fn bright_cliff(samples: &[f32]) -> Option<Cliff> {
    let len = samples.len();
    let last = len.checked_sub(1)?;
    let span = (len / 1000).max(1);
    let window_start = percentile::index_of_rank(len, 100.0 * (1.0 - CLIFF_WINDOW))?;
    let search_end = last.checked_sub(span)?;

    let mut best: Option<(usize, f32)> = None;
    for index in window_start..=search_end {
        let (lower, upper) = (samples[index], samples[index + span]);
        if lower <= 0.0 {
            continue;
        }
        let ratio = upper / lower;
        if ratio >= CLIFF_RATIO && best.is_none_or(|(_, previous)| ratio > previous) {
            best = Some((index, ratio));
        }
    }

    let (below, _ratio) = best?;
    let above = len - 1 - below;
    #[expect(
        clippy::cast_precision_loss,
        reason = "the share only has to be accurate enough to compare against a \
                  5% threshold, so full f32 precision on the count is irrelevant"
    )]
    let above_share = above as f32 / len as f32;
    Some(Cliff { below, above_share })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An image from a flat list of pixels.
    fn image(pixels: &[[f32; 3]]) -> LinearImage {
        let data = pixels.iter().flatten().copied().collect();
        let count = u32::try_from(pixels.len()).expect("test image is small");
        LinearImage::new(count, 1, data).expect("test image is well-formed")
    }

    #[test]
    fn detects_the_brightest_channel_level() {
        // Top 10% of the frame is clear base at 0.8. That is a substantial share,
        // so it is film rather than a hole, and the percentile should find it.
        let mut pixels = vec![[0.2f32; 3]; 900];
        pixels.extend(vec![[0.8f32; 3]; 100]);
        let base = FilmBase::detect(&image(&pixels), DEFAULT_PERCENTILE);
        assert!((base.transmittance[0] - 0.8).abs() < 1e-6, "got {base:?}");
    }

    #[test]
    fn a_sliver_of_bright_pixels_is_not_film() {
        // The real failure: sprocket holes and light table are far brighter than
        // the orange-masked base but cover well under 5% of the frame.
        let mut pixels = vec![[0.12f32; 3]; 9940];
        pixels.extend(vec![[0.62f32; 3]; 60]);
        let base = FilmBase::detect(&image(&pixels), DEFAULT_PERCENTILE);
        assert!(
            (base.transmittance[0] - 0.12).abs() < 1e-6,
            "the light table was measured instead of the film: {base:?}"
        );
    }

    #[test]
    fn one_bright_pixel_does_not_set_the_base() {
        let mut pixels = vec![[0.3f32; 3]; 999];
        pixels.push([0.95, 0.95, 0.95]);
        let base = FilmBase::detect(&image(&pixels), DEFAULT_PERCENTILE);
        assert!(
            (base.transmittance[0] - 0.3).abs() < 1e-6,
            "an outlier moved the base: {base:?}"
        );
    }

    #[test]
    fn measures_each_channel_separately() {
        // A base with a strong orange cast: red high, blue low.
        let pixels = vec![[0.9f32, 0.6, 0.25]; 100];
        let base = FilmBase::detect(&image(&pixels), DEFAULT_PERCENTILE);
        assert!((base.transmittance[0] - 0.9).abs() < 1e-6);
        assert!((base.transmittance[1] - 0.6).abs() < 1e-6);
        assert!((base.transmittance[2] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn a_gradient_with_no_cliff_falls_back_to_the_percentile() {
        // A smooth ramp has no second population, so nothing should be excluded.
        let pixels: Vec<[f32; 3]> = (0..1000u16).map(|i| [f32::from(i) / 1000.0; 3]).collect();
        let base = FilmBase::detect(&image(&pixels), DEFAULT_PERCENTILE);
        assert!((base.transmittance[0] - 0.995).abs() < 1e-3, "got {base:?}");
    }

    #[test]
    fn an_empty_image_yields_the_identity_base() {
        let empty = LinearImage::new(0, 0, Vec::new()).expect("empty image is valid");
        assert_eq!(
            FilmBase::detect(&empty, DEFAULT_PERCENTILE),
            FilmBase::IDENTITY
        );
    }

    #[test]
    fn a_black_frame_never_yields_a_zero_base() {
        let black = image(&[[0.0f32; 3]; 10]);
        let base = FilmBase::detect(&black, DEFAULT_PERCENTILE);
        assert!(
            base.transmittance.iter().all(|&t| t > 0.0),
            "a zero base would divide by zero downstream: {base:?}"
        );
    }

    #[test]
    fn percentile_is_clamped_rather_than_panicking() {
        let pixels = vec![[0.5f32; 3]; 10];
        for percentile in [-10.0, 0.0, 100.0, 250.0] {
            let base = FilmBase::detect(&image(&pixels), percentile);
            assert!((base.transmittance[0] - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn detection_is_deterministic() {
        let mut pixels = vec![[0.12f32; 3]; 9940];
        pixels.extend(vec![[0.62f32; 3]; 60]);
        let image = image(&pixels);
        assert_eq!(
            FilmBase::detect(&image, DEFAULT_PERCENTILE),
            FilmBase::detect(&image, DEFAULT_PERCENTILE)
        );
    }
}
