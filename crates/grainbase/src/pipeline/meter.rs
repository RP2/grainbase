//! Per-frame metering: the frame's integral color, against its own neutral.
//!
//! A camera scan is metered the way a photofinishing printer is. The frame's
//! *integral* density — the large-area average, which neither a dust mote nor a
//! patch of grain can move — carries the frame's own color: a film cast or a
//! scanning imbalance shifts it off neutral, and a neutral scene integrates flat
//! because dividing by the base already removed the mask's per-channel shape.
//! This module measures that integral and pulls it back toward the frame's own
//! neutral.
//!
//! One published caution comes with it. A scene that breaks the *integrating to
//! gray* assumption — one that is genuinely a single color — would be
//! "corrected" straight into neutrality by a printer that trusted its integral.
//! The photofinishing patent literature is explicit that the response is
//! coupled to the detected channel imbalance: full correction for a balanced
//! negative, reduced or zero correction for a *subject failure* scene
//! (US 3,653,759; US 4,159,174). The correction here is weighted the same way.
//! The measurement area is never gated; the response is.
//!
//! The **printer point** — the absolute level, not just the color — is *not*
//! here. Kodak publishes the LAD aims for a camera negative (see
//! [`LAD_CAMERA_NEGATIVE`]), but they are Status-M densities measured on film,
//! and this scan's base-relative density is on the camera's own scale: measured
//! on the maintainer's fixtures the two differ by a factor that is a property of
//! the rig, not of the film. Applying the aims to scan density directly was
//! measured and refused — see the commit that added this note. The scale between
//! the two is exactly the per-stock calibration the profile carries, so the
//! level belongs to a profile and not to this generic path.

use crate::image::LinearImage;

use super::{FilmBase, MAX_DENSITY};

/// Kodak's published LAD aims for a **camera** color negative: the Status-M
/// densities above base of a correctly exposed 16% `Munsell` N4.5 patch, in
/// red/green/blue order.
///
/// Cited from H-61a. These are the camera-negative aims; they are deliberately
/// *not* the duplicate-negative aims that also appear in the H-387 cross-check
/// table, and they are the numbers a profile's per-channel ranges are anchored
/// to.
///
/// They are **not** applied to scan density anywhere in the generic path. A
/// camera is not a densitometer: the scan's base-relative density is compressed
/// and spectrally shifted relative to Status-M by an unknown rig factor, and
/// comparing the two directly was measured to make every fixture worse. They
/// live here as the published reference a fitted profile works against.
pub const LAD_CAMERA_NEGATIVE: [f32; 3] = [0.80, 1.20, 1.60];

/// Side of one metering block, in source pixels.
///
/// The integral is an average, so it is naturally robust to grain; the block is
/// what makes it robust to *dust* and to a single hot pixel, which a per-pixel
/// average over a small area would not be.
const BLOCK: usize = 16;

/// Share trimmed from each end of the block densities before the mean.
///
/// The analysis region is inside the exposure, so a block should read picture.
/// A region that strays onto the rebate, or a block on a scratch, would drag a
/// plain mean; trimming the extremes keeps the integral honest without letting
/// an outlier set it.
const TRIM: f32 = 0.10;

/// Fewest blocks for the integral to mean anything.
const MIN_BLOCKS: usize = 16;

/// Channel spread, in density, at or below which the frame is balanced: the
/// channels sit as close to neutral as a real scene gets, so the integral is
/// trusted fully.
///
/// Measured on the maintainer's fixtures: a neutral scene integrates flat to
/// within about 0.05.
const BALANCED_SPREAD: f32 = 0.15;

/// Channel spread at or above which the frame is a subject failure: one channel
/// is far denser than the others, which is a scene that is genuinely one color,
/// so the integral is refused rather than neutralised into the scene's opposite.
const FAILURE_SPREAD: f32 = 1.20;

/// One frame's metered color.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metering {
    /// Measured integral density per channel, in log10 units above base.
    pub integral: [f32; 3],
    /// How far each channel's density moves to bring the integral back to the
    /// frame's own neutral, already weighted by [`Metering::confidence`].
    /// Applied to the density axis positive-up.
    pub correction: [f32; 3],
    /// How fully the integral is trusted: `1.0` for a balanced frame, falling to
    /// `0.0` as the channel imbalance grows into subject-failure territory.
    pub confidence: f32,
}

/// A pixel coordinate as a float.
///
/// Exact for anything that can be held in memory: `f32` represents integers
/// exactly up to 2^24, which is sixteen million pixels along one axis.
#[expect(
    clippy::cast_precision_loss,
    reason = "a coordinate beyond 2^24 on one axis is not a photograph"
)]
fn coordinate(value: usize) -> f32 {
    value as f32
}

/// Meters one frame's integral color.
///
/// `rect` is `[x0, y0, x1, y1]` in fractions of the image width and height — the
/// analysis region, which the caller derives from the frame's geometry, exactly
/// as [`FilmBase::detect_in_rect`] takes its rectangle. Returns `None` when the
/// region is too small to hold a measurement; callers then render as if no
/// metering had been asked for.
#[must_use]
pub fn meter(image: &LinearImage, base: &FilmBase, rect: [f32; 4]) -> Option<Metering> {
    let mut blocks = block_densities(image, base, rect);
    if blocks.len() < MIN_BLOCKS {
        return None;
    }

    let integral = [0, 1, 2].map(|channel| {
        let mut values: Vec<f32> = blocks.iter().map(|block| block[channel]).collect();
        trimmed_mean(&mut values)
    });
    blocks.clear();

    // The frame's own neutral: what a flat integral at this level would be. The
    // deviation from it is the frame's color, and a cast is a deviation the film
    // or the scan put there rather than the scene.
    let level = (integral[0] + integral[1] + integral[2]) / 3.0;
    let deviation = integral.map(|value| value - level);
    let spread = deviation.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - deviation.iter().copied().fold(f32::INFINITY, f32::min);
    let confidence =
        ((FAILURE_SPREAD - spread) / (FAILURE_SPREAD - BALANCED_SPREAD)).clamp(0.0, 1.0);

    Some(Metering {
        integral,
        correction: deviation.map(|value| -confidence * value),
        confidence,
    })
}

/// One density per block, from each block's mean transmittance.
///
/// Integral density is the density of the *average* transmittance, not the
/// average of the per-pixel densities: taking the mean first is what makes a
/// grain-free measurement of the light the film actually passed.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "block bounds are clamped to the image's pixel count before the \
              index casts, and source coordinates are photograph-sized"
)]
pub(crate) fn block_densities(
    image: &LinearImage,
    base: &FilmBase,
    rect: [f32; 4],
) -> Vec<[f32; 3]> {
    let (pixels, _remainder) = image.data.as_chunks::<3>();
    let width = image.width as usize;
    let height = image.height as usize;
    if pixels.is_empty() || width == 0 || height == 0 {
        return Vec::new();
    }

    let x0 = (rect[0] * coordinate(width))
        .clamp(0.0, coordinate(width))
        .floor() as usize;
    let y0 = (rect[1] * coordinate(height))
        .clamp(0.0, coordinate(height))
        .floor() as usize;
    let x1 = (rect[2] * coordinate(width))
        .clamp(0.0, coordinate(width))
        .ceil()
        .min(coordinate(width)) as usize;
    let y1 = (rect[3] * coordinate(height))
        .clamp(0.0, coordinate(height))
        .ceil()
        .min(coordinate(height)) as usize;
    if x1 <= x0 || y1 <= y0 {
        return Vec::new();
    }

    let mut blocks = Vec::new();
    let mut y = y0;
    while y < y1 {
        let row_end = (y + BLOCK).min(y1);
        let mut x = x0;
        while x < x1 {
            let column_end = (x + BLOCK).min(x1);
            let mut sums = [0.0f32; 3];
            let mut count = 0usize;
            for row in y..row_end {
                for column in x..column_end {
                    let pixel = pixels[row * width + column];
                    for (sum, channel) in sums.iter_mut().zip(pixel) {
                        *sum += channel;
                    }
                    count += 1;
                }
            }
            if count > 0 {
                let count = coordinate(count);
                let mean = sums.map(|sum| sum / count);
                blocks.push(
                    [0, 1, 2].map(|channel| density_of(mean[channel], base.transmittance[channel])),
                );
            }
            x = column_end;
        }
        y = row_end;
    }
    blocks
}

/// Optical density of one sample relative to the film base.
///
/// The same arithmetic the conversion core uses, so the meter measures the same
/// quantity the render will place. Clamped at the same ceiling for the same
/// reason: a fully opaque sample must yield a finite density.
fn density_of(value: f32, base: f32) -> f32 {
    let base = base.max(f32::MIN_POSITIVE);
    let transmittance = (value / base).clamp(0.0, 1.0);
    (-transmittance.log10()).min(MAX_DENSITY)
}

/// The mean of `values` with the extreme [`TRIM`] dropped from each end.
///
/// `values` is sorted in place.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the drop count is a small fraction of a block count and fits usize"
)]
fn trimmed_mean(values: &mut [f32]) -> f32 {
    values.sort_unstable_by(f32::total_cmp);
    let cap = values.len().saturating_sub(1) / 2;
    let drop = ((coordinate(values.len()) * TRIM).floor() as usize).min(cap);
    let kept = &values[drop..values.len() - drop];
    let sum: f32 = kept.iter().sum();
    sum / coordinate(kept.len().max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An image of one flat transmittance, at a size that holds many blocks.
    fn flat(transmittance: [f32; 3]) -> LinearImage {
        let pixels = 128u32 * 128;
        let data = (0..pixels).flat_map(|_| transmittance).collect();
        LinearImage::new(128, 128, data).expect("well-formed")
    }

    fn finished(base: [f32; 3]) -> FilmBase {
        FilmBase {
            transmittance: base,
        }
    }

    /// A flat frame whose transmittance is 10^(-density) times the base, per
    /// channel.
    fn frame_at(density: [f32; 3], base: [f32; 3]) -> LinearImage {
        let pixels = 128u32 * 128;
        let mut data = Vec::with_capacity(pixels as usize * 3);
        for _ in 0..pixels {
            for channel in 0..3 {
                data.push(base[channel] * 10f32.powf(-density[channel]));
            }
        }
        LinearImage::new(128, 128, data).expect("well-formed")
    }

    #[test]
    fn a_neutral_frame_needs_no_correction() {
        let base = [0.3, 0.12, 0.05];
        let image = frame_at([0.6, 0.6, 0.6], base);
        let metered = meter(&image, &finished(base), [0.0, 0.0, 1.0, 1.0]).expect("blocks");

        for channel in 0..3 {
            assert!(
                metered.correction[channel].abs() < 1e-3,
                "a neutral frame should not move: {metered:?}"
            );
        }
        assert!(metered.confidence > 0.99, "a balanced frame is trusted");
    }

    #[test]
    fn a_cast_is_pulled_toward_neutral() {
        // Blue denser than the rest: a cast. The correction lowers blue and
        // raises the others, so the integral comes back flat.
        let base = [0.3, 0.12, 0.05];
        let image = frame_at([0.6, 0.6, 0.8], base);
        let metered = meter(&image, &finished(base), [0.0, 0.0, 1.0, 1.0]).expect("blocks");

        assert!(
            metered.correction[2] < 0.0,
            "the densest channel comes down: {metered:?}"
        );
        assert!(
            metered.correction[0] > 0.0 && metered.correction[1] > 0.0,
            "the thinner channels come up: {metered:?}"
        );
        assert!(
            metered.confidence > 0.5,
            "a 0.2 spread is a cast, not a scene"
        );
    }

    #[test]
    fn a_strongly_colored_frame_lowers_confidence() {
        // The subject-failure case: one channel far denser than the rest. The
        // integral is not trusted, so the correction is reduced toward zero.
        let base = [0.3, 0.12, 0.05];
        let image = frame_at([0.2, 0.6, 1.6], base);
        let metered = meter(&image, &finished(base), [0.0, 0.0, 1.0, 1.0]).expect("blocks");

        assert!(
            metered.confidence < 0.01,
            "a 1.4-channel imbalance is a subject failure: {metered:?}"
        );
        for channel in 0..3 {
            assert!(
                metered.correction[channel].abs() < 1e-3,
                "a refused frame should not move: {metered:?}"
            );
        }
    }

    #[test]
    fn a_region_off_the_image_yields_nothing() {
        let base = [0.3, 0.12, 0.05];
        let image = flat([0.3, 0.12, 0.05]);
        assert_eq!(
            meter(&image, &finished(base), [0.9, 0.9, 0.91, 0.91]),
            None,
            "a region too small to hold blocks is not a measurement"
        );
    }

    #[test]
    fn a_small_image_yields_nothing() {
        let image = LinearImage::new(4, 4, vec![0.1; 48]).expect("well-formed");
        assert_eq!(
            meter(&image, &finished([0.3, 0.12, 0.05]), [0.0, 0.0, 1.0, 1.0]),
            None
        );
    }

    #[test]
    fn one_dust_block_does_not_move_the_integral() {
        // A brighter-than-base block (dust, a light leak) reads zero density.
        // One block in sixty-four is inside the trim; the integral does not move.
        let base = [0.3, 0.12, 0.05];
        let mut image = frame_at([0.6, 0.6, 0.8], base);
        let (pixels, _remainder) = image.data.as_chunks_mut::<3>();
        for row in 0..BLOCK {
            for column in 0..BLOCK {
                pixels[row * 128 + column] = [1.0, 1.0, 1.0];
            }
        }

        let clean = meter(&image, &finished(base), [0.0, 0.0, 1.0, 1.0]).expect("blocks");
        for channel in 0..3 {
            assert!(
                (clean.integral[channel] - [0.6, 0.6, 0.8][channel]).abs() < 0.05,
                "dust moved the integral: {clean:?}"
            );
        }
    }

    #[test]
    fn metering_is_deterministic() {
        let base = [0.3, 0.12, 0.05];
        let image = frame_at([0.5, 0.6, 0.8], base);
        let first = meter(&image, &finished(base), [0.0, 0.0, 1.0, 1.0]);
        let second = meter(&image, &finished(base), [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(first, second);
    }
}
