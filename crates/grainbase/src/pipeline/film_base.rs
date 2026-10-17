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
//!
//! Two detectors live here:
//!
//! * [`FilmBase::detect`] works on the whole image. It is the fallback when the
//!   film's mechanics cannot be located, and its heuristics are deliberately
//!   conservative: a bright population that covers more than a small sliver is
//!   assumed to be the light table, not film. That is enough for cropped frames
//!   and for scans where the rebate dominates, but it fails when the light table
//!   itself is a large share of the uncropped sensor.
//! * [`FilmBase::detect_in_regions`] samples only the rebate bands that
//!   [`crate::geometry`] identifies mechanically. The rebate is clear film *by
//!   the film's own structure*: the sprocket holes bound it on one side and the
//!   picture bounds it on the other. Sampling there excludes both the holes and
//!   the light table by construction, so the percentile lands on clear film even
//!   when a whole-frame detector would land in glare.
//!
//! The region path is preferred whenever geometry can find the strip. The
//! whole-image detector is kept as a trustworthy fallback for frames where the
//! band machinery does not apply.

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

/// Converts an integer pixel coordinate to `f32`.
///
/// `f32` represents integers exactly up to 2^24, which is sixteen million pixels
/// on one axis — larger than any photograph this code handles.
#[expect(
    clippy::cast_precision_loss,
    reason = "a coordinate beyond 2^24 on one axis is not a photograph"
)]
fn coordinate(value: usize) -> f32 {
    value as f32
}

/// One mechanically identified strip of clear film, in source pixel coordinates.
///
/// The geometry path returns the rebate rows above and below the picture as two
/// oriented horizontal bands. Each band is described by its centerline and a
/// half-height, so sampling it is a simple per-column y-range test. The slope
/// carries the strip's total skew (`tan(skew)`), because the rebate stays
/// parallel to the film as it tilts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FilmRegion {
    /// Row of the region's centerline at the image's horizontal center.
    pub center_y: f32,
    /// Half-height of the region, in source rows.
    pub half_height: f32,
    /// Slope `dy/dx` of the region's centerline, in source pixel units.
    ///
    /// This is `tan(total_skew)`: a small tilt shifts the centerline by a few
    /// rows across the width of the frame.
    pub slope: f32,
}

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
    ///
    /// This is the whole-image fallback. It is reliable when the rebate is the
    /// dominant bright area and the light table is only a small sliver; for
    /// uncropped scans where the light table is large, prefer
    /// [`Self::detect_in_regions`].
    #[must_use]
    pub fn detect(image: &LinearImage, percentile: f32) -> Self {
        let (pixels, _remainder) = image.data.as_chunks::<3>();
        if pixels.is_empty() {
            return Self::IDENTITY;
        }

        let step = (pixels.len() / SAMPLE_TARGET).max(1);
        let mut transmittance = [1.0f32; 3];

        for (channel, slot) in transmittance.iter_mut().enumerate() {
            let samples: Vec<f32> = pixels.iter().step_by(step).map(|p| p[channel]).collect();
            *slot = measure_channel(&samples, percentile);
        }

        Self { transmittance }
    }

    /// Estimates the base from mechanically identified rebate regions.
    ///
    /// Only pixels inside `regions` are sampled. The regions come from the film's
    /// own structure (sprocket holes on one side, picture on the other), so the
    /// light table and the holes are excluded by geometry rather than by a level
    /// heuristic. This is the path to use for uncropped scans with glare, where
    /// the whole-image detector would land in the bright population.
    ///
    /// Returns [`FilmBase::IDENTITY`] when `regions` is empty or covers no
    /// pixels; callers should fall back to [`Self::detect`] in that case.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "region rows are clamped to the image's pixel count before the \
                  index casts, and sample coordinates are photograph-sized, so \
                  f32 keeps the integer values exact"
    )]
    pub fn detect_in_regions(image: &LinearImage, regions: &[FilmRegion], percentile: f32) -> Self {
        let (pixels, _remainder) = image.data.as_chunks::<3>();
        if pixels.is_empty() || regions.is_empty() {
            return Self::IDENTITY;
        }

        let width = image.width as usize;
        let height = image.height as usize;
        let stride = region_stride(regions, width, height);
        let center_x = coordinate(width) / 2.0;
        let last_row = height.saturating_sub(1) as f32;

        let mut transmittance = [1.0f32; 3];
        for (channel, slot) in transmittance.iter_mut().enumerate() {
            let mut samples = Vec::new();
            for x in (0..width).step_by(stride) {
                let x_f = coordinate(x);
                for region in regions {
                    let y_center = region.center_y + region.slope * (x_f - center_x);
                    let y_low = (y_center - region.half_height).ceil().max(0.0);
                    let y_high = (y_center + region.half_height).floor().min(last_row);
                    if y_low > y_high {
                        continue;
                    }
                    for y in (y_low as usize..=y_high as usize).step_by(stride) {
                        samples.push(pixels[y * width + x][channel]);
                    }
                }
            }
            *slot = if samples.is_empty() {
                1.0
            } else {
                measure_channel(&samples, percentile)
            };
        }

        Self { transmittance }
    }

    /// Estimates the base from an authoritative axis-aligned rectangle.
    ///
    /// `rect` is `[x0, y0, x1, y1]` in fractions of the image width and height.
    /// It is intended for a manual override when the operator knows exactly
    /// where the rebate sits: a rectangle over the rebate beats every heuristic
    /// on glare-heavy rigs.
    ///
    /// This uses the same film-level machinery as [`Self::detect_in_regions`];
    /// the only difference is that the sample area is bounded in both axes.
    #[must_use]
    pub fn detect_in_rect(image: &LinearImage, rect: [f32; 4], percentile: f32) -> Self {
        let (pixels, _remainder) = image.data.as_chunks::<3>();
        if pixels.is_empty() {
            return Self::IDENTITY;
        }

        let width = image.width as usize;
        let height = image.height as usize;

        let x0f = (rect[0] * coordinate(width)).clamp(0.0, coordinate(width));
        let y0f = (rect[1] * coordinate(height)).clamp(0.0, coordinate(height));
        let x1f = (rect[2] * coordinate(width)).clamp(x0f, coordinate(width));
        let y1f = (rect[3] * coordinate(height)).clamp(y0f, coordinate(height));

        let x0 = floor_to_usize(x0f);
        let y0 = floor_to_usize(y0f);
        let x1 = ceil_to_usize(x1f).min(width);
        let y1 = ceil_to_usize(y1f).min(height);
        if x1 <= x0 || y1 <= y0 {
            return Self::IDENTITY;
        }

        let area = (x1 - x0) * (y1 - y0);
        let stride = (area / SAMPLE_TARGET).max(1).isqrt();
        let stride = stride.max(1);

        let mut transmittance = [1.0f32; 3];
        for (channel, slot) in transmittance.iter_mut().enumerate() {
            let mut samples = Vec::new();
            for y in (y0..y1).step_by(stride) {
                for x in (x0..x1).step_by(stride) {
                    samples.push(pixels[y * width + x][channel]);
                }
            }
            *slot = measure_channel(&samples, percentile);
        }

        Self { transmittance }
    }
}

/// Chooses a stride so that the region sample is roughly [`SAMPLE_TARGET`]
/// pixels, without oversampling small regions.
fn region_stride(regions: &[FilmRegion], width: usize, height: usize) -> usize {
    let area: f32 = regions
        .iter()
        .map(|region| {
            let h = region.half_height * 2.0;
            let vertical = h.min(coordinate(height));
            vertical * coordinate(width)
        })
        .sum();
    if area <= 0.0 {
        return 1;
    }
    let stride = (area / coordinate(SAMPLE_TARGET)).sqrt().max(1.0);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "stride is clamped to >= 1.0 before casting, so the value is a small positive integer"
    )]
    let stride = stride as usize;
    stride.max(1)
}

/// Measures one channel from a collected sample set.
fn measure_channel(samples: &[f32], percentile: f32) -> f32 {
    let mut sorted: Vec<f32> = samples.to_vec();
    sorted.sort_unstable_by(f32::total_cmp);
    film_level(&sorted, percentile).max(f32::MIN_POSITIVE)
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

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a rectangle coordinate: clamped to [0, dimension] before truncation"
)]
fn floor_to_usize(value: f32) -> usize {
    value.floor() as usize
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a rectangle coordinate: clamped to [0, dimension] before truncation"
)]
fn ceil_to_usize(value: f32) -> usize {
    value.ceil() as usize
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

    #[test]
    fn region_sampling_recovers_the_rebate_level() {
        // A 200x100 strip: bright light table margins, hole bands, a thin rebate
        // at the base level, then dark picture. The whole-image detector would
        // see a large bright population and pick the light table; the region
        // path must ignore everything outside the rebate bands.
        let width = 200u32;
        let height = 100u32;
        let base_level = [0.32f32, 0.12, 0.05];
        let hole = [0.9f32; 3];
        let light_table = [1.0f32; 3];
        let picture = [0.03f32; 3];

        let mut data = vec![0.0f32; width as usize * height as usize * 3];
        for y in 0..height {
            let row_value = if y < 8 {
                light_table
            } else if y < 16 {
                hole
            } else if y < 22 {
                base_level
            } else if y < 78 {
                picture
            } else if y < 84 {
                base_level
            } else if y < 92 {
                hole
            } else {
                light_table
            };
            for x in 0..width {
                let idx = (y as usize * width as usize + x as usize) * 3;
                data[idx..idx + 3].copy_from_slice(&row_value);
            }
        }

        let image = LinearImage::new(width, height, data).expect("well-formed");

        // Whole-image detector fails: the light table is most of the frame.
        let whole = FilmBase::detect(&image, DEFAULT_PERCENTILE);
        assert!(
            (whole.transmittance[0] - 1.0).abs() < 1e-3,
            "sanity: whole-image detector should land in the light table, got {whole:?}"
        );

        // Tight regions over the flat rebate rows only: rows 16..21 and 78..83.
        let regions = [
            FilmRegion {
                center_y: 18.5,
                half_height: 2.5,
                slope: 0.0,
            },
            FilmRegion {
                center_y: 80.5,
                half_height: 2.5,
                slope: 0.0,
            },
        ];
        let base = FilmBase::detect_in_regions(&image, &regions, DEFAULT_PERCENTILE);
        for (channel, &expected) in base_level.iter().enumerate() {
            assert!(
                (base.transmittance[channel] - expected).abs() < 1e-3,
                "channel {channel}: expected {expected}, got {base:?}"
            );
        }
    }

    #[test]
    #[expect(
        clippy::approx_constant,
        reason = "0.318 is the measured DSC02063 red base, not pi"
    )]
    fn region_sampling_excludes_bright_ends_and_soft_glare() {
        // Mirrors the DSC02063 failure shape: the rebate is clear film at the
        // known base, but the top and bottom of the strip are bright light table
        // and the transition into the rebate is a soft glare ramp rather than a
        // step. The region path must still recover the rebate level.
        #[expect(
            clippy::cast_precision_loss,
            reason = "test grid coordinates are small, integer-exact in f32"
        )]
        fn fraction(value: u32, total: u32) -> f32 {
            value as f32 / total as f32
        }

        let width = 400u32;
        let height = 300u32;
        let base_level = [0.318f32, 0.123, 0.050];

        let mut data = vec![0.0f32; width as usize * height as usize * 3];
        for y in 0..height {
            let row_frac = fraction(y, height);
            let row_value = if row_frac < 0.12 {
                // Top light table.
                [1.0f32; 3]
            } else if row_frac < 0.18 {
                // Soft glare ramp down into the rebate.
                let t = (row_frac - 0.12) / 0.06;
                base_level.map(|channel| channel + (1.0 - channel) * (1.0 - t))
            } else if row_frac < 0.22 {
                // Top rebate.
                base_level
            } else if row_frac < 0.78 {
                // Picture.
                [0.02f32; 3]
            } else if row_frac < 0.82 {
                // Bottom rebate.
                base_level
            } else if row_frac < 0.88 {
                // Soft glare ramp down to bottom light table.
                let t = (row_frac - 0.82) / 0.06;
                base_level.map(|channel| channel + (1.0 - channel) * t)
            } else {
                // Bottom light table.
                [1.0f32; 3]
            };
            for x in 0..width {
                let idx = (y as usize * width as usize + x as usize) * 3;
                data[idx..idx + 3].copy_from_slice(&row_value);
            }
        }

        let image = LinearImage::new(width, height, data).expect("well-formed");

        // Sample only the flat rebate core, not the surrounding glare ramps.
        let regions = [
            FilmRegion {
                center_y: 60.0,
                half_height: 5.0,
                slope: 0.0,
            },
            FilmRegion {
                center_y: 240.0,
                half_height: 5.0,
                slope: 0.0,
            },
        ];
        let base = FilmBase::detect_in_regions(&image, &regions, DEFAULT_PERCENTILE);
        for (channel, &expected) in base_level.iter().enumerate() {
            assert!(
                (base.transmittance[channel] - expected).abs() < 0.02,
                "channel {channel}: expected {expected}, got {base:?}"
            );
        }
    }

    #[test]
    fn manual_rect_override_samples_only_the_rect() {
        let mut data = vec![0.0f32; 100 * 100 * 3];
        for y in 0..100 {
            for x in 0..100 {
                let idx = (y * 100 + x) * 3;
                // A bright margin and a darker interior; the rect is over the
                // interior only.
                let value = if !(20..80).contains(&x) || !(20..80).contains(&y) {
                    [0.9f32; 3]
                } else {
                    [0.3f32; 3]
                };
                data[idx..idx + 3].copy_from_slice(&value);
            }
        }
        let image = LinearImage::new(100, 100, data).expect("well-formed");
        let base = FilmBase::detect_in_rect(&image, [0.25, 0.25, 0.75, 0.75], DEFAULT_PERCENTILE);
        assert!((base.transmittance[0] - 0.3).abs() < 1e-3, "got {base:?}");
    }

    #[test]
    fn empty_regions_fall_back_to_identity() {
        let image = LinearImage::new(10, 10, vec![0.5f32; 300]).expect("well-formed");
        assert_eq!(
            FilmBase::detect_in_regions(&image, &[], DEFAULT_PERCENTILE),
            FilmBase::IDENTITY
        );
    }

    #[test]
    fn region_sampling_honors_tilt() {
        // A narrow band tilted by slope 0.1. A horizontal band misses it; the
        // correct slope samples the right rows.
        let width = 200u32;
        let height = 200u32;
        let slope = 0.1f32;
        let center_y_at_mid = 100.0f32;
        let half_height = 5.0f32;
        let base_level = [0.4f32; 3];

        let mut data = vec![0.0f32; width as usize * height as usize * 3];
        #[expect(
            clippy::cast_precision_loss,
            reason = "test grid coordinates are small, integer-exact in f32"
        )]
        let to_f32 = |v: u32| v as f32;
        for y in 0..height {
            for x in 0..width {
                let x_f = to_f32(x);
                let y_center = center_y_at_mid + slope * (x_f - 100.0);
                let in_band = (to_f32(y) - y_center).abs() <= half_height;
                let idx = (y as usize * width as usize + x as usize) * 3;
                data[idx..idx + 3].copy_from_slice(if in_band {
                    &base_level
                } else {
                    &[0.02f32; 3]
                });
            }
        }

        let image = LinearImage::new(width, height, data).expect("well-formed");

        // A horizontal region placed well away from the tilted band should not
        // pick it up.
        let flat = FilmRegion {
            center_y: 50.0,
            half_height,
            slope: 0.0,
        };
        let flat_base = FilmBase::detect_in_regions(&image, &[flat], DEFAULT_PERCENTILE);
        assert!(
            flat_base.transmittance[0] < 0.1,
            "a flat region should miss the tilted band, got {flat_base:?}"
        );

        let tilted = FilmRegion {
            center_y: center_y_at_mid,
            half_height,
            slope,
        };
        let tilted_base = FilmBase::detect_in_regions(&image, &[tilted], DEFAULT_PERCENTILE);
        assert!(
            (tilted_base.transmittance[0] - 0.4).abs() < 1e-3,
            "a tilted region should recover the band, got {tilted_base:?}"
        );
    }
}
