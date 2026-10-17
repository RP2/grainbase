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
//! * [`FilmBase::detect_between_sprockets`] samples the film *between the
//!   perforations*, on the punch band's own rows. That film is unexposed by
//!   construction — the camera's gate never reaches the strip's edge — so it is
//!   clear film by the film's geometry alone, and the holes, the exposure and
//!   the light table are all excluded by *where* the samples are taken.
//!
//! The whole-image detector is not a conversion source: on the maintainer's
//! scans it lands in the hole population for red and renders the frame pink. It
//! is kept as a utility for the geometry pass, which needs a clear-film level
//! before it has a frame.

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
    /// [`Self::detect_between_sprockets`].
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

    /// Estimates the base from the film between the sprocket holes.
    ///
    /// The punch band runs across the strip at the film's edge, and the holes
    /// are punched out of it. The film *between* one hole and the next is
    /// unexposed by construction: the camera's gate never reaches the strip's
    /// edge, so nothing but the mask has ever been between that film and the
    /// light. The band's own rows, at the columns that are not holes, are
    /// therefore clear film by the film's geometry alone — no region has to be
    /// estimated, and the exposure, the light table, and the holes themselves
    /// are excluded by *where* the samples are taken rather than by a level.
    ///
    /// The holes are the bright minority of the band's area (measured 0.34–0.39
    /// of it on the maintainer's scans), so the median of the band's column
    /// means is the film, and the median is what this takes.
    ///
    /// Returns [`FilmBase::IDENTITY`] when there is no band to measure; callers
    /// must then ask the operator for a base rather than guess.
    #[must_use]
    pub fn detect_between_sprockets(image: &LinearImage, bands: &[FilmRegion]) -> Self {
        let values: Vec<[f32; 3]> = sample_values(image, bands)
            .into_iter()
            .map(|(_, value)| value)
            .collect();
        if values.is_empty() {
            return Self::IDENTITY;
        }

        // The base is the value most of the samples hold, not their average. A
        // strip whose punch is only found on part of its width puts a handful of
        // points on the other side, where the film is not the same colour; a
        // handful cannot move a majority.
        let mut best = 0usize;
        let mut best_count = 0usize;
        for (index, candidate) in values.iter().enumerate() {
            let count = values
                .iter()
                .filter(|other| agrees(*candidate, **other))
                .count();
            if count > best_count {
                best = index;
                best_count = count;
            }
        }

        // The floor: a value that most samples do not hold is not the film, and
        // the caller has to ask the operator instead of being handed one.
        if best_count * 2 < values.len() || best_count < SAMPLE_MIN {
            return Self::IDENTITY;
        }

        let mut transmittance = [1.0f32; 3];
        for (channel, slot) in transmittance.iter_mut().enumerate() {
            let mut agreeing: Vec<f32> = values
                .iter()
                .filter(|other| agrees(values[best], **other))
                .map(|value| value[channel])
                .collect();
            agreeing.sort_unstable_by(f32::total_cmp);
            *slot = agreeing[agreeing.len() / 2].max(f32::MIN_POSITIVE);
        }
        Self { transmittance }
    }

    /// The sprocket holes themselves, as their rectangles in source pixels.
    ///
    /// A hole is the bright run of columns inside a punch band: the band's own
    /// measured rows bound it above and below, and the run bounds it along the
    /// strip. Diagnostic as much as anything — it is what the sampler blocks,
    /// and drawing it is how a reader checks that the blocking is where they
    /// think it is.
    #[must_use]
    pub fn sprocket_holes(image: &LinearImage, bands: &[FilmRegion]) -> Vec<[f32; 4]> {
        let mut holes = Vec::new();
        for band in bands {
            let columns = sprocket_columns(image, band);
            let Some(bar) = sprocket_bar(&columns) else {
                // No lit perforations in this band: nothing in it can be trusted
                // to be the film *between* two holes. The other band is enough on
                // its own — one side's punch is all the base needs.
                continue;
            };

            // One run of hole columns is one hole, and runs stay inside their own
            // band: a hole of the upper band and one of the lower are two holes
            // however they line up.
            let mut band_holes: Vec<[f32; 4]> = Vec::new();
            let mut run: Option<[f32; 4]> = None;
            for column in columns {
                if column.least() >= bar {
                    run = Some(match run {
                        Some([x0, y0, _, y1]) => [
                            x0,
                            y0.min(column.y_low),
                            column.at[0],
                            y1.max(column.y_high),
                        ],
                        None => [column.at[0], column.y_low, column.at[0], column.y_high],
                    });
                } else if let Some(hole) = run.take() {
                    band_holes.push(hole);
                }
            }
            if let Some(hole) = run {
                band_holes.push(hole);
            }

            // A punch is a lattice. Bright runs that are too few, unevenly
            // spaced, or the wrong share of their pitch are not the film's
            // perforations — shadowed holes, an edge, a reflection — and the gaps
            // between them are not the film between holes. Only one band has to
            // hold a punch: one side's perforations are enough for a base.
            holes.extend(regular_holes(&band_holes));
        }
        holes
    }

    /// The columns the base was measured from, thinned to `limit` of them.
    ///
    /// Exactly the columns [`Self::detect_between_sprockets`] read that agree
    /// with the value it measured — the film between the holes, not the holes —
    /// so a diagnostic drawing of these is a drawing of the measurement.
    #[must_use]
    pub fn sprocket_sample_points(
        image: &LinearImage,
        bands: &[FilmRegion],
        limit: usize,
    ) -> Vec<[f32; 2]> {
        // Only the points that agree with the measured base are the measurement:
        // a point the vote set aside is where the film was not the same colour,
        // and drawing it as a sample would say otherwise.
        let base = Self::detect_between_sprockets(image, bands);
        if base == Self::IDENTITY {
            return Vec::new();
        }
        let mut points: Vec<[f32; 2]> = sample_values(image, bands)
            .into_iter()
            .filter(|(_, value)| agrees(*value, base.transmittance))
            .map(|(at, _)| at)
            .collect();
        if limit == 0 || points.len() <= limit {
            points.truncate(limit);
            return points;
        }
        let step = points.len().div_ceil(limit);
        points.into_iter().step_by(step).collect()
    }

    /// Estimates the base from an authoritative axis-aligned rectangle.
    ///
    /// `rect` is `[x0, y0, x1, y1]` in fractions of the image width and height.
    /// It is intended for a manual override when the operator knows exactly
    /// where the rebate sits: a rectangle over the rebate beats every heuristic
    /// on glare-heavy rigs.
    ///
    /// This uses the same film-level machinery as [`Self::detect`]; the only
    /// difference is that the sample area is bounded in both axes.
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

/// One column of a punch band: where it is, and what it reads.
struct SprocketColumn {
    /// The column's centre, in source coordinates.
    at: [f32; 2],
    /// The top and bottom of the band at that column, in source rows.
    y_low: f32,
    y_high: f32,
    /// The per-channel mean of the band's rows at that column.
    value: [f32; 3],
}

impl SprocketColumn {
    /// The column's least channel: the film's mask absorbs most there, and a
    /// hole has no mask at all in the way, so this is what tells them apart.
    fn least(&self) -> f32 {
        self.value[0].min(self.value[1]).min(self.value[2])
    }
}

/// The gap centres and what the film reads at each.
///
/// One pass over the sample points, so the measurement and any drawing of it are
/// the same points read the same way.
fn sample_values(image: &LinearImage, bands: &[FilmRegion]) -> Vec<([f32; 2], [f32; 3])> {
    gap_centres(image, bands)
        .into_iter()
        .filter_map(|point| window_mean(image, point).map(|value| (point, value)))
        .collect()
}

/// Where the base is read: the centre of each gap between two sprocket holes,
/// stepped just outside the punch band toward the film's edge.
///
/// The holes give the positions exactly. Between one hole and the next the
/// film is unexposed — the camera's gate never reaches the strip's edge — and
/// just outside the punch band, on the side away from the picture, it is
/// unexposed again: the film's own margin. Both are clear film by the film's
/// geometry, and the sample is taken where the geometry says, not where a level
/// says.
///
/// Where the band runs off the sensor, there is no film outside it to read, and
/// the point falls back inside the band's rows — still between two holes, still
/// unexposed.
#[must_use]
fn gap_centres(image: &LinearImage, bands: &[FilmRegion]) -> Vec<[f32; 2]> {
    let height = image.height as usize;
    let mut points = Vec::new();

    // Upper band first, so "outward" is known: the picture lies between them.
    let mut ordered: Vec<FilmRegion> = bands.to_vec();
    ordered.sort_by(|a, b| a.center_y.total_cmp(&b.center_y));

    for (index, band) in ordered.iter().enumerate() {
        let outward = if index == 0 { -1.0f32 } else { 1.0 };
        let edge = band.center_y + outward * band.half_height;
        let outside = edge + outward * EDGE_OFFSET;

        let holes = FilmBase::sprocket_holes(image, std::slice::from_ref(band));
        // Gap centres: between each pair of neighbouring holes.
        for pair in holes.windows(2) {
            let x = f32::midpoint(pair[0][2], pair[1][0]);
            let y = if outside >= 0.0 && outside < coordinate(height) {
                outside
            } else {
                band.center_y
            };
            points.push([x, y]);
        }
    }
    points
}

/// How far outside the punch band's outer edge a sample sits, in source pixels.
const EDGE_OFFSET: f32 = 2.0;

/// The window averaged at one sample point, in source pixels.
const SAMPLE_WINDOW: usize = 5;

/// The per-channel mean of a small window centered on a point, or [`None`] when
/// the point falls outside the image.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a point inside the image, rounded to its pixel"
)]
fn window_mean(image: &LinearImage, point: [f32; 2]) -> Option<[f32; 3]> {
    if point[0] < 0.0 || point[1] < 0.0 {
        return None;
    }
    let width = image.width as usize;
    let height = image.height as usize;
    let center_x = point[0].round() as usize;
    let center_y = point[1].round() as usize;
    if center_x >= width || center_y >= height {
        return None;
    }

    let half = SAMPLE_WINDOW / 2;
    let (pixels, _remainder) = image.data.as_chunks::<3>();
    let mut sums = [0.0f32; 3];
    let mut count = 0usize;
    for y in center_y.saturating_sub(half)..(center_y + half + 1).min(height) {
        for x in center_x.saturating_sub(half)..(center_x + half + 1).min(width) {
            let pixel = pixels[y * width + x];
            for (sum, channel) in sums.iter_mut().zip(pixel) {
                *sum += channel;
            }
            count += 1;
        }
    }
    let count = coordinate(count.max(1));
    Some(sums.map(|sum| sum / count))
}

/// The holes that fit their band's own lattice, or nothing when the band's
/// bright runs are not a punch at all.
///
/// A punch is a lattice: holes of one width at one pitch, covering a known share
/// of the strip's length. On 35 mm that is a 1.98 mm hole every 4.75 mm — a duty
/// of about 0.42, and the maintainer's scans measure 0.40 in a full band's rows.
/// Runs that do not fit that shape are not the film's perforations, and the gaps
/// between them are not the film between holes.
///
/// The fit is judged against the band's *own* median, and runs that miss it are
/// dropped rather than failing the band: a perforation cut by the sensor edge
/// (measured 132 px against a 286 px median) and one-column artifacts are both
/// normal, and the holes that remain still say where the film between them is.
/// A band left with too few holes after that is refused outright.
fn regular_holes(holes: &[[f32; 4]]) -> Vec<[f32; 4]> {
    let widths = |holes: &[[f32; 4]]| -> Vec<f32> {
        let mut values: Vec<f32> = holes.iter().map(|hole| hole[2] - hole[0]).collect();
        values.sort_by(f32::total_cmp);
        values
    };

    let mut widths = widths(holes);
    if widths.len() < PUNCH_MIN_HOLES {
        return Vec::new();
    }
    let median_width = widths[widths.len() / 2];
    if median_width <= 0.0 {
        return Vec::new();
    }

    let fitted: Vec<[f32; 4]> = holes
        .iter()
        .copied()
        .filter(|hole| (hole[2] - hole[0] - median_width).abs() <= PUNCH_TOLERANCE * median_width)
        .collect();
    if fitted.len() < PUNCH_MIN_HOLES {
        return Vec::new();
    }

    let mut pitches: Vec<f32> = fitted
        .windows(2)
        .map(|pair| f32::midpoint(pair[1][0], pair[1][2]) - f32::midpoint(pair[0][0], pair[0][2]))
        .collect();
    pitches.sort_by(f32::total_cmp);
    let median_pitch = pitches[pitches.len() / 2];
    if median_pitch <= 0.0 {
        return Vec::new();
    }

    let regular = pitches
        .iter()
        .all(|pitch| (pitch - median_pitch).abs() <= PUNCH_TOLERANCE * median_pitch);
    let duty = median_width / median_pitch;
    if !regular || !(PUNCH_DUTY_MIN..=PUNCH_DUTY_MAX).contains(&duty) {
        return Vec::new();
    }

    widths.clear();
    fitted
}

/// How far apart two samples' values may sit, per channel, and still count as
/// the same film, as a fraction of their mean.
///
/// Clear film is one colour: the mask's channel ratios are enormous by
/// comparison, and a sample on the wrong side of the strip, on a hole, or on a
/// light leak differs far beyond this. It also absorbs the light's falloff
/// across a frame, which is a few percent corner to corner.
const SAMPLE_AGREEMENT: f32 = 0.15;

/// Fewest samples that must agree before their value is taken as the base.
///
/// Below this there is not enough film read to be a measurement, and the caller
/// has to ask the operator rather than be handed a guess.
const SAMPLE_MIN: usize = 3;

/// Whether two samples read as the same film.
fn agrees(a: [f32; 3], b: [f32; 3]) -> bool {
    a.iter()
        .zip(b)
        .all(|(&left, right)| (left - right).abs() <= SAMPLE_AGREEMENT * f32::midpoint(left, right))
}

/// Fewest perforations for a band's spacing to mean anything.
const PUNCH_MIN_HOLES: usize = 3;

/// How far a hole's width or a gap's size may sit from the band's median, as a
/// fraction of it.
const PUNCH_TOLERANCE: f32 = 0.25;

/// The share of a punch's pitch that a hole occupies, as measured on 35 mm and
/// on the maintainer's scans.
const PUNCH_DUTY_MIN: f32 = 0.20;
const PUNCH_DUTY_MAX: f32 = 0.60;

/// The brightness that separates a hole from the film between holes.
///
/// The film is the majority of a punch band's area, so the median column reads
/// film; bare light through a hole is [`HOLE_BRIGHTNESS`] times that, the same
/// bar the frame detector uses for the holes.
fn sprocket_bar(columns: &[SprocketColumn]) -> Option<f32> {
    if columns.is_empty() {
        return None;
    }
    let mut levels: Vec<f32> = columns.iter().map(SprocketColumn::least).collect();
    levels.sort_unstable_by(f32::total_cmp);
    let median = levels[levels.len() / 2];
    let brightest = levels[levels.len() - 1];
    if std::env::var("GRAINBASE_TRACE").is_ok() {
        println!(
            "trace: band columns {} median {median:.4} brightest {brightest:.4}",
            levels.len()
        );
    }
    if median <= 0.0 || brightest < median * HOLE_BRIGHTNESS {
        return None;
    }
    Some(median * HOLE_BRIGHTNESS)
}

/// Bare light reads this many times the clear-film level.
const HOLE_BRIGHTNESS: f32 = 2.0;

/// Columns read across each punch band, before thinning for a drawing.
const COLUMN_SAMPLE_TARGET: usize = 512;

/// The columns of each punch band, with the mean of the band's rows at each.
///
/// A column of a punch band is either a hole or the film between two holes, so
/// one mean per column is one reading of one or the other, and the row range
/// comes from the band's own measured edges rather than from a level.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "band rows are clamped to the image's pixel count before the index \
              casts, and source coordinates are photograph-sized"
)]
fn sprocket_columns(image: &LinearImage, band: &FilmRegion) -> Vec<SprocketColumn> {
    let (pixels, _remainder) = image.data.as_chunks::<3>();
    let width = image.width as usize;
    let height = image.height as usize;
    if pixels.is_empty() || width == 0 || height == 0 {
        return Vec::new();
    }

    let center_x = coordinate(width) / 2.0;
    let last_row = height.saturating_sub(1) as f32;
    let column_stride = (width / COLUMN_SAMPLE_TARGET).max(1);

    let mut columns = Vec::new();
    for x in (0..width).step_by(column_stride) {
        let x_f = coordinate(x);
        let y_center = band.center_y + band.slope * (x_f - center_x);
        let y_low = (y_center - band.half_height).ceil().max(0.0);
        let y_high = (y_center + band.half_height).floor().min(last_row);
        if y_low > y_high {
            continue;
        }

        let mut sums = [0.0f32; 3];
        let mut count = 0usize;
        for y in y_low as usize..=y_high as usize {
            let pixel = pixels[y * width + x];
            for (sum, channel) in sums.iter_mut().zip(pixel) {
                *sum += channel;
            }
            count += 1;
        }
        if count == 0 {
            continue;
        }
        let count = coordinate(count);
        columns.push(SprocketColumn {
            at: [x_f, f32::midpoint(y_low, y_high)],
            y_low,
            y_high,
            value: sums.map(|sum| sum / count),
        });
    }
    columns
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
}
