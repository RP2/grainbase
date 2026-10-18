//! Locating and measuring a photographed `ColorChecker` Classic for calibration.
//!
//! The `ColorChecker` Classic is a 4 × 6 grid of patches surrounded by a black
//! border and separated by thin lines. On a film negative the border and
//! separators are unexposed film and therefore bright, while the exposed patches
//! are darker. This module finds the grid from those bright straight lines and
//! measures the per-patch optical densities that [`fit_profile`](super::fit)
//! needs.
//!
//! Detection is geometric: it looks for the evenly spaced lattice of separator
//! lines, not for patch colors. Level ratios are only used as bars against the
//! measured film base, mirroring the rest of the detector family.
//!
//! Orientation is validated after sampling by the densities themselves: the
//! white patch (bottom-left in a correctly oriented capture) is the densest
//! cell on a negative, and the black patch (bottom-right) is the least dense.
//! A wrong orientation is reported as a concrete next step, not silently
//! fitted.

use std::fmt;

use crate::image::LinearImage;
use crate::pipeline::FilmBase;
use crate::pipeline::fit::{self, FitError};
use crate::profile::{Method, Profile};

/// Number of `ColorChecker` Classic rows.
const ROWS: usize = 4;
/// Number of `ColorChecker` Classic columns.
const COLS: usize = 6;
/// Total patches on a `ColorChecker` Classic.
const CELLS: usize = ROWS * COLS;

/// A measured `ColorChecker` Classic: 24 ordered cells of per-channel density.
///
/// Cells are ordered row-major from the top-left patch to the bottom-right
/// patch, matching the `ColorChecker` Classic layout.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Measurements {
    /// Per-channel optical density above base for each of the 24 patches.
    pub cells: [[f32; 3]; CELLS],
}

/// Why a chart could not be measured or fitted.
#[derive(Debug, Clone, PartialEq)]
pub enum ChartError {
    /// The grid lines could not be located confidently.
    NoLattice {
        /// What went wrong, in terms a user can act on.
        reason: String,
    },
    /// The measured densities place the white patch somewhere unexpected.
    Orientation {
        /// Which flag would fix the orientation, if any.
        hint: &'static str,
        /// What the sampler saw.
        detail: String,
    },
    /// A patch's internal samples are too spread out for a trustworthy average.
    PatchSpread {
        /// Cell index in row-major order.
        patch: usize,
        /// Channel that failed the spread gate.
        channel: usize,
    },
    /// The fitter rejected the measured patches.
    Fit(FitError),
}

impl fmt::Display for ChartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoLattice { reason } => write!(
                f,
                "could not locate the `ColorChecker` grid: {reason}. Check that the card fills the frame, is in focus, and is evenly lit."
            ),
            Self::Orientation { hint, detail } => {
                write!(
                    f,
                    "{detail}. Re-capture with the white patch at the bottom-left, or pass {hint}"
                )
            }
            Self::PatchSpread { patch, channel } => write!(
                f,
                "patch {patch} has uneven samples in channel {channel}; the grid may be misaligned"
            ),
            Self::Fit(err) => write!(f, "could not fit the chart: {err}"),
        }
    }
}

impl std::error::Error for ChartError {}

impl From<FitError> for ChartError {
    fn from(err: FitError) -> Self {
        Self::Fit(err)
    }
}

/// Gretag-Macbeth/X-Rite published 8-bit sRGB patch values for the `ColorChecker`
/// Classic, normalized to 0..1.
///
/// Cross-checked against Danny Pascale's `BabelColor` survey and Charles
/// Poynton's `ColorChecker` page.
/// Values are in row-major order, top-left to bottom-right.
const PATCHES_SRGB: [[f32; 3]; CELLS] = [
    // Row 1.
    [115.0 / 255.0, 82.0 / 255.0, 68.0 / 255.0],
    [194.0 / 255.0, 150.0 / 255.0, 130.0 / 255.0],
    [98.0 / 255.0, 122.0 / 255.0, 157.0 / 255.0],
    [87.0 / 255.0, 108.0 / 255.0, 67.0 / 255.0],
    [133.0 / 255.0, 128.0 / 255.0, 177.0 / 255.0],
    [103.0 / 255.0, 189.0 / 255.0, 170.0 / 255.0],
    // Row 2.
    [214.0 / 255.0, 126.0 / 255.0, 44.0 / 255.0],
    [80.0 / 255.0, 91.0 / 255.0, 166.0 / 255.0],
    [193.0 / 255.0, 90.0 / 255.0, 99.0 / 255.0],
    [94.0 / 255.0, 60.0 / 255.0, 108.0 / 255.0],
    [157.0 / 255.0, 188.0 / 255.0, 64.0 / 255.0],
    [224.0 / 255.0, 163.0 / 255.0, 46.0 / 255.0],
    // Row 3.
    [56.0 / 255.0, 61.0 / 255.0, 150.0 / 255.0],
    [70.0 / 255.0, 148.0 / 255.0, 73.0 / 255.0],
    [175.0 / 255.0, 54.0 / 255.0, 60.0 / 255.0],
    [231.0 / 255.0, 199.0 / 255.0, 31.0 / 255.0],
    [187.0 / 255.0, 86.0 / 255.0, 149.0 / 255.0],
    [8.0 / 255.0, 133.0 / 255.0, 161.0 / 255.0],
    // Row 4 (neutral row, white at index 18, black at index 23).
    [243.0 / 255.0, 243.0 / 255.0, 242.0 / 255.0],
    [200.0 / 255.0, 200.0 / 255.0, 200.0 / 255.0],
    [160.0 / 255.0, 160.0 / 255.0, 160.0 / 255.0],
    [122.0 / 255.0, 122.0 / 255.0, 121.0 / 255.0],
    [85.0 / 255.0, 85.0 / 255.0, 85.0 / 255.0],
    [52.0 / 255.0, 52.0 / 255.0, 52.0 / 255.0],
];

/// The IEC 61966-2-1 sRGB electro-optical transfer function.
///
/// Converts normalized nonlinear sRGB values to linear-light values in 0..1.
#[must_use]
fn srgb_eotf(nonlinear: f32) -> f32 {
    // IEC 61966-2-1:1999, section 5.2.
    if nonlinear <= 0.04045 {
        nonlinear / 12.92
    } else {
        ((nonlinear + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear-light sRGB targets for the 18 chromatic patches.
#[must_use]
fn chromatic_targets() -> [[f32; 3]; 18] {
    let mut out = [[0.0f32; 3]; 18];
    for (i, patch) in PATCHES_SRGB[..18].iter().enumerate() {
        for (channel, value) in patch.iter().enumerate() {
            out[i][channel] = srgb_eotf(*value);
        }
    }
    out
}

/// Smallest image dimension the detector will attempt to grid.
const MIN_DIMENSION: u32 = 80;

/// How bright a separator pixel must be, as a fraction of the film base.
///
/// Separators are clear film, so they read at or just below the base level.
/// A small margin below base catches slight underexposure and noise without
/// letting dark patch interiors qualify.
pub(crate) const SEPARATOR_BRIGHTNESS: f32 = 0.90;

/// Window size for smoothing the 1-D separator projection.
pub(crate) const SMOOTH_RADIUS: usize = 1;

/// Fraction of a cell's width/height left as margin outside the sample window.
///
/// The central 50% of each cell is averaged, keeping the bright separator
/// borders out of the measurement.
pub(crate) const SAMPLE_INSET: f32 = 0.25;

/// Maximum allowed ratio of max sample to mean sample inside a patch window.
///
/// A uniform patch gives a ratio near 1.0; a window that crosses a separator
/// or edge gives a much larger ratio. The bound is generous to tolerate real
/// grain while still catching a misaligned grid.
pub(crate) const MAX_SPREAD_RATIO: f32 = 2.5;

/// Minimum fraction of the projection axis that must read as separator along
/// the detected lines for the lattice to be trusted.
///
/// Real separators span almost the full axis; a random-noise image cannot
/// produce a coherent set of equally spaced lines that scores this high.
pub(crate) const MIN_LINE_FILL: f32 = 0.70;

/// Maximum proportional deviation of detected spacing from the ideal equal
/// spacing for a card that fills the frame.
pub(crate) const MAX_SPACING_TOL: f32 = 0.15;

/// Locates the `ColorChecker` grid in `image` and measures per-patch densities.
///
/// `base` is the measured film base for the uncropped scan. Densities are
/// computed as `-log10(mean_transmittance / base_transmittance)` per channel,
/// clamped to the same maximum used by the conversion core.
///
/// # Errors
///
/// Returns [`ChartError::NoLattice`] when the separator lattice cannot be found,
/// [`ChartError::Orientation`] when the white patch is not in the bottom-left
/// cell, or [`ChartError::PatchSpread`] when a sample window is not uniform
/// enough.
pub fn measure(image: &LinearImage, base: &FilmBase) -> Result<Measurements, ChartError> {
    if image.width < MIN_DIMENSION || image.height < MIN_DIMENSION {
        return Err(ChartError::NoLattice {
            reason: format!(
                "image is too small ({}x{}); a chart capture should be much larger",
                image.width, image.height
            ),
        });
    }

    let (vertical, horizontal) = detect_lattice(image, base)?;
    let cells = sample_cells(image, base, &vertical, &horizontal)?;
    validate_orientation(&cells)?;
    Ok(Measurements { cells })
}

/// Detects the vertical and horizontal separator lines.
///
/// Returns the x-coordinates of the 7 vertical lines and the y-coordinates of
/// the 5 horizontal lines, both sorted from low to high.
fn detect_lattice(
    image: &LinearImage,
    base: &FilmBase,
) -> Result<(Vec<f32>, Vec<f32>), ChartError> {
    let base_min = base
        .transmittance
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min)
        .max(f32::MIN_POSITIVE);
    let threshold = base_min * SEPARATOR_BRIGHTNESS;

    let (vertical_proj, horizontal_proj) = separator_projections(image, threshold);

    let vertical =
        find_lines_1d(&vertical_proj, COLS + 1, COLS).map_err(|reason| ChartError::NoLattice {
            reason: format!("vertical separators: {reason}"),
        })?;
    let horizontal = find_lines_1d(&horizontal_proj, ROWS + 1, COLS).map_err(|reason| {
        ChartError::NoLattice {
            reason: format!("horizontal separators: {reason}"),
        }
    })?;

    Ok((vertical, horizontal))
}

/// Projects the bright-separator mask onto the x and y axes.
///
/// A pixel qualifies as separator-like when all three channels are above the
/// threshold. Using the least channel would let a bright channel from a colored
/// patch qualify; requiring all channels keeps the bar on clear film.
pub(crate) fn separator_projections(image: &LinearImage, threshold: f32) -> (Vec<f32>, Vec<f32>) {
    let width = image.width as usize;
    let height = image.height as usize;
    let (pixels, _remainder) = image.data.as_chunks::<3>();

    let mut vertical = vec![0.0f32; width];
    let mut horizontal = vec![0.0f32; height];

    for y in 0..height {
        for x in 0..width {
            let pixel = pixels[y * width + x];
            if pixel.iter().all(|channel| *channel >= threshold) {
                vertical[x] += 1.0;
                horizontal[y] += 1.0;
            }
        }
    }

    (vertical, horizontal)
}

/// Finds `count` equally spaced lines in a 1-D projection.
///
/// The capture protocol fills the frame with the card, so the grid spans nearly
/// the full axis and the lines are at integer fractions of it. The search
/// looks for the start offset and spacing that put the most separator mass on
/// the expected line positions.
#[expect(
    clippy::cast_precision_loss,
    reason = "search indices are bounded by the small projection length"
)]
pub(crate) fn find_lines_1d(
    projection: &[f32],
    count: usize,
    cell_count: usize,
) -> Result<Vec<f32>, String> {
    if count < 2 || projection.len() < count {
        return Err("projection too short".to_owned());
    }

    let len = projection.len();
    let max_value = projection
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
        .max(f32::MIN_POSITIVE);
    let smoothed = smooth(projection, SMOOTH_RADIUS);

    // Ideal spacing for a card that fills the axis.
    let ideal_step = len as f32 / (count - 1) as f32;
    let step_min = ideal_step * (1.0 - MAX_SPACING_TOL);
    let step_max = ideal_step * (1.0 + MAX_SPACING_TOL);
    // A lattice's offset is periodic in its spacing: an offset of one full
    // period is the same set of lines, so the search covers a whole period
    // rather than half of one. Half a period assumes the card fills the axis to
    // within half a cell, which a real capture — and a 20-step wedge in
    // particular — does not guarantee, and a missed offset then refines onto
    // cell edges instead of separators.
    let start_max = ideal_step;

    let mut best_score = f32::NEG_INFINITY;
    let mut best_lines: Option<Vec<f32>> = None;

    for step in linspace(step_min, step_max, 24) {
        if step <= 0.0 {
            continue;
        }
        for start in linspace(0.0, start_max, 12) {
            let lines: Vec<f32> = (0..count).map(|i| start + i as f32 * step).collect();
            if *lines.last().expect("count >= 2") > (len - 1) as f32 {
                continue;
            }
            let score = lines
                .iter()
                .map(|&pos| sample_smooth(&smoothed, pos))
                .sum::<f32>();
            if score > best_score {
                best_score = score;
                best_lines = Some(lines);
            }
        }
    }

    let Some(lines) = best_lines else {
        return Err("no line candidate fit inside the image".to_owned());
    };

    // Refine each line to the nearest local maximum in the smoothed projection.
    let refined: Vec<f32> = lines
        .iter()
        .map(|&pos| refine_to_peak(&smoothed, pos, cell_count))
        .collect();

    let refined_score = refined
        .iter()
        .map(|&pos| sample_smooth(&smoothed, pos))
        .sum::<f32>();
    let fill = refined_score / (count as f32 * max_value);
    if fill < MIN_LINE_FILL {
        return Err(format!(
            "separator response too weak ({:.0}%); expected strong bright lines",
            fill * 100.0
        ));
    }

    let spacings: Vec<f32> = refined.windows(2).map(|w| w[1] - w[0]).collect();
    let min_spacing = spacings.iter().copied().fold(f32::INFINITY, f32::min);
    let max_spacing = spacings.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max_spacing <= 0.0 || min_spacing / max_spacing < 0.75 {
        return Err("detected lines are not evenly spaced".to_owned());
    }

    Ok(refined)
}

/// Samples a smoothed projection at a fractional position.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "interpolation indices are clamped to the projection bounds"
)]
pub(crate) fn sample_smooth(smoothed: &[f32], pos: f32) -> f32 {
    let len = smoothed.len();
    if len == 0 {
        return 0.0;
    }
    let pos = pos.clamp(0.0, (len - 1) as f32);
    let i0 = pos.floor() as usize;
    let i1 = (i0 + 1).min(len - 1);
    let frac = pos - i0 as f32;
    smoothed[i0] * (1.0 - frac) + smoothed[i1] * frac
}

/// Refines a coarse line position to the nearest local maximum.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "projection indices are small positive integers derived from the image width"
)]
pub(crate) fn refine_to_peak(smoothed: &[f32], pos: f32, cell_count: usize) -> f32 {
    let len = smoothed.len();
    let center = pos.round().clamp(0.0, (len - 1) as f32) as usize;
    let search_radius = (len as f32 / (cell_count as f32 * 4.0)).max(2.0) as usize;

    let mut best = center;
    let mut best_value = smoothed[center];
    let first = center.saturating_sub(search_radius);
    let last = (center + search_radius).min(len - 1);
    for (i, value) in smoothed.iter().enumerate().take(last + 1).skip(first) {
        if *value > best_value {
            best = i;
            best_value = *value;
        }
    }

    // Sub-pixel parabolic refinement around the peak.
    let i = best;
    if i == 0 || i >= len - 1 {
        return best as f32;
    }
    let a = smoothed[i - 1];
    let b = smoothed[i];
    let c = smoothed[i + 1];
    let denom = a - 2.0 * b + c;
    if denom.abs() < f32::EPSILON {
        return best as f32;
    }
    let offset = 0.5 * (a - c) / denom;
    (best as f32 + offset).clamp(0.0, (len - 1) as f32)
}

/// Smooths a 1-D signal with a small box filter.
#[expect(
    clippy::cast_precision_loss,
    reason = "the box count is a small positive integer, well within f32 precision"
)]
pub(crate) fn smooth(values: &[f32], radius: usize) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0.0f32; values.len()];
    for (i, slot) in out.iter_mut().enumerate() {
        let from = i.saturating_sub(radius);
        let to = (i + radius).min(values.len() - 1);
        let count = to - from + 1;
        let sum: f32 = values[from..=to].iter().sum();
        *slot = sum / count as f32;
    }
    out
}

/// Generates `n` evenly spaced samples in `[start, end]`.
#[expect(
    clippy::cast_precision_loss,
    reason = "the search grid count is a small integer"
)]
pub(crate) fn linspace(start: f32, end: f32, n: usize) -> impl Iterator<Item = f32> {
    if n <= 1 {
        return vec![start].into_iter();
    }
    let step = (end - start) / (n - 1) as f32;
    (0..n)
        .map(move |i| start + i as f32 * step)
        .collect::<Vec<_>>()
        .into_iter()
}

/// Samples the central window of every cell and converts to density.
fn sample_cells(
    image: &LinearImage,
    base: &FilmBase,
    vertical: &[f32],
    horizontal: &[f32],
) -> Result<[[f32; 3]; CELLS], ChartError> {
    if vertical.len() != COLS + 1 || horizontal.len() != ROWS + 1 {
        return Err(ChartError::NoLattice {
            reason: "wrong number of grid lines".to_owned(),
        });
    }

    let mut cells = [[0.0f32; 3]; CELLS];
    for row in 0..ROWS {
        for col in 0..COLS {
            let x0 = vertical[col];
            let x1 = vertical[col + 1];
            let y0 = horizontal[row];
            let y1 = horizontal[row + 1];

            let inset_x = (x1 - x0) * SAMPLE_INSET;
            let inset_y = (y1 - y0) * SAMPLE_INSET;
            let window = Window {
                x0: x0 + inset_x,
                x1: x1 - inset_x,
                y0: y0 + inset_y,
                y1: y1 - inset_y,
            };

            let index = row * COLS + col;
            cells[index] =
                sample_window(image, base, &window).map_err(|channel| ChartError::PatchSpread {
                    patch: index,
                    channel,
                })?;
        }
    }
    Ok(cells)
}

/// A rectangular sample window in fractional pixel coordinates.
#[derive(Clone, Copy)]
pub(crate) struct Window {
    pub(crate) x0: f32,
    pub(crate) x1: f32,
    pub(crate) y0: f32,
    pub(crate) y1: f32,
}

/// Averages transmittance over a window and returns optical density.
///
/// Returns the failing channel index if the window's internal spread is too
/// high.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "window bounds are clamped to non-negative image dimensions before truncation"
)]
pub(crate) fn sample_window(
    image: &LinearImage,
    base: &FilmBase,
    window: &Window,
) -> Result<[f32; 3], usize> {
    let width = image.width as usize;
    let height = image.height as usize;
    let (pixels, _remainder) = image.data.as_chunks::<3>();

    let x0 = window.x0.clamp(0.0, width.saturating_sub(1) as f32) as usize;
    let x1 = window.x1.clamp(0.0, width.saturating_sub(1) as f32) as usize;
    let y0 = window.y0.clamp(0.0, height.saturating_sub(1) as f32) as usize;
    let y1 = window.y1.clamp(0.0, height.saturating_sub(1) as f32) as usize;

    if x1 <= x0 || y1 <= y0 {
        return Err(0);
    }

    let mut sums = [0.0f64; 3];
    let mut maxima = [0.0f32; 3];
    let mut count = 0u64;

    for y in y0..=y1 {
        for x in x0..=x1 {
            let pixel = pixels[y * width + x];
            for channel in 0..3 {
                sums[channel] += f64::from(pixel[channel]);
                if pixel[channel] > maxima[channel] {
                    maxima[channel] = pixel[channel];
                }
            }
            count += 1;
        }
    }

    if count == 0 {
        return Err(0);
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "pixel count in a sample window fits in f32"
    )]
    let count_f = count as f32;

    let mut density = [0.0f32; 3];
    for channel in 0..3 {
        let mean = (sums[channel] / f64::from(count_f)) as f32;
        if mean <= 0.0 {
            density[channel] = super::MAX_DENSITY;
        } else {
            let base_channel = base.transmittance[channel].max(f32::MIN_POSITIVE);
            density[channel] =
                (-((mean / base_channel).clamp(0.0, 1.0).log10())).min(super::MAX_DENSITY);
        }

        let mean_positive = mean.max(f32::MIN_POSITIVE);
        if maxima[channel] / mean_positive > MAX_SPREAD_RATIO {
            return Err(channel);
        }
    }

    Ok(density)
}

/// Validates orientation using measured densities.
///
/// On a negative, the white patch is the densest cell and the black patch is
/// the least dense. The white patch must sit in the bottom-left cell; the black
/// patch must sit in the bottom-right cell and be less dense than at least the
/// average chromatic patch.
fn validate_orientation(cells: &[[f32; 3]; CELLS]) -> Result<(), ChartError> {
    // Use a scalar density per cell: the mean of the three channels. Neutrals
    // are neutral, so the mean is representative; this is only for placement.
    let scalar: Vec<f32> = cells
        .iter()
        .map(|cell| (cell[0] + cell[1] + cell[2]) / 3.0)
        .collect();

    let white_index = scalar
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .expect("cells is non-empty");
    let black_index = scalar
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .expect("cells is non-empty");

    let white_row = white_index / COLS;
    let white_col = white_index % COLS;

    if white_row < ROWS / 2 {
        return Err(ChartError::Orientation {
            hint: "--flip-v",
            detail: format!(
                "the densest patch (white) is in row {white_row}, but it should be in the bottom row"
            ),
        });
    }
    if white_row == ROWS - 1 && white_col >= COLS / 2 {
        return Err(ChartError::Orientation {
            hint: "--flip-h",
            detail: format!(
                "the densest patch (white) is at row {white_row}, column {white_col}, but it should be at the bottom-left"
            ),
        });
    }

    let black_row = black_index / COLS;
    let black_col = black_index % COLS;
    let chromatic_avg: f32 = scalar[..18].iter().sum::<f32>() / 18.0;
    if black_row != ROWS - 1 || black_col != COLS - 1 || scalar[black_index] >= chromatic_avg {
        return Err(ChartError::Orientation {
            hint: "re-capture with the chart upright and unobscured",
            detail: format!(
                "the least dense patch (black) is at row {black_row}, column {black_col}; expected bottom-right and less dense than the chromatic average"
            ),
        });
    }

    Ok(())
}

/// Fits a profile from a measured ``ColorChecker`` capture.
///
/// Uses the bottom-right black patch as the dark anchor and the bottom-left
/// white patch as the light anchor. The 18 chromatic patches supply the
/// crosstalk gallery, with targets linearized from the published sRGB values
/// through the IEC 61966-2-1 EOTF.
///
/// # Errors
///
/// Propagates chart measurement errors and [`FitError`] from the fitter.
pub fn fit_capture(
    measurements: &Measurements,
    film: &str,
    camera: &str,
) -> Result<Profile, ChartError> {
    let cells = measurements.cells;
    let dark = cells[23];
    let light = cells[18];

    let targets = chromatic_targets();
    let patch: Vec<([f32; 3], [f32; 3])> = cells[..18]
        .iter()
        .zip(targets.iter())
        .map(|(&density, &target)| (density, target))
        .collect();

    let settings = fit::fit_profile(dark, light, &patch)?;

    Ok(Profile {
        film: film.to_owned(),
        camera: camera.to_owned(),
        family: None,
        method: Method::Fitted,
        frames: 1,
        settings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::IDENTITY;

    fn close(actual: f32, expected: f32, what: &str) {
        assert!(
            (actual - expected).abs() < 1e-3,
            "{what}: expected {expected}, got {actual}"
        );
    }

    /// Generates a synthetic negative of a ``ColorChecker`` Classic.
    ///
    /// `settings` supplies the ground-truth floor, ceiling, and crosstalk used
    /// to derive patch densities. The black and white neutrals are generated
    /// from the fitter's dark/light print tones so the anchors recover exactly.
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "synthetic image dimensions are small and coordinates are clamped before truncation"
    )]
    fn synthetic_checker(settings: &crate::pipeline::Settings) -> (LinearImage, FilmBase) {
        let width = 1200u32;
        let height = 800u32;
        let mut data = vec![0.0f32; width as usize * height as usize * 3];

        let floor = settings.density_floor.unwrap_or([0.0f32; 3]);
        let ceiling = settings
            .density_max
            .unwrap_or([crate::pipeline::DEFAULT_DENSITY_MAX; 3]);
        let range = [
            ceiling[0] - floor[0],
            ceiling[1] - floor[1],
            ceiling[2] - floor[2],
        ];

        // Invert the crosstalk matrix to go from corrected densities to raw
        // measured densities.
        let inv = invert_matrix(settings.crosstalk);

        let targets = chromatic_targets();
        let mut patch_densities = [[0.0f32; 3]; CELLS];
        for i in 0..18 {
            patch_densities[i] = density_from_target(targets[i], floor, range, inv);
        }
        patch_densities[18] = density_from_target([fit::LIGHT_PRINT_TONE; 3], floor, range, inv);
        patch_densities[23] = density_from_target([fit::DARK_PRINT_TONE; 3], floor, range, inv);
        // Other neutrals are not used by the fit, but give them plausible
        // densities so the orientation check sees a clear white/black.
        patch_densities[19] = density_from_target([0.75f32; 3], floor, range, inv);
        patch_densities[20] = density_from_target([0.55f32; 3], floor, range, inv);
        patch_densities[21] = density_from_target([0.35f32; 3], floor, range, inv);
        patch_densities[22] = density_from_target([0.20f32; 3], floor, range, inv);

        let base_transmittance = [0.8f32, 0.6f32, 0.4f32];
        let base = FilmBase {
            transmittance: base_transmittance,
        };

        // Card fills the frame with a small bright margin.
        let margin_x = width as f32 / 24.0;
        let margin_y = height as f32 / 24.0;
        let left = margin_x;
        let right = width as f32 - margin_x;
        let top = margin_y;
        let bottom = height as f32 - margin_y;
        let cell_w = (right - left) / COLS as f32;
        let cell_h = (bottom - top) / ROWS as f32;
        let sep_w = cell_w * 0.06f32;
        let sep_h = cell_h * 0.06f32;

        for y in 0..height as usize {
            for x in 0..width as usize {
                let xf = x as f32;
                let yf = y as f32;

                let value = if xf < left || xf >= right || yf < top || yf >= bottom {
                    // Bright border.
                    base_transmittance
                } else {
                    let col = ((xf - left) / cell_w) as usize;
                    let row = ((yf - top) / cell_h) as usize;
                    let in_sep_x = (xf - left) - col as f32 * cell_w < sep_w
                        || (left + (col + 1) as f32 * cell_w) - xf < sep_w;
                    let in_sep_y = (yf - top) - row as f32 * cell_h < sep_h
                        || (top + (row + 1) as f32 * cell_h) - yf < sep_h;
                    let col = col.min(COLS - 1);
                    let row = row.min(ROWS - 1);

                    if in_sep_x || in_sep_y {
                        base_transmittance
                    } else {
                        transmittance_from_density(
                            patch_densities[row * COLS + col],
                            base_transmittance,
                        )
                    }
                };

                let idx = (y * width as usize + x) * 3;
                data[idx] = value[0];
                data[idx + 1] = value[1];
                data[idx + 2] = value[2];
            }
        }

        (
            LinearImage::new(width, height, data).expect("well-formed"),
            base,
        )
    }

    fn density_from_target(
        target: [f32; 3],
        floor: [f32; 3],
        range: [f32; 3],
        inv_crosstalk: [[f32; 3]; 3],
    ) -> [f32; 3] {
        let corrected = [
            target[0] * range[0] + floor[0],
            target[1] * range[1] + floor[1],
            target[2] * range[2] + floor[2],
        ];
        apply_matrix(inv_crosstalk, corrected)
    }

    fn transmittance_from_density(density: [f32; 3], base: [f32; 3]) -> [f32; 3] {
        [
            base[0] * 10f32.powf(-density[0]),
            base[1] * 10f32.powf(-density[1]),
            base[2] * 10f32.powf(-density[2]),
        ]
    }

    fn apply_matrix(matrix: [[f32; 3]; 3], vector: [f32; 3]) -> [f32; 3] {
        matrix.map(|row| row[0] * vector[0] + row[1] * vector[1] + row[2] * vector[2])
    }

    fn invert_matrix(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        let inv_det = 1.0 / det;

        let mut out = [[0.0f32; 3]; 3];
        out[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
        out[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
        out[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
        out[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
        out[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
        out[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det;
        out[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
        out[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det;
        out[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;
        out
    }

    #[test]
    fn synthetic_checker_yields_24_cells() {
        let settings = crate::pipeline::Settings {
            density_max: Some([2.0f32; 3]),
            density_floor: Some([0.2f32; 3]),
            crosstalk: [
                [1.0f32, 0.05f32, 0.02f32],
                [0.03f32, 1.0f32, 0.04f32],
                [0.01f32, 0.02f32, 1.0f32],
            ],
            ..crate::pipeline::Settings::default()
        };
        let (image, base) = synthetic_checker(&settings);
        let measurements = measure(&image, &base).expect("checker should be found");
        assert_eq!(measurements.cells.len(), CELLS);
    }

    #[test]
    fn synthetic_checker_recovers_known_settings() {
        // Use identity crosstalk so the measured raw densities equal the
        // corrected densities; this lets the anchor fit recover the exact
        // generating floor and range, matching the fit.rs test idiom.
        let floor = [0.2f32; 3];
        let ceiling = [2.2f32; 3];
        let settings = crate::pipeline::Settings {
            density_max: Some(ceiling),
            density_floor: Some(floor),
            crosstalk: IDENTITY,
            ..crate::pipeline::Settings::default()
        };
        let (image, base) = synthetic_checker(&settings);
        let measurements = measure(&image, &base).expect("checker should be found");
        let profile = fit_capture(&measurements, "test", "test").expect("fit should succeed");

        let fitted_floor = profile.settings.density_floor.expect("fit emits floor");
        let fitted_ceiling = profile.settings.density_max.expect("fit emits ceiling");
        for channel in 0..3 {
            close(fitted_floor[channel], floor[channel], "floor");
            close(fitted_ceiling[channel], ceiling[channel], "ceiling");
        }
        for (row, expected_row) in IDENTITY.iter().enumerate() {
            for (col, expected) in expected_row.iter().enumerate() {
                close(profile.settings.crosstalk[row][col], *expected, "crosstalk");
            }
        }
    }

    #[test]
    fn vertically_flipped_checker_is_reported() {
        let settings = crate::pipeline::Settings {
            density_max: Some([2.0f32; 3]),
            density_floor: Some([0.2f32; 3]),
            crosstalk: IDENTITY,
            ..crate::pipeline::Settings::default()
        };
        let (image, base) = synthetic_checker(&settings);
        let flipped = image.flipped_vertical();
        let err = measure(&flipped, &base).expect_err("flipped chart should fail");
        let msg = err.to_string();
        assert!(
            msg.contains("--flip-v"),
            "error should suggest --flip-v: {msg}"
        );
    }

    #[test]
    fn horizontally_flipped_checker_is_reported() {
        let settings = crate::pipeline::Settings {
            density_max: Some([2.0f32; 3]),
            density_floor: Some([0.2f32; 3]),
            crosstalk: IDENTITY,
            ..crate::pipeline::Settings::default()
        };
        let (image, base) = synthetic_checker(&settings);
        let flipped = image.flipped_horizontal();
        let err = measure(&flipped, &base).expect_err("flipped chart should fail");
        let msg = err.to_string();
        assert!(
            msg.contains("--flip-h"),
            "error should suggest --flip-h: {msg}"
        );
    }

    #[test]
    #[expect(
        clippy::cast_precision_loss,
        reason = "u32 PRNG output normalized to 0..1; precision loss is intentional"
    )]
    fn random_noise_has_no_lattice() {
        let width = 400u32;
        let height = 300u32;
        let mut data = vec![0.0f32; width as usize * height as usize * 3];

        // Tiny deterministic xorshift32 PRNG; no dependency needed.
        let mut state = 0x1234_5678u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32) / (u32::MAX as f32)
        };

        for sample in &mut data {
            *sample = next();
        }
        let image = LinearImage::new(width, height, data).expect("well-formed");
        let base = FilmBase {
            transmittance: [0.8f32, 0.6f32, 0.4f32],
        };
        let err = measure(&image, &base).expect_err("noise should not form a lattice");
        assert!(
            matches!(err, ChartError::NoLattice { .. }),
            "expected NoLattice, got {err}"
        );
    }

    #[test]
    fn srgb_eotf_linearizes_the_endpoints() {
        // Black and white map to themselves-ish at the extremes.
        assert!(srgb_eotf(0.0) < 1e-6);
        assert!((srgb_eotf(1.0) - 1.0).abs() < 1e-6);
        // A mid-gray (sRGB 128) is below the linear midpoint.
        assert!(srgb_eotf(128.0 / 255.0) < 0.5);
    }
}
