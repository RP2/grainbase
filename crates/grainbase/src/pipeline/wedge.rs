//! Locating and measuring a photographed neutral gray wedge for rig calibration.
//!
//! A rig calibration is one measurement per camera + light table + process,
//! valid for every stock. The wedge carries only neutral steps, so it measures
//! the rig's per-channel response without confounding it with any film's dye
//! crosstalk. The resulting profile is therefore family-scoped (C-41, black and
//! white, or slide) rather than stock-scoped.
//!
//! Detection has two paths. A wedge whose steps are separated by dark lines
//! (which read as clear film on a negative) is located with the same 1-D lattice
//! machinery as [`chart`](super::chart). A printed gray scale whose steps abut —
//! the Kodak Q-13/Q-14 shape — has no such lines, so the frame is divided into
//! equal cells instead, which requires the strip to fill the exposure. Both
//! paths try a vertical layout (steps left-to-right) and then a horizontal one
//! (steps top-to-bottom), and the axis is chosen by whether the sampled steps
//! actually vary.
//!
//! Model under fit:
//!
//! ```text
//! per channel c:  d_c = m_c * D + b_c
//! D  = the step's nominal reflection density from the card
//! d_c = the step's measured density above base in channel c
//! ```
//!
//! `m_c` is negative on a negative: the lightest step (smallest D) is the
//! densest film. The fitted line is evaluated at the card's darkest step
//! (largest D) to get `density_floor[c]` and at the lightest step (smallest D)
//! to get `density_max[c]`. The card's own black and white points therefore pin
//! the channel's black and white points, and the per-channel range carries the
//! rig's gain.
//!
//! There is no gamma term: a neutral wedge gives one straight line per channel,
//! and any curvature would need a second capture to know its shape before it
//! could be corrected.

use std::fmt;

use crate::image::LinearImage;
use crate::pipeline::FilmBase;
use crate::pipeline::chart::{SAMPLE_INSET, SEPARATOR_BRIGHTNESS};
use crate::pipeline::chart::{Window, find_lines_1d, sample_window, separator_projections};
use crate::profile::{Method, Profile};

/// A registered gray-wedge card: named reflection-density steps plus citation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WedgeCard {
    /// Registry name used with `--wedge-card`.
    pub name: &'static str,
    /// Nominal reflection density of each step, lightest to darkest.
    pub steps: &'static [f32],
    /// Where the step values come from.
    pub cite: &'static str,
}

/// Kodak Gray Scale (Q-13 / Q-14) neutral wedge.
///
/// Kodak Gray Scale, Q-13 / Q-14 (Kodak Publication No. Q-14): 20 neutral steps
/// in nominal 0.10 reflection-density increments, white ~0.05 to black ~1.95.
const KODAK_GRAY_SCALE: WedgeCard = WedgeCard {
    name: "kodak-gray-scale",
    steps: &[
        0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45,
        1.55, 1.65, 1.75, 1.85, 1.95,
    ],
    cite: "Kodak Gray Scale, Q-13 / Q-14 (Kodak Publication No. Q-14): 20 neutral \
           steps in nominal 0.10 reflection-density increments, white ~0.05 to \
           black ~1.95.",
};

/// All wedge cards shipped with the engine.
const CARDS: &[WedgeCard] = &[KODAK_GRAY_SCALE];

/// Looks up a registered wedge card by name.
#[must_use]
pub fn card(name: &str) -> Option<&'static WedgeCard> {
    CARDS.iter().find(|card| card.name == name)
}

/// Names of all registered wedge cards, for error messages.
#[must_use]
pub fn card_names() -> Vec<&'static str> {
    CARDS.iter().map(|card| card.name).collect()
}

/// Which way the wedge steps run across the frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// Steps are vertical strips, read left to right.
    Vertical,
    /// Steps are horizontal strips, read top to bottom.
    Horizontal,
}

/// A measured gray wedge: per-step densities and the axis the detector won on.
#[derive(Debug, Clone, PartialEq)]
pub struct Measurement {
    /// Per-channel optical density above base for each step, in step order.
    pub cells: Vec<[f32; 3]>,
    /// Nominal reflection density for each step (same order as `cells`).
    pub steps: Vec<f32>,
    /// Which lattice orientation produced the cells.
    pub axis: Axis,
}

/// Smallest density spread across the sampled steps that means a wedge is
/// present at all, in log10 units.
///
/// The measured values are densities above base. A real neutral wedge spans a
/// large fraction of a density unit, while a flat field — or a frame so bright
/// that every pixel qualifies as a separator — spans nothing. Without this gate
/// the lattice finder happily divides a uniform field into equal cells and
/// "finds" a wedge that is not there.
const MIN_STEP_SPREAD: f32 = 0.05;

/// Smallest per-channel spread a fitted channel may have, in log10 units.
///
/// A channel that does not respond across the steps would emit a zero-height
/// density range, which the render cannot divide by. It is a measurement
/// failure, not a flat rig.
const MIN_CHANNEL_SPREAD: f32 = 1e-3;

/// Why a wedge could not be measured or fitted.
#[derive(Debug, Clone, PartialEq)]
pub enum WedgeError {
    /// The step list is too short to define a line.
    TooFewSteps {
        /// How many were supplied.
        got: usize,
    },
    /// The separator lattice could not be located confidently.
    NoLattice {
        /// What went wrong, in terms a user can act on.
        reason: String,
    },
    /// A step's internal samples are too spread out for a trustworthy average.
    StepSpread {
        /// Step index in step order.
        step: usize,
        /// Channel that failed the spread gate.
        channel: usize,
    },
    /// The fit collapsed (all nominal densities equal, or all measured
    /// densities equal).
    DegenerateFit,
}

impl fmt::Display for WedgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewSteps { got } => {
                write!(f, "a wedge needs at least 2 steps, got {got}")
            }
            Self::NoLattice { reason } => write!(
                f,
                "could not locate the wedge separators: {reason}. Check that the wedge fills the frame, is in focus, and is evenly lit."
            ),
            Self::StepSpread { step, channel } => write!(
                f,
                "step {step} has uneven samples in channel {channel}; the grid may be misaligned"
            ),
            Self::DegenerateFit => write!(
                f,
                "the wedge response is degenerate; the steps do not span a density range"
            ),
        }
    }
}

impl std::error::Error for WedgeError {}

/// Smallest image dimension the detector will attempt to grid.
const MIN_DIMENSION: u32 = 80;

/// Smallest span fraction the wedge separator search will accept.
///
/// The strip path deliberately requires the wedge to fill the frame, because
/// its perpendicular extent is the full frame and a strip that did not span it
/// would sample windows straddling the strip edge. This value preserves the
/// original `+/- 15%` step tolerance around a full-frame lattice.
const MIN_WEDGE_SPAN_FRACTION: f32 = 0.85;

/// Locates the wedge in `image` and measures per-step densities.
///
/// `base` is the measured film base for the uncropped scan. Densities are
/// computed as `-log10(mean_transmittance / base_transmittance)` per channel,
/// clamped to the same maximum used by the conversion core.
///
/// # Errors
///
/// Returns [`WedgeError::TooFewSteps`] when `steps` has fewer than two values,
/// [`WedgeError::NoLattice`] when the separator lattice cannot be found, or
/// [`WedgeError::StepSpread`] when a sample window is not uniform enough.
pub fn measure(
    image: &LinearImage,
    base: &FilmBase,
    steps: &[f32],
) -> Result<Measurement, WedgeError> {
    if steps.len() < 2 {
        return Err(WedgeError::TooFewSteps { got: steps.len() });
    }
    if image.width < MIN_DIMENSION || image.height < MIN_DIMENSION {
        return Err(WedgeError::NoLattice {
            reason: format!(
                "image is too small ({}x{}); a wedge capture should be much larger",
                image.width, image.height
            ),
        });
    }

    let base_min = base
        .transmittance
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min)
        .max(f32::MIN_POSITIVE);
    let threshold = base_min * SEPARATOR_BRIGHTNESS;

    let (vertical_proj, horizontal_proj) = separator_projections(image, threshold);

    // Each axis is a candidate. A lattice found on the wrong axis is a lattice
    // whose cells all cross the real steps, so the axis is selected by whether
    // the sampled steps actually vary — not by whether lines were found.
    for (projection, axis) in [
        (&vertical_proj, Axis::Vertical),
        (&horizontal_proj, Axis::Horizontal),
    ] {
        let Some(lines) = try_lattice(projection, steps.len(), axis) else {
            continue;
        };
        let cells = match axis {
            Axis::Vertical => sample_vertical_cells(image, base, &lines, steps.len()),
            Axis::Horizontal => sample_horizontal_cells(image, base, &lines, steps.len()),
        };
        let Ok(cells) = cells else {
            continue;
        };
        if !steps_differ(&cells) {
            continue;
        }
        return Ok(Measurement {
            cells,
            steps: steps.to_vec(),
            axis,
        });
    }

    // A printed gray scale has no dark separators between its steps: the patches
    // abut, and the only structure is the step change itself. When no separator
    // lattice is found, the frame is divided into equal cells instead — what a
    // strip framed to fill the exposure gives. The wrong axis is rejected by the
    // same test as the lattice: its cells would all span the real steps and so
    // would not differ.
    for axis in [Axis::Vertical, Axis::Horizontal] {
        let Ok(cells) = equal_cells(image, base, steps.len(), axis) else {
            continue;
        };
        if !steps_differ(&cells) {
            continue;
        }
        return Ok(Measurement {
            cells,
            steps: steps.to_vec(),
            axis,
        });
    }

    Err(WedgeError::NoLattice {
        reason: "no evenly spaced separator lattice or varying step sequence was found in \
                 either direction"
            .to_owned(),
    })
}

/// Divides the frame into `steps` equal cells, for a strip whose steps abut.
///
/// The strip must fill the frame: the cells are the frame divided equally, so a
/// margin shifts every sample. That is the documented capture requirement for a
/// contiguous printed card.
#[expect(
    clippy::cast_precision_loss,
    reason = "the axis length and step count are photograph-sized, well inside f32's \
              exact-integer range"
)]
fn equal_cells(
    image: &LinearImage,
    base: &FilmBase,
    steps: usize,
    axis: Axis,
) -> Result<Vec<[f32; 3]>, WedgeError> {
    let length = match axis {
        Axis::Vertical => image.width,
        Axis::Horizontal => image.height,
    };
    let lines: Vec<f32> = (0..=steps)
        .map(|index| length as f32 * index as f32 / steps as f32)
        .collect();

    match axis {
        Axis::Vertical => sample_vertical_cells(image, base, &lines, steps),
        Axis::Horizontal => sample_horizontal_cells(image, base, &lines, steps),
    }
}

/// Whether the sampled steps span a density range at all.
///
/// Even with a separator-contrast gate in the lattice finder, a uniform field
/// could still be divided into equal cells that all read the same density. The
/// gate that a real wedge passes and a flat field fails is that its steps
/// actually differ, so it is applied to the samples rather than to the
/// projection.
fn steps_differ(cells: &[[f32; 3]]) -> bool {
    let mut widest = 0.0f32;
    for channel in 0..3 {
        let mut low = f32::INFINITY;
        let mut high = f32::NEG_INFINITY;
        for cell in cells {
            low = low.min(cell[channel]);
            high = high.max(cell[channel]);
        }
        widest = widest.max(high - low);
    }

    widest > MIN_STEP_SPREAD
}

/// Tries to find `steps + 1` evenly spaced lines in one projection.
fn try_lattice(projection: &[f32], steps: usize, axis: Axis) -> Option<Vec<f32>> {
    let label = match axis {
        Axis::Vertical => "vertical",
        Axis::Horizontal => "horizontal",
    };
    find_lines_1d(projection, steps + 1, steps, MIN_WEDGE_SPAN_FRACTION)
        .map_err(|reason| format!("{label}: {reason}"))
        .ok()
}

/// Samples the central window of every vertical cell (left-to-right steps).
fn sample_vertical_cells(
    image: &LinearImage,
    base: &FilmBase,
    lines: &[f32],
    steps: usize,
) -> Result<Vec<[f32; 3]>, WedgeError> {
    if lines.len() != steps + 1 {
        return Err(WedgeError::NoLattice {
            reason: "wrong number of vertical separator lines".to_owned(),
        });
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "image dimensions are photograph-sized and fit exactly in f32"
    )]
    let height_f = image.height as f32;
    let mut cells = Vec::with_capacity(steps);

    for index in 0..steps {
        let x0 = lines[index];
        let x1 = lines[index + 1];
        let inset_x = (x1 - x0) * SAMPLE_INSET;
        let inset_y = height_f * SAMPLE_INSET;
        let window = Window {
            x0: x0 + inset_x,
            x1: x1 - inset_x,
            y0: inset_y,
            y1: height_f - inset_y,
        };

        cells.push(sample_window(image, base, &window).map_err(|channel| {
            WedgeError::StepSpread {
                step: index,
                channel,
            }
        })?);
    }
    Ok(cells)
}

/// Samples the central window of every horizontal cell (top-to-bottom steps).
fn sample_horizontal_cells(
    image: &LinearImage,
    base: &FilmBase,
    lines: &[f32],
    steps: usize,
) -> Result<Vec<[f32; 3]>, WedgeError> {
    if lines.len() != steps + 1 {
        return Err(WedgeError::NoLattice {
            reason: "wrong number of horizontal separator lines".to_owned(),
        });
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "image dimensions are photograph-sized and fit exactly in f32"
    )]
    let width_f = image.width as f32;
    let mut cells = Vec::with_capacity(steps);

    for index in 0..steps {
        let y0 = lines[index];
        let y1 = lines[index + 1];
        let inset_x = width_f * SAMPLE_INSET;
        let inset_y = (y1 - y0) * SAMPLE_INSET;
        let window = Window {
            x0: inset_x,
            x1: width_f - inset_x,
            y0: y0 + inset_y,
            y1: y1 - inset_y,
        };

        cells.push(sample_window(image, base, &window).map_err(|channel| {
            WedgeError::StepSpread {
                step: index,
                channel,
            }
        })?);
    }
    Ok(cells)
}

/// The result of fitting one wedge measurement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fit {
    /// Fitted density at the card's darkest step (largest reflection density).
    pub floor: [f32; 3],
    /// Fitted density at the card's lightest step (smallest reflection density).
    pub ceiling: [f32; 3],
    /// Fitted slope `m_c` per channel (`d_c = m_c * D + b_c`).
    pub slope: [f32; 3],
    /// Fitted intercept `b_c` per channel.
    pub intercept: [f32; 3],
    /// RMS residual of the fit per channel.
    pub residual: [f32; 3],
}

/// Fits the per-channel linear response from a wedge measurement.
///
/// # Errors
///
/// Returns [`WedgeError::DegenerateFit`] when the nominal densities do not span
/// a range or the measured densities do not.
pub fn fit(measurement: &Measurement) -> Result<Fit, WedgeError> {
    let n = measurement.steps.len();
    if n < 2 {
        return Err(WedgeError::TooFewSteps { got: n });
    }

    let min_d = measurement
        .steps
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let max_d = measurement
        .steps
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    if (max_d - min_d).abs() < f32::EPSILON {
        return Err(WedgeError::DegenerateFit);
    }

    let mut fit = Fit {
        floor: [0.0; 3],
        ceiling: [0.0; 3],
        slope: [0.0; 3],
        intercept: [0.0; 3],
        residual: [0.0; 3],
    };

    for channel in 0..3 {
        let d: Vec<f32> = measurement.cells.iter().map(|cell| cell[channel]).collect();
        let low = d.iter().copied().fold(f32::INFINITY, f32::min);
        let high = d.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if high - low <= MIN_CHANNEL_SPREAD {
            // A channel with no response would emit a zero-height density range,
            // which the render cannot divide by. Refuse rather than ship it.
            return Err(WedgeError::DegenerateFit);
        }
        let (m, b, residual) = fit_line(&measurement.steps, &d)?;
        fit.slope[channel] = m;
        fit.intercept[channel] = b;
        fit.floor[channel] = m * max_d + b;
        fit.ceiling[channel] = m * min_d + b;
        fit.residual[channel] = residual;
    }

    Ok(fit)
}

/// Least-squares line `density = slope * reflection_density + intercept` and
/// RMS residual.
#[expect(
    clippy::similar_names,
    reason = "sum_xx and sum_xy are the standard least-squares sums"
)]
fn fit_line(x: &[f32], y: &[f32]) -> Result<(f32, f32, f32), WedgeError> {
    let n = x.len();
    debug_assert_eq!(n, y.len());

    #[expect(
        clippy::cast_precision_loss,
        reason = "the step count is small and fits exactly in f32"
    )]
    let n_f = n as f32;
    let mean_x = x.iter().sum::<f32>() / n_f;
    let mean_y = y.iter().sum::<f32>() / n_f;

    let mut sum_xx = 0.0f32;
    let mut sum_xy = 0.0f32;
    for (&xi, &yi) in x.iter().zip(y) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        sum_xx += dx * dx;
        sum_xy += dx * dy;
    }

    if sum_xx.abs() < f32::EPSILON {
        return Err(WedgeError::DegenerateFit);
    }

    let slope = sum_xy / sum_xx;
    let intercept = mean_y - slope * mean_x;

    let residual = if n == 0 {
        0.0
    } else {
        let sum_sq: f32 = x
            .iter()
            .zip(y)
            .map(|(&xi, &yi)| {
                let predicted = slope * xi + intercept;
                (yi - predicted).powi(2)
            })
            .sum();
        (sum_sq / n_f).sqrt()
    };

    Ok((slope, intercept, residual))
}

/// Builds a rig-scoped profile from a wedge fit.
///
/// A rig calibration is not stock-specific, so `film` is empty. The crosstalk
/// matrix is identity because a neutral wedge carries no chromatic information.
///
/// # Errors
///
/// Propagates [`WedgeError::DegenerateFit`] from [`fit`] when the measurement
/// does not span a line.
pub fn profile(
    measurement: &Measurement,
    family: crate::pipeline::FilmMode,
    rig: &str,
) -> Result<Profile, WedgeError> {
    let fit = fit(measurement)?;
    Ok(Profile {
        film: String::new(),
        camera: rig.to_owned(),
        family: Some(family),
        method: Method::Fitted,
        frames: 1,
        settings: crate::pipeline::Settings {
            density_max: Some(fit.ceiling),
            density_floor: Some(fit.floor),
            gamma: None,
            pivot_density: None,
            crosstalk: crate::pipeline::IDENTITY,
        },
    })
}

/// Straightness ratio of the fit: residual divided by the fitted range.
///
/// A small ratio means the response is straight; a large ratio warns that a
/// single linear model is a poor description of the rig response.
#[must_use]
pub fn straightness_ratio(fit: &Fit, channel: usize) -> f32 {
    let range = (fit.ceiling[channel] - fit.floor[channel]).abs();
    if range < f32::EPSILON {
        0.0
    } else {
        fit.residual[channel] / range
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::LinearImage;
    use crate::pipeline::FilmBase;

    fn close(actual: f32, expected: f32, what: &str) {
        assert!(
            (actual - expected).abs() < 1e-3,
            "{what}: expected {expected}, got {actual}"
        );
    }

    /// Separator width used by the synthetic cards that have one.
    const SEPARATOR_SHARE: f32 = 0.06;

    /// Generates a synthetic negative of a vertical gray wedge.
    ///
    /// `floor` and `ceiling` are the per-channel densities at the darkest and
    /// lightest steps. `separator_share` is the width of the bright separator
    /// between steps as a fraction of a cell; zero makes the steps abut, which is
    /// the printed-gray-scale shape.
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "synthetic image dimensions are small and coordinates are clamped before truncation"
    )]
    fn synthetic_wedge(
        steps: &[f32],
        floor: [f32; 3],
        ceiling: [f32; 3],
        separator_share: f32,
    ) -> (LinearImage, FilmBase) {
        let width = 1200u32;
        let height = 800u32;
        let mut data = vec![0.0f32; width as usize * height as usize * 3];

        let base_transmittance = [0.8f32, 0.6f32, 0.4f32];
        let base = FilmBase {
            transmittance: base_transmittance,
        };

        let min_d = steps.iter().copied().fold(f32::INFINITY, f32::min);
        let max_d = steps.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let span_d = max_d - min_d;

        // Per-channel linear mapping from reflection density to measured density.
        // d = m * D + b, with d(min_d) = ceiling and d(max_d) = floor.
        let m = [
            (floor[0] - ceiling[0]) / span_d,
            (floor[1] - ceiling[1]) / span_d,
            (floor[2] - ceiling[2]) / span_d,
        ];
        let b = [
            ceiling[0] - m[0] * min_d,
            ceiling[1] - m[1] * min_d,
            ceiling[2] - m[2] * min_d,
        ];

        // Wedge fills the frame with a small bright margin.
        let margin_x = width as f32 / 24.0;
        let left = margin_x;
        let right = width as f32 - margin_x;
        let cell_w = (right - left) / steps.len() as f32;
        let sep_w = cell_w * separator_share;

        for y in 0..height as usize {
            for x in 0..width as usize {
                let xf = x as f32;
                let value = if xf < left || xf >= right {
                    base_transmittance
                } else {
                    let col = ((xf - left) / cell_w) as usize;
                    let col = col.min(steps.len() - 1);
                    let in_sep = (xf - left) - col as f32 * cell_w < sep_w
                        || (left + (col + 1) as f32 * cell_w) - xf < sep_w;

                    if in_sep {
                        base_transmittance
                    } else {
                        let d = [
                            m[0] * steps[col] + b[0],
                            m[1] * steps[col] + b[1],
                            m[2] * steps[col] + b[2],
                        ];
                        transmittance_from_density(d, base_transmittance)
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

    fn transmittance_from_density(density: [f32; 3], base: [f32; 3]) -> [f32; 3] {
        [
            base[0] * 10f32.powf(-density[0]),
            base[1] * 10f32.powf(-density[1]),
            base[2] * 10f32.powf(-density[2]),
        ]
    }

    #[test]
    fn kodak_gray_scale_has_twenty_steps() {
        let card = card("kodak-gray-scale").expect("registered card");
        assert_eq!(card.steps.len(), 20);
        assert!((card.steps[0] - 0.05).abs() < 1e-6);
        assert!((card.steps[19] - 1.95).abs() < 1e-6);
    }

    #[test]
    fn unknown_card_is_none_and_registry_lists_known_names() {
        assert!(card("no-such-card").is_none());
        let names = card_names();
        assert!(names.contains(&"kodak-gray-scale"));
    }

    #[test]
    fn synthetic_wedge_recovers_known_floor_and_ceiling() {
        let steps = card("kodak-gray-scale").unwrap().steps;
        let floor = [0.25f32, 0.30, 0.35];
        let ceiling = [1.80f32, 1.90, 2.00];
        let (image, base) = synthetic_wedge(steps, floor, ceiling, SEPARATOR_SHARE);

        let measurement = measure(&image, &base, steps).expect("wedge should be found");
        assert_eq!(measurement.cells.len(), steps.len());
        assert_eq!(measurement.axis, Axis::Vertical);

        let fit = fit(&measurement).expect("fit should succeed");
        for channel in 0..3 {
            close(fit.floor[channel], floor[channel], "floor");
            close(fit.ceiling[channel], ceiling[channel], "ceiling");
        }
    }

    #[test]
    fn lightest_step_maps_to_ceiling_and_darkest_to_floor() {
        let steps = &[0.1f32, 0.5, 1.0, 1.5];
        let floor = [0.2f32; 3];
        let ceiling = [2.0f32; 3];
        let (image, base) = synthetic_wedge(steps, floor, ceiling, SEPARATOR_SHARE);

        let measurement = measure(&image, &base, steps).expect("wedge should be found");
        let fit = fit(&measurement).expect("fit should succeed");

        for channel in 0..3 {
            assert!(
                fit.slope[channel] < 0.0,
                "slope should be negative on a negative, got {}",
                fit.slope[channel]
            );
            close(
                fit.floor[channel],
                floor[channel],
                "darkest step should be floor",
            );
            close(
                fit.ceiling[channel],
                ceiling[channel],
                "lightest step should be ceiling",
            );
        }
    }

    #[test]
    fn flat_field_has_no_lattice() {
        let width = 400u32;
        let height = 300u32;
        let data = vec![0.5f32; width as usize * height as usize * 3];
        let image = LinearImage::new(width, height, data).expect("well-formed");
        let base = FilmBase {
            transmittance: [0.8f32, 0.6f32, 0.4f32],
        };
        let steps = &[0.1f32, 0.5, 1.0];
        let err = measure(&image, &base, steps).expect_err("flat field should not form a lattice");
        assert!(
            matches!(err, WedgeError::NoLattice { .. }),
            "expected NoLattice, got {err}"
        );
    }

    #[test]
    fn profile_round_trips_with_family() {
        let steps = &[0.1f32, 0.5, 1.0, 1.5];
        let floor = [0.2f32; 3];
        let ceiling = [2.0f32; 3];
        let (image, base) = synthetic_wedge(steps, floor, ceiling, SEPARATOR_SHARE);
        let measurement = measure(&image, &base, steps).expect("wedge should be found");

        let profile = profile(
            &measurement,
            crate::pipeline::FilmMode::ColorNegative,
            "test-rig",
        )
        .expect("profile should build");
        let text = profile.to_string();
        let parsed = Profile::parse(&text).expect("profile should round-trip");
        assert_eq!(
            parsed.family,
            Some(crate::pipeline::FilmMode::ColorNegative)
        );
        assert_eq!(parsed.camera, "test-rig");
        assert_eq!(parsed.film, "");
    }

    #[test]
    fn too_few_steps_is_rejected() {
        let width = 400u32;
        let height = 300u32;
        let data = vec![0.5f32; width as usize * height as usize * 3];
        let image = LinearImage::new(width, height, data).expect("well-formed");
        let base = FilmBase {
            transmittance: [0.8f32, 0.6f32, 0.4f32],
        };
        let err = measure(&image, &base, &[0.5f32]).expect_err("one step is too few");
        assert!(matches!(err, WedgeError::TooFewSteps { got: 1 }));
    }

    #[test]
    fn a_horizontal_wedge_is_found_on_the_other_axis() {
        // The same synthetic wedge, transposed: the steps now run top to bottom,
        // so the vertical lattice must be rejected and the horizontal one used.
        let steps = &[0.1f32, 0.5, 1.0, 1.5];
        let floor = [0.2f32; 3];
        let ceiling = [2.0f32; 3];
        let (image, base) = synthetic_wedge(steps, floor, ceiling, SEPARATOR_SHARE);
        let transposed = image.transposed();

        let measurement = measure(&transposed, &base, steps).expect("wedge should be found");
        assert_eq!(measurement.axis, Axis::Horizontal);

        let fit = fit(&measurement).expect("fit should succeed");
        for channel in 0..3 {
            close(fit.floor[channel], floor[channel], "floor");
            close(fit.ceiling[channel], ceiling[channel], "ceiling");
        }
    }

    #[test]
    fn a_channel_with_no_response_is_a_degenerate_fit() {
        // The steps span nominally, but the measurement does not: emitting this
        // would give the channel a zero-height density range.
        let measurement = Measurement {
            cells: vec![[0.5f32; 3]; 3],
            steps: vec![0.1, 0.5, 1.0],
            axis: Axis::Vertical,
        };
        assert_eq!(fit(&measurement), Err(WedgeError::DegenerateFit));
    }

    #[test]
    fn a_contiguous_strip_is_divided_equally() {
        // A printed gray scale (Kodak Q-13/Q-14) has no dark separators between
        // its steps: the patches abut. The equal-division path is the one that
        // finds it, so it is the path a real card exercises.
        let steps = &[0.1f32, 0.5, 1.0, 1.5];
        let floor = [0.2f32; 3];
        let ceiling = [2.0f32; 3];
        let (image, base) = synthetic_wedge(steps, floor, ceiling, 0.0);

        let measurement = measure(&image, &base, steps).expect("contiguous strip should be found");
        assert_eq!(measurement.axis, Axis::Vertical);

        let fit = fit(&measurement).expect("fit should succeed");
        for channel in 0..3 {
            close(fit.floor[channel], floor[channel], "floor");
            close(fit.ceiling[channel], ceiling[channel], "ceiling");
        }
    }
}
