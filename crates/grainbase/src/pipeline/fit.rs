//! Fitting a calibration profile from a color-target capture.
//!
//! The capture protocol (maintainer's docs, sibling repo) defines the input:
//! one scanned frame of a ColorChecker-class panel, where a neutral dark and
//! a neutral light patch anchor the per-channel density range and the color
//! patches identify the dye crosstalk. Everything here is pure: patches in,
//! [`Settings`] out, and both stages are closed-form solves — there is no
//! optimizer to tune, so results are deterministic by construction.
//!
//! The fitter is generic. It knows nothing about any stock's numbers, and no
//! collected measurement lives here: the measurement corpus is data owned
//! outside this tree; a fitted profile file is only its visible artifact.
//!
//! Model under fit:
//!
//! ```text
//! corrected    = crosstalk * density          // per pixel
//! positive     = (corrected - floor) / (ceiling - floor)
//! ```
//!
//! `floor`/`ceiling` solve per channel from the two neutrals' measured
//! densities and intended print tones; `crosstalk` solves per row with the
//! diagonal held at `1.0` once the anchors are fixed.

use super::{IDENTITY, Settings};

/// The intended print value of the dark neutral patch.
///
/// Prints do not put true black at their darkest reachable sample: the
/// standard observation is that the darkest chart neutral sits just above
/// the paper's own floor, which leaves it a tenth of the way up the display
/// range.
pub const DARK_PRINT_TONE: f32 = 0.1;

/// The intended print value of the light neutral patch.
///
/// Symmetric to [`DARK_PRINT_TONE`]: the lightest chart neutral sits just
/// under the paper's shoulder, never at a clipped white.
pub const LIGHT_PRINT_TONE: f32 = 0.9;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FitError {
    /// The two anchor neutrals measure the same density in a channel, or the
    /// intended tones coincide: no range exists to fit.
    DegenerateAnchors,
    /// The color patches do not span the mixing plane: not enough
    /// independent patches identify the crosstalk row.
    UnderdeterminedCrosstalk,
}

/// One channel's density anchors: the black and white points, as absolute
/// densities above the film base.
///
/// The pair is the crossover correction — a channel's measured densities get
/// their own black and white points here, which absorbs the dye layer's
/// log-response difference directly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Anchors {
    pub floor: f32,
    pub ceiling: f32,
}

/// Fits one channel's anchors.
///
/// `dark`/`light` are the channel's measured densities for the chart's dark
/// and light neutral patches; `dark_tone`/`light_tone` are the print tones
/// they should occupy, between 0 and 1.
///
/// # Errors
///
/// [`FitError::DegenerateAnchors`] when the neutrals measure equal densities
/// or the intended tones coincide.
pub fn fit_anchor(
    dark: f32,
    light: f32,
    dark_tone: f32,
    light_tone: f32,
) -> Result<Anchors, FitError> {
    let tone_span = light_tone - dark_tone;
    let density_span = light - dark;
    if tone_span.abs() < f32::EPSILON || density_span.abs() < f32::EPSILON {
        return Err(FitError::DegenerateAnchors);
    }

    // positive = (x - floor)/range with the two neutrals pinned:
    //   light - dark = (light_tone - dark_tone) * range
    //   floor = dark - dark_tone * range
    let range = density_span / tone_span;
    let floor = dark - dark_tone * range;
    Ok(Anchors {
        floor,
        ceiling: floor + range,
    })
}

/// Fits all three channels' anchors from the two neutrals' density vectors.
///
/// # Errors
///
/// [`FitError::DegenerateAnchors`] when any channel is degenerate.
pub fn fit_anchors(dark: [f32; 3], light: [f32; 3]) -> Result<[Anchors; 3], FitError> {
    Ok([
        fit_anchor(dark[0], light[0], DARK_PRINT_TONE, LIGHT_PRINT_TONE)?,
        fit_anchor(dark[1], light[1], DARK_PRINT_TONE, LIGHT_PRINT_TONE)?,
        fit_anchor(dark[2], light[2], DARK_PRINT_TONE, LIGHT_PRINT_TONE)?,
    ])
}

/// Fits one crosstalk row's two off-diagonals by least squares.
///
/// For an output channel, the other two dye lines mix toward the target:
/// each patch gives one linear equation `x*a + y*b = r`, with `r` the
/// residual after the diagonal's own contribution. `x`/`y` are the normal
/// equations of the two-variable least squares, closed form; returns `None`
/// when the patch densities collapse rank (no spanning patch set).
fn fit_row(one: &[(f32, f32, f32)]) -> Option<(f32, f32)> {
    let (mut saa, mut sab, mut sbb) = (0.0f32, 0.0f32, 0.0f32);
    let (mut sar, mut sbr) = (0.0f32, 0.0f32);
    for &(a, b, r) in one {
        saa += a * a;
        sab += a * b;
        sbb += b * b;
        sar += a * r;
        sbr += b * r;
    }

    let det = saa * sbb - sab * sab;
    if det.abs() <= f32::EPSILON * (saa.abs() + sbb.abs()) {
        return None;
    }
    Some(((sbb * sar - sab * sbr) / det, (saa * sbr - sab * sar) / det))
}

/// Fits the crosstalk matrix given fixed anchors plus the color patches.
///
/// `patch` pairs a patch's measured density with its intended print tones
/// (per channel, from the checker's known reflectance targets). Each matrix
/// row solves independently with the diagonal held at 1.0; the target is
/// brought to the mixing space the same way the render maps out of it.
///
/// # Errors
///
/// [`FitError::UnderdeterminedCrosstalk`] when any row is rank-deficient,
/// which happens whenever the patch set is empty or does not span both other
/// dye columns.
///
/// # Panics
///
/// Panics if `anchors` is not exactly three channels or a patch's density is
/// not three samples: a type error in the caller, not bad data.
pub fn fit_crosstalk(
    anchors: &[Anchors; 3],
    patch: &[([f32; 3], [f32; 3])],
) -> Result<[[f32; 3]; 3], FitError> {
    assert!(anchors.len() == 3, "three channels under fit");
    assert!(
        !patch.iter().any(|(density, _)| density.len() < 3),
        "patches carry three densities"
    );
    if patch.is_empty() {
        return Err(FitError::UnderdeterminedCrosstalk);
    }

    let mut matrix = IDENTITY;
    for output in 0..3 {
        // Excluding the output dye's own density, the other two mix toward
        // the residual after holding the diagonal.
        let mut equations: Vec<(f32, f32, f32)> = Vec::with_capacity(patch.len());
        for (density, print) in patch {
            // (x) pairs with the first other-daughter dye column in ascending
            // order, (y) with the second.
            // The two other-dye densities, in ascending column order.
            let mut columns = (0..3).filter(|col| *col != output);
            let (a, b) = (
                density[columns.next().unwrap()],
                density[columns.next().unwrap()],
            );
            let anchor = anchors[output];
            let target = print[output] * (anchor.ceiling - anchor.floor) + anchor.floor;
            equations.push((a, b, target - density[output]));
        }
        let (x, y) = fit_row(&equations).ok_or(FitError::UnderdeterminedCrosstalk)?;

        // Writing the row back: the same two columns the equations read.
        let mut columns = (0..3).filter(|col| *col != output);
        matrix[output][columns.next().unwrap()] = x;
        matrix[output][columns.next().unwrap()] = y;
    }
    Ok(matrix)
}

/// Composes both stages into the settings a fitted profile carries.
///
/// # Errors
///
/// Both stages' errors propagate.
pub fn fit_profile(
    dark: [f32; 3],
    light: [f32; 3],
    patch: &[([f32; 3], [f32; 3])],
) -> Result<Settings, FitError> {
    let anchors = fit_anchors(dark, light)?;
    let crosstalk = fit_crosstalk(&anchors, patch)?;

    Ok(Settings {
        density_max: Some(anchors.map(|anchor| anchor.ceiling)),
        density_floor: Some(anchors.map(|anchor| anchor.floor)),
        gamma: None,
        pivot_density: None,
        crosstalk,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f32, expected: f32, what: &str) {
        assert!(
            (actual - expected).abs() < 1e-4,
            "{what}: expected {expected}, got {actual}"
        );
    }

    #[test]
    fn a_channel_anchor_fits_its_range() {
        // Ground truth chosen, neutrals computed from it, fit must recover it.
        let (floor, range) = (0.25f32, 2.0);
        let dark = floor + DARK_PRINT_TONE * range;
        let light = floor + LIGHT_PRINT_TONE * range;

        let fit = fit_anchor(dark, light, DARK_PRINT_TONE, LIGHT_PRINT_TONE).expect("range exists");
        close(fit.floor, floor, "black point");
        close(fit.ceiling - fit.floor, range, "density range");
    }

    #[test]
    fn degenerate_anchors_are_errors_not_zero_fits() {
        assert_eq!(
            fit_anchor(0.5, 0.5, DARK_PRINT_TONE, LIGHT_PRINT_TONE),
            Err(FitError::DegenerateAnchors)
        );
        assert_eq!(
            fit_anchor(0.3, 0.9, 0.5, 0.5),
            Err(FitError::DegenerateAnchors)
        );
    }

    #[test]
    fn crosstalk_recovers_a_known_gallery_row() {
        // Known mixing with a unit diagonal. Patches generated exactly from
        // the model; the fitter must recover the chosen rows.
        let chosen = [[1.0, 0.05, 0.02], [0.03, 1.0, 0.04], [0.01, 0.02, 1.0]];
        let (floor, range) = (0.25f32, 2.0);
        let anchor = Anchors {
            floor,
            ceiling: floor + range,
        };
        let anchors = [anchor; 3];

        let densities = [[0.50, 0.29, 0.20], [0.29, 0.60, 0.30], [0.21, 0.24, 0.55]];
        let patch: Vec<([f32; 3], [f32; 3])> = densities
            .iter()
            .map(|d| {
                let mut print = [0.0f32; 3];
                for o in 0..3 {
                    let corrected = chosen[o][0] * d[0] + chosen[o][1] * d[1] + chosen[o][2] * d[2];
                    print[o] = (corrected - floor) / range;
                }
                (*d, print)
            })
            .collect();

        let fit = fit_crosstalk(&anchors, &patch).expect("patches span the plane");
        for (row, expected_row) in chosen.iter().enumerate() {
            for (col, expected) in expected_row.iter().enumerate() {
                close(fit[row][col], *expected, "mixing recovered");
            }
        }
    }

    #[test]
    fn an_empty_patch_set_is_underetermined_not_silent() {
        let anchor = Anchors {
            floor: 0.25,
            ceiling: 2.25,
        };
        assert_eq!(
            fit_crosstalk(&[anchor; 3], &[]),
            Err(FitError::UnderdeterminedCrosstalk)
        );
    }

    #[test]
    fn fit_profile_composes_both_stages() {
        let chosen = [[1.0, 0.05, 0.02], [0.03, 1.0, 0.04], [0.01, 0.02, 1.0]];
        let (floor, range) = (0.25f32, 2.0);
        let densities = [[0.50, 0.29, 0.20], [0.29, 0.60, 0.30], [0.21, 0.24, 0.55]];
        // The two anchors: dark and light neutrals measured in a channel.
        let dark = [floor + DARK_PRINT_TONE * range; 3];
        let light = [floor + LIGHT_PRINT_TONE * range; 3];
        let patch: Vec<([f32; 3], [f32; 3])> = densities
            .iter()
            .map(|d| {
                let mut render = [0.0f32; 3];
                for o in 0..3 {
                    let corrected = chosen[o][0] * d[0] + chosen[o][1] * d[1] + chosen[o][2] * d[2];
                    render[o] = (corrected - floor) / range;
                }
                (*d, render)
            })
            .collect();

        let settings = fit_profile(dark, light, &patch).expect("nondegenerate");
        let floors = settings.density_floor.expect("a fit emits floors");
        let ceilings = settings.density_max.expect("a fit emits ceilings");
        for (i, actual_floor) in floors.iter().enumerate() {
            close(*actual_floor, floor, &format!("channel {i} black point"));
        }
        for (i, actual_ceiling) in ceilings.iter().enumerate() {
            close(
                *actual_ceiling,
                floor + range,
                &format!("channel {i} white point"),
            );
        }
        for (row, expected_row) in chosen.iter().enumerate() {
            for (col, expected) in expected_row.iter().enumerate() {
                close(settings.crosstalk[row][col], *expected, "mixing recovered");
            }
        }
    }
}
