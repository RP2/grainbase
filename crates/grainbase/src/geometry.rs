//! Finding the film frame in a scan: where it is, and how far it is tilted.
//!
//! Home scans are rarely straight, so a crop has to know the frame's angle as
//! well as its position: an axis-aligned box around a tilted frame either clips
//! the picture or keeps the rebate in the corners.
//!
//! The frame is found from the film's own mechanics first. The sprocket-hole
//! bands are parts of the strip — bare light through punched holes, reading
//! several times brighter than anything the film itself can show — and their
//! projection pins down three things at once: the strip's tilt, the rows the
//! picture may occupy, and the rebate margin around it. Hole rows never enter
//! the crop; the picture is never cut deeper than the rebate search allows.
//! The film's 3:2 aspect supplies the width, and the content's centre
//! positions it.
//!
//! Levels cannot do this alone. On an underexposed frame the rebate and the
//! picture sit at the same level; on a scan with bright content the picture
//! and the rebate trade places. Three per-pixel threshold rounds failed on
//! real scans exactly there before the bands took over the structural work
//! (measured: dark content at 0.145 against a base of 0.176; pale walls at
//! rebate level). What levels still do well is *local* verification — a step
//! across a boundary, a bright fraction in one row — and every threshold in
//! this module is a geometry bound or a contrast *difference* anchored to the
//! measured clear-film level, never an absolute brightness cut.
//!
//! When no hole band is visible (band-less strips), a line pipeline takes
//! over: strong straight edges are fit as near-vertical and near-horizontal
//! lines, and the frame is the rectangle whose four edges consistently read
//! clear film outside and picture inside. It is the fallback, not the
//! structure.
//!
//! This module is pure: image in, geometry out, no I/O.

use crate::image::LinearImage;
use crate::pipeline::{FilmBase, film_base};

use std::cmp::Reverse;

/// Largest tilt treated as genuine. Beyond this it is not a crooked scan, it is
/// a frame found wrongly, and rotating by it would wreck the picture.
const MAX_SKEW: f32 = 0.20; // radians, about 11.5 degrees

/// Target long side for the working image.
///
/// The real scan is 6000×4000. Scaling the longest side to about 1400 samples
/// throws away grain noise while leaving frame edges hundreds of pixels long.
const WORKING_LONG_SIDE: u32 = 1400;

/// Which gradient-magnitude percentile a candidate edge pixel must reach.
///
/// p90 of gradient magnitude: adaptive, no absolute level. High percentiles
/// put the sprocket-hole transitions above the bar and shut out the weaker —
/// and only frame-relevant — rebate step (p99 did, in a round the maintainer
/// saw as the crop keeping the sprocket band; the rebate step measures a few
/// times under the hole edges, but well above grain).
const GRADIENT_PERCENTILE: f32 = 0.90;

/// How bright a sample must be to read as bare light, as a multiple of the
/// clear-film base.
///
/// The light table is run neutral (~5500K), so bare light through the holes
/// is white and its least channel is high — but the hole-to-base *ratio*
/// still moves between scans (measured 3.4x to 10x) because the base
/// measurement itself shifts with the roll's content and the light's falloff
/// at the frame edges. A bar derived from one scan's ratio silently disables
/// the anchors on the next. Content never gets near the line: exposed film
/// absorbs in every layer, the rebate is the ceiling at roughly 1.2x the
/// base, and the 2x bar keeps clear water between holes and anything the
/// film itself can show.
const HOLE_BRIGHTNESS: f32 = 2.0;

/// The share of a row's columns that must read as bare light for the row to
/// belong to a hole band.
///
/// About half the columns are holes on band rows, but a band's outer rows
/// taper with tilt and blur; 8% keeps the whole band while never letting a
/// row of rebate between two holes qualify on its own.
const HOLE_ROW_FRACTION: f32 = 0.08;

/// How many dark rows a hole band may straddle in the projection.
///
/// The band's rows stay within a couple of pixels of one straight line: tilt
/// shifts the hole edges across the width by a few rows, never by a pitch.
const BAND_GAP_TOLERANCE: usize = 3;

/// The least height, in grid rows, for one hole band.
const MIN_BAND_ROWS: usize = 3;

/// How far the picture may sit from the hole band it faces, as a fraction of
/// the hole-to-hole span.
///
/// Measured at 2.7% on the maintainer's reference scan, whose rebate is a
/// narrow margin between picture and perforation band. 10% leaves room for
/// other stocks and framing; more than this and a candidate row range is not
/// a rebate, it is somewhere else in the picture.
const REBATE_MAX_FRACTION: f32 = 0.10;

/// Where the picture is placed when its rebate edge cannot be measured, as a
/// multiple of its hole band's height.
///
/// The reference scan measures 1.1 band heights from the band edge to the
/// picture. A rebate invisible under dark content is placed there: it keeps a
/// strip of rebate rather than risking the picture, and the bound is
/// deliberate — hole rows stay out, which is the one failure the maintainer
/// ruled out, and the picture is never cut by more than the rebate search
/// allows.
const REBATE_DEFAULT_FRACTION: f32 = 1.1;

/// Least support, as a share of the grid width, for a rebate edge to count as
/// measured.
///
/// Rebate is clear film — brighter than any exposed area — but only where
/// exposed content actually sits below the band. Invisible stretches
/// contribute nothing, and a quarter of the width rules out a scratch.
const REBATE_SUPPORT_MIN: f32 = 0.25;

/// Smallest step at the rebate→picture boundary, as a fraction of the base.
///
/// Rebate is clear film, brighter than any exposed area by at least the
/// film's color-mask absorption. Measured on dense content the step is a few
/// percent of the base; 4% clears grain noise (about 2% on these scans)
/// while staying far below a bright content seam.
const REBATE_STEP_MIN: f32 = 0.04;

/// Width over height of a full 35mm frame.
///
/// Bounding, not asserting: it keeps a candidate rectangle from chaining
/// across an inter-frame gap into the next frame, where a level-based edge
/// gate alone would keep walking.
const FRAME_ASPECT: f32 = 1.5;

/// The content level that separates exposed picture from rebate, as a
/// fraction of the base, when scanning a row for the picture's horizontal
/// extent.
///
/// An underexposed scan puts dark content arbitrarily close to the rebate
/// level, so the bar sits just under the base: anything meaningfully darker
/// than the rebate is picture. Rebate that reads darker than the base then
/// qualifies too — accepted, because the fallback placement keeps those
/// strips in the crop, which is the agreed trade — and the median extent
/// bounds the damage. Bright pixels inside the picture never end a run;
/// only rebate-level brightness on both sides does.
const CONTENT_LEVEL_FRACTION: f32 = 0.97;

/// Largest bright gap tolerated inside one content run, in grid columns.
///
/// The picture holds highlights, but never rebate-sized ones: a pitch of
/// dark rebate between two holes does not merge a run across it.
const CONTENT_GAP_TOLERANCE: usize = 6;

/// Angular resolution of the Hough accumulator, in radians.
///
/// 0.25 degrees is fine enough to resolve small tilt while keeping the
/// accumulator small.
const ANGLE_STEP: f32 = 0.004_363_323; // 0.25 degrees

/// Offset resolution of the Hough accumulator, in downsampled pixels.
///
/// One pixel matches the working-grid precision after the box downsample; any
/// coarser and the least-squares refinement would start too far from the edge.
const OFFSET_STEP: f32 = 1.0;

/// Maximum slope searched for either family of lines.
///
/// `tan(MAX_SKEW)`; slopes outside this range would imply a tilt larger than the
/// frame-level gate.
const MAX_LINE_SLOPE: f32 = 0.204_969_94; // tan(0.20)

/// Minimum votes in a Hough cell for it to be considered a peak.
///
/// Keeps accidental alignments of a handful of noise pixels from becoming line
/// candidates. Real edges produce dozens to hundreds of votes.
const MIN_VOTES: u32 = 5;

/// Half-width of the Hough peak-suppression window, in offset bins.
///
/// Candidate pixels of one real edge land in several adjacent offset bins; a
/// one-real-edge-per-region window stops the same edge from surfacing as
/// multiple peaks and eating the per-family peak budget.
const NMS_OFFSET_RADIUS: usize = 6;

/// Half-width of the Hough peak-suppression window, in angle bins.
///
/// With 0.25-degree bins this is one degree: the least-squares refinement
/// moves a fitted slope by several bins from its Hough bin, and without the
/// wider window the duplicates crowd out genuine lines.
const NMS_ANGLE_RADIUS: usize = 4;

/// Maximum number of peaks to refine per family.
///
/// A frame has four edges; keeping extras lets the rectangle stage choose
/// between overlapping candidates, and at the p90 candidate bar plenty of
/// picture-content edges compete for the same slots.
const MAX_PEAKS_PER_FAMILY: usize = 24;

/// Offset tolerance for treating two fitted lines as the same line, in
/// downsampled pixels. Tied Hough bins refine to one line; this drops the
/// remeasure.
const DUPLICATE_OFFSET: f32 = 3.0;

/// The same for slope, in the accumulator's slope units.
const DUPLICATE_SLOPE: f32 = 0.01;

/// How far from a frame's edge the image centre may sit and still count as
/// inside, in source pixels.
///
/// Containment is a tie-break on equal scores; a centre coincident with an
/// edge belongs to the rectangle it bounds. The length of a set of real
/// transitions is small against frame sizes, so a couple of pixels of drift
/// are nothing.
const CENTER_TOLERANCE: f32 = 2.0;

/// Maximum perpendicular distance, in downsampled pixels, for a candidate pixel
/// to be an inlier of a fitted line.
const INLIER_DISTANCE: f32 = 2.5;

/// Maximum standard deviation of perpendicular residuals for a line to be
/// considered straight.
///
/// Measured on synthetic data a perfect edge is sub-pixel; 1.5 px allows for
/// grain, optical blur, and the box downsample while rejecting ragged scratches.
const MAX_RESIDUAL_STD: f32 = 1.5;

/// Minimum fraction of the line's own extent that inliers must cover.
///
/// Not a fraction of the image axis: a portrait frame in a wide sensor area —
/// film stock and neighbour frames all around — has complete edges that span
/// only part of one axis, so a threshold there throws away real frames. This
/// gate only has to kill short scattered candidates: a strip of transition
/// pixels with big holes along its run. Anything that survives this but is not
/// a frame edge dies later, at the per-edge contrast gate in the rectangle
/// assembly, which is where edge-vs-not is actually decidable.
const MIN_SPAN_RATIO: f32 = 0.35;

/// Minimum number of inliers for a line.
///
/// A real frame edge is hundreds of pixels long; 20 is a small fraction of that
/// but is still enough for the synthetic tests.
const MIN_LINE_INLIERS: usize = 20;

/// Maximum angular disagreement between one edge and the median skew of a
/// candidate rectangle, in radians.
///
/// Opposite edges of a rectangle share one tilt; this rejects inconsistent
/// pairings of spurious lines.
const MAX_SKEW_DISAGREEMENT: f32 = 0.05;

/// Smallest dimension a found frame may have, as a fraction of the source image.
///
/// Generous: a 20% floor still passes the bounded-frame synthetic tests and
/// rejects tiny accidental rectangles formed by scratches.
const MIN_FRAME_EXTENT: f32 = 0.20;

/// How much the measured rectangle is pulled in, per side, in working pixels.
///
/// The fitted line runs through the middle of the transition between picture
/// and clear film, and the crop expands outward from it (floor on one side,
/// ceiling on the other). Pulled in half a sample beyond that expansion, the
/// axis-aligned crop holds picture pixels only — and everywhere else the soft,
/// half-film samples at the border stay out of the result. The maintainer has
/// ruled that keeping rebate looks worse than shaving a boundary sliver of the
/// picture, so the error here is deliberately toward cutting in.
const BOUNDARY_INSET: f32 = 0.7;

/// Width of the score band in which two rectangles count as equal.
///
/// Finding this frame and finding the *next* frame past the inter-frame gap are
/// locally indistinguishable: both are rectangles bounded by clear film on all
/// four sides, with the same edge contrasts. Differences below the grain-noise
/// floor are not evidence, so within the band the larger rectangle wins: the
/// operator frames the current frame almost fully, while an equally-scored
/// neighbour is the exceptional case.
const SCORE_TIE: f32 = 0.004;

/// Center offset of the contrast sampling band, in downsampled pixels.
///
/// Skips the smeared boundary; far enough to get a clean local mean, close
/// enough not to cross another edge.
const BAND_OFFSET: f32 = 8.0;

/// Half-width of the contrast band, in downsampled pixels.
///
/// The band is the mean of the two offsets `BAND_OFFSET ± BAND_HALF_WIDTH`.
const BAND_HALF_WIDTH: f32 = 2.0;

/// Which quantile of an edge's band samples the gates read.
///
/// An edge's outside band is not uniform along its length: where the rebate
/// narrows to nothing, sprocket holes sit directly above the picture. A mean
/// over the band would smear that in; a 60th-percentile ignores up to two
/// fifths of polluted samples while staying representative.
const BAND_QUANTILE: f32 = 0.60;

/// Minimum brightness step across a frame edge, in least-channel units.
///
/// The real picture/clear-film step is 0.02–0.03 and grain noise is about 0.002.
/// A fifth of the step (0.006) is therefore a bright-side-outside gate that
/// rejects interior tonal edges while accepting the real boundary.
const MIN_CONTRAST_STEP: f32 = 0.006;

/// Step between samples along an edge when measuring contrast, in downsampled
/// pixels.
const CONTRAST_SAMPLE_STEP: f32 = 2.0;

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

/// The film frame: a rectangle, possibly rotated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame {
    /// Centre of the frame, in source pixels.
    pub center: [f32; 2],
    /// Width and height of the frame, in source pixels. Rotation does not change
    /// these, so they can be used after deskewing.
    pub size: [f32; 2],
    /// How far the frame is rotated, in radians. Rotating the image by the
    /// negative of this makes the frame axis-aligned.
    pub skew: f32,
}

impl Frame {
    /// The whole image as a frame: no crop, no rotation.
    ///
    /// The answer whenever the frame cannot be located. Deliberately not an
    /// error: a bad crop is worse than no crop, so the fallback changes nothing.
    #[must_use]
    pub fn whole(image: &LinearImage) -> Self {
        let size = [
            coordinate(image.width as usize),
            coordinate(image.height as usize),
        ];
        Self {
            center: [size[0] / 2.0, size[1] / 2.0],
            size,
            skew: 0.0,
        }
    }

    /// The frame's color-sampling region: the measured frame pulled in by
    /// fractions of its own size, with the frame's skew carried through so a
    /// caller samples the deskewed region's corners correctly.
    ///
    /// The vertical inset is lighter than the horizontal one: the bands anchor
    /// the frame's height mechanically, while the horizontal sides are the
    /// content-derived approximations (the width comes from the film's aspect
    /// and the content's centre) and carry the greater error. Everything that
    /// *measures color* — calibration fitting, quality checks — samples inside
    /// this region rather than the frame at large, where rebate and film-edge
    /// dark bands sit.
    #[must_use]
    pub fn sample_region(self) -> Self {
        Self {
            center: self.center,
            size: [
                self.size[0] * (1.0 - 2.0 * SAMPLE_INSET_WIDTH),
                self.size[1] * (1.0 - 2.0 * SAMPLE_INSET_HEIGHT),
            ],
            skew: self.skew,
        }
    }
}

/// The color-sampling region's horizontal inset, as a fraction of the frame's
/// width per side.
///
/// The frame's horizontal sides are approximations: their width comes from the
/// film's aspect and the content's centre, not from four measured edges, so
/// 20% per side keeps the sample away from the parts most likely to hold
/// rebate or an inter-frame edge.
const SAMPLE_INSET_WIDTH: f32 = 0.20;

/// The color-sampling region's vertical inset, as a fraction of the frame's
/// height per side.
///
/// Mechanically anchored to the sprocket bands, the frame's height is
/// trustworthy; the 10% inset only trims the rebate the fallback placement
/// may keep and the rounded film corners the maintainer trims by hand.
const SAMPLE_INSET_HEIGHT: f32 = 0.10;

/// Finds the film frame, or `None` when the image has no pixels.
///
/// Falls back to [`Frame::whole`] whenever there is nothing to measure: no
/// strong straight edges, or no rectangle whose edges agree on a tilt and a
/// contrast sign.
#[must_use]
pub fn find(image: &LinearImage) -> Option<Frame> {
    let width = image.width as usize;
    let height = image.height as usize;
    if width == 0 || height == 0 {
        return None;
    }

    let (gray, scale) = downsample(image);

    // The clear-film level, for the per-edge gates in the assembly. Reuses the
    // film-base detector rather than measuring brightness here: the base *is*
    // the clear-film level, and that detector already knows to ignore the
    // sprocket holes, which are bare light rather than film.
    let detected = FilmBase::detect(image, film_base::DEFAULT_PERCENTILE);
    let base = detected
        .transmittance
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    if !base.is_finite() || base <= 0.0 {
        // Without a measurable clear-film level there is no frame to find.
        return Some(Frame::whole(image));
    }

    let src_w = coordinate(width);
    let src_h = coordinate(height);

    // Hole bands straighten the grid and anchor the frame mechanically.
    if let Some(coarse) = detect_bands(&gray, base).map(|bands| bands.skew) {
        let rotated = rotate_gray(&gray, -coarse);
        let anchored = detect_bands(&rotated, base)
            .and_then(|bands| band_frame(&rotated, scale, base, &bands));
        if let Some(second) = anchored {
            let skew = coarse + second.skew;
            if skew.abs() <= MAX_SKEW {
                return Some(map_back(second, coarse, src_w, src_h));
            }
        }
        // The bands were visible but their frame came out unusable: try the
        // content-based detector on the same straightened grid before
        // giving up on the crop entirely.
        let still = detect_frame(&rotated, scale, image.width, image.height, base);
        if let Some(second) = still {
            let skew = coarse + second.skew;
            if skew.abs() <= MAX_SKEW {
                return Some(map_back(second, coarse, src_w, src_h));
            }
        }
        return Some(Frame::whole(image));
    }

    // No hole bands: the band-less line pipeline, two passes, as before.
    let Some(first) = detect_frame(&gray, scale, image.width, image.height, base) else {
        return Some(Frame::whole(image));
    };

    // Straighten the analysis grid and detect again. Sizes are rotation
    // invariant; only the centre needs mapping back.
    let rotated = rotate_gray(&gray, -first.skew);
    let Some(second) = detect_frame(&rotated, scale, image.width, image.height, base) else {
        return Some(first);
    };

    let skew = first.skew + second.skew;
    if skew.abs() > MAX_SKEW {
        return Some(Frame::whole(image));
    }
    Some(map_back(second, first.skew, src_w, src_h))
}

/// Maps a frame measured on a straightened grid back into source
/// coordinates, composing the straightening tilt into `Frame::skew`.
fn map_back(frame: Frame, coarse_skew: f32, src_w: f32, src_h: f32) -> Frame {
    let (sin, cos) = coarse_skew.sin_cos();
    let qx = frame.center[0] - src_w / 2.0;
    let qy = frame.center[1] - src_h / 2.0;
    Frame {
        center: [
            src_w / 2.0 + cos * qx - sin * qy,
            src_h / 2.0 + sin * qx + cos * qy,
        ],
        size: frame.size,
        skew: coarse_skew + frame.skew,
    }
}

/// Runs the whole detection pipeline on one analysis grid.
///
/// The grid may already be straightened; coordinates come back in that grid's
/// space, scaled to source pixels. `base` is the least-channel clear-film
/// level, measured once by [`FilmBase`] and shared by both passes.
fn detect_frame(
    gray: &GrayImage,
    scale: f32,
    source_width: u32,
    source_height: u32,
    base: f32,
) -> Option<Frame> {
    let gradients = gradient(gray);
    let (vertical, horizontal) = candidates(gray, &gradients);
    if vertical.is_empty() || horizontal.is_empty() {
        return None;
    }

    let dw = gray.width as usize;
    let dh = gray.height as usize;
    let vertical_lines = find_lines(&vertical, dw, dh, Family::Vertical);
    let horizontal_lines = find_lines(&horizontal, dw, dh, Family::Horizontal);
    if vertical_lines.is_empty() || horizontal_lines.is_empty() {
        return None;
    }

    assemble_frame(
        &vertical_lines,
        &horizontal_lines,
        gray,
        scale,
        source_width,
        source_height,
        base,
    )
}

/// Rotates an analysis grid by `angle` about its centre.
///
/// Full extent, nothing cropped, and samples outside the grid clamp to the
/// edge — a lossy operation, which is why it happens only to the downsampled
/// analysis image and never to the scan itself.
fn rotate_gray(image: &GrayImage, angle: f32) -> GrayImage {
    let width = image.width as usize;
    let height = image.height as usize;
    let (cx, cy) = (coordinate(width) / 2.0, coordinate(height) / 2.0);
    let (sin, cos) = angle.sin_cos();

    let mut data = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let dx = coordinate(x) - cx;
            let dy = coordinate(y) - cy;
            // Inverse of "rotate the content by angle": sample the source.
            let sx = cx + cos * dx + sin * dy;
            let sy = cy - sin * dx + cos * dy;
            data[y * width + x] = sample_gray(image, width, height, sx, sy);
        }
    }

    GrayImage {
        width: image.width,
        height: image.height,
        data,
    }
}

/// One sprocket-hole band, as the row range `[start, end)` it occupies.
#[derive(Debug, Clone, Copy)]
struct Band {
    start: f32,
    end: f32,
}

impl Band {
    /// The band's height, in rows.
    fn height(self) -> f32 {
        self.end - self.start
    }
}

/// A strip's two hole bands and the mechanical tilt they carry.
struct Bands {
    top: Band,
    bottom: Band,
    skew: f32,
}

/// Detects the strip's sprocket-hole bands on one scan.
///
/// Returns `None` when either band is missing or too short: the strip never
/// shows a full film width, which is exactly the case where the mechanical
/// anchors are not available and detection must answer differently.
fn detect_bands(image: &GrayImage, base: f32) -> Option<Bands> {
    let width = image.width as usize;
    let height = image.height as usize;
    let bright = base * HOLE_BRIGHTNESS;

    // Share of each row that reads as bare light. Content never reaches this
    // bar; holes reach it on every row of their band.
    let mut fraction = vec![0.0f32; height];
    for (y, slot) in fraction.iter_mut().enumerate() {
        let mut count = 0usize;
        for x in 0..width {
            if image.get(x, y) >= bright {
                count += 1;
            }
        }
        *slot = coordinate(count) / coordinate(width.max(1));
    }

    let half = height / 2;
    let top = band_run(&fraction, 0..half, true)?;
    let bottom = band_run(&fraction, half..height, false)?;
    let skew = band_skew(image, base, top, bottom);

    Some(Bands { top, bottom, skew })
}

/// Runs of light rows in one vertical half of the grid, taking the run
/// nearest the strip's edge: first qualifying run for the top, last for the
/// bottom. A run grows through qualifying rows, tolerating the few dark rows
/// a tilted band straddles between successive holes; shorter-than-minimal
/// runs are not bands at all.
fn band_run(fraction: &[f32], rows: std::ops::Range<usize>, from_top: bool) -> Option<Band> {
    if rows.end > fraction.len() || rows.start >= rows.end {
        return None;
    }

    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut current: Option<(usize, usize)> = None;
    for row in rows.clone() {
        if fraction[row] >= HOLE_ROW_FRACTION {
            current = Some(match current {
                Some((from, last)) if row <= last + BAND_GAP_TOLERANCE => (from, row),
                _ => (row, row),
            });
        } else if let Some(run) = current.take() {
            runs.push(run);
        }
    }
    if let Some(run) = current {
        runs.push(run);
    }
    runs.retain(|(from, last)| last.saturating_sub(*from) + 1 >= MIN_BAND_ROWS);

    let chosen = if from_top {
        runs.into_iter()
            .min_by_key(|(from, last)| (*from, Reverse(last.saturating_sub(*from))))
    } else {
        runs.into_iter()
            .max_by_key(|(from, last)| (*last, last.saturating_sub(*from)))
    }?;

    Some(Band {
        start: coordinate(chosen.0),
        end: coordinate(chosen.1 + 1),
    })
}

/// The strip's tilt from the hole bands' axes, through a least-squares fit of
/// each band's per-column bright centroid.
///
/// Bands are parts of the film itself; their axis carries the strip's skew
/// with no content intercession anywhere in the measurement.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "band rows are non-negative small integers on the working grid"
)]
fn band_skew(image: &GrayImage, base: f32, top: Band, bottom: Band) -> f32 {
    let width = image.width as usize;
    let height = image.height as usize;
    let bright = base * HOLE_BRIGHTNESS;

    let slope = |band: Band| -> Option<f32> {
        let y0 = band.start.floor().max(0.0) as usize;
        let y1 = (band.end.ceil() as usize).min(height.saturating_sub(1));
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for x in 0..width {
            let mut weight = 0.0f32;
            let mut sum = 0.0f32;
            for y in y0..=y1 {
                if image.get(x, y) >= bright {
                    weight += 1.0;
                    sum += coordinate(y);
                }
            }
            if weight > 0.0 {
                xs.push(coordinate(x));
                ys.push(sum / weight);
            }
        }
        if xs.len() < 8 {
            return None;
        }
        let n = coordinate(xs.len());
        let mean_x = xs.iter().sum::<f32>() / n;
        let mean_y = ys.iter().sum::<f32>() / n;
        let mut cov = 0.0f32;
        let mut var = 0.0f32;
        for i in 0..xs.len() {
            let dx = xs[i] - mean_x;
            cov += dx * (ys[i] - mean_y);
            var += dx * dx;
        }
        if var <= 0.0 {
            return None;
        }
        Some(cov / var)
    };

    // Slope dy/dx of a near-horizontal edge is tan(skew).
    let slopes = [slope(top), slope(bottom)]
        .into_iter()
        .flatten()
        .collect::<Vec<f32>>();
    if slopes.is_empty() {
        return 0.0;
    }
    let mean = slopes.iter().sum::<f32>() / coordinate(slopes.len());
    f32::atan(mean)
}

/// The picture-side boundary of one hole band, as a working-grid line.
///
/// From the band edge, a rebate-sized zone is scanned per column for the
/// sharpest outer step — clear film toward the band, picture beyond it. The
/// columns whose step clears a small floor vote; with enough of them the
/// votes fit as one line. `None` when support is short of
/// [`REBATE_SUPPORT_MIN`]: under that much dark content the rebate is simply
/// not measurable and the caller places it by the reference offset.
///
/// `search` is the zone's depth, in grid rows — [`Band`]-derived and capped
/// by the rebate bounds, never content-derived.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "band rows are non-negative small integers on the working grid"
)]
fn rebate_edge(
    image: &GrayImage,
    base: f32,
    band: Band,
    from_top: bool,
    search: f32,
) -> Option<Line> {
    let width = image.width as usize;
    let height = image.height as usize;
    let step_min = base * REBATE_STEP_MIN;

    let (zone_start, zone_end) = if from_top {
        (
            band.end,
            (band.end + search).min(coordinate(height.saturating_sub(1))),
        )
    } else {
        ((band.start - search).max(0.0), band.start)
    };
    if zone_end - zone_start < 1.0 {
        return None;
    }

    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for x in 0..width {
        let mut best: Option<(f32, f32)> = None; // (row, step)
        let y0 = zone_start.ceil() as usize;
        let y1 = (zone_end.floor() as usize).min(height.saturating_sub(2));
        for y in y0..y1 {
            let above = image.get(x, y.saturating_sub(1));
            let below = image.get(x, (y + 1).min(height.saturating_sub(1)));
            // Toward the picture from the band, the rebate is the bright side.
            let step = if from_top {
                above - below
            } else {
                below - above
            };
            if step > best.map_or(f32::NEG_INFINITY, |(_, s)| s) {
                best = Some((coordinate(y), step));
            }
        }
        if best.is_some_and(|(_, step)| step >= step_min) {
            xs.push(coordinate(x));
            ys.push(best.expect("checked above").0);
        }
    }

    let support = coordinate(xs.len()) / coordinate(width.max(1));
    if support < REBATE_SUPPORT_MIN {
        return None;
    }

    // Fit the votes. In the straightened grid the boundary is near-level, so
    // this is a row line; width-spread votes with big residuals are distrust.
    let n = coordinate(xs.len());
    let mean_x = xs.iter().sum::<f32>() / n;
    let mean_y = ys.iter().sum::<f32>() / n;
    let mut cov = 0.0f32;
    let mut var = 0.0f32;
    let mut resid = 0.0f32;
    for i in 0..xs.len() {
        let dx = xs[i] - mean_x;
        cov += dx * (ys[i] - mean_y);
        var += dx * dx;
        let pred = mean_y + cov.max(0.0) / var.max(1.0) * dx;
        resid += (ys[i] - pred) * (ys[i] - pred);
    }
    if var <= 0.0 {
        return Some(Line { a: mean_y, m: 0.0 });
    }
    let m = cov / var;
    let a = mean_y - m * mean_x;
    resid /= n;
    if resid.sqrt() > 2.0 {
        return None;
    }
    Some(Line { a, m })
}

/// A band's reference-side boundary line: the rebate placed by the measured
/// default, at the band's row range.
fn band_line(band: Band, from_top: bool) -> Line {
    let offset = REBATE_DEFAULT_FRACTION * band.height();
    let line_row = if from_top {
        band.end + offset
    } else {
        band.start - offset
    };
    Line {
        a: line_row,
        m: 0.0,
    }
}

/// The picture content's horizontal extent between two boundary rows.
///
/// Rows are sampled across the band; a row qualifies any pixel noticeably
/// below the rebate as content and its extent is the widest qualifying run,
/// tolerating bright content inside the picture. The median extent wins:
/// rebate that reads darker than the base — an underexposed scan's usual
/// state — drags individual rows wide, and the median bounds the damage.
///
/// Generous by design: the vertical sides of a band-anchored frame are
/// approximations, with the frame's width coming from the film's aspect and
/// the position from the content's centre.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "grid coordinates are non-negative small integers"
)]
fn content_span(image: &GrayImage, base: f32, top: Line, bottom: Line) -> Option<(f32, f32)> {
    let width = image.width as usize;
    let height = image.height as usize;
    let level = base * CONTENT_LEVEL_FRACTION;

    let y_at = |line: Line, x: f32| line.a + line.m * x;
    let row_top = y_at(top, coordinate(width) / 2.0)
        .floor()
        .clamp(0.0, coordinate(height.saturating_sub(1))) as usize;
    let row_bottom = y_at(bottom, coordinate(width) / 2.0)
        .floor()
        .clamp(0.0, coordinate(height.saturating_sub(1))) as usize;
    if row_bottom <= row_top + 1 {
        return None;
    }

    let mut starts = Vec::new();
    let mut ends = Vec::new();
    let step_rows = ((row_bottom - row_top) / 60).max(2);
    let mut row = row_top + 2;
    while row < row_bottom.saturating_sub(2) {
        // One row's widest qualifying run, small bright gaps tolerated: the
        // picture holds highlight areas, but never rebate-sized ones.
        let mut best: Option<(usize, usize)> = None;
        let mut current: Option<(usize, usize)> = None;
        let mut last = 0usize;
        for x in 0..width {
            let qualifies = image.get(x, row) < level;
            if qualifies {
                current = Some(match current {
                    Some((from, _)) if x <= last + CONTENT_GAP_TOLERANCE => (from, x),
                    _ => (x, x),
                });
                last = x;
            } else if let Some(found) = current.take() {
                best = Some(match best {
                    Some(wider_of) => wider(found, wider_of),
                    None => found,
                });
            }
        }
        if let Some(found) = current {
            best = Some(match best {
                Some(wider_of) => wider(found, wider_of),
                None => found,
            });
        }
        if let Some((from, to)) = best {
            starts.push(coordinate(from));
            ends.push(coordinate(to + 1));
        }
        row += step_rows;
    }

    if starts.len() < 3 {
        return None;
    }
    starts.sort_by(f32::total_cmp);
    ends.sort_by(f32::total_cmp);
    let (start, end) = (median_of(&mut starts), median_of(&mut ends));
    if end - start < coordinate(width) / 10.0 {
        return None;
    }
    Some((start, end))
}

/// The wider of two `(start, end)` runs.
fn wider(a: (usize, usize), b: (usize, usize)) -> (usize, usize) {
    if b.1 - b.0 > a.1 - a.0 { b } else { a }
}

/// Median of an unsorted slice, in place.
fn median_of(values: &mut [f32]) -> f32 {
    values.sort_by(f32::total_cmp);
    values[values.len() / 2]
}

/// Builds the frame from the mechanical anchors: the bands set the vertical
/// bounds, refined by the rebate search inside their own bounds; the film's
/// aspect sets the width; the content's centre positions it.
///
/// Returns the frame in the grid's own coordinates, scaled to source pixels;
/// the caller folds it back through the straightening rotation.
fn band_frame(image: &GrayImage, scale: f32, base: f32, bands: &Bands) -> Option<Frame> {
    let width = image.width as usize;
    let height = image.height as usize;

    let inner = bands.bottom.start - bands.top.end;
    if inner <= 0.0 {
        return None;
    }
    let rebate_max = REBATE_MAX_FRACTION * inner;

    let top = rebate_edge(image, base, bands.top, true, rebate_max)
        .unwrap_or_else(|| band_line(bands.top, true));
    let bottom = rebate_edge(image, base, bands.bottom, false, rebate_max)
        .unwrap_or_else(|| band_line(bands.bottom, false));

    let y0 = top.a.max(0.0).min(coordinate(height));
    let y1 = bottom.a.max(y0).min(coordinate(height));
    let height_grid = y1 - y0;
    if height_grid <= 0.0 {
        return None;
    }

    // Width from the film's own aspect; position from the content. The
    // grid's long axis is the strip's length direction, cropped to one
    // frame's length. Bounded by the grid: a frame may run off the sensor
    // edge, and clamping into the sensor keeps the crop on film.
    let (content_start, content_end) = content_span(image, base, top, bottom)?;
    let content_center = f32::midpoint(content_start, content_end);
    let width_grid = FRAME_ASPECT * height_grid;
    let half = width_grid / 2.0;
    let x0 = (content_center - half).max(0.0);
    let x1 = (content_center + half).min(coordinate(width));

    let final_width = (x1 - x0) * scale - 2.0 * BOUNDARY_INSET * scale;
    let final_height = (y1 - y0) * scale - 2.0 * BOUNDARY_INSET * scale;
    if final_width <= 0.0 || final_height <= 0.0 {
        return None;
    }

    Some(Frame {
        center: [f32::midpoint(x0, x1) * scale, f32::midpoint(y0, y1) * scale],
        size: [final_width, final_height],
        skew: bands.skew,
    })
}

/// A downsampled grayscale image.
struct GrayImage {
    width: u32,
    height: u32,
    data: Vec<f32>,
}

impl GrayImage {
    fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width as usize + x]
    }
}

/// Builds the least-channel grayscale image and box-downsamples it so the
/// longest side is about [`WORKING_LONG_SIDE`] samples.
///
/// Returns the working image and the source-to-working scale factor.
fn downsample(image: &LinearImage) -> (GrayImage, f32) {
    let width = image.width as usize;
    let height = image.height as usize;
    let max_dim = image.width.max(image.height);

    let factor = if max_dim <= WORKING_LONG_SIDE {
        1
    } else {
        max_dim.div_ceil(WORKING_LONG_SIDE)
    } as usize;

    let gray: Vec<f32> = image
        .data
        .as_chunks::<3>()
        .0
        .iter()
        .map(least_channel)
        .collect();

    if factor == 1 {
        return (
            GrayImage {
                width: image.width,
                height: image.height,
                data: gray,
            },
            1.0,
        );
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "downsample factor is at most a small integer"
    )]
    let scale = factor as f32;
    let out_w = width.div_ceil(factor);
    let out_h = height.div_ceil(factor);
    let mut out = vec![0.0f32; out_w * out_h];

    for oy in 0..out_h {
        let y0 = oy * factor;
        let y1 = ((oy + 1) * factor).min(height);
        for ox in 0..out_w {
            let x0 = ox * factor;
            let x1 = ((ox + 1) * factor).min(width);

            let mut sum = 0.0f32;
            let mut count = 0usize;
            for y in y0..y1 {
                for x in x0..x1 {
                    sum += gray[y * width + x];
                    count += 1;
                }
            }
            out[oy * out_w + ox] = sum / coordinate(count);
        }
    }

    (
        GrayImage {
            width: u32::try_from(out_w).unwrap_or(1),
            height: u32::try_from(out_h).unwrap_or(1),
            data: out,
        },
        scale,
    )
}

/// Gradient components and magnitude on the working grid.
struct Gradient {
    gx: Vec<f32>,
    gy: Vec<f32>,
    mag: Vec<f32>,
}

/// Central-difference gradient magnitude on the least-channel image.
fn gradient(image: &GrayImage) -> Gradient {
    let width = image.width as usize;
    let height = image.height as usize;
    let len = width * height;
    let mut gx = vec![0.0f32; len];
    let mut gy = vec![0.0f32; len];
    let mut mag = vec![0.0f32; len];

    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let idx = y * width + x;
            let dx = (image.get(x + 1, y) - image.get(x - 1, y)) * 0.5;
            let dy = (image.get(x, y + 1) - image.get(x, y - 1)) * 0.5;
            gx[idx] = dx;
            gy[idx] = dy;
            mag[idx] = f32::hypot(dx, dy);
        }
    }

    Gradient { gx, gy, mag }
}

/// A candidate edge pixel on the working grid.
struct Candidate {
    x: f32,
    y: f32,
    gx: f32,
    gy: f32,
}

/// Splits candidate edge pixels into near-vertical and near-horizontal families.
///
/// Near-vertical edges have a strong horizontal gradient; near-horizontal edges
/// have a strong vertical gradient. Only the top percentile of gradient
/// magnitude is kept.
fn candidates(image: &GrayImage, grad: &Gradient) -> (Vec<Candidate>, Vec<Candidate>) {
    let width = image.width as usize;
    let height = image.height as usize;

    let threshold = percentile(&grad.mag, GRADIENT_PERCENTILE);

    let mut vertical = Vec::new();
    let mut horizontal = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let m = grad.mag[idx];
            // At or below the bar — which is zero at or below for an image with
            // no texture at all.
            if m <= threshold {
                continue;
            }
            let c = Candidate {
                x: coordinate(x),
                y: coordinate(y),
                gx: grad.gx[idx],
                gy: grad.gy[idx],
            };
            if c.gx.abs() > c.gy.abs() {
                vertical.push(c);
            } else {
                horizontal.push(c);
            }
        }
    }

    (vertical, horizontal)
}

/// Returns the value at the given fractional rank, or `0.0` for an empty slice.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the index is a non-negative fraction of the slice length and fits usize"
)]
fn percentile(values: &[f32], fraction: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f32> = values.to_vec();
    let index = (coordinate(sorted.len().saturating_sub(1)) * fraction.clamp(0.0, 1.0)) as usize;
    let (_left, value, _right) = sorted.select_nth_unstable_by(index, f32::total_cmp);
    *value
}

/// Which family a line belongs to.
#[derive(Debug, Clone, Copy)]
enum Family {
    /// `x = a + m * y`
    Vertical,
    /// `y = b + m * x`
    Horizontal,
}

/// A fitted line on the working grid.
#[derive(Clone, Copy)]
struct Line {
    /// Intercept: `a` for vertical family, `b` for horizontal family.
    a: f32,
    /// Slope: `dx/dy` for vertical family, `dy/dx` for horizontal family.
    m: f32,
}

/// Runs a Hough-style accumulator for one family and refines the strongest peaks.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "accumulator dimensions are derived from small positive counts and angles"
)]
fn find_lines(candidates: &[Candidate], width: usize, height: usize, family: Family) -> Vec<Line> {
    let angle_bins = (2.0 * MAX_SKEW / ANGLE_STEP).ceil() as usize + 1;
    let angle_min = -MAX_SKEW;
    let slopes: Vec<f32> = (0..angle_bins)
        .map(|i| {
            let angle = angle_min + coordinate(i) * ANGLE_STEP;
            f32::tan(angle)
        })
        .collect();

    let last_x = coordinate(width.saturating_sub(1));
    let last_y = coordinate(height.saturating_sub(1));
    let (offset_min, offset_max) = match family {
        Family::Vertical => (-MAX_LINE_SLOPE * last_y, last_x + MAX_LINE_SLOPE * last_y),
        Family::Horizontal => (-MAX_LINE_SLOPE * last_x, last_y + MAX_LINE_SLOPE * last_x),
    };
    let offset_bins = ((offset_max - offset_min) / OFFSET_STEP).ceil() as usize + 1;

    let mut acc = vec![vec![0u32; offset_bins]; angle_bins];

    for c in candidates {
        for (ai, &m) in slopes.iter().enumerate() {
            let rho = match family {
                Family::Vertical => c.x - m * c.y,
                Family::Horizontal => c.y - m * c.x,
            };
            let Some(oi) = bin_index(rho, offset_min, OFFSET_STEP, offset_bins) else {
                continue;
            };
            acc[ai][oi] += 1;
        }
    }

    let peaks = extract_peaks(&acc);

    let mut lines = Vec::new();
    for (ai, oi, _votes) in peaks.into_iter().take(MAX_PEAKS_PER_FAMILY) {
        let m0 = slopes[ai];
        let a0 = offset_min + coordinate(oi) * OFFSET_STEP;
        let Some(line) = refine_line(candidates, family, a0, m0, width, height) else {
            continue;
        };

        // Tied bins refine to the same line; keep the first fit.
        let duplicate = lines.iter().any(|fitted: &Line| {
            (fitted.a - line.a).abs() < DUPLICATE_OFFSET
                && (fitted.m - line.m).abs() < DUPLICATE_SLOPE
        });
        if !duplicate {
            lines.push(line);
        }
    }

    lines
}

/// Maps a continuous parameter to a bin index, or `None` if it lies outside.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "value is checked to be within [0, bins) before truncation"
)]
fn bin_index(value: f32, min: f32, step: f32, bins: usize) -> Option<usize> {
    let f = (value - min) / step;
    if f < 0.0 || f >= coordinate(bins) {
        return None;
    }
    Some(f as usize)
}

/// Extracts local maxima from the accumulator, sorted by vote count descending.
///
/// The comparison against peers is strict, not greater-or-equal: the transition
/// band between picture and film straddles one or two offset bins and those
/// bins usually tie, and with greater-or-equal each suppressed the other until
/// no peak survived at all. Both bins survive here and only ever refine to the
/// same line, which [`find_lines`] deduplicates.
fn extract_peaks(acc: &[Vec<u32>]) -> Vec<(usize, usize, u32)> {
    let angle_bins = acc.len();
    if angle_bins == 0 {
        return Vec::new();
    }
    let offset_bins = acc[0].len();

    let mut peaks = Vec::new();
    for ai in 0..angle_bins {
        for oi in 0..offset_bins {
            let count = acc[ai][oi];
            if count < MIN_VOTES {
                continue;
            }

            let mut is_max = true;
            for dai in 0..=2 * NMS_ANGLE_RADIUS {
                let ai_n = ai.saturating_add(dai).saturating_sub(NMS_ANGLE_RADIUS);
                if ai_n >= angle_bins {
                    continue;
                }
                for doi in 0..=2 * NMS_OFFSET_RADIUS {
                    let oi_n = oi.saturating_add(doi).saturating_sub(NMS_OFFSET_RADIUS);
                    if oi_n >= offset_bins || (ai_n == ai && oi_n == oi) {
                        continue;
                    }
                    if acc[ai_n][oi_n] > count {
                        is_max = false;
                        break;
                    }
                }
                if !is_max {
                    break;
                }
            }

            if is_max {
                peaks.push((ai, oi, count));
            }
        }
    }

    peaks.sort_by_key(|a| std::cmp::Reverse(a.2));
    peaks
}

/// Refines a Hough peak with least-squares and quality gates.
fn refine_line(
    candidates: &[Candidate],
    family: Family,
    a0: f32,
    m0: f32,
    width: usize,
    height: usize,
) -> Option<Line> {
    let (a, m) = fit_line(candidates, family, a0, m0)?;
    let (a, m) = fit_line(candidates, family, a, m)?;

    let stats = line_stats(candidates, family, a, m, width, height)?;
    if stats.count < MIN_LINE_INLIERS
        || stats.residual_std > MAX_RESIDUAL_STD
        || stats.span_ratio < MIN_SPAN_RATIO
    {
        return None;
    }

    Some(Line { a, m })
}

/// Fits a line to its inliers and returns the refined parameters.
fn fit_line(candidates: &[Candidate], family: Family, a: f32, m: f32) -> Option<(f32, f32)> {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for c in candidates {
        if distance_to_line(c, family, a, m).abs() <= INLIER_DISTANCE {
            xs.push(c.x);
            ys.push(c.y);
        }
    }

    let n = coordinate(xs.len());
    if xs.len() < 4 {
        return None;
    }

    let mean_x = xs.iter().sum::<f32>() / n;
    let mean_y = ys.iter().sum::<f32>() / n;

    match family {
        Family::Vertical => {
            let mut var_y = 0.0f32;
            let mut cov_xy = 0.0f32;
            for i in 0..xs.len() {
                let dy = ys[i] - mean_y;
                var_y += dy * dy;
                cov_xy += dy * (xs[i] - mean_x);
            }
            if var_y <= 0.0 {
                return None;
            }
            let m_new = cov_xy / var_y;
            let a_new = mean_x - m_new * mean_y;
            Some((a_new, m_new))
        }
        Family::Horizontal => {
            let mut var_x = 0.0f32;
            let mut cov_xy = 0.0f32;
            for i in 0..xs.len() {
                let dx = xs[i] - mean_x;
                var_x += dx * dx;
                cov_xy += dx * (ys[i] - mean_y);
            }
            if var_x <= 0.0 {
                return None;
            }
            let m_new = cov_xy / var_x;
            let b_new = mean_y - m_new * mean_x;
            Some((b_new, m_new))
        }
    }
}

/// Signed perpendicular distance from a candidate to a line.
fn distance_to_line(c: &Candidate, family: Family, a: f32, m: f32) -> f32 {
    let num = match family {
        Family::Vertical => c.x - a - m * c.y,
        Family::Horizontal => c.y - a - m * c.x,
    };
    num / f32::hypot(1.0, m)
}

/// Quality statistics for a fitted line.
struct LineStats {
    count: usize,
    residual_std: f32,
    span_ratio: f32,
}

/// Computes inlier count, residual spread, and span ratio for a line.
fn line_stats(
    candidates: &[Candidate],
    family: Family,
    a: f32,
    m: f32,
    width: usize,
    height: usize,
) -> Option<LineStats> {
    let mut residuals = Vec::new();
    let mut along = Vec::new();

    for c in candidates {
        let d = distance_to_line(c, family, a, m);
        if d.abs() <= INLIER_DISTANCE {
            residuals.push(d);
            along.push(match family {
                Family::Vertical => c.y,
                Family::Horizontal => c.x,
            });
        }
    }

    if residuals.is_empty() {
        return None;
    }

    let mean = residuals.iter().sum::<f32>() / coordinate(residuals.len());
    let variance = residuals
        .iter()
        .map(|r| (r - mean) * (r - mean))
        .sum::<f32>()
        / coordinate(residuals.len());
    let residual_std = variance.sqrt();

    let extent = line_extent(family, a, m, width, height);
    let span = along.iter().copied().fold(f32::INFINITY, f32::min)
        ..along.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let span_length = span.end - span.start;
    let span_ratio = if extent > 0.0 {
        span_length / extent
    } else {
        0.0
    };

    Some(LineStats {
        count: residuals.len(),
        residual_std,
        span_ratio,
    })
}

/// Length of the segment where the line lies inside the image.
fn line_extent(family: Family, a: f32, m: f32, width: usize, height: usize) -> f32 {
    let last_x = coordinate(width.saturating_sub(1));
    let last_y = coordinate(height.saturating_sub(1));

    match family {
        Family::Vertical => {
            // x = a + m * y must stay within [0, last_x] while y in [0, last_y].
            let mut y0 = 0.0f32;
            let mut y1 = last_y;
            if m.abs() > 1e-6 {
                let y_at_x0 = -a / m;
                let y_at_x1 = (last_x - a) / m;
                if m > 0.0 {
                    y0 = y0.max(y_at_x0);
                    y1 = y1.min(y_at_x1);
                } else {
                    y0 = y0.max(y_at_x1);
                    y1 = y1.min(y_at_x0);
                }
            } else if a < 0.0 || a > last_x {
                return 0.0;
            }
            (y1 - y0).max(0.0)
        }
        Family::Horizontal => {
            // y = a + m * x must stay within [0, last_y] while x in [0, last_x].
            let mut x0 = 0.0f32;
            let mut x1 = last_x;
            if m.abs() > 1e-6 {
                let x_at_y0 = -a / m;
                let x_at_y1 = (last_y - a) / m;
                if m > 0.0 {
                    x0 = x0.max(x_at_y0);
                    x1 = x1.min(x_at_y1);
                } else {
                    x0 = x0.max(x_at_y1);
                    x1 = x1.min(x_at_y0);
                }
            } else if a < 0.0 || a > last_y {
                return 0.0;
            }
            (x1 - x0).max(0.0)
        }
    }
}

/// Tries every ordered pair of vertical lines and every ordered pair of
/// horizontal lines, keeping the best-scoring candidate rectangle.
///
/// Ties break first toward a rectangle that contains the image centre, then
/// toward the larger area.
fn assemble_frame(
    vertical: &[Line],
    horizontal: &[Line],
    image: &GrayImage,
    scale: f32,
    source_width: u32,
    source_height: u32,
    base: f32,
) -> Option<Frame> {
    let dw = image.width as usize;
    let dh = image.height as usize;
    let src_w = coordinate(source_width as usize);
    let src_h = coordinate(source_height as usize);

    let mut candidates_valid: Option<(Frame, f32, bool)> = None;
    let mut best_area = 0.0f32;

    for left in vertical {
        for right in vertical {
            if std::ptr::eq(left, right) {
                continue;
            }
            for top in horizontal {
                for bottom in horizontal {
                    if std::ptr::eq(top, bottom) {
                        continue;
                    }

                    let Some((frame, score, contains_center)) = score_rectangle(
                        *left, *right, *top, *bottom, image, dw, dh, scale, src_w, src_h, base,
                    ) else {
                        continue;
                    };

                    let area = frame.size[0] * frame.size[1];
                    let better = match &candidates_valid {
                        None => true,
                        Some((_, best_score, best_contains)) => {
                            if score > *best_score + SCORE_TIE {
                                true
                            } else if score >= *best_score - SCORE_TIE {
                                match (contains_center, *best_contains) {
                                    (true, false) => true,
                                    (false, true) => false,
                                    // A smaller equal-scoring rectangle is
                                    // exactly the swallowed-rebate case; the
                                    // larger shape cannot be a frame whose
                                    // edges were missed, because any valid
                                    // interior rectangle has picture outside
                                    // its edges and fails the level window.
                                    _ => area < best_area,
                                }
                            } else {
                                false
                            }
                        }
                    };
                    if better {
                        candidates_valid = Some((frame, score, contains_center));
                        best_area = area;
                    }
                }
            }
        }
    }

    candidates_valid.map(|(frame, _, _)| frame)
}

/// Scores one candidate rectangle. Returns `None` if the rectangle fails any
/// consistency or clear-film gate.
///
/// `base` is the least-channel clear-film level for this frame, measured by
/// the film-base detector on the same image.
#[expect(
    clippy::too_many_arguments,
    reason = "rectangle scoring needs the four lines, the image, dimensions, scale, and base"
)]
fn score_rectangle(
    left: Line,
    right: Line,
    top: Line,
    bottom: Line,
    image: &GrayImage,
    dw: usize,
    dh: usize,
    scale: f32,
    src_w: f32,
    src_h: f32,
    base: f32,
) -> Option<(Frame, f32, bool)> {
    let skew_left = -f32::atan(left.m);
    let skew_right = -f32::atan(right.m);
    let skew_top = f32::atan(top.m);
    let skew_bottom = f32::atan(bottom.m);
    let skew = median4([skew_left, skew_right, skew_top, skew_bottom]);

    if [skew_left, skew_right, skew_top, skew_bottom]
        .iter()
        .any(|s| (*s - skew).abs() > MAX_SKEW_DISAGREEMENT)
    {
        return None;
    }
    if skew.abs() > MAX_SKEW {
        return None;
    }

    let cos = f32::cos(skew);

    // In deskewed coordinates vertical lines become x' = a * cos(skew) and
    // horizontal lines become y' = b * cos(skew).
    let mut x0 = left.a * cos;
    let mut x1 = right.a * cos;
    if x0 > x1 {
        std::mem::swap(&mut x0, &mut x1);
    }
    let mut y0 = top.a * cos;
    let mut y1 = bottom.a * cos;
    if y0 > y1 {
        std::mem::swap(&mut y0, &mut y1);
    }

    let inset = BOUNDARY_INSET;
    let width_source = ((x1 - x0) - 2.0 * inset) * scale;
    let height_source = ((y1 - y0) - 2.0 * inset) * scale;
    if width_source < MIN_FRAME_EXTENT * src_w || height_source < MIN_FRAME_EXTENT * src_h {
        return None;
    }

    let tl = intersect(left, top);
    let tr = intersect(right, top);
    let bl = intersect(left, bottom);
    let br = intersect(right, bottom);
    let corners = [tl, tr, bl, br];

    let left_out = normalize([-1.0, left.m]);
    let right_out = normalize([1.0, -right.m]);
    let top_out = normalize([top.m, -1.0]);
    let bottom_out = normalize([-bottom.m, 1.0]);

    let left_contrast = edge_quality(image, dw, dh, tl, bl, left_out, base)?;
    let right_contrast = edge_quality(image, dw, dh, tr, br, right_out, base)?;
    let top_contrast = edge_quality(image, dw, dh, tl, tr, top_out, base)?;
    let bottom_contrast = edge_quality(image, dw, dh, bl, br, bottom_out, base)?;

    let center_ds = [
        (tl.0 + tr.0 + bl.0 + br.0) * 0.25,
        (tl.1 + tr.1 + bl.1 + br.1) * 0.25,
    ];
    let center = [center_ds[0] * scale, center_ds[1] * scale];

    // Guard against lines that meet far outside the image.
    if center[0] < 0.0 || center[0] > src_w || center[1] < 0.0 || center[1] > src_h {
        return None;
    }

    // Additional guard: no corner may lie absurdly far from the image, which
    // would mean the four lines do not actually form a usable rectangle.
    let margin = src_w.max(src_h) * 2.0;
    if corners
        .iter()
        .any(|(x, y)| x.abs() > margin || y.abs() > margin)
    {
        return None;
    }

    let score = left_contrast + right_contrast + top_contrast + bottom_contrast;
    let frame = Frame {
        center,
        size: [width_source, height_source],
        skew,
    };

    // Does the rectangle contain the image centre? The operator frames the
    // current frame near the middle of the sensor; a clipped neighbour frame is
    // the exception. Used only as a tie-break between equal-scoring
    // rectangles, which happen when a neighbour frame is fully visible. The
    // tolerance lets a centre lying exactly on a boundary count as contained:
    // the fitted lines run through the middle of a finite-wide transition.
    let ix = src_w / 2.0;
    let iy = src_h / 2.0;
    let contains_center = (frame.center[0] - ix).abs() <= frame.size[0] / 2.0 + CENTER_TOLERANCE
        && (frame.center[1] - iy).abs() <= frame.size[1] / 2.0 + CENTER_TOLERANCE;

    Some((frame, score, contains_center))
}

/// Normalizes a vector to unit length, or returns the zero vector.
fn normalize(v: [f32; 2]) -> [f32; 2] {
    let len = f32::hypot(v[0], v[1]);
    if len == 0.0 {
        return [0.0; 2];
    }
    [v[0] / len, v[1] / len]
}

/// Intersection of a vertical-family line and a horizontal-family line.
fn intersect(vertical: Line, horizontal: Line) -> (f32, f32) {
    let denom = 1.0 - vertical.m * horizontal.m;
    if denom.abs() < 1e-6 {
        return (0.0, 0.0);
    }
    let x = (vertical.a + vertical.m * horizontal.a) / denom;
    let y = horizontal.a + horizontal.m * x;
    (x, y)
}

/// Median of four values.
fn median4(mut values: [f32; 4]) -> f32 {
    values.sort_by(f32::total_cmp);
    f32::midpoint(values[1], values[2])
}

/// How far below the clear-film level the outside band may sit, as a fraction
/// of the base.
///
/// The outside of every frame edge is clear film, so anything darker than a
/// fraction of the base is not film: the black sensor margin around the strip
/// measures a fiftieth of the base, and its edge lines are precisely what a
/// whole-strip rectangle is built from.
const OUTSIDE_BASE_MIN: f32 = 0.4;

/// How far above the clear-film level the outside band may sit, as a fraction
/// of the base.
///
/// Bare light — sprocket holes — measures about ten times the base on a real
/// scan, so a hole-band edge cannot become the outside of a frame side. The
/// window is deliberately generous: uneven illumination shifts the rebate a
/// little, the holes shift it an order of magnitude.
const OUTSIDE_BASE_MAX: f32 = 2.5;

/// Samples one edge and applies the frame's per-edge gates.
///
/// `outward` is the unit normal pointing toward the clear-film side, computed
/// per edge in [`score_rectangle`]: for the left edge it points toward negative
/// x, the right edge toward positive x, the top edge toward negative y, and the
/// bottom edge toward positive y.
///
/// Two gates, both on quantiles along the edge rather than means, because an
/// edge's band is not uniform: where the rebate narrows to nothing the holes
/// sit directly above the picture, and a mean over the whole edge would smear
/// that in. The 60th-percentile keeps the band's typical sample.
///
/// * the outside band must be clear film: its level sits between fractions of
///   the measured base ([`OUTSIDE_BASE_MIN`], [`OUTSIDE_BASE_MAX`]).
/// * the outside must be clearly brighter than the inside:
///   [`MIN_CONTRAST_STEP`].
///
/// Returns the measured step if both gates pass.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "step count is a positive length divided by a positive step size"
)]
fn edge_quality(
    image: &GrayImage,
    width: usize,
    height: usize,
    start: (f32, f32),
    end: (f32, f32),
    outward: [f32; 2],
    base: f32,
) -> Option<f32> {
    let dx = end.0 - start.0;
    let dy = end.1 - start.1;
    let length = f32::hypot(dx, dy);
    if length == 0.0 {
        return None;
    }

    let steps = ((length / CONTRAST_SAMPLE_STEP).ceil() as usize).max(1);

    let offsets = [BAND_OFFSET - BAND_HALF_WIDTH, BAND_OFFSET + BAND_HALF_WIDTH];
    let mut outside = Vec::with_capacity(steps * offsets.len());
    let mut inside = Vec::with_capacity(steps * offsets.len());

    for step in 0..steps {
        let t = coordinate(step) / coordinate(steps - 1);
        let px = start.0 + dx * t;
        let py = start.1 + dy * t;

        for &offset in &offsets {
            let ox = px + outward[0] * offset;
            let oy = py + outward[1] * offset;
            let ix = px - outward[0] * offset;
            let iy = py - outward[1] * offset;

            outside.push(sample_gray(image, width, height, ox, oy));
            inside.push(sample_gray(image, width, height, ix, iy));
        }
    }

    let out_level = percentile(&outside, BAND_QUANTILE);
    let in_level = percentile(&inside, BAND_QUANTILE);
    let diff = out_level - in_level;
    if out_level < OUTSIDE_BASE_MIN * base
        || out_level > OUTSIDE_BASE_MAX * base
        || diff < MIN_CONTRAST_STEP
    {
        return None;
    }
    Some(diff)
}

/// Bilinear sample of the working grayscale image, clamped to the edge.
fn sample_gray(image: &GrayImage, width: usize, height: usize, x: f32, y: f32) -> f32 {
    let last_x = coordinate(width.saturating_sub(1));
    let last_y = coordinate(height.saturating_sub(1));
    let clamp_x = |v: f32| v.clamp(0.0, last_x);
    let clamp_y = |v: f32| v.clamp(0.0, last_y);
    let (x, y) = (clamp_x(x), clamp_y(y));

    let x0 = floor_to_usize(x);
    let y0 = floor_to_usize(y);
    let x1 = (x0 + 1).min(width.saturating_sub(1));
    let y1 = (y0 + 1).min(height.saturating_sub(1));
    let fx = x - coordinate(x0);
    let fy = y - coordinate(y0);

    let top = image.get(x0, y0) * (1.0 - fx) + image.get(x1, y0) * fx;
    let bottom = image.get(x0, y1) * (1.0 - fx) + image.get(x1, y1) * fx;
    top * (1.0 - fy) + bottom * fy
}

/// The lowest channel value, which is what the orange mask suppresses.
///
/// Bare light and clear film both pass it; only the picture holds it back. Using
/// the least channel rather than a mean keeps a bright but coloured scene from
/// reading as clear film.
fn least_channel(pixel: &[f32; 3]) -> f32 {
    pixel[0].min(pixel[1]).min(pixel[2])
}

/// Tilt below which rotating costs more than it corrects.
///
/// At 0.001 rad across a 6000-pixel frame the drift is about three pixels, and a
/// straight integer crop is preferable to resampling the whole picture for it.
const NEGLIGIBLE_SKEW: f32 = 0.001;

/// Applies a frame: deskews when needed, then crops.
///
/// A negligible tilt takes the straight crop path, which resamples nothing. That
/// matters because resampling is the only step here that alters pixels, and grain
/// does not survive being interpolated more than once.
///
/// The result is always a rectangle. Nothing here modifies the source image, and
/// the caller can always ask for the uncropped frame instead, so the decision is
/// reversible.
#[must_use]
pub fn apply(image: &LinearImage, frame: &Frame) -> LinearImage {
    let cropped = if frame.skew.abs() < NEGLIGIBLE_SKEW {
        crop(image, frame)
    } else {
        deskew_and_crop(image, frame)
    };

    if cropped.width == 0 || cropped.height == 0 {
        image.clone()
    } else {
        cropped
    }
}

/// The integer rectangle a frame covers, clamped to the image.
fn crop_bounds(image: &LinearImage, frame: &Frame) -> (usize, usize, usize, usize) {
    let width = image.width as usize;
    let height = image.height as usize;

    let half = [frame.size[0] / 2.0, frame.size[1] / 2.0];
    let x0 = frame.center[0] - half[0];
    let y0 = frame.center[1] - half[1];
    let x1 = frame.center[0] + half[0];
    let y1 = frame.center[1] + half[1];

    (
        floor_to_usize(x0).min(width),
        floor_to_usize(y0).min(height),
        ceil_to_usize(x1).min(width),
        ceil_to_usize(y1).min(height),
    )
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a frame coordinate: the value is a position within the image, so it \
              fits usize once floored, and negatives clamp to zero"
)]
fn floor_to_usize(value: f32) -> usize {
    if value <= 0.0 {
        return 0;
    }
    value.floor() as usize
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a frame coordinate: the value is a position within the image, so it \
              fits usize once ceiled, and negatives clamp to zero"
)]
fn ceil_to_usize(value: f32) -> usize {
    if value <= 0.0 {
        return 0;
    }
    value.ceil() as usize
}

/// Copies the frame's rectangle out, resampling nothing.
fn crop(image: &LinearImage, frame: &Frame) -> LinearImage {
    let (x0, y0, x1, y1) = crop_bounds(image, frame);
    let (out_width, out_height) = (x1.saturating_sub(x0), y1.saturating_sub(y0));
    let width = image.width as usize;

    let (pixels, _remainder) = image.data.as_chunks::<3>();
    let mut data = Vec::with_capacity(out_width * out_height * 3);
    for row in y0..y1 {
        for column in x0..x1 {
            data.extend_from_slice(&pixels[row * width + column]);
        }
    }

    LinearImage::new(
        u32::try_from(out_width).unwrap_or(image.width),
        u32::try_from(out_height).unwrap_or(image.height),
        data,
    )
    .unwrap_or_else(|_| image.clone())
}

/// Rotates the frame upright and copies it out.
///
/// Maps each output pixel back into the source, which is the direction that
/// leaves no gaps: every output pixel gets a source sample, rather than every
/// source pixel being scattered somewhere.
fn deskew_and_crop(image: &LinearImage, frame: &Frame) -> LinearImage {
    let out_width = frame.size[0].round().max(1.0);
    let out_height = frame.size[1].round().max(1.0);
    let (out_w, out_h) = (as_usize(out_width), as_usize(out_height));

    let (sin, cos) = frame.skew.sin_cos();
    let (half_w, half_h) = (out_width / 2.0, out_height / 2.0);
    let (cx, cy) = (frame.center[0], frame.center[1]);

    let mut data = Vec::with_capacity(out_w * out_h * 3);
    for row in 0..out_h {
        for column in 0..out_w {
            let dx = coordinate(column) + 0.5 - half_w;
            let dy = coordinate(row) + 0.5 - half_h;
            // Inverse of "rotate the image by -skew".
            let sx = cx + cos * dx - sin * dy;
            let sy = cy + sin * dx + cos * dy;
            data.extend_from_slice(&sample(image, sx, sy));
        }
    }

    LinearImage::new(
        u32::try_from(out_w).unwrap_or(image.width),
        u32::try_from(out_h).unwrap_or(image.height),
        data,
    )
    .unwrap_or_else(|_| image.clone())
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "an output dimension: already rounded and bounded below by 1"
)]
fn as_usize(value: f32) -> usize {
    value as usize
}

/// Bilinear sample, clamping to the edge outside the image.
fn sample(image: &LinearImage, x: f32, y: f32) -> [f32; 3] {
    let width = image.width as usize;
    let height = image.height as usize;
    let (pixels, _remainder) = image.data.as_chunks::<3>();

    let clamp_x = |v: f32| v.clamp(0.0, coordinate(width.saturating_sub(1)));
    let clamp_y = |v: f32| v.clamp(0.0, coordinate(height.saturating_sub(1)));
    let (x, y) = (clamp_x(x), clamp_y(y));

    let x0 = floor_to_usize(x);
    let y0 = floor_to_usize(y);
    let x1 = (x0 + 1).min(width.saturating_sub(1));
    let y1 = (y0 + 1).min(height.saturating_sub(1));
    let (fx, fy) = (x - coordinate(x0), y - coordinate(y0));

    let mut out = [0.0f32; 3];
    for (channel, slot) in out.iter_mut().enumerate() {
        let top =
            pixels[y0 * width + x0][channel] * (1.0 - fx) + pixels[y0 * width + x1][channel] * fx;
        let bottom =
            pixels[y1 * width + x0][channel] * (1.0 - fx) + pixels[y1 * width + x1][channel] * fx;
        *slot = top * (1.0 - fy) + bottom * fy;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Clear film, much brighter than any picture.
    const CLEAR_FILM: f32 = 0.12;
    /// A dense picture.
    const PICTURE: f32 = 0.02;

    /// A picture rectangle on clear film, optionally rotated.
    fn scan_with_frame(width: u32, height: u32, frame: [f32; 4], skew: f32) -> LinearImage {
        let count = width as usize * height as usize;
        let mut data = vec![0.0f32; count * 3];

        let (cx, cy) = (
            coordinate(width as usize) / 2.0,
            coordinate(height as usize) / 2.0,
        );
        let (cos, sin) = (skew.cos(), skew.sin());

        for row in 0..height as usize {
            for column in 0..width as usize {
                let (x, y) = (coordinate(column) - cx, coordinate(row) - cy);
                // Into frame-local coordinates.
                let (fx, fy) = (x * cos + y * sin, -x * sin + y * cos);
                let inside = fx >= frame[0] && fx <= frame[2] && fy >= frame[1] && fy <= frame[3];

                let value = if inside { PICTURE } else { CLEAR_FILM };
                let at = (row * width as usize + column) * 3;
                data[at..at + 3].fill(value);
            }
        }

        LinearImage::new(width, height, data).expect("well-formed")
    }

    #[test]
    fn an_empty_image_has_no_frame() {
        let empty = LinearImage::new(0, 0, Vec::new()).expect("well-formed");
        assert_eq!(find(&empty), None);
    }

    #[test]
    fn an_image_that_is_all_picture_is_the_whole_image() {
        // No clear film anywhere, so there is no boundary. Saying so beats
        // inventing a crop.
        let image = scan_with_frame(64, 48, [-1000.0, -1000.0, 1000.0, 1000.0], 0.0);
        let frame = find(&image).expect("a frame");
        assert_eq!(frame.size, [64.0, 48.0]);
        assert!(frame.skew.abs() < 1e-3);
    }

    #[test]
    fn an_axis_aligned_frame_is_found_without_rotation() {
        // 100x80 image; picture from x 20..80 and y 10..70.
        let image = scan_with_frame(100, 80, [-30.0, -30.0, 30.0, 30.0], 0.0);
        let frame = find(&image).expect("a frame");

        assert!(
            frame.skew.abs() < 0.01,
            "skew should be ~0, got {}",
            frame.skew
        );
        assert!(
            (frame.size[0] - 60.0).abs() < 3.0,
            "width should be ~60, got {:?}",
            frame.size
        );
        assert!(
            (frame.size[1] - 60.0).abs() < 3.0,
            "height should be ~60, got {:?}",
            frame.size
        );
        assert!(
            (frame.center[0] - 50.0).abs() < 3.0,
            "centre should be ~50,40, got {:?}",
            frame.center
        );
    }

    #[test]
    fn a_tilted_frame_reports_its_tilt() {
        let skew = 0.06f32; // about 3.4 degrees
        let image = scan_with_frame(200, 200, [-60.0, -40.0, 60.0, 40.0], skew);
        let frame = find(&image).expect("a frame");

        assert!(
            (frame.skew - skew).abs() < 0.02,
            "expected skew near {skew}, got {}",
            frame.skew
        );
    }

    #[test]
    fn the_sign_of_the_tilt_is_reported() {
        let image = scan_with_frame(200, 200, [-60.0, -40.0, 60.0, 40.0], -0.06);
        let frame = find(&image).expect("a frame");
        assert!(frame.skew < 0.0, "got {}", frame.skew);
    }

    #[test]
    fn a_picture_that_does_not_reach_the_edges_is_bounded() {
        // Clear film on all four sides, as the real scan has: the picture must be
        // bounded by it, not spill to the sensor edge.
        let image = scan_with_frame(200, 200, [-90.0, -60.0, 80.0, 80.0], 0.0);
        let frame = find(&image).expect("a frame");

        assert!(
            frame.size[1] < 180.0,
            "height should be bounded by the clear film, got {:?}",
            frame.size
        );
    }

    #[test]
    fn sprocket_holes_do_not_become_the_frame() {
        // Bare light in the frame must not drag the detector toward the holes.
        let mut image = scan_with_frame(200, 200, [-80.0, -60.0, 80.0, 60.0], 0.0);
        let (pixels, _remainder) = image.data.as_chunks_mut::<3>();
        // A hole band above the picture, as a real scan has. Kept small: bare
        // light covering more than a few percent of the frame stops reading as
        // "not film", and that is a limitation rather than a test to defeat.
        for row in 10..20usize {
            for column in 10..190usize {
                if (column / 12) % 2 == 0 {
                    pixels[row * 200 + column] = [0.6, 0.6, 0.6];
                }
            }
        }

        let frame = find(&image).expect("a frame");
        assert!(
            (frame.size[1] - 120.0).abs() < 8.0,
            "the picture is 120 tall and the hole band is above it, got {:?}",
            frame.size
        );
    }

    #[test]
    fn a_next_frame_past_the_gap_is_not_part_of_this_frame() {
        // A scan that catches the start of the *next* frame, smaller than this one.
        // Taking a row's first and last picture pixel would swallow the gap and the
        // next frame with it.
        let mut image = scan_with_frame(400, 200, [-160.0, -80.0, 0.0, 80.0], 0.0);
        let (pixels, _remainder) = image.data.as_chunks_mut::<3>();
        for row in 0..200usize {
            for column in 0..400usize {
                // The gap, then the next frame, then clear film again.
                if (200..220).contains(&column) {
                    pixels[row * 400 + column] = [CLEAR_FILM; 3];
                } else if (220..280).contains(&column) {
                    pixels[row * 400 + column] = [PICTURE; 3];
                } else if column >= 280 {
                    pixels[row * 400 + column] = [CLEAR_FILM; 3];
                }
            }
        }

        let frame = find(&image).expect("a frame");
        assert!(
            (frame.size[0] - 160.0).abs() < 6.0,
            "the frame should stop at the gap rather than run into the next one, got {:?}",
            frame.size
        );
    }

    #[test]
    fn a_uniform_image_has_no_frame_to_find() {
        let image = LinearImage::new(4, 4, vec![0.05; 48]).expect("well-formed");
        let frame = find(&image).expect("a frame");
        assert_eq!(frame.size, [4.0, 4.0]);
    }

    /// Share of pixels that read as clear film rather than picture.
    fn clear_film_share(image: &LinearImage) -> f32 {
        let (pixels, _remainder) = image.data.as_chunks::<3>();
        let clear = pixels
            .iter()
            .filter(|pixel| least_channel(pixel) > f32::midpoint(PICTURE, CLEAR_FILM))
            .count();
        coordinate(clear) / coordinate(pixels.len().max(1))
    }

    #[test]
    fn an_axis_aligned_crop_extracts_the_frame() {
        let image = scan_with_frame(100, 80, [-30.0, -30.0, 30.0, 30.0], 0.0);
        let frame = find(&image).expect("a frame");
        let cropped = apply(&image, &frame);

        assert_eq!((cropped.width, cropped.height), (60, 60));
        assert_eq!(
            clear_film_share(&cropped),
            0.0,
            "a clean crop holds only picture"
        );
    }

    #[test]
    fn cropping_resamples_nothing_when_there_is_no_tilt() {
        // The whole point of the straight path: grain does not survive being
        // interpolated, so an untilted crop must copy pixels exactly.
        let image = scan_with_frame(100, 80, [-30.0, -30.0, 30.0, 30.0], 0.0);
        let frame = find(&image).expect("a frame");
        let cropped = apply(&image, &frame);

        for sample in &cropped.data {
            assert!(
                (*sample - PICTURE).abs() < 1e-9,
                "expected an exact copy, got {sample}"
            );
        }
    }

    #[test]
    fn deskewing_a_tilted_frame_removes_the_clear_film() {
        // The decisive test for the rotation direction. An absolute bound alone
        // would be brittle, because the crop is deliberately a pixel or two
        // generous — clipping the picture is worse than keeping a sliver of rebate.
        // So the comparison is against the same frame rotated the wrong way.
        let image = scan_with_frame(240, 240, [-90.0, -70.0, 90.0, 70.0], 0.08);
        let frame = find(&image).expect("a frame");
        assert!(frame.skew.abs() > NEGLIGIBLE_SKEW, "this frame is tilted");

        let deskewed = clear_film_share(&apply(&image, &frame));
        let flipped = clear_film_share(&apply(
            &image,
            &Frame {
                skew: -frame.skew,
                ..frame
            },
        ));

        assert!(
            deskewed * 2.0 < flipped,
            "the correct rotation left {deskewed} clear film and the wrong one \
             {flipped}; they should differ substantially"
        );
        assert!(
            deskewed < 0.05,
            "deskew left {deskewed} of the frame as film"
        );
    }

    #[test]
    fn an_untilted_frame_cropped_from_a_tilted_image_still_reads_as_picture() {
        // Not every scan needs deskewing; this pins that a frame found with a tiny
        // tilt still crops to the picture.
        let image = scan_with_frame(240, 240, [-90.0, -70.0, 90.0, 70.0], 0.0);
        let frame = find(&image).expect("a frame");
        let cropped = apply(&image, &frame);
        assert!(clear_film_share(&cropped) < 0.02);
    }

    #[test]
    fn the_sampling_region_insets_as_agreed() {
        // 20% width, 10% height, per side; centre and skew carried.
        let whole = Frame {
            center: [100.0, 50.0],
            size: [100.0, 50.0],
            skew: 0.01,
        };
        let region = whole.sample_region();
        assert!(
            (region.size[0] - 60.0).abs() < 1e-4 && (region.size[1] - 40.0).abs() < 1e-5,
            "got {region:?}"
        );
        assert_eq!(region.center, whole.center);
        assert_eq!(region.skew, whole.skew);
    }

    #[test]
    fn the_sampling_region_stays_inside_its_frame() {
        let whole = Frame::whole(
            &LinearImage::new(200, 100, vec![0.0; 200 * 100 * 3]).expect("well-formed"),
        );
        let region = whole.sample_region();
        // The region's diagonal never leaves the frame: sample corners sit in.
        let half = [region.size[0] / 2.0, region.size[1] / 2.0];
        assert!(half[0] < whole.size[0] / 2.0 && half[1] < whole.size[1] / 2.0);
    }

    #[test]
    fn an_empty_frame_leaves_the_image_alone() {
        let image = scan_with_frame(64, 48, [-10.0, -10.0, 10.0, 10.0], 0.0);
        let empty = Frame {
            center: [32.0, 24.0],
            size: [0.0, 0.0],
            skew: 0.0,
        };
        let result = apply(&image, &empty);
        assert_eq!((result.width, result.height), (64, 48));
    }

    #[test]
    fn an_interior_bright_patch_does_not_become_the_frame() {
        // Mirrors the measured real failure: bright picture regions whose level
        // approaches clear film. There is no level that can refute a *straight,
        // full-length* boundary of rebate-level content — the levels agree by
        // construction — but real thin picture areas never grow that shape:
        // they are irregular, the way content is. The patch here is shaped like
        // an actual bright content region, and the frame must still win.
        let width = 400u32;
        let height = 300u32;
        let mut data = vec![0.0f32; width as usize * height as usize * 3];

        // Picture spans 0.001–0.028, clear film is 0.06.
        let picture_left = 60usize;
        let picture_right = 340usize;
        let picture_top = 40usize;
        let picture_bottom = 260usize;

        for row in 0..height as usize {
            for column in 0..width as usize {
                let value = if row < picture_top
                    || row >= picture_bottom
                    || column < picture_left
                    || column >= picture_right
                {
                    0.06 // clear film
                } else {
                    // Vary the picture level so it is a continuum, not a single
                    // population that a level threshold could split.
                    let base = 0.001
                        + 0.027 * coordinate(column - picture_left)
                            / coordinate(picture_right - picture_left);
                    // A bright, irregular content region: a highlight blob
                    // centred on the picture, whose level reaches clear-film
                    // values and whose boundary is content-shaped, not straight.
                    let dx = coordinate(column) - 200.0;
                    let dy = coordinate(row) - 150.0;
                    let distance = f32::hypot(dx, dy * 1.5);
                    let blob = (45.0 - distance).max(0.0) / 45.0;
                    let highlight = blob * (0.04 + 0.012 * (coordinate(row) / 7.0).sin());
                    (base + highlight).min(0.058)
                };
                let at = (row * width as usize + column) * 3;
                data[at..at + 3].fill(value);
            }
        }

        let image = LinearImage::new(width, height, data).expect("well-formed");
        let frame = find(&image).expect("a frame");

        // The true picture width is 280 px; a frame formed from the patch edges
        // would be only 40 px wide.
        assert!(
            frame.size[0] > 200.0,
            "the true frame should be much wider than the patch, got {:?}",
            frame.size
        );
        assert!(
            frame.skew.abs() < 0.02,
            "the synthetic image is not tilted, got {}",
            frame.skew
        );

        let cropped = apply(&image, &frame);
        assert!(
            clear_film_share(&cropped) < 0.05,
            "the crop should be almost entirely picture"
        );
    }
}
