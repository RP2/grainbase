//! Finding the film frame in a scan: where it is, and how far it is tilted.
//!
//! Home scans are rarely straight, so a crop has to know the frame's angle as
//! well as its position: an axis-aligned box around a tilted frame either clips
//! the picture or keeps the rebate in the corners.
//!
//! The boundary being looked for is **exposed picture against clear film**, not
//! film against light. In a typical camera scan the film fills the frame and the
//! only bare light is whatever shows through the sprocket holes; the rebate and
//! the inter-frame gaps are film, just unexposed. Clear film is the brightest
//! thing the *film* shows, because nothing in it absorbs, and any exposure at all
//! makes the picture darker. Measured on a real frame the separation is about 5x,
//! which is enough to threshold.
//!
//! The threshold is found by Otsu's method rather than fixed, because the level
//! depends on the film, the light source, and the exposure. A fixed constant would
//! be a guess that happens to work on one rig.
//!
//! This module is pure: image in, geometry out, no I/O.

use crate::image::LinearImage;
use crate::pipeline::{FilmBase, film_base};

/// A row needs at least this share of picture pixels to count toward the extent,
/// which skips the ragged corners where two edges meet.
const MIN_COVERAGE: f32 = 0.25;

/// Share of rows, at each end, skipped when fitting an edge.
const EDGE_MARGIN_NUMERATOR: usize = 1;
const EDGE_MARGIN_DENOMINATOR: usize = 20; // 5%

/// Largest tilt treated as genuine. Beyond this it is not a crooked scan, it is a
/// frame found wrongly, and rotating by it would wreck the picture.
const MAX_SKEW: f32 = 0.20; // radians, about 11 degrees

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
}

/// Finds the film frame, or `None` when the image has no pixels.
///
/// Falls back to [`Frame::whole`] whenever there is nothing to measure: no
/// separation between picture and clear film, or a tilt too large to be a crooked
/// scan.
#[must_use]
pub fn find(image: &LinearImage) -> Option<Frame> {
    let (width, height) = (image.width as usize, image.height as usize);
    if width == 0 || height == 0 {
        return None;
    }

    let Some(threshold) = picture_threshold(image) else {
        return Some(Frame::whole(image));
    };

    let measured = extents(image, threshold);
    let picture_rows: Vec<usize> = (0..height)
        .filter(|row| measured.coverage[*row] >= MIN_COVERAGE)
        .collect();

    // No row is mostly picture, or every row is. Either way there is no boundary
    // to find, and inventing one would be worse than declining.
    if picture_rows.is_empty() || picture_rows.len() >= height {
        return Some(Frame::whole(image));
    }

    let left_edge = slope_of(&picture_rows, |row| measured.left[row]);
    let right_edge = slope_of(&picture_rows, |row| measured.right[row]);

    // A near-vertical edge of a frame rotated by theta has dx/dy = -tan(theta),
    // so the skew is the negation of the fitted slope.
    let Some(skew) = combine(&[left_edge, right_edge])
        .map(|slope| -slope)
        .filter(|slope| slope.abs() <= MAX_SKEW)
    else {
        return Some(Frame::whole(image));
    };

    let first = picture_rows[0];
    let last = picture_rows[picture_rows.len() - 1];
    let left = median(&measured.left[first..=last]);
    let right = median(&measured.right[first..=last]);

    Some(Frame {
        center: [
            f32::midpoint(left, right),
            f32::midpoint(coordinate(first), coordinate(last)),
        ],
        size: [right - left, coordinate(last) - coordinate(first)],
        skew,
    })
}

/// Per-row picture extent.
struct Extents {
    left: Vec<f32>,
    right: Vec<f32>,
    coverage: Vec<f32>,
}

/// The lowest channel value, which is what the orange mask suppresses.
///
/// Bare light and clear film both pass it; only the picture holds it back. Using
/// the least channel rather than a mean keeps a bright but coloured scene from
/// reading as clear film.
fn least_channel(pixel: &[f32; 3]) -> f32 {
    pixel[0].min(pixel[1]).min(pixel[2])
}

/// How far below the clear-film level the picture starts, as a fraction.
///
/// Clear film sits *at* the base: nothing in it absorbs. Any exposure puts the
/// picture below that, so the boundary is the base level less a margin for noise
/// in the clear film itself. 0.8 is about a tenth of a density step.
const PICTURE_FRACTION: f32 = 0.8;

/// The highest least-channel value that is still film, which is clear film.
///
/// Reuses the film-base detector rather than measuring brightness here: the base
/// *is* the clear-film level, and that detector already knows to ignore the
/// sprocket holes, which are not film at all.
fn clear_film_level(image: &LinearImage) -> Option<f32> {
    let base = FilmBase::detect(image, film_base::DEFAULT_PERCENTILE);
    let level = base
        .transmittance
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);

    (level.is_finite() && level > 0.0).then_some(level)
}

/// The level separating clear film from exposed picture.
///
/// Returns `None` when there is nothing to separate.
///
/// Deliberately *not* a global threshold like Otsu's. The film population is a
/// continuum from dense to thin with no bimodality between "picture" and "clear
/// film", so a variance-maximising split lands inside the picture's own tonal
/// range, fragmenting every row and reading the fragmentation as a tilt. Measured:
/// 8 degrees of skew on a frame that is straight.
fn picture_threshold(image: &LinearImage) -> Option<f32> {
    clear_film_level(image).map(|level| level * PICTURE_FRACTION)
}

/// Measures each row's picture extent.
///
/// A row is scanned for runs of picture pixels, and the longest run wins — with
/// small gaps *inside* a run tolerated and large ones not.
///
/// Both extremes fail on their own. A row's first-to-last picture pixel swallows
/// the inter-frame gap and the next frame with it. A strict longest-run breaks
/// wherever the picture varies across the threshold, so the run's start jumps
/// between rows and the edge fit reads that as tilt — measured at 6.7 degrees on a
/// frame that is straight. The two cases differ only in size: a few percent of the
/// width is threshold noise, and the inter-frame gap is wider than that.
fn extents(image: &LinearImage, threshold: f32) -> Extents {
    let width = image.width as usize;
    let height = image.height as usize;
    let (pixels, _remainder) = image.data.as_chunks::<3>();
    let max_gap = (width / 32).max(1);

    let mut left = vec![0.0f32; height];
    let mut right = vec![0.0f32; height];
    let mut coverage = vec![0.0f32; height];

    for row in 0..height {
        let mut best = (0usize, 0usize);
        let mut run_start: Option<usize> = None;
        let mut last_picture = 0usize;

        for column in 0..width {
            if least_channel(&pixels[row * width + column]) >= threshold {
                continue;
            }
            match run_start {
                None => run_start = Some(column),
                Some(start) if column - last_picture > max_gap => {
                    best = longer(best, (start, last_picture + 1));
                    run_start = Some(column);
                }
                Some(_) => {}
            }
            last_picture = column;
        }

        if let Some(start) = run_start {
            best = longer(best, (start, last_picture + 1));
        }

        let (start, end) = best;
        left[row] = coordinate(start);
        // `end` is exclusive; the fallback keeps a row with no picture from
        // claiming the whole width.
        right[row] = coordinate(end.saturating_sub(1).max(start));
        coverage[row] = coordinate(end - start) / coordinate(width.max(1));
    }

    Extents {
        left,
        right,
        coverage,
    }
}

/// Slope of an edge, as `dx/dy`, fitted over the middle of the rows.
///
/// The ends are skipped because that is where one edge meets the next and bends.
fn slope_of(rows: &[usize], position: impl Fn(usize) -> f32) -> Option<f32> {
    let count = rows.len();
    if count < 8 {
        return None;
    }
    // Integer arithmetic: a count-to-float round trip is exactly the kind of
    // precision loss that bit the percentile code.
    let margin = (count * EDGE_MARGIN_NUMERATOR / EDGE_MARGIN_DENOMINATOR).max(1);
    let middle = &rows[margin..count - margin];

    let n = coordinate(middle.len());
    let mean_row = middle.iter().map(|row| coordinate(*row)).sum::<f32>() / n;
    let mean_position = middle.iter().map(|row| position(*row)).sum::<f32>() / n;

    let mut covariance = 0.0f32;
    let mut variance = 0.0f32;
    for row in middle {
        let dy = coordinate(*row) - mean_row;
        covariance += dy * (position(*row) - mean_position);
        variance += dy * dy;
    }

    (variance > 0.0).then_some(covariance / variance)
}

/// The longer of two `(start, end)` runs.
fn longer(a: (usize, usize), b: (usize, usize)) -> (usize, usize) {
    if b.1 - b.0 > a.1 - a.0 { b } else { a }
}

/// Median of a slice, by value.
fn median(values: &[f32]) -> f32 {
    let mut sorted: Vec<f32> = values.to_vec();
    let middle = sorted.len() / 2;
    sorted.select_nth_unstable_by(middle, f32::total_cmp);
    sorted[middle]
}

/// Combines edge slopes into one skew.
///
/// Opposite edges of a rectangle tilt by the same amount, so the median is a
/// reasonable summary.
fn combine(slopes: &[Option<f32>]) -> Option<f32> {
    let present: Vec<f32> = slopes.iter().filter_map(|slope| *slope).collect();
    if present.is_empty() {
        return None;
    }
    Some(median(&present))
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
    fn the_threshold_lands_between_the_two_populations() {
        let image = scan_with_frame(100, 80, [-30.0, -30.0, 30.0, 30.0], 0.0);
        let threshold = picture_threshold(&image).expect("two populations");
        assert!(
            threshold > PICTURE && threshold < CLEAR_FILM,
            "threshold {threshold} should sit between {PICTURE} and {CLEAR_FILM}"
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
        // The real case: clear film along two edges, picture filling the rest.
        let image = scan_with_frame(200, 200, [-100.0, -60.0, 90.0, 100.0], 0.0);
        let frame = find(&image).expect("a frame");

        assert!(
            frame.size[1] < 180.0,
            "height should be bounded by the clear film, got {:?}",
            frame.size
        );
    }

    #[test]
    fn sprocket_holes_do_not_become_the_picture() {
        // Bare light in the frame must not drag the threshold toward it.
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

        let threshold = picture_threshold(&image).expect("two populations");
        assert!(
            threshold < CLEAR_FILM * 0.9,
            "the holes should be excluded from the split, got threshold {threshold}"
        );

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
    fn a_uniform_image_has_no_threshold_to_find() {
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
}
