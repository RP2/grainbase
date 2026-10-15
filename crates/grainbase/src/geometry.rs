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

/// Samples at or above this are bare light through a hole, not film at all.
///
/// Excluded before the threshold is chosen, or the holes dominate the split and
/// the result separates "film" from "holes" instead of "picture" from "clear
/// film".
const HOLE_LEVEL: f32 = 0.30;

/// A row needs at least this share of picture pixels to count toward the extent,
/// which skips the ragged corners where two edges meet.
const MIN_COVERAGE: f32 = 0.25;

/// Share of rows, at each end, skipped when fitting an edge.
const EDGE_MARGIN_NUMERATOR: usize = 1;
const EDGE_MARGIN_DENOMINATOR: usize = 20; // 5%

/// Largest tilt treated as genuine. Beyond this it is not a crooked scan, it is a
/// frame found wrongly, and rotating by it would wreck the picture.
const MAX_SKEW: f32 = 0.20; // radians, about 11 degrees

/// Bins used to build the histogram the threshold is chosen from.
const BINS: usize = 256;

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

/// The level separating clear film from exposed picture, by Otsu's method.
///
/// Returns `None` when there is nothing to separate.
fn picture_threshold(image: &LinearImage) -> Option<f32> {
    let (pixels, _remainder) = image.data.as_chunks::<3>();

    let mut histogram = [0u32; BINS];
    let mut total = 0u64;
    for pixel in pixels {
        let value = least_channel(pixel);
        // Holes are not film; letting them in makes the split about them.
        if !value.is_finite() || value >= HOLE_LEVEL {
            continue;
        }
        let bin = bin_of(value);
        histogram[bin] = histogram[bin].saturating_add(1);
        total += 1;
    }

    otsu(&histogram, total)
}

/// The bin a sample falls in.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is clamped to 0.0..=1.0, so the scaled bin is finite, \
              non-negative, and bounded by BINS - 1"
)]
fn bin_of(value: f32) -> usize {
    ((value.clamp(0.0, 1.0) * coordinate(BINS - 1)).round() as usize).min(BINS - 1)
}

/// Otsu's threshold: the bin that best separates two populations.
///
/// Maximises the variance *between* the classes, which needs no tuning and no
/// assumption about which class is which.
fn otsu(histogram: &[u32; BINS], total: u64) -> Option<f32> {
    if total == 0 {
        return None;
    }
    let total = coordinate(usize::try_from(total).ok()?);

    let mut sum_all = 0.0f64;
    for (bin, count) in histogram.iter().enumerate() {
        sum_all += f64::from(coordinate(bin)) * f64::from(*count);
    }

    let mut weight_below = 0.0f64;
    let mut sum_below = 0.0f64;
    let mut best_bin = 0usize;
    let mut best_variance = -1.0f64;

    for (bin, count) in histogram.iter().enumerate() {
        weight_below += f64::from(*count);
        if weight_below == 0.0 {
            continue;
        }
        let weight_above = f64::from(total) - weight_below;
        if weight_above <= 0.0 {
            break;
        }
        sum_below += f64::from(coordinate(bin)) * f64::from(*count);

        let mean_below = sum_below / weight_below;
        let mean_above = (sum_all - sum_below) / weight_above;
        let between = weight_below * weight_above * (mean_below - mean_above).powi(2);
        if between > best_variance {
            best_variance = between;
            best_bin = bin;
        }
    }

    // The boundary sits *between* the last bin of the lower class and the first
    // of the upper, not on the occupied bin itself. Returning the bin value puts
    // the threshold exactly on the lower population, and every pixel of it then
    // fails a strict comparison.
    Some((coordinate(best_bin) + 0.5) / coordinate(BINS - 1))
}

/// Measures each row's picture extent.
fn extents(image: &LinearImage, threshold: f32) -> Extents {
    let width = image.width as usize;
    let height = image.height as usize;
    let (pixels, _remainder) = image.data.as_chunks::<3>();

    let mut left = vec![0.0f32; height];
    let mut right = vec![0.0f32; height];
    let mut coverage = vec![0.0f32; height];

    for row in 0..height {
        let mut first = None;
        let mut last = None;
        let mut count = 0usize;

        for column in 0..width {
            // Picture is *below* the threshold: any exposure makes it darker than
            // clear film.
            if least_channel(&pixels[row * width + column]) >= threshold {
                continue;
            }
            count += 1;
            first.get_or_insert(column);
            last = Some(column);
        }

        left[row] = coordinate(first.unwrap_or(0));
        right[row] = coordinate(last.unwrap_or(width.saturating_sub(1)));
        coverage[row] = coordinate(count) / coordinate(width.max(1));
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
        let mut image = scan_with_frame(200, 200, [-80.0, -80.0, 80.0, 80.0], 0.0);
        let (pixels, _remainder) = image.data.as_chunks_mut::<3>();
        for row in 10..30usize {
            for column in 10..190usize {
                if (column / 12) % 2 == 0 {
                    pixels[row * 200 + column] = [0.6, 0.6, 0.6];
                }
            }
        }

        let threshold = picture_threshold(&image).expect("two populations");
        assert!(
            threshold < HOLE_LEVEL,
            "the holes should be excluded, got threshold {threshold}"
        );

        let frame = find(&image).expect("a frame");
        assert!(
            (frame.size[1] - 160.0).abs() < 6.0,
            "the hole band is clear film plus holes, so the picture is still bounded there, got {:?}",
            frame.size
        );
    }

    #[test]
    fn a_uniform_image_has_no_threshold_to_find() {
        let image = LinearImage::new(4, 4, vec![0.05; 48]).expect("well-formed");
        let frame = find(&image).expect("a frame");
        assert_eq!(frame.size, [4.0, 4.0]);
    }
}
