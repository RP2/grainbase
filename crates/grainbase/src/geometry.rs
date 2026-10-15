//! Finding the film frame in a scan: where it is, and how far it is tilted.
//!
//! Home scans are rarely straight, so a crop has to know the frame's angle as
//! well as its position: an axis-aligned box around a tilted frame either clips
//! the picture or keeps the light table in the corners.
//!
//! The frame is found from brightness structure rather than image content. The
//! light table and the sprocket holes let the full source through, while film —
//! including its clear base — blocks blue almost completely because of the
//! orange mask. So the *minimum* channel is a discriminator that works whatever
//! the photograph happens to contain: a bright sky running to the frame edge,
//! which defeats content-based edge detection, is still dense film here.
//!
//! This module is pure: image in, geometry out, no I/O.

use crate::image::LinearImage;

/// Smallest channel value counted as "light got through", as a fraction of the
/// brightest sample present.
///
/// Low enough that the dimmest light table still counts, high enough that the
/// film base never does. The base is orange, so its blue channel is a small
/// fraction of what the bare light source passes.
const LIGHT_FRACTION: f32 = 0.35;

/// A row or column needs at least this share of film pixels to count toward the
/// edge fit, which skips the ragged corners where two edges meet.
const MIN_EDGE_COVERAGE: f32 = 0.25;

/// Share of rows, at each end, skipped when fitting an edge.
///
/// Held as an integer fraction so the margin is computed without a trip through
/// floating point.
const EDGE_MARGIN_NUMERATOR: usize = 1;
const EDGE_MARGIN_DENOMINATOR: usize = 20; // 5%

/// Largest tilt treated as genuine. Beyond this it is not a crooked scan, it is
/// a frame found wrongly, and rotating by it would wreck the picture.
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
    /// Width and height of the frame, in source pixels. Rotation does not
    /// change these, so they can be used after deskewing.
    pub size: [f32; 2],
    /// How far the frame is rotated, in radians. Rotating the image by the
    /// negative of this makes the frame axis-aligned.
    pub skew: f32,
}

impl Frame {
    /// The whole image as a frame: no crop, no rotation.
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
/// When no light source shows in the frame there is nothing to locate, and the
/// whole image is returned as the frame — which is the honest answer, not a
/// failure.
#[must_use]
pub fn find(image: &LinearImage) -> Option<Frame> {
    let (width, height) = (image.width as usize, image.height as usize);
    if width == 0 || height == 0 {
        return None;
    }

    let Some(threshold) = light_threshold(image) else {
        return Some(Frame::whole(image));
    };

    let Rows {
        left,
        right,
        coverage: row_coverage,
    } = row_extents(image, threshold);

    // A row with almost no film is a row of bare light, outside the frame.
    let film_rows: Vec<usize> = (0..height)
        .filter(|row| row_coverage[*row] >= MIN_EDGE_COVERAGE)
        .collect();

    if film_rows.is_empty() {
        // Everything is light: no film in this scan at all.
        return Some(Frame::whole(image));
    }

    let top = coordinate(film_rows[0]);
    let bottom = coordinate(film_rows[film_rows.len() - 1]);
    let left_edge = slope_of(&film_rows, |row| left[row]);
    let right_edge = slope_of(&film_rows, |row| right[row]);

    // A near-vertical edge of a frame rotated by theta has dx/dy = -tan(theta),
    // so the skew is the negation of the fitted slope.
    let skew = -combine(&[left_edge, right_edge]).filter(|slope| slope.abs() <= MAX_SKEW)?;

    let first = film_rows[0];
    let last = film_rows[film_rows.len() - 1];
    let (Some(leftmost), Some(rightmost)) = (left.get(first..=last), right.get(first..=last))
    else {
        return Some(Frame::whole(image));
    };
    let left_bound = median(leftmost);
    let right_bound = median(rightmost);

    Some(Frame {
        center: [
            f32::midpoint(left_bound, right_bound),
            f32::midpoint(top, bottom),
        ],
        size: [right_bound - left_bound, bottom - top],
        skew,
    })
}

/// Per-row film extent.
struct Rows {
    left: Vec<f32>,
    right: Vec<f32>,
    coverage: Vec<f32>,
}

/// Brightness above which a sample is bare light rather than film, if any.
///
/// The brightest samples in the image are the light source passing through
/// nothing. Taking a fraction of that separates light from film without
/// assuming an exposure.
fn light_threshold(image: &LinearImage) -> Option<f32> {
    let brightest = image
        .data
        .iter()
        .copied()
        .filter(|sample| sample.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);

    (brightest > 0.0).then_some(brightest * LIGHT_FRACTION)
}

/// The lowest channel value, which is what the orange mask suppresses.
fn min_channel(pixel: &[f32; 3]) -> f32 {
    pixel[0].min(pixel[1]).min(pixel[2])
}

/// Measures each row's film extent.
fn row_extents(image: &LinearImage, threshold: f32) -> Rows {
    let width = image.width as usize;
    let height = image.height as usize;
    let (pixels, _) = image.data.as_chunks::<3>();

    let mut left = vec![0.0f32; height];
    let mut right = vec![0.0f32; height];
    let mut coverage = vec![0.0f32; height];

    for row in 0..height {
        let mut first = None;
        let mut last = None;
        let mut count = 0usize;

        for column in 0..width {
            if min_channel(&pixels[row * width + column]) >= threshold {
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

    Rows {
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

/// Combines edge slopes into one skew, rejecting disagreement.
///
/// Opposite edges of a rectangle tilt by the same amount, so a large
/// disagreement means an edge was fitted to something that is not the frame.
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

    /// A dark rectangle on a bright background, optionally rotated.
    fn scan_with_frame(
        width: u32,
        height: u32,
        frame: [f32; 4],
        skew: f32,
        background: f32,
        film: f32,
    ) -> LinearImage {
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

                let value = if inside { film } else { background };
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
    fn a_frame_with_no_light_around_it_is_the_whole_image() {
        // All film: there is nothing to locate, and saying so beats inventing a crop.
        let image = scan_with_frame(64, 48, [-100.0, -100.0, 100.0, 100.0], 0.0, 0.2, 0.2);
        let frame = find(&image).expect("a frame");
        assert_eq!(frame.size, [64.0, 48.0]);
        assert!(frame.skew.abs() < 1e-3);
    }

    #[test]
    fn an_axis_aligned_frame_is_found_without_rotation() {
        // 100x80 image, film from x 20..80 and y 10..70.
        let image = scan_with_frame(100, 80, [-30.0, -30.0, 30.0, 30.0], 0.0, 1.0, 0.1);
        let frame = find(&image).expect("a frame");

        assert!(
            frame.skew.abs() < 0.01,
            "skew should be ~0, got {}",
            frame.skew
        );
        assert!((frame.size[0] - 60.0).abs() < 2.0, "got {:?}", frame.size);
        assert!((frame.size[1] - 60.0).abs() < 2.0, "got {:?}", frame.size);
        assert!(
            (frame.center[0] - 50.0).abs() < 2.0,
            "got {:?}",
            frame.center
        );
        assert!(
            (frame.center[1] - 40.0).abs() < 2.0,
            "got {:?}",
            frame.center
        );
    }

    #[test]
    fn a_tilted_frame_reports_its_tilt() {
        let skew = 0.06f32; // about 3.4 degrees
        let image = scan_with_frame(200, 200, [-60.0, -40.0, 60.0, 40.0], skew, 1.0, 0.1);
        let frame = find(&image).expect("a frame");

        assert!(
            (frame.skew - skew).abs() < 0.02,
            "expected skew near {skew}, got {}",
            frame.skew
        );
    }

    #[test]
    fn the_sign_of_the_tilt_is_reported() {
        let image = scan_with_frame(200, 200, [-60.0, -40.0, 60.0, 40.0], -0.06, 1.0, 0.1);
        let frame = find(&image).expect("a frame");
        assert!(frame.skew < 0.0, "got {}", frame.skew);
    }

    #[test]
    fn a_dark_scene_inside_a_bright_frame_still_finds_the_frame() {
        // The point of using the least channel: a bright sky inside the picture
        // must not be mistaken for the light source.
        let mut image = scan_with_frame(100, 80, [-30.0, -30.0, 30.0, 30.0], 0.0, 1.0, 0.1);
        let (pixels, _) = image.data.as_chunks_mut::<3>();
        // Paint a very bright patch inside the frame, but keep blue suppressed as
        // film would: bright, yet not bare light.
        for row in 20..40usize {
            for column in 30..60usize {
                pixels[row * 100 + column] = [1.0, 0.9, 0.1];
            }
        }

        let frame = find(&image).expect("a frame");
        assert!((frame.size[0] - 60.0).abs() < 3.0, "got {:?}", frame.size);
    }
}
