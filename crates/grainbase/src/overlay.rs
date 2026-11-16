//! Drawing the conversion's own decisions onto the scan.
//!
//! A conversion is a chain of measurements a reader cannot see: where the frame
//! is, which rows are the punch, which pixels the film base was sampled from.
//! When one of them is wrong, the output is wrong in a way no number in a log
//! explains — and a bug report is worth ten times more when it carries a picture
//! of what the pipeline thought it was looking at.
//!
//! Diagnostic only. Nothing here feeds a conversion, and the marks are drawn on
//! a copy: the scan that gets converted is never touched.

use crate::geometry::Frame;
use crate::image::LinearImage;

/// What a crop cut out, drawn only when a crop was asked for.
const FRAME_COLOR: [f32; 3] = [1.0, 0.0, 0.0];

/// The sprocket lines: the punch bands, drawn as their edges.
///
/// Bare light at the white balance, and the one structure a scan states
/// reliably. Everything between them is film; everything beyond them is not.
const HOLE_COLOR: [f32; 3] = [1.0, 1.0, 0.0];

/// The points the film base was measured from, between the sprocket lines.
const SAMPLE_COLOR: [f32; 3] = [1.0, 0.0, 1.0];

/// The sprocket holes themselves: bare light, blocked from every sample.
const SPROCKET_COLOR: [f32; 3] = [1.0, 0.5, 0.0];

/// The region anything that reads the *picture* samples: the frame pulled in,
/// so an approximate frame still lands inside the exposure.
const COLOR_COLOR: [f32; 3] = [0.0, 1.0, 0.0];

/// The film between the punch bands, drawn as its edges.
///
/// The region a base sample *could* come from; the sample points themselves are
/// the marks that say where it did.
const FILM_COLOR: [f32; 3] = [0.0, 1.0, 1.0];

/// What to draw on the scan.
///
/// Every field is optional in the sense that an empty slice or a zero-sized
/// window draws nothing: a scan with no frame found, or with no probes, still
/// produces a usable annotated image.
pub struct Marks<'a> {
    /// The punch bands, drawn as their edges: bare light, never sampled.
    pub holes: &'a [Option<crate::pipeline::film_base::FilmRegion>; 2],
    /// The sprocket holes, as their rectangles: `[x0, y0, x1, y1]`.
    pub sprockets: &'a [[f32; 4]],
    /// The film between the punch bands, drawn as its edges.
    pub film: Option<&'a crate::pipeline::film_base::FilmRegion>,
    /// The points the film base was measured from.
    pub samples: &'a [[f32; 2]],
    /// The region color is read from: the frame pulled in by its own insets.
    ///
    /// Vertical edges are the ones a scan states worst, so every sample of the
    /// picture — calibration fitting, quality checks — is taken inside this
    /// region rather than inside the frame. An approximate frame still lands
    /// the sample in the exposure.
    pub color_region: Option<&'a Frame>,
    /// The crop rectangle, drawn only when a crop was asked for.
    pub crop: Option<&'a Frame>,
}

/// Returns a copy of `image` with the pipeline's marks drawn on it.
#[must_use]
pub fn annotate(image: &LinearImage, marks: &Marks) -> LinearImage {
    let mut marked = image.clone();
    let width = marked.width as usize;
    let height = marked.height as usize;
    if width == 0 || height == 0 {
        return marked;
    }

    // The punch bands first, so the samples read on top of them.
    for band in marks.holes.iter().flatten() {
        for side in [-1.0f32, 1.0] {
            let edge = band.center_y + side * band.half_height;
            let (from, to) = band_ends(edge, band.slope, &marked);
            line(&mut marked, from, to, HOLE_COLOR);
        }
    }

    if let Some(region) = marks.color_region {
        let corners = frame_corners(region);
        for index in 0..4 {
            let next = (index + 1) % 4;
            line(&mut marked, corners[index], corners[next], COLOR_COLOR);
        }
    }

    if let Some(frame) = marks.crop {
        let corners = frame_corners(frame);
        for index in 0..4 {
            let next = (index + 1) % 4;
            line(&mut marked, corners[index], corners[next], FRAME_COLOR);
        }
    }

    for hole in marks.sprockets {
        let corners = [
            [hole[0], hole[1]],
            [hole[2], hole[1]],
            [hole[2], hole[3]],
            [hole[0], hole[3]],
        ];
        for index in 0..4 {
            line(
                &mut marked,
                corners[index],
                corners[(index + 1) % 4],
                SPROCKET_COLOR,
            );
        }
    }

    if let Some(film) = marks.film {
        for side in [-1.0f32, 1.0] {
            let edge = film.center_y + side * film.half_height;
            let (from, to) = band_ends(edge, film.slope, &marked);
            line(&mut marked, from, to, FILM_COLOR);
        }
    }

    for point in marks.samples {
        dot(&mut marked, *point, SAMPLE_MARK / 2, SAMPLE_COLOR);
    }

    marked
}

/// The frame's four corners in source coordinates, clockwise from the top left.
fn frame_corners(frame: &Frame) -> [[f32; 2]; 4] {
    let (sin, cos) = frame.skew.sin_cos();
    let (half_w, half_h) = (frame.size[0] / 2.0, frame.size[1] / 2.0);
    let mut corners = [[0.0f32; 2]; 4];
    for (slot, (local_x, local_y)) in corners.iter_mut().zip([
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h),
        (-half_w, half_h),
    ]) {
        *slot = [
            frame.center[0] + cos * local_x - sin * local_y,
            frame.center[1] + sin * local_x + cos * local_y,
        ];
    }
    corners
}

/// The two ends of one hole-band edge, at the image's left and right edges.
fn band_ends(edge_y: f32, slope: f32, image: &LinearImage) -> ([f32; 2], [f32; 2]) {
    let width = coordinate(image.width as usize);
    let center_x = width / 2.0;
    let y_at = |x: f32| edge_y + slope * (x - center_x);
    ([0.0, y_at(0.0)], [width - 1.0, y_at(width - 1.0)])
}

/// A pixel coordinate as a float.
///
/// `f32` represents integers exactly up to 2^24, which is sixteen million pixels
/// on one axis.
#[expect(
    clippy::cast_precision_loss,
    reason = "a coordinate beyond 2^24 on one axis is not a photograph"
)]
fn coordinate(value: usize) -> f32 {
    value as f32
}

/// Draws a straight line, stepping one pixel at a time.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "line pixels are clamped into the image before the index casts"
)]
fn line(image: &mut LinearImage, from: [f32; 2], to: [f32; 2], color: [f32; 3]) {
    let steps = (to[0] - from[0]).abs().max((to[1] - from[1]).abs()).ceil() as usize;
    let steps = steps.max(1);
    for step in 0..=steps {
        #[expect(clippy::cast_precision_loss, reason = "a line's step index is small")]
        let t = step as f32 / steps as f32;
        let x = from[0] + (to[0] - from[0]) * t;
        let y = from[1] + (to[1] - from[1]) * t;
        if x < 0.0 || y < 0.0 {
            continue;
        }
        put(image, x as usize, y as usize, color);
    }
}

/// Side of the square drawn at one sample point, in source pixels.
///
/// The sample reads a small window; this marks where, without pretending to be
/// the window itself. Sized to stay visible when a 6000-pixel scan is viewed
/// whole.
const SAMPLE_MARK: usize = 21;

/// Draws a filled square centered on a point.
fn dot(image: &mut LinearImage, center: [f32; 2], half: usize, color: [f32; 3]) {
    let (x0, y0, x1, y1) = dot_bounds(center, half);
    for y in y0..y1 {
        for x in x0..x1 {
            put(image, x, y, color);
        }
    }
}

/// The integer bounds of a drawn dot, clamped to the image.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a dot is a few pixels around a point inside the image"
)]
fn dot_bounds(center: [f32; 2], half: usize) -> (usize, usize, usize, usize) {
    let cx = center[0].round().clamp(0.0, f32::from(u16::MAX)) as usize;
    let cy = center[1].round().clamp(0.0, f32::from(u16::MAX)) as usize;
    (
        cx.saturating_sub(half),
        cy.saturating_sub(half),
        cx.saturating_add(half).saturating_add(1),
        cy.saturating_add(half).saturating_add(1),
    )
}

/// Writes one pixel, ignoring coordinates outside the image.
fn put(image: &mut LinearImage, x: usize, y: usize, color: [f32; 3]) {
    let width = image.width as usize;
    let height = image.height as usize;
    if x >= width || y >= height {
        return;
    }
    let index = (y * width + x) * 3;
    image.data[index..index + 3].copy_from_slice(&color);
}

/// Annotates a chart capture with the detected lattice and sample regions.
///
/// The image is box-downsampled so its longest side is at most
/// `max_dimension`, which keeps the output small enough to view whole while
/// preserving the coordinate relationship. `vertical` and `horizontal` are the
/// 7 and 5 separator lines in source pixels; `sample_regions` are the 24 cell
/// windows in `[x0, y0, x1, y1]` source pixels.
#[must_use]
pub fn annotate_chart(
    image: &LinearImage,
    vertical: &[f32],
    horizontal: &[f32],
    sample_regions: &[[f32; 4]],
    crosses: &[[f32; 2]; 4],
    outer_border: &[[f32; 2]; 4],
    max_dimension: u32,
) -> LinearImage {
    let (downscaled, scale) = downsample(image, max_dimension);
    let mut marked = downscaled.clone();

    let outer_scaled: Vec<_> = outer_border
        .iter()
        .map(|&[x, y]| [x / scale, y / scale])
        .collect();
    for index in 0..4 {
        let next = (index + 1) % 4;
        line(
            &mut marked,
            outer_scaled[index],
            outer_scaled[next],
            CHART_OUTER_BORDER_COLOR,
        );
    }

    let corners = [
        [vertical[0], horizontal[0]],
        [vertical[vertical.len() - 1], horizontal[0]],
        [
            vertical[vertical.len() - 1],
            horizontal[horizontal.len() - 1],
        ],
        [vertical[0], horizontal[horizontal.len() - 1]],
    ];
    let scaled_corners: Vec<_> = corners
        .iter()
        .map(|&[x, y]| [x / scale, y / scale])
        .collect();
    for index in 0..4 {
        let next = (index + 1) % 4;
        line(
            &mut marked,
            scaled_corners[index],
            scaled_corners[next],
            CHART_FRAME_COLOR,
        );
    }

    let marked_height = coordinate(marked.height as usize) - 1.0;
    let marked_width = coordinate(marked.width as usize) - 1.0;

    for &x in vertical {
        let x = x / scale;
        line(
            &mut marked,
            [x, 0.0],
            [x, marked_height],
            CHART_VERTICAL_COLOR,
        );
    }

    for &y in horizontal {
        let y = y / scale;
        line(
            &mut marked,
            [0.0, y],
            [marked_width, y],
            CHART_HORIZONTAL_COLOR,
        );
    }

    for region in sample_regions {
        let [x0, y0, x1, y1] = region.map(|v| v / scale);
        let corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
        for index in 0..4 {
            let next = (index + 1) % 4;
            line(
                &mut marked,
                corners[index],
                corners[next],
                CHART_SAMPLE_COLOR,
            );
        }
    }

    for &[x, y] in crosses {
        dot(
            &mut marked,
            [x / scale, y / scale],
            (CROSS_MARK / 2).max(1),
            CHART_CROSS_COLOR,
        );
    }

    marked
}

/// The outer chart boundary drawn in diagnostic overlays.
const CHART_FRAME_COLOR: [f32; 3] = [1.0, 0.0, 0.0];
/// The detected outer border rectangle drawn in diagnostic overlays.
const CHART_OUTER_BORDER_COLOR: [f32; 3] = [1.0, 0.5, 0.0];
/// Vertical separator lines drawn in diagnostic overlays.
const CHART_VERTICAL_COLOR: [f32; 3] = [0.0, 1.0, 0.0];
/// Horizontal separator lines drawn in diagnostic overlays.
const CHART_HORIZONTAL_COLOR: [f32; 3] = [0.0, 0.0, 1.0];
/// Per-cell sample windows drawn in diagnostic overlays.
const CHART_SAMPLE_COLOR: [f32; 3] = [1.0, 1.0, 0.0];
/// Registration crosses drawn in diagnostic overlays.
const CHART_CROSS_COLOR: [f32; 3] = [1.0, 0.0, 1.0];
/// Side of the square drawn at one cross centre, in downsampled pixels.
const CROSS_MARK: usize = 11;

/// Box-downsamples `image` so its longest side is at most `max_dimension`.
///
/// Returns the downscaled image and the source-to-downsample scale factor.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "downsample factors and dimensions are small non-negative values"
)]
fn downsample(image: &LinearImage, max_dimension: u32) -> (LinearImage, f32) {
    let max_dim = image.width.max(image.height);
    if max_dim <= max_dimension {
        return (image.clone(), 1.0);
    }

    let factor = max_dim.div_ceil(max_dimension).max(2) as usize;
    let scale = factor as f32;
    let out_w = image.width as usize / factor;
    let out_h = image.height as usize / factor;
    let mut out = vec![0.0f32; out_w * out_h * 3];
    let src_w = image.width as usize;

    let (pixels, _remainder) = image.data.as_chunks::<3>();
    for oy in 0..out_h {
        let y0 = oy * factor;
        let y1 = ((oy + 1) * factor).min(image.height as usize);
        for ox in 0..out_w {
            let x0 = ox * factor;
            let x1 = ((ox + 1) * factor).min(image.width as usize);
            let mut sum = [0.0f32; 3];
            let mut count = 0usize;
            for y in y0..y1 {
                for x in x0..x1 {
                    let pixel = pixels[y * src_w + x];
                    for channel in 0..3 {
                        sum[channel] += pixel[channel];
                    }
                    count += 1;
                }
            }
            let idx = (oy * out_w + ox) * 3;
            for channel in 0..3 {
                out[idx + channel] = sum[channel] / coordinate(count);
            }
        }
    }

    (
        LinearImage::new(out_w as u32, out_h as u32, out).expect("well-formed"),
        scale,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn white(width: u32, height: u32) -> LinearImage {
        LinearImage::new(
            width,
            height,
            vec![1.0; width as usize * height as usize * 3],
        )
        .expect("well-formed")
    }

    fn pixel(image: &LinearImage, x: usize, y: usize) -> [f32; 3] {
        let width = image.width as usize;
        let index = (y * width + x) * 3;
        [
            image.data[index],
            image.data[index + 1],
            image.data[index + 2],
        ]
    }

    #[test]
    fn a_sample_point_is_drawn_where_it_was_read() {
        let image = white(80, 80);
        let samples = [[20.0f32, 20.0], [60.0, 60.0]];
        let marks = Marks {
            holes: &[None, None],
            sprockets: &[],
            film: None,
            samples: &samples,
            color_region: None,
            crop: None,
        };

        let marked = annotate(&image, &marks);
        assert_eq!(
            pixel(&marked, 20, 20),
            SAMPLE_COLOR,
            "the sample's own pixel"
        );
        assert_eq!(pixel(&marked, 60, 60), SAMPLE_COLOR, "the second sample");
        assert_eq!(pixel(&marked, 40, 40), [1.0, 1.0, 1.0], "untouched pixel");
    }

    #[test]
    fn marks_outside_the_image_are_ignored() {
        let image = white(10, 10);
        let marks = Marks {
            holes: &[None, None],
            sprockets: &[],
            film: None,
            samples: &[[-5.0, -5.0], [50.0, 50.0]],
            color_region: None,
            crop: None,
        };

        // The region is entirely off the image, so nothing is drawn; an
        // off-image mark must not panic or wrap.
        let marked = annotate(&image, &marks);
        assert_eq!(marked.data.len(), image.data.len());
    }

    #[test]
    fn a_hole_band_is_drawn_at_its_edges() {
        let image = white(20, 40);
        let holes = [
            Some(crate::pipeline::film_base::FilmRegion {
                center_y: 5.0,
                half_height: 2.0,
                slope: 0.0,
            }),
            None,
        ];
        let marks = Marks {
            holes: &holes,
            sprockets: &[],
            film: None,
            samples: &[],
            color_region: None,
            crop: None,
        };

        let marked = annotate(&image, &marks);
        assert_eq!(pixel(&marked, 10, 3), HOLE_COLOR, "the band's near edge");
        assert_eq!(pixel(&marked, 10, 7), HOLE_COLOR, "the band's far edge");
        assert_eq!(pixel(&marked, 10, 5), [1.0, 1.0, 1.0], "between the edges");
    }
}
