//! Black-and-white negative pipeline.
//!
//! A B&W negative has no orange mask and its image is a single silver density,
//! so the color model collapses to one channel. The output is gray: all three
//! channels carry the same scalar. The color channel choice (a creative filter,
//! e.g. red-key) is a later setting; this build collapses to luminance.
//!
//! A B&W profile carries its single value in channel 0 of the shared
//! [`Settings`] arrays, and writes it in all three channels for format
//! compatibility. Crosstalk is not applied: one density channel makes the 3x3
//! matrix degenerate.

use crate::image::LinearImage;
use crate::pipeline::film_base::FilmBase;
use crate::pipeline::{
    Settings, curve_from_settings, default_density_max, density_of, render as render_density,
};

/// Rec. 709 luma coefficients.
const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Renders a B&W negative as a gray positive.
///
/// Per pixel, the three channels are collapsed to Rec. 709 luma, the base is
/// collapsed the same way, and the resulting density is mapped through the same
/// floor/ceiling/curve machinery as the color negative core. Output is gray.
#[must_use]
pub fn render(image: &LinearImage, base: &FilmBase, settings: &Settings) -> LinearImage {
    // Channel 0 carries the single B&W value; the other two channels are ignored
    // so a profile stays a single plain-text line of three identical numbers.
    let density_max = settings
        .density_max
        .unwrap_or_else(default_density_max)
        .map(|value| value.max(f32::MIN_POSITIVE))[0];
    let floor = settings.density_floor.unwrap_or([0.0; 3])[0];
    let curve = curve_from_settings(settings, [floor; 3], [density_max; 3])
        .map(|(pivot, gamma)| ([pivot[0]; 3], [gamma[0]; 3]));

    let base_luma = dot(base.transmittance, LUMA);

    image.map_rgb(|pixel| {
        let luma = dot(pixel, LUMA);
        let density = density_of(luma, base_luma);
        let value = render_density(density, floor, density_max, curve.as_ref(), 0);
        [value; 3]
    })
}

/// Dot product of two three-element vectors.
#[must_use]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{DEFAULT_DENSITY_MAX, Settings};

    fn image(pixels: &[[f32; 3]]) -> LinearImage {
        let data = pixels.iter().flatten().copied().collect();
        let count = u32::try_from(pixels.len()).expect("test image is small");
        LinearImage::new(count, 1, data).expect("test image is well-formed")
    }

    fn base(transmittance: [f32; 3]) -> FilmBase {
        FilmBase { transmittance }
    }

    fn settings() -> Settings {
        Settings {
            density_max: Some([DEFAULT_DENSITY_MAX; 3]),
            ..Settings::default()
        }
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn a_neutral_frame_inverts_to_gray() {
        let source = image(&[[0.5, 0.5, 0.5]]);
        let result = render(&source, &base([1.0, 1.0, 1.0]), &settings());
        assert_close(result.data[0], result.data[1]);
        assert_close(result.data[1], result.data[2]);
    }

    #[test]
    fn a_colored_frame_collapses_to_its_luma() {
        // Rec. 709 luma of this patch.
        let pixel = [0.9f32, 0.4, 0.2];
        let luma = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2];

        let source = image(&[pixel]);
        let result = render(&source, &base([1.0, 1.0, 1.0]), &settings());

        // The same density rendered through the same ceiling.
        let expected =
            crate::pipeline::invert(&image(&[[luma; 3]]), &base([1.0, 1.0, 1.0]), &settings());
        assert_close(result.data[0], expected.data[0]);
    }

    #[test]
    fn the_base_luma_becomes_black() {
        let source = image(&[[0.8, 0.8, 0.8]]);
        let result = render(&source, &base([0.8, 0.8, 0.8]), &settings());
        for channel in result.data {
            assert_close(channel, 0.0);
        }
    }

    #[test]
    fn a_density_at_the_ceiling_becomes_white() {
        // luma = 0.01, base = 1.0 -> density = 2.0 = ceiling.
        let source = image(&[[0.01, 0.01, 0.01]]);
        let result = render(&source, &base([1.0, 1.0, 1.0]), &settings());
        for channel in result.data {
            assert_close(channel, 1.0);
        }
    }

    #[test]
    fn output_stays_in_range_and_monotone_across_a_sweep() {
        let mut pixels = Vec::new();
        for i in -20..=40 {
            #[expect(
                clippy::cast_precision_loss,
                reason = "the loop range fits exactly in f32"
            )]
            let density = i as f32 * 0.1f32;
            pixels.push([10f32.powf(-density); 3]);
        }
        let source = LinearImage::new(
            u32::try_from(pixels.len()).unwrap(),
            1,
            pixels.iter().flatten().copied().collect(),
        )
        .expect("well-formed");
        let result = render(&source, &base([1.0, 1.0, 1.0]), &settings());

        let pixels_out = result.data.len() / 3;
        for i in 0..pixels_out {
            let [red, green, blue] = [
                result.data[i * 3],
                result.data[i * 3 + 1],
                result.data[i * 3 + 2],
            ];
            assert!((0.0..=1.0).contains(&red), "red out of range: {red}");
            assert!((0.0..=1.0).contains(&green), "green out of range: {green}");
            assert!((0.0..=1.0).contains(&blue), "blue out of range: {blue}");
            if i > 0 {
                assert!(
                    result.data[i * 3] >= result.data[(i - 1) * 3] - 1e-6,
                    "luma is not monotonic at {i}"
                );
            }
        }
    }

    #[test]
    fn conversion_is_deterministic() {
        let source = image(&[[0.7, 0.4, 0.2], [0.1, 0.9, 0.3]]);
        let base = base([0.8, 0.5, 0.3]);
        let first = render(&source, &base, &settings());
        let second = render(&source, &base, &settings());
        assert_eq!(first.data, second.data);
    }
}
