//! Slide (E-6) pipeline.
//!
//! A slide scan is already the positive: no inversion at all. `VueScan` and
//! `NegPy` both open it as captured, with optional normalization.
//!
//! `normalize == false` returns the image unchanged (the scan is the output).
//! `normalize == true` divides each channel by the base transmittance, with no
//! log and no inversion. On a positive this is a per-channel white balance of
//! the film's Dmin (the clear base), NOT exposure compensation, and it needs a
//! measured base. If `normalize` is true and `base` is `None`, the image is
//! returned unchanged (defensive; the driver refuses earlier with a clear
//! error).

use crate::image::LinearImage;
use crate::pipeline::film_base::FilmBase;

/// Renders a slide scan as the positive.
///
/// No inversion: the scan is the output, optionally normalized by the clear
/// film base.
#[must_use]
pub fn render(image: &LinearImage, base: Option<&FilmBase>, normalize: bool) -> LinearImage {
    if !normalize {
        return image.clone();
    }
    let Some(base) = base else {
        return image.clone();
    };
    image.map_rgb(|pixel| {
        [
            pixel[0] / base.transmittance[0],
            pixel[1] / base.transmittance[1],
            pixel[2] / base.transmittance[2],
        ]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(pixels: &[[f32; 3]]) -> LinearImage {
        let data = pixels.iter().flatten().copied().collect();
        let count = u32::try_from(pixels.len()).expect("test image is small");
        LinearImage::new(count, 1, data).expect("test image is well-formed")
    }

    fn base(transmittance: [f32; 3]) -> FilmBase {
        FilmBase { transmittance }
    }

    #[test]
    fn normalize_false_returns_input_unchanged() {
        let source = image(&[[0.1, 0.2, 0.3]]);
        let result = render(&source, None, false);
        assert_eq!(result.data, source.data);
    }

    #[test]
    fn normalize_true_takes_the_base_to_white() {
        let source = image(&[[0.5, 0.5, 0.5]]);
        let result = render(&source, Some(&base([0.5, 0.5, 0.5])), true);
        assert!((result.data[0] - 1.0).abs() < 1e-6);
        assert!((result.data[1] - 1.0).abs() < 1e-6);
        assert!((result.data[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_true_preserves_channel_ratios() {
        // Base takes 0.5 to 1.0, so 0.25 becomes 0.5 and ratios stay fixed.
        let source = image(&[[0.25, 0.5, 0.75]]);
        let result = render(&source, Some(&base([0.5, 0.5, 0.5])), true);
        assert!((result.data[0] - 0.5).abs() < 1e-6);
        assert!((result.data[1] - 1.0).abs() < 1e-6);
        assert!((result.data[2] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn normalize_true_with_no_base_returns_input_unchanged() {
        let source = image(&[[0.1, 0.2, 0.3]]);
        let result = render(&source, None, true);
        assert_eq!(result.data, source.data);
    }
}
