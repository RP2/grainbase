//! Image types shared across the conversion pipeline.

use std::fmt;

/// A linear-light RGB image in the camera's own colour space.
///
/// Samples are interleaved RGB `f32` values, `width * height * 3` of them.
/// Straight out of decoding they are nominally in `0.0..=1.0`; later pipeline
/// stages may push extremes outside that range.
///
/// Construct with [`LinearImage::new`], which enforces the sample-count
/// invariant. Do not build one field-by-field: a consumer that indexes by
/// pixel would run off the end of a short buffer.
#[derive(Debug, Clone)]
pub struct LinearImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Interleaved RGB samples, exactly `width * height * 3` of them.
    pub data: Vec<f32>,
}

/// A [`LinearImage`] could not be built because the sample count did not match
/// the stated dimensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampleCountMismatch {
    /// Declared width.
    pub width: u32,
    /// Declared height.
    pub height: u32,
    /// Samples actually supplied.
    pub samples: usize,
    /// Samples required by `width * height * 3`.
    pub expected: usize,
}

impl fmt::Display for SampleCountMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}x{} needs {} samples, got {}",
            self.width, self.height, self.expected, self.samples
        )
    }
}

impl std::error::Error for SampleCountMismatch {}

impl LinearImage {
    /// Builds an image from interleaved RGB samples.
    ///
    /// # Errors
    ///
    /// Returns [`SampleCountMismatch`] when `data` does not hold exactly
    /// `width * height * 3` samples.
    pub fn new(width: u32, height: u32, data: Vec<f32>) -> Result<Self, SampleCountMismatch> {
        let expected = width as usize * height as usize * 3;
        if data.len() != expected {
            return Err(SampleCountMismatch {
                width,
                height,
                samples: data.len(),
                expected,
            });
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_a_sample_count_that_does_not_match_the_dimensions() {
        assert!(LinearImage::new(2, 2, vec![0.0; 11]).is_err());
        assert!(LinearImage::new(2, 2, vec![0.0; 13]).is_err());
        assert!(LinearImage::new(2, 2, vec![0.0; 12]).is_ok());
    }

    #[test]
    fn zero_sized_image_is_accepted() {
        assert!(LinearImage::new(0, 0, Vec::new()).is_ok());
    }

    #[test]
    fn mismatch_reports_what_it_expected() {
        let err = LinearImage::new(2, 3, vec![0.0; 5]).unwrap_err();
        assert_eq!(err.expected, 18);
        assert_eq!(err.samples, 5);
    }
}
