//! Image types shared across the conversion pipeline.

use std::fmt;

/// A linear-light RGB image in the camera's own color space.
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

    /// Derives a new image by transforming each pixel.
    ///
    /// Dimensions and the sample-count invariant are preserved by construction,
    /// so deriving beats building a [`LinearImage`] field-by-field. Pixel order
    /// is row-major, left to right, top to bottom.
    #[must_use]
    pub fn map_rgb(&self, mut f: impl FnMut([f32; 3]) -> [f32; 3]) -> Self {
        let (pixels, _remainder) = self.data.as_chunks::<3>();
        let mut data = Vec::with_capacity(self.data.len());
        for pixel in pixels {
            data.extend_from_slice(&f(*pixel));
        }
        Self {
            width: self.width,
            height: self.height,
            data,
        }
    }

    /// Mirrors left to right.
    ///
    /// This is the correction for a scan made through the *reverse* face of the
    /// film: the copy is a mirror image of what the film actually shows.
    #[must_use]
    pub fn flipped_horizontal(&self) -> Self {
        if self.width == 0 || self.height == 0 {
            return self.clone();
        }
        let width = self.width as usize;
        let mut data = Vec::with_capacity(self.data.len());
        for row in self.data.as_chunks::<3>().0.chunks_exact(width) {
            for pixel in row.iter().rev() {
                data.extend_from_slice(pixel);
            }
        }
        Self {
            width: self.width,
            height: self.height,
            data,
        }
    }

    /// Mirrors top to bottom.
    #[must_use]
    pub fn flipped_vertical(&self) -> Self {
        if self.width == 0 || self.height == 0 {
            return self.clone();
        }
        let row_samples = self.width as usize * 3;
        let mut data = Vec::with_capacity(self.data.len());
        for row in self.data.chunks_exact(row_samples).rev() {
            data.extend_from_slice(row);
        }
        Self {
            width: self.width,
            height: self.height,
            data,
        }
    }

    /// Rotates the image a half turn.
    ///
    /// Equivalent to mirroring both axes; dimensions are unchanged.
    #[must_use]
    pub fn rotated_180(&self) -> Self {
        self.flipped_horizontal().flipped_vertical()
    }

    /// Reflects the image across its main diagonal, swapping width and height.
    ///
    /// Half of the EXIF orientation values are only expressible as a transpose
    /// combined with a flip.
    #[must_use]
    pub fn transposed(&self) -> Self {
        let width = self.width as usize;
        let height = self.height as usize;
        let (pixels, _remainder) = self.data.as_chunks::<3>();
        let mut data = Vec::with_capacity(self.data.len());
        for column in 0..width {
            for row in 0..height {
                data.extend_from_slice(&pixels[row * width + column]);
            }
        }
        Self {
            width: self.height,
            height: self.width,
            data,
        }
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

    /// A 3x2 image whose red channel is `1..=6`, for checking index arithmetic.
    fn numbered() -> LinearImage {
        let data = (1..=6u16)
            .flat_map(|value| [f32::from(value), 0.0, 0.0])
            .collect();
        LinearImage::new(3, 2, data).expect("well-formed")
    }

    fn reds(image: &LinearImage) -> Vec<f32> {
        image.data.as_chunks::<3>().0.iter().map(|p| p[0]).collect()
    }

    #[test]
    fn horizontal_flip_reverses_each_row_only() {
        // rows are [1 2 3] and [4 5 6]
        assert_eq!(
            reds(&numbered().flipped_horizontal()),
            [3.0, 2.0, 1.0, 6.0, 5.0, 4.0]
        );
    }

    #[test]
    fn vertical_flip_reverses_the_row_order_only() {
        assert_eq!(
            reds(&numbered().flipped_vertical()),
            [4.0, 5.0, 6.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn a_half_turn_reverses_everything() {
        assert_eq!(
            reds(&numbered().rotated_180()),
            [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        );
    }

    #[test]
    fn flipping_twice_is_the_identity() {
        let image = numbered();
        assert_eq!(
            reds(&image.flipped_horizontal().flipped_horizontal()),
            reds(&image)
        );
        assert_eq!(
            reds(&image.flipped_vertical().flipped_vertical()),
            reds(&image)
        );
        assert_eq!(reds(&image.rotated_180().rotated_180()), reds(&image));
    }

    #[test]
    fn flip_preserves_dimensions_and_the_invariant() {
        for flipped in [
            numbered().flipped_horizontal(),
            numbered().flipped_vertical(),
            numbered().rotated_180(),
        ] {
            assert_eq!((flipped.width, flipped.height), (3, 2));
            assert_eq!(flipped.data.len(), 18);
        }
    }

    #[test]
    fn transpose_swaps_the_axes() {
        // 1 2 3        1 4
        // 4 5 6   ->   2 5
        //              3 6
        let transposed = numbered().transposed();
        assert_eq!((transposed.width, transposed.height), (2, 3));
        assert_eq!(reds(&transposed), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transposing_twice_is_the_identity() {
        let image = numbered();
        let twice = image.transposed().transposed();
        assert_eq!((twice.width, twice.height), (3, 2));
        assert_eq!(reds(&twice), reds(&image));
    }

    #[test]
    fn a_zero_width_image_can_be_flipped() {
        // chunks_exact(0) panics, so the degenerate case needs an explicit guard.
        let empty = LinearImage::new(0, 4, Vec::new()).expect("well-formed");
        assert_eq!(empty.flipped_horizontal().width, 0);
        assert_eq!(empty.flipped_vertical().height, 4);
    }
}
