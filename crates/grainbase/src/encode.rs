//! Writing a finished positive to disk.
//!
//! Format coverage is deliberately narrow for now: 16-bit PNG, tagged sRGB. PNG
//! is the only candidate whose writer can tag color honestly — the `tiff` crate
//! has no ICC or profile support at all, so a TIFF would have to ship untagged or
//! lie. More formats belong here later; they must all be able to carry a color
//! tag, because an untagged file is not color-managed, it is a guess.

use std::fmt;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use crate::image::LinearImage;

/// Why a positive could not be written.
#[derive(Debug)]
pub enum Error {
    /// The output file could not be created or written.
    Io(std::io::Error),
    /// The PNG encoder rejected the image.
    Png(png::EncodingError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "could not write output: {err}"),
            Self::Png(err) => write!(f, "PNG encoding failed: {err}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Png(err) => Some(err),
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<png::EncodingError> for Error {
    fn from(err: png::EncodingError) -> Self {
        Self::Png(err)
    }
}

/// Writes a positive as 16-bit sRGB PNG.
///
/// Input is linear light in sRGB primaries, which is what the pipeline produces.
/// The sRGB transfer function is applied here, at the boundary, so the file is
/// display-encoded as its tag claims. Nothing else is applied: grading and look
/// belong to a later stage, and blending them in here would make it impossible to
/// tell which part of the output is the negative and which is taste.
///
/// # Errors
///
/// Returns [`Error::Io`] if the file cannot be written, or [`Error::Png`] if the
/// encoder rejects the image.
pub fn encode(image: &LinearImage, path: impl AsRef<Path>) -> Result<(), Error> {
    let file = File::create(path)?;
    let mut encoder = png::Encoder::new(BufWriter::new(file), image.width, image.height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Sixteen);
    // The sRGB chunk, so a reader does not have to assume.
    encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);

    let mut writer = encoder.write_header()?;

    // PNG stores 16-bit samples big-endian.
    let mut samples = Vec::with_capacity(image.data.len() * 2);
    for sample in &image.data {
        samples.extend_from_slice(&to_srgb16(*sample).to_be_bytes());
    }

    writer.write_image_data(&samples)?;
    Ok(())
}

/// Encodes one linear sample as a 16-bit sRGB value.
///
/// The transfer function from IEC 61966-2-1. Linear segment near black, then the
/// power segment.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the encoded value is clamped to 0.0..=1.0 before scaling, so it is \
              finite, non-negative and below 65536"
)]
#[must_use]
pub fn to_srgb16(linear: f32) -> u16 {
    let clamped = linear.clamp(0.0, 1.0);
    let encoded = if clamped <= 0.003_130_8 {
        12.92 * clamped
    } else {
        1.055 * clamped.powf(1.0 / 2.4) - 0.055
    };
    (encoded * f32::from(u16::MAX)).round() as u16
}

/// Decodes one encoded sRGB sample to linear light.
///
/// The inverse of [`to_srgb16`], from the same IEC 61966-2-1 transfer
/// function: power segment above the linear knee.
#[must_use]
pub fn from_srgb16(encoded: f32) -> f32 {
    let encoded = encoded.clamp(0.0, 1.0);
    if encoded <= 12.92 * 0.003_130_8 {
        encoded / 12.92
    } else {
        ((encoded + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::BufReader;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("grainbase-encode-{}-{name}", std::process::id()));
        path
    }

    fn sample_image() -> LinearImage {
        // Black, mid, white; and the reverse, so a row-flip would show up.
        LinearImage::new(2, 1, vec![0.0, 0.5, 1.0, 1.0, 0.5, 0.0]).expect("well-formed")
    }

    #[test]
    fn the_transfer_function_hits_its_endpoints() {
        assert_eq!(to_srgb16(0.0), 0);
        assert_eq!(to_srgb16(1.0), u16::MAX);
        assert_eq!(to_srgb16(-1.0), 0, "below range clamps to black");
        assert_eq!(to_srgb16(2.0), u16::MAX, "above range clamps to white");
    }

    #[test]
    fn the_transfer_function_is_monotonic() {
        let mut previous = 0;
        for step in 0..=1000u16 {
            let linear = f32::from(step) / 1000.0;
            let encoded = to_srgb16(linear);
            assert!(
                encoded >= previous,
                "not monotonic at {linear}: {encoded} < {previous}"
            );
            previous = encoded;
        }
    }

    #[test]
    fn write_outputs_a_sixteen_bit_srgb_png() {
        let path = temp_path("basic.png");
        encode(&sample_image(), &path).expect("encode should succeed");

        let decoder = png::Decoder::new(BufReader::new(File::open(&path).expect("open output")));
        let reader = decoder.read_info().expect("read header");
        let info = reader.info();

        assert_eq!(info.width, 2);
        assert_eq!(info.height, 1);
        assert_eq!(info.bit_depth, png::BitDepth::Sixteen);
        assert_eq!(info.color_type, png::ColorType::Rgb);
        assert!(
            info.srgb.is_some(),
            "the file should carry an sRGB rendering intent, not leave it to be assumed"
        );

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn the_pixels_survive_the_round_trip_in_order() {
        let path = temp_path("roundtrip.png");
        encode(&sample_image(), &path).expect("encode should succeed");

        let mut decoder =
            png::Decoder::new(BufReader::new(File::open(&path).expect("open output")));
        decoder.set_transformations(png::Transformations::normalize_to_color8());
        let mut reader = decoder.read_info().expect("read header");
        let mut buffer = vec![0; reader.output_buffer_size().expect("known size")];
        let frame = reader.next_frame(&mut buffer).expect("read pixels");

        // normalize_to_color8 scales to 8 bits for comparison. The mid sample is
        // 188, not 128: the sRGB transfer of linear 0.5 is 0.735, so encoding is
        // visibly brighter than the linear value. That is the point of the test.
        assert_eq!(
            &buffer[..frame.buffer_size()],
            &[0, 188, 255, 255, 188, 0],
            "pixel order or transfer is wrong"
        );

        let _ = fs::remove_file(&path);
    }
}
