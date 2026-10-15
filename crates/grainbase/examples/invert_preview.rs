//! Writes a viewable preview of a converted scan, for judging quality by eye.
//!
//! ```text
//! cargo run --example invert_preview -- <scan.raw> <out.png> [--flip-h] [--flip-v]
//! ```
//!
//! This is an inspection aid, not the engine's encoder. It applies display gamma
//! and 8-bit quantisation, neither of which belongs to the engine: the real
//! `encode` unit writes 16-bit color-managed output, and the `look` unit owns
//! grading. Keep it around for eyeballing conversion changes before those land.

use grainbase::decode;
use grainbase::pipeline::{self, FilmBase, Settings, film_base, percentile};
use grainbase::{LinearImage, encode};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut positional = Vec::new();
    let mut flip_h = false;
    let mut flip_v = false;

    for argument in std::env::args().skip(1) {
        match argument.as_str() {
            "--flip-h" => flip_h = true,
            "--flip-v" => flip_v = true,
            _ => positional.push(argument),
        }
    }

    let [input, output] = positional.as_slice() else {
        return Err("usage: invert_preview <scan.raw> <out.png> [--flip-h] [--flip-v]".into());
    };

    let scan = decode::decode(input)?;
    println!(
        "decoded {}x{} from {} {} (EXIF orientation {:?})",
        scan.linear.width,
        scan.linear.height,
        scan.meta.make,
        scan.meta.model,
        scan.meta.orientation
    );
    let mut negative = scan.linear;

    if flip_h {
        negative = negative.flipped_horizontal();
        println!("mirrored left to right");
    }
    if flip_v {
        negative = negative.flipped_vertical();
        println!("mirrored top to bottom");
    }

    let base = FilmBase::detect(&negative, film_base::DEFAULT_PERCENTILE);
    println!(
        "film base   R {:.4}  G {:.4}  B {:.4}",
        base.transmittance[0], base.transmittance[1], base.transmittance[2]
    );

    let positive = pipeline::invert(&negative, &base, &Settings::default());

    report("negative", &negative, &base);
    report("positive", &positive, &base);

    diagnose_base(&negative);

    trace_tail(&negative);

    encode::encode(&positive, output)?;
    println!("wrote {output}");
    Ok(())
}

/// Traces the upper tail of the green channel, where the film/non-film
/// transition lives.
///
/// The shape matters: a *gap* between two populations needs one fix, a gradual
/// *density* change needs another.
fn trace_tail(negative: &LinearImage) {
    let (pixels, _remainder) = negative.data.as_chunks::<3>();
    let step = (pixels.len() / (1 << 18)).max(1);
    let mut samples: Vec<f32> = pixels.iter().step_by(step).map(|p| p[1]).collect();
    samples.sort_unstable_by(f32::total_cmp);

    println!("--- green channel upper tail ---");
    let mut percentile = 99.0f32;
    while percentile <= 100.0 {
        let value = percentile::value_at_sorted(&samples, percentile).unwrap_or(f32::NAN);
        println!("  p{percentile:<6.2} {value:.4}");
        percentile += 0.05;
    }
}

/// Tests whether the film base estimate is contaminated by bright areas that
/// are not film: sprocket holes, the light table, or the holder.
///
/// Those are brighter than the orange-masked base, so if they are in frame the
/// percentile lands on them and the "base" becomes the light source's color.
/// Measuring the center only should land on film.
fn diagnose_base(negative: &LinearImage) {
    let full = FilmBase::detect(negative, film_base::DEFAULT_PERCENTILE);
    let center = FilmBase::detect(
        &center_fraction(negative, 0.5),
        film_base::DEFAULT_PERCENTILE,
    );

    println!("--- base diagnosis ---");
    for percentile in [90.0, 99.0, 99.5, 99.9, 99.99] {
        let base = FilmBase::detect(negative, percentile);
        println!(
            "  p{percentile:<6} R {:.4}  G {:.4}  B {:.4}",
            base.transmittance[0], base.transmittance[1], base.transmittance[2]
        );
    }
    println!(
        "  full frame base  R {:.4}  G {:.4}  B {:.4}",
        full.transmittance[0], full.transmittance[1], full.transmittance[2]
    );
    println!(
        "  center 50% base  R {:.4}  G {:.4}  B {:.4}",
        center.transmittance[0], center.transmittance[1], center.transmittance[2]
    );

    // If the two disagree in *ratio*, the full-frame estimate is not looking at
    // film, and the resulting cast is an artefact rather than a scene property.
    let ratio = |base: &FilmBase| {
        [
            base.transmittance[0] / base.transmittance[1].max(f32::MIN_POSITIVE),
            base.transmittance[2] / base.transmittance[1].max(f32::MIN_POSITIVE),
        ]
    };
    let (full_ratio, center_ratio) = (ratio(&full), ratio(&center));
    println!(
        "  R/G B/G  full {:.3} {:.3}   center {:.3} {:.3}",
        full_ratio[0], full_ratio[1], center_ratio[0], center_ratio[1]
    );
}

/// The central `fraction` of the image, by both axes.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a diagnostic crop: the axis lengths of a real scan are far below \
              f32's exact-integer range, and fraction is a small positive"
)]
fn center_fraction(image: &LinearImage, fraction: f32) -> LinearImage {
    let width = image.width as usize;
    let height = image.height as usize;
    let crop_width = ((width as f32 * fraction) as usize).min(width);
    let crop_height = ((height as f32 * fraction) as usize).min(height);
    let left = (width - crop_width) / 2;
    let top = (height - crop_height) / 2;

    let (pixels, _remainder) = image.data.as_chunks::<3>();
    let mut data = Vec::with_capacity(crop_width * crop_height * 3);
    for row in top..top + crop_height {
        for column in left..left + crop_width {
            data.extend_from_slice(&pixels[row * width + column]);
        }
    }

    LinearImage::new(
        u32::try_from(crop_width).unwrap_or(image.width),
        u32::try_from(crop_height).unwrap_or(image.height),
        data,
    )
    .unwrap_or_else(|_| image.clone())
}

/// Prints the numbers that say whether the conversion did anything sensible.
///
/// The per-channel spread is the one that matters for color: the film base is
/// divided out per channel, so a *neutral* frame should come back with three
/// channel means that agree. A persistent spread is a cast, and this is how much.
fn report(label: &str, image: &LinearImage, base: &FilmBase) {
    let (min, median, max) = stats(&image.data);
    println!("{label:<10} min {min:.4}  median {median:.4}  max {max:.4}");

    let means = channel_means(image);
    println!(
        "{:<10} mean R {:.4}  G {:.4}  B {:.4}   (B/R {:.3})",
        format!("{label} ch"),
        means[0],
        means[1],
        means[2],
        means[2] / means[0].max(f64::MIN_POSITIVE)
    );

    let brightest_base = base.transmittance.iter().copied().fold(f32::MIN, f32::max);
    if label == "negative" {
        println!(
            "           base is {:.3}x the negative's brightest sample",
            brightest_base / max.max(f32::MIN_POSITIVE)
        );
    }
}

/// Per-channel means. Equal values in the positive mean a neutral conversion.
#[expect(
    clippy::cast_precision_loss,
    reason = "a diagnostic mean: the pixel count of a real scan is far below the \
              integer range f64 represents exactly"
)]
fn channel_means(image: &LinearImage) -> [f64; 3] {
    let mut sums = [0.0f64; 3];
    let mut count = 0u64;
    for pixel in image.data.as_chunks::<3>().0 {
        for (sum, sample) in sums.iter_mut().zip(pixel) {
            *sum += f64::from(*sample);
        }
        count += 1;
    }
    let count = count.max(1) as f64;
    [sums[0] / count, sums[1] / count, sums[2] / count]
}

/// Minimum, median, and maximum of a set of samples.
fn stats(samples: &[f32]) -> (f32, f32, f32) {
    let min = samples.iter().copied().fold(f32::INFINITY, f32::min);
    let max = samples.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // Selection needs a mutable slice; a copy is the price of not disturbing
    // the caller's data.
    let mut values = samples.to_vec();
    let median = percentile::value_at(&mut values, 50.0).unwrap_or(f32::NAN);
    (min, median, max)
}
