//! `parity_report`: measured per-channel difference between two same-sized
//! display-encoded sRGB images.
//!
//! The parity harness (U8): ours versus a reference render of the same frame.
//! Both files are read as encoded sRGB PNG values, so the comparison is
//! pixel-to-pixel with no transfer-function guessing. The reference must be
//! aligned to ours first (a resize, which biases large gradients); alignment is
//! the caller's job, and the tool refuses to compare different sizes.
//!
//! ## What it scores, and why not means
//!
//! A frame mean collapses three channels and the scene's own colours into one
//! number. It cannot see a per-channel error, and it reports a scene change as a
//! pipeline change — which is how four bad changes once looked like progress.
//! This tool therefore reports:
//!
//! - **each channel's distribution**, p5 / p50 / p95, ours against the reference;
//! - **a neutral reference, per channel**, as G/R and B/R against 1.0, when
//!   `--neutral` names a patch known to be neutral in the scene.
//!
//! The neutral readout is the one that matters: it is the only place the tool can
//! tell a film or scanning cast from a scene that is genuinely one colour.
//!
//! ```text
//! cargo run --example parity_report -- [--rotate180] \
//!   [--neutral x0,y0,x1,y1] ours.png reference.png
//! ```

fn main() -> Result<(), String> {
    let mut positional = Vec::new();
    let mut rotate = false;
    let mut neutral: Option<[f32; 4]> = None;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--rotate180" => rotate = true,
            "--neutral" => {
                let text = arguments
                    .next()
                    .ok_or("--neutral requires x0,y0,x1,y1 as fractions")?;
                neutral = Some(parse_rect(&text)?);
            }
            _ => positional.push(argument),
        }
    }
    let [ours_path, reference_path] = positional.as_slice() else {
        return Err(
            "usage: parity_report [--rotate180] [--neutral x0,y0,x1,y1] ours.png reference.png"
                .into(),
        );
    };

    let mut ours = decode_png(std::path::Path::new(ours_path))?;
    if rotate {
        ours.0 = ours.0.rotated_180();
        ours.1 = rotate_display(&ours.1, ours.0.width, ours.0.height);
    }
    let reference = decode_png(std::path::Path::new(reference_path))?;

    if ours.0.width != reference.0.width || ours.0.height != reference.0.height {
        return Err(format!(
            "size mismatch: ours {}x{}, reference {}x{} — align first (magick -resize)",
            ours.0.width, ours.0.height, reference.0.width, reference.0.height
        ));
    }

    let name = |channel: usize| ["red", "green", "blue"][channel];
    println!("per-channel distribution (display values, 0..1):");
    for channel in 0..3 {
        let ours_stats = distribution(&ours.1, channel);
        let reference_stats = distribution(&reference.1, channel);
        println!(
            "  {:5}  ours p5 {:.3} p50 {:.3} p95 {:.3}   ref p5 {:.3} p50 {:.3} p95 {:.3}   delta p5 {:+.3} p50 {:+.3} p95 {:+.3}",
            name(channel),
            ours_stats.p5,
            ours_stats.p50,
            ours_stats.p95,
            reference_stats.p5,
            reference_stats.p50,
            reference_stats.p95,
            ours_stats.p5 - reference_stats.p5,
            ours_stats.p50 - reference_stats.p50,
            ours_stats.p95 - reference_stats.p95,
        );
    }

    if let Some(rect) = neutral {
        let here = region_mean(&ours.0, &ours.1, rect);
        let there = region_mean(&reference.0, &reference.1, rect);
        match (here, there) {
            (Some(here), Some(there)) => {
                println!("neutral patch (fractions {rect:?}) — G/R and B/R against 1.0:");
                print_neutral("ours", &here);
                print_neutral("ref ", &there);
                println!(
                    "  delta     G/R {:+.3}  B/R {:+.3}",
                    ratio(here[1], here[0]) - ratio(there[1], there[0]),
                    ratio(here[2], here[0]) - ratio(there[2], there[0]),
                );
            }
            _ => println!("neutral patch: the rectangle covers no pixels; check the fractions"),
        }
    } else {
        println!(
            "neutral patch: not given. Pass --neutral x0,y0,x1,y1 over a patch known to be \
             neutral in the scene; without it this report cannot separate a cast from a scene \
             that is genuinely one colour."
        );
    }

    let color_diff = color_metric(&ours.1, &reference.1);
    println!(
        "color distance (delta-E76, first cut): mean {:.2}  p50 {:.2}  p95 {:.2}  p99 {:.2}",
        color_diff.mean, color_diff.median, color_diff.p95, color_diff.p99
    );
    Ok(())
}

/// A display value's G/R or B/R, guarding a zero red channel.
fn ratio(channel: f32, red: f32) -> f32 {
    channel / red.max(f32::MIN_POSITIVE)
}

/// Prints one image's neutral-patch means and their channel ratios.
fn print_neutral(label: &str, means: &[f32; 3]) {
    println!(
        "  {label}  R {:.3} G {:.3} B {:.3}   G/R {:.3}  B/R {:.3}",
        means[0],
        means[1],
        means[2],
        ratio(means[1], means[0]),
        ratio(means[2], means[0]),
    );
}

/// Parses `--neutral x0,y0,x1,y1` as fractions of the image.
fn parse_rect(text: &str) -> Result<[f32; 4], String> {
    let parts: Vec<&str> = text.split(',').collect();
    if parts.len() != 4 {
        return Err("--neutral expects four comma-separated fractions".to_owned());
    }
    let mut out = [0.0f32; 4];
    for (index, part) in parts.iter().enumerate() {
        out[index] = part
            .trim()
            .parse()
            .map_err(|_| format!("--neutral value `{part}` is not a number"))?;
    }
    Ok(out)
}

/// Reverses a display buffer in the same way [`LinearImage::rotated_180`] does.
///
/// The two buffers are decoded separately, so rotating one has to rotate the
/// other; otherwise the pixels compared would be ours-rotated against
/// reference-unrotated.
fn rotate_display(display: &[[f32; 3]], width: u32, height: u32) -> Vec<[f32; 3]> {
    let mut out = Vec::with_capacity(display.len());
    for row in (0..height as usize).rev() {
        for column in (0..width as usize).rev() {
            out.push(display[row * width as usize + column]);
        }
    }
    out
}

/// The mean display value per channel inside a fractional rectangle.
fn region_mean(
    image: &grainbase::LinearImage,
    display: &[[f32; 3]],
    rect: [f32; 4],
) -> Option<[f32; 3]> {
    let width = image.width as usize;
    let height = image.height as usize;
    if width == 0 || height == 0 || display.len() != width * height {
        return None;
    }

    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "rectangle bounds are clamped to the image and truncated to pixel indices"
    )]
    let span = |fraction: f32, extent: usize| -> usize {
        (fraction.clamp(0.0, 1.0) * extent as f32).floor() as usize
    };

    let x0 = span(rect[0], width);
    let y0 = span(rect[1], height);
    let x1 = span(rect[2], width).min(width);
    let y1 = span(rect[3], height).min(height);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }

    let mut sums = [0.0f64; 3];
    let mut count = 0usize;
    for row in y0..y1 {
        for column in x0..x1 {
            let pixel = display[row * width + column];
            for (sum, value) in sums.iter_mut().zip(pixel) {
                *sum += f64::from(value);
            }
            count += 1;
        }
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "the pixel count of a patch is far below f64's exact-integer range"
    )]
    let count = count as f64;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a mean of display values in 0..=1 narrows to f32 with room to spare"
    )]
    let means = sums.map(|sum| (sum / count) as f32);
    Some(means)
}

/// A channel's p5 / p50 / p95 display value.
struct Distribution {
    p5: f32,
    p50: f32,
    p95: f32,
}

fn distribution(display_sets: &[[f32; 3]], channel: usize) -> Distribution {
    let mut values: Vec<f32> = display_sets.iter().map(|pixel| pixel[channel]).collect();
    values.sort_by(f32::total_cmp);
    let at = |percentile: f32| {
        grainbase::pipeline::percentile::value_at_sorted(&values, percentile).unwrap_or(f32::NAN)
    };
    Distribution {
        p5: at(5.0),
        p50: at(50.0),
        p95: at(95.0),
    }
}

fn decode_png(path: &std::path::Path) -> Result<(grainbase::LinearImage, Vec<[f32; 3]>), String> {
    use png::{BitDepth, ColorType};
    let shown = path.display();
    let file = std::fs::File::open(path).map_err(|err| format!("{shown}: {err}"))?;
    let mut reader = png::Decoder::new(std::io::BufReader::new(file))
        .read_info()
        .map_err(|err| format!("{shown}: {err}"))?;
    let (color, depth) = reader.output_color_type();
    if color != ColorType::Rgb {
        return Err(format!(
            "{shown}: expected RGB (got {color:?}, depth {depth:?}); re-encode with 'magick -strip' if it carries alpha"
        ));
    }
    let width = reader.info().width;
    let height = reader.info().height;
    let count = (width as usize) * (height as usize);
    let buffer_size = reader
        .output_buffer_size()
        .ok_or_else(|| format!("{shown}: frame too large to decode"))?;
    let mut buffer = vec![0u8; buffer_size];
    reader
        .next_frame(&mut buffer)
        .map_err(|err| format!("{shown}: {err}"))?;

    // Display values normalized to 0..1 regardless of stored depth; an 8-bit
    // reference carries its producer's precision ceiling with it.
    let mut linear = Vec::with_capacity(count * 3);
    let mut display = vec![[0.0f32; 3]; count];
    match depth {
        BitDepth::Sixteen => {
            for sample in buffer.as_chunks::<2>().0 {
                let encoded = u16::from_be_bytes(*sample);
                let encoded_n = f32::from(encoded) / f32::from(u16::MAX);
                let i = linear.len();
                linear.push(grainbase::encode::from_srgb16(encoded_n));
                display[i / 3][i % 3] = encoded_n;
            }
        }
        BitDepth::Eight => {
            for encoded in &buffer {
                let encoded_n = f32::from(*encoded) / f32::from(u8::MAX);
                let i = linear.len();
                linear.push(grainbase::encode::from_srgb16(encoded_n));
                display[i / 3][i % 3] = encoded_n;
            }
        }
        other => return Err(format!("{shown}: unsupported depth {other:?}")),
    }
    let image = grainbase::LinearImage::new(width, height, linear)
        .map_err(|err| format!("{shown}: {err}"))?;
    Ok((image, display))
}

struct ColorDiff {
    mean: f32,
    median: f32,
    p95: f32,
    p99: f32,
}

/// delta-Lab distance per pixel (first cut, delta-E76), sampled striding.
fn color_metric(ours: &[[f32; 3]], reference: &[[f32; 3]]) -> ColorDiff {
    let step = (ours.len() / (1 << 17)).max(1);
    let mut diffs = Vec::new();
    for (ours_pixel, reference_pixel) in ours.iter().zip(reference).step_by(step) {
        let ours_lab = lab(ours_pixel);
        let reference_lab = lab(reference_pixel);
        let distance = (0..3)
            .map(|c| ours_lab[c] - reference_lab[c])
            .map(|difference| difference * difference)
            .sum::<f32>()
            .sqrt();
        diffs.push(distance);
    }
    diffs.sort_by(f32::total_cmp);
    #[expect(
        clippy::cast_precision_loss,
        reason = "mean of up to 40 million sampled distances needs no f32 \
                  mantissa exactness"
    )]
    let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
    // Integer rank arithmetic on the sorted slice; no float rounding involved.
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the ranks are non-negative and well inside usize for any \
                  sampled set a photograph produces"
    )]
    let rank_at = |share: f32| (diffs.len() as f32 * share) as usize;
    ColorDiff {
        mean,
        median: diffs[diffs.len() / 2],
        p95: diffs[rank_at(0.95)],
        p99: diffs[rank_at(0.99)],
    }
}

/// Stored (display) sRGB to CIELAB, D65, first cut: decode to linear, sRGB
/// matrix to XYZ, then the standard Lab transform.
fn lab(display: &[f32; 3]) -> [f32; 3] {
    let linear: [f32; 3] = display.map(grainbase::encode::from_srgb16);
    let xyz = xyz_from_linear_srgb(linear);
    lab_from_xyz(xyz)
}

/// The published sRGB (D65) matrix, as commonly recited in PSD/CSS primaries.
#[must_use]
fn xyz_from_linear_srgb(linear: [f32; 3]) -> [f32; 3] {
    let m = [
        [0.412_456_f32, 0.357_576, 0.180_437],
        [0.212_672, 0.715_152, 0.072_175],
        [0.019_333, 0.119_192, 0.950_304],
    ];
    let mut out = [0.0f32; 3];
    for (row, out_row) in m.iter().zip(&mut out) {
        *out_row = row[0] * linear[0] + row[1] * linear[1] + row[2] * linear[2];
    }
    out
}

/// CIE Lab, D65 white point, from XYZ in [0..1] normalized.
#[must_use]
fn lab_from_xyz(xyz: [f32; 3]) -> [f32; 3] {
    fn f(t: f32) -> f32 {
        const DELTA: f32 = 6.0 / 29.0;
        if t > DELTA * DELTA * DELTA {
            t.cbrt()
        } else {
            t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
        }
    }
    let white = [0.950_47, 1.0, 1.088_83];
    let scaled: [f32; 3] = std::array::from_fn(|c| xyz[c] / white[c]);
    let fx = f(scaled[0]);
    let fy = f(scaled[1]);
    let fz = f(scaled[2]);
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}
