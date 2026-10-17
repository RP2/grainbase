//! `parity_report`: measured per-channel difference between two same-sized
//! display-encoded sRGB images.
//!
//! The first cut of the parity harness (U8): ours versus a reference render
//! of the same frame. Both files are read as encoded sRGB PNG values, so the
//! comparison is pixel-to-pixel with no transfer-function guessing. The
//! reference must be aligned to ours first (a resize, which biases large
//! gradients); alignment is the caller's job, and the tool refuses to
//! compare different sizes.
//!
//! ```text
//! cargo run --example parity_report -- [--rotate180] ours.png reference.png
//! ```

fn main() -> Result<(), String> {
    let mut positional = Vec::new();
    let mut rotate = false;
    for argument in std::env::args().skip(1) {
        match argument.as_str() {
            "--rotate180" => rotate = true,
            _ => positional.push(argument),
        }
    }
    let [ours_path, reference_path] = positional.as_slice() else {
        return Err("usage: parity_report [--rotate180] ours.png reference.png".into());
    };

    let mut ours = decode_png(std::path::Path::new(ours_path))?;
    if rotate {
        ours.0 = ours.0.rotated_180();
    }
    let reference = decode_png(std::path::Path::new(reference_path))?;

    if ours.0.width != reference.0.width || ours.0.height != reference.0.height {
        return Err(format!(
            "size mismatch: ours {}x{}, reference {}x{} — align first (magick -resize)",
            ours.0.width, ours.0.height, reference.0.width, reference.0.height
        ));
    }

    let name = |channel: usize| ["red", "green", "blue"][channel];
    for channel in 0..3 {
        let ours_stats = statistics(&ours.1, channel);
        let reference_stats = statistics(&reference.1, channel);
        println!(
            "{:5} channel  ours mean {:.4} med {:.4}  ref mean {:.4} med {:.4}  mean-diff {:+.4}",
            name(channel),
            ours_stats.mean,
            ours_stats.median,
            reference_stats.mean,
            reference_stats.median,
            ours_stats.mean - reference_stats.mean
        );
    }

    let color_diff = color_metric(&ours.1, &reference.1);
    println!(
        "color distance (delta-E76, first cut): mean {:.2}  p50 {:.2}  p95 {:.2}  p99 {:.2}",
        color_diff.mean, color_diff.median, color_diff.p95, color_diff.p99
    );
    println!("ratio check: ours-vs-reference structured relation reported per channel above");
    Ok(())
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

struct Stats {
    mean: f32,
    median: f32,
}

fn statistics(display_sets: &[[f32; 3]], channel: usize) -> Stats {
    #[expect(
        clippy::cast_precision_loss,
        reason = "mean of up to 40 million samples needs no f32 mantissa exactness"
    )]
    let scale = |count: usize| count as f32;
    let mut values: Vec<f32> = display_sets.iter().map(|p| p[channel]).collect();
    let mean = values.iter().sum::<f32>() / scale(values.len());
    values.sort_by(f32::total_cmp);
    Stats {
        mean,
        median: values[values.len() / 2],
    }
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

/// Stored (display) sRGB to CIELAB, D65, first cut: dekode to linear, sRGB
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
