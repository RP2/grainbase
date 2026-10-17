//! Writes a viewable preview of a converted scan, for judging quality by eye.
//!
//! ```text
//! cargo run --example invert_preview -- [--profile <path>] [--base-sample x0,y0,x1,y1] <scan.raw> <out.png> [--flip-h] [--flip-v] [--no-crop]
//! ```
//!
//! This is an inspection aid, not the engine's encoder. It applies display gamma
//! and 8-bit quantisation, neither of which belongs to the engine: the real
//! `encode` unit writes 16-bit color-managed output, and the `look` unit owns
//! grading. Keep it around for eyeballing conversion changes before those land.
//!
//! The frame is found first, then the film base is measured from the frame's
//! rebate regions when geometry could locate them. A manual `--base-sample`
//! rectangle overrides that: it is the operator's statement of where the rebate
//! sits, and beats every heuristic on glare-heavy rigs. The crop still happens
//! after the base measurement, because cropping discards the rebate and the
//! rebate is the clear film the base is measured from.

use grainbase::decode;
use grainbase::pipeline::{self, FilmBase, Settings, film_base, meter, percentile};
use grainbase::profile::Profile;
use grainbase::{LinearImage, encode, geometry, overlay};

/// Sample points drawn in the overlay.
///
/// A drawing of every point the base sample reads is a filled rectangle; this is
/// enough marks to show the shape of the sample and which side of the strip it
/// came from.
const OVERLAY_SAMPLE_MARKS: usize = 400;

#[expect(
    clippy::too_many_lines,
    reason = "a diagnostic tool's main is a readout of every stage in order"
)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = parse_options()?;

    let scan = decode::decode(&options.input)?;
    println!(
        "decoded {}x{} from {} {} (EXIF orientation {:?})",
        scan.linear.width,
        scan.linear.height,
        scan.meta.make,
        scan.meta.model,
        scan.meta.orientation
    );

    // Orient first, then find: the frame is a property of the film, and the
    // flips are the operator's orientation fix, not part of the scan.
    let mut negative = scan.linear;
    if options.flip_h {
        negative = negative.flipped_horizontal();
        println!("mirrored left to right");
    }
    if options.flip_v {
        negative = negative.flipped_vertical();
        println!("mirrored top to bottom");
    }

    let frame = geometry::find(&negative).unwrap_or_else(|| geometry::Frame::whole(&negative));
    println!(
        "frame: centre {:.0},{:.0}  size {:.0}x{:.0}  skew {:.4} rad ({:.2} deg)",
        frame.center[0],
        frame.center[1],
        frame.size[0],
        frame.size[1],
        frame.skew,
        frame.skew.to_degrees()
    );

    // Base on the oriented scan, still uncropped, by the documented source
    // precedence (spec step 3): hand-supplied values, hand-placed rectangle,
    // the clear film just outside the frame, whole-frame statistics. The
    // whole-frame tier is VueScan's documented default estimate; its known
    // limit is light-table glare on copy-stand scans, for which the manual
    // inputs are the answer.
    if std::env::var("GRAINBASE_TRACE").is_ok() {
        diagnose_base(&negative);
        trace_tail(&negative);
    }

    // The base is the unexposed film, and the punch rows between the
    // perforations are the one place it is guaranteed to be: outside the
    // exposure, and clear of the light table.
    let bands: Vec<film_base::FilmRegion> = frame.holes.iter().flatten().copied().collect();

    let base = if let Some(values) = options.film_base {
        println!("film base (manual values)");
        FilmBase {
            transmittance: values.map(|value| value.max(f32::MIN_POSITIVE)),
        }
    } else if let Some(rect) = options.base_sample {
        println!("film base (manual sample rectangle)");
        FilmBase::detect_in_rect(&negative, rect, film_base::DEFAULT_PERCENTILE)
    } else if let Some(base) = measured_between_sprockets(&negative, &bands) {
        println!("film base (the film between the sprocket holes)");
        for band in &bands {
            println!(
                "  sprocket band: center_y {:.1} ±{:.1} slope {:.5}",
                band.center_y, band.half_height, band.slope
            );
        }
        base
    } else {
        eprintln!(
            "no film base could be measured: the scan shows no sprocket bands to measure \
             between. Pass --film-base r,g,b or --base-sample x0,y0,x1,y1."
        );
        std::process::exit(1);
    };

    // Metering is per frame and reads the picture: the frame's integral density is
    // pulled back toward its own neutral, weighted down when the frame is a scene
    // that is genuinely one color rather than a film or scanning cast. It runs on
    // the uncropped scan, before the crop, like every other measurement.
    let mut settings = options.settings;
    if options.meter {
        let region = frame.sample_region().exposure_region(&negative, &base);
        match meter::meter(&negative, &base, region.rect_fractions(&negative)) {
            Some(metered) => {
                println!(
                    "metering: integral R {:.3} G {:.3} B {:.3}  confidence {:.2}  \
                     correction R {:+.3} G {:+.3} B {:+.3}",
                    metered.integral[0],
                    metered.integral[1],
                    metered.integral[2],
                    metered.confidence,
                    metered.correction[0],
                    metered.correction[1],
                    metered.correction[2],
                );
                let floor = settings.density_floor.unwrap_or([0.0; 3]);
                settings.density_floor =
                    Some([0, 1, 2].map(|channel| floor[channel] - metered.correction[channel]));
            }
            None => println!("metering: the analysis region holds no blocks, not metered"),
        }
    }

    // The overlay is drawn in the uncropped scan's coordinates, so it has to
    // keep a copy when the crop moves the picture underneath it.
    let overlay_scan = if options.overlay && !options.no_crop {
        Some(negative.clone())
    } else {
        None
    };

    // Orientation first, then the frame: the frame is a property of the film, and
    // rotating the scan does not move it relative to the picture.
    if !options.no_crop {
        negative = geometry::apply(&negative, &frame);
        println!("cropped to {}x{}", negative.width, negative.height);
    }

    trace_min_channel(&negative);
    trace_densities(&negative, &base);

    println!(
        "film base   R {:.4}  G {:.4}  B {:.4}",
        base.transmittance[0], base.transmittance[1], base.transmittance[2]
    );

    let positive = pipeline::invert(&negative, &base, &settings);

    report("negative", &negative, &base);
    report("positive", &positive, &base);

    // The marks are drawn in the uncropped scan's coordinates, so the sample
    // points are read from it before the output takes it over.
    let samples = FilmBase::sprocket_sample_points(&negative, &bands, OVERLAY_SAMPLE_MARKS);
    let sprockets = FilmBase::sprocket_holes(&negative, &bands);
    let color_region = frame.sample_region().exposure_region(&negative, &base);
    println!(
        "color region: centre {:.0},{:.0} size {:.0}x{:.0} (from frame {:.0}x{:.0})",
        color_region.center[0],
        color_region.center[1],
        color_region.size[0],
        color_region.size[1],
        frame.size[0],
        frame.size[1]
    );

    let mut written = if options.negative_stage {
        // The negative itself is worth writing out, since a RAW file cannot be
        // opened in the usual viewers: this is the scan as captured.
        negative
    } else {
        positive
    };

    // With the crop off, the output *is* the scan the decisions were made on,
    // so the marks belong on it: one image shows the final frame and where the
    // pipeline looked, in the same coordinates. With the crop on, the output is
    // a sub-rectangle the marks would not line up with, so they go to a
    // separate annotated scan instead.
    if options.overlay {
        let marks = overlay::Marks {
            holes: &frame.holes,
            sprockets: &sprockets,
            film: frame.film.as_ref(),
            samples: &samples,
            color_region: Some(&color_region),
            crop: if options.no_crop { None } else { Some(&frame) },
        };
        written = draw_marks(&options.output, written, overlay_scan.as_ref(), &marks)?;
    }

    encode::encode(&written, &options.output)?;
    println!("wrote {}", options.output);
    Ok(())
}

/// The preview's parsed command line.
#[expect(
    clippy::struct_excessive_bools,
    reason = "a diagnostic tool's option switches are independent by nature: \
              they name operations, not states"
)]
struct Options {
    input: String,
    output: String,
    flip_h: bool,
    flip_v: bool,
    no_crop: bool,
    negative_stage: bool,
    overlay: bool,
    meter: bool,
    settings: Settings,
    film_base: Option<[f32; 3]>,
    base_sample: Option<[f32; 4]>,
}

/// Parses the preview's arguments, loading any profile eagerly.
fn parse_options() -> Result<Options, String> {
    let mut positional = Vec::new();
    let mut flip_h = false;
    let mut flip_v = false;
    let mut no_crop = false;
    let mut negative_stage = false;
    let mut overlay = false;
    let mut meter = false;
    let mut profile_path: Option<String> = None;
    let mut base_sample: Option<[f32; 4]> = None;
    let mut film_base: Option<[f32; 3]> = None;
    let mut density_max: Option<[f32; 3]> = None;
    let mut density_floor: Option<[f32; 3]> = None;
    let mut gamma: Option<[f32; 3]> = None;
    let mut pivot_density: Option<[f32; 3]> = None;

    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--flip-h" => flip_h = true,
            "--flip-v" => flip_v = true,
            "--no-crop" => no_crop = true,
            "--negative" => negative_stage = true,
            "--overlay" => overlay = true,
            "--meter" => meter = true,
            "--profile" => {
                let path = args.next().ok_or("--profile requires a path")?;
                profile_path = Some(path);
            }
            "--base-sample" => {
                let text = args.next().ok_or("--base-sample requires x0,y0,x1,y1")?;
                base_sample = Some(parse_base_sample(&text)?);
            }
            "--film-base" => {
                let text = args.next().ok_or("--film-base requires r,g,b")?;
                film_base = Some(parse_base_rgb(&text)?);
            }
            "--density-max" => {
                let text = args.next().ok_or("--density-max requires r,g,b")?;
                density_max = Some(parse_base_rgb(&text)?);
            }
            "--density-floor" => {
                let text = args.next().ok_or("--density-floor requires r,g,b")?;
                density_floor = Some(parse_base_rgb(&text)?);
            }
            "--gamma" => {
                let text = args.next().ok_or("--gamma requires r,g,b")?;
                gamma = Some(parse_base_rgb(&text)?);
            }
            "--pivot-density" => {
                let text = args.next().ok_or("--pivot-density requires r,g,b")?;
                pivot_density = Some(parse_base_rgb(&text)?);
            }
            _ => positional.push(argument),
        }
    }

    let [input, output] = positional.as_slice() else {
        return Err(
            "usage: invert_preview [--profile <path>] [--base-sample x0,y0,x1,y1] <scan.raw> <out.png> [--flip-h] [--flip-v] [--no-crop]"
                .into(),
        );
    };

    let settings = if let Some(path) = &profile_path {
        let text = std::fs::read_to_string(path)
            .map_err(|err| format!("could not read profile {path}: {err}"))?;
        Profile::parse(&text)
            .map_err(|err| format!("bad profile {path}: {err}"))?
            .settings
    } else {
        Settings::default()
    };
    // Flags win over a profile: tuning by eye is the loop this exists for.
    let settings = Settings {
        density_max: density_max.or(settings.density_max),
        density_floor: density_floor.or(settings.density_floor),
        gamma: gamma.or(settings.gamma),
        pivot_density: pivot_density.or(settings.pivot_density),
        ..settings
    };

    Ok(Options {
        input: input.clone(),
        output: output.clone(),
        flip_h,
        flip_v,
        no_crop,
        negative_stage,
        overlay,
        meter,
        settings,
        film_base,
        base_sample,
    })
}

/// Draws the marks on the output when it shares the scan's coordinates, or on a
/// copy of the scan beside it when a crop moved the picture underneath them.
fn draw_marks(
    output: &str,
    written: LinearImage,
    uncropped: Option<&LinearImage>,
    marks: &overlay::Marks<'_>,
) -> Result<LinearImage, Box<dyn std::error::Error>> {
    match uncropped {
        None => Ok(overlay::annotate(&written, marks)),
        Some(scan) => {
            write_overlay(output, scan, marks)?;
            Ok(written)
        }
    }
}

/// Writes the annotated scan beside the preview.
fn write_overlay(
    output: &str,
    negative: &LinearImage,
    marks: &overlay::Marks<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = overlay_path(output);
    encode::encode(&overlay::annotate(negative, marks), &path)?;
    println!("wrote {path}");
    Ok(())
}

/// The base measured from the film between the sprocket holes, or `None` when
/// there is nothing to measure.
///
/// The identity base means the sampler found nothing to read — a band with no
/// lit punch, or no gap between two holes. That is not a measurement, and
/// converting with it would render every frame wrong in silence.
fn measured_between_sprockets(
    image: &LinearImage,
    bands: &[film_base::FilmRegion],
) -> Option<FilmBase> {
    if bands.is_empty() {
        return None;
    }
    let base = FilmBase::detect_between_sprockets(image, bands);
    (base != FilmBase::IDENTITY).then_some(base)
}

/// The overlay for a preview goes beside it, marked.
fn overlay_path(output: &str) -> String {
    let path = std::path::Path::new(output);
    let stem = path.file_stem().map_or_else(
        || "overlay".to_owned(),
        |stem| stem.to_string_lossy().into_owned(),
    );
    let mut marked = path.to_path_buf();
    marked.set_file_name(format!("{stem}-overlay.png"));
    marked.to_string_lossy().into_owned()
}

/// Traces the per-channel density distributions the auto bounds read.
///
/// Auto white points come from percentiles of the *sampled* densities; this
/// shows those percentiles per channel, plus the same percentiles of the
/// linear values, so a wrong ceiling can be told apart from a wrong mask
/// division without guessing.
fn trace_densities(image: &LinearImage, base: &FilmBase) {
    use std::fmt::Write as _;

    let (pixels, _remainder) = image.data.as_chunks::<3>();
    let stride = (pixels.len() / (1 << 17)).max(1);

    println!("--- per-channel density percentiles (sampled) ---");
    for channel in 0..3 {
        let mut densities: Vec<f32> = pixels
            .iter()
            .step_by(stride)
            .map(|p| {
                let value = p[channel];
                let base_channel = base.transmittance[channel].max(f32::MIN_POSITIVE);
                // Same arithmetic as the inversion, for the same picture.
                (-((value / base_channel).clamp(0.0, 1.0).log10())).min(10.0)
            })
            .collect();

        densities.sort_unstable_by(f32::total_cmp);
        let mut line = String::new();
        for percentile_ in [50.0, 90.0, 99.0, 99.5, 99.9, 100.0] {
            let value = percentile::value_at_sorted(&densities, percentile_).unwrap_or(f32::NAN);
            let _ = writeln!(line, "  p{percentile_} {value:.3}");
        }
        println!("  ch {channel}:{line}");
    }

    println!("--- per-channel linear value percentiles (sampled) ---");
    for channel in 0..3 {
        let mut values: Vec<f32> = pixels.iter().step_by(stride).map(|p| p[channel]).collect();
        values.sort_unstable_by(f32::total_cmp);
        let mut line = String::new();
        for percentile_ in [10.0, 50.0, 90.0, 99.0, 99.9, 100.0] {
            let value = percentile::value_at_sorted(&values, percentile_).unwrap_or(f32::NAN);
            let _ = writeln!(line, "  p{percentile_} {value:.4}");
        }
        println!("  ch {channel}:{line}");
    }
}

/// Traces the distribution of the least channel.
///
/// The frame's boundary is between clear film (rebate, sprocket band, holes) and
/// exposed film, so the threshold has to sit in a much narrower band than
/// "light versus film" suggests. This is how that band gets chosen from data
/// rather than guessed.
fn trace_min_channel(image: &LinearImage) {
    let (pixels, _remainder) = image.data.as_chunks::<3>();

    // Where are the bad samples, if any? The answer says which stage produced
    // them: a border ring points at demosaic edge handling, scattered points at
    // sensor defects or arithmetic.
    let width = image.width as usize;
    let mut bad = 0usize;
    let mut first_bad = None;
    let mut last_bad = None;
    for (index, pixel) in pixels.iter().enumerate() {
        if pixel.iter().all(|channel| channel.is_finite()) {
            continue;
        }
        bad += 1;
        let (row, column) = (index / width, index % width);
        first_bad.get_or_insert((row, column));
        last_bad = Some((row, column));
    }
    println!("non-finite samples: {bad} of {} pixels", pixels.len());
    if let (Some(first), Some(last)) = (first_bad, last_bad) {
        println!("  first at row {}, column {}", first.0, first.1);
        println!("  last  at row {}, column {}", last.0, last.1);
    }

    let mut values: Vec<f32> = pixels
        .iter()
        .map(|p| p[0].min(p[1]).min(p[2]))
        .filter(|value| value.is_finite())
        .collect();
    values.sort_unstable_by(f32::total_cmp);

    println!("--- least-channel distribution ---");
    for percentile in [50.0f32, 90.0, 95.0, 99.0, 99.5, 99.9, 99.99, 100.0] {
        let value = percentile::value_at_sorted(&values, percentile).unwrap_or(f32::NAN);
        println!("  p{percentile:<6} {value:.4}");
    }

    println!("--- least channel by row band (top to bottom) ---");
    for band in 0..20 {
        let row = band * (image.height as usize / 20);
        println!("  row {:>5}  {:.4}", row, band_mean(image, band, 20, true));
    }

    println!("--- least channel by column band (left to right) ---");
    for band in 0..20 {
        let column = band * (image.width as usize / 20);
        println!(
            "  col {:>5}  {:.4}",
            column,
            band_mean(image, band, 20, false)
        );
    }
}

/// Mean least channel over one twentieth of the image, by row or by column.
///
/// Clear film sits at the top of the film's range and the picture below it, so
/// this shows where the rebate and the sprocket bands actually are — which is not
/// something to guess at.
fn band_mean(image: &LinearImage, band: usize, bands: usize, by_row: bool) -> f32 {
    let width = image.width as usize;
    let height = image.height as usize;
    let (pixels, _remainder) = image.data.as_chunks::<3>();

    let (outer, inner) = if by_row {
        (height, width)
    } else {
        (width, height)
    };
    let start = band * (outer / bands);
    let end = ((band + 1) * (outer / bands)).min(outer);

    let mut sum = 0.0f32;
    let mut count = 0usize;
    for a in start..end {
        for b in 0..inner {
            let (row, column) = if by_row { (a, b) } else { (b, a) };
            let pixel = &pixels[row * width + column];
            sum += pixel[0].min(pixel[1]).min(pixel[2]);
            count += 1;
        }
    }

    if count == 0 {
        return f32::NAN;
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "a diagnostic mean over a whole band"
    )]
    let count = count as f32;
    sum / count
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

/// Parses `--base-sample x0,y0,x1,y1` as fractions of the image.
fn parse_base_sample(text: &str) -> Result<[f32; 4], String> {
    let parts: Vec<&str> = text.split(',').collect();
    if parts.len() != 4 {
        return Err("--base-sample expects four comma-separated fractions".into());
    }
    let mut out = [0.0f32; 4];
    for (i, part) in parts.iter().enumerate() {
        out[i] = part
            .trim()
            .parse()
            .map_err(|_| format!("--base-sample value `{part}` is not a number"))?;
    }
    Ok(out)
}

fn parse_base_rgb(text: &str) -> Result<[f32; 3], String> {
    let parts: Vec<&str> = text.split(',').collect();
    if parts.len() != 3 {
        return Err("--film-base expects three comma-separated channel values".into());
    }
    let mut out = [0.0f32; 3];
    for (i, part) in parts.iter().enumerate() {
        out[i] = part
            .trim()
            .parse()
            .map_err(|_| format!("--film-base value `{part}` is not a number"))?;
    }
    Ok(out)
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
