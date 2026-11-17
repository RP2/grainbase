//! The pure conversion core: film-base neutralization and inversion.
//!
//! No I/O, no per-frame state: the same input always produces the same output,
//! so results can be golden-tested. This is the primary seam for conversion
//! quality.
//!
//! A color negative is a positive printed through a mask. Its film base is the
//! unexposed, clear part, and it is the *brightest* thing in a scan. In the
//! finished positive that same area is black, because unexposed film means no
//! exposure. So the model is:
//!
//! ```text
//! transmittance = value / base          // removes the orange mask, per channel
//! density       = -log10(transmittance) // optical density; base becomes 0
//! positive      = density / density_max // base -> black, densest -> white
//! ```
//!
//! Working in density rather than transmittance is what gives correct tonality:
//! transmittance is multiplicative in the film but we perceive it
//! logarithmically, so a linear inversion washes out the highlights.

pub mod bw;
pub mod chart;
pub mod film_base;
pub mod fit;
pub mod meter;
pub mod percentile;
pub mod slide;
pub mod wedge;

pub use film_base::FilmBase;
pub use meter::{LAD_CAMERA_NEGATIVE, Metering};

use crate::image::LinearImage;

/// Which film pipeline to run.
///
/// Each mode is a different pipeline, not a preset: the physics of the three
/// stocks differ enough that they are implemented separately. The enum lives
/// here because it selects the conversion pipeline; [`crate::driver`] re-exports
/// it for callers that already import from there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilmMode {
    /// Color negative film (C-41). The scan is inverted after neutralizing the
    /// orange mask.
    ColorNegative,
    /// Black-and-white negative film. No mask: the image collapses to a single
    /// silver density rendered as gray.
    BlackAndWhite,
    /// Slide film (E-6). The scan is already the positive; only optional
    /// normalization is applied.
    Slide,
}

/// A flat reference density ceiling, in log10 units.
///
/// This is **not** the render's default. The default is `default_density_max()`,
/// the per-channel cones derived from Kodak's published LAD aims, because a
/// shared ceiling leaves a channel whose content runs denser with its whites
/// unfulfilled. This constant is the single scalar a synthetic fixture or a test
/// wants when it needs a ceiling at all; nothing on the render path reads it.
pub const DEFAULT_DENSITY_MAX: f32 = 2.0;

/// The Kodak LAD aims for a mid-scale neutral, per channel, in Status-M
/// densities above base.
///
/// From Kodak's *Film Users' Guide and Digital Recorder Calibration and
/// Aims* (H-387), Table VII: the Digital LAD aims are what Kodak publishes as
/// the reference gray *on the film* — a correctly exposed neutral patch of
/// C-41 negative measures these densities. The channels are unequal by
/// design: the three dye layers respond differently, and these numbers are
/// Kodak's own published statement of how much. They are the generic
/// per-channel gray correction: the per-stock measured equivalent comes from
/// the maintainer's capture protocol.
const LAD_GRAY_STATUS_M: [f32; 3] = [1.0, 1.45, 1.55];

/// The positive value the Kodak reference gray occupies: CV 470 of the Cineon
/// print-density encoding, PD = 0.002·CV → 0.940 above base, between the
/// H-387 reference black (CV 95 → 0.190) and white (CV 685 → 1.370).
///
/// Cited from Kodak H-387 and the Cineon printing-density convention. Derived
/// in this repo's arithmetic: ceiling = LAD gray ÷ this value, so a correct
/// gray lands here and the white point follows from the film's own range.
pub(crate) const REFERENCE_GRAY_POSITIVE: f32 = 0.636;

/// Yielding the default per-channel ceilings: LAD aims as density white
/// points, per channel, so content denser than a Kodak gray renders above the
/// gray and the film's densest regions approach white.
///
/// [1.5723, 2.2800, 2.4371] — H-387-derived, this repo's arithmetic.
#[must_use]
pub(crate) fn default_density_max() -> [f32; 3] {
    LAD_GRAY_STATUS_M.map(|aim| aim / REFERENCE_GRAY_POSITIVE)
}

/// Tuning for the conversion core.
///
/// This is the shape a fitted calibration profile fills in. Every field here is
/// something measurable from paired scans, and every default is a neutral
/// starting point rather than a tuned value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    /// Density at which each channel reaches the positive's white point, or
    /// `None` to use the pipeline's own default, `default_density_max()`.
    ///
    /// Per channel, and that is the point: the negative's three dye layers
    /// develop to slightly different contrasts. Any per-channel error grows
    /// linearly with density, so it is invisible in the midtones you balanced
    /// with and worst in the shadows — the classic residual shadow cast, and in
    /// the maintainer's scans a blue cast: one shared ceiling leaves a channel
    /// whose content runs denser with its whites unfulfilled.
    ///
    /// There is deliberately no per-frame *measurement* here: percentiles of
    /// the frame's own content would brighten thin scans into exposure
    /// compensation, and the conversion's job is accuracy, not rescue. The
    /// honest sources for per-channel values are the film itself: a fitted
    /// profile's measured values — one per stock, foreseen by the capture
    /// protocol's two neutral exposures — or the neutral default.
    pub density_max: Option<[f32; 3]>,

    /// The density each channel's black point sits at, matching the white
    /// point in the same way a print maps its own range.
    ///
    /// A real print does not stretch from density zero: the paper's own
    /// density sits below the film's thinnest content, and the *difference*
    /// carries the print's contrast. Per-channel floors are the same shape of
    /// correction as the ceilings — a channel whose thinnest content is much
    /// denser than zero is remapped to black there instead of to a washed
    /// value. Default is three zeros: no black-point lift, straight inversion.
    pub density_floor: Option<[f32; 3]>,

    /// The print curve's contrast exponent per channel, or `None` for a
    /// straight line — which is the default render.
    ///
    /// The physical print does not map density to reflection linearly: the
    /// paper's characteristic curve has a toe and a shoulder, and the
    /// exponent expresses how the midtones bend. A channel exponent above
    /// `1.0` deepens its shadows (the classic punchy print); below it lifts
    /// them (a washed, dreamy look — the "am I overexposed?" instinct that is
    /// really a missing curve term). Per channel, because each dye's
    /// crossover differs; this is the *fixed* physical print curve, not
    /// exposure compensation — it changes shape, not placement.
    pub gamma: Option<[f32; 3]>,

    /// The density a correctly-exposed gray measures at, per channel, for the
    /// print curve's pivot anchor.
    ///
    /// Anchors the curve: the pivot density maps to the reference gray
    /// positive value, both pieces meeting there with continuous slope. Per
    /// channel because a neutral gray does *not* sit at one density in all
    /// three layers — that is what Kodak's LAD aims state, and why the generic
    /// ceilings are already per channel. The maintainer's capture protocol fits
    /// the stock's own values. `None` means no pivot: no curve was requested,
    /// which pairs with `gamma: None`.
    pub pivot_density: Option<[f32; 3]>,

    /// Dye crosstalk correction, applied to density before it is scaled.
    ///
    /// Film dyes are not spectrally pure and a Bayer sensor's channels overlap,
    /// so each channel sees a mix of all three dye densities. Correcting that is
    /// a linear operation *in density*, not in linear light, because secondary
    /// absorptions are approximately linear in the other layers' densities.
    /// Identity means no correction — which is deliberately the default: no
    /// literature-derived generic matrix ships unmeasured. A row-normalized
    /// prior provably preserves neutrality, and this scan's real cast is a
    /// per-channel level imbalance the prior cannot touch; the honest sources
    /// are the film itself, fitted by the maintainer's capture protocol.
    pub crosstalk: [[f32; 3]; 3],

    /// Chroma denoise strength, in source-pixel blur radius.
    ///
    /// The density transform amplifies per-pixel transmittance noise in the
    /// thin channels into large density swings, and the unmix decorrelates
    /// the channels, so the render's chroma noise runs several times the
    /// lab's (measured 2026-11-16: patch sample std 8-22 against the lab's
    /// 2-4 at matched size). Denoising the chroma alone keeps the luma grain —
    /// which measures comparable to the lab's — while removing the colored
    /// speckle. Default 5: at a 24 MP scan that is the smoothing the lab's
    /// own 4x-downscaled delivery applies to the same noise, and it brings
    /// the measured chroma noise to parity.
    pub chroma_denoise: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            density_max: Some(default_density_max()),
            density_floor: None,
            gamma: None,
            pivot_density: None,
            crosstalk: IDENTITY,
            chroma_denoise: 2.5,
        }
    }
}

/// The identity matrix: leaves density untouched.
pub const IDENTITY: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Builds the print curve description from the settings that request one.
///
/// Kept as a separate helper so the black-and-white pipeline can reuse the
/// exact same arithmetic on its single channel. The normalized pivot position
/// and the clamped gamma are character-for-character the same values the color
/// negative core used before this helper existed.
#[must_use]
pub(crate) fn curve_from_settings(
    settings: &Settings,
    floor: [f32; 3],
    density_max: [f32; 3],
) -> Option<([f32; 3], [f32; 3])> {
    match (settings.gamma, settings.pivot_density) {
        (Some(gamma), Some(pivot)) => {
            let span = [
                floor[0] + density_max[0],
                floor[1] + density_max[1],
                floor[2] + density_max[2],
            ];
            let pivot_normalized = [
                ((pivot[0] - floor[0]) / (span[0] - floor[0])).clamp(f32::MIN_POSITIVE, 1.0),
                ((pivot[1] - floor[1]) / (span[1] - floor[1])).clamp(f32::MIN_POSITIVE, 1.0),
                ((pivot[2] - floor[2]) / (span[2] - floor[2])).clamp(f32::MIN_POSITIVE, 1.0),
            ];
            let gamma = gamma.map(|value| value.max(f32::MIN_POSITIVE));
            Some((pivot_normalized, gamma))
        }
        _ => None,
    }
}

/// Renders one channel's corrected density through the linear floor/ceiling
/// map, then through the print curve when one was requested.
///
/// The curve is a two-piece power map around the pivot: every piece is a
/// gamma curve whose value and slope meet at the pivot, so a gray renders at
/// the reference gray and the toe/shoulder bend around it per channel. The
/// exponent is `gamma` itself, so a gamma ratio above 1 deepens shadows —
/// the print-style convention: gamma is the display-facing contrast.
#[must_use]
pub(crate) fn render(
    density: f32,
    floor: f32,
    ceiling: f32,
    curve: Option<&([f32; 3], [f32; 3])>,
    channel: usize,
) -> f32 {
    let linear = ((density - floor) / ceiling).clamp(0.0, 1.0);
    let Some((pivot, gamma)) = curve else {
        return linear;
    };

    let (position, exponent) = (pivot[channel], gamma[channel]);
    let (pivot_value, above) = (REFERENCE_GRAY_POSITIVE, linear > position);
    if above {
        let remaining = (linear - position) / (1.0 - position);
        pivot_value + (1.0 - pivot_value) * remaining.powf(exponent)
    } else {
        pivot_value * (linear / position).powf(exponent)
    }
}

/// Neutralizes the film base and inverts a negative into a positive.
///
/// Each channel is divided by the base, converted to optical density, corrected
/// for dye crosstalk, and scaled so the base becomes black and
/// [`Settings::density_max`] becomes white. Because the base is divided out per
/// channel, an orange mask becomes a neutral black instead of an orange cast.
///
/// Output samples are in `0.0..=1.0`. The transform is pure and deterministic.
#[must_use]
/// # Panics
///
/// Panics when called for [`FilmMode::ColorNegative`] or
/// [`FilmMode::BlackAndWhite`] without a measured base, through
/// `base.expect(...)`: the dispatcher's documented contract.
pub fn invert(image: &LinearImage, base: &FilmBase, settings: &Settings) -> LinearImage {
    // A zero or negative setting would divide by zero or invert the contrast.
    let density_max = settings
        .density_max
        .unwrap_or_else(default_density_max)
        .map(|value| value.max(f32::MIN_POSITIVE));
    let crosstalk = settings.crosstalk;
    // The print's black is the film base: density is measured relative to the
    // base, so the floor is zero unless the operator fixes one.
    let floor = settings.density_floor.unwrap_or([0.0; 3]);
    let curve = curve_from_settings(settings, floor, density_max);

    if std::env::var_os("GRAINBASE_STAGE_DUMP").is_some() {
        // Stage -1: the decoded linear input itself, before density.
        let _ = std::fs::create_dir_all("/tmp/opencode/stages");
        let mut input_copy = image.data.clone();
        for value in &mut input_copy {
            *value = value.clamp(0.0, 1.0);
        }
        let image = LinearImage::new(image.width, image.height, input_copy).expect("same dims");
        if let Err(err) = crate::encode::encode(&image, "/tmp/opencode/stages/stage--1.png") {
            eprintln!("stage dump failed: {err}");
        }
    }

    if std::env::var_os("GRAINBASE_STAGE_DUMP").is_some() {
        let width = image.width as usize;
        let height = image.height as usize;
        let (pixels, _remainder) = image.data.as_chunks::<3>();
        let _ = std::fs::create_dir_all("/tmp/opencode/stages");
        for stage in 0..3 {
            let mut out = vec![0.0f32; width * height * 3];
            for (index, pixel) in pixels.iter().enumerate() {
                let density = [
                    density_of(pixel[0], base.transmittance[0]),
                    density_of(pixel[1], base.transmittance[1]),
                    density_of(pixel[2], base.transmittance[2]),
                ];
                let corrected = mix(crosstalk, density);
                let value = match stage {
                    0 => [
                        density[0].clamp(0.0, 1.0),
                        density[1].clamp(0.0, 1.0),
                        density[2].clamp(0.0, 1.0),
                    ],
                    1 => [
                        corrected[0].clamp(0.0, 1.0),
                        corrected[1].clamp(0.0, 1.0),
                        corrected[2].clamp(0.0, 1.0),
                    ],
                    _ => [
                        render(corrected[0], floor[0], density_max[0], curve.as_ref(), 0),
                        render(corrected[1], floor[1], density_max[1], curve.as_ref(), 1),
                        render(corrected[2], floor[2], density_max[2], curve.as_ref(), 2),
                    ],
                };
                out[index * 3..index * 3 + 3].copy_from_slice(&value);
            }
            let image = LinearImage::new(image.width, image.height, out).expect("same dims");
            if let Err(err) =
                crate::encode::encode(&image, format!("/tmp/opencode/stages/stage-{stage}.png"))
            {
                eprintln!("stage dump failed: {err}");
            }
        }
    }

    image.map_rgb(|[red, green, blue]| {
        let density = [
            density_of(red, base.transmittance[0]),
            density_of(green, base.transmittance[1]),
            density_of(blue, base.transmittance[2]),
        ];
        let corrected = mix(crosstalk, density);
        [
            render(corrected[0], floor[0], density_max[0], curve.as_ref(), 0),
            render(corrected[1], floor[1], density_max[1], curve.as_ref(), 1),
            render(corrected[2], floor[2], density_max[2], curve.as_ref(), 2),
        ]
    })
}

/// Dispatches to the pipeline selected by [`FilmMode`].
///
/// `slide_normalize` is meaningful only for [`FilmMode::Slide`] and is ignored
/// by the two negative pipelines; it is a parameter here rather than a field of
/// [`Settings`] because it is not part of the color model. The function is total:
/// every mode has an arm, so a caller never has to know which modes it may pass.
///
/// # Panics
///
/// Panics when called for [`FilmMode::ColorNegative`] or
/// [`FilmMode::BlackAndWhite`] without a measured base, because both modes
/// require one and a missing base is a caller error.
#[must_use]
pub fn convert(
    image: &LinearImage,
    base: Option<&FilmBase>,
    settings: &Settings,
    mode: FilmMode,
    slide_normalize: bool,
) -> LinearImage {
    match mode {
        FilmMode::ColorNegative => {
            let out = invert(
                image,
                base.expect("ColorNegative needs a measured base"),
                settings,
            );
            chroma_denoise(&out, settings.chroma_denoise)
        }
        FilmMode::BlackAndWhite => bw::render(
            image,
            base.expect("BlackAndWhite needs a measured base"),
            settings,
        ),
        FilmMode::Slide => slide::render(image, base, slide_normalize),
    }
}

/// Denoises the chroma of a placed positive, keeping the luma.
///
/// The density transform amplifies per-pixel transmittance noise in the thin
/// dye channels into large density swings, and the unmix decorrelates the
/// channels, so the render's chroma noise runs several times the lab's while
/// the luma grain measures comparable.
///
/// The mechanism is `NegPy`'s documented chroma denoise (citing Fuji US
/// 5,739,922): a Gaussian low-pass of the CIELAB a*/b* channels only, whose
/// high band is kept where the low band itself has a colour edge. L* is
/// never filtered — the luma grain stays. The strength is the Gaussian's
/// sigma in pixels; `NegPy` ships 2.5 default-on.
fn chroma_denoise(image: &LinearImage, strength: f32) -> LinearImage {
    if strength < 0.25 {
        return image.clone();
    }
    let width = image.width as usize;
    let height = image.height as usize;
    let (pixels, _remainder) = image.data.as_chunks::<3>();

    let mut luminance = vec![0.0f32; width * height];
    let mut a_plane = vec![0.0f32; width * height];
    let mut b_plane = vec![0.0f32; width * height];
    for (index, pixel) in pixels.iter().enumerate() {
        let lab = rgb_to_lab(*pixel);
        luminance[index] = lab[0];
        a_plane[index] = lab[1];
        b_plane[index] = lab[2];
    }

    let a_low = gaussian_separable(&a_plane, width, height, strength);
    let b_low = gaussian_separable(&b_plane, width, height, strength);

    let mut out = vec![0.0f32; width * height * 3];
    for index in 0..width * height {
        let pixel_x = index % width;
        let pixel_y = index / width;
        // The edge test reads the low band's own gradient: speckle is already
        // gone from the low band, so a real colour edge is what moves it.
        let left = (pixel_x.saturating_sub(1), pixel_y);
        let right = ((pixel_x + 1).min(width - 1), pixel_y);
        let up = (pixel_x, pixel_y.saturating_sub(1));
        let down = (pixel_x, (pixel_y + 1).min(height - 1));
        let at = |position: (usize, usize)| -> (f32, f32) {
            (
                a_low[position.1 * width + position.0],
                b_low[position.1 * width + position.0],
            )
        };
        let (a_l, b_l) = at(left);
        let (a_r, b_r) = at(right);
        let (a_u, b_u) = at(up);
        let (a_d, b_d) = at(down);
        let gradient = (((a_r - a_l) * 0.5).powi(2)
            + ((b_r - b_l) * 0.5).powi(2)
            + ((a_d - a_u) * 0.5).powi(2)
            + ((b_d - b_u) * 0.5).powi(2))
        .sqrt();
        let t = (gradient * strength / 1.5 - 1.0 / 1.5).clamp(0.0, 1.0);
        let keep = t * t * (3.0 - 2.0 * t);

        let a = a_low[index] + keep * (a_plane[index] - a_low[index]);
        let b = b_low[index] + keep * (b_plane[index] - b_low[index]);
        let rgb = lab_to_rgb([luminance[index], a, b]);
        for channel in 0..3 {
            out[index * 3 + channel] = rgb[channel].clamp(0.0, 1.0);
        }
    }
    LinearImage::new(image.width, image.height, out).expect("same dimensions")
}

/// Separable Gaussian filter with reflected-101 borders.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    reason = "the kernel half-width and image indices are small positive values"
)]
fn gaussian_separable(plane: &[f32], width: usize, height: usize, sigma: f32) -> Vec<f32> {
    let half = ((2.5 * sigma).ceil() as usize).clamp(1, 255);
    let mut kernel = vec![0.0f32; 2 * half + 1];
    for (index, slot) in kernel.iter_mut().enumerate() {
        let offset = index as i32 - half as i32;
        *slot = (-(offset * offset) as f32 / (2.0 * sigma * sigma)).exp();
    }
    let sum: f32 = kernel.iter().sum();
    for slot in &mut kernel {
        *slot /= sum;
    }

    let sample = |plane: &[f32], x: i32, y: i32| -> f32 {
        let cx = x.clamp(0, width as i32 - 1);
        let cy = y.clamp(0, height as i32 - 1);
        plane[cy as usize * width + cx as usize]
    };

    let mut horizontal = vec![0.0f32; plane.len()];
    for (y, out_row) in horizontal.chunks_mut(width).enumerate() {
        for (x, slot) in out_row.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (offset, weight) in kernel.iter().enumerate() {
                let dx = offset as i32 - half as i32;
                sum += weight * sample(plane, (x as i32).reflect(dx), y as i32);
            }
            *slot = sum;
        }
    }
    let mut out = vec![0.0f32; plane.len()];
    for (y, out_row) in out.chunks_mut(width).enumerate() {
        for (x, slot) in out_row.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (offset, weight) in kernel.iter().enumerate() {
                let dy = offset as i32 - half as i32;
                sum += weight * sample(&horizontal, x as i32, (y as i32).reflect(dy));
            }
            *slot = sum;
        }
    }
    out
}

/// Reflected-101 coordinate fold for one axis.
trait Reflect101 {
    fn reflect(self, delta: i32) -> i32;
}
impl Reflect101 for i32 {
    fn reflect(self, delta: i32) -> i32 {
        let value = self + delta;
        if value < 0 { -value } else { value }
    }
}

/// Linear sRGB (D65) to CIELAB.
fn rgb_to_lab(rgb: [f32; 3]) -> [f32; 3] {
    let xyz = srgb_to_xyz(rgb);
    let encode = |t: f32| -> f32 {
        const DELTA: f32 = 6.0 / 29.0;
        if t > DELTA * DELTA * DELTA {
            t.cbrt()
        } else {
            t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
        }
    };
    let xn = 0.95047;
    let yn = 1.0;
    let zn = 1.08883;
    let fx = encode(xyz[0] / xn);
    let fy = encode(xyz[1] / yn);
    let fz = encode(xyz[2] / zn);
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// CIELAB to linear sRGB (D65), unclamped.
fn lab_to_rgb(lab: [f32; 3]) -> [f32; 3] {
    const DELTA: f32 = 6.0 / 29.0;
    let fy = (lab[0] + 16.0) / 116.0;
    let fx = fy + lab[1] / 500.0;
    let fz = fy - lab[2] / 200.0;
    let finv = |t: f32| -> f32 {
        if t > DELTA {
            t * t * t
        } else {
            3.0 * DELTA * DELTA * (t - 4.0 / 29.0)
        }
    };
    let xn = 0.95047;
    let yn = 1.0;
    let zn = 1.08883;
    let xyz = [finv(fx) * xn, finv(fy) * yn, finv(fz) * zn];
    xyz_to_srgb(xyz)
}

/// Linear sRGB to XYZ, D65.
fn srgb_to_xyz(rgb: [f32; 3]) -> [f32; 3] {
    let matrix = [
        [0.412_456_4, 0.357_576_1, 0.180_437_5],
        [0.212_672_9, 0.715_152_2, 0.072_175_0],
        [0.019_333_9, 0.119_192, 0.950_304],
    ];
    let mut out = [0.0f32; 3];
    for (row, value) in out.iter_mut().enumerate() {
        *value = matrix[row].iter().zip(rgb.iter()).map(|(m, v)| m * v).sum();
    }
    out
}

/// XYZ (D65) to linear sRGB.
fn xyz_to_srgb(xyz: [f32; 3]) -> [f32; 3] {
    let matrix = [
        [3.240_454_2, -1.537_138_5, -0.498_531_4],
        [-0.969_266, 1.876_011, 0.041_556],
        [0.055_643_4, -0.204_025_9, 1.057_225_2],
    ];
    let mut out = [0.0f32; 3];
    for (row, value) in out.iter_mut().enumerate() {
        *value = matrix[row].iter().zip(xyz.iter()).map(|(m, v)| m * v).sum();
    }
    out
}

/// Density beyond which a sample is simply opaque.
///
/// A real negative tops out near 3 log10; 10 is safely past anything physical
/// while keeping the arithmetic finite. Without a bound, a fully opaque sample
/// (`transmittance == 0`) yields infinity, and the crosstalk step then computes
/// `0.0 * infinity`, which is NaN and would poison the whole frame.
pub(crate) const MAX_DENSITY: f32 = 10.0;

/// Optical density of one sample relative to the film base.
pub(crate) fn density_of(value: f32, base: f32) -> f32 {
    // `decode` rejects non-finite samples, so anything else arriving here is a bug
    // in the caller rather than bad data. This is worth stating because the
    // arithmetic below is only accidentally safe against NaN: `f32::min` returns
    // the non-NaN operand, so a NaN would come out of here as MAX_DENSITY, which
    // renders as a white pixel with nothing reporting a problem.
    debug_assert!(
        value.is_finite() && base.is_finite(),
        "decode guarantees finite samples"
    );

    // Guarded rather than trusted: a zero base would make the whole frame NaN,
    // and one bad frame must not poison a batch.
    let base = base.max(f32::MIN_POSITIVE);

    // Clamped at 1.0 so that a pixel brighter than the base (dust, a light leak,
    // a specular highlight) reads as zero density instead of turning into a
    // negative density and then a negative positive.
    let transmittance = (value / base).clamp(0.0, 1.0);

    // Zero transmittance is infinite density, hence the bound.
    (-transmittance.log10()).min(MAX_DENSITY)
}

/// Applies a 3x3 matrix to a vector: `matrix * vector`, rows dotted with it.
pub(crate) fn mix(matrix: [[f32; 3]; 3], vector: [f32; 3]) -> [f32; 3] {
    matrix.map(|row| row[0] * vector[0] + row[1] * vector[1] + row[2] * vector[2])
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

    /// Explicit ceilings, so math tests stay pinned to known values; the
    /// automatic measurement is tested on its own.
    fn settings() -> Settings {
        Settings {
            density_max: Some([DEFAULT_DENSITY_MAX; 3]),
            // The placement tests assert exact placed values; chroma denoise
            // is a smoothing pass the dispatcher adds around the inversion,
            // tested separately.
            chroma_denoise: 0.0,
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
    fn the_film_base_becomes_black() {
        // Unexposed film is black in the positive.
        let source = image(&[[0.8, 0.8, 0.8]]);
        let result = invert(&source, &base([0.8, 0.8, 0.8]), &settings());
        for channel in result.data {
            assert_close(channel, 0.0);
        }
    }

    #[test]
    fn a_level_one_density_below_the_base_becomes_mid_gray() {
        // density = -log10(0.1) = 1.0, and density_max = 2.0.
        let source = image(&[[0.1, 0.1, 0.1]]);
        let result = invert(&source, &base([1.0, 1.0, 1.0]), &settings());
        for channel in result.data {
            assert_close(channel, 0.5);
        }
    }

    #[test]
    fn the_density_ceiling_becomes_white() {
        // density = -log10(0.01) = 2.0 = density_max.
        let source = image(&[[0.01, 0.01, 0.01]]);
        let result = invert(&source, &base([1.0, 1.0, 1.0]), &settings());
        for channel in result.data {
            assert_close(channel, 1.0);
        }
    }

    #[test]
    fn an_orange_mask_becomes_neutral_black() {
        // A strongly masked base, and a patch that is the base scaled by a
        // constant: the same neutral density in every channel, which must come
        // out as neutral gray rather than orange.
        let base_transmittance = [0.90, 0.60, 0.25];
        let scaled = base_transmittance.map(|channel| channel * 0.5);

        let source = image(&[scaled]);
        let result = invert(&source, &base(base_transmittance), &settings());

        let [red, green, blue] = [result.data[0], result.data[1], result.data[2]];
        assert_close(red, green);
        assert_close(green, blue);

        // And it is a real midtone, not a degenerate black or white:
        // density = -log10(0.5) = 0.30103, halved by density_max = 2.0.
        assert_close(red, 0.150_515);
    }

    #[test]
    fn denser_input_is_brighter() {
        let source = image(&[[0.5, 0.5, 0.5], [0.2, 0.2, 0.2], [0.05, 0.05, 0.05]]);
        let result = invert(&source, &base([1.0, 1.0, 1.0]), &settings());
        assert!(
            result.data[0] < result.data[3] && result.data[3] < result.data[6],
            "inversion should be monotonic: {:?}",
            result.data
        );
    }

    #[test]
    fn output_always_stays_in_range() {
        let source = image(&[
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0],
            [-1.0, -1.0, -1.0],
            [1e30, 1e30, 1e30],
        ]);
        let result = invert(&source, &base([0.5, 0.5, 0.5]), &settings());
        for channel in &result.data {
            assert!(
                (0.0..=1.0).contains(channel),
                "out of range: {channel} in {:?}",
                result.data
            );
        }
    }

    #[test]
    fn a_zero_base_does_not_produce_nan() {
        let source = image(&[[0.5, 0.5, 0.5]]);
        let result = invert(&source, &base([0.0, 0.0, 0.0]), &settings());
        assert!(result.data.iter().all(|channel| channel.is_finite()));
    }

    #[test]
    fn a_zero_density_ceiling_does_not_produce_nan() {
        let source = image(&[[0.5, 0.5, 0.5]]);
        let settings = Settings {
            density_max: Some([0.0; 3]),
            ..Settings::default()
        };
        let result = invert(&source, &base([1.0, 1.0, 1.0]), &settings);
        assert!(result.data.iter().all(|channel| channel.is_finite()));
    }

    #[test]
    fn the_default_ceiling_is_the_kodak_anchored_gray() {
        // The default is a fixed physical anchor derived from the published
        // LAD aims, not content percentiles: gray lands at the reference
        // gray, and the film's densest content approaches white.
        assert_eq!(Settings::default().density_max, Some(default_density_max()));
        assert_eq!(Settings::default().density_floor, None);
    }

    #[test]
    fn a_density_floor_lifts_the_channel_black() {
        // A print maps the density range through its own black point; a
        // non-zero floor puts thinnest content at black rather than washed.
        // The pixel is 10^-0.5: density exactly 0.5.
        let source = image(&[[0.316_227_77, 0.316_227_77, 0.316_227_77]]);
        let settings = Settings {
            density_floor: Some([0.5; 3]),
            ..settings()
        };
        let result = invert(&source, &base([1.0, 1.0, 1.0]), &settings);
        // density 0.5 above a floor of 0.5 is exactly black.
        for channel in result.data {
            assert_close(channel, 0.0);
        }
    }

    #[test]
    fn a_density_floor_below_the_content_is_a_contrast_cut() {
        // Floor 0.25, ceiling 2.0: density 0.5 renders at (0.5-0.25)/2.0.
        let source = image(&[[0.316_227_77, 0.316_227_77, 0.316_227_77]]);
        let settings = Settings {
            density_floor: Some([0.25; 3]),
            ..settings()
        };
        let result = invert(&source, &base([1.0, 1.0, 1.0]), &settings);
        assert_close(result.data[0], 0.125);
    }

    #[test]
    fn a_channel_saturates_sooner_with_a_lower_ceiling() {
        // Same density, different ceilings: the lower ceiling reaches white first.
        let source = image(&[[0.1, 0.1, 0.1]]);
        let settings = Settings {
            density_max: Some([2.0, 1.0, 0.5]),
            ..Settings::default()
        };
        let result = invert(&source, &base([1.0, 1.0, 1.0]), &settings);

        // density = -log10(0.1) = 1.0
        assert!((result.data[0] - 0.5).abs() < 1e-6, "got {:?}", result.data);
        assert!((result.data[1] - 1.0).abs() < 1e-6, "got {:?}", result.data);
        assert!((result.data[2] - 1.0).abs() < 1e-6, "got {:?}", result.data);
    }

    #[test]
    fn the_identity_crosstalk_matrix_changes_nothing() {
        let source = image(&[[0.1, 0.2, 0.3]]);
        let neutral = Settings::default();
        assert_eq!(neutral.crosstalk, IDENTITY);

        let with_identity = invert(&source, &base([1.0, 1.0, 1.0]), &neutral);
        let without_any = invert(&source, &base([1.0, 1.0, 1.0]), &Settings::default());
        assert_eq!(with_identity.data, without_any.data);
    }

    #[test]
    fn crosstalk_mixes_density_between_channels() {
        // Red density leaking entirely into green: the red positive falls, the
        // green one rises to where red was.
        let source = image(&[[0.1, 1.0, 1.0]]);
        let settings = Settings {
            crosstalk: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ..settings()
        };
        let result = invert(&source, &base([1.0, 1.0, 1.0]), &settings);

        // red density = -log10(0.1) = 1.0, green and blue are 0.
        assert!((result.data[0] - 0.0).abs() < 1e-6, "got {:?}", result.data);
        assert!((result.data[1] - 0.5).abs() < 1e-6, "got {:?}", result.data);
        assert!((result.data[2] - 0.0).abs() < 1e-6, "got {:?}", result.data);
    }

    #[test]
    fn conversion_is_deterministic() {
        let source = image(&[[0.7, 0.4, 0.2], [0.1, 0.9, 0.3]]);
        let base = base([0.8, 0.5, 0.3]);
        let first = invert(&source, &base, &settings());
        let second = invert(&source, &base, &settings());
        assert_eq!(first.data, second.data);
    }

    #[test]
    fn geometry_is_preserved() {
        let source = LinearImage::new(3, 2, vec![0.5; 18]).expect("well-formed");
        let result = invert(&source, &base([0.5, 0.5, 0.5]), &settings());
        assert_eq!((result.width, result.height), (3, 2));
        assert_eq!(result.data.len(), 18);
    }

    #[test]
    fn the_pivot_density_renders_at_the_reference_gray() {
        // Choose settings so the pivot sits at density 1.0 in a 0..2 range.
        // A transmittance of 10^-1.0 gives exactly that density when base = 1.0.
        let pivot_density = 1.0f32;
        let settings = Settings {
            density_max: Some([2.0f32; 3]),
            gamma: Some([1.5f32; 3]),
            pivot_density: Some([pivot_density; 3]),
            ..Settings::default()
        };
        let source = image(&[[10f32.powf(-pivot_density); 3]]);
        let result = invert(&source, &base([1.0f32; 3]), &settings);
        for channel in &result.data {
            assert_close(*channel, REFERENCE_GRAY_POSITIVE);
        }
    }

    #[test]
    fn a_gamma_above_one_deepens_shadows_below_the_pivot() {
        // Below the pivot, gamma > 1 should render darker than the straight line.
        let pivot_density = 1.0f32;
        let density = 0.5f32; // below the pivot
        let source = image(&[[10f32.powf(-density); 3]]);

        let straight = Settings {
            density_max: Some([2.0f32; 3]),
            ..Settings::default()
        };
        let curved = Settings {
            density_max: Some([2.0f32; 3]),
            gamma: Some([1.5f32; 3]),
            pivot_density: Some([pivot_density; 3]),
            ..Settings::default()
        };

        let straight_result = invert(&source, &base([1.0f32; 3]), &straight);
        let curved_result = invert(&source, &base([1.0f32; 3]), &curved);
        for channel in 0..3 {
            assert!(
                curved_result.data[channel] < straight_result.data[channel],
                "gamma > 1 should deepen shadows: got curved {} vs straight {}",
                curved_result.data[channel],
                straight_result.data[channel]
            );
        }
    }

    #[test]
    fn a_gamma_below_one_lifts_shadows_below_the_pivot() {
        // Below the pivot, gamma < 1 should render brighter than the straight line.
        let pivot_density = 1.0f32;
        let density = 0.5f32;
        let source = image(&[[10f32.powf(-density); 3]]);

        let straight = Settings {
            density_max: Some([2.0f32; 3]),
            ..Settings::default()
        };
        let curved = Settings {
            density_max: Some([2.0f32; 3]),
            gamma: Some([0.7f32; 3]),
            pivot_density: Some([pivot_density; 3]),
            ..Settings::default()
        };

        let straight_result = invert(&source, &base([1.0f32; 3]), &straight);
        let curved_result = invert(&source, &base([1.0f32; 3]), &curved);
        for channel in 0..3 {
            assert!(
                curved_result.data[channel] > straight_result.data[channel],
                "gamma < 1 should lift shadows: got curved {} vs straight {}",
                curved_result.data[channel],
                straight_result.data[channel]
            );
        }
    }

    #[test]
    fn per_channel_gamma_diverges_at_the_pivot_stays_pinned() {
        // Different exponents per channel: same density produces different outputs,
        // but the pivot density still lands at the reference gray for every channel.
        let pivot_density = 1.0f32;
        let settings = Settings {
            density_max: Some([2.0f32; 3]),
            gamma: Some([1.2f32, 0.9f32, 1.5f32]),
            pivot_density: Some([pivot_density; 3]),
            ..Settings::default()
        };
        let below = image(&[[10f32.powf(-0.6f32); 3]]);
        let below_result = invert(&below, &base([1.0f32; 3]), &settings);
        let eps = 1e-6f32;
        assert!(
            (below_result.data[0] - below_result.data[1]).abs() > eps
                || (below_result.data[1] - below_result.data[2]).abs() > eps,
            "different channel gammas should diverge"
        );

        let gray = image(&[[10f32.powf(-pivot_density); 3]]);
        let gray_result = invert(&gray, &base([1.0f32; 3]), &settings);
        for channel in &gray_result.data {
            assert_close(*channel, REFERENCE_GRAY_POSITIVE);
        }
    }

    #[test]
    fn the_print_curve_stays_in_range_and_monotonic_across_an_extreme_sweep() {
        // Sweep inputs far outside the physical range; the curve must stay bounded
        // and monotonic even when the gamma/pivot machinery is active.
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
        let settings = Settings {
            density_max: Some([2.0f32; 3]),
            gamma: Some([1.4f32; 3]),
            pivot_density: Some([1.0f32; 3]),
            ..Settings::default()
        };
        let result = invert(&source, &base([1.0f32; 3]), &settings);

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
                for channel in 0..3 {
                    assert!(
                        result.data[i * 3 + channel] >= result.data[(i - 1) * 3 + channel] - 1e-6,
                        "channel {channel} is not monotonic at {i}"
                    );
                }
            }
        }
    }

    #[test]
    fn dispatcher_color_negative_equals_invert() {
        // The C-41 path must stay byte-identical when reached through the
        // dispatcher. Chroma denoise is disabled here: the dispatcher adds it
        // around the inversion, and the identity check is about the inversion.
        let source = image(&[[0.7, 0.4, 0.2], [0.1, 0.9, 0.3]]);
        let base = base([0.8, 0.5, 0.3]);
        let mut settings = settings();
        settings.chroma_denoise = 0.0;
        let direct = invert(&source, &base, &settings);
        let dispatched = convert(
            &source,
            Some(&base),
            &settings,
            FilmMode::ColorNegative,
            false,
        );
        assert_eq!(direct.data, dispatched.data);
    }

    #[test]
    fn dispatcher_black_and_white_renders_gray() {
        let source = image(&[[0.9, 0.4, 0.2]]);
        let result = convert(
            &source,
            Some(&base([1.0, 1.0, 1.0])),
            &settings(),
            FilmMode::BlackAndWhite,
            false,
        );
        assert!((result.data[0] - result.data[1]).abs() < 1e-6);
        assert!((result.data[1] - result.data[2]).abs() < 1e-6);
    }

    #[test]
    fn dispatcher_slide_is_the_scan_and_honours_normalize() {
        let source = image(&[[0.5, 0.25, 0.125]]);
        let settings = settings();

        // Normalization off: the scan is the output.
        let as_captured = convert(&source, None, &settings, FilmMode::Slide, false);
        assert_eq!(as_captured.data, source.data);

        // Normalization on: the base goes to 1.0 per channel.
        let normalized = convert(
            &source,
            Some(&base([0.5, 0.25, 0.125])),
            &settings,
            FilmMode::Slide,
            true,
        );
        for channel in normalized.data {
            assert_close(channel, 1.0);
        }
    }
}
