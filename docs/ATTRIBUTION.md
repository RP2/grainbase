# Attribution

How grainbase uses external knowledge. The engine is original code and original
expression; where it is informed by published facts or by techniques observed in other
projects, this file records that.

The rule: **facts may be used with citation; nobody else's code, constants-of-convenience,
or undocumented numbers are used.** Every external constant carries its citation in the
doc comment where it lives; the summary lives here.

## Sources

| Source | License | What we took | How |
| --- | --- | --- | --- |
| Kodak, *Film Users' Guide and Digital Recorder Calibration and Aims* (H-387), including the Digital LAD tables | Kodak publication; used as published factual data | Status-M → printing-density relationships; reference black/white/grey printing density aims; the general fact of the Digital LAD system | Our own reading and derivation; values are cited at their point of use in `pipeline` |
| Kodak spectral dye-density publications for C-41 films (E-4050 and datasheet characteristic curves) | Kodak publication; used as published factual data | The existence and shape of the dye layers' density response; any per-layer D-max values the datasheets state | Our own reading and derivation |
| Gretag-Macbeth (later X-Rite/Calibrite) published 8-bit sRGB `ColorChecker` patch values; the sRGB transfer function (IEC 61966-2-1) | Manufacturer's published factual data; industry standard | Chart patch print targets + linearization for the calibration fitter (`pipeline/chart`) | Our own constants at the point of use, cross-checked against the Pascale survey and Poynton's `ColorChecker` page |
| Kodak Gray Scale Q-13/Q-14 (Kodak Publication No. Q-14) | Kodak publication; used as published factual data | The 20-step nominal reflection densities of the neutral wedge | Our own constants at the point of use in `pipeline/wedge` |
| Trumpy & Anderson, "From Negatives to Positives: Modeling the Photochemical Printing Process", IS&T Archiving Conference 2025 (DOI 10.2352/issn.2168-3204.2025.22.1.33) | Published paper; cited as fact and model reference | The photochemical print model's existence and its form; the H&D characteristic-curve shape (a straight middle between a soft toe and a soft shoulder) | Our own reading and derivation. A per-frame print stage was prototyped against this shape and **dropped** — see the rejected-technique note below |
| ISO 6846 / ISO-R photographic paper grade (the `R = 100·range/gamma` relation) | Published standard | The straight-line gamma as the negative's density range over the paper's exposure range | Recorded as the relation a print stage would use; nothing ships from it yet |
| Cineon digital film documentation (printing-density encoding: PD = 0.002·CV) | Industry-published specification | The printing-density scale convention | Our own implementation |
| NegPy (marcinz606) | GPL-3.0 | Observed as technique only: the density-domain preprocessing shape (base neutralisation → density → per-channel handling), the print-stage shape (an **unclamped** per-channel stretch to per-frame bounds measured on a block grid, then an H&D characteristic curve with soft knees, with cast removal as a per-channel re-fit against measured neutral references), and its crosstalk-gallery provenance policy; positive film is opened as captured rather than inverted | No code, no constants, no numbers taken. Read as a reference for method only; a per-frame print stage built along those lines was measured worse than the fixed cone on the maintainer's four fixtures and **dropped**. Nothing NegPy-derived ships |
| Kodak, *Printing from Color Negatives* (H-61) and *Set-Up an Electronic Color Analyzer and Printing Control* (H-61) | Kodak publication; used as published factual data | The printer-point method: a frame is metered from its **integral** (large-area) density against a reference, and the difference is the printer point. The analyzer is a visual matching instrument; the numeric anchor is the LAD patch | Our own reading; our own arithmetic |
| Kodak, *LAD for KODAK Color Negative Film* (H-61a) | Kodak publication; used as published factual data | The published LAD aims for a **camera** color negative: 0.80 R / 1.20 G / 1.60 B Status-M above base for a correctly exposed 16% `Munsell` N4.5 patch. Recorded as `pipeline::meter::LAD_CAMERA_NEGATIVE` | Cited as published. **Measured and not applied to scan density**: a camera is not a densitometer, and the scan's base-relative density sits on an unknown rig scale relative to Status-M. Kept as the reference a fitted profile works against |
| US 3,653,759 and US 4,159,174 (photofinishing exposure/color control) | Published patents | The *subject-failure* discipline: correction strength is coupled to the detected channel imbalance — full correction for a balanced negative, reduced or zero for a scene that is genuinely one color. The response is gated, not the measurement area | Our own reading; our own confidence weighting in `pipeline/meter.rs` |
| RapidRAW (CyberTimon) | AGPL-3.0 | Observed as technique only: percentile auto-bounds exist in the wild and how they behave on thin scans | Deliberately **not** implemented: content-derived white points compensate for exposure, which grainbase rejects. Nothing taken |
| darktable, `negadoctor` | GPL | Observed as technique only: per-channel density ceilings are a documented print-mapping idea; scanner-default concepts | No code or constants taken. Numbers come from Kodak documents, not from negadoctor sources |
| Filmomat, FilmLab, Negative Lab Pro | Proprietary | Behavior-level observations only (what their hosted tiers claim) | Nothing taken |
| VueScan (Hamrick) | Proprietary | Behavior-level observation only: the documented order of scan-to-positive processing (whole-image base estimate → per-line correction → linear black/white points → last-step color correction and gamma); positive film is opened as captured, with optional normalization | Nothing taken. Numbers and code are ours; the pipeline's sequencing follows the documented flow |
| Negative Lab Pro (marcinz606/nate) | Proprietary | Behavior-level observations only: border-sampling once per roll and syncing it; black/white clipping against the negative; two-rectangle discipline (output crop vs analysis region) | Nothing taken |
| SilverFast (Hamrick/LaserSoft) NegaFix, IFF+ engine | Proprietary | Observed as technique only: frame search is content-independent and tolerant of non-nominal film width; the analysis/crop boundary must contain nothing but exposed image | Nothing taken |
| Fuji Film, "Film scanning method and apparatus" (US5579132A) | Published patent | The mask-neutralization→density→crosstalk→exposure sequence as the established reading order for color negatives; leader-reference calibration precedent | Our own reading; our own arithmetic, distinct from theirs |
| Kodak electronic color analyzers (LAD/LID, H-61 suite) | Kodak publication; used as published factual data | The gray-track discipline: a standard reference patches the analyzer; per-frame gains aim at equal readings | Our own reading |

## Rejected techniques

- **Per-frame print stage (unclamped stretch + H&D curve + per-frame cast).** Built
  and measured against the maintainer's four fixtures, with `fixtures/processed/`
  (NegPy's render) as the target. Every variant rendered *worse* than the fixed
  LAD-anchored cone: the frame's own dense end lands about half the cone's, so a
  full per-frame stretch lifts the whole frame, and the per-frame cast reference
  cannot tell a film cast from a scene that is genuinely one colour (on a dark
  frame the lowest-chroma blocks are the frame's own tinted shadows, so
  neutralising them washes the frame). Dropped; the fixed cone stays the generic
  default and the cast belongs to a fitted profile, where the maintainer's
  measured values live.
- **LAD aims applied to scan density.** The H-61a camera-negative aims
  (0.80/1.20/1.60 Status-M) do not transfer: a camera is not a densitometer, and a
  neutral scene integrates *flat* in base-relative scan density because the base
  division already removed the aims' per-channel shape. Measured worse as ceilings
  on three of four fixtures, and as a meter target it refused three of four
  frames. Kept only as the reference a fitted profile is anchored to.
- **Percentile auto-bounds** (RapidRAW-style): content-derived white points
  compensate for exposure, which grainbase rejects.

## Maintenance

When a constant gains an external fact behind it, two places update in the same
commit: this table, and the constant's doc comment (the `measured:`/cite: style used
throughout). When a technique is rejected rather than implemented — like the percentile
bounds above — it is recorded here too: knowing what we chose against is part of how
choices are audited later.
