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
| Trumpy & Anderson, "From Negatives to Positives: Modeling the Photochemical Printing Process", IS&T Archiving Conference 2025 (DOI 10.2352/issn.2168-3204.2025.22.1.33) | Published paper; cited as fact and model reference | The photochemical print model's existence and its form | Our own reading and derivation |
| Cineon digital film documentation (printing-density encoding: PD = 0.002·CV) | Industry-published specification | The printing-density scale convention | Our own implementation |
| NegPy (marcinz606) | GPL-3.0 | Observed as technique only: the density-domain preprocessing shape (base neutralisation → density → per-channel handling) and its crosstalk-gallery provenance policy | No code, no constants, no numbers taken. Anything similar was written from first principles |
| RapidRAW (CyberTimon) | AGPL-3.0 | Observed as technique only: percentile auto-bounds exist in the wild and how they behave on thin scans | Deliberately **not** implemented: content-derived white points compensate for exposure, which grainbase rejects. Nothing taken |
| darktable, `negadoctor` | GPL | Observed as technique only: per-channel density ceilings are a documented print-mapping idea; scanner-default concepts | No code or constants taken. Numbers come from Kodak documents, not from negadoctor sources |
| Filmomat, FilmLab, Negative Lab Pro | Proprietary | Behavior-level observations only (what their hosted tiers claim) | Nothing taken |
| VueScan (Hamrick) | Proprietary | Behavior-level observation only: the documented order of scan-to-positive processing (whole-image base estimate → per-line correction → linear black/white points → last-step color correction and gamma) | Nothing taken. Numbers and code are ours; the pipeline's sequencing follows the documented flow |
| Negative Lab Pro (marcinz606/nate) | Proprietary | Behavior-level observations only: border-sampling once per roll and syncing it; black/white clipping against the negative; two-rectangle discipline (output crop vs analysis region) | Nothing taken |
| SilverFast (Hamrick/LaserSoft) NegaFix, IFF+ engine | Proprietary | Observed as technique only: frame search is content-independent and tolerant of non-nominal film width; the analysis/crop boundary must contain nothing but exposed image | Nothing taken |
| Fuji Film, "Film scanning method and apparatus" (US5579132A) | Published patent | The mask-neutralization→density→crosstalk→exposure sequence as the established reading order for color negatives; leader-reference calibration precedent | Our own reading; our own arithmetic, distinct from theirs |
| Kodak electronic color analyzers (LAD/LID, H-61 suite) | Kodak publication; used as published factual data | The gray-track discipline: a standard reference patches the analyzer; per-frame gains aim at equal readings | Our own reading |

## Maintenance

When a constant gains an external fact behind it, two places update in the same
commit: this table, and the constant's doc comment (the `measured:`/cite: style used
throughout). When a technique is rejected rather than implemented — like the percentile
bounds above — it is recorded here too: knowing what we chose against is part of how
choices are audited later.
