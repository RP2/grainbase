//! Fitted calibration profiles.
//!
//! A profile is the measured answer to "what does this film, shot on this rig,
//! need?" — the per-channel density ceilings and the dye crosstalk correction.
//! It is data, not code: a fitted profile can be shared, versioned, and edited
//! by hand, and it is what a calibration run produces and a conversion run
//! consumes.
//!
//! The format is deliberately plain text, one `key = value` per line, with `#`
//! comments. It is six fields; a parser for that is smaller than the dependency
//! it would take to avoid writing one. Unknown keys are an error rather than a
//! shrug, because a typo in a hand-edited profile should not silently fall back
//! to defaults and produce plausible-looking wrong output.

use std::fmt;

use crate::pipeline::{IDENTITY, Settings};

/// Format version, so an old profile can be rejected rather than misread.
pub const VERSION: u32 = 1;

/// How a profile's numbers were arrived at.
///
/// Mandatory provenance (ADR 0006). A hand-tuned profile is not wrong, but it
/// must say so: a consumer is entitled to weigh a number against how it was
/// produced, and a library of unlabelled numbers is worse than no library.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Method {
    /// Built-in neutral defaults. Nothing was fitted.
    #[default]
    Neutral,
    /// Fitted from paired scans on a known rig.
    Fitted,
    /// Derived from published spectral data rather than measured.
    Derived,
    /// Adjusted by hand until it looked right.
    Tuned,
}

impl fmt::Display for Method {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Neutral => "neutral",
            Self::Fitted => "fitted",
            Self::Derived => "derived",
            Self::Tuned => "tuned",
        })
    }
}

impl Method {
    /// Parses a method keyword, rejecting unknown spellings.
    fn parse(line: usize, key: &str, value: &str) -> Result<Self, ParseError> {
        match value {
            "neutral" => Ok(Self::Neutral),
            "fitted" => Ok(Self::Fitted),
            "derived" => Ok(Self::Derived),
            "tuned" => Ok(Self::Tuned),
            other => Err(ParseError::BadValue {
                line,
                key: key.to_owned(),
                reason: format!(
                    "`{other}` is not a method (expected neutral, fitted, derived or tuned)"
                ),
            }),
        }
    }
}

/// A fitted calibration profile.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Profile {
    /// Film stock the profile was fitted for, for the reader's benefit.
    pub film: String,
    /// Camera or rig the profile was fitted on. Crosstalk is a property of the
    /// whole scanning setup, not the film, so the same stock can need different
    /// values on a different rig.
    pub camera: String,
    /// How these numbers were arrived at.
    pub method: Method,
    /// Frames the fit used. A fit from three frames and one from sixty are not
    /// equally trustworthy, and should not look identical.
    pub frames: u32,
    /// The fitted conversion settings.
    ///
    /// Derived `Default` would leave the bounds `None` — meaning *measured* —
    /// which cannot serialise: a profile text is explicit by definition.
    /// Overridden here so the neutral default round-trips.
    pub settings: Settings,
}

impl Profile {
    /// The neutral built-in profile: no fit, no correction, explicit
    /// defaults.
    #[must_use]
    pub fn neutral() -> Self {
        Self {
            film: String::new(),
            camera: String::new(),
            method: Method::Neutral,
            frames: 0,
            settings: Settings {
                density_max: Some([crate::pipeline::DEFAULT_DENSITY_MAX; 3]),
                density_floor: None,
                crosstalk: crate::pipeline::IDENTITY,
            },
        }
    }
}

/// Why a profile could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// A line was neither blank, a comment, nor `key = value`.
    Malformed {
        /// 1-based line number.
        line: usize,
        /// The line as written.
        text: String,
    },
    /// A key the format does not define. Almost always a typo.
    UnknownKey {
        /// 1-based line number.
        line: usize,
        /// The unrecognised key.
        key: String,
    },
    /// A known key with a value that could not be interpreted.
    BadValue {
        /// 1-based line number.
        line: usize,
        /// The key whose value was rejected.
        key: String,
        /// What was wrong with it.
        reason: String,
    },
    /// A version this build does not understand.
    UnsupportedVersion {
        /// The version found in the file.
        found: u32,
    },
    /// A required key was absent.
    MissingKey {
        /// The key that was not found.
        key: &'static str,
    },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Malformed { line, text } => {
                write!(f, "line {line} is not `key = value`: {text}")
            }
            Self::UnknownKey { line, key } => {
                write!(f, "line {line}: unknown key `{key}` (typo?)")
            }
            Self::BadValue { line, key, reason } => {
                write!(f, "line {line}: bad value for `{key}`: {reason}")
            }
            Self::UnsupportedVersion { found } => write!(
                f,
                "profile version {found} is not supported, this build reads version {VERSION}"
            ),
            Self::MissingKey { key } => write!(f, "required key `{key}` is missing"),
        }
    }
}

impl std::error::Error for ParseError {}

impl fmt::Display for Profile {
    /// Renders the profile as text.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Some(max) = self.settings.density_max else {
            // Unreachable in practice: every constructor of `Profile` carries
            // explicit bounds, so a `None` here is a caller bug, and the
            // failure to serialise is the honest report.
            return Err(fmt::Error);
        };
        let matrix = self.settings.crosstalk;

        writeln!(f, "# grainbase calibration profile")?;
        writeln!(
            f,
            "# Fitted from paired scans. Edit by hand if you know why."
        )?;
        writeln!(f)?;
        writeln!(f, "version = {VERSION}")?;
        writeln!(f, "film = {}", quote(&self.film))?;
        writeln!(f, "camera = {}", quote(&self.camera))?;
        writeln!(f, "method = {}", self.method)?;
        writeln!(f, "frames = {}", self.frames)?;
        writeln!(f, "density_max = {:.6} {:.6} {:.6}", max[0], max[1], max[2])?;
        // Optional keys stay optional on both sides: an older profile parses
        // as well as a newer one.
        if let Some([red, green, blue]) = self.settings.density_floor {
            writeln!(f, "density_floor = {red:.6} {green:.6} {blue:.6}")?;
        }
        writeln!(f, "# Rows are output channels, columns are dye densities.")?;
        writeln!(
            f,
            "crosstalk = {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
            matrix[0][0],
            matrix[0][1],
            matrix[0][2],
            matrix[1][0],
            matrix[1][1],
            matrix[1][2],
            matrix[2][0],
            matrix[2][1],
            matrix[2][2],
        )?;
        Ok(())
    }
}

impl Profile {
    /// Reads a profile from text.
    ///
    /// # Errors
    ///
    /// Returns [`ParseError`] for a malformed line, an unknown key, an
    /// unsupported version, a missing required key, or a value that cannot be
    /// interpreted.
    pub fn parse(text: &str) -> Result<Self, ParseError> {
        let mut version = None;
        let mut film = None;
        let mut camera = None;
        let mut method = None;
        let mut frames = None;
        let mut density_max = None;
        let mut density_floor = None;
        let mut crosstalk = None;

        for (index, raw) in text.lines().enumerate() {
            let line = index + 1;
            let content = raw.trim();
            if content.is_empty() || content.starts_with('#') {
                continue;
            }

            let Some((key, value)) = content.split_once('=') else {
                return Err(ParseError::Malformed {
                    line,
                    text: raw.to_owned(),
                });
            };
            let (key, value) = (key.trim(), value.trim());

            match key {
                "version" => version = Some(parse_u32(line, key, value)?),
                "film" => film = Some(unquote(line, key, value)?),
                "camera" => camera = Some(unquote(line, key, value)?),
                "method" => method = Some(Method::parse(line, key, value)?),
                "frames" => frames = Some(parse_u32(line, key, value)?),
                "density_max" => density_max = Some(parse_floats::<3>(line, key, value)?),
                "density_floor" => density_floor = Some(parse_floats::<3>(line, key, value)?),
                "crosstalk" => {
                    let flat = parse_floats::<9>(line, key, value)?;
                    crosstalk = Some([
                        [flat[0], flat[1], flat[2]],
                        [flat[3], flat[4], flat[5]],
                        [flat[6], flat[7], flat[8]],
                    ]);
                }
                other => {
                    return Err(ParseError::UnknownKey {
                        line,
                        key: other.to_owned(),
                    });
                }
            }
        }

        let found = version.ok_or(ParseError::MissingKey { key: "version" })?;
        if found != VERSION {
            return Err(ParseError::UnsupportedVersion { found });
        }

        Ok(Self {
            film: film.ok_or(ParseError::MissingKey { key: "film" })?,
            camera: camera.ok_or(ParseError::MissingKey { key: "camera" })?,
            method: method.ok_or(ParseError::MissingKey { key: "method" })?,
            frames: frames.ok_or(ParseError::MissingKey { key: "frames" })?,
            settings: Settings {
                // A parsed profile's bounds are explicit by definition.
                density_max: Some(
                    density_max.ok_or(ParseError::MissingKey { key: "density_max" })?,
                ),
                density_floor,
                crosstalk: crosstalk.ok_or(ParseError::MissingKey { key: "crosstalk" })?,
            },
        })
    }
}

/// Wraps a string in quotes, escaping embedded quotes and backslashes.
fn quote(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

/// Undoes [`quote`].
fn unquote(line: usize, key: &str, value: &str) -> Result<String, ParseError> {
    let bad = |reason: &str| ParseError::BadValue {
        line,
        key: key.to_owned(),
        reason: reason.to_owned(),
    };

    let Some(inner) = value
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
    else {
        return Err(bad("expected a quoted string"));
    };

    let mut out = String::with_capacity(inner.len());
    let mut characters = inner.chars();
    while let Some(character) = characters.next() {
        if character != '\\' {
            out.push(character);
            continue;
        }
        match characters.next() {
            Some('\\') => out.push('\\'),
            Some('"') => out.push('"'),
            _ => return Err(bad("dangling escape")),
        }
    }
    Ok(out)
}

/// Parses a whole number.
fn parse_u32(line: usize, key: &str, value: &str) -> Result<u32, ParseError> {
    value.parse().map_err(|_| ParseError::BadValue {
        line,
        key: key.to_owned(),
        reason: format!("`{value}` is not a whole number"),
    })
}

/// Parses `N` whitespace-separated floats.
fn parse_floats<const N: usize>(
    line: usize,
    key: &str,
    value: &str,
) -> Result<[f32; N], ParseError> {
    let bad = |reason: String| ParseError::BadValue {
        line,
        key: key.to_owned(),
        reason,
    };

    let parts: Vec<&str> = value.split_whitespace().collect();
    if parts.len() != N {
        return Err(bad(format!("expected {N} numbers, found {}", parts.len())));
    }

    let mut out = [0.0f32; N];
    for (slot, part) in out.iter_mut().zip(parts) {
        let parsed: f32 = part
            .parse()
            .map_err(|_| bad(format!("`{part}` is not a number")))?;
        if !parsed.is_finite() {
            return Err(bad(format!("`{part}` is not finite")));
        }
        *slot = parsed;
    }
    Ok(out)
}

/// The identity crosstalk matrix, for tests and callers that want it by name.
pub const NO_CROSSTALK: [[f32; 3]; 3] = IDENTITY;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::DEFAULT_DENSITY_MAX;

    fn sample() -> Profile {
        Profile {
            film: "Portra 400".to_owned(),
            camera: "Sony ILCE-6000".to_owned(),
            method: Method::Fitted,
            frames: 12,
            settings: Settings {
                density_max: Some([1.85, 1.92, 1.74]),
                density_floor: None,
                crosstalk: [[1.0, 0.05, 0.02], [0.03, 1.0, 0.04], [0.01, 0.02, 1.0]],
            },
        }
    }

    #[test]
    fn a_profile_survives_a_round_trip() {
        let original = sample();
        let text = original.to_string();
        let parsed = Profile::parse(&text).expect("should parse");
        assert_eq!(parsed, original);
    }

    #[test]
    fn the_default_profile_is_valid_and_neutral() {
        let text = Profile::neutral().to_string();
        let parsed = Profile::parse(&text).expect("should parse");

        assert_eq!(parsed.settings.density_max, Some([DEFAULT_DENSITY_MAX; 3]));
        assert_eq!(
            parsed.settings.crosstalk, IDENTITY,
            "default means no correction"
        );
    }

    #[test]
    fn comments_and_blank_lines_are_ignored() {
        let text = "# a comment\n\nversion = 1\n\n# another\nfilm = \"x\"\ncamera = \"y\"\n\
                    method = fitted\nframes = 3\n\
                    density_max = 1 2 3\ncrosstalk = 1 0 0 0 1 0 0 0 1\n";
        assert!(Profile::parse(text).is_ok());
    }

    #[test]
    fn a_typo_in_a_key_is_an_error_not_a_silent_default() {
        // The failure this guards: a mistyped key silently falls back to the
        // default and produces plausible-looking wrong output.
        let text = "version = 1\nfilm = \"x\"\ncamera = \"y\"\ndensity_mx = 1 2 3\n\
                    crosstalk = 1 0 0 0 1 0 0 0 1\n";
        assert_eq!(
            Profile::parse(text),
            Err(ParseError::UnknownKey {
                line: 4,
                key: "density_mx".to_owned()
            })
        );
    }

    #[test]
    fn a_wrong_number_count_is_rejected() {
        let text = "version = 1\nfilm = \"x\"\ncamera = \"y\"\ndensity_max = 1 2\n\
                    crosstalk = 1 0 0 0 1 0 0 0 1\n";
        assert!(matches!(
            Profile::parse(text),
            Err(ParseError::BadValue { line: 4, .. })
        ));
    }

    #[test]
    fn a_missing_required_key_is_reported() {
        let text = "version = 1\nfilm = \"x\"\ncamera = \"y\"\nmethod = fitted\nframes = 3\n\
                    density_max = 1 2 3\n";
        assert_eq!(
            Profile::parse(text),
            Err(ParseError::MissingKey { key: "crosstalk" })
        );
    }

    #[test]
    fn provenance_is_required_and_not_defaulted() {
        // ADR 0006: a profile must say how its numbers were produced. Silently
        // assuming a method would let a guess pass as a measurement.
        for (omit, key) in [("method", "method"), ("frames", "frames")] {
            let mut lines = vec![
                "version = 1".to_owned(),
                "film = \"x\"".to_owned(),
                "camera = \"y\"".to_owned(),
                "method = fitted".to_owned(),
                "frames = 3".to_owned(),
                "density_max = 1 2 3".to_owned(),
                "crosstalk = 1 0 0 0 1 0 0 0 1".to_owned(),
            ];
            lines.retain(|line| !line.starts_with(omit));

            assert_eq!(
                Profile::parse(&lines.join("\n")),
                Err(ParseError::MissingKey { key }),
                "omitting {omit} should be reported, not defaulted"
            );
        }
    }

    #[test]
    fn an_unknown_method_is_rejected() {
        let text = "version = 1\nfilm = \"x\"\ncamera = \"y\"\nmethod = vibes\nframes = 3\n\
                    density_max = 1 2 3\ncrosstalk = 1 0 0 0 1 0 0 0 1\n";
        assert!(matches!(
            Profile::parse(text),
            Err(ParseError::BadValue { line: 4, .. })
        ));
    }

    #[test]
    fn the_built_in_default_declares_itself_unfitted() {
        assert_eq!(Profile::default().method, Method::Neutral);
        assert_eq!(Profile::default().frames, 0);
    }

    #[test]
    fn provenance_survives_a_round_trip() {
        let parsed = Profile::parse(&sample().to_string()).expect("should parse");
        assert_eq!(parsed.method, Method::Fitted);
        assert_eq!(parsed.frames, 12);
    }

    #[test]
    fn a_future_version_is_refused_rather_than_misread() {
        let text = "version = 99\nfilm = \"x\"\ncamera = \"y\"\ndensity_max = 1 2 3\n\
                    crosstalk = 1 0 0 0 1 0 0 0 1\n";
        assert_eq!(
            Profile::parse(text),
            Err(ParseError::UnsupportedVersion { found: 99 })
        );
    }

    #[test]
    fn a_non_finite_value_is_refused() {
        let text = "version = 1\nfilm = \"x\"\ncamera = \"y\"\ndensity_max = 1 inf 3\n\
                    crosstalk = 1 0 0 0 1 0 0 0 1\n";
        assert!(matches!(
            Profile::parse(text),
            Err(ParseError::BadValue { line: 4, .. })
        ));
    }

    #[test]
    fn a_line_that_is_not_key_value_is_rejected_with_its_number() {
        let text = "version = 1\ngarbage\n";
        assert_eq!(
            Profile::parse(text),
            Err(ParseError::Malformed {
                line: 2,
                text: "garbage".to_owned()
            })
        );
    }

    #[test]
    fn quoted_strings_round_trip_with_awkward_characters() {
        let profile = Profile {
            film: "Fuji \"Neopan\" 400".to_owned(),
            camera: "rig\\1".to_owned(),
            ..Profile::neutral()
        };
        let parsed = Profile::parse(&profile.to_string()).expect("should parse");
        assert_eq!(parsed.film, profile.film);
        assert_eq!(parsed.camera, profile.camera);
    }
}
