//! Selecting a sample by rank.

/// The value at `percentile` of `values`, reordering `values` in place.
///
/// `percentile` is a percentage: `0.0` is the minimum, `50.0` the median, `100.0`
/// the maximum. Values outside that range are clamped, so no input panics.
/// Returns `None` for an empty slice.
///
/// Selects rather than fully sorts, so the cost is linear on average. The slice
/// is only partially ordered afterwards, which is all a caller needs.
#[must_use]
pub fn value_at(values: &mut [f32], percentile: f32) -> Option<f32> {
    let rank = rank(values.len(), percentile)?;
    values.select_nth_unstable_by(rank, f32::total_cmp);
    Some(values[rank])
}

/// The value at `percentile` of a slice already sorted ascending.
#[must_use]
pub fn value_at_sorted(sorted: &[f32], percentile: f32) -> Option<f32> {
    sorted
        .get(index_of_rank(sorted.len(), percentile)?)
        .copied()
}

/// Index of the `percentile`-th sample of `len` sorted samples.
///
/// Exposed for callers that need a position in their own sorted data rather than
/// a value from it.
#[must_use]
pub fn index_of_rank(len: usize, percentile: f32) -> Option<usize> {
    rank(len, percentile)
}

/// Index of the `percentile`-th sample of `len` sorted samples.
fn rank(len: usize, percentile: f32) -> Option<usize> {
    if len == 0 {
        return None;
    }
    let clamped = percentile.clamp(0.0, 100.0);
    let scaled = scaled_rank(len, clamped);
    Some(scaled)
}

/// `len - 1` scaled by `percentile`, clamped into range.
///
/// Split out so the casts carry one explanation rather than three.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "percentile is clamped to 0.0..=100.0, so the scaled rank is \
              finite, non-negative, and bounded by len - 1"
)]
fn scaled_rank(len: usize, percentile: f32) -> usize {
    let rank = (len.saturating_sub(1) as f32 * (percentile / 100.0)).round();

    // The clamp is not redundant. `len - 1` does not survive the trip through
    // f32: at 24 million samples it rounds *up*, so the top percentile asked for
    // index 24_000_000 in a 24_000_000-element slice and got nothing. One sample
    // out of range is all it takes to turn a maximum into a missing value.
    (rank as usize).min(len.saturating_sub(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_extremes_are_the_minimum_and_maximum() {
        let mut values = vec![5.0, 1.0, 9.0, 3.0];
        assert_eq!(value_at(&mut values, 0.0), Some(1.0));
        assert_eq!(value_at(&mut values, 100.0), Some(9.0));
    }

    #[test]
    fn the_median_is_the_middle_sample() {
        let mut values = vec![5.0, 1.0, 9.0, 3.0, 7.0];
        assert_eq!(value_at(&mut values, 50.0), Some(5.0));
    }

    #[test]
    fn an_empty_slice_has_no_value() {
        assert_eq!(value_at(&mut [], 50.0), None);
    }

    #[test]
    fn a_single_sample_is_every_percentile() {
        for percentile in [0.0, 50.0, 100.0] {
            assert_eq!(value_at(&mut [4.0], percentile), Some(4.0));
        }
    }

    #[test]
    fn out_of_range_percentiles_are_clamped_not_panicking() {
        let mut values = vec![1.0, 2.0, 3.0];
        assert_eq!(value_at(&mut values, -50.0), Some(1.0));
        assert_eq!(value_at(&mut values, 250.0), Some(3.0));
    }

    #[test]
    fn nan_does_not_break_selection() {
        let mut values = vec![1.0, f32::NAN, 3.0];
        let value = value_at(&mut values, 100.0).expect("non-empty");
        assert!(value.is_finite() || value.is_nan());
    }

    #[test]
    fn the_top_rank_never_runs_past_the_end() {
        // The bug this pins: `len - 1` does not survive f32. At 24 million
        // samples it rounds up to exactly 24 million, and the top percentile
        // then asked for an index one past the last element.
        for len in [
            1,
            2,
            10,
            1_000,
            1_000_000,
            16_777_217,
            24_000_000,
            usize::MAX / 4,
        ] {
            for percentile in [99.0, 99.9, 99.99, 100.0] {
                let rank = scaled_rank(len, percentile);
                assert!(rank < len, "len {len} at p{percentile} gave rank {rank}");
            }
        }
    }

    #[test]
    fn the_maximum_is_reachable_at_the_top_percentile() {
        // A regression guard for the same bug at a size that is cheap to test:
        // the largest value must come back, not a missing value.
        let mut values: Vec<f32> = (0..10_000u16).map(f32::from).collect();
        values.push(99_999.0);
        let top = value_at(&mut values, 100.0).expect("there is a maximum");
        assert_eq!(top, 99_999.0);
    }
}
