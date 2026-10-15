//! Decodes a real scan when one is available.
//!
//! Real scans are private and third-party data, and this repository is public,
//! so no RAW fixture is committed. Point `GRAINBASE_FIXTURE_ARW` at a RAW file
//! to run this test; without it the test reports that it was skipped, so CI
//! stays green on a machine with no fixtures.

use grainbase::decode;

#[test]
fn decodes_a_real_scan_into_linear_rgb() {
    let Ok(path) = std::env::var("GRAINBASE_FIXTURE_ARW") else {
        eprintln!("skipped: set GRAINBASE_FIXTURE_ARW to a camera RAW file to run this test");
        return;
    };

    let scan = decode(&path).unwrap_or_else(|err| panic!("{path} should decode: {err}"));

    assert!(scan.linear.width > 0, "width should be non-zero");
    assert!(scan.linear.height > 0, "height should be non-zero");
    assert_eq!(
        scan.linear.data.len(),
        scan.linear.width as usize * scan.linear.height as usize * 3,
        "sample count should match the dimensions"
    );
    assert!(
        !scan.meta.model.is_empty(),
        "the camera model should be read from the file"
    );

    let min = scan
        .linear
        .data
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let max = scan
        .linear
        .data
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    // Linear light, unrendered: no gamma has been applied, so nothing dips
    // below zero, and a real negative is neither all-black nor constant.
    assert!(
        min >= 0.0,
        "linear samples should not be negative, got {min}"
    );
    assert!(max > 0.0, "the decoded frame is entirely black");
    assert!(
        max <= 1.0,
        "rescaled samples should stay in range, got {max}"
    );

    eprintln!(
        "decoded {}x{} from {}{}",
        scan.linear.width, scan.linear.height, scan.meta.make, scan.meta.model
    );
}
