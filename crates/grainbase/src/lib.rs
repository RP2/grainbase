//! The grainbase engine.
//!
//! Converts camera-scanned film negatives into finished positives.
//!
//! The engine performs no I/O beyond reading the scan it is handed, and carries
//! no front-end assumptions: the CLI is a thin adapter over this library, and
//! tests cross the same seams its callers do.

pub mod decode;
pub mod driver;
pub mod encode;
pub mod image;
pub mod pipeline;
pub mod profile;

pub use decode::decode;
pub use decode::{Scan, ScanMeta};
pub use image::LinearImage;
