//! Problem formulation: convert domain parameters into `min ‖Au − b‖²` form.
//!
//! Two formulations:
//!
//! - [`wls`](crate::cls::setup::wls) — Weighted least-squares with actuator-preference regularisation.
//!   Coefficient matrix is `(NV + NU) × NU`.
//! - [`ls`](crate::cls::setup::ls) — Plain (unregularised) least-squares.
//!   Coefficient matrix is `NV × NU`.

/// Unregularised least-squares formulation.
pub mod ls;
/// Weighted least-squares formulation (regularised).
pub mod wls;
