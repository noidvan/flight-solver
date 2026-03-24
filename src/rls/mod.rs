//! Recursive Least Squares (RLS) solvers for online parameter estimation.
//!
//! Two solver variants are provided:
//!
//! | Variant | Internal state | Numerical properties |
//! |---------|---------------|---------------------|
//! | [`standard`] | Covariance `P` | Requires numerical guards; matches indiflight C reference |
//! | [`inverse_qr`] | Info-matrix factor `L` | Inherently well-conditioned; no guards needed |
//!
//! Both support exponential forgetting (`λ`) and parallel multi-output mode
//! with shared regressors—the configuration used for flight controller system
//! identification (learning G1/G2 effectiveness matrices at 8 kHz).
//!
//! # Quick start
//!
//! ```no_run
//! use flight_solver::rls::{RlsParallel, CovarianceGuards};
//!
//! // 4 regressors, 3 parallel outputs (e.g. motor → roll/pitch/yaw)
//! let mut rls = RlsParallel::<4, 3>::new(1e2, 0.995, CovarianceGuards::default());
//!
//! // Each control loop iteration:
//! let a = nalgebra::SVector::<f32, 4>::new(0.1, -0.2, 0.3, 0.05);
//! let y = nalgebra::SVector::<f32, 3>::new(0.5, -0.3, 0.1);
//! let _stats = rls.update(&a, &y);
//!
//! let params = rls.params(); // 4×3 estimated parameter matrix
//! ```

pub mod inverse_qr;
pub mod standard;
pub mod types;

pub use self::types::{CovarianceGuards, ExitCode, UpdateStats};
pub use inverse_qr::InverseQrRls;
pub use standard::{Rls, RlsParallel};
