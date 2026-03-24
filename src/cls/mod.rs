//! Constrained least-squares (CLS) active-set solver and problem formulations.
//!
//! # Solver
//!
//! [`solve`] and [`solve_cls`] solve `min ‖Au − b‖²` subject to `umin ≤ u ≤ umax`
//! using an active-set method with incremental QR updates via Givens rotations.
//!
//! # Problem formulations
//!
//! | Module | Formulation | `A` size |
//! |--------|-------------|----------|
//! | [`setup::wls`] | Weighted LS with regularisation | `(NV+NU) × NU` |
//! | [`setup::ls`] | Plain LS (no regularisation) | `NV × NU` |

/// Low-level linear algebra: Householder QR, back-substitution, constraint checking.
pub mod linalg;
/// Problem formulations (WLS, unreg LS).
pub mod setup;
/// Active-set constrained least-squares solver.
pub mod solver;
/// Core types, constants, and nalgebra type aliases.
pub mod types;

pub use solver::{solve, solve_cls};
pub use types::{ExitCode, Mat, SolverStats, VecN};
