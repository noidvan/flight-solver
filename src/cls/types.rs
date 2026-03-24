//! Types for the constrained least-squares solver.

use nalgebra::{Const, OMatrix, OVector};

/// Constraint feasibility tolerance for f32 (matches C `AS_CONSTR_TOL` with `AS_SINGLE_FLOAT`).
pub const CONSTR_TOL: f32 = 1e-4;

/// Solver exit status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ExitCode {
    /// Optimal feasible solution found.
    Success = 0,
    /// Maximum iteration count reached without convergence.
    IterLimit = 1,
    // CostBelowTol (2) and CostPlateau (3) are not implemented — they only
    // applied to the Cholesky solver's optional cost-truncation mode, which
    // this crate does not port.
    /// NaN detected in QR solve (indicates singular or near-singular subproblem).
    NanFoundQ = 4,
    /// NaN detected in solution update.
    NanFoundUs = 5,
}

/// Statistics returned by the solver alongside the solution.
#[derive(Debug, Clone, Copy)]
pub struct SolverStats {
    /// How the solver terminated.
    pub exit_code: ExitCode,
    /// Number of active-set iterations performed.
    pub iterations: usize,
    /// Number of free (unconstrained) actuators at the solution.
    pub n_free: usize,
}

/// Convenience alias for an `NC × NU` coefficient matrix (column-major, f32).
pub type Mat<const NC: usize, const NU: usize> = OMatrix<f32, Const<NC>, Const<NU>>;

/// Convenience alias for an `N`-element column vector (f32).
pub type VecN<const N: usize> = OVector<f32, Const<N>>;
