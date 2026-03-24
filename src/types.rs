//! Shared types and constants used across all solvers.

use nalgebra::{Const, OMatrix, OVector};

/// Convenience alias for an `N`-element column vector (f32).
pub type VecN<const N: usize> = OVector<f32, Const<N>>;

/// Convenience alias for an `R × C` matrix (column-major, f32).
pub type Mat<const R: usize, const C: usize> = OMatrix<f32, Const<R>, Const<C>>;
