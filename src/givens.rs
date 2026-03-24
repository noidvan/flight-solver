//! Givens rotation primitives for numerically stable QR updates.
//!
//! Shared across all solvers in this crate: used by the inverse QR-RLS for
//! information-matrix updates, and by the WLS active-set solver for
//! incremental QR maintenance.
//!
//! The implementation uses [`libm::hypotf`] for the 2-norm computation,
//! which avoids overflow and underflow in the intermediate `a² + b²` term.

use nalgebra::{Const, DimName, OMatrix};

/// Compute Givens rotation parameters `(c, s)` that zero the second element.
///
/// Returns `(c, s)` such that the rotation matrix
///
/// ```text
/// G = [ c  -s ]
///     [ s   c ]
/// ```
///
/// applied to the column vector `[a, b]ᵀ` yields `[r, 0]ᵀ`
/// where `r = √(a² + b²)`.
///
/// When both `a` and `b` are zero, returns `(1, 0)` (identity rotation).
#[inline]
pub fn givens(a: f32, b: f32) -> (f32, f32) {
    let h = libm::hypotf(a, b);
    if h == 0.0 {
        return (1.0, 0.0);
    }
    let inv_h = 1.0 / h;
    (a * inv_h, -(b * inv_h))
}

/// Apply a Givens rotation from the left to two rows of a matrix.
///
/// For each column `k` in `0..n_cols`:
///
/// ```text
/// r[row1, k] ← c · r[row1, k] − s · r[row2, k]
/// r[row2, k] ← s · r[row1, k] + c · r[row2, k]
/// ```
///
/// This is equivalent to left-multiplying by the Givens matrix `G(row1, row2, c, s)`.
#[inline]
pub fn givens_left_apply<const R: usize, const C: usize>(
    r: &mut OMatrix<f32, Const<R>, Const<C>>,
    c: f32,
    s: f32,
    row1: usize,
    row2: usize,
    n_cols: usize,
) where
    Const<R>: DimName,
    Const<C>: DimName,
{
    for col in 0..n_cols {
        let r1 = r[(row1, col)];
        let r2 = r[(row2, col)];
        r[(row1, col)] = c * r1 - s * r2;
        r[(row2, col)] = s * r1 + c * r2;
    }
}

/// Apply a Givens rotation transpose from the right to two columns of a matrix.
///
/// For each row `i` in `0..n_rows`:
///
/// ```text
/// q[i, col1] ← c · q[i, col1] − s · q[i, col2]
/// q[i, col2] ← s · q[i, col1] + c · q[i, col2]
/// ```
///
/// This is equivalent to right-multiplying by `Gᵀ(col1, col2, c, s)`.
#[inline]
pub fn givens_right_apply_t<const R: usize, const C: usize>(
    q: &mut OMatrix<f32, Const<R>, Const<C>>,
    c: f32,
    s: f32,
    col1: usize,
    col2: usize,
    n_rows: usize,
) where
    Const<R>: DimName,
    Const<C>: DimName,
{
    for i in 0..n_rows {
        let q1 = q[(i, col1)];
        let q2 = q[(i, col2)];
        q[(i, col1)] = c * q1 - s * q2;
        q[(i, col2)] = s * q1 + c * q2;
    }
}
