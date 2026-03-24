//! Shared linear algebra utilities: constraint checking, Householder QR, and
//! triangular back-substitution.
//!
//! These operate on raw `[[f32; R]; C]` arrays (column-major: `arr[col][row]`)
//! because the naive solver's subproblem has a runtime column count that
//! nalgebra's static QR cannot express without `alloc`.

use super::types::CONSTR_TOL;

/// Check which elements of `x` violate bounds, with relative + absolute tolerance.
///
/// For each `i` in `0..n_check`, examines `x[perm[i]]` (or `x[i]` when `perm`
/// is `None`) against `xmin` / `xmax`. Returns the number of violations, and
/// writes `+1` (upper), `-1` (lower), or `0` (feasible) into `output`.
pub fn check_limits_tol<const N: usize>(
    n_check: usize,
    x: &[f32; N],
    xmin: &[f32; N],
    xmax: &[f32; N],
    output: &mut [i8; N],
    perm: Option<&[usize; N]>,
) -> usize {
    let tol = CONSTR_TOL;
    let mut count = 0usize;
    for i in 0..n_check {
        let ind = match perm {
            Some(p) => p[i],
            None => i,
        };
        let sign_max: f32 = if xmax[ind] > 0.0 { 1.0 } else { -1.0 };
        let upper = xmax[ind] * (1.0 + sign_max * tol) + tol;
        let sign_min: f32 = if xmin[ind] < 0.0 { 1.0 } else { -1.0 };
        let lower = xmin[ind] * (1.0 + sign_min * tol) - tol;

        if x[ind] >= upper {
            output[ind] = 1;
            count += 1;
        } else if x[ind] <= lower {
            output[ind] = -1;
            count += 1;
        } else {
            output[ind] = 0;
        }
    }
    count
}

// ---------------------------------------------------------------------------
// Householder QR with explicit Q recovery
// ---------------------------------------------------------------------------

/// Compute full Householder QR factorisation: A (m x n) = Q (m x m) * R (m x n).
///
/// Arrays are column-major: `mat[col][row]`.
/// `work` is the input; `q` and `r` are written on output.
#[allow(clippy::needless_range_loop)] // cross-column 2D array access prevents iterator use
pub fn householder_qr<const M: usize, const N: usize>(
    work: &[[f32; M]; N],
    q: &mut [[f32; M]; M],
    r: &mut [[f32; M]; N],
    m: usize,
    n: usize,
) {
    for j in 0..n {
        for i in 0..m {
            r[j][i] = work[j][i];
        }
    }
    for j in n..N {
        for i in 0..M {
            r[j][i] = 0.0;
        }
    }

    let mut tau = [0.0f32; N];
    let kmax = if m < n { m } else { n };

    for k in 0..kmax {
        let mut nu = 0.0f32;
        for i in (k + 1)..m {
            nu += r[k][i] * r[k][i];
        }
        nu = libm::sqrtf(nu);

        if nu < 1e-12 {
            tau[k] = 0.0;
            continue;
        }

        let beta = (if r[k][k] >= 0.0 { -1.0f32 } else { 1.0 }) * libm::hypotf(r[k][k], nu);
        tau[k] = (beta - r[k][k]) / beta;
        let scale = 1.0 / (r[k][k] - beta);
        for i in (k + 1)..m {
            r[k][i] *= scale;
        }
        r[k][k] = beta;

        for j in (k + 1)..n {
            let mut w = r[j][k];
            for i in (k + 1)..m {
                w += r[k][i] * r[j][i];
            }
            r[j][k] -= tau[k] * w;
            for i in (k + 1)..m {
                r[j][i] -= tau[k] * r[k][i] * w;
            }
        }
    }

    // Recover explicit Q from stored Householder vectors (reverse order)
    for j in 0..M {
        for i in 0..M {
            q[j][i] = if i == j { 1.0 } else { 0.0 };
        }
    }
    for k in (0..kmax).rev() {
        if tau[k] == 0.0 {
            continue;
        }
        for j in (k..m).rev() {
            let mut w = q[j][k];
            for i in (k + 1)..m {
                w += r[k][i] * q[j][i];
            }
            q[j][k] -= tau[k] * w;
            for i in (k + 1)..m {
                q[j][i] -= tau[k] * r[k][i] * w;
            }
        }
    }

    // Zero lower triangle of R (was Householder vector storage)
    for j in 0..n {
        for i in (j + 1)..m {
            r[j][i] = 0.0;
        }
    }
}

/// Back-substitute `Rx = b` where R is upper-triangular (first `n` rows/cols).
pub fn backward_tri_solve<const M: usize, const N: usize>(
    r: &[[f32; M]; N],
    b: &[f32; N],
    x: &mut [f32; N],
    n: usize,
) {
    if n == 0 {
        return;
    }
    x[n - 1] = b[n - 1] / r[n - 1][n - 1];
    for i in (0..n.saturating_sub(1)).rev() {
        let mut s = 0.0f32;
        for j in (i + 1)..n {
            s += r[j][i] * x[j];
        }
        x[i] = (b[i] - s) / r[i][i];
    }
}
