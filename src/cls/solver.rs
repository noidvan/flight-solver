use nalgebra::{allocator::Allocator, Const, DefaultAllocator, DimMin, DimName, OMatrix};

use super::linalg::check_limits_tol;
use super::types::{ExitCode, Mat, SolverStats, VecN, CONSTR_TOL};
use crate::givens::{givens, givens_left_apply, givens_right_apply_t};

fn qr_shift<const NC: usize, const NU: usize>(
    q: &mut OMatrix<f32, Const<NC>, Const<NU>>,
    r: &mut OMatrix<f32, Const<NU>, Const<NU>>,
    qtb: &mut [f32; NU],
    i: usize,
    j: usize,
) where
    Const<NC>: DimName,
    Const<NU>: DimName,
    DefaultAllocator: Allocator<Const<NC>, Const<NU>> + Allocator<Const<NU>, Const<NU>>,
{
    if i == j {
        return;
    }

    let n_givens: usize;
    if i > j {
        n_givens = i - j;
        for l in 0..NU {
            let tmp = r[(l, j)];
            for k in j..i {
                r[(l, k)] = r[(l, k + 1)];
            }
            r[(l, i)] = tmp;
        }
    } else {
        n_givens = j - i;
        for l in 0..NU {
            let tmp = r[(l, j)];
            for k in (i..j).rev() {
                r[(l, k + 1)] = r[(l, k)];
            }
            r[(l, i)] = tmp;
        }
    }

    for k in 0..n_givens {
        let (j1, i1) = if j > i {
            (j - k - 1, i)
        } else {
            (j + k, j + k)
        };
        let (c, s) = givens(r[(j1, i1)], r[(j1 + 1, i1)]);
        givens_left_apply(r, c, s, j1, j1 + 1, NU);
        givens_right_apply_t(q, c, s, j1, j1 + 1, NC);
        // Incrementally update cached Q^T * b
        let t1 = qtb[j1];
        let t2 = qtb[j1 + 1];
        qtb[j1] = c * t1 - s * t2;
        qtb[j1 + 1] = s * t1 + c * t2;
    }
}

fn backward_tri_solve<const NU: usize>(
    r: &OMatrix<f32, Const<NU>, Const<NU>>,
    b: &[f32; NU],
    x: &mut [f32; NU],
    n: usize,
) where
    Const<NU>: DimName,
    DefaultAllocator: Allocator<Const<NU>, Const<NU>>,
{
    if n == 0 {
        return;
    }
    x[n - 1] = b[n - 1] / r[(n - 1, n - 1)];
    for i in (0..n.saturating_sub(1)).rev() {
        let mut s = 0.0f32;
        for j in (i + 1)..n {
            s += r[(i, j)] * x[j];
        }
        x[i] = (b[i] - s) / r[(i, i)];
    }
}

// ---------------------------------------------------------------------------
// Incremental active-set solver
// ---------------------------------------------------------------------------

/// Active-set solver for the regularised WLS problem.
///
/// This is a convenience wrapper around [`solve_cls`] that enforces
/// `NC == NU + NV` (the augmented system produced by
/// [`setup::wls::setup_a`](crate::cls::setup::wls::setup_a) /
/// [`setup::wls::setup_b`](crate::cls::setup::wls::setup_b)).
#[allow(clippy::needless_range_loop)] // forwarding wrapper
pub fn solve<const NU: usize, const NV: usize, const NC: usize>(
    a: &Mat<NC, NU>,
    b: &VecN<NC>,
    umin: &VecN<NU>,
    umax: &VecN<NU>,
    us: &mut VecN<NU>,
    ws: &mut [i8; NU],
    imax: usize,
) -> SolverStats
where
    Const<NC>: DimName + DimMin<Const<NU>, Output = Const<NU>>,
    Const<NU>: DimName,
    Const<NV>: DimName,
    DefaultAllocator: Allocator<Const<NC>, Const<NU>>
        + Allocator<Const<NC>, Const<NC>>
        + Allocator<Const<NU>, Const<NU>>
        + Allocator<Const<NC>>
        + Allocator<Const<NU>>,
{
    debug_assert_eq!(NC, NU + NV);
    solve_cls(a, b, umin, umax, us, ws, imax)
}

/// General box-constrained least-squares solver.
///
/// Solves `min ‖Au − b‖²` subject to `umin ≤ u ≤ umax` using an active-set
/// method with incremental QR updates (Givens rotations).
///
/// Unlike [`solve`], this function does **not** require `NC == NU + NV` and
/// accepts any `NC ≥ NU`. Use it with the unregularised setup functions
/// ([`setup::ls`](crate::cls::setup::ls)) or with a custom `A` / `b`.
#[allow(clippy::needless_range_loop)] // multi-array index loops (ws, us, perm, bounds)
pub fn solve_cls<const NU: usize, const NC: usize>(
    a: &Mat<NC, NU>,
    b: &VecN<NC>,
    umin: &VecN<NU>,
    umax: &VecN<NU>,
    us: &mut VecN<NU>,
    ws: &mut [i8; NU],
    imax: usize,
) -> SolverStats
where
    Const<NC>: DimName + DimMin<Const<NU>, Output = Const<NU>>,
    Const<NU>: DimName,
    DefaultAllocator: Allocator<Const<NC>, Const<NU>>
        + Allocator<Const<NC>, Const<NC>>
        + Allocator<Const<NU>, Const<NU>>
        + Allocator<Const<NC>>
        + Allocator<Const<NU>>,
{
    let imax = if imax == 0 { 100 } else { imax };

    for i in 0..NU {
        if ws[i] == 0 {
            if us[i] > umax[i] {
                us[i] = umax[i];
            } else if us[i] < umin[i] {
                us[i] = umin[i];
            }
        } else {
            us[i] = if ws[i] > 0 { umax[i] } else { umin[i] };
        }
    }

    // Permutation: free first, bounded after
    let mut perm = [0usize; NU];
    let mut n_free: usize = 0;
    for i in 0..NU {
        if ws[i] == 0 {
            perm[n_free] = i;
            n_free += 1;
        }
    }
    let mut i_bnd: usize = 0;
    for i in 0..NU {
        if ws[i] != 0 {
            perm[n_free + i_bnd] = i;
            i_bnd += 1;
        }
    }

    // Permuted A → nalgebra QR → thin Q (NC×NU) and thin R (NU×NU)
    let mut a_perm: Mat<NC, NU> = Mat::zeros();
    for j in 0..NU {
        for i in 0..NC {
            a_perm[(i, j)] = a[(i, perm[j])];
        }
    }
    let qr_decomp = a_perm.qr();
    let mut q: OMatrix<f32, Const<NC>, Const<NU>> = qr_decomp.q();
    let mut r: OMatrix<f32, Const<NU>, Const<NU>> = qr_decomp.r();

    // Cache Q^T * b — updated incrementally via Givens in qr_shift
    let mut qtb = [0.0f32; NU];
    for i in 0..NU {
        let mut s = 0.0f32;
        for j in 0..NC {
            s += q[(j, i)] * b[j];
        }
        qtb[i] = s;
    }

    // Hoist bound arrays and scratch space outside the loop
    let mut umin_arr = [0.0f32; NU];
    let mut umax_arr = [0.0f32; NU];
    for i in 0..NU {
        umin_arr[i] = umin[i];
        umax_arr[i] = umax[i];
    }
    let mut w_temp = [0i8; NU];

    let mut z = [0.0f32; NU];
    let mut exit_code = ExitCode::IterLimit;

    let mut iter: usize = 0;
    while {
        iter += 1;
        iter <= imax
    } {
        // Use cached Q^T * b instead of recomputing from Q
        let mut c = [0.0f32; NU];
        c[..n_free].copy_from_slice(&qtb[..n_free]);

        for i in 0..n_free {
            for j in 0..(NU - n_free) {
                let pi = perm[n_free + j];
                let ub = if ws[pi] > 0 { umax[pi] } else { umin[pi] };
                c[i] -= r[(i, n_free + j)] * ub;
            }
        }

        let mut q_sol = [0.0f32; NU];
        backward_tri_solve(&r, &c, &mut q_sol, n_free);

        let mut nan_found = false;
        for i in 0..n_free {
            if f32::is_nan(q_sol[i]) {
                nan_found = true;
                break;
            }
            z[perm[i]] = q_sol[i];
        }
        if nan_found {
            exit_code = ExitCode::NanFoundQ;
            break;
        }
        for i in n_free..NU {
            z[perm[i]] = us[perm[i]];
        }

        let n_violated =
            check_limits_tol(n_free, &z, &umin_arr, &umax_arr, &mut w_temp, Some(&perm));

        if n_violated == 0 {
            for i in 0..n_free {
                us[perm[i]] = z[perm[i]];
            }

            if n_free == NU {
                exit_code = ExitCode::Success;
                break;
            }

            // Dual variables — use cached qtb instead of recomputing Q^T * b
            let mut d = [0.0f32; NU];
            d[n_free..NU].copy_from_slice(&qtb[n_free..NU]);
            for i in n_free..NU {
                for j in i..NU {
                    d[i] -= r[(i, j)] * us[perm[j]];
                }
            }

            let mut f_free: usize = 0;
            let mut maxlam: f32 = f32::NEG_INFINITY;
            for i in n_free..NU {
                let mut lam = 0.0f32;
                for j in n_free..=i {
                    lam += r[(j, i)] * d[j];
                }
                lam *= -f32::from(ws[perm[i]]);
                if lam > maxlam {
                    maxlam = lam;
                    f_free = i - n_free;
                }
            }

            if maxlam <= CONSTR_TOL {
                exit_code = ExitCode::Success;
                break;
            }

            qr_shift(&mut q, &mut r, &mut qtb, n_free, n_free + f_free);
            ws[perm[n_free + f_free]] = 0;
            let last_val = perm[n_free + f_free];
            for i in (1..=f_free).rev() {
                perm[n_free + i] = perm[n_free + i - 1];
            }
            perm[n_free] = last_val;
            n_free += 1;
        } else {
            let mut alpha: f32 = f32::INFINITY;
            let mut i_a: usize = 0;
            let mut f_bound: usize = 0;
            let mut i_s: i8 = 0;

            for f in 0..n_free {
                let ii = perm[f];
                let (tmp, ts) = if w_temp[ii] == -1 {
                    ((us[ii] - umin[ii]) / (us[ii] - z[ii]), -1i8)
                } else if w_temp[ii] == 1 {
                    ((umax[ii] - us[ii]) / (z[ii] - us[ii]), 1i8)
                } else {
                    continue;
                };
                if tmp < alpha {
                    alpha = tmp;
                    i_a = ii;
                    f_bound = f;
                    i_s = ts;
                }
            }

            let mut nan_found = false;
            for i in 0..NU {
                if i == i_a {
                    us[i] = if i_s == 1 { umax[i] } else { umin[i] };
                } else {
                    us[i] += alpha * (z[i] - us[i]);
                }
                if f32::is_nan(us[i]) {
                    nan_found = true;
                    break;
                }
            }
            if nan_found {
                exit_code = ExitCode::NanFoundUs;
                break;
            }

            qr_shift(&mut q, &mut r, &mut qtb, n_free - 1, f_bound);
            ws[i_a] = i_s;
            let first_val = perm[f_bound];
            for i in 0..(n_free - f_bound - 1) {
                perm[f_bound + i] = perm[f_bound + i + 1];
            }
            n_free -= 1;
            perm[n_free] = first_val;
        }
    }
    if exit_code == ExitCode::IterLimit {
        iter -= 1;
    }
    SolverStats {
        exit_code,
        iterations: iter,
        n_free,
    }
}
