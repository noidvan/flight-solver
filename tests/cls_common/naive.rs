use nalgebra::{allocator::Allocator, Const, DefaultAllocator, DimMin, DimName};

use flight_solver::cls::linalg::{backward_tri_solve, check_limits_tol, householder_qr};
use flight_solver::cls::types::{ExitCode, SolverStats};
use flight_solver::cls::{Mat, VecN};

/// Active-set solver using naive QR re-factorisation each iteration.
///
/// Each iteration extracts the free columns of A and solves the subproblem via
/// a full Householder QR. This uses a custom QR implementation because the
/// subproblem column count (`n_free`) varies at runtime, and nalgebra's
/// static-sized QR requires compile-time dimensions.
///
/// Translates `solveActiveSet_qr_naive.c`.
#[allow(dead_code)]
pub fn solve_naive<const NU: usize, const NV: usize, const NC: usize>(
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
    let imax = if imax == 0 { 100 } else { imax };

    // Initialise: clamp free vars, snap bounded to limits
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

    // Free-index bookkeeping
    let mut free_index = [0usize; NU];
    let mut free_index_lookup = [usize::MAX; NU];
    let mut n_free: usize = 0;
    for i in 0..NU {
        if ws[i] == 0 {
            free_index_lookup[i] = n_free;
            free_index[n_free] = i;
            n_free += 1;
        }
    }

    // Initial residual d = b - A*us
    let mut d = [0.0f32; NC];
    for i in 0..NC {
        d[i] = b[i];
        for j in 0..NU {
            d[i] -= a[(i, j)] * us[j];
        }
    }

    let mut exit_code = ExitCode::IterLimit;
    let mut us_prev = [0.0f32; NU];
    let mut p = [0.0f32; NU];
    let mut p_free = [0.0f32; NU];

    let mut iter: usize = 0;
    while {
        iter += 1;
        iter <= imax
    } {
        p.fill(0.0);

        // Build A_free: pack free columns into col-major scratch.
        // Raw array because the logical column count (n_free) is runtime — can't
        // use nalgebra's static QR here.
        let mut af: [[f32; NC]; NU] = [[0.0f32; NC]; NU];
        for jj in 0..n_free {
            let col = free_index[jj];
            for i in 0..NC {
                af[jj][i] = a[(i, col)];
            }
        }

        p_free.fill(0.0);

        if n_free > 0 {
            let mut q_mat: [[f32; NC]; NC] = [[0.0f32; NC]; NC];
            let mut r_mat: [[f32; NC]; NU] = [[0.0f32; NC]; NU];
            householder_qr(&af, &mut q_mat, &mut r_mat, NC, n_free);

            let mut c = [0.0f32; NU];
            for i in 0..n_free {
                let mut s = 0.0f32;
                for k in 0..NC {
                    s += q_mat[i][k] * d[k];
                }
                c[i] = s;
            }

            backward_tri_solve(&r_mat, &c, &mut p_free, n_free);
        }

        // Save us, map p_free back, update us
        for i in 0..NU {
            us_prev[i] = us[i];
        }
        let mut nan_found = false;
        for i in 0..n_free {
            if f32::is_nan(p_free[i]) {
                nan_found = true;
                break;
            }
            p[free_index[i]] = p_free[i];
            us[free_index[i]] += p_free[i];
        }
        if nan_found {
            exit_code = ExitCode::NanFoundQ;
            break;
        }

        // Check limits
        let mut us_arr = [0.0f32; NU];
        let mut umin_arr = [0.0f32; NU];
        let mut umax_arr = [0.0f32; NU];
        for i in 0..NU {
            us_arr[i] = us[i];
            umin_arr[i] = umin[i];
            umax_arr[i] = umax[i];
        }
        let mut limits_viol = [0i8; NU];
        check_limits_tol(NU, &us_arr, &umin_arr, &umax_arr, &mut limits_viol, None);
        let n_infeasible = limits_viol.iter().filter(|&&v| v != 0).count();

        if n_infeasible == 0 {
            let mut lambda = [0.0f32; NU];
            for i in 0..NC {
                for k in 0..n_free {
                    d[i] -= af[k][i] * p_free[k];
                }
                for k in 0..NU {
                    lambda[k] += a[(i, k)] * d[i];
                }
            }
            let mut break_flag = true;
            for i in 0..NU {
                lambda[i] *= f32::from(ws[i]);
                if lambda[i] < -f32::EPSILON {
                    break_flag = false;
                    ws[i] = 0;
                    if free_index_lookup[i] == usize::MAX {
                        free_index_lookup[i] = n_free;
                        free_index[n_free] = i;
                        n_free += 1;
                    }
                }
            }
            if break_flag {
                exit_code = ExitCode::Success;
                break;
            }
        } else {
            let mut alpha: f32 = f32::INFINITY;
            let mut id_alpha: usize = 0;
            for &id in &free_index[..n_free] {
                if limits_viol[id] != 0 {
                    let at = if p[id] < 0.0 {
                        (umin[id] - us_prev[id]) / p[id]
                    } else {
                        (umax[id] - us_prev[id]) / p[id]
                    };
                    if at < alpha {
                        alpha = at;
                        id_alpha = id;
                    }
                }
            }

            let mut nan_found = false;
            for i in 0..NU {
                let incr = alpha * p[i];
                if i == id_alpha {
                    us[i] = if p[i] > 0.0 { umax[i] } else { umin[i] };
                    ws[i] = if p[i] > 0.0 { 1 } else { -1 };
                } else {
                    us[i] = incr + us_prev[i];
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

            for i in 0..NC {
                for k in 0..n_free {
                    d[i] -= af[k][i] * alpha * p_free[k];
                }
            }

            let old_slot = free_index_lookup[id_alpha];
            n_free -= 1;
            free_index[old_slot] = free_index[n_free];
            free_index_lookup[free_index[old_slot]] = old_slot;
            free_index_lookup[id_alpha] = usize::MAX;
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
