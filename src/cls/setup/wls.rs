//! Weighted least-squares (WLS) problem formulation.
//!
//! Converts control-allocation parameters (effectiveness matrix `G`,
//! pseudo-control weights `Wv`, actuator weights `Wu`) into the regularised
//! `min ‖Au − b‖²` form that [`solve`](crate::cls::solve) accepts.
//! The coefficient matrix `A` is `(NV + NU) × NU`.

use nalgebra::{allocator::Allocator, Const, DefaultAllocator, DimMin, DimName};

use crate::cls::types::{Mat, VecN};

/// Minimum diagonal value used when normalising actuator weights.
const MIN_DIAG_CLAMP: f32 = 1e-6;

#[allow(clippy::needless_range_loop)]
fn gamma_estimator<const NV: usize>(a2: &[[f32; NV]; NV], cond_target: f32) -> (f32, f32) {
    let mut max_sig: f32 = 0.0;
    for i in 0..NV {
        let mut r: f32 = 0.0;
        for j in 0..NV {
            if j != i {
                r += libm::fabsf(a2[i][j]);
            }
        }
        let disk = a2[i][i] + r;
        if max_sig < disk {
            max_sig = disk;
        }
    }
    (libm::sqrtf(max_sig / cond_target), max_sig)
}

/// Convert WLS control allocation to a least-squares problem `min ‖Au − b‖²`.
///
/// `wu` is **normalized in-place** by its minimum value (matching the C code).
/// Returns `(A, gamma)`.
#[allow(clippy::needless_range_loop)]
pub fn setup_a<const NU: usize, const NV: usize, const NC: usize>(
    b_mat: &Mat<NV, NU>,
    wv: &VecN<NV>,
    wu: &mut VecN<NU>,
    theta: f32,
    cond_bound: f32,
) -> (Mat<NC, NU>, f32)
where
    Const<NC>: DimName + DimMin<Const<NU>, Output = Const<NU>>,
    Const<NU>: DimName,
    Const<NV>: DimName,
    DefaultAllocator: Allocator<Const<NC>, Const<NU>>
        + Allocator<Const<NC>, Const<NC>>
        + Allocator<Const<NU>, Const<NU>>
        + Allocator<Const<NC>>
        + Allocator<Const<NU>>
        + Allocator<Const<NV>>,
{
    debug_assert_eq!(NC, NU + NV);

    let mut a2 = [[0.0f32; NV]; NV];
    for i in 0..NV {
        for j in i..NV {
            let mut sum = 0.0f32;
            for k in 0..NU {
                sum += b_mat[(i, k)] * b_mat[(j, k)];
            }
            a2[i][j] = sum * wv[i] * wv[i];
            if i != j {
                a2[j][i] = a2[i][j];
            }
        }
    }

    let mut min_diag: f32 = f32::INFINITY;
    let mut max_diag: f32 = 0.0;
    for i in 0..NU {
        if wu[i] < min_diag {
            min_diag = wu[i];
        }
        if wu[i] > max_diag {
            max_diag = wu[i];
        }
    }
    if min_diag < MIN_DIAG_CLAMP {
        min_diag = MIN_DIAG_CLAMP;
    }
    let inv = 1.0 / min_diag;
    for i in 0..NU {
        wu[i] *= inv;
    }
    max_diag *= inv;

    let gamma = if cond_bound > 0.0 {
        let (ge, ms) = gamma_estimator(&a2, cond_bound);
        let gt = libm::sqrtf(ms) * theta / max_diag;
        if ge > gt {
            ge
        } else {
            gt
        }
    } else {
        let (_, ms) = gamma_estimator(&a2, 1.0);
        libm::sqrtf(ms) * theta / max_diag
    };

    let mut a: Mat<NC, NU> = Mat::zeros();
    for j in 0..NU {
        for i in 0..NV {
            a[(i, j)] = wv[i] * b_mat[(i, j)];
        }
        a[(NV + j, j)] = gamma * wu[j];
    }

    (a, gamma)
}

/// Compute the right-hand side `b` for the regularised WLS problem.
pub fn setup_b<const NU: usize, const NV: usize, const NC: usize>(
    v: &VecN<NV>,
    ud: &VecN<NU>,
    wv: &VecN<NV>,
    wu_norm: &VecN<NU>,
    gamma: f32,
) -> VecN<NC>
where
    Const<NC>: DimName + DimMin<Const<NU>, Output = Const<NU>>,
    Const<NU>: DimName,
    Const<NV>: DimName,
    DefaultAllocator: Allocator<Const<NC>, Const<NU>>
        + Allocator<Const<NC>, Const<NC>>
        + Allocator<Const<NU>, Const<NU>>
        + Allocator<Const<NC>>
        + Allocator<Const<NU>>
        + Allocator<Const<NV>>,
{
    debug_assert_eq!(NC, NU + NV);
    let mut b: VecN<NC> = VecN::zeros();
    for i in 0..NV {
        b[i] = wv[i] * v[i];
    }
    for i in 0..NU {
        b[NV + i] = gamma * wu_norm[i] * ud[i];
    }
    b
}
