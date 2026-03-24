//! Unregularised least-squares problem formulation.
//!
//! Builds `A = Wv · G` and `b = Wv · v` without the actuator-preference
//! regularisation term. The coefficient matrix is `NV × NU` (no extra rows).
//! Use with [`solve_cls`](crate::cls::solve_cls).

use nalgebra::{allocator::Allocator, Const, DefaultAllocator, DimMin, DimName};

use crate::cls::types::{Mat, VecN};

/// Build the unregularised coefficient matrix `A = Wv · G`.
///
/// Returns an `NV × NU` matrix suitable for [`solve_cls`](crate::cls::solve_cls).
pub fn setup_a<const NU: usize, const NV: usize>(b_mat: &Mat<NV, NU>, wv: &VecN<NV>) -> Mat<NV, NU>
where
    Const<NV>: DimName + DimMin<Const<NU>, Output = Const<NU>>,
    Const<NU>: DimName,
    DefaultAllocator: Allocator<Const<NV>, Const<NU>>
        + Allocator<Const<NV>, Const<NV>>
        + Allocator<Const<NU>, Const<NU>>
        + Allocator<Const<NV>>
        + Allocator<Const<NU>>,
{
    let mut a: Mat<NV, NU> = Mat::zeros();
    for j in 0..NU {
        for i in 0..NV {
            a[(i, j)] = wv[i] * b_mat[(i, j)];
        }
    }
    a
}

/// Build the unregularised right-hand side `b = Wv · v`.
pub fn setup_b<const NV: usize>(v: &VecN<NV>, wv: &VecN<NV>) -> VecN<NV>
where
    Const<NV>: DimName,
    DefaultAllocator: Allocator<Const<NV>>,
{
    let mut b: VecN<NV> = VecN::zeros();
    for i in 0..NV {
        b[i] = wv[i] * v[i];
    }
    b
}
