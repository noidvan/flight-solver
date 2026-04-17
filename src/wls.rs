//! Encapsulated weighted least-squares control allocator.

use nalgebra::{allocator::Allocator, Const, DefaultAllocator, DimMin, DimName, OMatrix, OVector};

use crate::cls::setup::wls as setup;
use crate::cls::solver;
use crate::cls::types::SolverStats;

/// Stateful WLS control allocator: owns the static problem data and the
/// warm-start solver state across solves.
///
/// Build once via [`ControlAllocator::new`] when the effectiveness matrix or
/// weights change, then call [`solve`](Self::solve) on every control tick.
/// The previous solution is automatically reused as the warm-start for the
/// next solve.
///
/// Const generics:
/// - `NU`: number of actuators
/// - `NV`: number of pseudo-controls
/// - `NC`: must equal `NU + NV` (compile-time checked)
///
/// # Example
///
/// ```
/// use flight_solver::wls::ControlAllocator;
/// use flight_solver::cls::ExitCode;
/// use nalgebra::{SMatrix, Vector3, Vector4};
///
/// // 3 pseudo-controls × 4 motors (e.g. roll/pitch/yaw mixer)
/// #[rustfmt::skip]
/// let g = SMatrix::<f32, 3, 4>::new(
///     -0.5,  0.5,  1.0,
///      0.5,  0.5, -1.0,
///     -0.5, -0.5, -1.0,
///      0.5, -0.5,  1.0,
/// );
/// let wv = Vector3::new(1.0_f32, 1.0_f32, 0.5_f32);
/// let wu = Vector4::from_element(1.0_f32);
///
/// let mut alloc = ControlAllocator::<4, 3, 7>::new(&g, &wv, wu, 2e-9, 4e5);
///
/// let v = Vector3::new(0.1_f32, -0.2_f32, 0.05_f32);
/// let ud = Vector4::from_element(0.5_f32);
/// let umin = Vector4::from_element(0.0_f32);
/// let umax = Vector4::from_element(1.0_f32);
///
/// let stats = alloc.solve(&v, &ud, &umin, &umax, 100);
/// assert_eq!(stats.exit_code, ExitCode::Success);
/// let u = alloc.solution();
/// ```
pub struct ControlAllocator<const NU: usize, const NV: usize, const NC: usize>
where
    Const<NC>: DimName,
    Const<NU>: DimName,
    Const<NV>: DimName,
    DefaultAllocator: Allocator<Const<NC>, Const<NU>> + Allocator<Const<NU>> + Allocator<Const<NV>>,
{
    a: OMatrix<f32, Const<NC>, Const<NU>>,
    wv: OVector<f32, Const<NV>>,
    wu_norm: OVector<f32, Const<NU>>,
    gamma: f32,
    us: OVector<f32, Const<NU>>,
    ws: [i8; NU],
}

impl<const NU: usize, const NV: usize, const NC: usize> ControlAllocator<NU, NV, NC>
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
    /// Build the allocator: factor the augmented matrix `A`, compute the
    /// regularisation scalar `γ`, and normalise the actuator weights.
    ///
    /// `wu` is consumed so the in-place normalisation is fully internal — the
    /// caller's data is never mutated through aliasing. The warm-start is
    /// initialised to zero; use [`set_warmstart`](Self::set_warmstart) to
    /// seed a non-zero initial guess before the first [`solve`](Self::solve).
    pub fn new(
        g: &OMatrix<f32, Const<NV>, Const<NU>>,
        wv: &OVector<f32, Const<NV>>,
        mut wu: OVector<f32, Const<NU>>,
        theta: f32,
        cond_bound: f32,
    ) -> Self {
        const { assert!(NC == NU + NV, "ControlAllocator requires NC == NU + NV") };
        let (a, gamma) = setup::setup_a::<NU, NV, NC>(g, wv, &mut wu, theta, cond_bound);
        Self {
            a,
            wv: wv.clone_owned(),
            wu_norm: wu,
            gamma,
            us: OVector::zeros(),
            ws: [0i8; NU],
        }
    }

    /// Run one constrained least-squares solve.
    ///
    /// Builds the right-hand side `b` from `v` and `ud` using the stored
    /// weights and `γ`, then runs the active-set solver continuing from the
    /// current warm-start. The solution is left in the allocator and can be
    /// read via [`solution`](Self::solution).
    pub fn solve(
        &mut self,
        v: &OVector<f32, Const<NV>>,
        ud: &OVector<f32, Const<NU>>,
        umin: &OVector<f32, Const<NU>>,
        umax: &OVector<f32, Const<NU>>,
        imax: usize,
    ) -> SolverStats {
        let b = setup::setup_b::<NU, NV, NC>(v, ud, &self.wv, &self.wu_norm, self.gamma);
        solver::solve::<NU, NV, NC>(&self.a, &b, umin, umax, &mut self.us, &mut self.ws, imax)
    }

    /// The current actuator solution (also the warm-start for the next solve).
    pub fn solution(&self) -> &OVector<f32, Const<NU>> {
        &self.us
    }

    /// The regularisation scalar `γ` chosen at construction time.
    pub fn gamma(&self) -> f32 {
        self.gamma
    }

    /// Seed the warm-start with an explicit initial guess and clear the
    /// active set. Call before [`solve`](Self::solve) when starting a new
    /// trajectory; otherwise the previous solution is reused automatically.
    pub fn set_warmstart(&mut self, us: &OVector<f32, Const<NU>>) {
        self.us = us.clone_owned();
        self.ws = [0i8; NU];
    }

    /// Zero the warm-start solution and clear the active set.
    pub fn reset_warmstart(&mut self) {
        self.us = OVector::zeros();
        self.ws = [0i8; NU];
    }
}
