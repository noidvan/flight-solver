// Hot-path numerical code: explicit index loops match the C reference
// algorithm and produce identical codegen under LTO. Iterator style
// would obscure the matrix-index correspondence with no perf benefit.
#![allow(clippy::needless_range_loop, clippy::manual_memcpy)]

//! Standard (covariance-form) Recursive Least Squares.
//!
//! Two variants:
//!
//! - [`Rls<N, D>`] — Standard RLS with `D`-dimensional observation. Uses
//!   Cholesky decomposition of the `D × D` innovation matrix `M` when `D > 1`.
//!   Model: `y = A x` where `x ∈ ℝⁿ`, `y ∈ ℝᵈ`, `A ∈ ℝᵈˣⁿ`.
//!
//! - [`RlsParallel<N, P>`] — `P` independent single-output RLS instances
//!   sharing one covariance matrix. The regressor `a ∈ ℝⁿ` is a vector
//!   (scalar observation per output), so the denominator is a scalar—no
//!   Cholesky required. Model: `yᵀ = a X` where `X ∈ ℝⁿˣᵖ`.
//!
//! Both include the numerical guards from the indiflight C reference:
//! covariance explosion detection, order decrement limiting, and diagonal
//! clamping.

use nalgebra::{Const, DimName, OMatrix, OVector};

use super::types::{CovarianceGuards, ExitCode, UpdateStats};

// ═══════════════════════════════════════════════════════════════════════════
//  Rls<N, D> — standard RLS with D-dimensional observation
// ═══════════════════════════════════════════════════════════════════════════

/// Standard RLS maintaining the covariance matrix `P` directly.
///
/// Supports multi-dimensional observations (`D > 1`) via Cholesky
/// decomposition of the `D × D` innovation matrix `M = λI + A P Aᵀ`.
///
/// # Model
///
/// ```text
/// y = A x    where  y ∈ ℝᵈ,  x ∈ ℝⁿ,  A ∈ ℝᵈˣⁿ
/// ```
///
/// The regressor matrix is passed as `Aᵀ ∈ ℝⁿˣᵈ` (column-major, matching
/// the indiflight C convention).
///
/// # Type parameters
///
/// - `N` — number of parameters (regressor dimension)
/// - `D` — observation dimension (number of outputs per sample)
///
/// # Example
///
/// ```no_run
/// use flight_solver::rls::{Rls, CovarianceGuards};
/// use nalgebra::SMatrix;
///
/// // 3 parameters, 2-dimensional observations
/// let mut rls = Rls::<3, 2>::new(1e2, 0.995, CovarianceGuards::default());
///
/// let a_t = SMatrix::<f32, 3, 2>::new(
///     0.034, 0.227,
///     0.135, -0.436,
///     -0.169, -0.059,
/// );
/// let y = nalgebra::SVector::<f32, 2>::new(0.338, -0.165);
///
/// let stats = rls.update(&a_t, &y);
/// ```
#[derive(Clone)]
pub struct Rls<const N: usize, const D: usize> {
    /// Parameter vector `x ∈ ℝⁿ`.
    x: OVector<f32, Const<N>>,
    /// Covariance matrix `P ∈ ℝⁿˣⁿ` (symmetric positive definite).
    p: OMatrix<f32, Const<N>, Const<N>>,
    /// Forgetting factor `λ ∈ (0, 1]`.
    lambda: f32,
    /// Number of samples processed.
    samples: u32,
    /// Numerical guard configuration.
    guards: CovarianceGuards,
}

impl<const N: usize, const D: usize> Rls<N, D>
where
    Const<N>: DimName,
    Const<D>: DimName,
{
    /// Create a new standard RLS with initial covariance `P = γ I`.
    ///
    /// # Arguments
    ///
    /// - `gamma` — initial covariance diagonal (higher = more uncertain = faster learning)
    /// - `lambda` — forgetting factor in `(0, 1]` (lower = faster forgetting of old data)
    /// - `guards` — numerical guard configuration
    pub fn new(gamma: f32, lambda: f32, guards: CovarianceGuards) -> Self {
        let mut p = OMatrix::<f32, Const<N>, Const<N>>::zeros();
        for i in 0..N {
            p[(i, i)] = gamma;
        }
        Self {
            x: OVector::<f32, Const<N>>::zeros(),
            p,
            lambda,
            samples: 0,
            guards,
        }
    }

    /// Process one observation and update parameter estimate.
    ///
    /// # Arguments
    ///
    /// - `a_t` — regressor matrix transposed, `Aᵀ ∈ ℝⁿˣᵈ` (column-major).
    ///   Each column is one regressor vector.
    /// - `y` — observation vector `y ∈ ℝᵈ`
    ///
    /// # Algorithm
    ///
    /// 1. Covariance explosion check (temporarily increase λ if `P[i,i] > cov_max`)
    /// 2. Form `M = λI + A P Aᵀ` and Cholesky-solve for `K = P Aᵀ M⁻¹`
    /// 3. Order decrement limiting on `trace(K A P)`
    /// 4. Covariance update: `P ← (KAPmult · P − P Aᵀ Kᵀ) / λ`
    /// 5. Parameter update: `x ← x + K (y − A x)`
    #[inline]
    pub fn update(
        &mut self,
        a_t: &OMatrix<f32, Const<N>, Const<D>>,
        y: &OVector<f32, Const<D>>,
    ) -> UpdateStats {
        self.samples += 1;

        // ── Step 1: Covariance explosion detection ──────────────────────
        let mut lam = self.lambda;
        let mut trace_p: f32 = 0.0;
        let mut exit_code = ExitCode::Success;
        for i in 0..N {
            trace_p += self.p[(i, i)];
            if self.p[(i, i)] > self.guards.cov_max {
                lam = 1.0 + 0.1 * (1.0 - self.lambda);
                exit_code = ExitCode::CovarianceExplosion;
            }
        }

        // ── Step 2: Compute P Aᵀ (n×d) ─────────────────────────────────
        // PAT[i,j] = Σ_k P[i,k] * AT[k,j]
        let mut pat = [[0.0f32; D]; N]; // pat[row][col], row-major for cache
        for j in 0..D {
            for i in 0..N {
                let mut sum = 0.0f32;
                for k in 0..N {
                    sum += self.p[(k, i)] * a_t[(k, j)];
                }
                pat[i][j] = sum;
            }
        }

        // ── Step 3: Form M = λI + Aᵀᵀ P Aᵀ = λI + (P Aᵀ)ᵀ Aᵀ ────────
        // M is d×d symmetric
        let mut m = [[0.0f32; D]; D];
        for i in 0..D {
            m[i][i] = lam;
        }
        for i in 0..D {
            for j in 0..D {
                let mut sum = 0.0f32;
                for k in 0..N {
                    sum += pat[k][i] * a_t[(k, j)];
                }
                m[i][j] += sum;
            }
        }

        // ── Step 4: Cholesky decomposition M = UᵀU and solve K ─────────
        // Matches indiflight's chol(): upper triangular U where M = UᵀU
        // Storage: u[i][j] = U(j, i) in mathematical notation
        // (indiflight uses column-major U[i*d+j] → element at col=i, row=j)
        let mut u = [[0.0f32; D]; D];
        let mut i_diag = [0.0f32; D];
        for i in 0..D {
            for j in 0..=i {
                let mut s = 0.0f32;
                for k in 0..j {
                    s += u[i][k] * u[j][k];
                }
                if i == j {
                    u[i][j] = libm::sqrtf(m[i][i] - s);
                    i_diag[j] = 1.0 / u[i][j];
                } else {
                    u[i][j] = i_diag[j] * (m[i][j] - s);
                }
            }
        }

        // AP = (P Aᵀ)ᵀ = A P  (d×n), stored as ap[col][row] column-major
        // ap[col_n][row_d] = pat[col_n][row_d] transposed
        // In indiflight: AP[row + col*d] = PAT[col + row*n]

        // Solve Kᵀ = M⁻¹ A P  column by column via Cholesky forward/backward sub.
        // KT is d×n: kt[col_n][row_d]
        let mut kt = [[0.0f32; D]; N];
        for col in 0..N {
            // RHS for this column: AP[:,col] = PAT[col,:] (d elements)
            let mut b = [0.0f32; D];
            for row in 0..D {
                b[row] = pat[col][row]; // AP[row, col] = PAT[col][row]
            }

            // Forward substitution: Uᵀ y = b
            // Uᵀ[j,k] = U[k,j]. Cholesky stores u[i][j] = U[i*d+j] which
            // is element at (col=i, row=j). So U(row=k, col=j) = u[j][k].
            // For forward sub: need Uᵀ(j,k) for k < j, i.e. U(k,j) = u[j][k].
            let mut x_sol = [0.0f32; D];
            for j in 0..D {
                let mut t = b[j];
                for k in 0..j {
                    t -= u[j][k] * x_sol[k];
                }
                x_sol[j] = t * i_diag[j];
            }

            // Backward substitution: U x = y
            // Need U(j,k) for k > j, i.e. u[k][j].
            for j in (0..D).rev() {
                let mut t = x_sol[j];
                for k in (j + 1)..D {
                    t -= u[k][j] * x_sol[k];
                }
                x_sol[j] = t * i_diag[j];
            }

            for row in 0..D {
                kt[col][row] = x_sol[row];
            }
        }

        // ── Step 5: Order decrement limiting ────────────────────────────
        // trace(K A P) = trace(PAT · KT) = Σ_i dot(AP_col_i, KT_col_i)
        // where AP_col_i[row_d] = pat[i][row_d] and KT_col_i[row_d] = kt[i][row_d]
        let mut trace_kap: f32 = 0.0;
        for i in 0..N {
            for j in 0..D {
                trace_kap += pat[i][j] * kt[i][j];
            }
        }

        let mut kap_mult: f32 = 1.0;
        if trace_kap > self.guards.cov_min {
            kap_mult = ((1.0 - self.guards.max_order_decrement) * (trace_p / trace_kap)).min(1.0);
        }

        // ── Step 6: Covariance update ───────────────────────────────────
        // P = (KAPmult · P − PAT · KT) / λ
        // Matching indiflight: SGEMM(n, n, d, PAT, KT, P, -KAPmult, -ilam)
        //   → P = (-1/λ)((-KAPmult)·P + PAT·KT)
        //   → P = (KAPmult·P − PAT·KT) / λ
        // Fused form: P[r,c] = (kap/λ)*P[r,c] - (1/λ)*Σ_k pat[r][k]*kt[c][k]
        let kap_ilam = kap_mult / lam;
        let neg_ilam = -1.0 / lam;
        for col in 0..N {
            for row in 0..N {
                let mut rank_d = 0.0f32;
                for k in 0..D {
                    rank_d += pat[row][k] * kt[col][k];
                }
                self.p[(row, col)] = kap_ilam * self.p[(row, col)] + neg_ilam * rank_d;
            }
        }

        // ── Step 7: Compute error and update parameters ─────────────────
        // e = y − A x = y − Aᵀᵀ x
        // eᵀ = yᵀ − xᵀ Aᵀ  →  e[j] = y[j] − Σ_k x[k] · AT[k,j]
        let mut e = [0.0f32; D];
        for j in 0..D {
            let mut sum = 0.0f32;
            for k in 0..N {
                sum += self.x[k] * a_t[(k, j)];
            }
            e[j] = y[j] - sum;
        }

        // dx = K e = Kᵀᵀ e  →  dx[i] = Σ_j KT[i][j] · e[j]
        // But KT is stored as kt[col_n][row_d], and K = KTᵀ.
        // dx[i] = Σ_j K[i,j] · e[j] = Σ_j KT[j,i] · e[j]
        // With our storage: KT[j,i] = kt[j][i]... hmm.
        // Actually: kt[col_n][row_d] stores KT which is d×n.
        // KT(row_d, col_n) → kt[col_n][row_d]. So KT[j_d, i_n] = kt[i_n][j_d].
        // dx[i] = Σ_j KT(j, i) · e[j] = Σ_j kt[i][j] · e[j]
        for i in 0..N {
            let mut dx = 0.0f32;
            for j in 0..D {
                dx += kt[i][j] * e[j];
            }
            self.x[i] += dx;
        }

        UpdateStats {
            exit_code,
            samples: self.samples,
        }
    }

    /// Current parameter estimate `x ∈ ℝⁿ`.
    #[inline]
    pub fn params(&self) -> &OVector<f32, Const<N>> {
        &self.x
    }

    /// Mutable access to the parameter estimate.
    #[inline]
    pub fn params_mut(&mut self) -> &mut OVector<f32, Const<N>> {
        &mut self.x
    }

    /// Current covariance matrix `P ∈ ℝⁿˣⁿ`.
    #[inline]
    pub fn covariance(&self) -> &OMatrix<f32, Const<N>, Const<N>> {
        &self.p
    }

    /// Forgetting factor `λ`.
    #[inline]
    pub fn lambda(&self) -> f32 {
        self.lambda
    }

    /// Set the forgetting factor `λ ∈ (0, 1]`.
    #[inline]
    pub fn set_lambda(&mut self, lambda: f32) {
        self.lambda = lambda;
    }

    /// Number of samples processed so far.
    #[inline]
    pub fn samples(&self) -> u32 {
        self.samples
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  RlsParallel<N, P> — parallel multi-output with shared covariance
// ═══════════════════════════════════════════════════════════════════════════

/// Parallel multi-output RLS with shared covariance.
///
/// Runs `P` independent scalar-output RLS instances that share the same
/// regressor vector `a ∈ ℝⁿ` and covariance matrix `P ∈ ℝⁿˣⁿ`. Since
/// each observation is scalar, the denominator `λ + a P aᵀ` is a scalar—no
/// Cholesky decomposition required.
///
/// # Model
///
/// ```text
/// yᵀ = a X    where  yᵀ ∈ ℝᵖ,  X ∈ ℝⁿˣᵖ,  a ∈ ℝⁿ
/// ```
///
/// Each column `xⱼ` of `X` is an independent parameter vector. All `P`
/// outputs share the covariance update but have independent parameter
/// updates weighted by their individual prediction errors.
///
/// # Type parameters
///
/// - `N` — number of parameters per output (regressor dimension)
/// - `P` — number of parallel outputs
///
/// # Example
///
/// ```no_run
/// use flight_solver::rls::{RlsParallel, CovarianceGuards};
///
/// // 4 parameters, 3 parallel outputs (e.g. G1 learning for roll/pitch/yaw)
/// let mut rls = RlsParallel::<4, 3>::new(1e2, 0.995, CovarianceGuards::default());
///
/// let a = nalgebra::SVector::<f32, 4>::new(0.1, -0.2, 0.3, 0.05);
/// let y = nalgebra::SVector::<f32, 3>::new(0.5, -0.3, 0.1);
///
/// let stats = rls.update(&a, &y);
/// ```
#[derive(Clone)]
pub struct RlsParallel<const N: usize, const P: usize> {
    /// Parameter matrix `X ∈ ℝⁿˣᵖ` (column `j` = parameters for output `j`).
    x: OMatrix<f32, Const<N>, Const<P>>,
    /// Shared covariance matrix `P ∈ ℝⁿˣⁿ` (symmetric positive definite).
    p: OMatrix<f32, Const<N>, Const<N>>,
    /// Forgetting factor `λ ∈ (0, 1]`.
    lambda: f32,
    /// Number of samples processed.
    samples: u32,
    /// Numerical guard configuration.
    guards: CovarianceGuards,
}

impl<const N: usize, const P: usize> RlsParallel<N, P>
where
    Const<N>: DimName,
    Const<P>: DimName,
{
    /// Create a new parallel RLS with initial covariance `P = γ I`.
    ///
    /// # Arguments
    ///
    /// - `gamma` — initial covariance diagonal
    /// - `lambda` — forgetting factor in `(0, 1]`
    /// - `guards` — numerical guard configuration
    pub fn new(gamma: f32, lambda: f32, guards: CovarianceGuards) -> Self {
        let mut p = OMatrix::<f32, Const<N>, Const<N>>::zeros();
        for i in 0..N {
            p[(i, i)] = gamma;
        }
        Self {
            x: OMatrix::<f32, Const<N>, Const<P>>::zeros(),
            p,
            lambda,
            samples: 0,
            guards,
        }
    }

    /// Create from a time-constant–based forgetting factor, matching
    /// indiflight's `rlsParallelInit`.
    ///
    /// ```text
    /// λ = (1 − ln 2)^(Ts / Tchar)
    /// ```
    ///
    /// At time `Tchar`, the weight of the oldest sample has decayed to 50%.
    ///
    /// # Arguments
    ///
    /// - `gamma` — initial covariance diagonal
    /// - `ts` — sampling period in seconds (e.g. `1.0 / 8000.0`)
    /// - `t_char` — characteristic forgetting time in seconds
    /// - `guards` — numerical guard configuration
    pub fn from_time_constant(gamma: f32, ts: f32, t_char: f32, guards: CovarianceGuards) -> Self {
        let lambda = libm::powf(1.0 - core::f32::consts::LN_2, ts / t_char);
        Self::new(gamma, lambda, guards)
    }

    /// Process one observation and update parameter estimates.
    ///
    /// # Arguments
    ///
    /// - `a` — shared regressor vector `a ∈ ℝⁿ`
    /// - `y` — observation vector `y ∈ ℝᵖ` (one scalar per output)
    ///
    /// # Algorithm
    ///
    /// 1. Covariance explosion check
    /// 2. Compute `P aᵀ` and scalar `a P aᵀ`
    /// 3. Gain vector: `k = P aᵀ / (λ + a P aᵀ)`
    /// 4. Per-diagonal order decrement limiting
    /// 5. Covariance update: `P ← (KAPmult · P − k (P aᵀ)ᵀ) / λ`
    /// 6. Parameter update: `X ← X + k eᵀ`
    #[inline]
    pub fn update(
        &mut self,
        a: &OVector<f32, Const<N>>,
        y: &OVector<f32, Const<P>>,
    ) -> UpdateStats {
        self.samples += 1;

        // ── Step 1: Covariance explosion detection ──────────────────────
        let mut lam = self.lambda;
        let mut diag_p = [0.0f32; N];
        let mut exit_code = ExitCode::Success;
        for i in 0..N {
            diag_p[i] = self.p[(i, i)];
            if diag_p[i] > self.guards.cov_max {
                lam = 1.0 + 0.1 * (1.0 - self.lambda);
                exit_code = ExitCode::CovarianceExplosion;
            }
        }

        // ── Step 2: Compute P aᵀ (n-vector) ────────────────────────────
        // Matching indiflight: SGEMVt(n, n, P, aT, PaT)
        // PaT[i] = Σ_k P^T[i,k] * a[k] = Σ_k P[k,i] * a[k]
        let mut pa = [0.0f32; N];
        for i in 0..N {
            let mut sum = 0.0f32;
            for k in 0..N {
                sum += self.p[(k, i)] * a[k];
            }
            pa[i] = sum;
        }

        // ── Step 3: Scalar denominator a P aᵀ ──────────────────────────
        let mut a_pa: f32 = 0.0;
        for i in 0..N {
            a_pa += a[i] * pa[i];
        }

        // ── Step 4: Prediction errors (computed before param update) ────
        // eT[j] = y[j] - aᵀ X[:,j]
        let mut e = [0.0f32; P];
        for j in 0..P {
            let mut ax = 0.0f32;
            for k in 0..N {
                ax += a[k] * self.x[(k, j)];
            }
            e[j] = y[j] - ax;
        }

        // ── Step 5: Gain vector k = P aᵀ / (λ + a P aᵀ) ───────────────
        let i_sig = 1.0 / (lam + a_pa);
        let mut k = [0.0f32; N];
        for i in 0..N {
            k[i] = pa[i] * i_sig;
        }

        // ── Step 6: Per-diagonal order decrement limiting ───────────────
        // diagKAP[i] = k[i] * PaT[i] (diagonal of rank-1 matrix k ⊗ PaT)
        let mut max_diag_ratio: f32 = 0.0;
        for i in 0..N {
            let diag_kap = k[i] * pa[i];
            if diag_kap > 1e-6 {
                let ratio = diag_p[i] / diag_kap;
                if ratio > max_diag_ratio {
                    max_diag_ratio = ratio;
                }
            }
        }

        let mut kap_mult: f32 = 1.0;
        if max_diag_ratio > self.guards.cov_min {
            kap_mult = ((1.0 - self.guards.max_order_decrement) * max_diag_ratio).min(1.0);
        }

        // ── Step 7: Covariance update ───────────────────────────────────
        // P = (KAPmult · P − k ⊗ PaT) / λ
        // Matching indiflight: SGEMM(n, n, 1, k, PaT, P, -KAPmult, -ilam)
        // Fused form: P[r,c] = (kap_mult/λ)*P[r,c] - (1/λ)*k[r]*pa[c]
        // Saves one multiply per inner iteration vs the two-step version.
        let kap_ilam = kap_mult / lam;
        let neg_ilam = -1.0 / lam;
        for col in 0..N {
            let pa_col_scaled = pa[col] * neg_ilam;
            for row in 0..N {
                self.p[(row, col)] = kap_ilam * self.p[(row, col)] + k[row] * pa_col_scaled;
            }
        }

        // ── Step 8: Parameter update X += k eᵀ ─────────────────────────
        // Matching indiflight: SGEMM(n, p, 1, k, eT, X, 1, 1)
        for j in 0..P {
            for i in 0..N {
                self.x[(i, j)] += k[i] * e[j];
            }
        }

        UpdateStats {
            exit_code,
            samples: self.samples,
        }
    }

    /// Current parameter matrix `X ∈ ℝⁿˣᵖ`.
    #[inline]
    pub fn params(&self) -> &OMatrix<f32, Const<N>, Const<P>> {
        &self.x
    }

    /// Mutable access to the parameter matrix.
    #[inline]
    pub fn params_mut(&mut self) -> &mut OMatrix<f32, Const<N>, Const<P>> {
        &mut self.x
    }

    /// Current shared covariance matrix `P ∈ ℝⁿˣⁿ`.
    #[inline]
    pub fn covariance(&self) -> &OMatrix<f32, Const<N>, Const<N>> {
        &self.p
    }

    /// Forgetting factor `λ`.
    #[inline]
    pub fn lambda(&self) -> f32 {
        self.lambda
    }

    /// Set the forgetting factor `λ ∈ (0, 1]`.
    #[inline]
    pub fn set_lambda(&mut self, lambda: f32) {
        self.lambda = lambda;
    }

    /// Number of samples processed so far.
    #[inline]
    pub fn samples(&self) -> u32 {
        self.samples
    }
}
