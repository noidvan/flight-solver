#![allow(clippy::needless_range_loop)]

//! Inverse QR-RLS solver using Givens rotations on the information matrix factor.
//!
//! Maintains `L` where `P⁻¹ = L Lᵀ` (lower-triangular Cholesky factor of
//! the information matrix). Updates via a sweep of `N` Givens rotations on
//! an augmented pre-array—numerically superior to the standard covariance
//! form because information updates involve only additions, never subtractions.
//!
//! No covariance guards are needed: the information form cannot suffer from
//! the catastrophic cancellation that plagues direct `P` maintenance.
//!
//! # References
//!
//! - Haykin, *Adaptive Filter Theory*, 5th ed., Ch. 15 (Square-Root
//!   Adaptive Filtering Algorithms), §15.1 Table 15.1 Part 2 and §15.3
//!   Table 15.2.

use nalgebra::{Const, DimName, OMatrix, OVector};

use super::types::UpdateStats;

/// Inverse QR-RLS solver with parallel multi-output support.
///
/// Maintains the lower-triangular Cholesky factor `L` of the information
/// matrix (`P⁻¹ = L Lᵀ`) and updates it via Givens rotations. This is
/// the recommended solver for embedded use: it is numerically robust,
/// requires no guards, and produces the Kalman gain as a byproduct of
/// the rotation sweep.
///
/// # Type parameters
///
/// - `N` — regressor dimension (number of parameters per output)
/// - `P` — number of parallel outputs sharing the same regressor
///
/// # Algorithm (per sample)
///
/// 1. Build augmented pre-array and sweep `N` Givens rotations → updated `L`, gain helper `k̄`, scale `γ`
/// 2. Solve `Lᵀ v = k̄` for Kalman gain `v` (triangular back-substitution)
/// 3. For each output `j`: `eⱼ = yⱼ − aᵀ xⱼ`, then `xⱼ += v · eⱼ`
///
/// Total cost: `O(N² + N·P)` per sample.
#[derive(Clone)]
pub struct InverseQrRls<const N: usize, const P: usize> {
    /// Parameter matrix `X ∈ ℝⁿˣᵖ` (column `j` = parameters for output `j`).
    x: OMatrix<f32, Const<N>, Const<P>>,
    /// Lower-triangular Cholesky factor `L` where `P⁻¹ = L Lᵀ`.
    l: OMatrix<f32, Const<N>, Const<N>>,
    /// Forgetting factor `λ ∈ (0, 1]`.
    lambda: f32,
    /// Cached `√λ` — avoids one `sqrtf` per frame.
    sqrt_lambda: f32,
    /// Number of samples processed.
    samples: u32,
}

impl<const N: usize, const P: usize> InverseQrRls<N, P>
where
    Const<N>: DimName,
    Const<P>: DimName,
{
    /// Create a new inverse QR-RLS with initial covariance `P₀ = γ I`.
    ///
    /// The information matrix factor is initialized as `L = (1/√γ) I` so
    /// that `P⁻¹ = L Lᵀ = (1/γ) I`.
    ///
    /// # Arguments
    ///
    /// - `gamma` — initial covariance diagonal (higher = more uncertain)
    /// - `lambda` — forgetting factor in `(0, 1]`
    pub fn new(gamma: f32, lambda: f32) -> Self {
        let inv_sqrt_gamma = 1.0 / libm::sqrtf(gamma);
        let mut l = OMatrix::<f32, Const<N>, Const<N>>::zeros();
        for i in 0..N {
            l[(i, i)] = inv_sqrt_gamma;
        }
        Self {
            x: OMatrix::<f32, Const<N>, Const<P>>::zeros(),
            l,
            lambda,
            sqrt_lambda: libm::sqrtf(lambda),
            samples: 0,
        }
    }

    /// Create from a time-constant–based forgetting factor.
    ///
    /// See [`RlsParallel::from_time_constant`](super::standard::RlsParallel::from_time_constant).
    pub fn from_time_constant(gamma: f32, ts: f32, t_char: f32) -> Self {
        let lambda = libm::powf(1.0 - core::f32::consts::LN_2, ts / t_char);
        Self::new(gamma, lambda)
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
    /// Augmented pre-array update via Givens rotations (Haykin Table 15.2):
    ///
    /// ```text
    /// [ √λ Lᵀ  |  0 ]        [ L_newᵀ  |  k̄ ]
    /// [  aᵀ    |  1 ]   →    [   0     |  γ  ]
    /// ```
    ///
    /// The unitary rotation `Θ = ∏ Θₖ` is a product of `N` Givens rotations,
    /// each zeroing one element of the regressor row against the corresponding
    /// diagonal of `Lᵀ` (Haykin Fig. 15.2b boundary cell).
    ///
    /// From the post-array identity `L_new k̄ = a`, the Kalman gain is
    /// `K = L_new⁻ᵀ k̄` (one triangular back-substitution). Then for each
    /// output: `xⱼ ← xⱼ + K (yⱼ − aᵀ xⱼ)`.
    ///
    /// The conversion factor `γ = ∏ cₖ` (Haykin Eq. 15.53) satisfies
    /// `γ² = 1 − aᵀ P_new a`.
    #[inline]
    pub fn update(
        &mut self,
        a: &OVector<f32, Const<N>>,
        y: &OVector<f32, Const<P>>,
    ) -> UpdateStats {
        self.samples += 1;

        // ── Step 1: Scale L by √λ (in-place, lower triangle only) ───
        let sqrt_lambda = self.sqrt_lambda;
        for col in 0..N {
            for row in col..N {
                self.l[(row, col)] *= sqrt_lambda;
            }
        }

        // ── Step 2: Givens rotation sweep ───────────────────────────
        // Working copy of regressor (bottom row of pre-array left block)
        let mut a_work = [0.0f32; N];
        for i in 0..N {
            a_work[i] = a[i];
        }

        // Right column: k_bar (rows 0..N-1, initially 0) and gamma (row N, initially 1)
        let mut k_bar = [0.0f32; N];
        let mut gamma: f32 = 1.0;

        // Each rotation zeros a_work[j] against L's diagonal L[(j,j)].
        // Row j of the pre-array (top block) stores row j of √λ Lᵀ,
        // accessed as column j of L: Lᵀ[j,k] = L[(k,j)] for k ≥ j.
        for j in 0..N {
            let (c, s) = crate::givens::givens(self.l[(j, j)], a_work[j]);

            // Rotate columns j..N-1 of the left block
            for k in j..N {
                let lkj = self.l[(k, j)]; // Lᵀ[j,k] = L[k,j]
                let ak = a_work[k];
                self.l[(k, j)] = c * lkj - s * ak;
                a_work[k] = s * lkj + c * ak;
            }

            // Rotate right column (k_bar[j] is 0 here — untouched by prior rotations)
            k_bar[j] = -s * gamma;
            gamma *= c;
        }

        // ── Step 3: Back-substitution for Kalman gain K ─────────────
        // Solve L_newᵀ K = k̄  where L_newᵀ[j,k] = L_new[(k,j)]
        let mut gain = k_bar;
        for j in (0..N).rev() {
            for k in (j + 1)..N {
                gain[j] -= self.l[(k, j)] * gain[k];
            }
            gain[j] /= self.l[(j, j)];
        }

        // ── Step 4: Parameter update for each output ────────────────
        for p in 0..P {
            // A priori prediction error: e = y_p - aᵀ x_p
            let mut e = y[p];
            for i in 0..N {
                e -= a[i] * self.x[(i, p)];
            }
            for i in 0..N {
                self.x[(i, p)] += gain[i] * e;
            }
        }

        UpdateStats {
            exit_code: super::types::ExitCode::Success,
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

    /// Current information matrix factor `L` where `P⁻¹ = L Lᵀ`.
    #[inline]
    pub fn info_factor(&self) -> &OMatrix<f32, Const<N>, Const<N>> {
        &self.l
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
        self.sqrt_lambda = libm::sqrtf(lambda);
    }

    /// Number of samples processed so far.
    #[inline]
    pub fn samples(&self) -> u32 {
        self.samples
    }
}
