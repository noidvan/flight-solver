//! Cross-check: verify standard and inverse QR-RLS produce identical results.
//!
//! Both solvers implement the same RLS recurrence — they must agree to within
//! floating-point tolerance on identical inputs. Any divergence indicates a
//! bug in one of the implementations.

use flight_solver::rls::{CovarianceGuards, InverseQrRls, RlsParallel};
use nalgebra::SVector;

/// Maximum acceptable relative/absolute error between the two solvers.
///
/// The covariance form and information form are mathematically equivalent but
/// accumulate f32 rounding differently. Measured worst case across all test
/// configurations is ~5e-2 (lambda=0.999, 100 steps). Tolerance set at 2× that.
const CROSS_TOL: f32 = 1e-1;

fn max_param_diff<const N: usize, const P: usize>(
    std: &RlsParallel<N, P>,
    iqr: &InverseQrRls<N, P>,
) -> f32 {
    let mut max_diff: f32 = 0.0;
    for col in 0..P {
        for row in 0..N {
            let a = std.params()[(row, col)];
            let b = iqr.params()[(row, col)];
            let diff = (a - b).abs();
            let scale = a.abs().max(b.abs()).max(1e-10);
            let rel = diff / scale;
            max_diff = max_diff.max(rel).max(diff); // check both relative and absolute
        }
    }
    max_diff
}

/// Helper: run both solvers on identical random-ish data and compare.
fn run_parallel_cross_check<const N: usize, const P: usize>(
    gamma: f32,
    lambda: f32,
    n_steps: usize,
    seed: u32,
) {
    // Disable all numerical guards so standard RLS matches the pure
    // mathematical RLS recurrence (same as what inverse QR implements).
    // Setting cov_min to MAX prevents the order decrement logic from triggering.
    let guards = CovarianceGuards {
        cov_max: f32::MAX,
        cov_min: f32::MAX, // prevents order decrement limiting from triggering
        max_order_decrement: 0.1,
    };

    let mut std_rls = RlsParallel::<N, P>::new(gamma, lambda, guards);
    let mut iqr_rls = InverseQrRls::<N, P>::new(gamma, lambda);

    // Simple xorshift32 PRNG for reproducibility (no_std compatible)
    let mut rng = seed;
    let mut next_f32 = || -> f32 {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        // Map to [-1, 1]
        (rng as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };

    for step in 0..n_steps {
        // Generate regressor
        let mut a_data = [0.0f32; N];
        for v in a_data.iter_mut() {
            *v = next_f32();
        }
        let a = SVector::<f32, N>::from_column_slice(&a_data);

        // Generate observation
        let mut y_data = [0.0f32; P];
        for v in y_data.iter_mut() {
            *v = next_f32();
        }
        let y = SVector::<f32, P>::from_column_slice(&y_data);

        std_rls.update(&a, &y);
        iqr_rls.update(&a, &y);

        // Check agreement every 10 steps
        if (step + 1) % 10 == 0 || step == n_steps - 1 {
            let diff = max_param_diff(&std_rls, &iqr_rls);
            assert!(
                diff < CROSS_TOL,
                "Solvers diverged at step {} (N={}, P={}): max diff = {:.2e} (tol = {:.2e})",
                step + 1,
                N,
                P,
                diff,
                CROSS_TOL
            );
        }
    }
}

// ── Basic convergence tests ─────────────────────────────────────────────

#[test]
fn cross_check_n4_p1_basic() {
    run_parallel_cross_check::<4, 1>(1e2, 0.995, 100, 42);
}

#[test]
fn cross_check_n4_p3_basic() {
    run_parallel_cross_check::<4, 3>(1e2, 0.995, 100, 123);
}

#[test]
fn cross_check_n8_p3_basic() {
    run_parallel_cross_check::<8, 3>(1e2, 0.995, 100, 7);
}

// ── Different forgetting factors ────────────────────────────────────────

#[test]
fn cross_check_lambda_0999() {
    run_parallel_cross_check::<4, 1>(1e2, 0.999, 100, 99);
}

#[test]
fn cross_check_lambda_090() {
    // Aggressive forgetting amplifies numerical differences faster
    run_parallel_cross_check::<4, 1>(1e2, 0.90, 50, 55);
}

#[test]
fn cross_check_lambda_1() {
    run_parallel_cross_check::<4, 1>(1e2, 1.0, 100, 77);
}

// ── Different initial covariance ────────────────────────────────────────

#[test]
fn cross_check_high_gamma() {
    // High initial covariance causes catastrophic cancellation in the
    // standard form — a short run suffices to verify algorithm agreement
    run_parallel_cross_check::<4, 3>(1e4, 0.995, 30, 333);
}

#[test]
fn cross_check_low_gamma() {
    run_parallel_cross_check::<4, 3>(1.0, 0.995, 100, 444);
}

// ── Flight-realistic dimensions ─────────────────────────────────────────

#[test]
fn cross_check_motor_rls_n4_p1() {
    run_parallel_cross_check::<4, 1>(1e2, 0.995, 100, 8000);
}

#[test]
fn cross_check_g1g2_rls_n8_p3() {
    run_parallel_cross_check::<8, 3>(1e2, 0.995, 100, 8001);
}

#[test]
fn cross_check_force_rls_n4_p3() {
    run_parallel_cross_check::<4, 3>(1e2, 0.995, 100, 8002);
}

// ── Indiflight test case (from rls.c rlsTest) ──────────────────────────

#[test]
fn cross_check_indiflight_parallel_test_case() {
    let gamma = 1e3f32;
    // Compute lambda matching indiflight: powf(1 - ln2, Ts/Tchar)
    let lambda = libm::powf(1.0 - core::f32::consts::LN_2, 0.005 / 0.011212807);

    let guards = CovarianceGuards {
        cov_max: f32::MAX,
        cov_min: f32::MAX,
        max_order_decrement: 0.1,
    };

    let mut std_rls = RlsParallel::<3, 2>::new(gamma, lambda, guards);
    let mut iqr_rls = InverseQrRls::<3, 2>::new(gamma, lambda);

    // Set initial parameters (matching indiflight test)
    std_rls.params_mut()[(0, 0)] = 1.0;
    std_rls.params_mut()[(1, 0)] = 1.0;
    std_rls.params_mut()[(2, 0)] = 1.0;
    std_rls.params_mut()[(0, 1)] = 2.0;
    std_rls.params_mut()[(1, 1)] = 2.0;
    std_rls.params_mut()[(2, 1)] = 2.0;

    iqr_rls.params_mut()[(0, 0)] = 1.0;
    iqr_rls.params_mut()[(1, 0)] = 1.0;
    iqr_rls.params_mut()[(2, 0)] = 1.0;
    iqr_rls.params_mut()[(0, 1)] = 2.0;
    iqr_rls.params_mut()[(1, 1)] = 2.0;
    iqr_rls.params_mut()[(2, 1)] = 2.0;

    let a = SVector::<f32, 3>::new(0.03428682, 0.13472362, -0.16932012);
    let y = SVector::<f32, 2>::new(0.33754053, -0.16460538);

    std_rls.update(&a, &y);
    iqr_rls.update(&a, &y);

    let diff = max_param_diff(&std_rls, &iqr_rls);
    assert!(
        diff < CROSS_TOL,
        "Indiflight test case: solvers diverged, max diff = {:.2e}",
        diff
    );
}
