#![allow(clippy::excessive_precision)]
//! Golden tests: verify standard RLS matches indiflight C reference output.
//!
//! Test cases and expected values from `indiflight/src/main/common/rls.c`
//! function `rlsTest()`.

use flight_solver::rls::{CovarianceGuards, Rls, RlsParallel};
use nalgebra::{SMatrix, SVector};

const TOL: f32 = 2e-3;

fn assert_vec_close<const N: usize>(actual: &SVector<f32, N>, expected: &[f32; N], tol: f32) {
    for i in 0..N {
        let diff = (actual[i] - expected[i]).abs();
        let scale = expected[i].abs().max(1.0);
        assert!(
            diff / scale < tol,
            "Element [{}]: actual={:.8}, expected={:.8}, diff={:.2e}",
            i,
            actual[i],
            expected[i],
            diff
        );
    }
}

fn assert_mat_close<const R: usize, const C: usize>(
    actual: &SMatrix<f32, R, C>,
    expected: &[[f32; C]; R],
    tol: f32,
) {
    for row in 0..R {
        for col in 0..C {
            let diff = (actual[(row, col)] - expected[row][col]).abs();
            let scale = expected[row][col].abs().max(1.0);
            assert!(
                diff / scale < tol,
                "Element [{}, {}]: actual={:.8}, expected={:.8}, diff={:.2e}",
                row,
                col,
                actual[(row, col)],
                expected[row][col],
                diff
            );
        }
    }
}

/// Golden test from indiflight's rlsTest() — Rls<3, 2> with Cholesky
///
/// gamma=1e3, lambda=0.9, initial x=[1,2,3]
/// Expected: newP and newx from rls.c commented reference values.
#[test]
fn golden_standard_rls_3x2() {
    let gamma = 1e3f32;
    let lambda = 0.9f32;
    let guards = CovarianceGuards::default();

    let mut rls = Rls::<3, 2>::new(gamma, lambda, guards);
    *rls.params_mut() = SVector::<f32, 3>::new(1.0, 2.0, 3.0);

    // AT is 3×2 (n×d) — nalgebra new() fills row-major
    let a_t = SMatrix::<f32, 3, 2>::new(
        0.03428682,
        0.22687657,
        0.13472362,
        -0.43582882,
        -0.16932012,
        -0.05850954,
    );
    let y = SVector::<f32, 2>::new(0.33754053, -0.16460538);

    rls.update(&a_t, &y);

    assert_vec_close(rls.params(), &[2.72319665, 1.79674801, 0.06989191], TOL);

    let expected_p = [
        [740.73667109, 328.91447157, 401.56846372],
        [328.91447157, 150.83991394, 177.92090952],
        [401.56846372, 177.92090952, 248.03972859],
    ];
    assert_mat_close(rls.covariance(), &expected_p, TOL);
}

/// Verify RlsParallel structural correctness and convergence.
#[test]
fn golden_parallel_structural() {
    let gamma = 1e3f32;
    let lambda = 0.9f32;
    let guards = CovarianceGuards::default();

    let mut rls = RlsParallel::<3, 2>::new(gamma, lambda, guards);
    rls.params_mut()[(0, 0)] = 1.0;
    rls.params_mut()[(1, 0)] = 1.0;
    rls.params_mut()[(2, 0)] = 1.0;
    rls.params_mut()[(0, 1)] = 2.0;
    rls.params_mut()[(1, 1)] = 2.0;
    rls.params_mut()[(2, 1)] = 2.0;

    let a = SVector::<f32, 3>::new(0.03428682, 0.13472362, -0.16932012);
    let y = SVector::<f32, 2>::new(0.33754053, -0.16460538);

    let stats = rls.update(&a, &y);
    assert_eq!(stats.samples, 1);

    // Covariance must stay symmetric and positive definite
    let p = rls.covariance();
    for i in 0..3 {
        for j in 0..3 {
            let diff = (p[(i, j)] - p[(j, i)]).abs();
            assert!(
                diff < 1e-3,
                "P not symmetric: P[{},{}]={}, P[{},{}]={}",
                i,
                j,
                p[(i, j)],
                j,
                i,
                p[(j, i)]
            );
        }
        assert!(p[(i, i)] > 0.0, "P diagonal non-positive");
    }

    // Covariance trace must be finite and reasonable
    let trace: f32 = (0..3).map(|i| p[(i, i)]).sum();
    assert!(trace.is_finite(), "P trace not finite");
    assert!(trace > 0.0, "P trace not positive");

    // Parameters must have moved
    assert!((rls.params()[(0, 0)] - 1.0).abs() > 1e-6);
}

/// Verify convergence: RlsParallel learns a known linear model.
#[test]
fn golden_parallel_convergence() {
    let mut rls = RlsParallel::<2, 1>::new(1e2, 0.995, CovarianceGuards::default());

    // True model: y = 3*a[0] + (-2)*a[1]
    let true_params = [3.0f32, -2.0];

    let mut rng: u32 = 12345;
    let mut next_f32 = || -> f32 {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        (rng as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };

    for _ in 0..200 {
        let a0 = next_f32();
        let a1 = next_f32();
        let y_true = true_params[0] * a0 + true_params[1] * a1;
        let noise = next_f32() * 0.01;
        let a = SVector::<f32, 2>::new(a0, a1);
        let y = SVector::<f32, 1>::new(y_true + noise);
        rls.update(&a, &y);
    }

    let est = rls.params();
    let err0 = (est[(0, 0)] - true_params[0]).abs();
    let err1 = (est[(1, 0)] - true_params[1]).abs();
    assert!(
        err0 < 0.1,
        "param[0]={:.4} not close to {}",
        est[(0, 0)],
        true_params[0]
    );
    assert!(
        err1 < 0.1,
        "param[1]={:.4} not close to {}",
        est[(1, 0)],
        true_params[1]
    );
}

/// Verify convergence: InverseQrRls learns the same model.
#[test]
fn golden_inverse_qr_convergence() {
    use flight_solver::rls::InverseQrRls;

    let mut rls = InverseQrRls::<2, 1>::new(1e2, 0.995);
    let true_params = [3.0f32, -2.0];

    let mut rng: u32 = 12345;
    let mut next_f32 = || -> f32 {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        (rng as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };

    for _ in 0..200 {
        let a0 = next_f32();
        let a1 = next_f32();
        let y_true = true_params[0] * a0 + true_params[1] * a1;
        let noise = next_f32() * 0.01;
        let a = SVector::<f32, 2>::new(a0, a1);
        let y = SVector::<f32, 1>::new(y_true + noise);
        rls.update(&a, &y);
    }

    let est = rls.params();
    let err0 = (est[(0, 0)] - true_params[0]).abs();
    let err1 = (est[(1, 0)] - true_params[1]).abs();
    assert!(
        err0 < 0.1,
        "param[0]={:.4} not close to {}",
        est[(0, 0)],
        true_params[0]
    );
    assert!(
        err1 < 0.1,
        "param[1]={:.4} not close to {}",
        est[(1, 0)],
        true_params[1]
    );
}
