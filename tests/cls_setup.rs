#[path = "cls_common/mod.rs"]
mod common;

use common::{mat, max_abs_diff, vec_from};
use flight_solver::cls::setup::ls::{setup_a as setup_a_unreg, setup_b as setup_b_unreg};
use flight_solver::cls::setup::wls::{setup_a, setup_b};
use flight_solver::cls::{solve, solve_cls, ExitCode};
use nalgebra::SMatrix;

const TOL: f32 = 1e-4;

// ---------------------------------------------------------------------------
// Shared test geometry: NU=4, NV=4 quad
// ---------------------------------------------------------------------------

#[rustfmt::skip]
const G_4X4: [[f32; 4]; 4] = [
    [-8.0,  0.5,  0.5,  0.1],
    [-8.0, -0.5,  0.5, -0.1],
    [-8.0,  0.5, -0.5, -0.1],
    [-8.0, -0.5, -0.5,  0.1],
];
const WV4: [f32; 4] = [10.0, 10.0, 10.0, 1.0];
const LB4: [f32; 4] = [0.0; 4];
const UB4: [f32; 4] = [1.0; 4];

// ---------------------------------------------------------------------------
// 1. Analytical: square unconstrained → u = G⁻¹·v
// ---------------------------------------------------------------------------

#[test]
fn square_unconstrained_matches_inverse() {
    // Pick v such that u = G⁻¹·v is well inside [0, 1].
    // Pre-computed G⁻¹ · v for v = [-4.0, 0.0, 0.0, 0.0]:
    //   G·u = v ⇒ all rows of G dot u = v.
    //   With G[:,0] = [-8,-8,-8,-8], v = [-4,0,0,0]:
    //     u = [0.5, 0, 0, 0] gives G·u = [-4,-4,-4,-4] ≠ v.
    //   Use nalgebra to invert at test time instead.
    let g = mat(G_4X4);
    let g_inv = g.try_inverse().expect("G must be invertible");

    // Choose v so that u = G⁻¹·v ∈ (0.05, 0.95) — comfortably inside bounds.
    // v = G · [0.5, 0.5, 0.5, 0.5] ensures the solution is centered.
    let u_mid = vec_from([0.5f32; 4]);
    let v_vec = g * u_mid;
    let v_arr = [v_vec[0], v_vec[1], v_vec[2], v_vec[3]];
    let v = vec_from(v_arr);
    let u_expected = g_inv * v;

    // Verify expected solution is inside bounds (test sanity check).
    for i in 0..4 {
        assert!(
            u_expected[i] > LB4[i] + 0.01 && u_expected[i] < UB4[i] - 0.01,
            "u_expected[{i}] = {} not inside bounds — pick a different v",
            u_expected[i]
        );
    }

    let a = setup_a_unreg::<4, 4>(&g, &vec_from(WV4));
    let b = setup_b_unreg(&vec_from(v_arr), &vec_from(WV4));

    let mut us = vec_from([0.5; 4]);
    let mut ws = [0i8; 4];
    let stats = solve_cls::<4, 4>(
        &a,
        &b,
        &vec_from(LB4),
        &vec_from(UB4),
        &mut us,
        &mut ws,
        100,
    );

    assert_eq!(
        stats.exit_code,
        ExitCode::Success,
        "solver did not converge"
    );
    let diff = max_abs_diff(&us.data.0[0], &u_expected.data.0[0]);
    assert!(diff < TOL, "max_diff={diff:.6e}, expected < {TOL}");
}

// ---------------------------------------------------------------------------
// 2. Analytical: identity effectiveness
// ---------------------------------------------------------------------------

#[test]
fn identity_effectiveness_unconstrained() {
    // G = I, Wv = I → A = I, b = v → u = v (trivially).
    let g = mat([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let wv = vec_from([1.0; 3]);
    let v_arr = [0.3f32, 0.5, 0.7];

    let a = setup_a_unreg::<3, 3>(&g, &wv);
    let b = setup_b_unreg(&vec_from(v_arr), &wv);

    let mut us = vec_from([0.0; 3]);
    let mut ws = [0i8; 3];
    let stats = solve_cls::<3, 3>(
        &a,
        &b,
        &vec_from([0.0; 3]),
        &vec_from([1.0; 3]),
        &mut us,
        &mut ws,
        100,
    );

    assert_eq!(stats.exit_code, ExitCode::Success);
    for i in 0..3 {
        assert!(
            (us[i] - v_arr[i]).abs() < TOL,
            "us[{i}]={}, expected {}",
            us[i],
            v_arr[i]
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Wv weighting is respected
// ---------------------------------------------------------------------------

#[test]
fn wv_weighting_changes_solution() {
    // Overdetermined (NV=3, NU=2): two different Wv should give different u.
    #[rustfmt::skip]
    let g: SMatrix<f32, 3, 2> = mat([
        [1.0, 0.0, 0.5],
        [0.0, 1.0, 0.5],
    ]);
    let v_arr = [1.0f32, 1.0, 0.0];
    let lb = vec_from([0.0; 2]);
    let ub = vec_from([2.0; 2]);

    let wv_uniform = vec_from([1.0, 1.0, 1.0]);
    let a1 = setup_a_unreg::<2, 3>(&g, &wv_uniform);
    let b1 = setup_b_unreg(&vec_from(v_arr), &wv_uniform);
    let mut us1 = vec_from([0.5; 2]);
    let mut ws1 = [0i8; 2];
    let s1 = solve_cls::<2, 3>(&a1, &b1, &lb, &ub, &mut us1, &mut ws1, 100);
    assert_eq!(s1.exit_code, ExitCode::Success);

    // Heavily weight third row (the 0.5·u1 + 0.5·u2 = 0 constraint).
    let wv_heavy = vec_from([1.0, 1.0, 100.0]);
    let a2 = setup_a_unreg::<2, 3>(&g, &wv_heavy);
    let b2 = setup_b_unreg(&vec_from(v_arr), &wv_heavy);
    let mut us2 = vec_from([0.5; 2]);
    let mut ws2 = [0i8; 2];
    let s2 = solve_cls::<2, 3>(&a2, &b2, &lb, &ub, &mut us2, &mut ws2, 100);
    assert_eq!(s2.exit_code, ExitCode::Success);

    // Solutions must differ (heavy Wv pulls u toward satisfying row 3).
    let diff = max_abs_diff(&us1.data.0[0], &us2.data.0[0]);
    assert!(
        diff > 0.01,
        "Wv weighting should change solution, diff={diff:.6e}"
    );
}

// ---------------------------------------------------------------------------
// 4. Constraint handling — single motor saturates
// ---------------------------------------------------------------------------

#[test]
fn single_motor_saturates() {
    let g = mat(G_4X4);
    let g_inv = g.try_inverse().unwrap();
    // Pick v so unconstrained solution has at least one component outside [0,1].
    let v_arr = [-12.0f32, 1.0, -1.0, 0.5];
    let u_unc = g_inv * vec_from(v_arr);
    assert!(
        (0..4).any(|i| u_unc[i] < LB4[i] || u_unc[i] > UB4[i]),
        "test setup: v should cause saturation"
    );
    let a = setup_a_unreg::<4, 4>(&g, &vec_from(WV4));
    let b = setup_b_unreg(&vec_from(v_arr), &vec_from(WV4));

    let mut us = vec_from([0.5; 4]);
    let mut ws = [0i8; 4];
    let stats = solve_cls::<4, 4>(
        &a,
        &b,
        &vec_from(LB4),
        &vec_from(UB4),
        &mut us,
        &mut ws,
        100,
    );

    assert_eq!(stats.exit_code, ExitCode::Success);
    for i in 0..4 {
        assert!(
            us[i] >= LB4[i] - TOL && us[i] <= UB4[i] + TOL,
            "us[{i}]={} outside bounds",
            us[i]
        );
    }
    // At least one motor should be at or very near a bound.
    let at_bound = (0..4)
        .filter(|&i| (us[i] - UB4[i]).abs() < TOL || (us[i] - LB4[i]).abs() < TOL)
        .count();
    assert!(at_bound >= 1, "expected at least 1 saturated motor");
}

// ---------------------------------------------------------------------------
// 5. Full saturation — all motors clamp
// ---------------------------------------------------------------------------

#[test]
fn full_saturation_all_at_bounds() {
    let g = mat(G_4X4);
    // Extreme demand — impossible to satisfy within [0,1].
    let v_arr = [-500.0f32, 50.0, -50.0, 20.0];
    let a = setup_a_unreg::<4, 4>(&g, &vec_from(WV4));
    let b = setup_b_unreg(&vec_from(v_arr), &vec_from(WV4));

    let mut us = vec_from([0.5; 4]);
    let mut ws = [0i8; 4];
    let stats = solve_cls::<4, 4>(
        &a,
        &b,
        &vec_from(LB4),
        &vec_from(UB4),
        &mut us,
        &mut ws,
        100,
    );

    assert_eq!(stats.exit_code, ExitCode::Success);
    for i in 0..4 {
        assert!(
            us[i] >= LB4[i] - TOL && us[i] <= UB4[i] + TOL,
            "us[{i}]={} outside bounds",
            us[i]
        );
    }
}

// ---------------------------------------------------------------------------
// 6. Zero demand → solution near zero (lower bound)
// ---------------------------------------------------------------------------

#[test]
fn zero_demand() {
    let g = mat(G_4X4);
    let v_arr = [0.0f32; 4];
    let a = setup_a_unreg::<4, 4>(&g, &vec_from(WV4));
    let b = setup_b_unreg(&vec_from(v_arr), &vec_from(WV4));

    let mut us = vec_from([0.5; 4]);
    let mut ws = [0i8; 4];
    let stats = solve_cls::<4, 4>(
        &a,
        &b,
        &vec_from(LB4),
        &vec_from(UB4),
        &mut us,
        &mut ws,
        100,
    );

    assert_eq!(stats.exit_code, ExitCode::Success);
    // G⁻¹·0 = 0, which is on the lower bound. Solver should return ~0.
    for i in 0..4 {
        assert!(
            us[i].abs() < TOL,
            "zero demand: us[{i}]={}, expected ~0",
            us[i]
        );
    }
}

// ---------------------------------------------------------------------------
// 7. Equivalence with regularised path (small gamma)
// ---------------------------------------------------------------------------

#[test]
fn small_gamma_approx_unreg() {
    let g = mat(G_4X4);
    let v_arr = [-4.0f32, 0.1, -0.1, 0.02];
    let wv = vec_from(WV4);

    // Unregularised solve.
    let a_unreg = setup_a_unreg::<4, 4>(&g, &wv);
    let b_unreg = setup_b_unreg(&vec_from(v_arr), &wv);
    let mut us_unreg = vec_from([0.5; 4]);
    let mut ws_unreg = [0i8; 4];
    let su = solve_cls::<4, 4>(
        &a_unreg,
        &b_unreg,
        &vec_from(LB4),
        &vec_from(UB4),
        &mut us_unreg,
        &mut ws_unreg,
        100,
    );
    assert_eq!(su.exit_code, ExitCode::Success, "unreg did not converge");

    // Regularised with very small theta (drives gamma toward zero).
    let mut wu = vec_from([1.0; 4]);
    let (a_reg, gamma) = setup_a::<4, 4, 8>(&g, &wv, &mut wu, 1e-15, 0.0);
    let b_reg = setup_b::<4, 4, 8>(&vec_from(v_arr), &vec_from([0.5; 4]), &wv, &wu, gamma);
    let mut us_reg = vec_from([0.5; 4]);
    let mut ws_reg = [0i8; 4];
    let sr = solve::<4, 4, 8>(
        &a_reg,
        &b_reg,
        &vec_from(LB4),
        &vec_from(UB4),
        &mut us_reg,
        &mut ws_reg,
        100,
    );
    assert_eq!(sr.exit_code, ExitCode::Success, "reg did not converge");

    // Should be very close (gamma is tiny, regularisation effect negligible).
    let diff = max_abs_diff(&us_unreg.data.0[0], &us_reg.data.0[0]);
    // Relax tolerance — gamma is small but nonzero, so slight difference is expected.
    assert!(
        diff < 1e-2,
        "unreg vs small-gamma reg: max_diff={diff:.6e}, expected < 1e-2"
    );
}

// ---------------------------------------------------------------------------
// 8. Overdetermined: NV=6, NU=4 (no regularisation)
// ---------------------------------------------------------------------------

#[rustfmt::skip]
const G_6X4: [[f32; 6]; 4] = [
    [-1.0, 0.5,  0.5,  0.1,  0.02, 0.02],
    [-1.0,-0.5,  0.5, -0.1, -0.02, 0.02],
    [-1.0, 0.5, -0.5, -0.1,  0.02,-0.02],
    [-1.0,-0.5, -0.5,  0.1, -0.02,-0.02],
];
const WV6: [f32; 6] = [10.0, 10.0, 10.0, 1.0, 0.5, 0.5];

#[test]
fn overdetermined_nominal() {
    let g = mat(G_6X4);
    let v_arr = [-2.0f32, 0.1, -0.1, 0.02, 0.0, 0.0];
    let wv = vec_from(WV6);
    let a = setup_a_unreg::<4, 6>(&g, &wv);
    let b = setup_b_unreg(&vec_from(v_arr), &wv);

    let mut us = vec_from([0.5; 4]);
    let mut ws = [0i8; 4];
    let stats = solve_cls::<4, 6>(
        &a,
        &b,
        &vec_from(LB4),
        &vec_from(UB4),
        &mut us,
        &mut ws,
        100,
    );

    assert_eq!(stats.exit_code, ExitCode::Success);
    for i in 0..4 {
        assert!(
            us[i] >= LB4[i] - TOL && us[i] <= UB4[i] + TOL,
            "us[{i}]={} outside bounds",
            us[i]
        );
    }
}

#[test]
fn overdetermined_saturating() {
    let g = mat(G_6X4);
    let v_arr = [-100.0f32, 10.0, -10.0, 5.0, 1.0, -1.0];
    let wv = vec_from(WV6);
    let a = setup_a_unreg::<4, 6>(&g, &wv);
    let b = setup_b_unreg(&vec_from(v_arr), &wv);

    let mut us = vec_from([0.5; 4]);
    let mut ws = [0i8; 4];
    let stats = solve_cls::<4, 6>(
        &a,
        &b,
        &vec_from(LB4),
        &vec_from(UB4),
        &mut us,
        &mut ws,
        100,
    );

    assert_eq!(stats.exit_code, ExitCode::Success);
    for i in 0..4 {
        assert!(
            us[i] >= LB4[i] - TOL && us[i] <= UB4[i] + TOL,
            "us[{i}]={} outside bounds",
            us[i]
        );
    }
}

// ---------------------------------------------------------------------------
// 9. Warmstart: re-solve from previous solution in 0–1 iterations
// ---------------------------------------------------------------------------

#[test]
fn warmstart_converges_immediately() {
    let g = mat(G_4X4);
    let v_arr = [-4.0f32, 0.1, -0.1, 0.02];
    let wv = vec_from(WV4);
    let a = setup_a_unreg::<4, 4>(&g, &wv);
    let b = setup_b_unreg(&vec_from(v_arr), &wv);
    let lb = vec_from(LB4);
    let ub = vec_from(UB4);

    // Cold start.
    let mut us = vec_from([0.5; 4]);
    let mut ws = [0i8; 4];
    let s1 = solve_cls::<4, 4>(&a, &b, &lb, &ub, &mut us, &mut ws, 100);
    assert_eq!(s1.exit_code, ExitCode::Success);
    let us_ref = us;

    // Warmstart with converged solution — should exit in 1 iteration.
    let s2 = solve_cls::<4, 4>(&a, &b, &lb, &ub, &mut us, &mut ws, 100);
    assert_eq!(s2.exit_code, ExitCode::Success);
    assert!(
        s2.iterations <= 1,
        "warmstart took {} iterations, expected ≤ 1",
        s2.iterations
    );
    let diff = max_abs_diff(&us.data.0[0], &us_ref.data.0[0]);
    assert!(diff < TOL, "warmstart changed solution: diff={diff:.6e}");
}

// ---------------------------------------------------------------------------
// 10. Negative bounds (bidirectional actuators)
// ---------------------------------------------------------------------------

#[test]
fn negative_bounds() {
    let g = mat(G_4X4);
    let v_arr = [0.0f32, 0.3, -0.2, 0.05];
    let wv = vec_from(WV4);
    let a = setup_a_unreg::<4, 4>(&g, &wv);
    let b = setup_b_unreg(&vec_from(v_arr), &wv);

    let lb = vec_from([-1.0; 4]);
    let ub = vec_from([1.0; 4]);

    let mut us = vec_from([0.0; 4]);
    let mut ws = [0i8; 4];
    let stats = solve_cls::<4, 4>(&a, &b, &lb, &ub, &mut us, &mut ws, 100);

    assert_eq!(stats.exit_code, ExitCode::Success);
    for i in 0..4 {
        assert!(
            us[i] >= -1.0 - TOL && us[i] <= 1.0 + TOL,
            "us[{i}]={} outside [-1, 1]",
            us[i]
        );
    }

    // With zero thrust demand, the solution should also be verifiable via G⁻¹.
    let g_mat = mat(G_4X4);
    let g_inv = g_mat.try_inverse().unwrap();
    let u_expected = g_inv * vec_from(v_arr);
    // If unconstrained solution is inside bounds, solver should match it.
    let all_inside = (0..4).all(|i| u_expected[i] > -1.0 + 0.01 && u_expected[i] < 1.0 - 0.01);
    if all_inside {
        let diff = max_abs_diff(&us.data.0[0], &u_expected.data.0[0]);
        assert!(diff < TOL, "neg bounds analytical: diff={diff:.6e}");
    }
}

// ---------------------------------------------------------------------------
// 11. Asymmetric bounds
// ---------------------------------------------------------------------------

#[test]
fn asymmetric_bounds() {
    let g = mat(G_4X4);
    let v_arr = [-4.0f32, 0.1, -0.1, 0.02];
    let wv = vec_from(WV4);
    let a = setup_a_unreg::<4, 4>(&g, &wv);
    let b = setup_b_unreg(&vec_from(v_arr), &wv);

    // Tighter upper bound on motors 0 and 1.
    let lb = vec_from([0.0; 4]);
    let ub = vec_from([0.6, 0.6, 1.0, 1.0]);

    let mut us = vec_from([0.3; 4]);
    let mut ws = [0i8; 4];
    let stats = solve_cls::<4, 4>(&a, &b, &lb, &ub, &mut us, &mut ws, 100);

    assert_eq!(stats.exit_code, ExitCode::Success);
    for i in 0..4 {
        assert!(
            us[i] >= lb[i] - TOL && us[i] <= ub[i] + TOL,
            "us[{i}]={} outside [{}, {}]",
            us[i],
            lb[i],
            ub[i]
        );
    }
}

// ---------------------------------------------------------------------------
// 12. Randomised NU=4, NV=4 unreg (50 cases, xorshift32)
// ---------------------------------------------------------------------------

struct Rng(u32);
impl Rng {
    fn new(seed: u32) -> Self {
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() & 0x7FFFFF) as f32 / 0x7FFFFF as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }
}

#[test]
fn random_4x4_unreg_50() {
    let mut rng = Rng::new(0xDEAD);
    for t in 0..50 {
        // Random G.
        let mut g_data = [[0.0f32; 4]; 4];
        for col in &mut g_data {
            for x in col.iter_mut() {
                *x = rng.range(-10.0, 10.0);
            }
        }
        let g = mat(g_data);

        // Skip if G is singular (det ≈ 0).
        if g.determinant().abs() < 1e-3 {
            continue;
        }

        let wv = vec_from([
            rng.range(1.0, 30.0),
            rng.range(1.0, 30.0),
            rng.range(1.0, 30.0),
            rng.range(0.5, 5.0),
        ]);
        let v = vec_from([
            rng.range(-20.0, 20.0),
            rng.range(-5.0, 5.0),
            rng.range(-5.0, 5.0),
            rng.range(-1.0, 1.0),
        ]);

        let a = setup_a_unreg::<4, 4>(&g, &wv);
        let b = setup_b_unreg(&v, &wv);

        let lb = vec_from([0.0; 4]);
        let ub = vec_from([1.0; 4]);
        let mut us = vec_from([0.5; 4]);
        let mut ws = [0i8; 4];
        let stats = solve_cls::<4, 4>(&a, &b, &lb, &ub, &mut us, &mut ws, 100);

        assert_eq!(
            stats.exit_code,
            ExitCode::Success,
            "rng_unreg_{t}: did not converge"
        );
        for i in 0..4 {
            assert!(
                us[i] >= -TOL && us[i] <= 1.0 + TOL,
                "rng_unreg_{t}: us[{i}]={} outside bounds",
                us[i]
            );
        }

        // If unconstrained solution is feasible, verify it matches G⁻¹·v.
        let g_inv = g.try_inverse().unwrap();
        let u_exact = g_inv * v;
        let all_inside = (0..4).all(|i| u_exact[i] > TOL && u_exact[i] < 1.0 - TOL);
        if all_inside {
            let diff = max_abs_diff(&us.data.0[0], &u_exact.data.0[0]);
            assert!(diff < TOL, "rng_unreg_{t}: feasible case diff={diff:.6e}");
        }
    }
}

// ---------------------------------------------------------------------------
// 13. Randomised overdetermined NV=6, NU=4 unreg (50 cases)
// ---------------------------------------------------------------------------

#[test]
fn random_6x4_unreg_50() {
    let mut rng = Rng::new(0xBEEF);
    for t in 0..50 {
        let mut g_data = [[0.0f32; 6]; 4];
        for col in &mut g_data {
            for x in col.iter_mut() {
                *x = rng.range(-10.0, 10.0);
            }
        }
        let g = mat(g_data);
        let wv = vec_from([
            rng.range(1.0, 30.0),
            rng.range(1.0, 30.0),
            rng.range(1.0, 30.0),
            rng.range(0.5, 5.0),
            rng.range(0.5, 2.0),
            rng.range(0.5, 2.0),
        ]);
        let v = vec_from([
            rng.range(-20.0, 20.0),
            rng.range(-5.0, 5.0),
            rng.range(-5.0, 5.0),
            rng.range(-1.0, 1.0),
            rng.range(-0.5, 0.5),
            rng.range(-0.5, 0.5),
        ]);

        let a = setup_a_unreg::<4, 6>(&g, &wv);
        let b = setup_b_unreg(&v, &wv);

        let lb = vec_from([0.0; 4]);
        let ub = vec_from([1.0; 4]);
        let mut us = vec_from([0.5; 4]);
        let mut ws = [0i8; 4];
        let stats = solve_cls::<4, 6>(&a, &b, &lb, &ub, &mut us, &mut ws, 100);

        assert_eq!(
            stats.exit_code,
            ExitCode::Success,
            "rng_od_{t}: did not converge"
        );
        for i in 0..4 {
            assert!(
                us[i] >= -TOL && us[i] <= 1.0 + TOL,
                "rng_od_{t}: us[{i}]={} outside bounds",
                us[i]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 14. solve_cls produces same result as solve for regularised problems
// ---------------------------------------------------------------------------

#[test]
fn solve_cls_matches_solve_regularised() {
    let g = mat(G_4X4);
    let wv = vec_from(WV4);
    let mut wu = vec_from([1.0; 4]);
    let (a, gamma) = setup_a::<4, 4, 8>(&g, &wv, &mut wu, 2e-9, 4e5);
    let b = setup_b::<4, 4, 8>(
        &vec_from([-4.0, 0.1, -0.1, 0.02]),
        &vec_from([0.5; 4]),
        &wv,
        &wu,
        gamma,
    );
    let lb = vec_from(LB4);
    let ub = vec_from(UB4);

    let mut us_solve = vec_from([0.5; 4]);
    let mut ws_solve = [0i8; 4];
    let s1 = solve::<4, 4, 8>(&a, &b, &lb, &ub, &mut us_solve, &mut ws_solve, 100);

    let mut us_cls = vec_from([0.5; 4]);
    let mut ws_cls = [0i8; 4];
    let s2 = solve_cls::<4, 8>(&a, &b, &lb, &ub, &mut us_cls, &mut ws_cls, 100);

    assert_eq!(s1.exit_code, ExitCode::Success);
    assert_eq!(s2.exit_code, ExitCode::Success);
    let diff = max_abs_diff(&us_solve.data.0[0], &us_cls.data.0[0]);
    assert!(
        diff < 1e-6,
        "solve vs solve_cls: diff={diff:.6e}, expected identical"
    );
    assert_eq!(s1.iterations, s2.iterations);
}
