#[path = "cls_common/mod.rs"]
mod common;

use common::naive::solve_naive;
use common::{mat, max_abs_diff, vec_from};
use flight_solver::cls::setup::wls::{setup_a, setup_b};
use flight_solver::cls::{solve, ExitCode};
use nalgebra::{SMatrix, SVector};

const TOL: f32 = 1e-4;

// ---------------------------------------------------------------------------
// Shared test problem: NU=4, NV=4
// ---------------------------------------------------------------------------

#[rustfmt::skip]
const EDGE_G: [[f32; 4]; 4] = [
    [-8.0,  0.5,  0.5,  0.1],
    [-8.0, -0.5,  0.5, -0.1],
    [-8.0,  0.5, -0.5, -0.1],
    [-8.0, -0.5, -0.5,  0.1],
];
const EDGE_WV: [f32; 4] = [10.0, 10.0, 10.0, 1.0];
const EDGE_WU: [f32; 4] = [1.0; 4];
const EDGE_V: [f32; 4] = [-4.0, 0.3, -0.2, 0.05];
const EDGE_LB: [f32; 4] = [0.0; 4];
const EDGE_UB: [f32; 4] = [1.0; 4];
const EDGE_UP: [f32; 4] = [0.5; 4];

fn edge_ab(v: [f32; 4]) -> (SMatrix<f32, 8, 4>, SVector<f32, 8>) {
    let mut wu = vec_from(EDGE_WU);
    let (a, gamma) = setup_a::<4, 4, 8>(&mat(EDGE_G), &vec_from(EDGE_WV), &mut wu, 2.0e-9, 4e5);
    let b = setup_b::<4, 4, 8>(
        &vec_from(v),
        &vec_from(EDGE_UP),
        &vec_from(EDGE_WV),
        &wu,
        gamma,
    );
    (a, b)
}

/// Run both solvers on a NU=4 problem, assert they agree within TOL.
fn assert_solvers_agree_4(
    a: &SMatrix<f32, 8, 4>,
    b: &SVector<f32, 8>,
    lb: &SVector<f32, 4>,
    ub: &SVector<f32, 4>,
    u0: [f32; 4],
    ws0: [i8; 4],
    label: &str,
) {
    let mut us_n = vec_from(u0);
    let mut ws_n = ws0;
    let sn = solve_naive::<4, 4, 8>(a, b, lb, ub, &mut us_n, &mut ws_n, 100);

    let mut us_i = vec_from(u0);
    let mut ws_i = ws0;
    let si = solve::<4, 4, 8>(a, b, lb, ub, &mut us_i, &mut ws_i, 100);

    assert_eq!(
        sn.exit_code,
        ExitCode::Success,
        "{label}: naive did not converge"
    );
    assert_eq!(
        si.exit_code,
        ExitCode::Success,
        "{label}: incr did not converge"
    );
    let diff = max_abs_diff(&us_n.data.0[0], &us_i.data.0[0]);
    assert!(diff < TOL, "{label}: max_diff={diff:.6e}");
}

// ---------------------------------------------------------------------------
// Basic edge cases (NU=4, NV=4)
// ---------------------------------------------------------------------------

#[test]
fn all_free_cold_start() {
    let (a, b) = edge_ab(EDGE_V);
    assert_solvers_agree_4(
        &a,
        &b,
        &vec_from(EDGE_LB),
        &vec_from(EDGE_UB),
        [0.5; 4],
        [0; 4],
        "all_free_cold",
    );
}

#[test]
fn zero_demand() {
    let (a, b) = edge_ab([0.0; 4]);
    assert_solvers_agree_4(
        &a,
        &b,
        &vec_from(EDGE_LB),
        &vec_from(EDGE_UB),
        [0.5; 4],
        [0; 4],
        "zero_demand",
    );
}

#[test]
fn large_demand() {
    let (a, b) = edge_ab([-100.0, 10.0, -10.0, 5.0]);
    let lb = vec_from(EDGE_LB);
    let ub = vec_from(EDGE_UB);

    let mut us_n = vec_from([0.5; 4]);
    let mut ws_n = [0i8; 4];
    let sn = solve_naive::<4, 4, 8>(&a, &b, &lb, &ub, &mut us_n, &mut ws_n, 100);

    let mut us_i = vec_from([0.5; 4]);
    let mut ws_i = [0i8; 4];
    let si = solve::<4, 4, 8>(&a, &b, &lb, &ub, &mut us_i, &mut ws_i, 100);

    assert_eq!(sn.exit_code, ExitCode::Success);
    assert_eq!(si.exit_code, ExitCode::Success);
    assert!(max_abs_diff(&us_n.data.0[0], &us_i.data.0[0]) < TOL);
    for i in 0..4 {
        assert!(us_n.data.0[0][i] >= EDGE_LB[i] - 1e-6);
        assert!(us_n.data.0[0][i] <= EDGE_UB[i] + 1e-6);
    }
}

#[test]
fn negative_g_entries() {
    #[rustfmt::skip]
    let g: SMatrix<f32, 4, 4> = mat([
        [ 8.0,  0.5,  0.5,  0.1], [-8.0, -0.5,  0.5, -0.1],
        [ 8.0,  0.5, -0.5, -0.1], [-8.0, -0.5, -0.5,  0.1],
    ]);
    let wv = vec_from([10.0, 10.0, 10.0, 1.0]);
    let mut wu = vec_from([1.0; 4]);
    let (a, gamma) = setup_a::<4, 4, 8>(&g, &wv, &mut wu, 2.0e-9, 4e5);
    let b = setup_b::<4, 4, 8>(
        &vec_from([-4.0, 0.3, -0.2, 0.05]),
        &vec_from([0.0; 4]),
        &wv,
        &wu,
        gamma,
    );
    assert_solvers_agree_4(
        &a,
        &b,
        &vec_from([-1.0; 4]),
        &vec_from([1.0; 4]),
        [0.0; 4],
        [0; 4],
        "neg_g",
    );
}

#[test]
fn single_free_warmstart() {
    let (a, b) = edge_ab(EDGE_V);
    assert_solvers_agree_4(
        &a,
        &b,
        &vec_from(EDGE_LB),
        &vec_from(EDGE_UB),
        [0.0, 0.0, 0.5, 0.0],
        [-1, -1, 0, 1],
        "single_free",
    );
}

// ---------------------------------------------------------------------------
// Warmstart correctness
// ---------------------------------------------------------------------------

#[test]
fn warmstart_matches_cold_start() {
    let (a, b) = edge_ab(EDGE_V);
    let lb = vec_from(EDGE_LB);
    let ub = vec_from(EDGE_UB);

    let mut us_cold = vec_from([0.5; 4]);
    let mut ws_cold = [0i8; 4];
    let sc = solve_naive::<4, 4, 8>(&a, &b, &lb, &ub, &mut us_cold, &mut ws_cold, 100);
    assert_eq!(sc.exit_code, ExitCode::Success);

    let mut us_wn = us_cold;
    let mut ws_wn = ws_cold;
    let swn = solve_naive::<4, 4, 8>(&a, &b, &lb, &ub, &mut us_wn, &mut ws_wn, 1);
    assert_eq!(swn.exit_code, ExitCode::Success, "naive warmstart failed");

    let mut us_wi = us_cold;
    let mut ws_wi = ws_cold;
    let swi = solve::<4, 4, 8>(&a, &b, &lb, &ub, &mut us_wi, &mut ws_wi, 1);
    assert_eq!(swi.exit_code, ExitCode::Success, "incr warmstart failed");

    assert!(max_abs_diff(&us_wn.data.0[0], &us_cold.data.0[0]) < TOL);
    assert!(max_abs_diff(&us_wi.data.0[0], &us_cold.data.0[0]) < TOL);
}

// ---------------------------------------------------------------------------
// Numerical stress tests
// ---------------------------------------------------------------------------

#[test]
fn high_condition_number() {
    #[rustfmt::skip]
    let g: SMatrix<f32, 4, 4> = mat([
        [-8.0,  0.5,  0.5,  0.1], [-8.0, -0.5,  0.5, -0.1],
        [-8.0,  0.5, -0.5, -0.1], [-8.0, -0.5, -0.5,  0.1],
    ]);
    let wv = vec_from([1.0, 1.0, 100.0, 100.0]);
    let mut wu = vec_from([1.0; 4]);
    let (a, gamma) = setup_a::<4, 4, 8>(&g, &wv, &mut wu, 2.0e-9, 4e5);
    let b = setup_b::<4, 4, 8>(
        &vec_from([-4.0, 0.3, -0.2, 0.05]),
        &vec_from([0.5; 4]),
        &wv,
        &wu,
        gamma,
    );
    assert_solvers_agree_4(
        &a,
        &b,
        &vec_from([0.0; 4]),
        &vec_from([1.0; 4]),
        [0.5; 4],
        [0; 4],
        "high_cond",
    );
}

#[test]
fn near_zero_effectiveness_column() {
    #[rustfmt::skip]
    let g: SMatrix<f32, 4, 4> = mat([
        [-8.0, 0.5, 0.5, 0.1], [-8.0, -0.5, 0.5, -0.1],
        [1e-7, 1e-7, 1e-7, 1e-7], [-8.0, -0.5, -0.5, 0.1],
    ]);
    let mut wu = vec_from([1.0; 4]);
    let (a, gamma) = setup_a::<4, 4, 8>(&g, &vec_from(EDGE_WV), &mut wu, 2.0e-9, 4e5);
    let b = setup_b::<4, 4, 8>(
        &vec_from(EDGE_V),
        &vec_from(EDGE_UP),
        &vec_from(EDGE_WV),
        &wu,
        gamma,
    );
    assert_solvers_agree_4(
        &a,
        &b,
        &vec_from([0.0; 4]),
        &vec_from([1.0; 4]),
        [0.5; 4],
        [0; 4],
        "near_zero_col",
    );
}

#[test]
fn wide_range_g_entries() {
    #[rustfmt::skip]
    let g: SMatrix<f32, 4, 4> = mat([
        [-8000.0, 0.5, 0.5, 0.0001], [-8000.0, -0.5, 0.5, -0.0001],
        [-0.001, 500.0, -500.0, -0.0001], [-0.001, -500.0, -500.0, 100.0],
    ]);
    let mut wu = vec_from([1.0; 4]);
    let (a, gamma) = setup_a::<4, 4, 8>(&g, &vec_from([1.0; 4]), &mut wu, 2.0e-9, 4e5);
    let b = setup_b::<4, 4, 8>(
        &vec_from([-4.0, 0.3, -0.2, 0.05]),
        &vec_from([0.5; 4]),
        &vec_from([1.0; 4]),
        &wu,
        gamma,
    );
    assert_solvers_agree_4(
        &a,
        &b,
        &vec_from([0.0; 4]),
        &vec_from([1.0; 4]),
        [0.5; 4],
        [0; 4],
        "wide_range_g",
    );
}

// ---------------------------------------------------------------------------
// NU=4, NV=6 — quad dimensions (4 motors, 6 pseudo-controls)
// ---------------------------------------------------------------------------

fn assert_solvers_agree_4_6(
    a: &SMatrix<f32, 10, 4>,
    b: &SVector<f32, 10>,
    lb: &SVector<f32, 4>,
    ub: &SVector<f32, 4>,
    u0: [f32; 4],
    ws0: [i8; 4],
    label: &str,
) {
    let mut us_n = vec_from(u0);
    let mut ws_n = ws0;
    let sn = solve_naive::<4, 6, 10>(a, b, lb, ub, &mut us_n, &mut ws_n, 100);
    let mut us_i = vec_from(u0);
    let mut ws_i = ws0;
    let si = solve::<4, 6, 10>(a, b, lb, ub, &mut us_i, &mut ws_i, 100);
    assert_eq!(sn.exit_code, ExitCode::Success, "{label}: naive failed");
    assert_eq!(si.exit_code, ExitCode::Success, "{label}: incr failed");
    assert!(
        max_abs_diff(&us_n.data.0[0], &us_i.data.0[0]) < TOL,
        "{label}: solvers disagree"
    );
}

#[rustfmt::skip]
const QUAD_G: [[f32; 6]; 4] = [
    [-1.0, 0.5,  0.5,  0.1,  0.02, 0.02],
    [-1.0,-0.5,  0.5, -0.1, -0.02, 0.02],
    [-1.0, 0.5, -0.5, -0.1,  0.02,-0.02],
    [-1.0,-0.5, -0.5,  0.1, -0.02,-0.02],
];
const QUAD_WV: [f32; 6] = [10.0, 10.0, 10.0, 1.0, 0.5, 0.5];

fn quad_ab(v: [f32; 6]) -> (SMatrix<f32, 10, 4>, SVector<f32, 10>) {
    let mut wu = vec_from([1.0; 4]);
    let (a, gamma) = setup_a::<4, 6, 10>(&mat(QUAD_G), &vec_from(QUAD_WV), &mut wu, 2.0e-9, 4e5);
    let b = setup_b::<4, 6, 10>(
        &vec_from(v),
        &vec_from([0.5; 4]),
        &vec_from(QUAD_WV),
        &wu,
        gamma,
    );
    (a, b)
}

#[test]
fn quad_4_6_nominal() {
    let (a, b) = quad_ab([-4.0, 0.3, -0.2, 0.05, 0.0, 0.0]);
    assert_solvers_agree_4_6(
        &a,
        &b,
        &vec_from([0.0; 4]),
        &vec_from([1.0; 4]),
        [0.5; 4],
        [0; 4],
        "quad_nominal",
    );
}

#[test]
fn quad_4_6_zero_demand() {
    let (a, b) = quad_ab([0.0; 6]);
    assert_solvers_agree_4_6(
        &a,
        &b,
        &vec_from([0.0; 4]),
        &vec_from([1.0; 4]),
        [0.5; 4],
        [0; 4],
        "quad_zero",
    );
}

#[test]
fn quad_4_6_saturating_demand() {
    let (a, b) = quad_ab([-100.0, 10.0, -10.0, 5.0, 1.0, -1.0]);
    assert_solvers_agree_4_6(
        &a,
        &b,
        &vec_from([0.0; 4]),
        &vec_from([1.0; 4]),
        [0.5; 4],
        [0; 4],
        "quad_saturating",
    );
}

#[test]
fn quad_4_6_warmstart() {
    let (a, b) = quad_ab([-4.0, 0.3, -0.2, 0.05, 0.0, 0.0]);
    let lb = vec_from([0.0; 4]);
    let ub = vec_from([1.0; 4]);
    let mut us_ref = vec_from([0.5; 4]);
    let mut ws_ref = [0i8; 4];
    let sr = solve_naive::<4, 6, 10>(&a, &b, &lb, &ub, &mut us_ref, &mut ws_ref, 100);
    assert_eq!(sr.exit_code, ExitCode::Success);
    let mut us_w = us_ref;
    let mut ws_w = ws_ref;
    let sw = solve::<4, 6, 10>(&a, &b, &lb, &ub, &mut us_w, &mut ws_w, 1);
    assert_eq!(sw.exit_code, ExitCode::Success, "quad warmstart failed");
    assert!(max_abs_diff(&us_w.data.0[0], &us_ref.data.0[0]) < TOL);
}

// ---------------------------------------------------------------------------
// Randomised NU=4, NV=4 (xorshift32)
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

fn random_problem_4_4(
    rng: &mut Rng,
) -> (
    SMatrix<f32, 8, 4>,
    SVector<f32, 8>,
    SVector<f32, 4>,
    SVector<f32, 4>,
) {
    let mut g_data = [[0.0f32; 4]; 4];
    for col in &mut g_data {
        for x in col.iter_mut() {
            *x = rng.range(-10.0, 10.0);
        }
    }
    let wv = vec_from([
        rng.range(1.0, 30.0),
        rng.range(1.0, 30.0),
        rng.range(1.0, 30.0),
        rng.range(0.5, 5.0),
    ]);
    let mut wu = vec_from([
        rng.range(0.5, 2.0),
        rng.range(0.5, 2.0),
        rng.range(0.5, 2.0),
        rng.range(0.5, 2.0),
    ]);
    let v = vec_from([
        rng.range(-20.0, 20.0),
        rng.range(-5.0, 5.0),
        rng.range(-5.0, 5.0),
        rng.range(-1.0, 1.0),
    ]);
    let (a, gamma) = setup_a::<4, 4, 8>(&mat(g_data), &wv, &mut wu, 2.0e-9, 4e5);
    let b = setup_b::<4, 4, 8>(&v, &vec_from([0.5; 4]), &wv, &wu, gamma);
    (a, b, vec_from([0.0; 4]), vec_from([1.0; 4]))
}

#[test]
fn random_4x4_50() {
    let mut rng = Rng::new(42);
    for t in 0..50 {
        let (a, b, lb, ub) = random_problem_4_4(&mut rng);
        assert_solvers_agree_4(&a, &b, &lb, &ub, [0.5; 4], [0; 4], &format!("rng4_{t}"));
    }
}
