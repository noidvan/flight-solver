#[path = "wls_regression/c_ref.rs"]
mod c_ref;
#[path = "wls_regression/c_test_cases.rs"]
mod c_test_cases;

#[path = "cls_common/mod.rs"]
mod common;

use common::naive::solve_naive;
use common::{mat, max_abs_diff, vec_from};
use flight_solver::cls::setup::wls::{setup_a, setup_b};
use flight_solver::cls::ExitCode;
use flight_solver::wls::ControlAllocator;
use nalgebra::{SMatrix, SVector};

fn has_nan(a: &[f32]) -> bool {
    a.iter().any(|x| x.is_nan())
}

const SOLVER_TOL: f32 = 1e-4;
const C_REF_TOL: f32 = 1e-4;

fn run_case(idx: usize) -> ([f32; 6], [f32; 6], ExitCode, ExitCode) {
    let tc = &c_test_cases::CASES[idx];

    let g: SMatrix<f32, 4, 6> = mat(tc.jg);
    let wv: SVector<f32, 4> = vec_from(tc.wv);
    let wu: SVector<f32, 6> = vec_from(tc.wu);
    let v: SVector<f32, 4> = vec_from(tc.v);
    let up: SVector<f32, 6> = vec_from(tc.up);
    let lb: SVector<f32, 6> = vec_from(tc.lb);
    let ub: SVector<f32, 6> = vec_from(tc.ub);

    // Naive reference path — still uses the raw building blocks so we can
    // feed solve_naive the same A/b the encapsulated allocator sees.
    let mut wu_raw = wu;
    let (a, gamma) = setup_a::<6, 4, 10>(&g, &wv, &mut wu_raw, 2.0e-9, 4e5);
    let b = setup_b::<6, 4, 10>(&v, &up, &wv, &wu_raw, gamma);

    let mut us_n: SVector<f32, 6> = vec_from(tc.u0);
    let mut ws_n = [0i8; 6];
    let sn = solve_naive::<6, 4, 10>(&a, &b, &lb, &ub, &mut us_n, &mut ws_n, 100);

    // Incremental path — driven through the encapsulated ControlAllocator API.
    let mut alloc = ControlAllocator::<6, 4, 10>::new(&g, &wv, wu, 2.0e-9, 4e5);
    alloc.set_warmstart(&vec_from(tc.u0));
    let si = alloc.solve(&v, &up, &lb, &ub, 100);
    let us_i = *alloc.solution();

    (us_n.data.0[0], us_i.data.0[0], sn.exit_code, si.exit_code)
}

#[test]
fn all_cases_naive_vs_incremental() {
    let mut fails = 0;
    let mut worst: f32 = 0.0;
    for idx in 0..c_test_cases::N_CASES {
        let (us_n, us_i, ec_n, ec_i) = run_case(idx);
        assert!(!has_nan(&us_n) && !has_nan(&us_i), "NaN at case {idx}");
        let d = max_abs_diff(&us_n, &us_i);
        if d > worst {
            worst = d;
        }
        if d > SOLVER_TOL {
            eprintln!("FAIL case {idx}: diff={d:.6e} ec_n={ec_n:?} ec_i={ec_i:?}");
            fails += 1;
        }
    }
    eprintln!(
        "naive vs incr: {}/{} ok, max_diff={worst:.6e}",
        c_test_cases::N_CASES - fails,
        c_test_cases::N_CASES
    );
    assert_eq!(fails, 0);
}

#[test]
fn all_cases_naive_vs_c_ref() {
    let mut fails = 0;
    let mut worst: f32 = 0.0;
    for idx in 0..c_test_cases::N_CASES {
        let (us_n, _, ec_n, _) = run_case(idx);
        let cr = &c_ref::C_REFS[idx];
        if (ec_n as u8) != cr.exit_code {
            fails += 1;
            continue;
        }
        let d = max_abs_diff(&us_n, &cr.us);
        if d > worst {
            worst = d;
        }
        if d > C_REF_TOL {
            fails += 1;
        }
    }
    eprintln!(
        "naive vs C ref: {}/{} ok, max_diff={worst:.6e}",
        c_test_cases::N_CASES - fails,
        c_test_cases::N_CASES
    );
    assert_eq!(fails, 0);
}

#[test]
fn all_cases_incremental_vs_c_ref() {
    let mut fails = 0;
    let mut worst: f32 = 0.0;
    for idx in 0..c_test_cases::N_CASES {
        let (_, us_i, _, ec_i) = run_case(idx);
        let cr = &c_ref::C_REFS[idx];
        if (ec_i as u8) != cr.exit_code {
            fails += 1;
            continue;
        }
        let d = max_abs_diff(&us_i, &cr.us);
        if d > worst {
            worst = d;
        }
        if d > C_REF_TOL {
            fails += 1;
        }
    }
    eprintln!(
        "incr vs C ref: {}/{} ok, max_diff={worst:.6e}",
        c_test_cases::N_CASES - fails,
        c_test_cases::N_CASES
    );
    assert_eq!(fails, 0);
}

#[test]
fn all_cases_no_nan() {
    for idx in 0..c_test_cases::N_CASES {
        let (us_n, us_i, _, _) = run_case(idx);
        assert!(!has_nan(&us_n), "NaN in naive at case {idx}");
        assert!(!has_nan(&us_i), "NaN in incr at case {idx}");
    }
}

#[test]
fn all_cases_within_bounds() {
    for idx in 0..c_test_cases::N_CASES {
        let tc = &c_test_cases::CASES[idx];
        let (us_n, us_i, ec_n, ec_i) = run_case(idx);
        let tol = 1e-3;
        for (label, us, ec) in [("naive", &us_n, ec_n), ("incr", &us_i, ec_i)] {
            if ec == ExitCode::Success {
                for (j, ((&u, &lo), &hi)) in
                    us.iter().zip(tc.lb.iter()).zip(tc.ub.iter()).enumerate()
                {
                    assert!(
                        u >= lo - tol && u <= hi + tol,
                        "{label} case {idx} u[{j}]={u:.6} out of bounds",
                    );
                }
            }
        }
    }
}
