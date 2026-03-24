//! Rust benchmark for flight-solver CLS — times both solvers on all 1000 C test cases.
//!
//! Build:  cargo build --release --bin bench_rust
//! Run:    ./target/release/bench_rust > bench/results_rust.csv

// Re-use the test case data from the regression tests
#[path = "../tests/wls_regression/c_test_cases.rs"]
mod c_test_cases;

use flight_solver::cls::setup::wls::{setup_a, setup_b};
use flight_solver::cls::{solve, ExitCode};
use nalgebra::{ArrayStorage, Matrix, SMatrix, SVector};
use std::time::Instant;

fn mat<const R: usize, const C: usize>(data: [[f32; R]; C]) -> SMatrix<f32, R, C> {
    Matrix::from_data(ArrayStorage(data))
}
fn vec_from<const N: usize>(data: [f32; N]) -> SVector<f32, N> {
    Matrix::from_data(ArrayStorage([data]))
}

const N_WARMUP: usize = 50;
const N_ITERS: usize = 1000;

fn bench_case(idx: usize) -> (f64, ExitCode, usize, [f32; 6]) {
    let tc = &c_test_cases::CASES[idx];

    let g: SMatrix<f32, 4, 6> = mat(tc.jg);
    let wv: SVector<f32, 4> = vec_from(tc.wv);
    let v: SVector<f32, 4> = vec_from(tc.v);
    let up: SVector<f32, 6> = vec_from(tc.up);
    let lb: SVector<f32, 6> = vec_from(tc.lb);
    let ub: SVector<f32, 6> = vec_from(tc.ub);

    // Setup once (not timed — matches C benchmark structure)
    let mut wu: SVector<f32, 6> = vec_from(tc.wu);
    let (a, gamma) = setup_a::<6, 4, 10>(&g, &wv, &mut wu, 2.0e-9, 4e5);
    let b = setup_b::<6, 4, 10>(&v, &up, &wv, &wu, gamma);

    // Warmup
    for _ in 0..N_WARMUP {
        let mut us: SVector<f32, 6> = vec_from(tc.u0);
        let mut ws = [0i8; 6];
        let _ = solve::<6, 4, 10>(&a, &b, &lb, &ub, &mut us, &mut ws, 100);
    }

    // Timed runs
    let mut last_us = [0.0f32; 6];
    let mut last_ec = ExitCode::IterLimit;
    let mut last_iter = 0usize;

    let start = Instant::now();
    for r in 0..N_ITERS {
        let mut us: SVector<f32, 6> = vec_from(tc.u0);
        let mut ws = [0i8; 6];
        let stats = solve::<6, 4, 10>(&a, &b, &lb, &ub, &mut us, &mut ws, 100);

        if r == N_ITERS - 1 {
            for i in 0..6 {
                last_us[i] = us[i];
            }
            last_ec = stats.exit_code;
            last_iter = stats.iterations;
        }
    }
    let elapsed_ns = start.elapsed().as_nanos() as f64 / N_ITERS as f64;

    (elapsed_ns, last_ec, last_iter, last_us)
}

fn main() {
    println!(
        "case,solver,exit_code,iterations,time_ns,\
         us0,us1,us2,us3,us4,us5"
    );

    for c in 0..c_test_cases::N_CASES {
        let (ns, ec, iter, us) = bench_case(c);
        println!(
            "{},incremental,{},{},{:.1},\
             {:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e}",
            c, ec as u8, iter, ns, us[0], us[1], us[2], us[3], us[4], us[5]
        );
    }
}
