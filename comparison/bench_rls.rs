#![allow(clippy::needless_range_loop)]
//! RLS benchmark binary: throughput and numerical stability.
//!
//! Usage:
//!   bench_rls throughput   — CSV: ns/step for standard, inverse QR, and C-equivalent
//!   bench_rls stability    — CSV: per-step drift from f64 reference
//!
//! Build: cargo build --release --bin bench_rls
//! Run:   ./target/release/bench_rls throughput > comparison/results_rust.csv
//!        ./target/release/bench_rls stability  > comparison/stability.csv

use flight_solver::rls::{CovarianceGuards, InverseQrRls, RlsParallel};
use nalgebra::SVector;
use std::env;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════════
//  Shared PRNG
// ═══════════════════════════════════════════════════════════════════════════

struct Xor32(u32);
impl Xor32 {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }
}

fn gen_data<const N: usize, const P: usize>(
    rng: &mut Xor32,
    n_steps: usize,
) -> (Vec<SVector<f32, N>>, Vec<SVector<f32, P>>) {
    let regressors = (0..n_steps)
        .map(|_| {
            let mut d = [0.0f32; N];
            for v in d.iter_mut() { *v = rng.next_f32(); }
            SVector::<f32, N>::from_column_slice(&d)
        })
        .collect();
    let observations = (0..n_steps)
        .map(|_| {
            let mut d = [0.0f32; P];
            for v in d.iter_mut() { *v = rng.next_f32(); }
            SVector::<f32, P>::from_column_slice(&d)
        })
        .collect();
    (regressors, observations)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Throughput benchmark
// ═══════════════════════════════════════════════════════════════════════════

const N_WARMUP: usize = 50;
const N_ITERS: usize = 10_000;
const N_STEPS: usize = 100;

fn bench_throughput_one<const N: usize, const P: usize>(
    label: &str, solver: &str, gamma: f32, lambda: f32, seed: u32,
) {
    let mut rng = Xor32(seed);
    let (regs, obs) = gen_data::<N, P>(&mut rng, N_STEPS);
    let guards = CovarianceGuards::default();

    // Warmup + timed runs
    for phase in 0..2 {
        let iters = if phase == 0 { N_WARMUP } else { N_ITERS };
        let start = Instant::now();
        let mut x00 = 0.0f32;
        for r in 0..iters {
            match solver {
                "standard" => {
                    let mut rls = RlsParallel::<N, P>::new(gamma, lambda, guards);
                    for i in 0..N_STEPS { rls.update(&regs[i], &obs[i]); }
                    if phase == 1 && r == iters - 1 { x00 = rls.params()[(0, 0)]; }
                }
                "inverse_qr" => {
                    let mut rls = InverseQrRls::<N, P>::new(gamma, lambda);
                    for i in 0..N_STEPS { rls.update(&regs[i], &obs[i]); }
                    if phase == 1 && r == iters - 1 { x00 = rls.params()[(0, 0)]; }
                }
                _ => unreachable!(),
            }
        }
        if phase == 1 {
            let ns_seq = start.elapsed().as_nanos() as f64 / iters as f64;
            let ns_step = ns_seq / N_STEPS as f64;
            println!("{},{},{},{:.1},{:.1},{:.8e}", label, N_STEPS, solver, ns_seq, ns_step, x00);
        }
    }
}

fn run_throughput() {
    println!("config,steps,solver,ns_per_sequence,ns_per_step,final_x00");
    for solver in &["standard", "inverse_qr"] {
        bench_throughput_one::<4, 1>("motor_n4p1", solver, 100.0, 0.995, 42);
        bench_throughput_one::<8, 3>("g1g2_n8p3", solver, 100.0, 0.995, 123);
        bench_throughput_one::<4, 3>("force_n4p3", solver, 100.0, 0.995, 77);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Stability benchmark (f64 reference comparison)
// ═══════════════════════════════════════════════════════════════════════════

struct RlsF64<const N: usize, const P: usize> {
    x: [[f64; P]; N],
    p: [[f64; N]; N],
    lambda: f64,
}

impl<const N: usize, const P: usize> RlsF64<N, P> {
    fn new(gamma: f64, lambda: f64) -> Self {
        let mut p = [[0.0f64; N]; N];
        for i in 0..N { p[i][i] = gamma; }
        Self { x: [[0.0; P]; N], lambda, p }
    }

    fn update(&mut self, a: &[f64; N], y: &[f64; P]) {
        let mut pa = [0.0f64; N];
        for i in 0..N { for k in 0..N { pa[i] += self.p[k][i] * a[k]; } }
        let mut apa = 0.0f64;
        for i in 0..N { apa += a[i] * pa[i]; }
        let mut e = [0.0f64; P];
        for j in 0..P { e[j] = y[j]; for k in 0..N { e[j] -= a[k] * self.x[k][j]; } }
        let isig = 1.0 / (self.lambda + apa);
        let mut k = [0.0f64; N];
        for i in 0..N { k[i] = pa[i] * isig; }
        let ilam = 1.0 / self.lambda;
        for col in 0..N { for row in 0..N {
            self.p[row][col] = (self.p[row][col] - k[row] * pa[col]) * ilam;
        }}
        for j in 0..P { for i in 0..N { self.x[i][j] += k[i] * e[j]; } }
    }

    fn param_error(&self, true_x: &[[f64; P]; N]) -> f64 {
        let mut m = 0.0f64;
        for i in 0..N { for j in 0..P {
            let e = (self.x[i][j] - true_x[i][j]).abs();
            if e > m { m = e; }
        }}
        m
    }
}

fn max_err_f32_vs_f64<const N: usize, const P: usize>(
    f32p: &nalgebra::OMatrix<f32, nalgebra::Const<N>, nalgebra::Const<P>>,
    f64r: &RlsF64<N, P>,
) -> f32 {
    let mut m = 0.0f32;
    for i in 0..N { for j in 0..P {
        let e = (f32p[(i, j)] - f64r.x[i][j] as f32).abs();
        if e > m { m = e; }
    }}
    m
}

fn max_err_vs_true<const N: usize, const P: usize>(
    f32p: &nalgebra::OMatrix<f32, nalgebra::Const<N>, nalgebra::Const<P>>,
    true_x: &[[f64; P]; N],
) -> f32 {
    let mut m = 0.0f32;
    for i in 0..N { for j in 0..P {
        let e = (f32p[(i, j)] - true_x[i][j] as f32).abs();
        if e > m { m = e; }
    }}
    m
}

fn run_scenario<const N: usize, const P: usize>(
    name: &str, gamma: f32, lambda: f32,
    true_x: &[[f64; P]; N], data: &[([f32; N], [f32; P])],
) {
    let guards = CovarianceGuards { cov_max: f32::MAX, cov_min: f32::MAX, max_order_decrement: 0.1 };
    let mut std_rls = RlsParallel::<N, P>::new(gamma, lambda, guards);
    let mut iqr_rls = InverseQrRls::<N, P>::new(gamma, lambda);
    let mut ref_rls = RlsF64::<N, P>::new(gamma as f64, lambda as f64);

    for (step, (a_arr, y_arr)) in data.iter().enumerate() {
        let a = SVector::<f32, N>::from_column_slice(a_arr);
        let y = SVector::<f32, P>::from_column_slice(y_arr);
        let mut a64 = [0.0f64; N]; for i in 0..N { a64[i] = a_arr[i] as f64; }
        let mut y64 = [0.0f64; P]; for i in 0..P { y64[i] = y_arr[i] as f64; }

        std_rls.update(&a, &y);
        iqr_rls.update(&a, &y);
        ref_rls.update(&a64, &y64);

        println!("{},{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}",
            name, step + 1,
            max_err_f32_vs_f64(std_rls.params(), &ref_rls),
            max_err_f32_vs_f64(iqr_rls.params(), &ref_rls),
            ref_rls.param_error(true_x) as f32,
            max_err_vs_true(std_rls.params(), true_x),
            max_err_vs_true(iqr_rls.params(), true_x),
        );
    }
}

fn run_stability() {
    println!("scenario,step,std_vs_ref,iqr_vs_ref,ref_vs_true,std_vs_true,iqr_vs_true");

    // High gamma: catastrophic cancellation in covariance form
    {
        let mut rng = Xor32(42);
        let tx = [[0.5f64], [-0.3], [0.8], [0.1]];
        let d: Vec<_> = (0..1000).map(|_| {
            let mut a = [0.0f32; 4]; for v in a.iter_mut() { *v = rng.next_f32(); }
            let y = [(tx[0][0]*a[0] as f64 + tx[1][0]*a[1] as f64 + tx[2][0]*a[2] as f64
                + tx[3][0]*a[3] as f64 + 0.01*rng.next_f32() as f64) as f32];
            (a, y)
        }).collect();
        run_scenario("high_gamma", 1e6, 0.995, &tx, &d);
    }

    // Low excitation burst: quiet then loud
    {
        let mut rng = Xor32(123);
        let tx = [[1.0f64, -0.5, 0.2], [0.3, 0.7, -0.4]];
        let d: Vec<_> = (0..1000).map(|step| {
            let s = if step < 500 { 1e-3f32 } else { 1.0 };
            let mut a = [0.0f32; 2]; for v in a.iter_mut() { *v = rng.next_f32() * s; }
            let mut y = [0.0f32; 3];
            for j in 0..3 { y[j] = (tx[0][j]*a[0] as f64 + tx[1][j]*a[1] as f64 + 0.001*rng.next_f32() as f64) as f32; }
            (a, y)
        }).collect();
        run_scenario("low_excitation", 1e2, 0.995, &tx, &d);
    }

    // Long convergence: slow forgetting, 2000 steps
    {
        let mut rng = Xor32(777);
        let tx = [[0.3f64,-0.1,0.5],[-0.2,0.4,0.1],[0.6,-0.3,-0.2],[0.1,0.2,-0.4]];
        let d: Vec<_> = (0..2000).map(|_| {
            let mut a = [0.0f32; 4]; for v in a.iter_mut() { *v = rng.next_f32(); }
            let mut y = [0.0f32; 3];
            for j in 0..3 { let mut v = 0.0f64; for k in 0..4 { v += tx[k][j]*a[k] as f64; }
                y[j] = (v + 0.01*rng.next_f32() as f64) as f32; }
            (a, y)
        }).collect();
        run_scenario("long_convergence", 1e4, 0.999, &tx, &d);
    }
}

// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    let args: Vec<String> = env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("throughput");

    match cmd {
        "throughput" => run_throughput(),
        "stability" => run_stability(),
        _ => {
            eprintln!("Usage: bench_rls [throughput|stability]");
            std::process::exit(1);
        }
    }
}
