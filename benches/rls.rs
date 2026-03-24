#![allow(clippy::excessive_precision)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use flight_solver::rls::{CovarianceGuards, InverseQrRls, RlsParallel};
use nalgebra::SVector;

// ── Deterministic test data (xorshift32) ────────────────────────────────

struct Xor32(u32);
impl Xor32 {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }
}

fn make_regressor<const N: usize>(rng: &mut Xor32) -> SVector<f32, N> {
    let mut data = [0.0f32; N];
    for v in data.iter_mut() {
        *v = rng.next_f32();
    }
    SVector::<f32, N>::from_column_slice(&data)
}

fn make_observation<const P: usize>(rng: &mut Xor32) -> SVector<f32, P> {
    let mut data = [0.0f32; P];
    for v in data.iter_mut() {
        *v = rng.next_f32();
    }
    SVector::<f32, P>::from_column_slice(&data)
}

// ── Motor RLS: n=4, p=1 ────────────────────────────────────────────────

fn bench_motor_rls(c: &mut Criterion) {
    let mut group = c.benchmark_group("motor_n4_p1");

    // Pre-generate data for 100 steps
    let mut rng = Xor32(42);
    let regressors: Vec<_> = (0..100).map(|_| make_regressor::<4>(&mut rng)).collect();
    let observations: Vec<_> = (0..100).map(|_| make_observation::<1>(&mut rng)).collect();

    group.bench_function("standard", |ben| {
        ben.iter(|| {
            let mut rls = RlsParallel::<4, 1>::new(100.0, 0.995, CovarianceGuards::default());
            for i in 0..100 {
                rls.update(black_box(&regressors[i]), black_box(&observations[i]));
            }
            rls
        })
    });

    group.bench_function("inverse_qr", |ben| {
        ben.iter(|| {
            let mut rls = InverseQrRls::<4, 1>::new(100.0, 0.995);
            for i in 0..100 {
                rls.update(black_box(&regressors[i]), black_box(&observations[i]));
            }
            rls
        })
    });

    group.finish();
}

// ── G1/G2 RLS: n=8, p=3 ────────────────────────────────────────────────

fn bench_g1g2_rls(c: &mut Criterion) {
    let mut group = c.benchmark_group("g1g2_n8_p3");

    let mut rng = Xor32(123);
    let regressors: Vec<_> = (0..100).map(|_| make_regressor::<8>(&mut rng)).collect();
    let observations: Vec<_> = (0..100).map(|_| make_observation::<3>(&mut rng)).collect();

    group.bench_function("standard", |ben| {
        ben.iter(|| {
            let mut rls = RlsParallel::<8, 3>::new(100.0, 0.995, CovarianceGuards::default());
            for i in 0..100 {
                rls.update(black_box(&regressors[i]), black_box(&observations[i]));
            }
            rls
        })
    });

    group.bench_function("inverse_qr", |ben| {
        ben.iter(|| {
            let mut rls = InverseQrRls::<8, 3>::new(100.0, 0.995);
            for i in 0..100 {
                rls.update(black_box(&regressors[i]), black_box(&observations[i]));
            }
            rls
        })
    });

    group.finish();
}

// ── Single-step latency (what matters at 8 kHz) ────────────────────────

fn bench_single_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_step");

    // Pre-warm solvers (100 steps) so covariance is realistic
    let mut rng = Xor32(777);
    let warmup_a: Vec<_> = (0..100).map(|_| make_regressor::<4>(&mut rng)).collect();
    let warmup_y: Vec<_> = (0..100).map(|_| make_observation::<1>(&mut rng)).collect();

    let mut std_rls = RlsParallel::<4, 1>::new(100.0, 0.995, CovarianceGuards::default());
    let mut iqr_rls = InverseQrRls::<4, 1>::new(100.0, 0.995);
    for i in 0..100 {
        std_rls.update(&warmup_a[i], &warmup_y[i]);
        iqr_rls.update(&warmup_a[i], &warmup_y[i]);
    }

    let a = make_regressor::<4>(&mut rng);
    let y = make_observation::<1>(&mut rng);

    group.bench_function("standard_n4_p1", |ben| {
        let mut rls = std_rls.clone();
        ben.iter(|| rls.update(black_box(&a), black_box(&y)))
    });

    group.bench_function("inverse_qr_n4_p1", |ben| {
        let mut rls = iqr_rls.clone();
        ben.iter(|| rls.update(black_box(&a), black_box(&y)))
    });

    // n=8, p=3
    let mut rng = Xor32(888);
    let warmup_a8: Vec<_> = (0..100).map(|_| make_regressor::<8>(&mut rng)).collect();
    let warmup_y3: Vec<_> = (0..100).map(|_| make_observation::<3>(&mut rng)).collect();

    let mut std_rls8 = RlsParallel::<8, 3>::new(100.0, 0.995, CovarianceGuards::default());
    let mut iqr_rls8 = InverseQrRls::<8, 3>::new(100.0, 0.995);
    for i in 0..100 {
        std_rls8.update(&warmup_a8[i], &warmup_y3[i]);
        iqr_rls8.update(&warmup_a8[i], &warmup_y3[i]);
    }

    let a8 = make_regressor::<8>(&mut rng);
    let y3 = make_observation::<3>(&mut rng);

    group.bench_function("standard_n8_p3", |ben| {
        let mut rls = std_rls8.clone();
        ben.iter(|| rls.update(black_box(&a8), black_box(&y3)))
    });

    group.bench_function("inverse_qr_n8_p3", |ben| {
        let mut rls = iqr_rls8.clone();
        ben.iter(|| rls.update(black_box(&a8), black_box(&y3)))
    });

    group.finish();
}

criterion_group!(benches, bench_motor_rls, bench_g1g2_rls, bench_single_step);
criterion_main!(benches);
