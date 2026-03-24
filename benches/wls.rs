#![allow(clippy::excessive_precision)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use flight_solver::cls::setup::wls::{setup_a, setup_b};
use flight_solver::cls::solve;
use nalgebra::{ArrayStorage, Matrix, SMatrix, SVector};

fn mat<const R: usize, const C: usize>(data: [[f32; R]; C]) -> SMatrix<f32, R, C> {
    Matrix::from_data(ArrayStorage(data))
}
fn vec_from<const N: usize>(data: [f32; N]) -> SVector<f32, N> {
    Matrix::from_data(ArrayStorage([data]))
}

// ---- Hex: 6 actuators, 4 pseudo-controls ----

#[rustfmt::skip]
const HEX_JG: [[f32; 4]; 6] = [
    [-8.0, -1.1698896, 2.0263083, 0.05],
    [-8.0, -2.6163429, 0.0000000, -0.05],
    [-8.0, -1.2556785, -2.174899, 0.05],
    [-8.0,  1.223678, -2.1194725, -0.05],
    [-8.0,  2.2348890, 0.0000000, 0.05],
    [-8.0,  1.1174387, 1.9354606, -0.05],
];

fn hex_problem() -> (
    SMatrix<f32, 10, 6>,
    SVector<f32, 10>,
    SVector<f32, 6>,
    SVector<f32, 6>,
) {
    let mut wu = vec_from([1.0246610, 1.5560965, 1.3817812, 1.4521209, 1.0, 1.6243161]);
    let wv = vec_from([10.0, 31.622776, 31.622776, 1.0]);
    let (a, gamma) = setup_a::<6, 4, 10>(&mat(HEX_JG), &wv, &mut wu, 2.0e-9, 4e5);
    let b = setup_b::<6, 4, 10>(
        &vec_from([-17.588512, 0.817079, -0.943119, -0.574301]),
        &vec_from([0.0; 6]),
        &wv,
        &wu,
        gamma,
    );
    (a, b, vec_from([0.0; 6]), vec_from([1.0; 6]))
}

// ---- Quad: 4 actuators, 6 pseudo-controls ----

#[rustfmt::skip]
const QUAD_G: [[f32; 6]; 4] = [
    [-1.0, 0.5,  0.5,  0.1,  0.02, 0.02],
    [-1.0,-0.5,  0.5, -0.1, -0.02, 0.02],
    [-1.0, 0.5, -0.5, -0.1,  0.02,-0.02],
    [-1.0,-0.5, -0.5,  0.1, -0.02,-0.02],
];

fn quad_problem() -> (
    SMatrix<f32, 10, 4>,
    SVector<f32, 10>,
    SVector<f32, 4>,
    SVector<f32, 4>,
) {
    let mut wu = vec_from([1.0; 4]);
    let wv = vec_from([10.0, 10.0, 10.0, 1.0, 0.5, 0.5]);
    let (a, gamma) = setup_a::<4, 6, 10>(&mat(QUAD_G), &wv, &mut wu, 2.0e-9, 4e5);
    let b = setup_b::<4, 6, 10>(
        &vec_from([-4.0, 0.3, -0.2, 0.05, 0.0, 0.0]),
        &vec_from([0.5; 4]),
        &wv,
        &wu,
        gamma,
    );
    (a, b, vec_from([0.0; 4]), vec_from([1.0; 4]))
}

fn bench_solvers(c: &mut Criterion) {
    let (ha, hb, hlb, hub) = hex_problem();
    let (qa, qb, qlb, qub) = quad_problem();

    c.bench_function("hex_6x4", |ben| {
        ben.iter(|| {
            let mut us = vec_from([0.5; 6]);
            let mut ws = [0i8; 6];
            solve::<6, 4, 10>(
                black_box(&ha),
                black_box(&hb),
                &hlb,
                &hub,
                &mut us,
                &mut ws,
                100,
            )
        })
    });

    c.bench_function("quad_4x6", |ben| {
        ben.iter(|| {
            let mut us = vec_from([0.5; 4]);
            let mut ws = [0i8; 4];
            solve::<4, 6, 10>(
                black_box(&qa),
                black_box(&qb),
                &qlb,
                &qub,
                &mut us,
                &mut ws,
                100,
            )
        })
    });
}

criterion_group!(benches, bench_solvers);
criterion_main!(benches);
