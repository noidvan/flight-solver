use nalgebra::{ArrayStorage, Matrix, SMatrix, SVector};

/// Naive QR solver — structurally independent reimplementation used as a test
/// oracle to cross-check the incremental solver.
pub mod naive;

pub fn mat<const R: usize, const C: usize>(data: [[f32; R]; C]) -> SMatrix<f32, R, C> {
    Matrix::from_data(ArrayStorage(data))
}

pub fn vec_from<const N: usize>(data: [f32; N]) -> SVector<f32, N> {
    Matrix::from_data(ArrayStorage([data]))
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}
