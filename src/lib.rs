//! Real-time solvers for flight controllers.
//!
//! `flight-solver` provides `no_std`, fully stack-allocated solvers designed
//! for real-time system identification and control allocation on embedded
//! targets. All dimensions are const-generic for
//! zero-overhead monomorphization.
//!
//! # Solvers
//!
//! | Module | Algorithm | Use case |
//! |--------|-----------|----------|
//! | [`rls`] | Recursive Least Squares | Online parameter estimation |
//! | [`cls`] | Constrained Least Squares | Box-constrained allocation |
//! | [`wls`] | Encapsulated WLS allocator | High-level control allocation API |
//!
//! # Example: RLS online parameter estimation
//!
//! ```no_run
//! use flight_solver::rls::{RlsParallel, CovarianceGuards};
//!
//! let mut rls = RlsParallel::<4, 3>::new(1e2, 0.995, CovarianceGuards::default());
//!
//! let a = nalgebra::SVector::<f32, 4>::new(0.1, -0.2, 0.3, 0.05);
//! let y = nalgebra::SVector::<f32, 3>::new(0.5, -0.3, 0.1);
//! rls.update(&a, &y);
//!
//! let estimate = rls.params(); // 4×3 parameter matrix
//! ```
//!
//! # Example: WLS constrained allocation (encapsulated API — recommended)
//!
//! [`wls::ControlAllocator`] owns the static problem (effectiveness matrix,
//! weights, `γ`) and the warm-start solver state. Build once, then call
//! [`solve`](wls::ControlAllocator::solve) on every control tick.
//!
//! ```no_run
//! use flight_solver::wls::ControlAllocator;
//! use flight_solver::cls::{ExitCode, Mat, VecN};
//!
//! let g: Mat<6, 4> = Mat::zeros();  // effectiveness matrix
//! let wv = VecN::<6>::from_column_slice(&[10.0, 10.0, 10.0, 1.0, 0.5, 0.5]);
//! let wu = VecN::<4>::from_column_slice(&[1.0; 4]);
//!
//! let mut alloc = ControlAllocator::<4, 6, 10>::new(&g, &wv, wu, 2e-9, 4e5);
//!
//! let v = VecN::<6>::zeros();
//! let ud = VecN::<4>::from_column_slice(&[0.5; 4]);
//! let umin = VecN::<4>::from_column_slice(&[0.0; 4]);
//! let umax = VecN::<4>::from_column_slice(&[1.0; 4]);
//!
//! let stats = alloc.solve(&v, &ud, &umin, &umax, 100);
//! assert_eq!(stats.exit_code, ExitCode::Success);
//! let u = alloc.solution();
//! ```
//!
//! # Example: WLS constrained allocation (raw building blocks)
//!
//! For advanced use — custom `A` matrices or non-standard pipeline composition —
//! the [`cls::setup::wls`] and [`cls`] modules expose the free functions
//! directly.
//!
//! ```no_run
//! use flight_solver::cls::{solve, ExitCode, Mat, VecN};
//! use flight_solver::cls::setup::wls::{setup_a, setup_b};
//!
//! let g: Mat<6, 4> = Mat::zeros();
//! let wv = VecN::<6>::from_column_slice(&[10.0, 10.0, 10.0, 1.0, 0.5, 0.5]);
//! let mut wu = VecN::<4>::from_column_slice(&[1.0; 4]);
//!
//! let (a, gamma) = setup_a::<4, 6, 10>(&g, &wv, &mut wu, 2e-9, 4e5);
//! let v = VecN::<6>::zeros();
//! let ud = VecN::<4>::from_column_slice(&[0.5; 4]);
//! let b = setup_b::<4, 6, 10>(&v, &ud, &wv, &wu, gamma);
//!
//! let umin = VecN::<4>::from_column_slice(&[0.0; 4]);
//! let umax = VecN::<4>::from_column_slice(&[1.0; 4]);
//! let mut us = VecN::<4>::from_column_slice(&[0.5; 4]);
//! let mut ws = [0i8; 4];
//! let stats = solve::<4, 6, 10>(&a, &b, &umin, &umax, &mut us, &mut ws, 100);
//! assert_eq!(stats.exit_code, ExitCode::Success);
//! ```
//!
//! # Example: Unregularised LS allocation
//!
//! ```no_run
//! use flight_solver::cls::{solve_cls, ExitCode, Mat, VecN};
//! use flight_solver::cls::setup::ls;
//!
//! let g: Mat<4, 4> = Mat::zeros();  // square system
//! let wv = VecN::<4>::from_column_slice(&[1.0; 4]);
//!
//! let a = ls::setup_a::<4, 4>(&g, &wv);
//! let v = VecN::<4>::zeros();
//! let b = ls::setup_b(&v, &wv);
//!
//! let umin = VecN::<4>::from_column_slice(&[0.0; 4]);
//! let umax = VecN::<4>::from_column_slice(&[1.0; 4]);
//! let mut us = VecN::<4>::from_column_slice(&[0.5; 4]);
//! let mut ws = [0i8; 4];
//! let stats = solve_cls::<4, 4>(&a, &b, &umin, &umax, &mut us, &mut ws, 100);
//! assert_eq!(stats.exit_code, ExitCode::Success);
//! ```

#![no_std]
#![warn(missing_docs)]

pub mod cls;
pub mod givens;
pub mod rls;
pub mod wls;
