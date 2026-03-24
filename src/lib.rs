//! Real-time solvers for flight controllers.
//!
//! `flight-solver` provides `no_std`, fully stack-allocated solvers designed
//! for real-time system identification and control allocation on embedded
//! targets (Cortex-M7 at 8 kHz). All dimensions are const-generic for
//! zero-overhead monomorphization.
//!
//! # Solvers
//!
//! | Module | Algorithm | Use case |
//! |--------|-----------|----------|
//! | [`rls`] | Recursive Least Squares | Online parameter estimation (G1/G2 learning) |
//!
//! # Example: RLS system identification
//!
//! ```no_run
//! use flight_solver::rls::{RlsParallel, CovarianceGuards};
//!
//! // Learn a 4→3 mapping (4 motor regressors, 3 torque outputs)
//! let mut rls = RlsParallel::<4, 3>::new(1e2, 0.995, CovarianceGuards::default());
//!
//! // Each control loop iteration at 8 kHz:
//! let regressor = nalgebra::SVector::<f32, 4>::new(0.1, -0.2, 0.3, 0.05);
//! let observation = nalgebra::SVector::<f32, 3>::new(0.5, -0.3, 0.1);
//! rls.update(&regressor, &observation);
//!
//! let g1_estimate = rls.params(); // 4×3 matrix
//! ```

#![no_std]
#![warn(missing_docs)]

pub mod givens;
pub mod rls;
pub mod types;
