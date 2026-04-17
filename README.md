# Flight Solver

[![CI](https://github.com/noidvan/flight-solver/actions/workflows/ci.yml/badge.svg)](https://github.com/noidvan/flight-solver/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/flight-solver.svg)](https://crates.io/crates/flight-solver)
[![docs.rs](https://docs.rs/flight-solver/badge.svg)](https://docs.rs/flight-solver)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Real-time solvers for flight controllers. `no_std`, fully stack-allocated, const-generic over all dimensions.

## Solvers

| Module | Algorithm | Description |
|--------|-----------|-------------|
| `cls` | Constrained Least Squares | Active-set solver with incremental Givens QR |
| `cls::setup::wls` | WLS formulation | Weighted LS with actuator-preference regularisation |
| `cls::setup::ls` | LS formulation | Plain (unregularised) least-squares |
| `rls::standard` | Standard RLS | Covariance-form with numerical guards |
| `rls::inverse_qr` | Inverse QR-RLS | Information-form via Givens rotations |

## Quick start

### RLS - online parameter estimation

```rust
use flight_solver::rls::{InverseQrRls, RlsParallel, CovarianceGuards};

// Inverse QR-RLS: 4 regressors, 3 parallel outputs
let mut rls = InverseQrRls::<4, 3>::new(1e2, 0.995);

let a = nalgebra::SVector::<f32, 4>::new(0.1, -0.2, 0.3, 0.05);
let y = nalgebra::SVector::<f32, 3>::new(0.5, -0.3, 0.1);
rls.update(&a, &y);
```

### WLS control allocation (high-level API)

`flight_solver::wls::ControlAllocator` owns the problem configuration (`A`,
`γ`, normalized `wu`) and the warm-start solver state across solves. Build
once, then call `solve()` every control tick to compute the optimal control
allocation for a given desired pseudo-control and preferred motor command.

```rust
use flight_solver::wls::ControlAllocator;
use flight_solver::cls::{ExitCode, Mat, VecN};

let g: Mat<6, 4> = Mat::zeros();
let wv = VecN::<6>::from_column_slice(&[10.0, 10.0, 10.0, 1.0, 0.5, 0.5]);
let wu = VecN::<4>::from_column_slice(&[1.0; 4]);

// One-time setup: factor A, compute γ, normalize wu
let mut alloc = ControlAllocator::<4, 6, 10>::new(&g, &wv, wu, 2e-9, 4e5);

// Per-tick solve — warm-start is persisted automatically across calls
let v = VecN::<6>::zeros();
let ud = VecN::<4>::from_column_slice(&[0.5; 4]);
let umin = VecN::<4>::from_column_slice(&[0.0; 4]);
let umax = VecN::<4>::from_column_slice(&[1.0; 4]);

let stats = alloc.solve(&v, &ud, &umin, &umax, 100);
assert_eq!(stats.exit_code, ExitCode::Success);
let u = alloc.solution(); // optimal motor commands
```

### CLS - raw building blocks

For advanced use — custom `A` matrices, the unregularised CLS variant, or
non-standard pipeline composition — the `cls` module exposes the underlying
free functions directly.

```rust
use flight_solver::cls::{solve, ExitCode, Mat, VecN};
use flight_solver::cls::setup::wls::{setup_a, setup_b};

let g: Mat<6, 4> = Mat::zeros();
let wv = VecN::<6>::from_column_slice(&[10.0, 10.0, 10.0, 1.0, 0.5, 0.5]);
let mut wu = VecN::<4>::from_column_slice(&[1.0; 4]);

let (a, gamma) = setup_a::<4, 6, 10>(&g, &wv, &mut wu, 2e-9, 4e5);
let b = setup_b::<4, 6, 10>(&VecN::zeros(), &VecN::from_column_slice(&[0.5; 4]), &wv, &wu, gamma);

let mut us = VecN::<4>::from_column_slice(&[0.5; 4]);
let mut ws = [0i8; 4];
let stats = solve::<4, 6, 10>(&a, &b, &VecN::zeros(), &VecN::from_element(1.0), &mut us, &mut ws, 100);
```

## References

- Haykin, S. *Adaptive Filter Theory*, 5th ed., Pearson, 2014. Ch. 15 - Square-root adaptive filtering (inverse QR-RLS derivation).
- [ActiveSetCtlAlloc](https://github.com/tudelft/ActiveSetCtlAlloc) - C reference implementation of the active-set WLS solver.
- [Indiflight](https://github.com/tudelft/indiflight) - C reference implementation of the standard RLS with numerical guards.