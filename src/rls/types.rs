//! Types specific to the RLS solvers.

/// How an RLS update terminated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ExitCode {
    /// Normal update completed successfully.
    Success = 0,
    /// Covariance explosion detected — forgetting factor was temporarily increased.
    ///
    /// The update still completed, but with a modified λ to prevent
    /// unbounded covariance growth. This typically indicates insufficient
    /// excitation in the regressor signal.
    CovarianceExplosion = 1,
}

/// Statistics returned after each RLS update step.
#[derive(Debug, Clone, Copy)]
pub struct UpdateStats {
    /// How the update terminated.
    pub exit_code: ExitCode,
    /// Total number of samples processed (including this one).
    pub samples: u32,
}

/// Numerical guard configuration for the standard (covariance-form) RLS.
///
/// These guards prevent numerical instability that arises from maintaining
/// the covariance matrix `P` directly. The inverse QR-RLS does not need
/// these guards because it works with the information matrix factor, where
/// updates involve only additions (never subtractions).
///
/// Default values match the indiflight C reference implementation.
#[derive(Debug, Clone, Copy)]
pub struct CovarianceGuards {
    /// Maximum allowed diagonal value in `P`.
    ///
    /// When any `P[i,i]` exceeds this threshold, the forgetting factor `λ`
    /// is temporarily increased toward 1.0 to slow covariance growth.
    ///
    /// Default: `1e10` (matches `RLS_COV_MAX` in indiflight).
    pub cov_max: f32,

    /// Minimum threshold for trace/diagonal ratio checks.
    ///
    /// Guards against division by very small numbers in the order
    /// decrement limiting logic.
    ///
    /// Default: `1e-10` (matches `RLS_COV_MIN` in indiflight).
    pub cov_min: f32,

    /// Maximum relative trace reduction per update step.
    ///
    /// Limits how much `trace(P)` can decrease in a single update,
    /// preventing numerical instability from aggressive rank-1 updates.
    /// A value of 0.1 means at most 10% of the trace can be removed
    /// per step.
    ///
    /// Default: `0.1` (matches `RLS_MAX_P_ORDER_DECREMENT` in indiflight).
    pub max_order_decrement: f32,
}

impl Default for CovarianceGuards {
    fn default() -> Self {
        Self {
            cov_max: 1e10,
            cov_min: 1e-10,
            max_order_decrement: 0.1,
        }
    }
}
