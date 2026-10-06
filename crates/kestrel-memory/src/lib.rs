//! Kestrel's memory orchestrator: per-tier budgets ([`Ledger`]), the
//! tier-aware [`WeightStore`] with streaming and prefetch, the
//! [`MemoryGuard`] that enforces budgets on *measured* memory, the
//! [`LfruCache`] policy for MoE experts, and metrics.
//!
//! Nothing here depends on an inference backend.

pub mod guard;
pub mod io;
pub mod ledger;
pub mod lfru;
pub mod metrics;
pub mod store;

pub use guard::{MemoryGuard, Pressure};
pub use ledger::{BudgetError, Ledger, Reservation, Tier, TierUsage};
pub use lfru::LfruCache;
pub use metrics::MetricsSnapshot;
pub use store::{Lease, RingPolicy, StoreConfig, StoreError, WeightStore};
