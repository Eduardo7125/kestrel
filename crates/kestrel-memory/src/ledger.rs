//! Per-tier budgets. Every byte Kestrel owns is reserved here first; a
//! reservation past the limit is an error value, never an OOM.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    Vram,
    Ram,
    Disk,
}

impl Tier {
    pub fn name(self) -> &'static str {
        match self {
            Tier::Vram => "VRAM",
            Tier::Ram => "RAM",
            Tier::Disk => "NVMe",
        }
    }
    fn idx(self) -> usize {
        self as usize
    }
}

#[derive(Debug, thiserror::Error)]
#[error("{tier} budget exhausted: requested {requested} bytes for {purpose}, {reserved}/{limit} reserved ({breakdown})")]
pub struct BudgetError {
    pub tier: &'static str,
    pub requested: u64,
    pub purpose: String,
    pub reserved: u64,
    pub limit: u64,
    pub breakdown: String,
}

#[derive(Default)]
struct Account {
    limit: AtomicU64,
    reserved: AtomicU64,
    peak: AtomicU64,
    by_purpose: Mutex<std::collections::BTreeMap<String, u64>>,
}

#[derive(Default)]
pub struct Ledger {
    accounts: [Account; 3],
}

#[derive(Clone, Debug, Serialize)]
pub struct TierUsage {
    pub tier: Tier,
    pub limit: u64,
    pub reserved: u64,
    pub peak: u64,
    pub by_purpose: std::collections::BTreeMap<String, u64>,
}

impl Ledger {
    pub fn new(vram: u64, ram: u64, disk: u64) -> Arc<Self> {
        let l = Ledger::default();
        l.accounts[0].limit.store(vram, Ordering::Relaxed);
        l.accounts[1].limit.store(ram, Ordering::Relaxed);
        l.accounts[2].limit.store(disk, Ordering::Relaxed);
        Arc::new(l)
    }

    pub fn set_limit(&self, tier: Tier, limit: u64) {
        self.accounts[tier.idx()].limit.store(limit, Ordering::Relaxed);
    }

    pub fn reserve(self: &Arc<Self>, tier: Tier, bytes: u64, purpose: &str) -> Result<Reservation, BudgetError> {
        let a = &self.accounts[tier.idx()];
        let limit = a.limit.load(Ordering::Relaxed);
        let mut cur = a.reserved.load(Ordering::Relaxed);
        loop {
            let next = cur.saturating_add(bytes);
            if next > limit {
                let bp = a.by_purpose.lock().unwrap();
                let breakdown = bp.iter().map(|(k, v)| format!("{k} {}", kestrel_hw::fmt_bytes(*v))).collect::<Vec<_>>().join(", ");
                return Err(BudgetError {
                    tier: tier.name(),
                    requested: bytes,
                    purpose: purpose.to_string(),
                    reserved: cur,
                    limit,
                    breakdown,
                });
            }
            match a.reserved.compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Relaxed) {
                Ok(_) => {
                    a.peak.fetch_max(next, Ordering::Relaxed);
                    *a.by_purpose.lock().unwrap().entry(purpose.to_string()).or_insert(0) += bytes;
                    return Ok(Reservation { ledger: self.clone(), tier, bytes, purpose: purpose.to_string() });
                }
                Err(v) => cur = v,
            }
        }
    }

    pub fn usage(&self, tier: Tier) -> TierUsage {
        let a = &self.accounts[tier.idx()];
        TierUsage {
            tier,
            limit: a.limit.load(Ordering::Relaxed),
            reserved: a.reserved.load(Ordering::Relaxed),
            peak: a.peak.load(Ordering::Relaxed),
            by_purpose: a.by_purpose.lock().unwrap().clone(),
        }
    }

    pub fn available(&self, tier: Tier) -> u64 {
        let a = &self.accounts[tier.idx()];
        a.limit.load(Ordering::Relaxed).saturating_sub(a.reserved.load(Ordering::Relaxed))
    }

    fn release(&self, tier: Tier, bytes: u64, purpose: &str) {
        let a = &self.accounts[tier.idx()];
        a.reserved.fetch_sub(bytes, Ordering::AcqRel);
        let mut bp = a.by_purpose.lock().unwrap();
        if let Some(v) = bp.get_mut(purpose) {
            *v = v.saturating_sub(bytes);
            if *v == 0 {
                bp.remove(purpose);
            }
        }
    }
}

/// RAII budget reservation; released on drop.
pub struct Reservation {
    ledger: Arc<Ledger>,
    tier: Tier,
    bytes: u64,
    purpose: String,
}

impl Reservation {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn tier(&self) -> Tier {
        self.tier
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.ledger.release(self.tier, self.bytes, &self.purpose);
    }
}

impl std::fmt::Debug for Reservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Reservation({} {} {})", self.tier.name(), self.bytes, self.purpose)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_enforced_and_released() {
        let l = Ledger::new(0, 1000, 0);
        let a = l.reserve(Tier::Ram, 600, "weights").unwrap();
        let err = l.reserve(Tier::Ram, 500, "kv").unwrap_err();
        assert!(err.to_string().contains("weights"), "{err}");
        let b = l.reserve(Tier::Ram, 400, "kv").unwrap();
        assert_eq!(l.available(Tier::Ram), 0);
        drop(a);
        assert_eq!(l.available(Tier::Ram), 600);
        drop(b);
        let u = l.usage(Tier::Ram);
        assert_eq!(u.reserved, 0);
        assert_eq!(u.peak, 1000);
        assert!(u.by_purpose.is_empty());
        assert!(l.reserve(Tier::Vram, 1, "x").is_err());
    }

    #[test]
    fn concurrent_reservations_never_exceed() {
        let l = Ledger::new(0, 10_000, 0);
        let held = std::sync::Mutex::new(Vec::new());
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    for _ in 0..1000 {
                        if let Ok(r) = l.reserve(Tier::Ram, 7, "x") {
                            held.lock().unwrap().push(r);
                        }
                    }
                });
            }
        });
        assert!(l.usage(Tier::Ram).reserved <= 10_000);
        assert_eq!(held.lock().unwrap().len(), 10_000 / 7);
    }
}
