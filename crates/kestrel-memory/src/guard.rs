//! Budget enforcement on *measured* memory, and the rebalancer that moves
//! groups between the resident and streamed tiers at safe points.
//!
//! Projections are estimates; Colibrì's issue #403 saw a projected peak of
//! 74 GB turn into a measured 116 GB and three OOM kills. Kestrel therefore
//! samples RSS and system available memory between tokens and demotes
//! resident groups when either crosses its threshold. Demotion is always
//! possible because the model file still holds every byte.

use crate::ledger::Tier;
use crate::store::WeightStore;
use serde::Serialize;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum Pressure {
    /// Within budget; `spare` bytes could be promoted safely.
    Ok { spare: u64 },
    /// Over budget by `excess` bytes (RSS above limit or system memory low).
    Over { excess: u64 },
}

#[derive(Clone, Debug, Serialize)]
pub struct MemoryGuard {
    /// Bytes this process may hold resident.
    pub rss_limit: u64,
    /// System available memory below which we shed load regardless of RSS.
    pub min_available: u64,
    /// Fraction of the limit tolerated above it before acting (noise filter).
    pub tolerance: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct MemSample {
    pub rss: u64,
    pub available: u64,
}

impl MemSample {
    pub fn now() -> Option<Self> {
        let rss = kestrel_hw::process_rss()?;
        let m = kestrel_hw::HardwareProfile::discover(&[]).ram;
        Some(MemSample { rss, available: m.effective_available() })
    }
}

impl MemoryGuard {
    pub fn assess(&self, s: MemSample) -> Pressure {
        let slack = (self.rss_limit as f64 * self.tolerance) as u64 + (256 << 20);
        let over_rss = s.rss.saturating_sub(self.rss_limit + slack);
        let over_sys = self.min_available.saturating_sub(s.available);
        let excess = over_rss.max(over_sys);
        if excess > 0 {
            Pressure::Over { excess }
        } else {
            let spare = self.rss_limit.saturating_sub(s.rss).min(s.available.saturating_sub(self.min_available));
            Pressure::Ok { spare }
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub enum RebalanceAction {
    Demoted { group: usize, bytes: u64 },
    Promoted { group: usize, bytes: u64 },
}

/// Moves groups between tiers at safe points (no leases held).
///
/// By default the plan is static and the rebalancer only protects it:
/// it demotes under memory pressure and, once memory is back, restores the
/// groups it demoted. With `beyond_plan` it also promotes streamed groups the
/// plan never made resident (opportunistic adaptation).
pub struct Rebalancer {
    pub guard: MemoryGuard,
    /// Consecutive `Ok` assessments with enough headroom before promoting.
    pub promote_after: u32,
    pub allow_promotion: bool,
    /// Promote groups beyond the plan's resident set, not only restore demoted ones.
    pub beyond_plan: bool,
    calm: u32,
    /// Groups that must stay resident (e.g. embeddings used by every token).
    pub pinned: Vec<usize>,
    /// Groups demoted under pressure, waiting to be restored.
    demoted: Vec<usize>,
}

impl Rebalancer {
    pub fn new(guard: MemoryGuard) -> Self {
        Rebalancer { guard, promote_after: 3, allow_promotion: true, beyond_plan: false, calm: 0, pinned: Vec::new(), demoted: Vec::new() }
    }

    /// Groups demoted under pressure and not yet restored.
    pub fn demoted(&self) -> &[usize] {
        &self.demoted
    }

    /// One rebalancing step from a memory sample. Call between tokens.
    pub fn tick(&mut self, store: &WeightStore, sample: MemSample) -> Vec<RebalanceAction> {
        let mut actions = Vec::new();
        match self.guard.assess(sample) {
            Pressure::Over { excess } => {
                self.calm = 0;
                let mut freed = 0u64;
                while freed < excess {
                    let Some(g) = self.pick(store, true) else { break };
                    let bytes = store.group_buf_len(g) as u64;
                    if store.demote(g).is_err() {
                        break;
                    }
                    freed += bytes;
                    // Lower the ledger ceiling so the freed room is not refilled.
                    let ledger = store.ledger();
                    let lim = ledger.usage(Tier::Ram).limit;
                    ledger.set_limit(Tier::Ram, lim.saturating_sub(bytes));
                    self.demoted.push(g);
                    actions.push(RebalanceAction::Demoted { group: g, bytes });
                }
            }
            Pressure::Ok { spare } => {
                self.calm += 1;
                // Promotions load in the background and are installed here,
                // at a safe point, so decode never waits for them.
                if let Ok(done) = store.poll_promotions() {
                    for g in done {
                        actions.push(RebalanceAction::Promoted { group: g, bytes: store.group_buf_len(g) as u64 });
                    }
                }
                if self.allow_promotion && self.calm >= self.promote_after && store.pending_promotions() == 0 {
                    if let Some(g) = self.pick(store, false) {
                        let bytes = store.group_buf_len(g) as u64;
                        let ledger = store.ledger();
                        let restore = self.demoted.contains(&g);
                        // Keep a full group of headroom beyond the promoted one.
                        if spare > 2 * bytes {
                            // Restoring gives back the ledger room taken at demotion.
                            let lim = ledger.usage(Tier::Ram).limit;
                            if restore {
                                ledger.set_limit(Tier::Ram, lim + bytes);
                            }
                            if ledger.available(Tier::Ram) >= bytes && store.promote_async(g).unwrap_or(false) {
                                self.calm = 0;
                                self.demoted.retain(|&d| d != g);
                            } else if restore {
                                ledger.set_limit(Tier::Ram, lim);
                            }
                        }
                    }
                }
            }
        }
        actions
    }

    /// Choose a layer group to demote (resident → streamed) or promote.
    ///
    /// Streamed groups should be spread out between resident ones so each
    /// load has resident compute to hide behind. Demotion therefore picks the
    /// resident group furthest from any streamed group; promotion picks the
    /// streamed group with the least resident compute before it.
    fn pick(&self, store: &WeightStore, demote: bool) -> Option<usize> {
        let model = store.model();
        let mask = store.resident_mask();
        let layer_groups: Vec<usize> = model.groups.iter().filter(|g| g.layer.is_some() && !store.is_external(g.id)).map(|g| g.id).collect();
        if layer_groups.is_empty() {
            return None;
        }
        let n = layer_groups.len();
        let gap = |i: usize, want_resident: bool| -> usize {
            // distance to the nearest group (cyclically) whose residency != want
            (1..n).find(|&k| mask[layer_groups[(i + k) % n]] != want_resident || mask[layer_groups[(i + n - k) % n]] != want_resident).unwrap_or(n)
        };
        if demote {
            (0..n)
                .filter(|&i| mask[layer_groups[i]] && !self.pinned.contains(&layer_groups[i]))
                .max_by_key(|&i| (gap(i, true), model.groups[layer_groups[i]].bytes))
                .map(|i| layer_groups[i])
        } else {
            (0..n)
                .filter(|&i| !mask[layer_groups[i]] && (self.beyond_plan || self.demoted.contains(&layer_groups[i])))
                .max_by_key(|&i| gap(i, false))
                .map(|i| layer_groups[i])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assess() {
        let g = MemoryGuard { rss_limit: 1 << 30, min_available: 512 << 20, tolerance: 0.02 };
        assert!(matches!(g.assess(MemSample { rss: 900 << 20, available: 4 << 30 }), Pressure::Ok { .. }));
        assert!(matches!(g.assess(MemSample { rss: 2 << 30, available: 4 << 30 }), Pressure::Over { .. }));
        assert!(matches!(g.assess(MemSample { rss: 100 << 20, available: 100 << 20 }), Pressure::Over { .. }));
    }

    fn rig(beyond_plan: bool) -> (tempfile::TempDir, WeightStore, Rebalancer) {
        use crate::ledger::Ledger;
        use crate::store::{tests_support::model, StoreConfig};
        let d = tempfile::tempdir().unwrap();
        let m = model(d.path(), 6);
        let n = m.groups.len();
        // Layer groups alternate resident / streamed.
        let resident: Vec<bool> = (0..n).map(|g| m.groups[g].layer.is_none() || g % 2 == 0).collect();
        let store = WeightStore::new(m, &resident, (0..n).collect(), StoreConfig::default(), Ledger::new(0, 1 << 30, 0)).unwrap();
        let mut r = Rebalancer::new(MemoryGuard { rss_limit: 1 << 30, min_available: 0, tolerance: 0.0 });
        r.promote_after = 1;
        r.beyond_plan = beyond_plan;
        (d, store, r)
    }
    const CALM: MemSample = MemSample { rss: 1 << 20, available: 8 << 30 };

    /// Tick until no promotion is pending; returns the resident mask.
    fn settle(r: &mut Rebalancer, store: &WeightStore, ticks: usize) -> Vec<bool> {
        for _ in 0..ticks {
            r.tick(store, CALM);
            while store.pending_promotions() > 0 {
                std::thread::sleep(std::time::Duration::from_millis(2));
                r.tick(store, CALM);
            }
        }
        store.resident_mask()
    }

    #[test]
    fn static_plan_restores_only_what_pressure_demoted() {
        let (_d, store, mut r) = rig(false);
        let plan = store.resident_mask();
        // Calm memory: the plan is left alone.
        assert_eq!(settle(&mut r, &store, 5), plan);
        // Pressure (1 byte over limit + slack): one resident layer group is demoted.
        let acts = r.tick(&store, MemSample { rss: (1 << 30) + (256 << 20) + 1, available: 8 << 30 });
        assert_eq!(acts.len(), 1, "{acts:?}");
        let RebalanceAction::Demoted { group, .. } = acts[0] else { panic!("{acts:?}") };
        assert!(!store.resident_mask()[group]);
        assert_eq!(r.demoted(), &[group]);
        // Memory is back: exactly the plan is restored, nothing more.
        assert_eq!(settle(&mut r, &store, 5), plan);
        assert!(r.demoted().is_empty());
    }

    #[test]
    fn beyond_plan_promotes_streamed_layers() {
        let (_d, store, mut r) = rig(true);
        let before = store.resident_mask().iter().filter(|&&x| x).count();
        let after = settle(&mut r, &store, 5).iter().filter(|&&x| x).count();
        assert!(after > before, "{before} -> {after}");
    }
}
