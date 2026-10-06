//! LFRU admission/eviction for stochastic, skewed accesses (MoE experts).
//!
//! Adapted from Colibrì's `c/tier.h`: frequency dominates and recency only
//! breaks ties (`score = heat << 8 | recency`, one frequency count outweighs
//! any recency), a candidate must beat the victim by 25% + 4 counts to be
//! admitted (hysteresis against ping-pong), and heat decays by halving.
//! Capacity is counted in entries; entries can be leased, and leased entries
//! are never evicted.

use std::collections::HashMap;
use std::hash::Hash;

#[derive(Debug)]
struct Entry<V> {
    value: V,
    leases: u32,
}

pub struct LfruCache<K, V> {
    capacity: usize,
    entries: HashMap<K, Entry<V>>,
    heat: HashMap<K, u32>,
    last: HashMap<K, u64>,
    clock: u64,
    pub hits: u64,
    pub misses: u64,
    pub admissions: u64,
    pub rejections: u64,
}

impl<K: Hash + Eq + Clone, V> LfruCache<K, V> {
    pub fn new(capacity: usize) -> Self {
        LfruCache {
            capacity,
            entries: HashMap::new(),
            heat: HashMap::new(),
            last: HashMap::new(),
            clock: 0,
            hits: 0,
            misses: 0,
            admissions: 0,
            rejections: 0,
        }
    }

    fn score(&self, k: &K) -> u64 {
        let heat = *self.heat.get(k).unwrap_or(&0) as u64;
        let age = self.clock - self.last.get(k).copied().unwrap_or(0);
        let recent = 255u64.saturating_sub(age);
        (heat << 8) | recent
    }

    /// Record an access. Returns true on a cache hit.
    pub fn touch(&mut self, k: &K) -> bool {
        self.clock += 1;
        let h = self.heat.entry(k.clone()).or_insert(0);
        *h = h.saturating_add(1);
        self.last.insert(k.clone(), self.clock);
        if self.entries.contains_key(k) {
            self.hits += 1;
            true
        } else {
            self.misses += 1;
            false
        }
    }

    pub fn get(&self, k: &K) -> Option<&V> {
        self.entries.get(k).map(|e| &e.value)
    }

    pub fn contains(&self, k: &K) -> bool {
        self.entries.contains_key(k)
    }

    /// Decide whether `k` should be admitted, and which key to evict for it.
    /// `Ok(None)`: admit into free space. `Ok(Some(v))`: admit, evicting `v`.
    /// `Err(())`: not hot enough to displace any resident entry.
    pub fn admission(&self, k: &K) -> Result<Option<K>, ()> {
        if self.entries.contains_key(k) {
            return Err(());
        }
        if self.entries.len() < self.capacity {
            return Ok(None);
        }
        let victim = self.entries.iter().filter(|(_, e)| e.leases == 0).min_by_key(|(key, _)| self.score(key)).map(|(key, _)| key.clone());
        let Some(v) = victim else { return Err(()) };
        let (cs, hs) = (self.score(&v), self.score(k));
        if hs > cs + (cs >> 2) + (4 << 8) {
            Ok(Some(v))
        } else {
            Err(())
        }
    }

    /// Insert after a successful [`admission`](Self::admission), evicting as decided.
    pub fn insert(&mut self, k: K, v: V, evict: Option<K>) -> Option<V> {
        let old = evict.and_then(|e| self.entries.remove(&e)).map(|e| e.value);
        self.entries.insert(k, Entry { value: v, leases: 0 });
        self.admissions += 1;
        old
    }

    pub fn try_admit(&mut self, k: K, make: impl FnOnce() -> V) -> (bool, Option<V>) {
        match self.admission(&k) {
            Ok(evict) => {
                let old = self.insert(k, make(), evict);
                (true, old)
            }
            Err(()) => {
                self.rejections += 1;
                (false, None)
            }
        }
    }

    pub fn lease(&mut self, k: &K) -> Option<&V> {
        let e = self.entries.get_mut(k)?;
        e.leases += 1;
        Some(&e.value)
    }

    pub fn release(&mut self, k: &K) {
        if let Some(e) = self.entries.get_mut(k) {
            e.leases = e.leases.saturating_sub(1);
        }
    }

    /// Halve all heat (call periodically so the cache follows the workload).
    pub fn decay(&mut self) {
        self.heat.retain(|_, h| {
            *h >>= 1;
            *h > 0
        });
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn hit_rate(&self) -> f64 {
        let t = self.hits + self.misses;
        if t == 0 {
            0.0
        } else {
            self.hits as f64 / t as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hysteresis_and_frequency() {
        let mut c = LfruCache::new(2);
        for _ in 0..10 {
            c.touch(&1);
        }
        c.touch(&2);
        assert!(c.try_admit(1, || "a").0);
        assert!(c.try_admit(2, || "b").0);
        // 3 is touched once: must not displace 2 (heat 1) without margin.
        c.touch(&3);
        assert!(!c.try_admit(3, || "c").0);
        for _ in 0..10 {
            c.touch(&3);
        }
        let (ok, old) = c.try_admit(3, || "c");
        assert!(ok);
        assert_eq!(old, Some("b"), "coldest entry evicted, hot entry 1 kept");
        assert!(c.contains(&1));
    }

    #[test]
    fn leased_entries_survive() {
        let mut c = LfruCache::new(1);
        c.touch(&1);
        c.try_admit(1, || 1);
        c.lease(&1);
        for _ in 0..50 {
            c.touch(&2);
        }
        assert!(!c.try_admit(2, || 2).0);
        c.release(&1);
        assert!(c.try_admit(2, || 2).0);
    }

    #[test]
    fn skewed_workload_beats_capacity_ratio() {
        // Zipf-like accesses over 64 keys with room for 8: hit rate should be
        // far above 8/64 once warm.
        let mut c = LfruCache::new(8);
        let mut x = 12345u64;
        for i in 0..20000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let u = (x % 1000) as f64 / 1000.0;
            let k = ((u * u * u) * 64.0) as u32;
            if !c.touch(&k) {
                c.try_admit(k, || ());
            }
            if i % 2000 == 0 {
                c.decay();
            }
        }
        assert!(c.hit_rate() > 0.4, "{}", c.hit_rate());
    }
}
