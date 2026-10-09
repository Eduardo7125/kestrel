//! The expert store: expert-granular residency for MoE models.
//!
//! This is the generalization of Colibrì's per-layer expert slots
//! (`ESlot`, `c/colibri.c`) and its `ColiExpertStore` lease contract
//! (`c/expert_store.h`). The unit of placement and transfer is **one routed
//! expert**: its slices of the stacked `ffn_{gate,up,down}_exps` tensors, read
//! together into one aligned buffer.
//!
//! * Experts live in a global cache with a fixed capacity (from the RAM
//!   budget). Admission and eviction follow LFRU with hysteresis
//!   (Colibrì `c/tier.h`): frequency dominates, recency breaks ties, and a
//!   newcomer must beat the victim by 25% + 4 counts. `Lru` is an ablation.
//! * An expert that is not admitted is still served, through a small pool of
//!   scratch buffers that are not cached. A cold expert never evicts a hot one.
//! * Every buffer handed out is an `Arc`. Eviction only considers entries whose
//!   `Arc` is not shared and whose load has completed, which is the lease
//!   contract.
//! * [`ExpertStore::prefetch`] issues asynchronous loads for predicted experts
//!   (router lookahead). It is advisory: it never evicts a leased, loading or
//!   hotter entry.
//! * Usage counts can be saved and loaded so the next session starts warm.

use crate::io::{IoEngine, Ticket};
use crate::ledger::{BudgetError, Ledger, Reservation, Tier};
use crate::store::{ExtentLayout, StoreError};
use kestrel_hw::fileio::{AlignedBuf, IoMode, ReadFile, DIRECT_ALIGN};
use kestrel_model::{Extent, ModelDesc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExpertPolicy {
    Lfru,
    Lru,
}

/// Where the three projections of every expert live in the file.
#[derive(Clone, Debug)]
pub struct ExpertGeometry {
    /// Per MoE layer: (layer index, [tensor index; 3] for gate, up, down).
    pub layers: Vec<(u32, [usize; 3])>,
    pub n_expert: u32,
}

impl ExpertGeometry {
    pub fn from_model(m: &ModelDesc) -> Option<Self> {
        let moe = m.moe.as_ref()?;
        let mut layers = Vec::new();
        for l in 0..m.hparams.n_layer {
            let find = |s: &str| m.tensors.iter().position(|t| t.name == format!("blk.{l}.ffn_{s}_exps.weight"));
            if let (Some(g), Some(u), Some(d)) = (find("gate"), find("up"), find("down")) {
                layers.push((l, [g, u, d]));
            }
        }
        (!layers.is_empty()).then_some(ExpertGeometry { layers, n_expert: moe.n_expert })
    }
}

pub struct ExpertBuf {
    buf: AlignedBuf,
    ticket: Arc<Ticket>,
    /// (start, len) of gate, up, down inside `buf`.
    parts: [(usize, usize); 3],
    _res: Reservation,
}

impl ExpertBuf {
    /// Bytes of projection `i` (0 = gate, 1 = up, 2 = down).
    pub fn part(&self, i: usize) -> &[u8] {
        let (s, l) = self.parts[i];
        &self.buf.as_slice()[s..s + l]
    }
}

/// A leased expert. Keeps the buffer alive and un-evictable while held.
pub type ExpertLease = Arc<ExpertBuf>;

struct Entry {
    buf: Arc<ExpertBuf>,
    prefetched: bool,
    /// Loaded by `request` for an imminent demand; its miss is already counted.
    demand: bool,
}

#[derive(Default)]
struct State {
    cache: HashMap<(u32, u32), Entry>,
    heat: HashMap<(u32, u32), u32>,
    last: HashMap<(u32, u32), u64>,
    /// Lifetime usage counts (persisted).
    usage: HashMap<(u32, u32), u64>,
    clock: u64,
    scratch: Vec<Arc<ExpertBuf>>,
    /// Speculative loads (router lookahead) that the cache would not admit.
    spec: Vec<((u32, u32), Arc<ExpertBuf>)>,
    /// Adaptive gate for speculative loads: (issued, used) in the current
    /// window, and how many prefetch calls to skip while paused.
    spec_window: (u64, u64),
    spec_paused: u64,
    /// Recycled buffers from evictions, reused to avoid reallocation.
    free: Vec<(AlignedBuf, Reservation)>,
}

#[derive(Default)]
struct Metrics {
    requests: AtomicU64,
    hits: AtomicU64,
    late: AtomicU64,
    misses: AtomicU64,
    scratch_loads: AtomicU64,
    bytes_read: AtomicU64,
    stall_ns: AtomicU64,
    prefetch_issued: AtomicU64,
    prefetch_used: AtomicU64,
    prefetch_wasted: AtomicU64,
    evictions: AtomicU64,
    lookahead_predicted: AtomicU64,
    lookahead_correct: AtomicU64,
    spec_pauses: AtomicU64,
    reads: AtomicU64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ExpertMetrics {
    pub capacity: usize,
    pub cached: usize,
    pub requests: u64,
    pub hits: u64,
    /// Requested while its (pre)fetch was still in flight.
    pub late: u64,
    pub misses: u64,
    pub scratch_loads: u64,
    pub bytes_read: u64,
    pub stall_s: f64,
    pub prefetch_issued: u64,
    pub prefetch_used: u64,
    pub prefetch_wasted: u64,
    pub evictions: u64,
    pub hit_rate: f64,
    /// Router-lookahead recall: predicted experts that were then routed.
    pub lookahead_predicted: u64,
    pub lookahead_correct: u64,
    pub lookahead_recall: f64,
    /// Times the speculative pool was paused for low accuracy.
    pub spec_pauses: u64,
    /// File extents read (one expert is 3 in a plain GGUF, 1 when prepared).
    pub reads: u64,
}

impl ExpertMetrics {
    /// Counters accumulated since `earlier`.
    pub fn delta(&self, earlier: &ExpertMetrics) -> ExpertMetrics {
        let mut d = self.clone();
        d.requests -= earlier.requests;
        d.hits -= earlier.hits;
        d.late -= earlier.late;
        d.misses -= earlier.misses;
        d.scratch_loads -= earlier.scratch_loads;
        d.bytes_read -= earlier.bytes_read;
        d.stall_s -= earlier.stall_s;
        d.prefetch_issued -= earlier.prefetch_issued;
        d.prefetch_used -= earlier.prefetch_used;
        d.prefetch_wasted -= earlier.prefetch_wasted;
        d.evictions -= earlier.evictions;
        d.lookahead_predicted -= earlier.lookahead_predicted;
        d.lookahead_correct -= earlier.lookahead_correct;
        d.spec_pauses -= earlier.spec_pauses;
        d.reads -= earlier.reads;
        d.hit_rate = if d.requests == 0 { 0.0 } else { d.hits as f64 / d.requests as f64 };
        d.lookahead_recall = if d.lookahead_predicted == 0 { 0.0 } else { d.lookahead_correct as f64 / d.lookahead_predicted as f64 };
        d
    }
}

/// See [`ExpertStore::map`].
#[derive(Clone, Debug, Serialize)]
pub struct ExpertMap {
    pub layers: Vec<u32>,
    pub n_expert: u32,
    pub capacity: usize,
    /// 0 = not cached, 1 = loading, 2 = cached.
    pub cached: Vec<u8>,
    pub usage: Vec<u64>,
    /// Expert requests since this expert was last routed (`u64::MAX`: never).
    pub age: Vec<u64>,
    pub clock: u64,
}

pub struct ExpertStore {
    geom: ExpertGeometry,
    /// Per MoE layer index → position in `geom.layers`.
    layer_pos: HashMap<u32, usize>,
    /// Per layer: byte length of one expert's gate, up, down slices.
    slice: Vec<[u64; 3]>,
    model: Arc<ModelDesc>,
    file: Arc<ReadFile>,
    /// Declared before `state`: dropping it joins the workers before the
    /// buffers their queued reads target are freed.
    io: Arc<IoEngine>,
    ledger: Arc<Ledger>,
    pub capacity: usize,
    policy: ExpertPolicy,
    state: Mutex<State>,
    m: Metrics,
    scratch_slots: usize,
    /// Buffers for speculative loads outside the cache (0 disables).
    pub spec_slots: usize,
    /// Minimum fraction of speculative loads that must be used for the pool
    /// to stay enabled. On a saturated disk a wasted speculative read costs
    /// as much as a hit saves (Colibrì observed the same for PILOT), so the
    /// pool is paused when measured accuracy is below this and re-probed later.
    /// Default 0.5: a used speculative read saves one demand read, a wasted one
    /// costs one, so below half it cannot pay off on a disk-bound host.
    pub spec_min_accuracy: f64,
}

impl ExpertStore {
    /// `capacity` experts are cached (RAM-reserved lazily as they load);
    /// `scratch_slots` buffers serve non-admitted experts.
    pub fn new(model: Arc<ModelDesc>, capacity: usize, scratch_slots: usize, io_mode: IoMode, io_workers: usize, policy: ExpertPolicy, ledger: Arc<Ledger>) -> Result<Self, StoreError> {
        let geom = ExpertGeometry::from_model(&model).ok_or_else(|| StoreError::Config("model has no routed experts".into()))?;
        let file = Arc::new(ReadFile::open(&model.path, io_mode).map_err(|e| StoreError::Io(e.to_string()))?);
        let n = geom.n_expert as u64;
        let mut slice = Vec::new();
        for (_, ts) in &geom.layers {
            let s: Vec<u64> = ts.iter().map(|&t| model.tensors[t].size / n).collect();
            slice.push([s[0], s[1], s[2]]);
        }
        let layer_pos = geom.layers.iter().enumerate().map(|(i, (l, _))| (*l, i)).collect();
        Ok(ExpertStore {
            geom,
            layer_pos,
            slice,
            model,
            file,
            io: IoEngine::new(io_workers),
            ledger,
            capacity,
            policy,
            state: Mutex::new(State::default()),
            m: Metrics::default(),
            scratch_slots: scratch_slots.max(1),
            spec_slots: scratch_slots.max(1),
            spec_min_accuracy: 0.5,
        })
    }

    /// The file ranges of expert `e`'s gate, up and down slices. In a plain
    /// GGUF they are three scattered ranges; in a prepared container
    /// (`kestrel prepare`) they are adjacent.
    fn parts_of(model: &ModelDesc, geom: &ExpertGeometry, li: usize, e: u32) -> [Extent; 3] {
        let ts = geom.layers[li].1;
        [0, 1, 2].map(|k| {
            let t = &model.tensors[ts[k]];
            Extent { offset: t.slice_offset(e as u64), len: t.slice_bytes() }
        })
    }

    /// The reads that load expert `e`: its parts, with adjacent ones merged.
    fn extents_of(parts: &[Extent; 3]) -> Vec<Extent> {
        let mut ext = parts.to_vec();
        ext.sort_by_key(|x| x.offset);
        kestrel_model::coalesce(&ext, DIRECT_ALIGN as u64)
    }

    pub fn bytes_per_expert(&self, layer: u32) -> u64 {
        self.slice[self.layer_pos[&layer]].iter().sum()
    }

    pub fn is_moe_layer(&self, layer: u32) -> bool {
        self.layer_pos.contains_key(&layer)
    }

    /// Buffer size that fits any expert of any layer: an extent of `len`
    /// bytes at an arbitrary offset spans at most `align_up(len) + align`.
    fn buf_len(&self) -> usize {
        let a = DIRECT_ALIGN as u64;
        self.slice.iter().map(|s| s.iter().map(|&len| len.div_ceil(a) * a + a).sum::<u64>() as usize).max().unwrap_or(DIRECT_ALIGN)
    }

    fn new_buffer(&self, st: &mut State, purpose: &str) -> Result<(AlignedBuf, Reservation), StoreError> {
        if let Some(b) = st.free.pop() {
            return Ok(b);
        }
        let len = self.buf_len();
        let r = self.ledger.reserve(Tier::Ram, len as u64, purpose).map_err(|e: BudgetError| StoreError::Budget(e))?;
        Ok((AlignedBuf::new(len, DIRECT_ALIGN).map_err(|e| StoreError::Io(e.to_string()))?, r))
    }

    fn load(&self, layer: u32, e: u32, (mut buf, res): (AlignedBuf, Reservation)) -> Arc<ExpertBuf> {
        let li = self.layer_pos[&layer];
        let pe = Self::parts_of(&self.model, &self.geom, li, e);
        let ext = Self::extents_of(&pe);
        let layout = ExtentLayout::new(&ext);
        let ticket = layout.issue(&self.io, &self.file, &mut buf);
        let parts = pe.map(|x| (layout.position(x.offset, x.len), x.len as usize));
        self.m.bytes_read.fetch_add(ext.iter().map(|x| x.len).sum(), Relaxed);
        self.m.reads.fetch_add(ext.len() as u64, Relaxed);
        Arc::new(ExpertBuf { buf, ticket, parts, _res: res })
    }

    fn score(st: &State, k: &(u32, u32), policy: ExpertPolicy) -> u64 {
        let last = st.last.get(k).copied().unwrap_or(0);
        match policy {
            ExpertPolicy::Lru => last,
            ExpertPolicy::Lfru => {
                let heat = *st.heat.get(k).unwrap_or(&0) as u64;
                let age = st.clock - last;
                (heat << 8) | 255u64.saturating_sub(age)
            }
        }
    }

    /// Victim for admitting `k`, or None if `k` should not displace anyone.
    fn victim(&self, st: &State, k: &(u32, u32), force: bool) -> Option<(u32, u32)> {
        let evictable = st.cache.iter().filter(|(_, en)| Arc::strong_count(&en.buf) == 1 && en.buf.ticket.is_done());
        let (vk, _) = evictable.min_by_key(|(kk, _)| Self::score(st, kk, self.policy))?;
        if force || self.policy == ExpertPolicy::Lru {
            return Some(*vk);
        }
        let (cs, hs) = (Self::score(st, vk, self.policy), Self::score(st, k, self.policy));
        (hs > cs + (cs >> 2) + (4 << 8)).then_some(*vk)
    }

    /// Insert `k` (loading it) if admissible; returns the new entry.
    fn admit(&self, st: &mut State, k: (u32, u32), prefetch: bool) -> Result<Option<Arc<ExpertBuf>>, StoreError> {
        let buf = if st.cache.len() < self.capacity {
            self.new_buffer(st, "expert-cache")?
        } else {
            let Some(v) = self.victim(st, &k, false) else { return Ok(None) };
            let old = st.cache.remove(&v).unwrap();
            self.m.evictions.fetch_add(1, Relaxed);
            if old.prefetched {
                self.m.prefetch_wasted.fetch_add(1, Relaxed);
            }
            match Arc::try_unwrap(old.buf) {
                Ok(b) => (b.buf, b._res),
                Err(_) => unreachable!("victims are unshared"),
            }
        };
        let eb = self.load(k.0, k.1, buf);
        st.cache.insert(k, Entry { buf: eb.clone(), prefetched: prefetch, demand: false });
        Ok(Some(eb))
    }

    fn touch(st: &mut State, k: (u32, u32)) {
        st.clock += 1;
        let h = st.heat.entry(k).or_insert(0);
        *h = h.saturating_add(1);
        st.last.insert(k, st.clock);
        *st.usage.entry(k).or_insert(0) += 1;
    }

    /// Lease expert `e` of `layer`, loading it if needed. Blocks until ready.
    pub fn get(&self, layer: u32, e: u32) -> Result<ExpertLease, StoreError> {
        let k = (layer, e);
        self.m.requests.fetch_add(1, Relaxed);
        let lease = {
            let mut st = self.state.lock().unwrap();
            Self::touch(&mut st, k);
            if let Some(en) = st.cache.get_mut(&k) {
                if en.prefetched {
                    en.prefetched = false;
                    self.m.prefetch_used.fetch_add(1, Relaxed);
                }
                if en.demand {
                    en.demand = false; // counted as a miss by `request`
                } else if en.buf.ticket.is_done() {
                    self.m.hits.fetch_add(1, Relaxed);
                } else {
                    self.m.late.fetch_add(1, Relaxed);
                }
                en.buf.clone()
            } else if let Some(i) = st.spec.iter().position(|(kk, _)| *kk == k) {
                // Predicted by lookahead but not cached: served from the
                // speculative pool (then recycled as scratch).
                let (_, b) = st.spec.swap_remove(i);
                st.spec_window.1 += 1;
                self.m.prefetch_used.fetch_add(1, Relaxed);
                if b.ticket.is_done() {
                    self.m.hits.fetch_add(1, Relaxed);
                } else {
                    self.m.late.fetch_add(1, Relaxed);
                }
                st.scratch.push(b.clone());
                b
            } else {
                self.m.misses.fetch_add(1, Relaxed);
                match self.admit(&mut st, k, false)? {
                    Some(b) => b,
                    None => {
                        // Not hot enough to cache: stream through scratch.
                        self.m.scratch_loads.fetch_add(1, Relaxed);
                        let idx = st.scratch.iter().position(|s| Arc::strong_count(s) == 1 && s.ticket.is_done());
                        let buf = match idx {
                            Some(i) => {
                                let old = st.scratch.swap_remove(i);
                                match Arc::try_unwrap(old) {
                                    Ok(b) => (b.buf, b._res),
                                    Err(_) => unreachable!(),
                                }
                            }
                            None if st.scratch.len() < self.scratch_slots => self.new_buffer(&mut st, "expert-scratch")?,
                            None => {
                                // All scratch buffers busy: wait for one.
                                let t = st.scratch[0].ticket.clone();
                                drop(st);
                                let _ = t.wait();
                                return self.get_uncounted(layer, e);
                            }
                        };
                        let eb = self.load(layer, e, buf);
                        st.scratch.push(eb.clone());
                        eb
                    }
                }
            }
        };
        let waited = lease.ticket.wait().map_err(StoreError::Io)?;
        self.m.stall_ns.fetch_add(waited.as_nanos() as u64, Relaxed);
        Ok(lease)
    }

    fn get_uncounted(&self, layer: u32, e: u32) -> Result<ExpertLease, StoreError> {
        self.m.requests.fetch_sub(1, Relaxed);
        self.m.misses.fetch_sub(1, Relaxed);
        self.m.scratch_loads.fetch_sub(1, Relaxed);
        {
            let mut st = self.state.lock().unwrap();
            st.clock -= 1;
            if let Some(u) = st.usage.get_mut(&(layer, e)) {
                *u -= 1;
            }
            if let Some(h) = st.heat.get_mut(&(layer, e)) {
                *h = h.saturating_sub(1);
            }
        }
        self.get(layer, e)
    }

    /// Cached and fully loaded.
    pub fn is_ready(&self, layer: u32, e: u32) -> bool {
        self.state.lock().unwrap().cache.get(&(layer, e)).is_some_and(|en| en.buf.ticket.is_done())
    }

    /// Advisory: start loading predicted experts that are not cached, if the
    /// cache would admit them. Never blocks on I/O.
    pub fn prefetch(&self, layer: u32, experts: &[u32]) {
        if !self.is_moe_layer(layer) {
            return;
        }
        let mut st = self.state.lock().unwrap();
        for &e in experts {
            let k = (layer, e);
            if st.cache.contains_key(&k) {
                continue;
            }
            // Admission for a prediction uses its *current* heat (no touch):
            // a guess must not inflate the heat of what it guesses.
            if let Ok(Some(_)) = self.admit(&mut st, k, true) {
                self.m.prefetch_issued.fetch_add(1, Relaxed);
                continue;
            }
            // Not admissible: load into the speculative pool instead, never
            // displacing a cached expert — if the pool has been paying off.
            if st.spec_paused > 0 {
                st.spec_paused -= 1;
                if st.spec_paused == 0 {
                    st.spec_window = (0, 0); // re-probe
                }
                continue;
            }
            if self.spec_slots == 0 || st.spec.iter().any(|(kk, _)| *kk == k) {
                continue;
            }
            if st.spec_window.0 >= 128 {
                let acc = st.spec_window.1 as f64 / st.spec_window.0 as f64;
                st.spec_window = (0, 0);
                if acc < self.spec_min_accuracy {
                    self.m.spec_pauses.fetch_add(1, Relaxed);
                    st.spec_paused = 4096;
                    continue;
                }
            }
            let buf = if st.spec.len() < self.spec_slots {
                // Reuse an idle scratch buffer or allocate a new one.
                match st.scratch.iter().position(|b| Arc::strong_count(b) == 1 && b.ticket.is_done()).filter(|_| st.scratch.len() > self.scratch_slots) {
                    Some(i) => match Arc::try_unwrap(st.scratch.swap_remove(i)) {
                        Ok(b) => Some((b.buf, b._res)),
                        Err(_) => None,
                    },
                    None => self.new_buffer(&mut st, "expert-prefetch").ok(),
                }
            } else {
                // Replace the oldest idle speculative load (stale prediction).
                match st.spec.iter().position(|(_, b)| Arc::strong_count(b) == 1 && b.ticket.is_done()) {
                    Some(i) => {
                        self.m.prefetch_wasted.fetch_add(1, Relaxed);
                        match Arc::try_unwrap(st.spec.remove(i).1) {
                            Ok(b) => Some((b.buf, b._res)),
                            Err(_) => None,
                        }
                    }
                    None => None,
                }
            };
            if let Some(buf) = buf {
                let eb = self.load(layer, e, buf);
                st.spec.push((k, eb));
                st.spec_window.0 += 1;
                self.m.prefetch_issued.fetch_add(1, Relaxed);
            }
        }
    }

    /// Announce the experts a layer is about to use (batch union) so the
    /// missing ones load in parallel. Unlike [`prefetch`](Self::prefetch) these
    /// are real demands: each one not cached counts as a miss now, and the
    /// following [`get`](Self::get) does not count it again.
    pub fn request(&self, layer: u32, experts: &[u32]) -> Result<(), StoreError> {
        let mut st = self.state.lock().unwrap();
        for &e in experts {
            let k = (layer, e);
            if st.cache.contains_key(&k) {
                continue;
            }
            if let Some(_) = self.admit(&mut st, k, false)? {
                self.m.misses.fetch_add(1, Relaxed);
                st.cache.get_mut(&k).unwrap().demand = true;
            }
        }
        Ok(())
    }

    /// Record router-lookahead accuracy (engine-supplied).
    pub fn record_lookahead(&self, predicted: u64, correct: u64) {
        self.m.lookahead_predicted.fetch_add(predicted, Relaxed);
        self.m.lookahead_correct.fetch_add(correct, Relaxed);
    }

    /// Halve all heat so placement follows the workload.
    pub fn decay(&self) {
        let mut st = self.state.lock().unwrap();
        st.heat.retain(|_, h| {
            *h >>= 1;
            *h > 0
        });
    }

    pub fn metrics(&self) -> ExpertMetrics {
        let g = |a: &AtomicU64| a.load(Relaxed);
        let req = g(&self.m.requests);
        let pred = g(&self.m.lookahead_predicted);
        ExpertMetrics {
            capacity: self.capacity,
            cached: self.state.lock().unwrap().cache.len(),
            requests: req,
            hits: g(&self.m.hits),
            late: g(&self.m.late),
            misses: g(&self.m.misses),
            scratch_loads: g(&self.m.scratch_loads),
            bytes_read: g(&self.m.bytes_read),
            stall_s: g(&self.m.stall_ns) as f64 * 1e-9,
            prefetch_issued: g(&self.m.prefetch_issued),
            prefetch_used: g(&self.m.prefetch_used),
            prefetch_wasted: g(&self.m.prefetch_wasted),
            evictions: g(&self.m.evictions),
            hit_rate: if req == 0 { 0.0 } else { g(&self.m.hits) as f64 / req as f64 },
            lookahead_predicted: pred,
            lookahead_correct: g(&self.m.lookahead_correct),
            lookahead_recall: if pred == 0 { 0.0 } else { g(&self.m.lookahead_correct) as f64 / pred as f64 },
            spec_pauses: g(&self.m.spec_pauses),
            reads: g(&self.m.reads),
        }
    }

    /// Lifetime usage counts, for persistence.
    pub fn usage(&self) -> Vec<(u32, u32, u64)> {
        let st = self.state.lock().unwrap();
        let mut v: Vec<(u32, u32, u64)> = st.usage.iter().map(|(&(l, e), &c)| (l, e, c)).collect();
        v.sort_unstable();
        v
    }

    /// A snapshot of every routed expert for monitoring: whether it is
    /// cached, its lifetime use count, and how many requests ago it was last
    /// routed. Arrays are `[layer][expert]`, flattened in `layers` order.
    pub fn map(&self) -> ExpertMap {
        let st = self.state.lock().unwrap();
        let n = self.geom.n_expert as usize;
        let layers: Vec<u32> = self.geom.layers.iter().map(|(l, _)| *l).collect();
        let total = layers.len() * n;
        let (mut cached, mut usage, mut age) = (vec![0u8; total], vec![0u64; total], vec![u64::MAX; total]);
        for (li, &l) in layers.iter().enumerate() {
            for e in 0..n {
                let k = (l, e as u32);
                let i = li * n + e;
                if let Some(en) = st.cache.get(&k) {
                    cached[i] = if en.buf.ticket.is_done() { 2 } else { 1 };
                }
                usage[i] = st.usage.get(&k).copied().unwrap_or(0);
                if let Some(&t) = st.last.get(&k) {
                    age[i] = st.clock.saturating_sub(t);
                }
            }
        }
        ExpertMap { layers, n_expert: n as u32, capacity: self.capacity, cached, usage, age, clock: st.clock }
    }

    /// Total number of routed experts in the model.
    pub fn n_experts_total(&self) -> usize {
        self.geom.layers.len() * self.geom.n_expert as usize
    }

    /// Load every expert (when the cache can hold them all). Returns the
    /// number loaded.
    pub fn preload_all(&self) -> Result<usize, StoreError> {
        if self.capacity < self.n_experts_total() {
            return Ok(0);
        }
        let mut st = self.state.lock().unwrap();
        let mut pending = Vec::new();
        for &(l, _) in &self.geom.layers {
            for e in 0..self.geom.n_expert {
                if !st.cache.contains_key(&(l, e)) {
                    if let Some(b) = self.admit(&mut st, (l, e), false)? {
                        pending.push(b);
                    }
                }
            }
        }
        drop(st);
        let n = pending.len();
        for b in pending {
            b.ticket.wait().map_err(StoreError::Io)?;
        }
        Ok(n)
    }

    /// Warm start: seed heat from a usage history and preload the hottest
    /// experts up to capacity. Returns the number preloaded.
    pub fn warm_start(&self, usage: &[(u32, u32, u64)]) -> Result<usize, StoreError> {
        let mut ranked: Vec<&(u32, u32, u64)> = usage.iter().filter(|(l, e, _)| self.is_moe_layer(*l) && *e < self.geom.n_expert).collect();
        ranked.sort_by(|a, b| b.2.cmp(&a.2));
        let mut st = self.state.lock().unwrap();
        let max = ranked.first().map(|x| x.2).unwrap_or(1).max(1);
        for &&(l, e, c) in &ranked {
            // Scale persisted counts into the session heat range.
            st.heat.insert((l, e), ((c * 64) / max).max(1) as u32);
            *st.usage.entry((l, e)).or_insert(0) += c;
        }
        let mut n = 0;
        let mut pending = Vec::new();
        for &&(l, e, _) in ranked.iter().take(self.capacity) {
            if let Some(b) = self.admit(&mut st, (l, e), false)? {
                pending.push(b);
                n += 1;
            }
        }
        drop(st);
        for b in pending {
            b.ticket.wait().map_err(StoreError::Io)?;
        }
        Ok(n)
    }
}

/// Usage-history persistence next to the model: `<model>.kestrel-usage.json`,
/// keyed by the model fingerprint so a different model never loads it.
#[derive(Serialize, Deserialize)]
pub struct UsageFile {
    pub fingerprint: String,
    pub n_expert: u32,
    /// (layer, expert, count)
    pub usage: Vec<(u32, u32, u64)>,
}

impl UsageFile {
    pub fn path_for(model: &ModelDesc) -> std::path::PathBuf {
        let mut p = model.path.clone().into_os_string();
        p.push(".kestrel-usage.json");
        p.into()
    }
    pub fn load(model: &ModelDesc) -> Option<Vec<(u32, u32, u64)>> {
        let text = std::fs::read_to_string(Self::path_for(model)).ok()?;
        let f: UsageFile = serde_json::from_str(&text).ok()?;
        (f.fingerprint == model.fingerprint).then_some(f.usage)
    }
    pub fn save(model: &ModelDesc, usage: Vec<(u32, u32, u64)>) -> std::io::Result<()> {
        let f = UsageFile { fingerprint: model.fingerprint.clone(), n_expert: model.moe.as_ref().map(|m| m.n_expert).unwrap_or(0), usage };
        let p = Self::path_for(model);
        let tmp = p.with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_string(&f).unwrap())?;
        std::fs::rename(tmp, p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kestrel_gguf::writer::{GgufWriter, TensorData};
    use kestrel_gguf::Value;

    fn moe_model(dir: &std::path::Path) -> Arc<ModelDesc> {
        let p = dir.join("moe.gguf");
        let mut w = GgufWriter::new();
        w.kv("general.architecture", Value::String("qwen3moe".into()));
        w.kv("qwen3moe.block_count", Value::U32(2));
        w.kv("qwen3moe.embedding_length", Value::U32(64));
        w.kv("qwen3moe.attention.head_count", Value::U32(4));
        w.kv("qwen3moe.expert_count", Value::U32(8));
        w.kv("qwen3moe.expert_used_count", Value::U32(2));
        for l in 0..2u32 {
            for (k, name) in ["gate", "up", "down"].iter().enumerate() {
                // value = layer*1000 + expert*10 + projection, constant per expert slice
                let data: Vec<f32> = (0..8u32).flat_map(|e| std::iter::repeat((l * 1000 + e * 10 + k as u32) as f32).take(64 * 32)).collect();
                w.tensor(&format!("blk.{l}.ffn_{name}_exps.weight"), &[64, 32, 8], TensorData::F32(data));
            }
        }
        w.write(&p).unwrap();
        Arc::new(ModelDesc::open(&p).unwrap())
    }

    fn val(b: &ExpertBuf, part: usize) -> f32 {
        f32::from_le_bytes(b.part(part)[..4].try_into().unwrap())
    }

    #[test]
    fn serves_correct_slices_and_respects_capacity() {
        let d = tempfile::tempdir().unwrap();
        let m = moe_model(d.path());
        let ledger = Ledger::new(0, 1 << 30, 0);
        let s = ExpertStore::new(m, 3, 2, IoMode::Direct, 2, ExpertPolicy::Lfru, ledger.clone()).unwrap();
        for _ in 0..3 {
            for l in 0..2 {
                for e in 0..8 {
                    let x = s.get(l, e).unwrap();
                    for k in 0..3 {
                        assert_eq!(val(&x, k), (l * 1000 + e * 10 + k as u32) as f32);
                        assert_eq!(x.part(k).len(), 64 * 32 * 4);
                    }
                }
            }
        }
        let mt = s.metrics();
        assert!(mt.cached <= 3);
        assert!(mt.scratch_loads > 0, "cold experts stream through scratch: {mt:?}");
        // RAM: at most capacity + scratch buffers.
        assert!(ledger.usage(Tier::Ram).reserved <= 5 * s.buf_len() as u64);
    }

    #[test]
    fn prepared_container_serves_identical_experts_in_one_read() {
        let d = tempfile::tempdir().unwrap();
        let m = moe_model(d.path());
        let g = kestrel_gguf::GgufFile::open(&m.path).unwrap();
        let out = d.path().join("moe.kgguf");
        kestrel_model::prepare::prepare(&g, &m, &out, true, |_, _| {}).unwrap();
        let pm = Arc::new(ModelDesc::open(&out).unwrap());
        let open = |m: &Arc<ModelDesc>| ExpertStore::new(m.clone(), 16, 2, IoMode::Direct, 2, ExpertPolicy::Lfru, Ledger::new(0, 1 << 30, 0)).unwrap();
        let (a, b) = (open(&m), open(&pm));
        for l in 0..2 {
            for e in 0..8 {
                let (x, y) = (a.get(l, e).unwrap(), b.get(l, e).unwrap());
                for k in 0..3 {
                    assert_eq!(x.part(k), y.part(k), "layer {l} expert {e} part {k}");
                }
            }
        }
        assert_eq!(a.metrics().reads, 3 * 16);
        assert_eq!(b.metrics().reads, 16);
        assert_eq!(a.metrics().bytes_read, b.metrics().bytes_read);
    }

    #[test]
    fn hot_experts_stay_cached_under_lfru() {
        let d = tempfile::tempdir().unwrap();
        let m = moe_model(d.path());
        let s = ExpertStore::new(m, 2, 2, IoMode::Buffered, 2, ExpertPolicy::Lfru, Ledger::new(0, 1 << 30, 0)).unwrap();
        // Experts 0 and 1 are hot; 2..8 appear once each between them.
        for round in 0..20u32 {
            s.get(0, 0).unwrap();
            s.get(0, 1).unwrap();
            s.get(0, 2 + round % 6).unwrap();
        }
        let mt = s.metrics();
        assert!(mt.hits >= 36, "hot set stays resident: {mt:?}");
        let lru = ExpertStore::new(s.model.clone(), 2, 2, IoMode::Buffered, 2, ExpertPolicy::Lru, Ledger::new(0, 1 << 30, 0)).unwrap();
        for round in 0..20u32 {
            lru.get(0, 0).unwrap();
            lru.get(0, 1).unwrap();
            lru.get(0, 2 + round % 6).unwrap();
        }
        assert!(lru.metrics().hits < mt.hits, "LRU lets one-off experts evict hot ones");
    }

    #[test]
    fn prefetch_then_hit_and_warm_start() {
        let d = tempfile::tempdir().unwrap();
        let m = moe_model(d.path());
        let s = ExpertStore::new(m.clone(), 4, 2, IoMode::Direct, 2, ExpertPolicy::Lfru, Ledger::new(0, 1 << 30, 0)).unwrap();
        s.prefetch(1, &[3, 5]);
        let x = s.get(1, 3).unwrap();
        assert_eq!(val(&x, 2), 1032.0);
        let mt = s.metrics();
        assert_eq!((mt.prefetch_issued, mt.prefetch_used, mt.misses), (2, 1, 0));
        s.get(1, 5).unwrap();
        UsageFile::save(&m, s.usage()).unwrap();
        let hist = UsageFile::load(&m).unwrap();
        let s2 = ExpertStore::new(m, 4, 2, IoMode::Direct, 2, ExpertPolicy::Lfru, Ledger::new(0, 1 << 30, 0)).unwrap();
        assert_eq!(s2.warm_start(&hist).unwrap(), 2);
        s2.get(1, 5).unwrap();
        assert_eq!(s2.metrics().hits, 1);
    }
}
