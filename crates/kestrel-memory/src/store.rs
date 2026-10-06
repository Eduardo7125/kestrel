//! The tier-aware weight store.
//!
//! Every weight access of the native executor goes through
//! [`WeightStore::lease`]. A group is either **resident** (loaded once into an
//! owned buffer) or **streamed** (read from the model file into one slot of a
//! bounded ring). Leasing a streamed group automatically prefetches the next
//! `prefetch_depth` streamed groups in execution order, so I/O overlaps
//! compute.
//!
//! Ring eviction is **Belady-optimal** for the deterministic, cyclic access
//! order of dense layers: the victim is the slot whose group is needed
//! furthest in the future. LRU is available only as an ablation; on a cyclic
//! scan larger than the ring its hit rate is zero (`docs/scheduler-design.md`).

use crate::io::{ChunkJob, DstPtr, IoEngine, Ticket, CHUNK};
use crate::ledger::{BudgetError, Ledger, Reservation, Tier};
use crate::metrics::{MetricsSnapshot, StoreMetrics};
use kestrel_hw::fileio::{AlignedBuf, IoMode, ReadFile, DIRECT_ALIGN};
use kestrel_model::{Extent, ModelDesc};
use serde::{Deserialize, Serialize};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error(transparent)]
    Budget(#[from] BudgetError),
    #[error("I/O error: {0}")]
    Io(String),
    #[error("{0}")]
    Config(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RingPolicy {
    /// Evict the group needed furthest in the future (optimal for known order).
    Belady,
    /// Evict the least recently used group (ablation).
    Lru,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoreConfig {
    pub io_mode: IoMode,
    /// Streamed groups to load ahead of the one being used.
    pub prefetch_depth: usize,
    /// Ring slots; at least `prefetch_depth + 1`.
    pub ring_slots: usize,
    pub io_workers: usize,
    pub policy: RingPolicy,
    /// In buffered mode, advise the OS to drop pages after reading so the
    /// page cache does not hold a second copy. `false` emulates mmap-style
    /// page-cache streaming.
    pub drop_page_cache: bool,
    /// Routed-expert groups are managed by an [`crate::ExpertStore`]: the
    /// weight store neither loads nor streams them.
    #[serde(default)]
    pub external_experts: bool,
}

impl Default for StoreConfig {
    fn default() -> Self {
        StoreConfig { io_mode: IoMode::Direct, prefetch_depth: 2, ring_slots: 3, io_workers: 8, policy: RingPolicy::Belady, drop_page_cache: true, external_experts: false }
    }
}

/// A set of file extents laid out in one aligned buffer so that each can be
/// read with direct I/O straight into place.
#[derive(Debug, Clone)]
pub struct ExtentLayout {
    /// (extent, buffer position of the aligned span, head = offset % align, span)
    pieces: Vec<(Extent, usize, usize, usize)>,
    pub buf_len: usize,
}

impl ExtentLayout {
    pub fn new(extents: &[Extent]) -> Self {
        let a = DIRECT_ALIGN as u64;
        let mut pieces = Vec::new();
        let mut pos = 0usize;
        for e in extents {
            let start = e.offset / a * a;
            let head = (e.offset - start) as usize;
            let span = ((e.offset + e.len).div_ceil(a) * a - start) as usize;
            pieces.push((*e, pos, head, span));
            pos += span;
        }
        ExtentLayout { pieces, buf_len: pos.max(DIRECT_ALIGN) }
    }

    /// Buffer position of file range `[offset, offset+len)`, which must lie
    /// inside one extent.
    pub fn position(&self, offset: u64, len: u64) -> usize {
        let (e, bpos, head, _) = self.pieces.iter().find(|(e, ..)| offset >= e.offset && offset + len <= e.offset + e.len).expect("range inside the layout's extents");
        bpos + head + (offset - e.offset) as usize
    }

    /// Plan the chunk reads that fill `buf` and submit them.
    pub(crate) fn issue(&self, io: &IoEngine, file: &Arc<ReadFile>, buf: &mut AlignedBuf) -> Arc<Ticket> {
        // The I/O workers write through raw pointers: never trust the caller.
        assert!(buf.len() >= self.buf_len, "buffer of {} bytes for a layout of {}", buf.len(), self.buf_len);
        let base = buf.as_mut_slice().as_mut_ptr();
        let mut plan = Vec::new();
        for &(e, bpos, head, span) in &self.pieces {
            let file_start = e.offset - head as u64;
            let needed = head + e.len as usize;
            let mut k = 0usize;
            while k < span {
                let len = CHUNK.min(span - k);
                let required = needed.saturating_sub(k).min(len);
                if required > 0 {
                    // SAFETY: bpos + k + len <= buf_len by construction.
                    plan.push((file_start + k as u64, DstPtr(unsafe { base.add(bpos + k) }), len, required));
                }
                k += len;
            }
        }
        let ticket = Ticket::new(plan.len());
        io.submit(
            plan.into_iter()
                .map(|(file_offset, dst, len, required)| ChunkJob { file: file.clone(), file_offset, dst, len, required, ticket: ticket.clone() })
                .collect(),
        );
        ticket
    }
}

/// Where each group's bytes sit inside its buffer.
#[derive(Debug)]
pub struct GroupLayout {
    extents: ExtentLayout,
    /// (tensor index, start in buffer, length)
    tensors: Vec<(usize, usize, usize)>,
    pub buf_len: usize,
    pub bytes: u64,
}

impl GroupLayout {
    fn new(model: &ModelDesc, group: usize) -> Self {
        let g = &model.groups[group];
        let extents = ExtentLayout::new(&g.extents);
        let tensors = g.tensors.iter().map(|&ti| {
            let t = &model.tensors[ti];
            (ti, extents.position(t.offset, t.size), t.size as usize)
        }).collect();
        GroupLayout { buf_len: extents.buf_len, extents, tensors, bytes: g.bytes }
    }

    fn issue(&self, io: &IoEngine, file: &Arc<ReadFile>, buf: &mut AlignedBuf) -> Arc<Ticket> {
        self.extents.issue(io, file, buf)
    }
}

/// An owned buffer holding one resident group.
pub struct GroupBuf {
    buf: AlignedBuf,
    _res: Reservation,
}

enum Placement {
    Resident(Arc<GroupBuf>),
    Streamed,
    /// Managed elsewhere (routed experts → ExpertStore).
    External,
}

struct Slot {
    buf: AlignedBuf,
    group: Option<usize>,
    ticket: Option<Arc<Ticket>>,
    leases: usize,
    last_use: u64,
    /// Loaded by prefetch and not yet leased.
    prefetched: bool,
    _res: Reservation,
}

struct State {
    placement: Vec<Placement>,
    slots: Vec<Slot>,
    clock: u64,
    /// Execution-order position of the most recently leased group.
    cur_pos: usize,
}

struct Inner {
    model: Arc<ModelDesc>,
    layouts: Vec<GroupLayout>,
    /// Position of each group in execution order.
    pos: Vec<usize>,
    /// Execution order of group ids.
    order: Vec<usize>,
    file: Arc<ReadFile>,
    cfg: StoreConfig,
    io: Arc<IoEngine>,
    ledger: Arc<Ledger>,
    state: Mutex<State>,
    metrics: StoreMetrics,
}

#[derive(Clone)]
pub struct WeightStore {
    inner: Arc<Inner>,
}

/// A borrowed view of one group's bytes. While alive, the group's buffer
/// cannot be evicted, reused or freed.
pub struct Lease {
    store: Option<Arc<Inner>>,
    slot: usize,
    resident: Option<Arc<GroupBuf>>,
    ptr: *const u8,
    len: usize,
    group: usize,
}

unsafe impl Send for Lease {}

impl Lease {
    fn bytes(&self) -> &[u8] {
        // SAFETY: the buffer is pinned by the Arc (resident) or by the slot's
        // lease count (streamed) for the lifetime of self.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    /// Bytes of tensor `tensor_idx` (index into `ModelDesc::tensors`).
    pub fn tensor(&self, tensor_idx: usize) -> &[u8] {
        let layouts = match &self.store {
            Some(s) => &s.layouts,
            None => unreachable!(),
        };
        let &(_, start, len) = layouts[self.group]
            .tensors
            .iter()
            .find(|(t, ..)| *t == tensor_idx)
            .unwrap_or_else(|| panic!("tensor {tensor_idx} is not in group {}", self.group));
        &self.bytes()[start..start + len]
    }
    pub fn group(&self) -> usize {
        self.group
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        if self.resident.is_none() {
            if let Some(s) = &self.store {
                let mut st = s.state.lock().unwrap();
                st.slots[self.slot].leases -= 1;
            }
        }
    }
}

impl WeightStore {
    /// Build the store: load every group marked `resident` (in parallel) and
    /// reserve the streaming ring. `resident[g]` selects the tier of group g;
    /// `order` is the execution order of all groups.
    pub fn new(model: Arc<ModelDesc>, resident: &[bool], order: Vec<usize>, cfg: StoreConfig, ledger: Arc<Ledger>) -> Result<Self, StoreError> {
        let n = model.groups.len();
        if resident.len() != n {
            return Err(StoreError::Config(format!("placement has {} entries for {n} groups", resident.len())));
        }
        let mut pos = vec![usize::MAX; n];
        for (i, &g) in order.iter().enumerate() {
            pos[g] = i;
        }
        if pos.contains(&usize::MAX) {
            return Err(StoreError::Config("execution order must list every group".into()));
        }
        let file = Arc::new(ReadFile::open(&model.path, cfg.io_mode).map_err(|e| StoreError::Io(format!("{}: {e}", model.path.display())))?);
        let mut cfg = cfg;
        cfg.io_mode = file.mode();
        let layouts: Vec<GroupLayout> = (0..n).map(|g| GroupLayout::new(&model, g)).collect();
        let io = IoEngine::new(cfg.io_workers);
        let external: Vec<bool> = model.groups.iter().map(|g| cfg.external_experts && g.kind == kestrel_model::GroupKind::Experts).collect();
        let resident: Vec<bool> = resident.iter().zip(&external).map(|(r, e)| *r && !e).collect();

        // Resident groups: reserve, allocate, read in parallel.
        let mut placement = Vec::with_capacity(n);
        let mut pending = Vec::new();
        let mut bufs: Vec<Option<(AlignedBuf, Reservation)>> = Vec::with_capacity(n);
        for (g, &res) in resident.iter().enumerate() {
            if res {
                let r = ledger.reserve(Tier::Ram, layouts[g].buf_len as u64, "weights")?;
                let mut buf = AlignedBuf::new(layouts[g].buf_len, DIRECT_ALIGN).map_err(|e| StoreError::Io(e.to_string()))?;
                pending.push(layouts[g].issue(&io, &file, &mut buf));
                bufs.push(Some((buf, r)));
            } else {
                bufs.push(None);
            }
        }
        for t in pending {
            t.wait().map_err(StoreError::Io)?;
        }
        if cfg.io_mode == IoMode::Buffered && cfg.drop_page_cache {
            file.drop_cache(0, model.file_size);
        }
        for (g, b) in bufs.into_iter().enumerate() {
            placement.push(match b {
                Some((buf, r)) => Placement::Resident(Arc::new(GroupBuf { buf, _res: r })),
                None if external[g] => Placement::External,
                None => Placement::Streamed,
            });
        }

        // Ring: sized for the largest streamed group (or any group, so that
        // later demotions can stream through it).
        let any_streamed = resident.iter().zip(&external).any(|(r, e)| !r && !e);
        let slot_len = layouts.iter().enumerate().filter(|(g, _)| !resident[*g] && !external[*g]).map(|(_, l)| l.buf_len).max().unwrap_or(0);
        let mut slots = Vec::new();
        if any_streamed {
            if cfg.ring_slots < cfg.prefetch_depth + 1 {
                cfg.ring_slots = cfg.prefetch_depth + 1;
            }
            for _ in 0..cfg.ring_slots {
                let r = ledger.reserve(Tier::Ram, slot_len as u64, "stream-ring")?;
                slots.push(Slot {
                    buf: AlignedBuf::new(slot_len, DIRECT_ALIGN).map_err(|e| StoreError::Io(e.to_string()))?,
                    group: None,
                    ticket: None,
                    leases: 0,
                    last_use: 0,
                    prefetched: false,
                    _res: r,
                });
            }
        }
        Ok(WeightStore {
            inner: Arc::new(Inner {
                model,
                layouts,
                pos,
                order,
                file,
                cfg,
                io,
                ledger,
                state: Mutex::new(State { placement, slots, clock: 0, cur_pos: 0 }),
                metrics: StoreMetrics::default(),
            }),
        })
    }

    pub fn config(&self) -> &StoreConfig {
        &self.inner.cfg
    }
    pub fn model(&self) -> &Arc<ModelDesc> {
        &self.inner.model
    }
    pub fn ledger(&self) -> &Arc<Ledger> {
        &self.inner.ledger
    }

    pub fn is_resident(&self, group: usize) -> bool {
        matches!(self.inner.state.lock().unwrap().placement[group], Placement::Resident(_))
    }

    pub fn is_external(&self, group: usize) -> bool {
        matches!(self.inner.state.lock().unwrap().placement[group], Placement::External)
    }

    pub fn resident_mask(&self) -> Vec<bool> {
        let st = self.inner.state.lock().unwrap();
        st.placement.iter().map(|p| matches!(p, Placement::Resident(_))).collect()
    }

    /// Lease a group's bytes, loading it if necessary. Leasing a streamed
    /// group triggers prefetch of the following streamed groups.
    pub fn lease(&self, group: usize) -> Result<Lease, StoreError> {
        let inner = &self.inner;
        let m = &inner.metrics;
        m.leases.fetch_add(1, Relaxed);
        m.lease_bytes.fetch_add(inner.layouts[group].bytes, Relaxed);
        let mut st = inner.state.lock().unwrap();
        st.clock += 1;
        st.cur_pos = inner.pos[group];
        if let Placement::Resident(buf) = &st.placement[group] {
            let buf = buf.clone();
            drop(st);
            m.resident_hits.fetch_add(1, Relaxed);
            m.resident_bytes.fetch_add(inner.layouts[group].bytes, Relaxed);
            let (ptr, len) = (buf.buf.as_slice().as_ptr(), buf.buf.len());
            return Ok(Lease { store: Some(inner.clone()), slot: usize::MAX, resident: Some(buf), ptr, len, group });
        }

        if matches!(st.placement[group], Placement::External) {
            return Err(StoreError::Config(format!("group {group} is managed by the expert store")));
        }
        let clock = st.clock;
        let slot = match st.slots.iter().position(|s| s.group == Some(group)) {
            Some(s) => {
                let sl = &mut st.slots[s];
                if sl.prefetched {
                    sl.prefetched = false;
                    m.prefetch_used.fetch_add(1, Relaxed);
                }
                if sl.ticket.as_ref().is_none_or(|t| t.is_done()) {
                    m.ring_ready.fetch_add(1, Relaxed);
                } else {
                    m.ring_late.fetch_add(1, Relaxed);
                }
                s
            }
            None => {
                m.demand_loads.fetch_add(1, Relaxed);
                loop {
                    if let Some(v) = self.victim(&st, group, true) {
                        self.load_into(&mut st, v, group, false);
                        break v;
                    }
                    // All unleased slots have loads in flight: wait for one.
                    let busy = st.slots.iter().find(|s| s.leases == 0).and_then(|s| s.ticket.clone()).ok_or_else(|| {
                        StoreError::Config(format!("streaming ring exhausted: all {} slots are leased", st.slots.len()))
                    })?;
                    drop(st);
                    let waited = busy.wait().map_err(StoreError::Io)?;
                    m.stall_ns.fetch_add(waited.as_nanos() as u64, Relaxed);
                    st = inner.state.lock().unwrap();
                    if let Some(s) = st.slots.iter().position(|s| s.group == Some(group)) {
                        break s; // someone else loaded it meanwhile
                    }
                }
            }
        };
        st.slots[slot].leases += 1;
        st.slots[slot].last_use = clock;
        let ticket = st.slots[slot].ticket.clone();
        // Prefetch what follows while we (possibly) wait.
        self.prefetch_locked(&mut st, group);
        let (ptr, len) = (st.slots[slot].buf.as_slice().as_ptr(), st.slots[slot].buf.len());
        drop(st);

        let lease = Lease { store: Some(inner.clone()), slot, resident: None, ptr, len, group };
        if let Some(t) = ticket {
            let waited = t.wait().map_err(StoreError::Io)?;
            m.stall_ns.fetch_add(waited.as_nanos() as u64, Relaxed);
        }
        Ok(lease)
    }

    /// Choose a slot to (re)fill for `group`. With `force`, always returns an
    /// unleased slot if one exists; otherwise only evicts a slot whose group
    /// is needed later than `group` (never displace something needed sooner).
    fn victim(&self, st: &State, group: usize, force: bool) -> Option<usize> {
        let n = self.inner.order.len();
        let dist = |g: usize| (self.inner.pos[g] + n - st.cur_pos) % n;
        let target = dist(group);
        // Never refill a slot whose previous load is still in flight: its
        // late-completing chunks would overwrite the new group's bytes.
        let free = st.slots.iter().enumerate().filter(|(_, s)| s.leases == 0 && s.ticket.as_ref().is_none_or(|t| t.is_done()));
        if let Some((i, _)) = free.clone().find(|(_, s)| s.group.is_none()) {
            return Some(i);
        }
        match self.inner.cfg.policy {
            RingPolicy::Belady => {
                let (i, s) = free.max_by_key(|(_, s)| dist(s.group.unwrap()))?;
                // The group just used (distance 0 after the cursor moved past it
                // is impossible: the current group is leased) or later groups.
                if force || dist(s.group.unwrap()) > target {
                    Some(i)
                } else {
                    None
                }
            }
            RingPolicy::Lru => free.min_by_key(|(_, s)| s.last_use).map(|(i, _)| i),
        }
    }

    fn load_into(&self, st: &mut State, slot: usize, group: usize, prefetch: bool) {
        let inner = &self.inner;
        let s = &mut st.slots[slot];
        if s.prefetched {
            inner.metrics.prefetch_wasted.fetch_add(1, Relaxed);
        }
        let layout = &inner.layouts[group];
        let ticket = layout.issue(&inner.io, &inner.file, &mut s.buf);
        s.group = Some(group);
        s.ticket = Some(ticket);
        s.prefetched = prefetch;
        inner.metrics.stream_bytes.fetch_add(layout.bytes, Relaxed);
        if prefetch {
            inner.metrics.prefetch_issued.fetch_add(1, Relaxed);
        }
        if inner.cfg.io_mode == IoMode::Buffered && inner.cfg.drop_page_cache {
            for e in &inner.model.groups[group].extents {
                inner.file.drop_cache(e.offset, e.len);
            }
        }
    }

    /// Issue loads for the next `prefetch_depth` streamed groups after `group`.
    fn prefetch_locked(&self, st: &mut State, group: usize) {
        let inner = &self.inner;
        let d = inner.cfg.prefetch_depth;
        if d == 0 || st.slots.is_empty() {
            return;
        }
        let n = inner.order.len();
        let start = inner.pos[group];
        let mut found = 0;
        for k in 1..n {
            if found >= d {
                break;
            }
            let g = inner.order[(start + k) % n];
            if !matches!(st.placement[g], Placement::Streamed) {
                continue;
            }
            found += 1;
            if st.slots.iter().any(|s| s.group == Some(g)) {
                continue;
            }
            match self.victim(st, g, inner.cfg.policy == RingPolicy::Lru) {
                Some(v) => self.load_into(st, v, g, true),
                None => break,
            }
        }
    }

    /// Promote a streamed group to resident RAM. Blocks until loaded.
    pub fn promote(&self, group: usize) -> Result<(), StoreError> {
        let inner = &self.inner;
        if self.is_resident(group) {
            return Ok(());
        }
        let layout = &inner.layouts[group];
        let r = inner.ledger.reserve(Tier::Ram, layout.buf_len as u64, "weights")?;
        let mut buf = AlignedBuf::new(layout.buf_len, DIRECT_ALIGN).map_err(|e| StoreError::Io(e.to_string()))?;
        let t = layout.issue(&inner.io, &inner.file, &mut buf);
        t.wait().map_err(StoreError::Io)?;
        let mut st = inner.state.lock().unwrap();
        st.placement[group] = Placement::Resident(Arc::new(GroupBuf { buf, _res: r }));
        // A ring copy is now redundant; free the slot for others.
        for s in st.slots.iter_mut() {
            if s.group == Some(group) && s.leases == 0 && s.ticket.as_ref().is_none_or(|t| t.is_done()) {
                s.group = None;
                s.ticket = None;
                s.prefetched = false;
            }
        }
        inner.metrics.promotions.fetch_add(1, Relaxed);
        Ok(())
    }

    /// Demote a resident group to streamed. Its buffer (and RAM reservation)
    /// is freed as soon as the last outstanding lease drops. If no streaming
    /// ring exists yet (the plan started fully resident), one is allocated
    /// after the group's memory is released, sized for the largest layer
    /// group so later demotions fit too; if even that fails the demotion is
    /// reverted.
    pub fn demote(&self, group: usize) -> Result<(), StoreError> {
        let inner = &self.inner;
        let mut st = inner.state.lock().unwrap();
        if !matches!(st.placement[group], Placement::Resident(_)) {
            return Ok(());
        }
        let need = inner.layouts[group].buf_len;
        let ring_ok = !st.slots.is_empty() && st.slots[0].buf.len() >= need;
        if !ring_ok && st.slots.iter().any(|s| s.leases > 0 || s.ticket.as_ref().is_some_and(|t| !t.is_done())) {
            return Err(StoreError::Config("cannot resize the streaming ring while it is in use".into()));
        }
        let old = std::mem::replace(&mut st.placement[group], Placement::Streamed);
        if !ring_ok {
            drop(old); // release the group's reservation first (if unleased)
            let slot_len = inner.model.groups.iter().filter(|g| g.layer.is_some()).map(|g| inner.layouts[g.id].buf_len).max().unwrap_or(need).max(need);
            let n = inner.cfg.ring_slots.max(inner.cfg.prefetch_depth + 1);
            let mut slots = Vec::with_capacity(n);
            for _ in 0..n {
                let alloc = inner
                    .ledger
                    .reserve(Tier::Ram, slot_len as u64, "stream-ring")
                    .map_err(StoreError::from)
                    .and_then(|r| AlignedBuf::new(slot_len, DIRECT_ALIGN).map(|b| (b, r)).map_err(|e| StoreError::Io(e.to_string())));
                match alloc {
                    Ok((buf, r)) => slots.push(Slot { buf, group: None, ticket: None, leases: 0, last_use: 0, prefetched: false, _res: r }),
                    Err(e) => {
                        drop(slots);
                        drop(st);
                        // Revert: bring the group back (its bytes are on disk).
                        let _ = self.promote(group);
                        inner.metrics.promotions.fetch_sub(1, Relaxed);
                        return Err(e);
                    }
                }
            }
            st.slots = slots;
        }
        inner.metrics.demotions.fetch_add(1, Relaxed);
        Ok(())
    }

    /// Bytes of each streamed group per full pass over the model.
    pub fn streamed_bytes_per_pass(&self) -> u64 {
        let st = self.inner.state.lock().unwrap();
        st.placement.iter().enumerate().filter(|(_, p)| matches!(p, Placement::Streamed)).map(|(g, _)| self.inner.layouts[g].bytes).sum()
    }

    pub fn slot_len(&self) -> usize {
        self.inner.state.lock().unwrap().slots.first().map(|s| s.buf.len()).unwrap_or(0)
    }
    pub fn group_buf_len(&self, group: usize) -> usize {
        self.inner.layouts[group].buf_len
    }

    pub fn metrics(&self) -> MetricsSnapshot {
        self.inner.metrics.snapshot(self.inner.io.service_ns.load(Relaxed), self.inner.io.n_workers)
    }

    pub fn io_mode(&self) -> IoMode {
        self.inner.cfg.io_mode
    }

    /// Wait for all in-flight ring loads (used before measurements).
    pub fn quiesce(&self) {
        let tickets: Vec<Arc<Ticket>> = self.inner.state.lock().unwrap().slots.iter().filter_map(|s| s.ticket.clone()).collect();
        for t in tickets {
            let _ = t.wait();
        }
    }

    /// Mean disk latency per streamed group load observed so far.
    pub fn mean_load_latency(&self) -> Option<Duration> {
        let st = self.inner.state.lock().unwrap();
        let l: Vec<Duration> = st.slots.iter().filter_map(|s| s.ticket.as_ref().and_then(|t| t.latency())).collect();
        (!l.is_empty()).then(|| l.iter().sum::<Duration>() / l.len() as u32)
    }
}

#[cfg(test)]
mod tests_support {
    use super::*;
    use kestrel_gguf::writer::{GgufWriter, TensorData};
    use kestrel_gguf::Value;

    /// A model with `layers` layers whose tensor bytes encode (tensor, index).
    pub fn model(dir: &std::path::Path, layers: u32) -> Arc<ModelDesc> {
        let p = dir.join("m.gguf");
        let mut w = GgufWriter::new();
        w.kv("general.architecture", Value::String("llama".into()));
        w.kv("llama.block_count", Value::U32(layers));
        w.kv("llama.embedding_length", Value::U32(64));
        w.kv("llama.attention.head_count", Value::U32(4));
        let pat = |seed: u32, n: usize| (0..n).map(|i| (seed * 1000 + i as u32 % 997) as f32).collect::<Vec<f32>>();
        w.tensor("token_embd.weight", &[64, 16], TensorData::F32(pat(1, 1024)));
        for l in 0..layers {
            w.tensor(&format!("blk.{l}.attn_q.weight"), &[64, 100], TensorData::F32(pat(10 + l, 6400)));
            w.tensor(&format!("blk.{l}.ffn_up.weight"), &[64, 300], TensorData::F32(pat(100 + l, 19200)));
        }
        w.tensor("output.weight", &[64, 16], TensorData::F32(pat(2, 1024)));
        w.write(&p).unwrap();
        Arc::new(ModelDesc::open(&p).unwrap())
    }

    pub fn check(store: &WeightStore, m: &ModelDesc, raw: &[u8], g: usize) {
        let lease = store.lease(g).unwrap();
        for &ti in &m.groups[g].tensors {
            let t = &m.tensors[ti];
            assert_eq!(lease.tensor(ti), &raw[t.offset as usize..(t.offset + t.size) as usize], "tensor {}", t.name);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::tests_support::*;
    use super::*;

    fn run(policy: RingPolicy, depth: usize, slots: usize, resident_every: usize, mode: IoMode) -> MetricsSnapshot {
        let d = tempfile::tempdir().unwrap();
        let m = model(d.path(), 6);
        let raw = std::fs::read(&m.path).unwrap();
        let n = m.groups.len();
        let resident: Vec<bool> = (0..n).map(|g| resident_every > 0 && g % resident_every == 0).collect();
        let order: Vec<usize> = (0..n).collect();
        let cfg = StoreConfig { io_mode: mode, prefetch_depth: depth, ring_slots: slots, io_workers: 3, policy, drop_page_cache: true, external_experts: false };
        let ledger = Ledger::new(0, 1 << 30, 0);
        let store = WeightStore::new(m.clone(), &resident, order, cfg, ledger.clone()).unwrap();
        for _token in 0..4 {
            for g in 0..n {
                check(&store, &m, &raw, g);
            }
        }
        store.quiesce();
        let snap = store.metrics();
        drop(store);
        assert_eq!(ledger.usage(Tier::Ram).reserved, 0, "all reservations released");
        snap
    }

    #[test]
    fn streaming_is_bit_exact_in_all_modes() {
        for mode in [IoMode::Direct, IoMode::Buffered] {
            for policy in [RingPolicy::Belady, RingPolicy::Lru] {
                for (depth, slots) in [(0, 1), (1, 2), (2, 3), (3, 6)] {
                    for every in [0, 2, 3] {
                        run(policy, depth, slots, every, mode);
                    }
                }
            }
        }
    }

    #[test]
    fn prefetch_turns_demand_loads_into_hits() {
        let no_pf = run(RingPolicy::Belady, 0, 1, 0, IoMode::Direct);
        assert_eq!(no_pf.prefetch_issued, 0);
        assert!(no_pf.demand_loads > 0);
        let pf = run(RingPolicy::Belady, 2, 3, 0, IoMode::Direct);
        // Only the very first lease is a demand load; everything after is prefetched.
        assert_eq!(pf.demand_loads, 1, "{pf:?}");
        assert!(pf.prefetch_accuracy > 0.9, "{pf:?}");
    }

    #[test]
    fn lru_thrashes_on_cyclic_scan_belady_does_not() {
        // 14 groups, 13 slots, no prefetch: LRU misses every time; Belady keeps 12.
        let lru = run(RingPolicy::Lru, 0, 13, 0, IoMode::Buffered);
        let bel = run(RingPolicy::Belady, 0, 13, 0, IoMode::Buffered);
        assert_eq!(lru.ring_ready, 0, "LRU on a cyclic scan never hits: {lru:?}");
        assert!(bel.ring_ready > lru.ring_ready * 2 + 20, "belady {bel:?}");
    }

    #[test]
    fn budget_is_enforced() {
        let d = tempfile::tempdir().unwrap();
        let m = model(d.path(), 4);
        let n = m.groups.len();
        let ledger = Ledger::new(0, 50_000, 0);
        let err = WeightStore::new(m, &vec![true; n], (0..n).collect(), StoreConfig::default(), ledger).err().unwrap();
        assert!(matches!(err, StoreError::Budget(_)), "{err}");
    }

    #[test]
    fn promote_and_demote_preserve_contents() {
        let d = tempfile::tempdir().unwrap();
        let m = model(d.path(), 4);
        let raw = std::fs::read(&m.path).unwrap();
        let n = m.groups.len();
        let ledger = Ledger::new(0, 1 << 30, 0);
        let store = WeightStore::new(m.clone(), &vec![false; n], (0..n).collect(), StoreConfig::default(), ledger.clone()).unwrap();
        let before = ledger.usage(Tier::Ram).reserved;
        let held = store.lease(3).unwrap();
        store.promote(3).unwrap();
        assert!(store.is_resident(3));
        assert!(ledger.usage(Tier::Ram).reserved > before);
        drop(held);
        for g in 0..n {
            check(&store, &m, &raw, g);
        }
        let held = store.lease(3).unwrap();
        store.demote(3).unwrap();
        // Buffer stays alive while leased.
        assert!(ledger.usage(Tier::Ram).reserved > before);
        drop(held);
        assert_eq!(ledger.usage(Tier::Ram).reserved, before);
        for g in 0..n {
            check(&store, &m, &raw, g);
        }
        let s = store.metrics();
        assert_eq!((s.promotions, s.demotions), (1, 1));
    }
}

#[cfg(test)]
mod rebalance_tests {
    use super::tests_support::*;
    use super::*;
    use crate::guard::{MemSample, MemoryGuard, RebalanceAction, Rebalancer};

    #[test]
    fn pressure_demotes_then_calm_promotes_without_changing_bytes() {
        let d = tempfile::tempdir().unwrap();
        let m = model(d.path(), 6);
        let raw = std::fs::read(&m.path).unwrap();
        let n = m.groups.len();
        let ledger = Ledger::new(0, 1 << 30, 0);
        let cfg = StoreConfig { ring_slots: 3, prefetch_depth: 2, ..StoreConfig::default() };
        let store = WeightStore::new(m.clone(), &vec![true; n], (0..n).collect(), cfg, ledger.clone()).unwrap();
        let mut r = Rebalancer::new(MemoryGuard { rss_limit: 100 << 20, min_available: 0, tolerance: 0.0 });
        r.pinned = m.groups.iter().filter(|g| g.layer.is_none()).map(|g| g.id).collect();
        let limit0 = ledger.usage(Tier::Ram).limit;

        // RSS 1 MB over the limit (+256 MiB slack built into the guard).
        let acts = r.tick(&store, MemSample { rss: (100 << 20) + (256 << 20) + (1 << 20), available: 8 << 30 });
        assert!(!acts.is_empty() && acts.iter().all(|a| matches!(a, RebalanceAction::Demoted { .. })), "{acts:?}");
        assert!(ledger.usage(Tier::Ram).limit < limit0, "ceiling lowered so freed room is not refilled");
        let demoted = store.resident_mask().iter().filter(|r| !**r).count();
        assert!(demoted >= 1);
        for _ in 0..2 {
            for g in 0..n {
                check(&store, &m, &raw, g);
            }
        }
        // Calm for `promote_after` ticks with plenty of headroom: promote one back.
        ledger.set_limit(Tier::Ram, limit0);
        let calm = MemSample { rss: 1 << 20, available: 8 << 30 };
        let mut promoted = false;
        for _ in 0..r.promote_after {
            promoted |= r.tick(&store, calm).iter().any(|a| matches!(a, RebalanceAction::Promoted { .. }));
        }
        assert!(promoted);
        for g in 0..n {
            check(&store, &m, &raw, g);
        }
    }
}
