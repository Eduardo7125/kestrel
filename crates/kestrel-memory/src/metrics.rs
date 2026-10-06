use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Counters updated lock-free on the hot path.
#[derive(Default)]
pub struct StoreMetrics {
    pub leases: AtomicU64,
    pub lease_bytes: AtomicU64,
    /// Leases served by a resident group.
    pub resident_hits: AtomicU64,
    pub resident_bytes: AtomicU64,
    /// Streamed group already loaded when leased.
    pub ring_ready: AtomicU64,
    /// Streamed group whose prefetch was still in flight (late prefetch).
    pub ring_late: AtomicU64,
    /// Streamed group not in the ring at all: demand load.
    pub demand_loads: AtomicU64,
    pub stall_ns: AtomicU64,
    pub prefetch_issued: AtomicU64,
    pub prefetch_used: AtomicU64,
    /// Prefetched groups evicted before being used.
    pub prefetch_wasted: AtomicU64,
    pub stream_bytes: AtomicU64,
    pub promotions: AtomicU64,
    pub demotions: AtomicU64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct MetricsSnapshot {
    pub leases: u64,
    pub lease_bytes: u64,
    pub resident_hits: u64,
    pub resident_bytes: u64,
    pub ring_ready: u64,
    pub ring_late: u64,
    pub demand_loads: u64,
    pub stall_s: f64,
    pub prefetch_issued: u64,
    pub prefetch_used: u64,
    pub prefetch_wasted: u64,
    pub stream_bytes: u64,
    pub disk_service_s: f64,
    pub io_workers: usize,
    pub promotions: u64,
    pub demotions: u64,
    /// Fraction of leased bytes served without waiting for I/O.
    pub hit_rate: f64,
    /// Prefetches that were used ÷ prefetches issued.
    pub prefetch_accuracy: f64,
    /// Streamed leases that had to wait ÷ streamed leases.
    pub late_fraction: f64,
    /// Effective disk bandwidth while busy (bytes ÷ per-worker service time × workers).
    pub disk_bw: f64,
}

impl StoreMetrics {
    pub fn snapshot(&self, disk_service_ns: u64, io_workers: usize) -> MetricsSnapshot {
        let g = |a: &AtomicU64| a.load(Relaxed);
        let streamed = g(&self.ring_ready) + g(&self.ring_late) + g(&self.demand_loads);
        let issued = g(&self.prefetch_issued);
        let svc = disk_service_ns as f64 * 1e-9;
        MetricsSnapshot {
            leases: g(&self.leases),
            lease_bytes: g(&self.lease_bytes),
            resident_hits: g(&self.resident_hits),
            resident_bytes: g(&self.resident_bytes),
            ring_ready: g(&self.ring_ready),
            ring_late: g(&self.ring_late),
            demand_loads: g(&self.demand_loads),
            stall_s: g(&self.stall_ns) as f64 * 1e-9,
            prefetch_issued: issued,
            prefetch_used: g(&self.prefetch_used),
            prefetch_wasted: g(&self.prefetch_wasted),
            stream_bytes: g(&self.stream_bytes),
            disk_service_s: svc,
            io_workers,
            promotions: g(&self.promotions),
            demotions: g(&self.demotions),
            hit_rate: if g(&self.leases) == 0 { 0.0 } else { (g(&self.resident_hits) + g(&self.ring_ready)) as f64 / g(&self.leases) as f64 },
            prefetch_accuracy: if issued == 0 { 0.0 } else { g(&self.prefetch_used) as f64 / issued as f64 },
            late_fraction: if streamed == 0 { 0.0 } else { (g(&self.ring_late) + g(&self.demand_loads)) as f64 / streamed as f64 },
            disk_bw: if svc > 0.0 { g(&self.stream_bytes) as f64 / svc * io_workers as f64 } else { 0.0 },
        }
    }
}

impl MetricsSnapshot {
    /// Counters accumulated between two snapshots.
    pub fn delta(&self, earlier: &MetricsSnapshot) -> MetricsSnapshot {
        let mut d = self.clone();
        d.leases -= earlier.leases;
        d.lease_bytes -= earlier.lease_bytes;
        d.resident_hits -= earlier.resident_hits;
        d.resident_bytes -= earlier.resident_bytes;
        d.ring_ready -= earlier.ring_ready;
        d.ring_late -= earlier.ring_late;
        d.demand_loads -= earlier.demand_loads;
        d.stall_s -= earlier.stall_s;
        d.prefetch_issued -= earlier.prefetch_issued;
        d.prefetch_used -= earlier.prefetch_used;
        d.prefetch_wasted -= earlier.prefetch_wasted;
        d.stream_bytes -= earlier.stream_bytes;
        d.disk_service_s -= earlier.disk_service_s;
        d.promotions -= earlier.promotions;
        d.demotions -= earlier.demotions;
        let streamed = d.ring_ready + d.ring_late + d.demand_loads;
        d.hit_rate = if d.leases == 0 { 0.0 } else { (d.resident_hits + d.ring_ready) as f64 / d.leases as f64 };
        d.prefetch_accuracy = if d.prefetch_issued == 0 { 0.0 } else { d.prefetch_used as f64 / d.prefetch_issued as f64 };
        d.late_fraction = if streamed == 0 { 0.0 } else { (d.ring_late + d.demand_loads) as f64 / streamed as f64 };
        d.disk_bw = if d.disk_service_s > 0.0 { d.stream_bytes as f64 / d.disk_service_s * d.io_workers as f64 } else { 0.0 };
        d
    }
}
