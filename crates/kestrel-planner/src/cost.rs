//! The decode cost model.
//!
//! Single-stream decode of a transformer is memory-bound: each token reads
//! (nearly) every active weight once. Per-token time is therefore modelled as
//! bytes touched in each tier divided by that tier's bandwidth, plus KV reads:
//!
//! ```text
//! t_gpu  = bytes_vram / bw_vram + n_gpu_layers × launch_overhead
//! t_cpu  = (bytes_ram + bytes_streamed) / bw_cpu_compute
//! t_disk = bytes_streamed / bw_disk
//! t      = t_gpu + max(t_cpu, t_disk) + t_kv     (prefetch overlaps I/O and compute)
//! t      = t_gpu + t_cpu + t_disk + t_kv         (no overlap: demand paging)
//! ```
//!
//! Every bandwidth comes from a measurement when one exists; otherwise the
//! estimate is named in [`Bandwidths::estimated`] and the plan says so.

use kestrel_hw::fileio::IoMode;
use kestrel_hw::HardwareProfile;
use kestrel_memory::Tier;
use kestrel_model::ModelDesc;
use serde::Serialize;
use std::path::Path;

#[derive(Clone, Debug, Serialize)]
pub struct Bandwidths {
    pub ram: f64,
    /// Weight bytes/s the native executor's kernels consume (decode GEMV).
    pub cpu_native: f64,
    /// Weight bytes/s llama.cpp's CPU kernels consume.
    pub cpu_llamacpp: f64,
    pub vram: f64,
    pub pcie: f64,
    pub disk: f64,
    /// Which of the above are estimates rather than measurements.
    pub estimated: Vec<String>,
}

impl Bandwidths {
    pub fn from_profile(hw: &HardwareProfile, model_path: &Path, dominant_type: Option<&str>) -> Self {
        let mut est = Vec::new();
        let m = hw.measured.as_ref();
        let ram = match m.map(|m| m.ram_read_bw).filter(|v| *v > 0.0) {
            Some(v) => v,
            None => {
                est.push("ram: 20 GB/s assumed".into());
                20e9
            }
        };
        let cpu_native = match (m, dominant_type) {
            (Some(m), Some(t)) if m.cpu_gemv_bytes_per_s.contains_key(t) => m.cpu_gemv_bytes_per_s[t].min(ram),
            _ => {
                est.push("native kernels: 20% of RAM bandwidth assumed".into());
                ram * 0.2
            }
        };
        // llama.cpp's SIMD q8-activation kernels run close to memory bandwidth.
        est.push("llama.cpp CPU: 70% of RAM bandwidth assumed".into());
        let cpu_llamacpp = ram * 0.7;
        let gpu = hw.gpus.first();
        let vram = gpu.and_then(|g| g.mem_bandwidth).map(|v| v as f64).unwrap_or(0.0);
        if gpu.is_some_and(|g| g.mem_bandwidth_is_estimate) {
            est.push("vram: vendor specification".into());
        }
        let pcie = gpu.and_then(|g| g.pcie_bandwidth).map(|v| v as f64).unwrap_or(12e9);
        let disk = match m.and_then(|m| m.disk_for(model_path)) {
            Some(d) if d.mode == IoMode::Direct || !d.cache_suspect => d.read_bw,
            _ => {
                let kind = hw.storage_for(model_path).map(|s| s.kind.as_str()).unwrap_or("unknown");
                let (v, label) = match kind {
                    "nvme" => (3.0e9, "3 GB/s (NVMe)"),
                    "ssd" => (0.5e9, "0.5 GB/s (SATA SSD)"),
                    "hdd" => (0.15e9, "0.15 GB/s (HDD)"),
                    _ => (1.0e9, "1 GB/s (unknown device)"),
                };
                est.push(format!("disk: {label} assumed"));
                v
            }
        };
        Bandwidths { ram, cpu_native, cpu_llamacpp, vram, pcie, disk, estimated: est }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Estimate {
    pub t_token_s: f64,
    pub tok_s: f64,
    pub t_gpu_s: f64,
    pub t_cpu_s: f64,
    pub t_disk_s: f64,
    pub t_kv_s: f64,
    /// Bytes read from disk per token.
    pub disk_bytes_per_token: f64,
    pub bottleneck: String,
    pub overlapped: bool,
}

/// Demand paging (mmap) without lookahead reaches only part of the device
/// bandwidth; explicit prefetch with deep queues approaches it. 0.5 is an
/// assumption to be replaced by measurement (benchmark arm 7).
const DEMAND_PAGING_EFFICIENCY: f64 = 0.5;
const GPU_LAYER_OVERHEAD_S: f64 = 25e-6;

#[allow(clippy::too_many_arguments)]
pub fn estimate(model: &ModelDesc, tiers: &[Tier], kv_bytes: u64, kv_tier: Tier, n_ctx: u64, bw: &Bandwidths, cpu_bw: f64, overlapped: bool, n_gpu_layers: u32) -> Estimate {
    let (mut gpu, mut cpu, mut disk) = (0f64, 0f64, 0f64);
    for g in &model.groups {
        let touched = g.bytes as f64 * g.touch_per_token;
        match tiers[g.id] {
            Tier::Vram => gpu += touched,
            Tier::Ram => cpu += touched,
            Tier::Disk => {
                // A streamed dense group is read whole; experts by touch.
                disk += touched;
                cpu += touched;
            }
        }
    }
    let t_gpu = if gpu > 0.0 && bw.vram > 0.0 { gpu / bw.vram + n_gpu_layers as f64 * GPU_LAYER_OVERHEAD_S } else { 0.0 };
    let t_cpu = cpu / cpu_bw.max(1.0);
    let disk_bw = if overlapped { bw.disk } else { bw.disk * DEMAND_PAGING_EFFICIENCY };
    let t_disk = disk / disk_bw.max(1.0);
    // Average decode position ~ half of a typical 2k conversation.
    let avg_ctx = n_ctx.min(2048) as f64 / 2.0;
    let kv_read = kv_bytes as f64 / n_ctx.max(1) as f64 * avg_ctx;
    let t_kv = kv_read / if kv_tier == Tier::Vram && bw.vram > 0.0 { bw.vram } else { bw.ram };
    let t = if overlapped { t_gpu + t_cpu.max(t_disk) + t_kv } else { t_gpu + t_cpu + t_disk + t_kv };
    let parts = [("gpu", t_gpu), ("cpu", t_cpu), ("disk", t_disk), ("kv", t_kv)];
    let bottleneck = parts.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).map(|p| p.0).unwrap_or("cpu").to_string();
    Estimate {
        t_token_s: t,
        tok_s: if t > 0.0 { 1.0 / t } else { 0.0 },
        t_gpu_s: t_gpu,
        t_cpu_s: t_cpu,
        t_disk_s: t_disk,
        t_kv_s: t_kv,
        disk_bytes_per_token: disk,
        bottleneck,
        overlapped,
    }
}
