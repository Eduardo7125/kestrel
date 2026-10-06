//! The execution planner.
//!
//! Inputs: a [`ModelDesc`], a [`HardwareProfile`] (with measured bandwidths
//! when available) and user overrides. Output: an [`ExecutionPlan`] that
//! assigns every tensor group to a tier, sizes the KV cache, the streaming
//! ring and the safety margins, ranks the alternative strategies with a
//! bandwidth cost model, and explains itself. When nothing fits, the result
//! is an [`Infeasible`] diagnosis with concrete remedies instead of a crash.

mod cost;
mod render;

pub use cost::{Bandwidths, Estimate};

use kestrel_gguf::GgmlType;
use kestrel_hw::fileio::IoMode;
use kestrel_hw::HardwareProfile;
use kestrel_memory::{RingPolicy, StoreConfig, Tier};
use kestrel_model::{GroupKind, ModelDesc};
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    Native,
    LlamaCpp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum StrategyKind {
    /// Everything (weights + KV) in VRAM.
    GpuFull,
    /// Leading layers in RAM on the CPU, the rest in VRAM.
    GpuRamHybrid,
    /// Everything resident in RAM, CPU compute.
    RamOnly,
    /// Part resident in RAM, the rest streamed from NVMe every token.
    RamNvme,
    /// VRAM + RAM + NVMe (llama.cpp with mmap paging for the overflow).
    VramRamNvme,
}

impl StrategyKind {
    pub fn label(self) -> &'static str {
        match self {
            StrategyKind::GpuFull => "VRAM only",
            StrategyKind::GpuRamHybrid => "Hybrid VRAM/RAM",
            StrategyKind::RamOnly => "RAM only (CPU)",
            StrategyKind::RamNvme => "RAM + NVMe streaming",
            StrategyKind::VramRamNvme => "Hybrid VRAM/RAM/NVMe",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PlacementOrder {
    /// Streamed groups spread evenly between resident ones (default).
    Interleaved,
    /// The last groups are streamed back to back (ablation).
    Contiguous,
}

#[derive(Clone, Debug)]
pub struct PlanRequest {
    pub backend: Option<Backend>,
    pub n_ctx: Option<u64>,
    pub kv_type: GgmlType,
    pub vram_budget: Option<u64>,
    pub ram_budget: Option<u64>,
    pub kv_budget: Option<u64>,
    pub allow_overcommit: bool,
    pub prefetch_depth: Option<usize>,
    pub io_mode: Option<IoMode>,
    pub io_workers: Option<usize>,
    pub ring_policy: RingPolicy,
    pub strategy: Option<StrategyKind>,
    /// Refuse NVMe streaming (everything must be resident).
    pub no_stream: bool,
    pub placement: PlacementOrder,
    pub threads: Option<usize>,
    /// Whether the native executor can run this model (from kestrel-engine).
    pub native_support: Result<(), String>,
    /// Whether a llama.cpp server binary is available.
    pub llamacpp_available: bool,
}

impl Default for PlanRequest {
    fn default() -> Self {
        PlanRequest {
            backend: None,
            n_ctx: None,
            kv_type: GgmlType::F16,
            vram_budget: None,
            ram_budget: None,
            kv_budget: None,
            allow_overcommit: false,
            prefetch_depth: None,
            io_mode: None,
            io_workers: None,
            ring_policy: RingPolicy::Belady,
            strategy: None,
            no_stream: false,
            placement: PlacementOrder::Interleaved,
            threads: None,
            native_support: Ok(()),
            llamacpp_available: false,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct TierBudget {
    pub total: u64,
    pub available: u64,
    pub safety: u64,
    pub overhead: u64,
    pub usable: u64,
    pub explicit: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Budgets {
    pub vram: TierBudget,
    pub ram: TierBudget,
    pub disk_free: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Candidate {
    pub kind: StrategyKind,
    pub backend: Backend,
    /// Tier of each tensor group (index = group id).
    pub tiers: Vec<Tier>,
    pub vram_weights: u64,
    pub ram_weights: u64,
    pub disk_weights: u64,
    pub kv_tier: Tier,
    pub kv_bytes: u64,
    /// Part of the KV cache held in VRAM (KV of GPU-offloaded layers).
    pub kv_vram_bytes: u64,
    /// VRAM / RAM totals including KV, buffers and the streaming ring.
    pub vram_total: u64,
    pub ram_total: u64,
    pub n_gpu_layers: Option<u32>,
    pub experts_on_cpu: bool,
    pub ring_bytes: u64,
    pub estimate: Estimate,
    pub feasible: bool,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ExecutionPlan {
    pub model_name: String,
    pub model_path: String,
    pub arch: String,
    pub n_params: u64,
    pub weight_bytes: u64,
    pub quant: String,
    pub hardware: String,
    pub budgets: Budgets,
    pub bandwidths: Bandwidths,
    pub n_ctx: u64,
    pub kv_type: String,
    pub threads: usize,
    pub chosen: Candidate,
    pub alternatives: Vec<Candidate>,
    pub store: StoreConfig,
    pub placement: PlacementOrder,
    pub llamacpp_args: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Infeasible {
    pub model_name: String,
    pub required_vram: u64,
    pub required_ram: u64,
    pub required_disk: u64,
    pub available_vram: u64,
    pub available_ram: u64,
    pub available_disk: u64,
    pub reasons: Vec<String>,
    pub remedies: Vec<String>,
    pub candidates: Vec<Candidate>,
}

/// Bytes the runtime itself needs in RAM besides weights and KV: tokenizer,
/// activations, logits, thread stacks.
pub fn runtime_overhead(model: &ModelDesc, n_ctx: u64) -> u64 {
    let h = &model.hparams;
    let act = 64 * (h.n_embd as u64 + h.n_ff as u64) * 4 * 4; // n_batch rows of activations
    let logits = h.n_vocab as u64 * 4 * 2;
    let attn = n_ctx * h.n_head as u64 * 4;
    256 * (1 << 20) + act + logits + attn
}

/// GPU compute-buffer estimate for llama.cpp at a given context and batch.
fn gpu_overhead(model: &ModelDesc, n_ctx: u64) -> u64 {
    let h = &model.hparams;
    let batch = 512u64;
    let act = batch * (h.n_embd as u64 * 4 + h.n_ff as u64 * 2) * 4;
    let attn = batch * n_ctx.min(8192) * h.n_head as u64 * 4 / 4; // flash-attn-ish working set
    let logits = batch.min(32) * h.n_vocab as u64 * 4;
    300 * (1 << 20) + act + attn + logits
}

pub fn budgets(hw: &HardwareProfile, req: &PlanRequest, model: &ModelDesc, n_ctx: u64) -> Budgets {
    let gpu = hw.gpus.first();
    let (vtotal, vfree) = gpu.map(|g| (g.vram_total, g.vram_free)).unwrap_or((0, 0));
    let vsafety = if vtotal > 0 { (512u64 << 20).max(vtotal / 12) } else { 0 };
    let vover = if vtotal > 0 { gpu_overhead(model, n_ctx) } else { 0 };
    let vusable = match req.vram_budget {
        Some(b) if req.allow_overcommit => b,
        Some(b) => b.min(vfree.saturating_sub(vsafety)),
        None => vfree.saturating_sub(vsafety),
    }
    .saturating_sub(vover);
    let rtotal = hw.ram.total;
    let ravail = hw.ram.effective_available();
    let rsafety = (1536u64 << 20).max(rtotal / 10);
    let rover = runtime_overhead(model, n_ctx);
    let rusable = match req.ram_budget {
        Some(b) if req.allow_overcommit => b,
        Some(b) => b.min(ravail.saturating_sub(rsafety)),
        None => ravail.saturating_sub(rsafety),
    }
    .saturating_sub(rover);
    let disk_free = hw.storage_for(&model.path).map(|s| s.available).unwrap_or(0);
    Budgets {
        vram: TierBudget { total: vtotal, available: vfree, safety: vsafety, overhead: vover, usable: vusable, explicit: req.vram_budget.is_some() },
        ram: TierBudget { total: rtotal, available: ravail, safety: rsafety, overhead: rover, usable: rusable, explicit: req.ram_budget.is_some() },
        disk_free,
    }
}

/// Context length: user choice, else the training context capped at 4096
/// (and by the KV budget, if given).
fn choose_ctx(model: &ModelDesc, req: &PlanRequest) -> u64 {
    let per_tok = model.kv_bytes_per_token(req.kv_type).max(1);
    let mut n = req.n_ctx.unwrap_or_else(|| (model.hparams.n_ctx_train as u64).clamp(256, 4096));
    if let Some(kb) = req.kv_budget {
        n = n.min(kb / per_tok).max(16);
    }
    n
}

/// Van der Corput order over `n` items: every prefix is spread evenly.
fn spread_order(n: usize) -> Vec<usize> {
    let bits = usize::BITS - n.max(1).leading_zeros();
    let mut v: Vec<usize> = (0..(1usize << bits)).map(|i| i.reverse_bits() >> (usize::BITS - bits)).filter(|&i| i < n).collect();
    v.dedup();
    v
}

pub fn plan(model: &ModelDesc, hw: &HardwareProfile, req: &PlanRequest) -> Result<ExecutionPlan, Infeasible> {
    let n_ctx = choose_ctx(model, req);
    let b = budgets(hw, req, model, n_ctx);
    let bw = Bandwidths::from_profile(hw, &model.path, model.dominant_type());
    let threads = req.threads.unwrap_or(hw.cpu.physical_cores).max(1);
    let kv_bytes = model.kv_bytes(n_ctx, req.kv_type);
    let has_gpu = hw.gpus.first().is_some_and(|g| g.vram_total > 0);
    let mut warnings = Vec::new();

    let mut cands = Vec::new();
    let native_ok = req.native_support.is_ok();
    let want = |be: Backend| req.backend.is_none_or(|x| x == be);
    if want(Backend::LlamaCpp) && (req.llamacpp_available || req.backend == Some(Backend::LlamaCpp)) {
        if has_gpu {
            for experts_on_cpu in [false, true] {
                if experts_on_cpu && model.moe.is_none() {
                    continue;
                }
                cands.push(llamacpp_gpu(model, &b, &bw, req, n_ctx, kv_bytes, experts_on_cpu));
            }
        }
        cands.push(cpu_candidate(model, &b, &bw, req, n_ctx, kv_bytes, Backend::LlamaCpp));
    }
    if want(Backend::Native) {
        let mut c = cpu_candidate(model, &b, &bw, req, n_ctx, kv_bytes, Backend::Native);
        if let Err(e) = &req.native_support {
            c.feasible = false;
            c.reason = Some(format!("native executor: {e}"));
        }
        cands.push(c);
    }
    if let Some(k) = req.strategy {
        for c in cands.iter_mut().filter(|c| c.kind != k) {
            if c.feasible {
                c.feasible = false;
                c.reason = Some(format!("strategy forced to {}", k.label()));
            }
        }
    }
    if req.no_stream {
        for c in cands.iter_mut().filter(|c| c.disk_weights > 0) {
            if c.feasible {
                c.feasible = false;
                c.reason = Some("NVMe streaming disabled (--no-stream)".into());
            }
        }
    }
    cands.sort_by(|a, b| b.feasible.cmp(&a.feasible).then(b.estimate.tok_s.partial_cmp(&a.estimate.tok_s).unwrap()));

    let Some(best) = cands.iter().find(|c| c.feasible).cloned() else {
        return Err(diagnose(model, &b, req, n_ctx, kv_bytes, cands));
    };
    if !native_ok && req.backend.is_none() && !req.llamacpp_available {
        warnings.push("the native executor cannot run this model and no llama.cpp server was found".into());
    }
    if bw.estimated.iter().any(|s| s.contains("disk")) && best.disk_weights > 0 {
        warnings.push("disk bandwidth is an estimate; run `kestrel hardware --bench` for a measured plan".into());
    }
    if best.disk_weights > 0 {
        warnings.push(format!(
            "{} of weights stream from disk every token: decode is bounded by disk bandwidth ({}/s), not compute",
            kestrel_hw::fmt_bytes(best.disk_weights),
            kestrel_hw::fmt_bytes(bw.disk as u64)
        ));
    }
    if n_ctx > model.hparams.n_ctx_train as u64 {
        warnings.push(format!("context {n_ctx} exceeds the training context {}", model.hparams.n_ctx_train));
    }

    let depth = req.prefetch_depth.unwrap_or(2);
    let store = StoreConfig {
        io_mode: req.io_mode.unwrap_or(IoMode::Direct),
        prefetch_depth: depth,
        ring_slots: depth + 1,
        io_workers: req.io_workers.unwrap_or(8),
        policy: req.ring_policy,
        drop_page_cache: req.io_mode != Some(IoMode::Buffered),
    };
    let llamacpp_args = if best.backend == Backend::LlamaCpp { llamacpp_args(model, &best, n_ctx, req, threads) } else { Vec::new() };
    let alternatives = cands.into_iter().filter(|c| !(c.kind == best.kind && c.backend == best.backend && c.experts_on_cpu == best.experts_on_cpu)).collect();
    Ok(ExecutionPlan {
        model_name: model.name.clone(),
        model_path: model.path.display().to_string(),
        arch: model.arch.clone(),
        n_params: model.n_params,
        weight_bytes: model.weight_bytes(),
        quant: model.file_type.clone().or_else(|| model.dominant_type().map(str::to_string)).unwrap_or_default(),
        hardware: render::hardware_line(hw),
        budgets: b,
        bandwidths: bw,
        n_ctx,
        kv_type: req.kv_type.name().to_string(),
        threads,
        chosen: best,
        alternatives,
        store,
        placement: req.placement,
        llamacpp_args,
        warnings,
    })
}

/// CPU-side candidate: everything in RAM if it fits, otherwise stream the
/// overflow from NVMe (native: explicit ring; llama.cpp: mmap paging).
fn cpu_candidate(model: &ModelDesc, b: &Budgets, bw: &Bandwidths, req: &PlanRequest, n_ctx: u64, kv_bytes: u64, backend: Backend) -> Candidate {
    let n = model.groups.len();
    let mut tiers = vec![Tier::Ram; n];
    let total = model.weight_bytes();
    let ram_for_weights = b.ram.usable.saturating_sub(kv_bytes);
    let mut ring_bytes = 0;
    let mut reason = None;
    let mut feasible = true;
    if total > ram_for_weights {
        let depth = req.prefetch_depth.unwrap_or(2) as u64;
        // Stream layer groups (never embed/head) until the rest fits.
        let layer: Vec<usize> = model.groups.iter().filter(|g| g.layer.is_some()).map(|g| g.id).collect();
        let order: Vec<usize> = match req.placement {
            PlacementOrder::Interleaved => spread_order(layer.len()).into_iter().map(|i| layer[i]).collect(),
            PlacementOrder::Contiguous => layer.iter().rev().copied().collect(),
        };
        let mut resident = total;
        let mut max_streamed = 0u64;
        for g in order {
            let ring = (depth + 1) * max_streamed.max(model.groups[g].bytes);
            if resident + ring <= ram_for_weights {
                break;
            }
            tiers[g] = Tier::Disk;
            resident -= model.groups[g].bytes;
            max_streamed = max_streamed.max(model.groups[g].bytes);
        }
        ring_bytes = (depth + 1) * max_streamed;
        if resident + ring_bytes > ram_for_weights {
            feasible = false;
            reason = Some(format!(
                "even streaming every layer needs {} resident in RAM (embeddings, head, ring, KV); usable RAM is {}",
                kestrel_hw::fmt_bytes(resident + ring_bytes + kv_bytes),
                kestrel_hw::fmt_bytes(b.ram.usable)
            ));
        }
    }
    let disk_weights: u64 = model.groups.iter().filter(|g| tiers[g.id] == Tier::Disk).map(|g| g.bytes).sum();
    let ram_weights = total - disk_weights;
    let kind = if disk_weights > 0 { StrategyKind::RamNvme } else { StrategyKind::RamOnly };
    let compute_bw = match backend {
        Backend::Native => bw.cpu_native,
        Backend::LlamaCpp => bw.cpu_llamacpp,
    };
    let overlapped = backend == Backend::Native && req.prefetch_depth != Some(0);
    let estimate = cost::estimate(model, &tiers, kv_bytes, Tier::Ram, n_ctx, bw, compute_bw, overlapped, 0);
    Candidate {
        kind,
        backend,
        tiers,
        vram_weights: 0,
        ram_weights,
        disk_weights,
        kv_tier: Tier::Ram,
        kv_bytes,
        kv_vram_bytes: 0,
        vram_total: 0,
        ram_total: ram_weights + kv_bytes + ring_bytes + b.ram.overhead,
        n_gpu_layers: if backend == Backend::LlamaCpp { Some(0) } else { None },
        experts_on_cpu: false,
        ring_bytes,
        estimate,
        feasible,
        reason,
    }
}

/// llama.cpp with GPU offload. llama.cpp offloads the *last* `n_gpu_layers`
/// layers (and the output layer when `ngl > n_layer`); KV of offloaded layers
/// lives in VRAM. With `experts_on_cpu`, routed experts stay in RAM
/// (`--override-tensor exps=CPU`) so attention and shared tensors of every
/// layer fit in VRAM.
fn llamacpp_gpu(model: &ModelDesc, b: &Budgets, bw: &Bandwidths, req: &PlanRequest, n_ctx: u64, kv_bytes: u64, experts_on_cpu: bool) -> Candidate {
    let nl = model.hparams.n_layer;
    let kv_per_layer = kv_bytes / nl.max(1) as u64;
    let layer_vram = |l: u32| -> u64 {
        model
            .groups_of_layer(l)
            .filter(|g| !(experts_on_cpu && g.kind == GroupKind::Experts))
            .map(|g| g.bytes)
            .sum::<u64>()
            + kv_per_layer
    };
    let head_bytes: u64 = model.groups.iter().filter(|g| g.kind == GroupKind::Head).map(|g| g.bytes).sum();
    let mut used = 0u64;
    let mut ngl = 0u32;
    for l in (0..nl).rev() {
        let need = layer_vram(l);
        if used + need > b.vram.usable {
            break;
        }
        used += need;
        ngl += 1;
    }
    let mut output_on_gpu = false;
    if ngl == nl && used + head_bytes <= b.vram.usable {
        used += head_bytes;
        output_on_gpu = true;
    }
    let mut tiers = vec![Tier::Ram; model.groups.len()];
    for g in &model.groups {
        let on_gpu = match (g.kind, g.layer) {
            (GroupKind::Experts, Some(_)) if experts_on_cpu => false,
            (_, Some(l)) => l >= nl - ngl,
            (GroupKind::Head, _) => output_on_gpu,
            _ => false,
        };
        if on_gpu {
            tiers[g.id] = Tier::Vram;
        }
    }
    let vram_weights: u64 = model.groups.iter().filter(|g| tiers[g.id] == Tier::Vram).map(|g| g.bytes).sum();
    let cpu_weights = model.weight_bytes() - vram_weights;
    let kv_ram = kv_per_layer * (nl - ngl) as u64;
    let ram_room = b.ram.usable.saturating_sub(kv_ram);
    // Overflow beyond RAM is paged from NVMe by mmap (OS-managed).
    let mut disk_weights = 0;
    if cpu_weights > ram_room {
        let mut over = cpu_weights - ram_room;
        let layer_ids: Vec<usize> = model.groups.iter().filter(|g| tiers[g.id] == Tier::Ram && g.layer.is_some()).map(|g| g.id).collect();
        for &g in layer_ids.iter() {
            if over == 0 {
                break;
            }
            tiers[g] = Tier::Disk;
            disk_weights += model.groups[g].bytes;
            over = over.saturating_sub(model.groups[g].bytes);
        }
    }
    let ram_weights = cpu_weights - disk_weights;
    let kind = if vram_weights == model.weight_bytes() {
        StrategyKind::GpuFull
    } else if disk_weights > 0 {
        StrategyKind::VramRamNvme
    } else {
        StrategyKind::GpuRamHybrid
    };
    let mut feasible = ngl > 0 || experts_on_cpu;
    let mut reason = (!feasible).then(|| format!("not a single layer fits in usable VRAM ({})", kestrel_hw::fmt_bytes(b.vram.usable)));
    if ram_weights + kv_ram > b.ram.usable + disk_weights && disk_weights == 0 {
        feasible = false;
        reason = Some("CPU-side weights do not fit in RAM".into());
    }
    let estimate = cost::estimate(model, &tiers, kv_bytes, if ngl == nl { Tier::Vram } else { Tier::Ram }, n_ctx, bw, bw.cpu_llamacpp, false, ngl);
    let _ = req;
    Candidate {
        kind,
        backend: Backend::LlamaCpp,
        tiers,
        vram_weights,
        ram_weights,
        disk_weights,
        kv_tier: if ngl > 0 { Tier::Vram } else { Tier::Ram },
        kv_bytes,
        kv_vram_bytes: kv_per_layer * ngl as u64,
        vram_total: used + b.vram.overhead,
        ram_total: ram_weights + kv_ram + b.ram.overhead,
        n_gpu_layers: Some(if output_on_gpu { nl + 1 } else { ngl }),
        experts_on_cpu,
        ring_bytes: 0,
        estimate,
        feasible,
        reason,
    }
}

fn llamacpp_args(model: &ModelDesc, c: &Candidate, n_ctx: u64, req: &PlanRequest, threads: usize) -> Vec<String> {
    let mut a = vec!["--model".into(), model.path.display().to_string(), "--ctx-size".into(), n_ctx.to_string(), "--threads".into(), threads.to_string()];
    if let Some(ngl) = c.n_gpu_layers {
        a.push("--n-gpu-layers".into());
        a.push(ngl.to_string());
    }
    if c.experts_on_cpu {
        a.push("--override-tensor".into());
        a.push(r"\.ffn_(up|down|gate)_exps\.=CPU".into());
    }
    if req.kv_type != GgmlType::F16 {
        let t = req.kv_type.name().to_lowercase();
        a.extend(["--cache-type-k".into(), t.clone(), "--cache-type-v".into(), t]);
    }
    if c.disk_weights == 0 && c.ram_weights > 0 && c.ram_total < c.ram_weights * 2 {
        // Everything fits: keep default mmap (pages stay clean and evictable).
    }
    a
}

fn diagnose(model: &ModelDesc, b: &Budgets, req: &PlanRequest, n_ctx: u64, kv_bytes: u64, candidates: Vec<Candidate>) -> Infeasible {
    let total = model.weight_bytes();
    let embed_head: u64 = model.groups.iter().filter(|g| g.layer.is_none()).map(|g| g.bytes).sum();
    let max_group = model.groups.iter().filter(|g| g.layer.is_some()).map(|g| g.bytes).max().unwrap_or(0);
    let min_ram = embed_head + 3 * max_group + kv_bytes + b.ram.overhead;
    let mut reasons: Vec<String> = candidates.iter().filter_map(|c| c.reason.clone().map(|r| format!("{} ({:?}): {r}", c.kind.label(), c.backend))).collect();
    reasons.dedup();
    let mut remedies = Vec::new();
    let gib = |x: u64| kestrel_hw::fmt_bytes(x);
    let bpw = total as f64 * 8.0 / model.n_params.max(1) as f64;
    if bpw > 5.5 {
        let q4 = (model.n_params as f64 * 4.85 / 8.0) as u64;
        remedies.push(format!("Use a Q4_K_M quantization (~{} instead of {})", gib(q4), gib(total)));
    }
    if req.no_stream {
        remedies.push("Enable NVMe tiering (drop --no-stream): only embeddings, head, KV and a small ring must be resident".into());
    }
    if kv_bytes > (512 << 20) {
        remedies.push(format!("Reduce the context (--ctx {}) or quantize the KV cache (--kv-type q8_0): KV is {}", n_ctx / 2, gib(kv_bytes)));
    }
    if b.ram.explicit {
        remedies.push("Raise --ram-budget (it is lower than what this machine has available)".into());
    }
    remedies.push(format!("Close memory-heavy applications: {} RAM is available of {}", gib(b.ram.available), gib(b.ram.total)));
    remedies.push("Use a smaller model".into());
    Infeasible {
        model_name: model.name.clone(),
        required_vram: 0,
        required_ram: min_ram,
        required_disk: if b.disk_free > 0 { 0 } else { total },
        available_vram: b.vram.usable,
        available_ram: b.ram.usable + b.ram.overhead,
        available_disk: b.disk_free,
        reasons,
        remedies,
        candidates,
    }
}

impl ExecutionPlan {
    pub fn text(&self) -> String {
        render::plan_text(self)
    }
    pub fn resident_mask(&self) -> Vec<bool> {
        self.chosen.tiers.iter().map(|t| *t != Tier::Disk).collect()
    }
}

impl Infeasible {
    pub fn text(&self) -> String {
        render::infeasible_text(self)
    }
}

#[cfg(test)]
mod tests;
