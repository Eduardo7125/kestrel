//! Shared CLI plumbing: model resolution, overrides, hardware profile.

use anyhow::{bail, Context, Result};
use clap::{Args, ValueEnum};
use kestrel_gguf::GgmlType;
use kestrel_hw::bench;
use kestrel_hw::fileio::IoMode;
use kestrel_hw::{fmt_bytes, HardwareProfile};
use kestrel_memory::RingPolicy;
use kestrel_model::ModelDesc;
use kestrel_planner::{Backend, PlacementOrder, PlanRequest, StrategyKind};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Parse sizes like `7G`, `512M`, `1.5GiB`, `200GB`, `1048576`.
pub fn parse_size(s: &str) -> Result<u64, String> {
    let t = s.trim();
    let split = t.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(t.len());
    let (num, unit) = t.split_at(split);
    let v: f64 = num.parse().map_err(|_| format!("invalid size '{s}'"))?;
    let mult = match unit.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1.0,
        "k" | "kb" | "kib" => 1024.0,
        "m" | "mb" | "mib" => 1024.0 * 1024.0,
        "g" | "gb" | "gib" => 1024.0 * 1024.0 * 1024.0,
        "t" | "tb" | "tib" => 1024.0f64.powi(4),
        u => return Err(format!("unknown size unit '{u}' in '{s}'")),
    };
    Ok((v * mult) as u64)
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum BackendArg {
    Auto,
    Native,
    Llamacpp,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum IoArg {
    Direct,
    Buffered,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum PolicyArg {
    Belady,
    Lru,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum ExpertPolicyArg {
    Lfru,
    Lru,
}

impl Overrides {
    pub fn native_options(&self) -> kestrel_backends::NativeOptions {
        kestrel_backends::NativeOptions {
            adaptive: !self.no_adapt,
            expert_policy: match self.expert_policy {
                ExpertPolicyArg::Lfru => kestrel_memory::ExpertPolicy::Lfru,
                ExpertPolicyArg::Lru => kestrel_memory::ExpertPolicy::Lru,
            },
            usage_history: !self.no_usage_history,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum StrategyArg {
    GpuFull,
    Hybrid,
    Ram,
    RamNvme,
    VramRamNvme,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum PlacementArg {
    Interleaved,
    Contiguous,
}

/// Overrides for the automatic plan. Everything is optional.
#[derive(Args, Clone, Debug)]
pub struct Overrides {
    /// Execution backend.
    #[arg(long, value_enum, default_value = "auto")]
    pub backend: BackendArg,
    /// Context length (tokens).
    #[arg(long = "ctx", alias = "ctx-size")]
    pub ctx: Option<u64>,
    /// VRAM budget, e.g. 7G.
    #[arg(long, value_parser = parse_size)]
    pub vram_budget: Option<u64>,
    /// RAM budget, e.g. 24G.
    #[arg(long, value_parser = parse_size)]
    pub ram_budget: Option<u64>,
    /// Upper bound for the KV cache, e.g. 2G (limits the context).
    #[arg(long = "kv-cache", value_parser = parse_size)]
    pub kv_cache: Option<u64>,
    /// KV cache type (llama.cpp backend): f16, q8_0.
    #[arg(long, default_value = "f16")]
    pub kv_type: String,
    /// Disk cache budget (reserved for re-packed layouts; GGUF is read in place).
    #[arg(long, value_parser = parse_size)]
    pub disk_cache: Option<u64>,
    /// Streamed groups loaded ahead of use (0 = no prefetch).
    #[arg(long)]
    pub prefetch_depth: Option<usize>,
    /// Streaming I/O mode.
    #[arg(long, value_enum)]
    pub io: Option<IoArg>,
    /// I/O worker threads.
    #[arg(long)]
    pub io_workers: Option<usize>,
    /// Ring eviction policy (lru is an ablation).
    #[arg(long, value_enum, default_value = "belady")]
    pub policy: PolicyArg,
    /// Force a strategy.
    #[arg(long, value_enum)]
    pub strategy: Option<StrategyArg>,
    /// Forbid NVMe streaming.
    #[arg(long)]
    pub no_stream: bool,
    /// Where streamed groups sit in the layer order.
    #[arg(long, value_enum, default_value = "interleaved")]
    pub placement: PlacementArg,
    /// Compute threads (default: physical cores).
    #[arg(long, short = 't')]
    pub threads: Option<usize>,
    /// Allow budgets larger than currently available memory.
    #[arg(long)]
    pub allow_overcommit: bool,
    /// Disable runtime promotion/demotion of layers.
    #[arg(long)]
    pub no_adapt: bool,
    /// Expert-cache policy for MoE models (lru is an ablation).
    #[arg(long, value_enum, default_value = "lfru")]
    pub expert_policy: ExpertPolicyArg,
    /// Do not load or save the expert usage history (MoE warm start).
    #[arg(long)]
    pub no_usage_history: bool,
    /// Skip the automatic first-run hardware benchmark.
    #[arg(long)]
    pub no_bench: bool,
}

impl Overrides {
    pub fn request(&self, model: &ModelDesc) -> Result<PlanRequest> {
        let kv_type = match self.kv_type.to_ascii_lowercase().as_str() {
            "f16" => GgmlType::F16,
            "q8_0" | "q8" => GgmlType::Q8_0,
            "q4_0" => GgmlType::Q4_0,
            "f32" => GgmlType::F32,
            other => bail!("unsupported --kv-type {other}"),
        };
        let native_support = kestrel_engine::check_support(model);
        if kv_type != GgmlType::F16 && !matches!(self.backend, BackendArg::Llamacpp) {
            // The native executor keeps an f16 cache; size the plan for that.
        }
        Ok(PlanRequest {
            backend: match self.backend {
                BackendArg::Auto => None,
                BackendArg::Native => Some(Backend::Native),
                BackendArg::Llamacpp => Some(Backend::LlamaCpp),
            },
            n_ctx: self.ctx,
            kv_type,
            vram_budget: self.vram_budget,
            ram_budget: self.ram_budget,
            kv_budget: self.kv_cache,
            allow_overcommit: self.allow_overcommit,
            prefetch_depth: self.prefetch_depth,
            io_mode: self.io.map(|i| match i {
                IoArg::Direct => IoMode::Direct,
                IoArg::Buffered => IoMode::Buffered,
            }),
            io_workers: self.io_workers,
            ring_policy: match self.policy {
                PolicyArg::Belady => RingPolicy::Belady,
                PolicyArg::Lru => RingPolicy::Lru,
            },
            strategy: self.strategy.map(|s| match s {
                StrategyArg::GpuFull => StrategyKind::GpuFull,
                StrategyArg::Hybrid => StrategyKind::GpuRamHybrid,
                StrategyArg::Ram => StrategyKind::RamOnly,
                StrategyArg::RamNvme => StrategyKind::RamNvme,
                StrategyArg::VramRamNvme => StrategyKind::VramRamNvme,
            }),
            no_stream: self.no_stream,
            placement: match self.placement {
                PlacementArg::Interleaved => PlacementOrder::Interleaved,
                PlacementArg::Contiguous => PlacementOrder::Contiguous,
            },
            threads: self.threads,
            native_support,
            llamacpp_available: kestrel_backends::llamacpp::server_available(),
        })
    }
}

fn model_dirs() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(p) = std::env::var_os("KESTREL_MODELS") {
        v.extend(std::env::split_paths(&p));
    }
    v.push(kestrel_hw::cache_dir().join("models"));
    v.push(PathBuf::from("models"));
    v.push(PathBuf::from("."));
    v
}

/// All `.gguf` files under the model directories (one level deep).
pub fn list_models() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for d in model_dirs() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                if let Ok(sub) = std::fs::read_dir(&p) {
                    out.extend(sub.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "gguf")));
                }
            } else if p.extension().is_some_and(|x| x == "gguf") {
                out.push(p);
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Resolve a model argument: a path, or a (partial) name such as
/// `Qwen/Qwen3-32B` or `qwen3-32b-q4_k_m` matched against local GGUF files.
pub fn resolve_model(arg: &str) -> Result<PathBuf> {
    let p = Path::new(arg);
    if p.is_file() {
        return Ok(p.to_path_buf());
    }
    let key = arg.rsplit('/').next().unwrap_or(arg).to_ascii_lowercase();
    let mut hits: Vec<PathBuf> = list_models()
        .into_iter()
        .filter(|m| m.file_name().map(|f| f.to_string_lossy().to_ascii_lowercase().contains(&key)).unwrap_or(false))
        .collect();
    // Exact file-stem match first, then prefix matches, then a Q4_K_M file
    // when several quantizations match.
    hits.sort_by_key(|m| {
        let stem = m.file_stem().map(|s| s.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
        (stem != key, !stem.starts_with(&key), !stem.contains("q4_k_m"), stem.len())
    });
    if let Some(h) = hits.into_iter().next() {
        return Ok(h);
    }
    bail!(
        "model '{arg}' not found locally.\n\
         Kestrel never downloads anything without being asked. Download a GGUF, e.g.\n\
         \x20 huggingface-cli download <repo>-GGUF --include '*Q4_K_M*.gguf' --local-dir {}\n\
         or pass a path, or set KESTREL_MODELS to your model directories.",
        kestrel_hw::cache_dir().join("models").display()
    )
}

pub fn profile_cache() -> PathBuf {
    kestrel_hw::cache_dir().join("hwprofile.json")
}

/// Discover the hardware; attach cached measurements, or run a quick
/// benchmark (a few seconds) the first time unless `no_bench`.
pub fn hardware(model: Option<&ModelDesc>, no_bench: bool, quiet: bool) -> HardwareProfile {
    let mut paths = vec![kestrel_hw::cache_dir()];
    if let Some(m) = model {
        paths.insert(0, m.path.parent().map(Path::to_path_buf).unwrap_or_default());
        paths[0] = std::fs::canonicalize(&paths[0]).unwrap_or(paths[0].clone());
    }
    let mut hw = HardwareProfile::discover(&paths);
    hw.load_cached_measurements(&profile_cache());
    let need_disk = model.is_some_and(|m| hw.measured.as_ref().is_none_or(|x| x.disk_for(&m.path).is_none()));
    if !no_bench && (hw.measured.is_none() || need_disk) {
        if !quiet {
            eprintln!("measuring this machine once (RAM, disk, CPU kernels; cached in {})…", profile_cache().display());
        }
        run_bench(&mut hw, model, true);
        let _ = hw.save_measurements(&profile_cache());
    }
    hw
}

/// Measure bandwidths. `quick` keeps it to a few seconds.
pub fn run_bench(hw: &mut HardwareProfile, model: Option<&ModelDesc>, quick: bool) {
    let mut m = hw.measured.clone().unwrap_or_default();
    let ram_bytes = if quick { 256 << 20 } else { 1 << 30 };
    m.ram_read_bw = bench::ram_bandwidth(ram_bytes, hw.cpu.physical_cores);
    m.ram_read_bw_1t = bench::ram_bandwidth(ram_bytes / 2, 1);
    let dur = Duration::from_millis(if quick { 1200 } else { 4000 });
    // Disk: read the model file itself when it is large enough to defeat
    // caches; otherwise a scratch file next to the model cache.
    let target = match model {
        Some(md) if md.file_size >= 512 << 20 => Some(md.path.clone()),
        _ => bench::scratch_file(&kestrel_hw::cache_dir(), if quick { 256 << 20 } else { 1 << 30 }).ok(),
    };
    if let Some(t) = target {
        if let Ok(d) = bench::disk_bandwidth(&t, IoMode::Direct, 4 << 20, 8, dur) {
            m.disks.retain(|x| x.path != d.path);
            m.disks.push(d);
        }
    }
    let types: Vec<GgmlType> = match model.and_then(|md| md.dominant_type()) {
        Some(t) => all_kernel_types().into_iter().filter(|x| x.name() == t).collect(),
        None => vec![GgmlType::Q4_K, GgmlType::Q8_0],
    };
    let types = if quick { types } else { all_kernel_types() };
    for t in types {
        let rows = 2048;
        let cols = 4096;
        let bw = kestrel_engine::quant::bench_gemv(t, rows, cols, if quick { 3 } else { 10 });
        m.cpu_gemv_bytes_per_s.insert(t.name().to_string(), bw);
    }
    m.when = bench::now_unix();
    hw.measured = Some(m);
}

pub fn all_kernel_types() -> Vec<GgmlType> {
    use GgmlType::*;
    vec![F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K, F32]
}

pub fn open_model(arg: &str) -> Result<(kestrel_gguf::GgufFile, ModelDesc)> {
    let path = resolve_model(arg)?;
    let g = kestrel_gguf::GgufFile::open(&path).with_context(|| format!("reading {}", path.display()))?;
    let m = ModelDesc::from_gguf(&g)?;
    Ok((g, m))
}

pub fn gb(x: u64) -> String {
    fmt_bytes(x)
}
