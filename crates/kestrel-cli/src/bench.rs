//! `kestrel benchmark`: compare scheduling strategies on this machine.
//!
//! Methodology (docs/benchmark-plan.md): every arm runs in a fresh child
//! process (so peak RSS is per arm), arms are interleaved across repetitions
//! (A B C A B C …), the model's page-cache pages are dropped before each run,
//! and every arm's greedy output must be token-identical to the first arm's.

use crate::common::{self, gb, Overrides};
use anyhow::{bail, Context, Result};
use clap::Args;
use kestrel_backends::{InferenceSession, NativeOptions, NativeSession};
use kestrel_engine::{GenParams, SamplerConfig};
use kestrel_hw::fileio::IoMode;
use kestrel_memory::{RingPolicy, Tier};
use kestrel_planner::{Backend, PlacementOrder};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Args, Clone, Debug)]
pub struct BenchArgs {
    pub model: String,
    /// Comma-separated arms, or "auto". Native arms: resident, kestrel,
    /// no-prefetch, page-cache, contiguous, lru-cache, belady-cache.
    /// llama.cpp arms (need llama-bench): llamacpp-cpu, llamacpp-auto.
    #[arg(long, default_value = "auto")]
    pub arms: String,
    /// Fraction of layer weights to stream in the streaming arms.
    #[arg(long, default_value_t = 0.5)]
    pub stream_fraction: f64,
    #[arg(long, default_value_t = 3)]
    pub runs: usize,
    /// Tokens to generate per run.
    #[arg(long, default_value_t = 32)]
    pub tokens: usize,
    /// Prompt length in tokens (approximate).
    #[arg(long, default_value_t = 32)]
    pub prompt_tokens: usize,
    #[arg(long, short = 't')]
    pub threads: Option<usize>,
    /// Sampling temperature (fixed seed, so arms stay comparable). Default:
    /// 0 (greedy) for dense models, 1.0 for MoE models, where greedy decoding
    /// of random weights repeats one token and freezes the routing.
    #[arg(long)]
    pub temperature: Option<f32>,
    /// Write raw results here.
    #[arg(long)]
    pub json: Option<std::path::PathBuf>,
    /// Internal: run one arm in this process and print its JSON result.
    #[arg(long, hide = true)]
    pub single_arm: Option<String>,
    #[arg(long, hide = true)]
    pub ram_budget_bytes: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArmRun {
    pub arm: String,
    pub run: usize,
    pub load_s: f64,
    pub prompt_tokens: usize,
    pub prefill_tok_s: f64,
    pub ttft_s: f64,
    pub decode_tok_s: f64,
    pub generated: usize,
    pub peak_rss: u64,
    pub ram_budget: u64,
    pub resident_weight_bytes: u64,
    pub streamed_bytes_per_token: u64,
    pub hit_rate: f64,
    pub stall_s: f64,
    pub stream_bytes: u64,
    pub disk_bw: f64,
    pub prefetch_accuracy: f64,
    pub late_fraction: f64,
    /// Growth of the OS page cache during the run (Linux `Cached`), bytes.
    pub page_cache_growth: i64,
    pub output_hash: String,
    pub note: String,
    /// MoE: expert-cache hit rate, router-lookahead recall, expert bytes read.
    #[serde(default)]
    pub expert_hit_rate: Option<f64>,
    #[serde(default)]
    pub lookahead_recall: Option<f64>,
    #[serde(default)]
    pub expert_bytes: Option<u64>,
}

fn meminfo_cached() -> i64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("Cached:")).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<i64>().ok()))
        .map(|kb| kb * 1024)
        .unwrap_or(0)
}

fn drop_file_cache(path: &std::path::Path) {
    if let Ok(f) = kestrel_hw::fileio::ReadFile::open(path, IoMode::Buffered) {
        f.drop_cache(0, 0);
    }
}

const NATIVE_ARMS: &[&str] = &["resident", "kestrel", "no-prefetch", "page-cache", "contiguous", "lru-cache", "belady-cache"];
/// MoE models: the expert cache is the variable (`--stream-fraction` = share
/// of expert bytes that do not fit in it).
const MOE_ARMS: &[&str] = &["resident", "kestrel", "lookahead-always", "no-lookahead", "lru-experts"];

pub fn run(a: BenchArgs) -> Result<()> {
    if let Some(arm) = a.single_arm.clone() {
        let r = run_arm(&a, &arm)?;
        println!("KESTREL_ARM_RESULT {}", serde_json::to_string(&r)?);
        return Ok(());
    }
    let (_, m) = common::open_model(&a.model)?;
    let hw = common::hardware(Some(&m), false, false);
    let default_arms = if m.moe.is_some() { MOE_ARMS } else { NATIVE_ARMS };
    let mut arms: Vec<String> = if a.arms == "auto" { default_arms.iter().map(|s| s.to_string()).collect() } else { a.arms.split(',').map(|s| s.trim().to_string()).collect() };
    let llama_bench = kestrel_backends::llamacpp::find_tool("llama-bench");
    if a.arms == "auto" && llama_bench.is_some() {
        arms.extend(["llamacpp-cpu".to_string()]);
    }
    // RAM budget for streaming arms: keep (1 - f) of layer bytes resident.
    let layer_bytes: u64 = m.groups.iter().filter(|g| g.layer.is_some()).map(|g| g.bytes).sum();
    let other: u64 = m.weight_bytes() - layer_bytes;
    let max_group = m.groups.iter().filter(|g| g.layer.is_some()).map(|g| g.bytes).max().unwrap_or(0);
    let n_ctx = 512u64;
    let kv = m.kv_bytes(n_ctx, kestrel_gguf::GgmlType::F16);
    let mut budget = other + ((1.0 - a.stream_fraction) * layer_bytes as f64) as u64 + 3 * max_group + kv + 64 * 4096 + kestrel_planner::runtime_overhead(&m, n_ctx);
    if let Some(moe) = &m.moe {
        // Dense part resident; the fraction applies to routed-expert bytes.
        let expert_bytes: u64 = m.groups.iter().filter(|g| g.kind == kestrel_model::GroupKind::Experts).map(|g| g.bytes).sum();
        let unit = moe.bytes_per_expert + 3 * 4096;
        let scratch = 4 * moe.n_expert_used as u64 * unit;
        let cached = ((1.0 - a.stream_fraction) * expert_bytes as f64) as u64;
        let n_tot = moe.n_expert as u64 * moe.moe_layers as u64;
        budget = m.weight_bytes() - expert_bytes + cached + cached / moe.bytes_per_expert.max(1) * 3 * 4096 + scratch + kv + 64 * 4096 + kestrel_planner::runtime_overhead(&m, n_ctx);
        eprintln!("MoE: {} experts of {} each; cache for ~{:.0}% of them", n_tot, gb(moe.bytes_per_expert), (1.0 - a.stream_fraction) * 100.0);
    }
    eprintln!(
        "model {} · {} weights ({} in layers) · streaming arms: RAM budget {} (~{:.0}% of layer bytes streamed)",
        m.name,
        gb(m.weight_bytes()),
        gb(layer_bytes),
        gb(budget),
        a.stream_fraction * 100.0
    );
    eprintln!("arms: {}  ·  runs: {} (interleaved)  ·  {} tokens", arms.join(", "), a.runs, a.tokens);

    let exe = std::env::current_exe()?;
    let mut results: Vec<ArmRun> = Vec::new();
    for run in 0..a.runs {
        for arm in &arms {
            drop_file_cache(&m.path);
            let r = if arm.starts_with("llamacpp") {
                match &llama_bench {
                    Some(lb) => llamacpp_arm(lb, &m.path.to_string_lossy(), arm, &a, run),
                    None => {
                        eprintln!("  skip {arm}: llama-bench not found");
                        continue;
                    }
                }
            } else {
                let mut cmd = std::process::Command::new(&exe);
                cmd.args(["benchmark", &m.path.to_string_lossy(), "--single-arm", arm, "--tokens", &a.tokens.to_string(), "--prompt-tokens", &a.prompt_tokens.to_string(), "--runs", "1"]);
                cmd.args(["--ram-budget-bytes", &budget.to_string()]);
                if let Some(t) = a.threads {
                    cmd.args(["--threads", &t.to_string()]);
                }
                if let Some(t) = a.temperature {
                    cmd.args(["--temperature", &t.to_string()]);
                }
                if arm == "no-lookahead" {
                    cmd.env("KESTREL_LOOKAHEAD", "0");
                }
                if arm == "lookahead-always" {
                    cmd.env("KESTREL_SPEC_MIN_ACCURACY", "0");
                }
                let out = cmd.output()?;
                let stdout = String::from_utf8_lossy(&out.stdout);
                match stdout.lines().find_map(|l| l.strip_prefix("KESTREL_ARM_RESULT ")) {
                    Some(j) => {
                        let mut r: ArmRun = serde_json::from_str(j)?;
                        r.run = run;
                        Ok(r)
                    }
                    None => Err(anyhow::anyhow!("{arm} failed: {}", String::from_utf8_lossy(&out.stderr).lines().last().unwrap_or(""))),
                }
            };
            match r {
                Ok(r) => {
                    eprintln!("  run {} {:<13} decode {:>7.2} tok/s · prefill {:>7.1} tok/s · peak RSS {:>9} · hit {:>4.0}% · stall {:>6.2}s", run + 1, r.arm, r.decode_tok_s, r.prefill_tok_s, gb(r.peak_rss), r.hit_rate * 100.0, r.stall_s);
                    results.push(r);
                }
                Err(e) => eprintln!("  run {} {arm}: {e}", run + 1),
            }
        }
    }
    summarize(&results, &arms);
    if let Some(path) = &a.json {
        let doc = serde_json::json!({
            "kestrel_version": env!("CARGO_PKG_VERSION"),
            "hardware": hw,
            "model": {"name": m.name, "path": m.path, "arch": m.arch, "file_size": m.file_size, "weight_bytes": m.weight_bytes(), "fingerprint": m.fingerprint, "quant": m.file_type},
            "settings": {"stream_fraction": a.stream_fraction, "stream_ram_budget": budget, "tokens": a.tokens, "prompt_tokens": a.prompt_tokens, "runs": a.runs},
            "runs": results,
        });
        std::fs::write(path, serde_json::to_string_pretty(&doc)?)?;
        eprintln!("raw results → {}", path.display());
    }
    Ok(())
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() {
        0.0
    } else {
        v[v.len() / 2]
    }
}

fn summarize(results: &[ArmRun], arms: &[String]) {
    println!("\n{:<14} {:>12} {:>12} {:>10} {:>11} {:>7} {:>9} {:>10} {:>9}  output", "ARM", "decode tok/s", "prefill t/s", "peak RSS", "streamed/t", "hit", "stall/tok", "disk BW", "prefetch");
    let reference: Option<&str> = results.iter().find(|r| !r.output_hash.is_empty()).map(|r| r.output_hash.as_str());
    let mut by: BTreeMap<&str, Vec<&ArmRun>> = BTreeMap::new();
    for r in results {
        by.entry(r.arm.as_str()).or_default().push(r);
    }
    for arm in arms {
        let Some(rs) = by.get(arm.as_str()) else { continue };
        let dec: Vec<f64> = rs.iter().map(|r| r.decode_tok_s).collect();
        let (lo, hi) = (dec.iter().cloned().fold(f64::MAX, f64::min), dec.iter().cloned().fold(0.0, f64::max));
        let r0 = rs[0];
        let same = match (reference, r0.output_hash.as_str()) {
            (_, "") => "n/a",
            (Some(h), x) if rs.iter().all(|r| r.output_hash == x) && x == h => "identical",
            _ => "DIFFERS",
        };
        println!(
            "{:<14} {:>5.2} [{:.2}-{:.2}] {:>12.1} {:>10} {:>11} {:>6.0}% {:>8.3}s {:>8}/s {:>8.0}%  {}",
            arm,
            median(dec),
            lo,
            hi,
            median(rs.iter().map(|r| r.prefill_tok_s).collect()),
            gb(rs.iter().map(|r| r.peak_rss).max().unwrap_or(0)),
            gb(r0.streamed_bytes_per_token),
            median(rs.iter().map(|r| r.hit_rate).collect()) * 100.0,
            median(rs.iter().map(|r| r.stall_s / r.generated.max(1) as f64).collect()),
            gb(median(rs.iter().map(|r| r.disk_bw).collect()) as u64),
            median(rs.iter().map(|r| r.prefetch_accuracy).collect()) * 100.0,
            same
        );
        if let Some(eh) = r0.expert_hit_rate {
            let recall: Vec<f64> = rs.iter().filter_map(|r| r.lookahead_recall).collect();
            println!(
                "{:<14} experts: hit {:.0}% · read {}/token · lookahead recall {}",
                "",
                median(rs.iter().filter_map(|r| r.expert_hit_rate).collect()) * 100.0,
                gb(r0.expert_bytes.unwrap_or(0) / (r0.generated.max(1) + 1) as u64),
                if recall.is_empty() { "off".to_string() } else { format!("{:.0}%", median(recall) * 100.0) }
            );
            let _ = eh;
        }
        if !r0.note.is_empty() {
            println!("{:<14} note: {}", "", r0.note);
        }
    }
}

/// Run one arm in this process.
fn run_arm(a: &BenchArgs, arm: &str) -> Result<ArmRun> {
    let (g, m) = common::open_model(&a.model)?;
    let budget = a.ram_budget_bytes.context("--ram-budget-bytes")?;
    let streaming = arm != "resident";
    let o = Overrides {
        backend: common::BackendArg::Native,
        ctx: Some(512),
        vram_budget: None,
        ram_budget: if streaming { Some(budget) } else { None },
        kv_cache: None,
        kv_type: "f16".into(),
        disk_cache: None,
        prefetch_depth: Some(match arm {
            "no-prefetch" | "page-cache" | "lru-cache" => 0,
            _ => 2,
        }),
        io: Some(if arm == "page-cache" { common::IoArg::Buffered } else { common::IoArg::Direct }),
        io_workers: None,
        policy: if arm == "lru-cache" { common::PolicyArg::Lru } else { common::PolicyArg::Belady },
        strategy: None,
        no_stream: arm == "resident",
        placement: if arm == "contiguous" { common::PlacementArg::Contiguous } else { common::PlacementArg::Interleaved },
        threads: a.threads,
        allow_overcommit: streaming,
        no_adapt: true,
        no_bench: true,
        expert_policy: if arm == "lru-experts" { common::ExpertPolicyArg::Lru } else { common::ExpertPolicyArg::Lfru },
        no_usage_history: true,
    };
    let hw = common::hardware(Some(&m), true, true);
    let req = o.request(&m)?;
    let mut plan = match kestrel_planner::plan(&m, &hw, &req) {
        Ok(p) => p,
        Err(inf) => bail!("{}", inf.text().lines().next().unwrap_or("infeasible")),
    };
    if plan.chosen.backend != Backend::Native {
        bail!("arm {arm} needs the native backend");
    }
    let mut note = String::new();
    if matches!(arm, "lru-cache" | "belady-cache") {
        // Same RAM, used as a cache instead of a static pin: no resident
        // layer groups, as many ring slots as the resident layers occupied.
        let slot = m.groups.iter().filter(|g| g.layer.is_some()).map(|g| g.bytes).max().unwrap_or(1);
        let resident_layer_bytes: u64 = m.groups.iter().filter(|g| g.layer.is_some() && plan.chosen.tiers[g.id] != Tier::Disk).map(|g| g.bytes).sum();
        let slots = ((resident_layer_bytes + plan.chosen.ring_bytes) / slot).max(1) as usize;
        for g in m.groups.iter().filter(|g| g.layer.is_some()) {
            plan.chosen.tiers[g.id] = Tier::Disk;
        }
        plan.store.ring_slots = slots;
        plan.store.policy = if arm == "lru-cache" { RingPolicy::Lru } else { RingPolicy::Belady };
        note = format!("all layers through a {slots}-slot {:?} cache of the same RAM", plan.store.policy);
    }
    if arm == "adaptive" {
        note = "starts at the streaming plan; rebalancer promotes in the background as RAM allows".into();
    }
    if plan.placement == PlacementOrder::Contiguous {
        note = "streamed layers contiguous".into();
    }
    let resident_weight_bytes: u64 = m.groups.iter().filter(|g| plan.chosen.tiers[g.id] != Tier::Disk).map(|g| g.bytes).sum();
    let ram_budget = plan.budgets.ram.usable;
    let m = std::sync::Arc::new(m);
    let cache0 = meminfo_cached();
    let mut sess = NativeSession::load(plan, &g, m.clone(), NativeOptions { adaptive: arm == "adaptive", ..o.native_options() })?;
    if arm == "adaptive" {
        // Start from the streaming plan, then let the rebalancer use the
        // RAM this machine actually has: promotions load in the background
        // and are installed between tokens.
        sess.ledger.set_limit(Tier::Ram, u64::MAX / 4);
        sess.configure_rebalancer(2, 1, kestrel_hw::HardwareProfile::discover(&[]).ram.total * 3 / 4);
    }
    let load_s = sess.load_s;

    // A deterministic prompt of roughly the requested length.
    let base = "The quick brown fox jumps over the lazy dog while the scheduler streams layers from disk. ";
    let mut text = String::new();
    while sess.tokenize(&text).len() < a.prompt_tokens {
        text.push_str(base);
    }
    let mut toks = sess.tokenize(&text);
    toks.truncate(a.prompt_tokens.max(2));
    let temp = a.temperature.unwrap_or(if m.moe.is_some() { 1.0 } else { 0.0 });
    let sampler = if temp > 0.0 { SamplerConfig { temperature: temp, top_k: 0, top_p: 1.0, min_p: 0.0, repeat_penalty: 1.0, repeat_last_n: 0, seed: 42 } } else { SamplerConfig::greedy() };
    let params = GenParams { max_tokens: a.tokens, sampler, ignore_eos: true, ..Default::default() };
    if arm == "migration" {
        // RQ 7: cost of moving groups between tiers at runtime. Promote every
        // streamed group (disk → RAM, blocking), then demote them again,
        // timing each; then decode with everything resident.
        let streamed: Vec<usize> = (0..sess.store.model().groups.len())
            .filter(|&g| !sess.store.is_resident(g) && !sess.store.is_external(g))
            .collect();
        // This arm measures transfer cost, not budgeting: lift the ceiling.
        sess.ledger.set_limit(Tier::Ram, u64::MAX / 4);
        let (mut p_bytes, mut p_s, mut d_s) = (0u64, 0f64, 0f64);
        for &g in &streamed {
            let t = std::time::Instant::now();
            sess.store.promote(g)?;
            p_s += t.elapsed().as_secs_f64();
            p_bytes += sess.store.group_buf_len(g) as u64;
        }
        let promoted = sess.generate(&toks, &params, &mut |_| true)?;
        for &g in &streamed {
            let t = std::time::Instant::now();
            sess.store.demote(g)?;
            d_s += t.elapsed().as_secs_f64();
        }
        note = format!(
            "{} groups promoted: {:.1} ms each ({}/s), demoted: {:.3} ms each; decode after promotion {:.2} tok/s",
            streamed.len(),
            p_s * 1e3 / streamed.len().max(1) as f64,
            gb((p_bytes as f64 / p_s.max(1e-9)) as u64),
            d_s * 1e3 / streamed.len().max(1) as f64,
            promoted.decode_tok_s
        );
    }
    let st = sess.generate(&toks, &params, &mut |_| true)?;
    let mem = st.memory.clone().unwrap_or_default();
    let mut h: u64 = 0xcbf29ce484222325;
    for t in &st.tokens {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    Ok(ArmRun {
        arm: arm.to_string(),
        run: 0,
        load_s,
        prompt_tokens: toks.len(),
        prefill_tok_s: st.prompt_tok_s,
        ttft_s: st.ttft_s,
        decode_tok_s: st.decode_tok_s,
        generated: st.generated,
        peak_rss: kestrel_hw::process_peak_rss().unwrap_or(0),
        ram_budget,
        resident_weight_bytes,
        streamed_bytes_per_token: sess.store.streamed_bytes_per_pass(),
        hit_rate: st.experts.as_ref().map(|e| e.hit_rate).unwrap_or(mem.hit_rate),
        stall_s: mem.stall_s + st.experts.as_ref().map(|e| e.stall_s).unwrap_or(0.0),
        stream_bytes: mem.stream_bytes,
        disk_bw: mem.disk_bw,
        prefetch_accuracy: mem.prefetch_accuracy,
        late_fraction: mem.late_fraction,
        page_cache_growth: meminfo_cached() - cache0,
        output_hash: format!("{h:016x}"),
        note: if arm == "adaptive" { format!("{note}; {} promotions during the run", sess.totals.rebalance.len()) } else { note },
        expert_hit_rate: st.experts.as_ref().map(|e| e.hit_rate),
        lookahead_recall: st.experts.as_ref().filter(|e| e.lookahead_predicted > 0).map(|e| e.lookahead_recall),
        expert_bytes: st.experts.as_ref().map(|e| e.bytes_read),
    })
}

fn llamacpp_arm(bin: &std::path::Path, model: &str, arm: &str, a: &BenchArgs, run: usize) -> Result<ArmRun> {
    let mut cmd = std::process::Command::new(bin);
    cmd.args(["-m", model, "-p", &a.prompt_tokens.to_string(), "-n", &a.tokens.to_string(), "-r", "1", "-o", "json"]);
    if arm == "llamacpp-cpu" {
        cmd.args(["-ngl", "0"]);
    }
    if let Some(t) = a.threads {
        cmd.args(["-t", &t.to_string()]);
    }
    let out = cmd.output()?;
    if !out.status.success() {
        bail!("llama-bench failed: {}", String::from_utf8_lossy(&out.stderr).lines().last().unwrap_or(""));
    }
    let v: serde_json::Value = serde_json::from_slice(&out.stdout)?;
    let rows = v.as_array().cloned().unwrap_or_default();
    let ts = |is_gen: bool| {
        rows.iter()
            .find(|r| (r["n_gen"].as_u64().unwrap_or(0) > 0) == is_gen)
            .and_then(|r| r["avg_ts"].as_f64())
            .unwrap_or(0.0)
    };
    Ok(ArmRun {
        arm: arm.into(),
        run,
        load_s: 0.0,
        prompt_tokens: a.prompt_tokens,
        prefill_tok_s: ts(false),
        ttft_s: 0.0,
        decode_tok_s: ts(true),
        generated: a.tokens,
        peak_rss: 0,
        ram_budget: 0,
        resident_weight_bytes: 0,
        streamed_bytes_per_token: 0,
        hit_rate: 0.0,
        stall_s: 0.0,
        stream_bytes: 0,
        disk_bw: 0.0,
        prefetch_accuracy: 0.0,
        late_fraction: 0.0,
        page_cache_growth: 0,
        output_hash: String::new(),
        note: "llama.cpp (mmap, own kernels, own placement); RSS not measured".into(),
        expert_hit_rate: None,
        lookahead_recall: None,
        expert_bytes: None,
    })
}
