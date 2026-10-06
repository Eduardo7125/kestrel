//! `kestrel` — run models that don't fit, by planning across VRAM, RAM and NVMe.

mod bench;
mod common;

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use common::{gb, Overrides};
use kestrel_backends::{InferenceSession, NativeSession};
use kestrel_engine::{ChatMessage, GenParams, SamplerConfig};
use kestrel_gguf::GgmlType;
use kestrel_model::GroupKind;
use kestrel_planner::Backend;
use std::io::{BufRead, Write};
use std::sync::{Arc, Mutex};

#[derive(Parser)]
#[command(name = "kestrel", version, about = "Local LLM runtime that plans across VRAM, RAM and NVMe")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// List local GGUF models (KESTREL_MODELS, ~/.cache/kestrel/models, ./models).
    Models,
    /// Describe a model: architecture, tensors, quantization, memory needs.
    Inspect {
        model: String,
        #[arg(long)]
        json: bool,
        /// List every tensor.
        #[arg(long)]
        tensors: bool,
    },
    /// Describe this machine; --bench measures bandwidths.
    Hardware {
        #[arg(long)]
        bench: bool,
        /// Also benchmark the disk holding this file or directory.
        #[arg(long)]
        path: Option<std::path::PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Show the execution plan Kestrel would use.
    Plan {
        model: String,
        #[command(flatten)]
        o: Overrides,
        #[arg(long)]
        json: bool,
    },
    /// Run a model: one prompt (-p) or an interactive chat.
    Run {
        model: String,
        #[command(flatten)]
        o: Overrides,
        /// Prompt; omit for interactive chat.
        #[arg(short, long)]
        prompt: Option<String>,
        /// Treat the prompt as raw text (no chat template).
        #[arg(long)]
        raw: bool,
        #[arg(short = 'n', long, default_value_t = 256)]
        max_tokens: usize,
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Print memory and speed statistics after each answer.
        #[arg(long)]
        stats: bool,
        /// System prompt for chat mode.
        #[arg(long)]
        system: Option<String>,
    },
    /// Serve an OpenAI-compatible API.
    Serve {
        model: String,
        #[command(flatten)]
        o: Overrides,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        #[arg(long, default_value_t = 512)]
        max_tokens: usize,
    },
    /// Benchmark scheduling strategies on this machine (see docs/benchmark-plan.md).
    Benchmark(bench::BenchArgs),
    /// Rewrite a GGUF once into a container laid out for Kestrel's I/O
    /// (lossless: same tensors, same bytes; experts packed one read each).
    Prepare {
        model: String,
        /// Output path (default: next to the model, `<name>.kgguf`).
        #[arg(short, long)]
        output: Option<std::path::PathBuf>,
        /// Overwrite an existing output.
        #[arg(long)]
        force: bool,
        /// Skip re-reading both files to compare every tensor.
        #[arg(long)]
        no_verify: bool,
        /// Only print what would be written.
        #[arg(long)]
        dry_run: bool,
    },
}

fn main() {
    // Behave like a normal Unix tool when piped into `head`.
    #[cfg(unix)]
    unsafe {
        libc_sigpipe_default();
    }
    let cli = Cli::parse();
    let r = match cli.cmd {
        Cmd::Models => cmd_models(),
        Cmd::Inspect { model, json, tensors } => cmd_inspect(&model, json, tensors),
        Cmd::Hardware { bench, path, json } => cmd_hardware(bench, path, json),
        Cmd::Plan { model, o, json } => cmd_plan(&model, &o, json),
        Cmd::Run { model, o, prompt, raw, max_tokens, temperature, seed, stats, system } => cmd_run(&model, &o, prompt, raw, max_tokens, temperature, seed, stats, system),
        Cmd::Serve { model, o, host, port, max_tokens } => cmd_serve(&model, &o, &host, port, max_tokens),
        Cmd::Benchmark(a) => bench::run(a),
        Cmd::Prepare { model, output, force, no_verify, dry_run } => cmd_prepare(&model, output, force, !no_verify, dry_run),
    };
    if let Err(e) = r {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn cmd_prepare(arg: &str, output: Option<std::path::PathBuf>, force: bool, verify: bool, dry_run: bool) -> Result<()> {
    use kestrel_model::prepare;
    let (g, m) = common::open_model(arg)?;
    let out = output.unwrap_or_else(|| prepare::default_output(&g.path));
    if out.exists() && !force && !dry_run {
        bail!("{} exists (use --force to overwrite)", out.display());
    }
    let layout = prepare::plan_layout(&g, &m, &out);
    let r = &layout.report;
    println!("{} → {}", g.path.display(), out.display());
    if m.prepared.is_some() {
        println!("  note            the source is already a prepared container");
    }
    if r.experts_packed > 0 {
        println!(
            "  experts         {} experts in {} MoE layers packed: {:.1} → {:.1} reads per expert",
            r.experts_packed, r.moe_layers_packed, r.reads_per_expert_before, r.reads_per_expert_after
        );
    } else {
        println!("  experts         none (dense model): only group order and page alignment change");
    }
    println!("  dense groups    {} split into several extents in the source", r.split_groups_before);
    println!("  padding         {} added for 4 KiB alignment", gb(r.padding_bytes));
    let need = g.file_size + r.padding_bytes + (64 << 20);
    let dir = out.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(std::path::Path::new("."));
    if let Some(s) = kestrel_hw::describe_storage(dir) {
        println!("  disk            needs ~{}, {} free on {}", gb(need), gb(s.available), s.mount_point.display());
        if s.available < need && !dry_run {
            bail!("not enough free disk space for the prepared copy ({} needed, {} free)", gb(need), gb(s.available));
        }
    }
    if dry_run {
        return Ok(());
    }
    let total = g.total_tensor_bytes() * if verify { 2 } else { 1 };
    let (mut done, mut last) = (0u64, std::time::Instant::now());
    let r = prepare::prepare(&g, &m, &out, verify, |phase, n| {
        done += n;
        if last.elapsed().as_millis() > 500 {
            last = std::time::Instant::now();
            eprint!("\r  {phase:<7} {:>5.1}%", 100.0 * done as f64 / total.max(1) as f64);
        }
    })?;
    eprintln!("\r                    ");
    // Carry the expert usage history over: same experts, new file.
    let new = kestrel_model::ModelDesc::open(&out)?;
    if let Some(u) = kestrel_memory::expert::UsageFile::load(&m) {
        kestrel_memory::expert::UsageFile::save(&new, u)?;
        println!("  usage history   carried over");
    }
    println!(
        "  wrote           {} in {:.1} s ({}/s){}",
        gb(r.output_bytes),
        r.seconds,
        gb((g.total_tensor_bytes() as f64 / r.seconds.max(1e-9)) as u64),
        if r.verified { " · verified: every tensor identical to the source" } else { "" }
    );
    println!("  run it with     kestrel run {}", out.display());
    println!("  the source GGUF is unchanged; keep it for llama.cpp or delete it to save {}", gb(g.file_size));
    Ok(())
}

fn cmd_models() -> Result<()> {
    let ms = common::list_models();
    if ms.is_empty() {
        println!("no GGUF models found (searched KESTREL_MODELS, {}, ./models, .)", kestrel_hw::cache_dir().join("models").display());
        return Ok(());
    }
    println!("{:<50} {:>10}  {:<10} {:<10}", "MODEL", "SIZE", "ARCH", "QUANT");
    for p in ms {
        match kestrel_model::ModelDesc::open(&p) {
            Ok(m) => println!("{:<50} {:>10}  {:<10} {:<10}", p.display(), gb(m.file_size), m.arch, m.file_type.unwrap_or_default()),
            Err(e) => println!("{:<50} (unreadable: {e})", p.display()),
        }
    }
    Ok(())
}

fn cmd_inspect(arg: &str, json: bool, tensors: bool) -> Result<()> {
    let (g, m) = common::open_model(arg)?;
    let kv4k = m.kv_bytes(4096, GgmlType::F16);
    let embed_head: u64 = m.groups.iter().filter(|g| g.layer.is_none()).map(|g| g.bytes).sum();
    let max_layer_group = m.groups.iter().filter(|g| g.layer.is_some()).map(|g| g.bytes).max().unwrap_or(0);
    let overhead = kestrel_planner::runtime_overhead(&m, 4096);
    let resident = m.weight_bytes() + kv4k + overhead;
    let min_stream = embed_head + 3 * max_layer_group + kv4k + overhead;
    if json {
        let v = serde_json::json!({
            "model": m,
            "metadata": g.metadata.iter().filter(|(k, _)| !k.starts_with("tokenizer.ggml.")).map(|(k, v)| (k.clone(), v.summary())).collect::<std::collections::BTreeMap<_, _>>(),
            "estimates": {"kv_bytes_4k_f16": kv4k, "full_residency_bytes": resident, "min_streaming_ram_bytes": min_stream, "disk_bytes": m.file_size},
            "native_support": kestrel_engine::check_support(&m).err(),
        });
        println!("{}", serde_json::to_string_pretty(&v)?);
        return Ok(());
    }
    let h = &m.hparams;
    println!("{} ({})", m.name, m.path.display());
    println!("  architecture    {} — {}", m.arch, kestrel_model::adapter_for(&m.arch).description);
    println!("  parameters      {:.3}B", m.n_params as f64 / 1e9);
    println!("  file            {} · GGUF v{} · {} tensors · {}", gb(m.file_size), g.version, m.tensors.len(), m.file_type.clone().unwrap_or_default());
    if let Some(p) = &m.prepared {
        println!("  prepared        Kestrel container v{} from {} · experts packed: {} · native backend only", p.version, p.source, if p.packed_experts { "yes" } else { "no" });
    }
    println!("  layers          {} · embd {} · ff {} · heads {}/{} kv · head dim {} · vocab {}", h.n_layer, h.n_embd, h.n_ff, h.n_head, h.n_head_kv.first().unwrap_or(&0), h.head_dim_k, h.n_vocab);
    println!("  context         {} trained · rope base {}", h.n_ctx_train, h.rope_freq_base);
    if let Some(moe) = &m.moe {
        println!("  MoE             {} experts, {} active, {} shared · {} per expert · {} MoE layers", moe.n_expert, moe.n_expert_used, moe.n_expert_shared, gb(moe.bytes_per_expert), moe.moe_layers);
    }
    println!("  quantization    {}", m.bytes_by_type.iter().map(|(t, b)| format!("{t} {}", gb(*b))).collect::<Vec<_>>().join(", "));
    let by_kind = |k: GroupKind| m.groups.iter().filter(|g| g.kind == k).map(|g| g.bytes).sum::<u64>();
    println!("  weights         {} total: embed {}, attention {}, ffn {}, experts {}, head {}", gb(m.weight_bytes()), gb(by_kind(GroupKind::Embed)), gb(by_kind(GroupKind::Attn)), gb(by_kind(GroupKind::Ffn)), gb(by_kind(GroupKind::Experts)), gb(by_kind(GroupKind::Head)));
    println!("  KV cache        {} per token (f16) · {} at 4096 ctx", gb(m.kv_bytes_per_token(GgmlType::F16)), gb(kv4k));
    println!("  touched/token   {} (decode reads this much weight data per token)", gb(m.bytes_touched_per_token() as u64));
    println!("\nmemory estimates (4096 ctx, f16 KV)");
    println!("  fully resident  {} (VRAM+RAM combined)", gb(resident));
    println!("  minimum RAM     {} with NVMe streaming (embeddings, head, KV, 3-slot ring)", gb(min_stream));
    println!("  disk            {}", gb(m.file_size));
    match kestrel_engine::check_support(&m) {
        Ok(()) => println!("  native executor supported"),
        Err(e) => println!("  native executor: {e}"),
    }
    if tensors {
        println!("\n{:<40} {:<8} {:>22} {:>12} {:>14}", "TENSOR", "TYPE", "SHAPE", "SIZE", "OFFSET");
        for t in &m.tensors {
            println!("{:<40} {:<8} {:>22} {:>12} {:>14}", t.name, t.ggml_type.name(), format!("{:?}", t.dims), gb(t.size), t.offset);
        }
    }
    Ok(())
}

fn cmd_hardware(bench: bool, path: Option<std::path::PathBuf>, json: bool) -> Result<()> {
    let mut paths = vec![kestrel_hw::cache_dir()];
    if let Some(p) = &path {
        paths.insert(0, std::fs::canonicalize(p)?);
    }
    let mut hw = kestrel_hw::HardwareProfile::discover(&paths);
    hw.load_cached_measurements(&common::profile_cache());
    if bench {
        let model = match &path {
            Some(p) if p.is_file() => kestrel_model::ModelDesc::open(p).ok(),
            _ => None,
        };
        eprintln!("benchmarking (≈20 s)…");
        common::run_bench(&mut hw, model.as_ref(), false);
        if let Some(p) = path.filter(|p| p.is_file()) {
            if let Ok(d) = kestrel_hw::bench::disk_bandwidth(&p, kestrel_hw::fileio::IoMode::Buffered, 4 << 20, 8, std::time::Duration::from_secs(2)) {
                hw.measured.as_mut().unwrap().disks.push(d);
            }
        }
        hw.save_measurements(&common::profile_cache())?;
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&hw)?);
        return Ok(());
    }
    println!("OS      {} {}", hw.os, hw.arch);
    println!("CPU     {} · {} cores / {} threads · {} NUMA node(s)", hw.cpu.model, hw.cpu.physical_cores, hw.cpu.logical_cores, hw.cpu.numa_nodes);
    println!("        SIMD: {}", if hw.cpu.simd.is_empty() { "-".into() } else { hw.cpu.simd.join(" ") });
    if !hw.cpu.caches.is_empty() {
        println!("        cache: {}", hw.cpu.caches.iter().map(|c| format!("L{} {} {}", c.level, c.kind.to_lowercase(), gb(c.size))).collect::<Vec<_>>().join(", "));
    }
    println!("RAM     {} total · {} available{}", gb(hw.ram.total), gb(hw.ram.available), hw.ram.cgroup_limit.map(|l| format!(" · container limit {}", gb(l))).unwrap_or_default());
    if hw.gpus.is_empty() {
        println!("GPU     none detected (CPU execution)");
    }
    for g in &hw.gpus {
        println!(
            "GPU {}   {} · {} total / {} free · {:?}{}{}",
            g.index,
            g.name,
            gb(g.vram_total),
            gb(g.vram_free),
            g.backends,
            g.compute_capability.as_ref().map(|c| format!(" · sm {c}")).unwrap_or_default(),
            g.mem_bandwidth.map(|b| format!(" · {}/s{}", gb(b), if g.mem_bandwidth_is_estimate { " (spec)" } else { "" })).unwrap_or_default()
        );
    }
    for s in &hw.storage {
        println!("DISK    {} · {} · {} · {} free of {}{}", s.mount_point.display(), s.kind, s.filesystem.clone().unwrap_or_default(), gb(s.available), gb(s.capacity), s.model.as_ref().map(|m| format!(" · {m}")).unwrap_or_default());
    }
    match &hw.measured {
        None => println!("\nnot measured yet: run `kestrel hardware --bench`"),
        Some(m) => {
            println!("\nMEASURED");
            println!("  RAM read          {}/s ({} threads), {}/s single thread", gb(m.ram_read_bw as u64), hw.cpu.physical_cores, gb(m.ram_read_bw_1t as u64));
            for d in &m.disks {
                println!(
                    "  disk {:?}  {}/s at QD{} · {}/s QD1 · 4K latency {:.0} µs{} [{}]",
                    d.mode,
                    gb(d.read_bw as u64),
                    d.threads,
                    gb(d.read_bw_qd1 as u64),
                    d.latency_4k_us,
                    if d.cache_suspect { " (may be page cache)" } else { "" },
                    d.path.display()
                );
            }
            for (t, bw) in &m.cpu_gemv_bytes_per_s {
                println!("  native GEMV {t:<5} {}/s of weights", gb(*bw as u64));
            }
        }
    }
    Ok(())
}

fn make_plan(arg: &str, o: &Overrides, quiet: bool) -> Result<(kestrel_gguf::GgufFile, Arc<kestrel_model::ModelDesc>, kestrel_planner::ExecutionPlan)> {
    let (g, m) = common::open_model(arg)?;
    let hw = common::hardware(Some(&m), o.no_bench, quiet);
    let req = o.request_with(&m, &hw)?;
    match kestrel_planner::plan(&m, &hw, &req) {
        Ok(p) => Ok((g, Arc::new(m), p)),
        Err(inf) => {
            eprint!("{}", inf.text());
            bail!("no feasible execution plan")
        }
    }
}

fn cmd_plan(arg: &str, o: &Overrides, json: bool) -> Result<()> {
    let (_, _, p) = make_plan(arg, o, json)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&p)?);
    } else {
        print!("{}", p.text());
    }
    Ok(())
}

fn load_native(arg: &str, o: &Overrides) -> Result<NativeSession> {
    let (g, m, p) = make_plan(arg, o, false)?;
    if p.chosen.backend != Backend::Native {
        bail!("internal: load_native called for a {:?} plan", p.chosen.backend);
    }
    eprintln!(
        "kestrel: {} · {} · {} resident, {} streamed · ctx {}",
        m.name,
        p.chosen.kind.label(),
        gb(p.chosen.ram_weights),
        gb(p.chosen.disk_weights),
        p.n_ctx
    );
    NativeSession::load(p, &g, m, o.native_options())
}

fn print_stats(st: &kestrel_engine::GenStats, sess: &NativeSession) {
    let m = st.memory.clone().unwrap_or_default();
    eprintln!(
        "[{} prompt ({} new) @ {:.1} tok/s · {} generated @ {:.2} tok/s · ttft {:.2}s]",
        st.prompt_tokens, st.prompt_evaluated, st.prompt_tok_s, st.generated, st.decode_tok_s, st.ttft_s
    );
    eprintln!(
        "[weights: hit {:.0}% · stall {:.2}s · streamed {} ({}/s) · prefetch {:.0}% accurate · RSS {}]",
        m.hit_rate * 100.0,
        m.stall_s,
        gb(m.stream_bytes),
        gb(m.disk_bw as u64),
        m.prefetch_accuracy * 100.0,
        gb(kestrel_hw::process_rss().unwrap_or(0))
    );
    if let Some(e) = &st.experts {
        eprintln!(
            "[experts: hit {:.0}% ({} of {} requests) · {} cached / {} capacity · scratch {} · lookahead recall {:.0}% · prefetch {} issued, {} used · stall {:.2}s]",
            e.hit_rate * 100.0, e.hits, e.requests, e.cached, e.capacity, e.scratch_loads, e.lookahead_recall * 100.0, e.prefetch_issued, e.prefetch_used, e.stall_s
        );
    }
    if let Some(p) = &sess.engine.tf.profile {
        eprintln!("[profile: matmul {:.2}s · attention {:.2}s · weight wait {:.2}s · other {:.2}s]", p.matmul_s, p.attention_s, p.lease_s, p.other_s);
    }
    if !sess.totals.rebalance.is_empty() {
        eprintln!("[rebalance: {:?}]", sess.totals.rebalance.last().unwrap());
    }
}

#[allow(clippy::too_many_arguments)]
fn cmd_run(arg: &str, o: &Overrides, prompt: Option<String>, raw: bool, max_tokens: usize, temperature: f32, seed: u64, stats: bool, system: Option<String>) -> Result<()> {
    let (_, _, p) = make_plan(arg, o, true)?;
    if p.chosen.backend == Backend::LlamaCpp {
        // Kestrel planned; llama.cpp executes.
        let Some(cli) = kestrel_backends::llamacpp::find_tool("llama-cli") else { bail!("plan selected llama.cpp but llama-cli was not found (KESTREL_LLAMA_CLI / KESTREL_LLAMA_CPP / PATH)") };
        eprint!("{}", p.text());
        let mut cmd = std::process::Command::new(cli);
        cmd.args(&p.llamacpp_args).args(["-n", &max_tokens.to_string(), "--temp", &temperature.to_string(), "--seed", &seed.to_string()]);
        if let Some(pr) = &prompt {
            cmd.args(["-p", pr, "-no-cnv"]);
        }
        let st = cmd.status()?;
        if !st.success() {
            bail!("llama-cli exited with {st}");
        }
        return Ok(());
    }
    let mut sess = load_native(arg, o)?;
    let params = GenParams {
        max_tokens,
        sampler: SamplerConfig { temperature, seed, ..Default::default() },
        ..Default::default()
    };
    let stdout = std::io::stdout();
    let mut emit = |s: &str| {
        let mut l = stdout.lock();
        let _ = l.write_all(s.as_bytes());
        let _ = l.flush();
        true
    };
    if let Some(pr) = prompt {
        let text = if raw { pr } else { sess.render_chat(&[ChatMessage { role: "user".into(), content: pr }])? };
        let toks = sess.tokenize(&text);
        let st = sess.generate(&toks, &params, &mut emit)?;
        println!();
        if stats {
            print_stats(&st, &sess);
        }
        return Ok(());
    }
    eprintln!("chat mode — empty line or Ctrl-D to quit, /reset to clear history");
    let mut history: Vec<ChatMessage> = system.into_iter().map(|s| ChatMessage { role: "system".into(), content: s }).collect();
    let stdin = std::io::stdin();
    loop {
        eprint!("› ");
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 || line.trim().is_empty() {
            break;
        }
        if line.trim() == "/reset" {
            history.retain(|m| m.role == "system");
            continue;
        }
        history.push(ChatMessage { role: "user".into(), content: line.trim_end().to_string() });
        let toks = sess.tokenize(&sess.render_chat(&history)?);
        let mut answer = String::new();
        let st = sess.generate(&toks, &params, &mut |s| {
            answer.push_str(s);
            emit(s)
        })?;
        println!();
        history.push(ChatMessage { role: "assistant".into(), content: answer });
        if stats {
            print_stats(&st, &sess);
        }
    }
    Ok(())
}

fn cmd_serve(arg: &str, o: &Overrides, host: &str, port: u16, max_tokens: usize) -> Result<()> {
    let (_, _, p) = make_plan(arg, o, false)?;
    if p.chosen.backend == Backend::LlamaCpp {
        eprint!("{}", p.text());
        eprintln!("\nlaunching llama-server on http://{host}:{port}/v1 with Kestrel's placement");
        let mut child = kestrel_backends::llamacpp::spawn_server(&p.llamacpp_args, host, port, &[])?;
        let st = child.wait()?;
        if !st.success() {
            bail!("llama-server exited with {st}");
        }
        return Ok(());
    }
    let sess = load_native(arg, o)?;
    eprintln!("serving {} on http://{host}:{port}/v1 (OpenAI-compatible; /metrics for Prometheus)", sess.model_name());
    let shared: kestrel_server::SharedSession = Arc::new(Mutex::new(Box::new(sess) as Box<dyn InferenceSession>));
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build()?;
    rt.block_on(kestrel_server::serve(shared, &format!("{host}:{port}"), max_tokens))
}

#[cfg(unix)]
unsafe fn libc_sigpipe_default() {
    extern "C" {
        fn signal(sig: i32, handler: usize) -> usize;
    }
    const SIGPIPE: i32 = 13;
    signal(SIGPIPE, 0);
}
