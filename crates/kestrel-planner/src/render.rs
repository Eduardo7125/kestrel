use crate::{Backend, Candidate, ExecutionPlan, Infeasible};
use kestrel_hw::{fmt_bytes as fb, HardwareProfile};
use std::fmt::Write;

pub fn hardware_line(hw: &HardwareProfile) -> String {
    let mut s = format!("{} ({} cores / {} threads), {} RAM", hw.cpu.model, hw.cpu.physical_cores, hw.cpu.logical_cores, fb(hw.ram.total));
    for g in &hw.gpus {
        let _ = write!(s, ", {} {}", g.name, fb(g.vram_total));
    }
    if hw.gpus.is_empty() {
        s.push_str(", no GPU");
    }
    s
}

fn backend_name(b: Backend) -> &'static str {
    match b {
        Backend::Native => "native",
        Backend::LlamaCpp => "llama.cpp",
    }
}

pub fn plan_text(p: &ExecutionPlan) -> String {
    let c = &p.chosen;
    let mut s = String::new();
    let _ = writeln!(s, "MODEL\n  {} · {} · {:.2}B params · {} · {}", p.model_name, p.arch, p.n_params as f64 / 1e9, p.quant, fb(p.weight_bytes));
    let _ = writeln!(s, "\nHARDWARE\n  {}", p.hardware);
    let b = &p.budgets;
    let _ = writeln!(s, "\nBUDGETS (available − safety − overhead = usable)");
    if b.vram.total > 0 {
        let _ = writeln!(s, "  VRAM  {} − {} − {} = {}", fb(b.vram.available), fb(b.vram.safety), fb(b.vram.overhead), fb(b.vram.usable));
    }
    let _ = writeln!(s, "  RAM   {} − {} − {} = {}", fb(b.ram.available), fb(b.ram.safety), fb(b.ram.overhead), fb(b.ram.usable));

    let _ = writeln!(s, "\nPLAN  [{} · backend {}]", c.kind.label(), backend_name(c.backend));
    if c.vram_weights > 0 {
        let _ = writeln!(
            s,
            "  VRAM  {} weights{} + KV {}",
            fb(c.vram_weights),
            c.n_gpu_layers.map(|n| format!(" (n_gpu_layers {n})")).unwrap_or_default(),
            fb(c.kv_vram_bytes)
        );
    }
    let kv_ram = c.kv_bytes - c.kv_vram_bytes;
    let _ = writeln!(s, "  RAM   {} weights resident{}", fb(c.ram_weights), if kv_ram > 0 { format!(" + KV {}", fb(kv_ram)) } else { String::new() });
    if c.disk_weights > 0 {
        let how = if c.backend == Backend::Native { format!("streamed through a {} ring, prefetch depth {}", fb(c.ring_bytes), p.store.prefetch_depth) } else { "paged by mmap".into() };
        let _ = writeln!(s, "  NVMe  {} weights {how}", fb(c.disk_weights));
    }
    if let Some(cap) = c.expert_cache {
        let _ = writeln!(s, "  MoE   expert cache holds {cap} routed experts (LFRU, router-lookahead prefetch); the rest stream on demand");
    }
    if c.experts_on_cpu {
        let _ = writeln!(s, "  MoE   routed experts on CPU, attention/shared in VRAM");
    }
    let e = &c.estimate;
    let _ = writeln!(s, "\nESTIMATE");
    if c.vram_total > 0 {
        let _ = writeln!(s, "  VRAM          {}", fb(c.vram_total));
    }
    let _ = writeln!(s, "  RAM           {}", fb(c.ram_total));
    if e.disk_bytes_per_token > 0.0 {
        let _ = writeln!(s, "  Disk read     {}/token", fb(e.disk_bytes_per_token as u64));
    }
    let _ = writeln!(s, "  Context       {} tokens (KV {} {})", p.n_ctx, p.kv_type, fb(c.kv_bytes));
    let _ = writeln!(s, "  Decode        ~{:.1} tok/s (bottleneck: {})", e.tok_s, e.bottleneck);

    let _ = writeln!(s, "\nSTRATEGIES");
    let mut all: Vec<&Candidate> = vec![c];
    all.extend(p.alternatives.iter());
    for (i, a) in all.iter().enumerate() {
        let tag = (b'A' + i as u8) as char;
        let status = if i == 0 {
            "selected".to_string()
        } else if a.feasible {
            String::new()
        } else {
            format!("infeasible: {}", a.reason.clone().unwrap_or_default())
        };
        let moe = if a.experts_on_cpu { " (experts on CPU)" } else { "" };
        let _ = writeln!(s, "  {tag}  {:<22}{:<11}{:>8}  {}", format!("{}{moe}", a.kind.label()), backend_name(a.backend), if a.feasible { format!("{:.1} tok/s", a.estimate.tok_s) } else { "—".into() }, status);
    }
    if !p.bandwidths.estimated.is_empty() {
        let _ = writeln!(s, "\nASSUMPTIONS\n  {}", p.bandwidths.estimated.join("\n  "));
    }
    for w in &p.warnings {
        let _ = writeln!(s, "\nWARNING  {w}");
    }
    if !p.llamacpp_args.is_empty() {
        let _ = writeln!(s, "\nLLAMA.CPP\n  llama-server {}", p.llamacpp_args.iter().map(|a| if a.contains(' ') || a.contains('\\') { format!("'{a}'") } else { a.clone() }).collect::<Vec<_>>().join(" "));
    }
    s
}

pub fn infeasible_text(x: &Infeasible) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "Model cannot currently be executed: {}\n", x.model_name);
    if x.required_ram > 0 {
        let _ = writeln!(s, "Required (minimum, with NVMe streaming):");
        let _ = writeln!(s, "  RAM:  {}\n", fb(x.required_ram));
    }
    let _ = writeln!(s, "Available:");
    if x.available_vram > 0 {
        let _ = writeln!(s, "  VRAM: {} usable", fb(x.available_vram));
    }
    let _ = writeln!(s, "  RAM:  {}", fb(x.available_ram));
    let _ = writeln!(s, "  Disk: {} free", fb(x.available_disk));
    if !x.reasons.is_empty() {
        let _ = writeln!(s, "\nWhy:");
        for r in &x.reasons {
            let _ = writeln!(s, "  - {r}");
        }
    }
    let _ = writeln!(s, "\nPossible solutions:");
    for (i, r) in x.remedies.iter().enumerate() {
        let _ = writeln!(s, "  {}. {r}", i + 1);
    }
    s
}
