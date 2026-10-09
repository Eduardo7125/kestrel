//! `kestrel setup`: from nothing to a running model in one step.
//!
//! detect the machine → list the catalog against it → recommend a model →
//! download it (resumable, SHA-256 verified) → plan it → remember it as the
//! default model. Every step is idempotent: running setup again resumes a
//! download, and once a model is configured it goes straight to starting it.

use crate::common::{self, gb};
use crate::download;
use anyhow::{bail, Context, Result};
use kestrel_hw::HardwareProfile;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, IsTerminal, Write};
use std::path::PathBuf;

/// A model Kestrel's native executor runs, in a single-file GGUF.
pub struct CatalogEntry {
    pub id: &'static str,
    pub name: &'static str,
    pub repo: &'static str,
    pub params: &'static str,
    /// Approximate Q4_K_M download size, GB. The exact size comes from the Hub.
    pub size_gb: f64,
    /// Weight bytes read per generated token, GB (smaller than the size for MoE).
    pub active_gb: f64,
    /// RAM needed when routed experts stream from disk (MoE only), GB.
    pub stream_ram_gb: Option<f64>,
    pub note: &'static str,
}

/// Ordered from smallest to most capable. Only the repository names are
/// fixed: file names, sizes and hashes are read from the Hub at download time.
pub const CATALOG: &[CatalogEntry] = &[
    CatalogEntry { id: "qwen2.5-0.5b", name: "Qwen2.5 0.5B Instruct", repo: "bartowski/Qwen2.5-0.5B-Instruct-GGUF", params: "0.5B", size_gb: 0.4, active_gb: 0.4, stream_ram_gb: None, note: "tiny; for testing" },
    CatalogEntry { id: "qwen2.5-1.5b", name: "Qwen2.5 1.5B Instruct", repo: "bartowski/Qwen2.5-1.5B-Instruct-GGUF", params: "1.5B", size_gb: 1.1, active_gb: 1.1, stream_ram_gb: None, note: "small and fast" },
    CatalogEntry { id: "llama3.2-3b", name: "Llama 3.2 3B Instruct", repo: "bartowski/Llama-3.2-3B-Instruct-GGUF", params: "3B", size_gb: 2.0, active_gb: 2.0, stream_ram_gb: None, note: "" },
    CatalogEntry { id: "qwen2.5-3b", name: "Qwen2.5 3B Instruct", repo: "bartowski/Qwen2.5-3B-Instruct-GGUF", params: "3B", size_gb: 1.9, active_gb: 1.9, stream_ram_gb: None, note: "" },
    CatalogEntry { id: "qwen2.5-7b", name: "Qwen2.5 7B Instruct", repo: "bartowski/Qwen2.5-7B-Instruct-GGUF", params: "7.6B", size_gb: 4.7, active_gb: 4.7, stream_ram_gb: None, note: "good all-rounder" },
    CatalogEntry { id: "llama3.1-8b", name: "Llama 3.1 8B Instruct", repo: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", params: "8B", size_gb: 4.9, active_gb: 4.9, stream_ram_gb: None, note: "" },
    CatalogEntry { id: "qwen3-8b", name: "Qwen3 8B", repo: "bartowski/Qwen_Qwen3-8B-GGUF", params: "8.2B", size_gb: 5.0, active_gb: 5.0, stream_ram_gb: None, note: "thinks before answering" },
    CatalogEntry { id: "qwen2.5-14b", name: "Qwen2.5 14B Instruct", repo: "bartowski/Qwen2.5-14B-Instruct-GGUF", params: "14.8B", size_gb: 9.0, active_gb: 9.0, stream_ram_gb: None, note: "" },
    CatalogEntry { id: "qwen3-30b-a3b", name: "Qwen3 30B-A3B (MoE)", repo: "bartowski/Qwen_Qwen3-30B-A3B-GGUF", params: "30.5B, 3.3B active", size_gb: 18.6, active_gb: 2.0, stream_ram_gb: Some(6.0), note: "MoE: fast for its size, experts can stream from disk" },
    CatalogEntry { id: "qwen2.5-32b", name: "Qwen2.5 32B Instruct", repo: "bartowski/Qwen2.5-32B-Instruct-GGUF", params: "32.8B", size_gb: 19.9, active_gb: 19.9, stream_ram_gb: None, note: "most capable; slow on CPU" },
];

pub const DEFAULT_QUANT: &str = "Q4_K_M";
/// Decode speed below which a model is not recommended (tokens/s).
const COMFORTABLE_TOK_S: f64 = 4.0;

/// What `setup` remembers, in `$KESTREL_CACHE/setup.json`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SetupConfig {
    pub model: PathBuf,
    #[serde(default)]
    pub catalog_id: Option<String>,
    #[serde(default)]
    pub repo: Option<String>,
    #[serde(default = "default_port")]
    pub port: u16,
}

fn default_port() -> u16 {
    8080
}

impl SetupConfig {
    pub fn path() -> PathBuf {
        kestrel_hw::cache_dir().join("setup.json")
    }
    pub fn load() -> Option<Self> {
        serde_json::from_str(&std::fs::read_to_string(Self::path()).ok()?).ok()
    }
    pub fn save(&self) -> Result<()> {
        let p = Self::path();
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d)?;
        }
        std::fs::write(&p, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }
}

/// The model argument of `run`/`serve`/`plan`/`inspect`, or the one `setup`
/// configured.
pub fn model_or_default(arg: Option<String>) -> Result<String> {
    if let Some(a) = arg {
        return Ok(a);
    }
    match SetupConfig::load() {
        Some(c) if c.model.is_file() => Ok(c.model.to_string_lossy().into_owned()),
        Some(c) => bail!("the configured model {} is missing; run `kestrel setup` again", c.model.display()),
        None => bail!("no model yet: run ./start-here.sh (or `kestrel setup`), or pass a model file"),
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Fit {
    Ram,
    Streaming,
    No,
}

struct Assessed<'a> {
    e: &'a CatalogEntry,
    fit: Fit,
    tok_s: f64,
    why: String,
}

const GB: f64 = 1e9;

fn usable_ram(hw: &HardwareProfile) -> f64 {
    let total = hw.ram.total as f64;
    let safety = (1.5 * GB).max(total / 10.0);
    (hw.ram.effective_available() as f64 - safety).max(0.0)
}

/// Bytes per second the native decode can stream through weights.
fn decode_bw(hw: &HardwareProfile) -> f64 {
    let m = hw.measured.as_ref();
    let ram = m.map(|m| m.ram_read_bw).filter(|&b| b > 0.0).unwrap_or(10.0 * GB) * 0.7;
    let kern = m.and_then(|m| m.cpu_gemv_bytes_per_s.get("Q4_K").copied()).filter(|&b| b > 0.0).unwrap_or(ram);
    ram.min(kern)
}

fn assess<'a>(e: &'a CatalogEntry, hw: &HardwareProfile, disk_free: f64) -> Assessed<'a> {
    let ram = usable_ram(hw);
    let need = e.size_gb * 1.1 + 0.6; // weights + KV at 4K context + runtime
    let tok_s = decode_bw(hw) / (e.active_gb * GB);
    let (fit, why) = if disk_free < e.size_gb * GB + 1.0 * GB {
        (Fit::No, format!("needs {:.0} GB free disk", e.size_gb + 1.0))
    } else if need * GB <= ram {
        (Fit::Ram, "fits in RAM".into())
    } else if let Some(s) = e.stream_ram_gb.filter(|&s| s * GB <= ram) {
        let _ = s;
        (Fit::Streaming, "experts stream from disk".into())
    } else {
        (Fit::No, format!("needs ~{:.0} GB free RAM", e.stream_ram_gb.unwrap_or(need).ceil()))
    };
    // Streaming MoE reads part of the experts from disk: count it as slower.
    let tok_s = if fit == Fit::Streaming { tok_s * 0.5 } else { tok_s };
    Assessed { e, fit, tok_s, why }
}

/// The most capable model that runs comfortably here, else the fastest that fits.
fn recommend(list: &[Assessed]) -> Option<usize> {
    let fits = |a: &&Assessed| a.fit != Fit::No;
    list.iter().enumerate().rev().find(|(_, a)| fits(a) && a.tok_s >= COMFORTABLE_TOK_S).map(|(i, _)| i).or_else(|| list.iter().position(|a| fits(&a)))
}

fn machine_line(hw: &HardwareProfile, disk_free: f64) -> String {
    let simd = if hw.cpu.simd.iter().any(|s| s == "avx2") { " (AVX2)" } else { "" };
    let gpu = match hw.gpus.first() {
        Some(g) => format!("{} ({})", g.name, gb(g.vram_total)),
        None => "no GPU".into(),
    };
    format!(
        "{} cores{simd} · {} RAM ({} free) · {} free disk · {gpu}",
        hw.cpu.physical_cores,
        gb(hw.ram.total),
        gb(hw.ram.effective_available()),
        gb(disk_free as u64)
    )
}

fn print_list(list: &[Assessed], rec: Option<usize>) {
    println!("\n   #  {:<24} {:<18} {:>8}  {:>11}", "MODEL", "PARAMETERS", "SIZE", "SPEED HERE");
    for (i, a) in list.iter().enumerate() {
        let speed = if a.fit == Fit::No { "—".to_string() } else { format!("~{:.0} tok/s", a.tok_s.max(0.5)) };
        let mark = if Some(i) == rec { "  ← recommended" } else { "" };
        let note = if a.e.note.is_empty() || a.fit == Fit::No { String::new() } else { format!(" · {}", a.e.note) };
        println!("  {:>2}  {:<24} {:<18} {:>5.1} GB  {:>11}  {}{note}{mark}", i + 1, a.e.name, a.e.params, a.e.size_gb, speed, a.why);
    }
    println!("\n  Speeds are estimates from this machine's measured memory bandwidth (CPU, native executor).");
}

fn ask(prompt: &str) -> Result<String> {
    eprint!("{prompt}");
    std::io::stderr().flush()?;
    let mut s = String::new();
    std::io::stdin().lock().read_line(&mut s)?;
    Ok(s.trim().to_string())
}

pub struct SetupArgs {
    pub model: Option<String>,
    pub repo: Option<String>,
    pub quant: String,
    pub model_file: Option<PathBuf>,
    pub dir: Option<PathBuf>,
    pub list: bool,
    pub yes: bool,
    pub reconfigure: bool,
    pub port: u16,
}

/// Run the setup. Returns the configured model path, or None for `--list`.
pub fn run(a: SetupArgs) -> Result<Option<PathBuf>> {
    let interactive = !a.yes && std::io::stdin().is_terminal();

    // Already set up: start straight away.
    if !a.reconfigure && !a.list && a.model.is_none() && a.repo.is_none() && a.model_file.is_none() {
        if let Some(c) = SetupConfig::load().filter(|c| c.model.is_file()) {
            eprintln!("kestrel: using {} (run `kestrel setup --reconfigure` to choose another model)", c.model.display());
            return Ok(Some(c.model));
        }
    }

    // A model already on disk.
    if let Some(p) = a.model_file {
        let p = std::fs::canonicalize(&p).with_context(|| format!("{} not found", p.display()))?;
        return finish(p, None, None, a.port).map(Some);
    }

    let dir = a.dir.clone().unwrap_or_else(|| kestrel_hw::cache_dir().join("models"));
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    eprintln!("Kestrel setup\n\nLooking at this machine…");
    let hw = common::hardware(None, false, true);
    let disk_free = kestrel_hw::describe_storage(&dir).map(|s| s.available as f64).unwrap_or(f64::MAX);
    println!("  machine   {}", machine_line(&hw, disk_free));
    println!("  models in {}", dir.display());

    let (repo, catalog_id) = match (&a.repo, &a.model) {
        (Some(r), _) => (r.clone(), None),
        (None, Some(id)) => {
            let e = CATALOG.iter().find(|e| e.id.eq_ignore_ascii_case(id)).with_context(|| {
                format!("unknown model '{id}'. Known: {}. Or use --repo <owner/name> for any GGUF repository", CATALOG.iter().map(|e| e.id).collect::<Vec<_>>().join(", "))
            })?;
            (e.repo.to_string(), Some(e.id.to_string()))
        }
        (None, None) => {
            let list: Vec<Assessed> = CATALOG.iter().map(|e| assess(e, &hw, disk_free)).collect();
            let rec = recommend(&list);
            print_list(&list, rec);
            if a.list {
                return Ok(None);
            }
            let Some(rec) = rec else {
                bail!("no model in the catalog fits this machine (see the list above); free some RAM or disk, or use --model-file with a smaller GGUF");
            };
            let choice = if interactive {
                let s = ask(&format!("\nWhich model? [Enter = {}] ", rec + 1))?;
                if s.is_empty() {
                    rec
                } else {
                    let n: usize = s.parse().ok().filter(|n| (1..=list.len()).contains(n)).with_context(|| format!("'{s}' is not a number from the list"))?;
                    if list[n - 1].fit == Fit::No {
                        bail!("{} does not fit this machine: {}", list[n - 1].e.name, list[n - 1].why);
                    }
                    n - 1
                }
            } else {
                eprintln!("\nTaking the recommendation: {}", list[rec].e.name);
                rec
            };
            (list[choice].e.repo.to_string(), Some(list[choice].e.id.to_string()))
        }
    };

    eprintln!("\nFinding {} in {repo}…", a.quant);
    let files = download::list_gguf(&repo)?;
    let Some(file) = download::pick_quant(&files, &a.quant) else {
        let have: Vec<&str> = files.iter().map(|f| f.path.as_str()).collect();
        bail!("{repo} has no single-file {} GGUF. Files: {}", a.quant, if have.is_empty() { "none".into() } else { have.join(", ") });
    };
    let dest = dir.join(&file.path);
    if disk_free < file.size as f64 && !dest.is_file() {
        bail!("{} needs {} free on the disk; {} is free. Use --dir with a folder on a bigger disk", file.path, gb(file.size), gb(disk_free as u64));
    }
    if interactive && !dest.is_file() {
        let s = ask(&format!("Download {} ({})? [Y/n] ", file.path, gb(file.size)))?;
        if s.eq_ignore_ascii_case("n") || s.eq_ignore_ascii_case("no") {
            bail!("cancelled");
        }
    }
    eprintln!("Downloading {} ({}) — stop any time, run again to resume", file.path, gb(file.size));
    let t0 = std::time::Instant::now();
    let res = download::download(file, &dest, |done, total| {
        let rate = done as f64 / t0.elapsed().as_secs_f64().max(0.1);
        eprint!("\r  {:>5.1}%  {} / {}  {}/s     ", 100.0 * done as f64 / total.max(1) as f64, gb(done), gb(total), gb(rate as u64));
    });
    if res.is_err() {
        eprintln!();
    }
    res?;
    eprintln!("\n  ✓ downloaded{}", if file.sha256.is_some() { " and verified (SHA-256)" } else { "" });
    finish(dest, catalog_id, Some(repo), a.port).map(Some)
}

/// Plan the model once (so problems show now, not at first use) and save it
/// as the default.
fn finish(model: PathBuf, catalog_id: Option<String>, repo: Option<String>, port: u16) -> Result<PathBuf> {
    let (_, m) = common::open_model(&model.to_string_lossy())?;
    if let Err(e) = kestrel_engine::check_support(&m) {
        eprintln!("  note: the native executor cannot run this model ({e}); it needs the llama.cpp backend");
    }
    let hw = common::hardware(Some(&m), false, true);
    let o = common::Overrides::default_for_setup();
    match kestrel_planner::plan(&m, &hw, &o.request_with(&m, &hw)?) {
        Ok(p) => println!(
            "  plan      {} · {} in RAM{} · ~{:.1} tok/s estimated · context {}",
            p.chosen.kind.label(),
            gb(p.chosen.ram_weights),
            if p.chosen.disk_weights > 0 { format!(", {} streamed from disk", gb(p.chosen.disk_weights)) } else { String::new() },
            p.chosen.estimate.tok_s,
            p.n_ctx
        ),
        Err(inf) => {
            eprint!("{}", inf.text());
            bail!("the model was downloaded but cannot run here right now");
        }
    }
    SetupConfig { model: model.clone(), catalog_id, repo, port }.save()?;
    println!("  saved     as the default model ({})", SetupConfig::path().display());
    Ok(model)
}

/// `kestrel status`: the configured model and whether a server answers.
pub fn status() -> Result<()> {
    let Some(c) = SetupConfig::load() else {
        println!("not set up: run ./start-here.sh (or `kestrel setup`)");
        return Ok(());
    };
    let present = c.model.is_file();
    println!("model     {}{}", c.model.display(), if present { "" } else { "  (missing: run `kestrel setup`)" });
    if present {
        if let Ok(md) = std::fs::metadata(&c.model) {
            println!("size      {}", gb(md.len()));
        }
    }
    let url = format!("http://127.0.0.1:{}", c.port);
    let up = ureq::AgentBuilder::new().timeout(std::time::Duration::from_secs(2)).build().get(&format!("{url}/health")).call().is_ok();
    println!("server    {}", if up { format!("running · OpenAI base URL {url}/v1") } else { format!("not running (start it with `kestrel serve`; it will listen on {url})") });
    println!("chat      kestrel chat");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hw(ram_gb: f64, bw_gb: f64) -> HardwareProfile {
        let mut h = HardwareProfile::discover(&[]);
        h.ram.total = (ram_gb * GB) as u64;
        h.ram.available = (ram_gb * GB * 0.8) as u64;
        h.ram.cgroup_limit = None;
        let mut m = h.measured.clone().unwrap_or_default();
        m.ram_read_bw = bw_gb * GB / 0.7;
        m.cpu_gemv_bytes_per_s.insert("Q4_K".into(), bw_gb * GB);
        h.measured = Some(m);
        h
    }

    #[test]
    fn recommends_the_most_capable_comfortable_model() {
        // 32 GB, 30 GB/s: the 7-8B class runs at ~6 tok/s; the MoE is faster still.
        let h = hw(32.0, 30.0);
        let list: Vec<Assessed> = CATALOG.iter().map(|e| assess(e, &h, 500.0 * GB)).collect();
        let r = recommend(&list).unwrap();
        assert_eq!(list[r].e.id, "qwen3-30b-a3b");
        // 8 GB laptop: the MoE does not fit fully but streams its experts.
        let h = hw(8.0, 20.0);
        let list: Vec<Assessed> = CATALOG.iter().map(|e| assess(e, &h, 500.0 * GB)).collect();
        let moe = list.iter().find(|a| a.e.id == "qwen3-30b-a3b").unwrap();
        assert!(moe.fit != Fit::Ram);
        let r = recommend(&list).unwrap();
        assert!(list[r].fit != Fit::No && list[r].tok_s >= COMFORTABLE_TOK_S, "{}", list[r].e.id);
    }

    #[test]
    fn small_disk_rules_models_out() {
        let h = hw(64.0, 30.0);
        let list: Vec<Assessed> = CATALOG.iter().map(|e| assess(e, &h, 3.0 * GB)).collect();
        // 1 GB of headroom beyond the download is required.
        assert!(list.iter().filter(|a| a.fit != Fit::No).all(|a| a.e.size_gb + 1.0 <= 3.0));
        assert!(list.iter().any(|a| a.fit == Fit::No));
    }
}
