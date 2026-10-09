//! The dashboard's Settings page, seen from the CLI: the settings it edits,
//! how they map onto [`Overrides`], and the load / plan / tune hooks the
//! server calls. Also the runtime file that lets `kestrel stop` and
//! `kestrel status` find a running server.

use crate::common::{self, ExpertPolicyArg, IoArg, Overrides};
use anyhow::{bail, Context, Result};
use kestrel_backends::{InferenceSession, NativeSession};
use kestrel_planner::Backend;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::BufRead;
use std::path::PathBuf;
use std::sync::Arc;

/// What the Settings page can change. `None` means automatic.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WebSettings {
    pub threads: Option<usize>,
    pub ram_budget_gb: Option<f64>,
    pub ctx: Option<u64>,
    pub prefetch_depth: Option<usize>,
    /// "direct" or "buffered".
    pub io: Option<String>,
    pub io_workers: Option<usize>,
    /// "lfru" or "lru".
    pub expert_policy: Option<String>,
    /// Router-lookahead expert prefetch (MoE). Default on.
    pub lookahead: Option<bool>,
    /// Memory guard: demote under pressure, restore after. Default on.
    pub guard: Option<bool>,
    /// Promote streamed layers beyond the plan. Default off.
    pub adapt: Option<bool>,
    /// Allow streaming weights from disk. Default on.
    pub stream: Option<bool>,
    /// Fill automatic knobs from a saved autotune profile. Default on.
    pub tune_profile: Option<bool>,
}

impl WebSettings {
    /// The settings a command line corresponds to.
    pub fn from_overrides(o: &Overrides) -> Self {
        WebSettings {
            threads: o.threads,
            ram_budget_gb: o.ram_budget.map(|b| b as f64 / 1e9),
            ctx: o.ctx,
            prefetch_depth: o.prefetch_depth,
            io: o.io.map(|m| match m {
                IoArg::Direct => "direct".into(),
                IoArg::Buffered => "buffered".into(),
            }),
            io_workers: o.io_workers,
            expert_policy: Some(match o.expert_policy {
                ExpertPolicyArg::Lfru => "lfru".into(),
                ExpertPolicyArg::Lru => "lru".into(),
            }),
            lookahead: Some(std::env::var("KESTREL_LOOKAHEAD").map(|v| v != "0").unwrap_or(true)),
            guard: Some(!o.no_adapt),
            adapt: Some(o.adapt),
            stream: Some(!o.no_stream),
            tune_profile: Some(!o.no_tune_profile),
        }
    }

    /// `base` with every knob this page controls replaced (auto → unset).
    pub fn apply(&self, base: &Overrides) -> Result<Overrides> {
        let mut o = base.clone();
        o.threads = self.threads.filter(|&t| t > 0);
        o.ram_budget = match self.ram_budget_gb {
            Some(g) if g.is_finite() && g > 0.0 => Some((g * 1e9) as u64),
            Some(_) => bail!("the RAM budget must be a positive number of GB"),
            None => None,
        };
        o.ctx = self.ctx.filter(|&c| c > 0);
        o.prefetch_depth = self.prefetch_depth;
        o.io = match self.io.as_deref() {
            None | Some("") | Some("auto") => None,
            Some("direct") => Some(IoArg::Direct),
            Some("buffered") => Some(IoArg::Buffered),
            Some(x) => bail!("unknown I/O mode '{x}' (direct or buffered)"),
        };
        o.io_workers = self.io_workers.filter(|&w| w > 0);
        o.expert_policy = match self.expert_policy.as_deref() {
            None | Some("lfru") => ExpertPolicyArg::Lfru,
            Some("lru") => ExpertPolicyArg::Lru,
            Some(x) => bail!("unknown expert policy '{x}' (lfru or lru)"),
        };
        o.no_adapt = !self.guard.unwrap_or(true);
        o.adapt = self.adapt.unwrap_or(false) && !o.no_adapt;
        o.no_stream = !self.stream.unwrap_or(true);
        o.no_tune_profile = !self.tune_profile.unwrap_or(true);
        // The first-run hardware benchmark already ran when the server started.
        o.no_bench = true;
        Ok(o)
    }
}

fn parse(v: &Value) -> Result<WebSettings> {
    if v.is_null() {
        return Ok(WebSettings::default());
    }
    serde_json::from_value(v.clone()).context("reading the settings")
}

/// Load a native session with these settings, profiled for the dashboard.
pub fn load(model: &str, o: &Overrides, s: &WebSettings) -> Result<NativeSession> {
    let (g, m, p) = crate::make_plan(model, o, true)?;
    if p.chosen.backend != Backend::Native {
        bail!("the plan chose llama.cpp; the dashboard manages the native backend only (set the backend to native)");
    }
    let mut sess = NativeSession::load(p, &g, m, o.native_options())?;
    sess.enable_profile();
    sess.engine.tf.lookahead = s.lookahead.unwrap_or(true);
    Ok(sess)
}

/// Plan for these settings without loading, as the Settings page shows it.
pub fn plan(model: &str, o: &Overrides) -> Result<Value> {
    let (_, m) = common::open_model(model)?;
    let hw = common::hardware(Some(&m), true, true);
    let req = o.request_with(&m, &hw)?;
    let p = match kestrel_planner::plan(&m, &hw, &req) {
        Ok(p) => p,
        Err(inf) => bail!("{}", inf.text()),
    };
    Ok(json!({
        "strategy": p.chosen.kind.label(),
        "backend": p.chosen.backend,
        "tok_s": p.chosen.estimate.tok_s,
        "bottleneck": p.chosen.estimate.bottleneck,
        "ram_weights": p.chosen.ram_weights,
        "disk_weights": p.chosen.disk_weights,
        "kv_bytes": p.chosen.kv_bytes,
        "expert_cache": p.chosen.expert_cache,
        "ram_usable": p.budgets.ram.usable,
        "ram_available": p.budgets.ram.available,
        "n_ctx": p.n_ctx,
        "threads": p.threads,
        "prefetch_depth": p.store.prefetch_depth,
        "io_workers": p.store.io_workers,
        "io_mode": p.store.io_mode,
        "warnings": p.warnings,
        "text": p.text(),
    }))
}

/// Run `kestrel benchmark <model> --tune` as a child process, passing its
/// output on line by line. Returns its conclusion (the last line it printed).
pub fn tune(model: &str, on_line: &mut dyn FnMut(String)) -> Result<String> {
    let exe = std::env::current_exe()?;
    let mut child = std::process::Command::new(exe)
        .args(["benchmark", model, "--tune"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .context("starting the autotuner")?;
    let (tx, rx) = std::sync::mpsc::channel::<(bool, String)>();
    let readers: Vec<_> = [(true, child.stdout.take().map(|r| Box::new(r) as Box<dyn std::io::Read + Send>)), (false, child.stderr.take().map(|r| Box::new(r) as Box<dyn std::io::Read + Send>))]
        .into_iter()
        .filter_map(|(out, r)| r.map(|r| (out, r)))
        .map(|(out, r)| {
            let tx = tx.clone();
            std::thread::spawn(move || {
                for line in std::io::BufReader::new(r).lines().map_while(Result::ok) {
                    let _ = tx.send((out, line));
                }
            })
        })
        .collect();
    drop(tx);
    let (mut last_out, mut last_err) = (String::new(), String::new());
    for (out, line) in rx {
        let line = line.trim_end().to_string();
        if line.is_empty() {
            continue;
        }
        if out {
            last_out = line.clone();
        } else {
            last_err = line.clone();
        }
        on_line(line);
    }
    for r in readers {
        let _ = r.join();
    }
    let status = child.wait()?;
    if !status.success() {
        bail!("{}", if last_err.is_empty() { format!("the autotuner exited with {status}") } else { last_err });
    }
    Ok(if last_out.is_empty() { last_err } else { last_out })
}

/// Everything the server needs to manage `model` from the dashboard.
pub fn control(model: String, base: Overrides) -> kestrel_server::Control {
    let initial = WebSettings::from_overrides(&base);
    let (m1, b1) = (model.clone(), base.clone());
    let (m2, b2) = (model.clone(), base.clone());
    kestrel_server::Control {
        load: Arc::new(move |v: &Value| {
            let s = parse(v)?;
            let o = s.apply(&b1)?;
            Ok(Box::new(load(&m1, &o, &s)?) as Box<dyn InferenceSession>)
        }),
        plan: Arc::new(move |v: &Value| plan(&m2, &parse(v)?.apply(&b2)?)),
        tune: Arc::new(move |on_line: &mut dyn FnMut(String)| tune(&model, on_line)),
        settings: serde_json::to_value(initial).unwrap_or_default(),
    }
}

/// Where a running server leaves its address, for `kestrel stop`/`status`.
#[derive(Debug, Serialize, Deserialize)]
pub struct ServerFile {
    pub pid: u32,
    pub url: String,
    pub model: String,
}

impl ServerFile {
    pub fn path() -> PathBuf {
        kestrel_hw::cache_dir().join("server.json")
    }
    pub fn write(url: &str, model: &str) {
        let f = ServerFile { pid: std::process::id(), url: url.to_string(), model: model.to_string() };
        let _ = std::fs::create_dir_all(kestrel_hw::cache_dir());
        let _ = std::fs::write(Self::path(), serde_json::to_string_pretty(&f).unwrap_or_default());
    }
    pub fn read() -> Option<Self> {
        serde_json::from_str(&std::fs::read_to_string(Self::path()).ok()?).ok()
    }
    /// Remove the file if it is ours.
    pub fn remove_own() {
        if Self::read().is_some_and(|f| f.pid == std::process::id()) {
            let _ = std::fs::remove_file(Self::path());
        }
    }
}

fn agent() -> ureq::Agent {
    // Local requests never go through a proxy.
    ureq::AgentBuilder::new().timeout(std::time::Duration::from_secs(5)).build()
}

/// The URL of a running Kestrel server, if one answers.
pub fn running_url() -> Option<String> {
    let mut candidates = Vec::new();
    if let Some(f) = ServerFile::read() {
        candidates.push(f.url);
    }
    if let Some(c) = crate::setup::SetupConfig::load() {
        candidates.push(format!("http://127.0.0.1:{}", c.port));
    }
    candidates.push("http://127.0.0.1:8080".into());
    candidates.dedup();
    candidates.into_iter().find(|u| agent().get(&format!("{u}/health")).call().is_ok())
}

/// `kestrel stop`: ask the running server to shut down and free its memory.
pub fn stop() -> Result<()> {
    let Some(url) = running_url() else {
        println!("Kestrel is not running.");
        let _ = std::fs::remove_file(ServerFile::path());
        return Ok(());
    };
    agent()
        .post(&format!("{url}/api/shutdown"))
        .set("Content-Type", "application/json")
        .send_string("{}")
        .map_err(|e| anyhow::anyhow!("asking {url} to stop: {e}"))?;
    // Wait until it is gone, so the memory is really free when we return.
    for _ in 0..50 {
        std::thread::sleep(std::time::Duration::from_millis(200));
        if agent().get(&format!("{url}/health")).call().is_err() {
            println!("Kestrel stopped ({url}); its memory is free.");
            return Ok(());
        }
    }
    bail!("{url} did not stop within 10 s; close its window or end kestrel in the task manager")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_round_trip_and_auto() {
        let base = Overrides::default_for_setup();
        let s = WebSettings::from_overrides(&base);
        let o = s.apply(&base).unwrap();
        assert_eq!(o.threads, None);
        assert!(!o.no_adapt && !o.adapt && !o.no_stream && !o.no_tune_profile);
        let s = WebSettings { threads: Some(8), ram_budget_gb: Some(9.5), io: Some("buffered".into()), guard: Some(false), adapt: Some(true), stream: Some(false), ..Default::default() };
        let o = s.apply(&base).unwrap();
        assert_eq!(o.threads, Some(8));
        assert_eq!(o.ram_budget, Some(9_500_000_000));
        assert!(matches!(o.io, Some(IoArg::Buffered)));
        // Promotion needs the guard: with the guard off it stays off.
        assert!(o.no_adapt && !o.adapt && o.no_stream);
        assert!(WebSettings { ram_budget_gb: Some(-1.0), ..Default::default() }.apply(&base).is_err());
        assert!(WebSettings { io: Some("mmap".into()), ..Default::default() }.apply(&base).is_err());
    }
}
