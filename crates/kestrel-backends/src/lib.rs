//! Backends turn an [`ExecutionPlan`] into something that runs.
//!
//! * [`NativeSession`]: Kestrel's own CPU executor. Every weight access goes
//!   through Kestrel's `WeightStore`, the RAM budget is enforced by the
//!   ledger, and the rebalancer adapts placement between tokens.
//! * [`llamacpp`]: translates the plan into llama.cpp placement parameters and
//!   launches `llama-server` (GPU execution via ggml).

pub mod llamacpp;

use anyhow::{Context, Result};
use kestrel_engine::{ChatMessage, Engine, GenParams, GenStats};
use kestrel_gguf::GgufFile;
use kestrel_hw::fmt_bytes;
use kestrel_memory::guard::{MemSample, RebalanceAction, Rebalancer};
use kestrel_memory::{ExpertPolicy, ExpertStore, Ledger, MemoryGuard, Tier, UsageFile, WeightStore};
use kestrel_model::ModelDesc;
use kestrel_planner::ExecutionPlan;
use serde::Serialize;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Live state of a session, readable while it generates (it never takes the
/// session lock). The flag asks for the per-expert map of MoE models.
pub type Monitor = Arc<dyn Fn(bool) -> serde_json::Value + Send + Sync>;

/// Turns kept for the dashboard's history.
const TURN_HISTORY: usize = 60;

/// One generation request, as the dashboard charts it.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Turn {
    /// Unix time (s) when the turn finished.
    pub at: u64,
    pub prompt_tokens: usize,
    pub prompt_evaluated: usize,
    pub generated: usize,
    pub ttft_s: f64,
    pub prefill_s: f64,
    pub decode_s: f64,
    pub prefill_tok_s: f64,
    pub decode_tok_s: f64,
    /// Time spent waiting for weights (streamed layers and experts).
    pub weight_wait_s: f64,
    /// Forward-pass phases (wall time, prefill and decode together).
    pub matmul_s: f64,
    pub attention_s: f64,
    pub other_s: f64,
    pub layer_hit_rate: f64,
    pub streamed_bytes: u64,
    pub expert_hit_rate: Option<f64>,
    pub expert_bytes: Option<u64>,
    pub rebalance_actions: usize,
    pub stop_reason: String,
}

/// A loaded model that can generate text.
pub trait InferenceSession: Send {
    fn model_name(&self) -> &str;
    fn render_chat(&self, messages: &[ChatMessage]) -> Result<String>;
    fn tokenize(&self, text: &str) -> Vec<u32>;
    fn generate(&mut self, prompt: &[u32], params: &GenParams, on_text: &mut dyn FnMut(&str) -> bool) -> Result<GenStats>;
    fn status(&self) -> serde_json::Value;
    /// Static description for dashboards: model, plan, placement.
    fn info(&self) -> serde_json::Value {
        serde_json::json!({ "model": { "name": self.model_name() } })
    }
    /// Live state that can be read during generation.
    fn monitor(&self) -> Option<Monitor> {
        None
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct SessionTotals {
    pub requests: u64,
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
    pub decode_s: f64,
    pub prefill_s: f64,
    pub last_decode_tok_s: f64,
    pub rebalance: Vec<RebalanceAction>,
}

pub struct NativeSession {
    pub engine: Engine,
    pub store: WeightStore,
    pub ledger: Arc<Ledger>,
    pub plan: ExecutionPlan,
    pub load_s: f64,
    rebalancer: Option<Rebalancer>,
    /// Tokens between rebalancer checks.
    pub rebalance_interval: usize,
    pub totals: SessionTotals,
    name: String,
    usage_history: bool,
    turns: Arc<Mutex<VecDeque<Turn>>>,
    busy: Arc<AtomicBool>,
    /// Tokens generated so far in the running turn.
    live_tokens: Arc<AtomicU64>,
}

pub struct NativeOptions {
    /// Adapt placement at runtime (promote/demote between RAM and NVMe).
    pub adaptive: bool,
    /// Expert-cache policy for MoE models.
    pub expert_policy: ExpertPolicy,
    /// Load and persist expert usage next to the model (warm starts).
    pub usage_history: bool,
    /// Promote streamed layers beyond the plan when memory frees up. Off by
    /// default: the plan is static, and the memory guard only demotes under
    /// pressure and restores what it demoted.
    pub promote_beyond_plan: bool,
}

impl Default for NativeOptions {
    fn default() -> Self {
        NativeOptions { adaptive: true, expert_policy: ExpertPolicy::Lfru, usage_history: true, promote_beyond_plan: false }
    }
}

impl NativeSession {
    pub fn load(plan: ExecutionPlan, gguf: &GgufFile, model: Arc<ModelDesc>, opts: NativeOptions) -> Result<Self> {
        let t0 = Instant::now();
        let b = &plan.budgets;
        // The ledger holds weights + ring + KV; runtime overhead is outside it.
        let ledger = Ledger::new(0, b.ram.usable, b.disk_free);
        let order: Vec<usize> = (0..model.groups.len()).collect();
        let store = WeightStore::new(model.clone(), &plan.resident_mask(), order, plan.store.clone(), ledger.clone())
            .with_context(|| format!("allocating weights under a {} RAM budget", fmt_bytes(b.ram.usable)))?;
        let experts = match (plan.chosen.expert_cache, &model.moe) {
            (Some(cap), Some(moe)) => {
                let mut es = ExpertStore::new(
                    model.clone(),
                    cap as usize,
                    2 * moe.n_expert_used as usize,
                    plan.store.io_mode,
                    plan.store.io_workers,
                    opts.expert_policy,
                    ledger.clone(),
                )?;
                if let Some(v) = std::env::var("KESTREL_SPEC_MIN_ACCURACY").ok().and_then(|v| v.parse().ok()) {
                    es.spec_min_accuracy = v;
                }
                if es.capacity >= es.n_experts_total() {
                    // Everything fits: experts are resident, load them now.
                    es.preload_all()?;
                } else if let Some(hist) = UsageFile::load(&model).filter(|_| opts.usage_history) {
                    let n = es.warm_start(&hist)?;
                    eprintln!("kestrel: warm start: {n} experts preloaded from usage history");
                }
                Some(Arc::new(es))
            }
            _ => None,
        };
        let engine = Engine::new(gguf, model.clone(), store.clone(), experts.clone(), plan.n_ctx as usize, plan.threads, &ledger)?;
        let rebalancer = opts.adaptive.then(|| {
            let mut r = Rebalancer::new(MemoryGuard {
                rss_limit: b.ram.usable + b.ram.overhead,
                min_available: b.ram.safety / 2,
                tolerance: 0.02,
            });
            // Restores layers demoted under pressure once memory frees up;
            // promotes beyond the plan only when asked.
            r.allow_promotion = true;
            r.beyond_plan = opts.promote_beyond_plan;
            r.pinned = model.groups.iter().filter(|g| g.layer.is_none()).map(|g| g.id).collect();
            r
        });
        Ok(NativeSession {
            name: model.name.clone(),
            engine,
            store,
            ledger,
            plan,
            load_s: t0.elapsed().as_secs_f64(),
            rebalancer,
            rebalance_interval: 16,
            totals: SessionTotals::default(),
            usage_history: opts.usage_history,
            turns: Arc::new(Mutex::new(VecDeque::new())),
            busy: Arc::new(AtomicBool::new(false)),
            live_tokens: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Tune the rebalancer: check every `interval` tokens, promote after
    /// `promote_after` calm checks, and allow `rss_limit` bytes of RSS.
    pub fn configure_rebalancer(&mut self, interval: usize, promote_after: u32, rss_limit: u64) {
        self.rebalance_interval = interval.max(1);
        if let Some(r) = self.rebalancer.as_mut() {
            r.promote_after = promote_after;
            r.guard.rss_limit = rss_limit;
            r.allow_promotion = true;
            r.beyond_plan = true;
        }
    }

    /// Time the forward-pass phases (matmul, attention, weight waits) on every
    /// turn. The cost is two clock reads per matrix product.
    pub fn enable_profile(&mut self) {
        if self.engine.tf.profile.is_none() {
            self.engine.tf.profile = Some(Default::default());
        }
    }

    pub fn rebalance_now(&mut self) -> Vec<RebalanceAction> {
        let (Some(r), Some(s)) = (self.rebalancer.as_mut(), MemSample::now()) else { return Vec::new() };
        let acts = r.tick(&self.store, s);
        self.totals.rebalance.extend(acts.iter().cloned());
        acts
    }
}

impl NativeSession {
    fn record_turn(&self, st: &GenStats, prof0: Option<kestrel_engine::Profile>, rebalance_actions: usize) {
        let m = st.memory.clone().unwrap_or_default();
        let (mut matmul_s, mut attention_s, mut other_s) = (0.0, 0.0, 0.0);
        if let (Some(a), Some(b)) = (prof0, self.engine.tf.profile.as_ref()) {
            matmul_s = b.matmul_s - a.matmul_s;
            attention_s = b.attention_s - a.attention_s;
            other_s = (b.other_s - a.other_s) + (b.head_s - a.head_s);
        }
        let t = Turn {
            at: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
            prompt_tokens: st.prompt_tokens,
            prompt_evaluated: st.prompt_evaluated,
            generated: st.generated,
            ttft_s: st.ttft_s,
            prefill_s: st.prefill_s,
            decode_s: st.decode_s,
            prefill_tok_s: st.prompt_tok_s,
            decode_tok_s: st.decode_tok_s,
            weight_wait_s: m.stall_s + st.experts.as_ref().map(|e| e.stall_s).unwrap_or(0.0),
            matmul_s,
            attention_s,
            other_s,
            layer_hit_rate: m.hit_rate,
            streamed_bytes: m.stream_bytes,
            expert_hit_rate: st.experts.as_ref().map(|e| e.hit_rate),
            expert_bytes: st.experts.as_ref().map(|e| e.bytes_read),
            rebalance_actions,
            stop_reason: st.stop_reason.clone(),
        };
        let mut turns = self.turns.lock().unwrap();
        if turns.len() == TURN_HISTORY {
            turns.pop_front();
        }
        turns.push_back(t);
    }
}

impl InferenceSession for NativeSession {
    fn model_name(&self) -> &str {
        &self.name
    }

    fn render_chat(&self, messages: &[ChatMessage]) -> Result<String> {
        Ok(self.engine.render_chat(messages)?)
    }

    fn tokenize(&self, text: &str) -> Vec<u32> {
        self.engine.tokenize_prompt(text)
    }

    fn generate(&mut self, prompt: &[u32], params: &GenParams, on_text: &mut dyn FnMut(&str) -> bool) -> Result<GenStats> {
        // Rebalance between tokens: the text callback runs at a safe point
        // (the forward pass has returned and released every lease).
        let interval = self.rebalance_interval.max(1);
        let store = self.store.clone();
        let mut rebal = self.rebalancer.take();
        let mut actions = Vec::new();
        let mut n = 0usize;
        let prof0 = self.engine.tf.profile.clone();
        self.busy.store(true, Ordering::Relaxed);
        self.live_tokens.store(0, Ordering::Relaxed);
        let live = self.live_tokens.clone();
        let res = self.engine.generate(prompt, params, |piece| {
            live.fetch_add(1, Ordering::Relaxed);
            n += 1;
            if n.is_multiple_of(interval) {
                if let (Some(r), Some(s)) = (rebal.as_mut(), MemSample::now()) {
                    actions.extend(r.tick(&store, s));
                }
            }
            on_text(piece)
        });
        self.rebalancer = rebal;
        self.busy.store(false, Ordering::Relaxed);
        let n_actions = actions.len();
        self.totals.rebalance.extend(actions);
        let stats = res?;
        self.record_turn(&stats, prof0, n_actions);
        if let (true, Some(ex)) = (self.usage_history, &self.engine.tf.experts) {
            let _ = UsageFile::save(&self.engine.tf.model, ex.usage());
            ex.decay();
        }
        let t = &mut self.totals;
        t.requests += 1;
        t.prompt_tokens += stats.prompt_tokens as u64;
        t.generated_tokens += stats.generated as u64;
        t.decode_s += stats.decode_s;
        t.prefill_s += stats.prefill_s;
        t.last_decode_tok_s = stats.decode_tok_s;
        Ok(stats)
    }

    fn info(&self) -> serde_json::Value {
        let m = self.store.model();
        let p = &self.plan;
        let groups: Vec<serde_json::Value> = m
            .groups
            .iter()
            .map(|g| serde_json::json!({ "label": g.label(), "kind": g.kind, "layer": g.layer, "bytes": g.bytes, "tier": p.chosen.tiers.get(g.id) }))
            .collect();
        serde_json::json!({
            "model": {
                "name": m.name, "arch": m.arch, "file": m.path.file_name().map(|f| f.to_string_lossy().to_string()),
                "file_size": m.file_size, "n_params": m.n_params, "quant": m.file_type, "n_layer": m.hparams.n_layer,
                "n_embd": m.hparams.n_embd, "n_vocab": m.hparams.n_vocab, "moe": m.moe, "prepared": m.prepared.is_some(),
            },
            "plan": {
                "strategy": p.chosen.kind.label(), "backend": "native", "n_ctx": p.n_ctx, "threads": p.threads,
                "ram_weights": p.chosen.ram_weights, "disk_weights": p.chosen.disk_weights, "kv_bytes": p.chosen.kv_bytes,
                "ring_bytes": p.chosen.ring_bytes, "expert_cache": p.chosen.expert_cache, "estimate": p.chosen.estimate,
                "budgets": p.budgets, "warnings": p.warnings, "io_mode": self.store.io_mode(), "prefetch_depth": p.store.prefetch_depth,
                "alternatives": p.alternatives.iter().map(|c| serde_json::json!({
                    "strategy": c.kind.label(), "backend": c.backend, "tok_s": c.estimate.tok_s, "feasible": c.feasible, "reason": c.reason,
                })).collect::<Vec<_>>(),
                "text": p.text(),
            },
            "groups": groups,
            "load_s": self.load_s,
        })
    }

    fn monitor(&self) -> Option<Monitor> {
        let (store, ledger, experts) = (self.store.clone(), self.ledger.clone(), self.engine.tf.experts.clone());
        let (turns, busy, live) = (self.turns.clone(), self.busy.clone(), self.live_tokens.clone());
        Some(Arc::new(move |with_experts: bool| {
            serde_json::json!({
                "busy": busy.load(Ordering::Relaxed),
                "live_tokens": live.load(Ordering::Relaxed),
                "rss": kestrel_hw::process_rss(),
                "ram": ledger.usage(Tier::Ram),
                "resident": store.resident_mask(),
                "ring": store.ring_contents(),
                "store": store.metrics(),
                "experts": experts.as_ref().map(|e| e.metrics()),
                "expert_map": if with_experts { experts.as_ref().map(|e| serde_json::to_value(e.map()).unwrap_or_default()) } else { None },
                "turns": turns.lock().unwrap().iter().cloned().collect::<Vec<_>>(),
            })
        }))
    }

    fn status(&self) -> serde_json::Value {
        let ram = self.ledger.usage(Tier::Ram);
        serde_json::json!({
            "model": self.name,
            "backend": "native",
            "strategy": self.plan.chosen.kind.label(),
            "n_ctx": self.plan.n_ctx,
            "load_s": self.load_s,
            "rss_bytes": kestrel_hw::process_rss(),
            "ram_budget": ram,
            "store": self.store.metrics(),
            "experts": self.engine.tf.experts.as_ref().map(|e| e.metrics()),
            "streamed_bytes_per_token": self.store.streamed_bytes_per_pass(),
            "io_mode": self.store.io_mode(),
            "totals": self.totals,
        })
    }
}
