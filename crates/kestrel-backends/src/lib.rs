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
use std::sync::Arc;
use std::time::Instant;

/// A loaded model that can generate text.
pub trait InferenceSession: Send {
    fn model_name(&self) -> &str;
    fn render_chat(&self, messages: &[ChatMessage]) -> Result<String>;
    fn tokenize(&self, text: &str) -> Vec<u32>;
    fn generate(&mut self, prompt: &[u32], params: &GenParams, on_text: &mut dyn FnMut(&str) -> bool) -> Result<GenStats>;
    fn status(&self) -> serde_json::Value;
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
}

pub struct NativeOptions {
    /// Adapt placement at runtime (promote/demote between RAM and NVMe).
    pub adaptive: bool,
    /// Expert-cache policy for MoE models.
    pub expert_policy: ExpertPolicy,
    /// Load and persist expert usage next to the model (warm starts).
    pub usage_history: bool,
}

impl Default for NativeOptions {
    fn default() -> Self {
        NativeOptions { adaptive: true, expert_policy: ExpertPolicy::Lfru, usage_history: true }
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
            // Promotion is a no-op while nothing is streamed; after a demotion
            // under pressure it lets a resident plan recover once memory frees up.
            r.allow_promotion = true;
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
        })
    }

    pub fn rebalance_now(&mut self) -> Vec<RebalanceAction> {
        let (Some(r), Some(s)) = (self.rebalancer.as_mut(), MemSample::now()) else { return Vec::new() };
        let acts = r.tick(&self.store, s);
        self.totals.rebalance.extend(acts.iter().cloned());
        acts
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
        let res = self.engine.generate(prompt, params, |piece| {
            n += 1;
            if n.is_multiple_of(interval) {
                if let (Some(r), Some(s)) = (rebal.as_mut(), MemSample::now()) {
                    actions.extend(r.tick(&store, s));
                }
            }
            on_text(piece)
        });
        self.rebalancer = rebal;
        self.totals.rebalance.extend(actions);
        let stats = res?;
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
