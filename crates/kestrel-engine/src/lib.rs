//! Kestrel's native CPU executor.
//!
//! It exists so that *every* weight access goes through Kestrel's
//! [`WeightStore`](kestrel_memory::WeightStore), which is what lets streaming,
//! prefetch and rebalancing be implemented and measured truthfully. It is not
//! a performance competitor to llama.cpp's hand-tuned CPU kernels; benchmarks
//! report both side by side.

#[cfg(target_arch = "x86_64")]
mod avx2;
pub mod chat;
pub mod qdot;
pub mod quant;
pub mod sampler;
pub mod tokenizer;
mod transformer;

pub use chat::{ChatMessage, ChatTemplate};
pub use sampler::{Sampler, SamplerConfig};
pub use tokenizer::{Tokenizer, Utf8Stream};
pub use transformer::{check_support, EngineError, Profile, Transformer};

use kestrel_gguf::GgufFile;
use kestrel_memory::{Ledger, MetricsSnapshot, WeightStore};
use kestrel_model::ModelDesc;
use serde::Serialize;
use std::sync::Arc;
use std::time::Instant;

pub struct Engine {
    pub tf: Transformer,
    pub tokenizer: Tokenizer,
    pub chat: ChatTemplate,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct GenStats {
    pub prompt_tokens: usize,
    /// Prompt tokens actually evaluated (excludes reused KV prefix).
    pub prompt_evaluated: usize,
    pub generated: usize,
    pub prefill_s: f64,
    pub decode_s: f64,
    pub ttft_s: f64,
    pub prompt_tok_s: f64,
    pub decode_tok_s: f64,
    pub stop_reason: String,
    pub memory: Option<MetricsSnapshot>,
    pub experts: Option<kestrel_memory::ExpertMetrics>,
    /// Generated token ids (for output-equivalence checks).
    #[serde(skip)]
    pub tokens: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct GenParams {
    pub max_tokens: usize,
    pub sampler: SamplerConfig,
    pub stop: Vec<String>,
    /// Keep generating past end-of-generation tokens (benchmarks).
    pub ignore_eos: bool,
}

impl Default for GenParams {
    fn default() -> Self {
        GenParams { max_tokens: 256, sampler: SamplerConfig::default(), stop: Vec::new(), ignore_eos: false }
    }
}

impl Engine {
    pub fn new(gguf: &GgufFile, model: Arc<ModelDesc>, store: WeightStore, experts: Option<Arc<kestrel_memory::ExpertStore>>, n_ctx: usize, threads: usize, ledger: &Arc<Ledger>) -> Result<Self, EngineError> {
        let tokenizer = Tokenizer::from_gguf(gguf).map_err(|e| EngineError::Unsupported(e.to_string()))?;
        let tok_text = |id: Option<u32>| id.map(|i| tokenizer.token_text(i).to_string()).unwrap_or_default();
        let chat = ChatTemplate::new(gguf.get_str("tokenizer.chat_template"), &tok_text(tokenizer.bos), &tok_text(tokenizer.eos));
        let tf = Transformer::new(model, store, experts, n_ctx, threads, ledger)?;
        Ok(Engine { tf, tokenizer, chat })
    }

    pub fn render_chat(&self, messages: &[ChatMessage]) -> Result<String, EngineError> {
        self.chat.render(messages, true).map_err(|e| EngineError::Other(format!("chat template: {e}")))
    }

    /// Tokenize a rendered prompt. Chat-rendered text already contains BOS
    /// as text when the template emits it, so BOS is only added if absent.
    pub fn tokenize_prompt(&self, text: &str) -> Vec<u32> {
        let mut toks = self.tokenizer.encode(text, true, true);
        if let (Some(b), [first, second, ..]) = (self.tokenizer.bos, toks.as_slice()) {
            if *first == b && *second == b {
                toks.remove(0);
            }
        }
        toks
    }

    /// Generate from `prompt` tokens. `on_text` receives decoded text pieces
    /// and returns `false` to stop. KV of a shared prefix with the previous
    /// call is reused.
    pub fn generate(&mut self, prompt: &[u32], params: &GenParams, mut on_text: impl FnMut(&str) -> bool) -> Result<GenStats, EngineError> {
        let mut stats = GenStats { prompt_tokens: prompt.len(), ..Default::default() };
        let mem0 = self.tf.store.metrics();
        let ex0 = self.tf.experts.as_ref().map(|e| e.metrics());
        let t0 = Instant::now();

        // Reuse the longest cached prefix, but always evaluate at least one
        // prompt token to obtain fresh logits.
        let common = self.tf.cached.iter().zip(prompt).take_while(|(a, b)| a == b).count();
        let keep = common.min(prompt.len().saturating_sub(1));
        self.tf.truncate(keep);
        let to_eval = &prompt[keep..];
        stats.prompt_evaluated = to_eval.len();
        let budget = params.max_tokens.min(self.tf.n_ctx.saturating_sub(prompt.len()));
        let mut logits = self.tf.forward(to_eval)?;
        stats.prefill_s = t0.elapsed().as_secs_f64();
        stats.ttft_s = stats.prefill_s;

        let mut sampler = Sampler::new(params.sampler.clone());
        let mut history: Vec<u32> = prompt.to_vec();
        let mut utf8 = Utf8Stream::default();
        let mut text = String::new();
        let t1 = Instant::now();
        stats.stop_reason = "length".into();
        for i in 0..budget {
            let tok = sampler.sample(&logits, &history);
            if self.tokenizer.is_eog(tok) && !params.ignore_eos {
                stats.stop_reason = "stop".into();
                break;
            }
            history.push(tok);
            stats.tokens.push(tok);
            stats.generated += 1;
            let piece = utf8.push(&self.tokenizer.token_bytes(tok, false));
            if !piece.is_empty() {
                text.push_str(&piece);
                if let Some(s) = params.stop.iter().find(|s| !s.is_empty() && text.ends_with(s.as_str())) {
                    let _ = s;
                    stats.stop_reason = "stop".into();
                    break;
                }
                if !on_text(&piece) {
                    stats.stop_reason = "cancelled".into();
                    break;
                }
            }
            if i + 1 == budget {
                break;
            }
            logits = self.tf.forward(&[tok])?;
        }
        let tail = utf8.flush();
        if !tail.is_empty() {
            on_text(&tail);
        }
        stats.decode_s = t1.elapsed().as_secs_f64();
        if stats.prefill_s > 0.0 {
            stats.prompt_tok_s = stats.prompt_evaluated as f64 / stats.prefill_s;
        }
        if stats.generated > 1 && stats.decode_s > 0.0 {
            // The first generated token's forward is part of prefill.
            stats.decode_tok_s = (stats.generated - 1) as f64 / stats.decode_s;
        }
        stats.memory = Some(self.tf.store.metrics().delta(&mem0));
        if let (Some(ex), Some(e0)) = (&self.tf.experts, ex0) {
            stats.experts = Some(ex.metrics().delta(&e0));
        }
        Ok(stats)
    }
}
