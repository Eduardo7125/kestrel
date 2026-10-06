//! The llama-family forward pass (llama, qwen2, qwen3) over Kestrel's
//! [`WeightStore`]: every weight is reached through a lease, so the same code
//! runs with the model fully resident, partially streamed, or streamed
//! layer by layer from NVMe.

use crate::quant::{self, matmul};
use kestrel_gguf::GgmlType;
use kestrel_memory::{Ledger, Lease, Reservation, StoreError, Tier, WeightStore};
use kestrel_model::{GroupKind, ModelDesc};
use rayon::prelude::*;
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("model not supported by the native executor: {0}")]
    Unsupported(String),
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error(transparent)]
    Budget(#[from] kestrel_memory::BudgetError),
    #[error("context full: {needed} positions needed, n_ctx = {n_ctx}")]
    ContextFull { needed: usize, n_ctx: usize },
    #[error("{0}")]
    Other(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arch {
    Llama,
    Qwen2,
    Qwen3,
}

#[derive(Clone, Copy, Debug)]
struct T {
    idx: usize,
    ty: GgmlType,
    rows: usize,
    cols: usize,
}

#[derive(Clone, Copy)]
struct LayerW {
    attn_group: usize,
    ffn_group: usize,
    attn_norm: T,
    wq: T,
    wk: T,
    wv: T,
    wo: T,
    bq: Option<T>,
    bk: Option<T>,
    bv: Option<T>,
    q_norm: Option<T>,
    k_norm: Option<T>,
    ffn_norm: T,
    gate: T,
    up: T,
    down: T,
}

/// Check that the native executor can run `model`.
pub fn check_support(model: &ModelDesc) -> Result<(), String> {
    if !matches!(model.arch.as_str(), "llama" | "qwen2" | "qwen3") {
        return Err(format!("architecture '{}' (native: llama, qwen2, qwen3; use --backend llamacpp)", model.arch));
    }
    if model.moe.is_some() {
        return Err("MoE execution is not implemented in the native executor yet".into());
    }
    for t in &model.tensors {
        if !quant::is_supported(t.ggml_type) {
            return Err(format!("tensor {} has type {} (native kernels: F32 F16 BF16 Q8_0 Q4_0 Q4_1 Q5_0 Q5_1 Q4_K Q5_K Q6_K)", t.name, t.ggml_type));
        }
    }
    Ok(())
}

pub struct Transformer {
    pub model: Arc<ModelDesc>,
    pub store: WeightStore,
    n_embd: usize,
    n_head: usize,
    n_kv: usize,
    hd: usize,
    n_ff: usize,
    n_vocab: usize,
    eps: f32,
    rope_neox: bool,
    rope_dim: usize,
    inv_freq: Vec<f32>,
    layers: Vec<LayerW>,
    embed_group: usize,
    embed: T,
    head_group: usize,
    out_norm: T,
    output: (usize, T),
    pub n_ctx: usize,
    k_cache: Vec<Vec<half::f16>>,
    v_cache: Vec<Vec<half::f16>>,
    /// Tokens whose K/V are in the cache, in order.
    pub cached: Vec<u32>,
    pool: rayon::ThreadPool,
    _kv_res: Reservation,
    /// Prompt rows processed per forward call.
    pub n_batch: usize,
}

impl Transformer {
    pub fn new(model: Arc<ModelDesc>, store: WeightStore, n_ctx: usize, threads: usize, ledger: &Arc<Ledger>) -> Result<Self, EngineError> {
        check_support(&model).map_err(EngineError::Unsupported)?;
        let arch = match model.arch.as_str() {
            "llama" => Arch::Llama,
            "qwen2" => Arch::Qwen2,
            _ => Arch::Qwen3,
        };
        let hp = &model.hparams;
        let find = |name: &str| -> Option<T> {
            let i = model.tensors.iter().position(|t| t.name == name)?;
            let t = &model.tensors[i];
            Some(T { idx: i, ty: t.ggml_type, cols: t.dims[0] as usize, rows: t.rows() as usize })
        };
        let req = |name: String| find(&name).ok_or_else(|| EngineError::Unsupported(format!("missing tensor {name}")));
        let group_of = |kind: GroupKind, layer: Option<u32>| -> Result<usize, EngineError> {
            model.groups.iter().find(|g| g.kind == kind && g.layer == layer).map(|g| g.id).ok_or_else(|| EngineError::Unsupported(format!("missing {kind:?} group for layer {layer:?}")))
        };
        let mut layers = Vec::new();
        for l in 0..hp.n_layer {
            let p = |s: &str| format!("blk.{l}.{s}");
            layers.push(LayerW {
                attn_group: group_of(GroupKind::Attn, Some(l))?,
                ffn_group: group_of(GroupKind::Ffn, Some(l))?,
                attn_norm: req(p("attn_norm.weight"))?,
                wq: req(p("attn_q.weight"))?,
                wk: req(p("attn_k.weight"))?,
                wv: req(p("attn_v.weight"))?,
                wo: req(p("attn_output.weight"))?,
                bq: find(&p("attn_q.bias")),
                bk: find(&p("attn_k.bias")),
                bv: find(&p("attn_v.bias")),
                q_norm: find(&p("attn_q_norm.weight")),
                k_norm: find(&p("attn_k_norm.weight")),
                ffn_norm: req(p("ffn_norm.weight"))?,
                gate: req(p("ffn_gate.weight"))?,
                up: req(p("ffn_up.weight"))?,
                down: req(p("ffn_down.weight"))?,
            });
        }
        let embed_group = group_of(GroupKind::Embed, None)?;
        let head_group = group_of(GroupKind::Head, None)?;
        let embed = req("token_embd.weight".into())?;
        let output = match find("output.weight") {
            Some(t) => (head_group, t),
            None => (embed_group, embed), // tied embeddings
        };
        let n_head = hp.n_head as usize;
        let n_kv = hp.n_head_kv.first().copied().unwrap_or(hp.n_head) as usize;
        if hp.n_head_kv.iter().any(|&k| k as usize != n_kv) {
            return Err(EngineError::Unsupported("per-layer KV head counts".into()));
        }
        let hd = hp.head_dim_k as usize;
        if hp.head_dim_v != hp.head_dim_k {
            return Err(EngineError::Unsupported("key/value head sizes differ".into()));
        }
        let rope_dim = hp.rope_dim as usize;
        let mut inv_freq: Vec<f32> = (0..rope_dim / 2).map(|i| hp.rope_freq_base.powf(-(2.0 * i as f32) / rope_dim as f32)).collect();
        // Llama 3 style frequency factors.
        if let Some(rf) = find("rope_freqs.weight") {
            let g = model.groups.iter().find(|g| g.tensors.contains(&rf.idx)).unwrap().id;
            let lease = store.lease(g)?;
            let mut f = vec![0f32; rf.cols * rf.rows];
            quant::dequant_row(rf.ty, lease.tensor(rf.idx), &mut f);
            for (i, v) in inv_freq.iter_mut().enumerate() {
                *v /= f[i];
            }
        }
        let n_ctx = n_ctx.max(1);
        let kv_elems = n_ctx * n_kv * hd;
        let kv_res = ledger.reserve(Tier::Ram, (2 * hp.n_layer as usize * kv_elems * 2) as u64, "kv")?;
        let k_cache = (0..hp.n_layer).map(|_| vec![half::f16::ZERO; kv_elems]).collect();
        let v_cache = (0..hp.n_layer).map(|_| vec![half::f16::ZERO; kv_elems]).collect();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads.max(1)).thread_name(|i| format!("kestrel-cpu-{i}")).build().map_err(|e| EngineError::Other(e.to_string()))?;
        Ok(Transformer {
            n_embd: hp.n_embd as usize,
            n_head,
            n_kv,
            hd,
            n_ff: hp.n_ff as usize,
            n_vocab: embed.rows,
            eps: hp.rms_eps,
            rope_neox: arch != Arch::Llama,
            rope_dim,
            inv_freq,
            layers,
            embed_group,
            embed,
            head_group,
            out_norm: req("output_norm.weight".into())?,
            output,
            n_ctx,
            k_cache,
            v_cache,
            cached: Vec::new(),
            pool,
            _kv_res: kv_res,
            n_batch: 64,
            model,
            store,
        })
    }

    pub fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    /// Forget cached positions from `keep` onwards.
    pub fn truncate(&mut self, keep: usize) {
        self.cached.truncate(keep);
    }

    /// Feed `tokens` after the cached ones; returns logits of the last token.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>, EngineError> {
        let start = self.cached.len();
        if start + tokens.len() > self.n_ctx {
            return Err(EngineError::ContextFull { needed: start + tokens.len(), n_ctx: self.n_ctx });
        }
        if tokens.is_empty() {
            return Err(EngineError::Other("forward() needs at least one token".into()));
        }
        let pool = std::mem::replace(&mut self.pool, rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap());
        let mut logits = Vec::new();
        let mut res = Ok(());
        pool.install(|| {
            for chunk in tokens.chunks(self.n_batch.max(1)) {
                match self.forward_chunk(chunk) {
                    Ok(l) => logits = l,
                    Err(e) => {
                        res = Err(e);
                        return;
                    }
                }
            }
        });
        self.pool = pool;
        res.map(|_| logits)
    }

    fn mm(&self, lease: &Lease, t: T, x: &[f32]) -> Vec<f32> {
        let s = x.len() / t.cols;
        let mut out = vec![0f32; s * t.rows];
        matmul(t.ty, lease.tensor(t.idx), t.rows, t.cols, x, &mut out);
        out
    }

    fn vec_of(&self, lease: &Lease, t: T) -> Vec<f32> {
        let mut v = vec![0f32; t.cols * t.rows];
        quant::dequant_row(t.ty, lease.tensor(t.idx), &mut v);
        v
    }

    fn rmsnorm(&self, x: &[f32], w: &[f32], dim: usize) -> Vec<f32> {
        let mut out = vec![0f32; x.len()];
        for (xi, oi) in x.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
            let ss: f32 = xi.iter().map(|v| v * v).sum::<f32>() / dim as f32;
            let r = 1.0 / (ss + self.eps).sqrt();
            for j in 0..dim {
                oi[j] = xi[j] * r * w[j];
            }
        }
        out
    }

    fn rope(&self, x: &mut [f32], n_heads: usize, pos0: usize) {
        let (hd, rd) = (self.hd, self.rope_dim);
        let s_rows = x.len() / (n_heads * hd);
        for s in 0..s_rows {
            let p = (pos0 + s) as f32;
            for h in 0..n_heads {
                let v = &mut x[(s * n_heads + h) * hd..][..hd];
                for i in 0..rd / 2 {
                    let (sin, cos) = (p * self.inv_freq[i]).sin_cos();
                    let (a, b) = if self.rope_neox { (i, i + rd / 2) } else { (2 * i, 2 * i + 1) };
                    let (x0, x1) = (v[a], v[b]);
                    v[a] = x0 * cos - x1 * sin;
                    v[b] = x0 * sin + x1 * cos;
                }
            }
        }
    }

    fn forward_chunk(&mut self, tokens: &[u32]) -> Result<Vec<f32>, EngineError> {
        let (d, s_rows) = (self.n_embd, tokens.len());
        let pos0 = self.cached.len();

        let mut x = vec![0f32; s_rows * d];
        {
            let lease = self.store.lease(self.embed_group)?;
            for (s, &t) in tokens.iter().enumerate() {
                if t as usize >= self.n_vocab {
                    return Err(EngineError::Other(format!("token id {t} out of range")));
                }
                quant::get_row(self.embed.ty, lease.tensor(self.embed.idx), d, t as usize, &mut x[s * d..(s + 1) * d]);
            }
        }

        for l in 0..self.layers.len() {
            // ---- attention ----
            let attn_out = {
                let lw = self.layers[l];
                let lease = self.store.lease(lw.attn_group)?;
                let h = self.rmsnorm(&x, &self.vec_of(&lease, lw.attn_norm), d);
                let mut q = self.mm(&lease, lw.wq, &h);
                let mut k = self.mm(&lease, lw.wk, &h);
                let mut v = self.mm(&lease, lw.wv, &h);
                for (b, y) in [(lw.bq, &mut q), (lw.bk, &mut k), (lw.bv, &mut v)] {
                    if let Some(b) = b {
                        let bias = self.vec_of(&lease, b);
                        for row in y.chunks_exact_mut(bias.len()) {
                            row.iter_mut().zip(&bias).for_each(|(a, b)| *a += b);
                        }
                    }
                }
                if let Some(qn) = lw.q_norm {
                    q = self.rmsnorm(&q, &self.vec_of(&lease, qn), self.hd);
                }
                if let Some(kn) = lw.k_norm {
                    k = self.rmsnorm(&k, &self.vec_of(&lease, kn), self.hd);
                }
                self.rope(&mut q, self.n_head, pos0);
                self.rope(&mut k, self.n_kv, pos0);
                let kvw = self.n_kv * self.hd;
                for s in 0..s_rows {
                    let p = pos0 + s;
                    for j in 0..kvw {
                        self.k_cache[l][p * kvw + j] = half::f16::from_f32(k[s * kvw + j]);
                        self.v_cache[l][p * kvw + j] = half::f16::from_f32(v[s * kvw + j]);
                    }
                }
                let ctx = self.attention(l, &q, pos0, s_rows);
                self.mm(&lease, lw.wo, &ctx)
            };
            x.iter_mut().zip(&attn_out).for_each(|(a, b)| *a += b);

            // ---- feed-forward (SwiGLU) ----
            let ffn_out = {
                let lw = self.layers[l];
                let lease = self.store.lease(lw.ffn_group)?;
                let h = self.rmsnorm(&x, &self.vec_of(&lease, lw.ffn_norm), d);
                let mut g = self.mm(&lease, lw.gate, &h);
                let u = self.mm(&lease, lw.up, &h);
                g.iter_mut().zip(&u).for_each(|(gv, uv)| *gv = *gv / (1.0 + (-*gv).exp()) * uv);
                debug_assert_eq!(g.len(), s_rows * self.n_ff);
                self.mm(&lease, lw.down, &g)
            };
            x.iter_mut().zip(&ffn_out).for_each(|(a, b)| *a += b);
        }
        self.cached.extend_from_slice(tokens);

        // Logits of the last row only.
        let last = &x[(s_rows - 1) * d..];
        let normed = {
            let lease = self.store.lease(self.head_group)?;
            self.rmsnorm(last, &self.vec_of(&lease, self.out_norm), d)
        };
        let (og, ot) = self.output;
        let lease = self.store.lease(og)?;
        Ok(self.mm(&lease, ot, &normed))
    }

    /// Causal GQA attention for `s_rows` new queries at positions `pos0..`.
    fn attention(&self, l: usize, q: &[f32], pos0: usize, s_rows: usize) -> Vec<f32> {
        let (hd, nh, nkv) = (self.hd, self.n_head, self.n_kv);
        let group = nh / nkv;
        let scale = 1.0 / (hd as f32).sqrt();
        let (kc, vc) = (&self.k_cache[l], &self.v_cache[l]);
        let kvw = nkv * hd;
        let mut out = vec![0f32; s_rows * nh * hd];
        out.par_chunks_mut(hd).enumerate().for_each(|(i, o)| {
            let (s, h) = (i / nh, i % nh);
            let kvh = h / group;
            let qv = &q[(s * nh + h) * hd..][..hd];
            let n = pos0 + s + 1;
            let mut scores = Vec::with_capacity(n);
            let mut kbuf = vec![0f32; hd];
            for p in 0..n {
                let kr = &kc[p * kvw + kvh * hd..][..hd];
                for (a, b) in kbuf.iter_mut().zip(kr) {
                    *a = b.to_f32();
                }
                scores.push(quant::dot(qv, &kbuf) * scale);
            }
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for sc in scores.iter_mut() {
                *sc = (*sc - m).exp();
                sum += *sc;
            }
            for (p, &w) in scores.iter().enumerate() {
                let w = w / sum;
                let vr = &vc[p * kvw + kvh * hd..][..hd];
                for (a, b) in o.iter_mut().zip(vr) {
                    *a += w * b.to_f32();
                }
            }
        });
        out
    }
}
