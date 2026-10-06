//! The llama-family forward pass (llama, qwen2, qwen3, and their MoE
//! variants qwen2moe, qwen3moe, Mixtral-style llama) over Kestrel's
//! [`WeightStore`]: every weight is reached through a lease, so the same code
//! runs with the model fully resident, partially streamed, or streamed
//! layer by layer from NVMe.

use crate::quant::{self, matmul};
use kestrel_gguf::GgmlType;
use kestrel_memory::{ExpertStore, Ledger, Lease, Reservation, StoreError, Tier, WeightStore};
use kestrel_model::{GroupKind, ModelDesc, MoeInfo};
use std::sync::Mutex;
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
    // Dense FFN.
    gate: Option<T>,
    up: Option<T>,
    down: Option<T>,
    // MoE FFN: router, stacked experts (gate, up, down), shared expert.
    router: Option<T>,
    exps: Option<[T; 3]>,
    sh_inp: Option<T>,
    sh_gate: Option<T>,
    sh_up: Option<T>,
    sh_down: Option<T>,
}

/// Check that the native executor can run `model`.
pub fn check_support(model: &ModelDesc) -> Result<(), String> {
    if !matches!(model.arch.as_str(), "llama" | "qwen2" | "qwen3" | "qwen2moe" | "qwen3moe") {
        return Err(format!("architecture '{}' (native: llama, qwen2, qwen3, qwen2moe, qwen3moe; use --backend llamacpp)", model.arch));
    }
    if model.moe.is_some() && kestrel_memory::expert::ExpertGeometry::from_model(model).is_none() {
        return Err("MoE model without stacked *_exps expert tensors".into());
    }
    for t in &model.tensors {
        if !quant::is_supported(t.ggml_type) {
            return Err(format!("tensor {} has type {} (native kernels: F32 F16 BF16 Q8_0 Q4_0 Q4_1 Q5_0 Q5_1 Q4_K Q5_K Q6_K)", t.name, t.ggml_type));
        }
    }
    Ok(())
}

static MM_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LEASE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ATTN_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

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
    /// Seconds per phase (weights wait, matmul, attention, other), when
    /// `KESTREL_PROFILE=1`.
    pub profile: Option<Profile>,
    /// Routed experts (MoE models).
    pub experts: Option<Arc<ExpertStore>>,
    moe: Option<MoeInfo>,
    /// Router-lookahead prefetch: predict layer l+1's experts from layer l's
    /// post-attention state (Colibrì's PILOT). `KESTREL_LOOKAHEAD=0` disables.
    pub lookahead: bool,
    /// Predicted experts per layer, to score recall when the layer routes.
    predicted: Mutex<Vec<Option<Vec<u32>>>>,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Profile {
    pub lease_s: f64,
    pub matmul_s: f64,
    pub attention_s: f64,
    pub other_s: f64,
    pub head_s: f64,
}

impl Transformer {
    pub fn new(model: Arc<ModelDesc>, store: WeightStore, experts: Option<Arc<ExpertStore>>, n_ctx: usize, threads: usize, ledger: &Arc<Ledger>) -> Result<Self, EngineError> {
        check_support(&model).map_err(EngineError::Unsupported)?;
        if model.moe.is_some() && experts.is_none() {
            return Err(EngineError::Unsupported("MoE model needs an expert store".into()));
        }
        let arch = match model.arch.as_str() {
            "llama" => Arch::Llama,
            "qwen2" | "qwen2moe" => Arch::Qwen2,
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
                gate: find(&p("ffn_gate.weight")),
                up: find(&p("ffn_up.weight")),
                down: find(&p("ffn_down.weight")),
                router: find(&p("ffn_gate_inp.weight")),
                exps: match (find(&p("ffn_gate_exps.weight")), find(&p("ffn_up_exps.weight")), find(&p("ffn_down_exps.weight"))) {
                    (Some(g), Some(u), Some(d)) => Some([g, u, d]),
                    _ => None,
                },
                sh_inp: find(&p("ffn_gate_inp_shexp.weight")),
                sh_gate: find(&p("ffn_gate_shexp.weight")),
                sh_up: find(&p("ffn_up_shexp.weight")),
                sh_down: find(&p("ffn_down_shexp.weight")),
            });
            let lw = layers.last().unwrap();
            let dense = lw.gate.is_some() && lw.up.is_some() && lw.down.is_some();
            let moe = lw.router.is_some() && lw.exps.is_some();
            if !dense && !moe {
                return Err(EngineError::Unsupported(format!("layer {l} has neither a dense nor a routed FFN")));
            }
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
            n_batch: 256,
            profile: std::env::var("KESTREL_PROFILE").ok().filter(|v| v == "1").map(|_| Profile::default()),
            lookahead: std::env::var("KESTREL_LOOKAHEAD").map(|v| v != "0").unwrap_or(true),
            predicted: Mutex::new(vec![None; hp.n_layer as usize]),
            moe: model.moe.clone(),
            experts,
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
        let t0 = std::time::Instant::now();
        matmul(t.ty, lease.tensor(t.idx), t.rows, t.cols, x, &mut out);
        if self.profile.is_some() {
            MM_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        }
        out
    }

    fn lease(&self, g: usize) -> Result<Lease, EngineError> {
        let t0 = std::time::Instant::now();
        let l = self.store.lease(g)?;
        if self.profile.is_some() {
            LEASE_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(l)
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
        let t_all = std::time::Instant::now();
        let (m0, l0, a0) = (MM_NS.load(std::sync::atomic::Ordering::Relaxed), LEASE_NS.load(std::sync::atomic::Ordering::Relaxed), ATTN_NS.load(std::sync::atomic::Ordering::Relaxed));
        let r = self.forward_chunk_inner(tokens);
        if let Some(p) = self.profile.as_mut() {
            let g = |a: &std::sync::atomic::AtomicU64, b: u64| (a.load(std::sync::atomic::Ordering::Relaxed) - b) as f64 * 1e-9;
            let (mm, ls, at) = (g(&MM_NS, m0), g(&LEASE_NS, l0), g(&ATTN_NS, a0));
            p.matmul_s += mm;
            p.lease_s += ls;
            p.attention_s += at;
            p.other_s += (t_all.elapsed().as_secs_f64() - mm - ls - at).max(0.0);
        }
        r
    }

    fn forward_chunk_inner(&mut self, tokens: &[u32]) -> Result<Vec<f32>, EngineError> {
        let (d, s_rows) = (self.n_embd, tokens.len());
        let pos0 = self.cached.len();

        let mut x = vec![0f32; s_rows * d];
        {
            let lease = self.lease(self.embed_group)?;
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
                let lease = self.lease(lw.attn_group)?;
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
                let ta = std::time::Instant::now();
                let ctx = self.attention(l, &q, pos0, s_rows);
                if self.profile.is_some() {
                    ATTN_NS.fetch_add(ta.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                }
                self.mm(&lease, lw.wo, &ctx)
            };
            x.iter_mut().zip(&attn_out).for_each(|(a, b)| *a += b);

            // Router lookahead: guess the next MoE layer's experts from this
            // layer's post-attention state and start loading them now.
            if self.lookahead && s_rows <= 4 && l + 1 < self.layers.len() && self.layers[l + 1].router.is_some() {
                self.lookahead_prefetch(l + 1, &x)?;
            }

            // ---- feed-forward (SwiGLU, dense or routed experts) ----
            let ffn_out = {
                let lw = self.layers[l];
                let lease = self.lease(lw.ffn_group)?;
                let h = self.rmsnorm(&x, &self.vec_of(&lease, lw.ffn_norm), d);
                if lw.router.is_some() {
                    self.moe_ffn(l, &lease, &h)?
                } else {
                    let (gate, up, down) = (lw.gate.unwrap(), lw.up.unwrap(), lw.down.unwrap());
                    let mut g = self.mm(&lease, gate, &h);
                    let u = self.mm(&lease, up, &h);
                    g.iter_mut().zip(&u).for_each(|(gv, uv)| *gv = *gv / (1.0 + (-*gv).exp()) * uv);
                    debug_assert_eq!(g.len(), s_rows * self.n_ff);
                    self.mm(&lease, down, &g)
                }
            };
            x.iter_mut().zip(&ffn_out).for_each(|(a, b)| *a += b);
        }
        self.cached.extend_from_slice(tokens);

        // Logits of the last row only.
        let last = &x[(s_rows - 1) * d..];
        let normed = {
            let lease = self.lease(self.head_group)?;
            self.rmsnorm(last, &self.vec_of(&lease, self.out_norm), d)
        };
        let (og, ot) = self.output;
        let lease = self.lease(og)?;
        Ok(self.mm(&lease, ot, &normed))
    }

    /// Router: softmax over experts, top-k, optional renormalization and
    /// scaling (llama.cpp `build_moe_ffn` semantics). Returns per row the
    /// selected (expert, weight) pairs.
    fn route(&self, lease: &Lease, router: T, h: &[f32]) -> Vec<Vec<(u32, f32)>> {
        let moe = self.moe.as_ref().unwrap();
        let (ne, k) = (moe.n_expert as usize, moe.n_expert_used as usize);
        let logits = self.mm(lease, router, h);
        logits
            .chunks_exact(ne)
            .map(|row| {
                let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut p: Vec<f32> = row.iter().map(|v| (v - m).exp()).collect();
                let sum: f32 = p.iter().sum();
                p.iter_mut().for_each(|v| *v /= sum);
                let mut idx: Vec<usize> = (0..ne).collect();
                idx.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
                let mut sel: Vec<(u32, f32)> = idx[..k].iter().map(|&e| (e as u32, p[e])).collect();
                if moe.norm_topk {
                    let s: f32 = sel.iter().map(|x| x.1).sum::<f32>().max(6.103_515_6e-5);
                    sel.iter_mut().for_each(|x| x.1 /= s);
                }
                if moe.weights_scale != 0.0 && moe.weights_scale != 1.0 {
                    sel.iter_mut().for_each(|x| x.1 *= moe.weights_scale);
                }
                sel
            })
            .collect()
    }

    fn lookahead_prefetch(&self, next: usize, x: &[f32]) -> Result<(), EngineError> {
        let lw = self.layers[next];
        let lease = self.lease(lw.ffn_group)?;
        let h = self.rmsnorm(x, &self.vec_of(&lease, lw.ffn_norm), self.n_embd);
        let mut pred: Vec<u32> = self.route(&lease, lw.router.unwrap(), &h).into_iter().flatten().map(|(e, _)| e).collect();
        pred.sort_unstable();
        pred.dedup();
        self.experts.as_ref().unwrap().prefetch(next as u32, &pred);
        self.predicted.lock().unwrap()[next] = Some(pred);
        Ok(())
    }

    /// Mixture-of-experts FFN with batch union: every expert routed by any
    /// row is fetched once and applied to all of its rows. Cached experts are
    /// computed first while missing ones load in parallel.
    fn moe_ffn(&self, l: usize, lease: &Lease, h: &[f32]) -> Result<Vec<f32>, EngineError> {
        let lw = self.layers[l];
        let d = self.n_embd;
        let s_rows = h.len() / d;
        let ex = self.experts.as_ref().unwrap();
        let ne = self.moe.as_ref().unwrap().n_expert as usize;
        let routes = self.route(lease, lw.router.unwrap(), h);
        if let Some(pred) = self.predicted.lock().unwrap()[l].take() {
            let actual: std::collections::BTreeSet<u32> = routes.iter().flatten().map(|x| x.0).collect();
            let correct = pred.iter().filter(|e| actual.contains(e)).count() as u64;
            ex.record_lookahead(pred.len() as u64, correct);
        }
        let mut assign: std::collections::BTreeMap<u32, Vec<(usize, f32)>> = Default::default();
        for (s, sel) in routes.iter().enumerate() {
            for &(e, w) in sel {
                assign.entry(e).or_default().push((s, w));
            }
        }
        let needed: Vec<u32> = assign.keys().copied().collect();
        ex.request(l as u32, &needed)?;
        let mut order = needed.clone();
        order.sort_by_key(|&e| !ex.is_ready(l as u32, e));

        let [tg, tu, td] = lw.exps.unwrap();
        let ff = tg.rows / ne;
        let mut out = vec![0f32; s_rows * d];
        // Experts are *computed* in residency order (cached first), but their
        // contributions are *summed* in expert-id order, so the result is
        // bit-identical whatever the cache state (float addition does not
        // commute bitwise).
        let mut ys: std::collections::BTreeMap<u32, Vec<f32>> = Default::default();
        for e in order {
            let rows = &assign[&e];
            let t0 = std::time::Instant::now();
            let eb = ex.get(l as u32, e)?;
            if self.profile.is_some() {
                LEASE_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            }
            let xs: Vec<f32> = rows.iter().flat_map(|&(s, _)| h[s * d..(s + 1) * d].iter().copied()).collect();
            let n = rows.len();
            let tm = std::time::Instant::now();
            let mut g = vec![0f32; n * ff];
            let mut u = vec![0f32; n * ff];
            matmul(tg.ty, eb.part(0), ff, d, &xs, &mut g);
            matmul(tu.ty, eb.part(1), ff, d, &xs, &mut u);
            g.iter_mut().zip(&u).for_each(|(gv, uv)| *gv = *gv / (1.0 + (-*gv).exp()) * uv);
            let mut y = vec![0f32; n * d];
            matmul(td.ty, eb.part(2), d, ff, &g, &mut y);
            if self.profile.is_some() {
                MM_NS.fetch_add(tm.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            }
            ys.insert(e, y);
        }
        for (e, y) in &ys {
            for (i, &(s, w)) in assign[e].iter().enumerate() {
                out[s * d..(s + 1) * d].iter_mut().zip(&y[i * d..(i + 1) * d]).for_each(|(o, v)| *o += w * v);
            }
        }
        // Shared expert (qwen2moe): always active, scaled by sigmoid(gate·h).
        if let (Some(sg), Some(su), Some(sd)) = (lw.sh_gate, lw.sh_up, lw.sh_down) {
            let mut g = self.mm(lease, sg, h);
            let u = self.mm(lease, su, h);
            g.iter_mut().zip(&u).for_each(|(gv, uv)| *gv = *gv / (1.0 + (-*gv).exp()) * uv);
            let y = self.mm(lease, sd, &g);
            let gate: Vec<f32> = match lw.sh_inp {
                Some(si) => {
                    let w = self.vec_of(lease, si);
                    (0..s_rows).map(|s| 1.0 / (1.0 + (-quant::dot(&w, &h[s * d..(s + 1) * d])).exp())).collect()
                }
                None => vec![1.0; s_rows],
            };
            for s in 0..s_rows {
                out[s * d..(s + 1) * d].iter_mut().zip(&y[s * d..(s + 1) * d]).for_each(|(o, v)| *o += gate[s] * v);
            }
        }
        Ok(out)
    }

    /// Causal GQA attention for `s_rows` new queries at positions `pos0..`.
    /// The layer's K/V history is converted to f32 once per call, then every
    /// (query, head) pair runs SIMD dot products over it.
    fn attention(&self, l: usize, q: &[f32], pos0: usize, s_rows: usize) -> Vec<f32> {
        let (hd, nh, nkv) = (self.hd, self.n_head, self.n_kv);
        let group = nh / nkv;
        let scale = 1.0 / (hd as f32).sqrt();
        let kvw = nkv * hd;
        let n_tot = pos0 + s_rows;
        let as_bits = |v: &[half::f16]| -> &[u16] {
            // SAFETY: half::f16 is repr(transparent) over u16.
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u16, v.len()) }
        };
        let mut kf = vec![0f32; n_tot * kvw];
        let mut vf = vec![0f32; n_tot * kvw];
        kf.par_chunks_mut(kvw * 64).zip(as_bits(&self.k_cache[l][..n_tot * kvw]).par_chunks(kvw * 64)).for_each(|(d, s)| quant::f16_slice_to_f32(s, d));
        vf.par_chunks_mut(kvw * 64).zip(as_bits(&self.v_cache[l][..n_tot * kvw]).par_chunks(kvw * 64)).for_each(|(d, s)| quant::f16_slice_to_f32(s, d));
        let mut out = vec![0f32; s_rows * nh * hd];
        out.par_chunks_mut(hd).enumerate().for_each(|(i, o)| {
            let (s, h) = (i / nh, i % nh);
            let kvh = h / group;
            let qv = &q[(s * nh + h) * hd..][..hd];
            let n = pos0 + s + 1;
            let mut scores: Vec<f32> = (0..n).map(|p| quant::dot(qv, &kf[p * kvw + kvh * hd..][..hd]) * scale).collect();
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for sc in scores.iter_mut() {
                *sc = (*sc - m).exp();
                sum += *sc;
            }
            let inv = 1.0 / sum;
            for (p, &w) in scores.iter().enumerate() {
                let w = w * inv;
                let vr = &vf[p * kvw + kvh * hd..][..hd];
                for (a, b) in o.iter_mut().zip(vr) {
                    *a += w * b;
                }
            }
        });
        out
    }
}
