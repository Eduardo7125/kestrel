//! Architecture-neutral model description.
//!
//! A [`ModelDesc`] is what the planner and memory orchestrator know about a
//! model: hyper-parameters, the KV-cache geometry, the MoE geometry, and the
//! weights partitioned into [`TensorGroup`]s, which are the units of placement
//! (see `docs/memory-model.md`). Building one reads only the GGUF header.

mod adapters;

pub use adapters::{adapter_for, all as all_adapters, ArchAdapter, ArchSupport};

use kestrel_gguf::{GgmlType, GgufFile, TensorInfo};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error(transparent)]
    Gguf(#[from] kestrel_gguf::GgufError),
    #[error("model is missing required metadata '{0}'")]
    Missing(String),
    #[error("{0}")]
    Invalid(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GroupKind {
    /// `token_embd`: one row gathered per token.
    Embed,
    /// Attention block of one layer (norm, q/k/v/o, biases, q/k norms).
    Attn,
    /// Dense FFN of one layer, or the always-active part of an MoE layer
    /// (router, shared experts, norms).
    Ffn,
    /// Routed experts of one layer (the stacked `*_exps` tensors).
    Experts,
    /// Output norm and LM head.
    Head,
    /// Anything else not tied to a layer (e.g. `rope_freqs`).
    Other,
}

/// A contiguous byte range of the model file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Extent {
    pub offset: u64,
    pub len: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct TensorGroup {
    pub id: usize,
    pub kind: GroupKind,
    pub layer: Option<u32>,
    /// Indices into [`ModelDesc::tensors`].
    pub tensors: Vec<usize>,
    /// Coalesced file extents covering all tensors of the group.
    pub extents: Vec<Extent>,
    pub bytes: u64,
    /// Expected fraction of this group's bytes touched per decoded token.
    /// 1.0 for dense groups, k/E for routed experts, ~row/total for embeddings.
    pub touch_per_token: f64,
}

impl TensorGroup {
    pub fn label(&self) -> String {
        match (self.kind, self.layer) {
            (GroupKind::Embed, _) => "embed".into(),
            (GroupKind::Head, _) => "head".into(),
            (GroupKind::Other, _) => "other".into(),
            (k, Some(l)) => format!("blk.{l}.{}", match k {
                GroupKind::Attn => "attn",
                GroupKind::Ffn => "ffn",
                GroupKind::Experts => "experts",
                _ => "?",
            }),
            (_, None) => "?".into(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct HParams {
    pub n_layer: u32,
    pub n_embd: u32,
    pub n_head: u32,
    /// KV heads per layer (GQA). Usually constant.
    pub n_head_kv: Vec<u32>,
    pub head_dim_k: u32,
    pub head_dim_v: u32,
    pub n_ff: u32,
    pub n_vocab: u32,
    pub n_ctx_train: u32,
    pub rope_freq_base: f32,
    pub rope_dim: u32,
    pub rms_eps: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct MoeInfo {
    pub n_expert: u32,
    pub n_expert_used: u32,
    pub n_expert_shared: u32,
    /// Bytes of one routed expert in one layer (gate+up+down slices).
    pub bytes_per_expert: u64,
    /// Number of layers with routed experts.
    pub moe_layers: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelDesc {
    pub path: PathBuf,
    pub name: String,
    pub arch: String,
    pub support: ArchSupport,
    pub file_size: u64,
    pub fingerprint: String,
    pub file_type: Option<String>,
    pub hparams: HParams,
    pub moe: Option<MoeInfo>,
    pub n_params: u64,
    pub tensors: Vec<TensorInfo>,
    pub groups: Vec<TensorGroup>,
    /// Bytes per ggml type across all tensors.
    pub bytes_by_type: BTreeMap<String, u64>,
    pub has_chat_template: bool,
}

impl ModelDesc {
    pub fn from_gguf(g: &GgufFile) -> Result<Self, ModelError> {
        let arch = g.architecture().ok_or_else(|| ModelError::Missing("general.architecture".into()))?.to_string();
        let adapter = adapter_for(&arch);
        let hparams = read_hparams(g, &arch)?;
        let tensors = g.tensors.clone();

        let mut bytes_by_type = BTreeMap::new();
        for t in &tensors {
            *bytes_by_type.entry(t.ggml_type.name().to_string()).or_insert(0) += t.size;
        }

        // Partition tensors into groups.
        let mut keyed: BTreeMap<(u8, u32, GroupKind), Vec<usize>> = BTreeMap::new();
        for (i, t) in tensors.iter().enumerate() {
            let (layer, kind) = classify(&t.name);
            // Execution order: embed, per-layer [attn, ffn, experts], head, other.
            let (major, l) = match (kind, layer) {
                (GroupKind::Embed, _) => (0, 0),
                (_, Some(l)) => (1, l),
                (GroupKind::Head, _) => (2, 0),
                _ => (3, 0),
            };
            keyed.entry((major, l, kind)).or_default().push(i);
        }
        let mut groups = Vec::new();
        for ((_, _, kind), idx) in keyed {
            let layer = classify(&tensors[idx[0]].name).0;
            let mut ext: Vec<Extent> = idx.iter().map(|&i| Extent { offset: tensors[i].offset, len: tensors[i].size }).collect();
            ext.sort_by_key(|e| e.offset);
            let extents = coalesce(&ext, g.alignment);
            let bytes = idx.iter().map(|&i| tensors[i].size).sum();
            groups.push(TensorGroup { id: groups.len(), kind, layer, tensors: idx, extents, bytes, touch_per_token: 1.0 });
        }

        // MoE geometry.
        let n_expert = g.arch_u64("expert_count").unwrap_or(0) as u32;
        let n_expert_used = g.arch_u64("expert_used_count").unwrap_or(0) as u32;
        let moe = if n_expert > 1 {
            let exp_groups: Vec<&TensorGroup> = groups.iter().filter(|gr| gr.kind == GroupKind::Experts).collect();
            let bytes_per_expert = exp_groups.first().map(|gr| gr.bytes / n_expert as u64).unwrap_or(0);
            Some(MoeInfo {
                n_expert,
                n_expert_used: n_expert_used.max(1),
                n_expert_shared: g.arch_u64("expert_shared_count").unwrap_or(0) as u32,
                bytes_per_expert,
                moe_layers: exp_groups.len() as u32,
            })
        } else {
            None
        };
        for gr in &mut groups {
            gr.touch_per_token = match gr.kind {
                GroupKind::Experts => moe.as_ref().map(|m| m.n_expert_used as f64 / m.n_expert as f64).unwrap_or(1.0),
                GroupKind::Embed if hparams.n_vocab > 0 => 1.0 / hparams.n_vocab as f64,
                _ => 1.0,
            };
        }

        let n_params = tensors.iter().map(|t| t.n_elements()).sum();
        let name = g
            .get_str("general.name")
            .map(str::to_string)
            .unwrap_or_else(|| g.path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default());
        Ok(ModelDesc {
            path: g.path.clone(),
            name,
            support: adapter.support,
            arch,
            file_size: g.file_size,
            fingerprint: g.fingerprint(),
            file_type: g.get_u64("general.file_type").map(|f| kestrel_gguf::file_type_name(f as u32).to_string()),
            hparams,
            moe,
            n_params,
            tensors,
            groups,
            bytes_by_type,
            has_chat_template: g.get_str("tokenizer.chat_template").is_some(),
        })
    }

    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, ModelError> {
        Self::from_gguf(&GgufFile::open(path)?)
    }

    pub fn weight_bytes(&self) -> u64 {
        self.groups.iter().map(|g| g.bytes).sum()
    }

    /// KV-cache bytes per token of context, summed over layers.
    pub fn kv_bytes_per_token(&self, kv_type: GgmlType) -> u64 {
        let h = &self.hparams;
        let per_elem = kv_type.type_size() as f64 / kv_type.block_size() as f64;
        let elems: u64 = (0..h.n_layer as usize)
            .map(|l| {
                let kv = *h.n_head_kv.get(l).or(h.n_head_kv.last()).unwrap_or(&h.n_head) as u64;
                kv * (h.head_dim_k as u64 + h.head_dim_v as u64)
            })
            .sum();
        (elems as f64 * per_elem).ceil() as u64
    }

    pub fn kv_bytes(&self, n_ctx: u64, kv_type: GgmlType) -> u64 {
        self.kv_bytes_per_token(kv_type) * n_ctx
    }

    pub fn groups_of_layer(&self, layer: u32) -> impl Iterator<Item = &TensorGroup> {
        self.groups.iter().filter(move |g| g.layer == Some(layer))
    }

    /// Bytes touched per decoded token if every weight were resident:
    /// dense groups fully, experts by their activation probability.
    pub fn bytes_touched_per_token(&self) -> f64 {
        self.groups.iter().map(|g| g.bytes as f64 * g.touch_per_token).sum()
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// The dominant weight quantization, by bytes.
    pub fn dominant_type(&self) -> Option<&str> {
        self.bytes_by_type.iter().max_by_key(|(_, b)| **b).map(|(t, _)| t.as_str())
    }
}

/// Map a GGUF tensor name to (layer, group kind).
pub fn classify(name: &str) -> (Option<u32>, GroupKind) {
    if name.starts_with("token_embd") {
        return (None, GroupKind::Embed);
    }
    if name.starts_with("output") {
        return (None, GroupKind::Head);
    }
    if let Some(rest) = name.strip_prefix("blk.") {
        let mut parts = rest.splitn(2, '.');
        let layer = parts.next().and_then(|s| s.parse::<u32>().ok());
        let t = parts.next().unwrap_or("");
        if let Some(l) = layer {
            let kind = if t.contains("_exps") {
                GroupKind::Experts
            } else if t.starts_with("ffn") || t.starts_with("post_ffw") || t.starts_with("layer_output") {
                GroupKind::Ffn
            } else {
                GroupKind::Attn
            };
            return (Some(l), kind);
        }
    }
    (None, GroupKind::Other)
}

/// Merge extents separated only by alignment padding.
fn coalesce(sorted: &[Extent], alignment: u64) -> Vec<Extent> {
    let mut out: Vec<Extent> = Vec::new();
    for e in sorted {
        if let Some(last) = out.last_mut() {
            let end = last.offset + last.len;
            if e.offset >= end && e.offset - end < alignment {
                last.len = e.offset + e.len - last.offset;
                continue;
            }
        }
        out.push(*e);
    }
    out
}

fn read_hparams(g: &GgufFile, arch: &str) -> Result<HParams, ModelError> {
    let req = |k: &str| g.arch_u64(k).ok_or_else(|| ModelError::Missing(format!("{arch}.{k}")));
    let n_layer = req("block_count")? as u32;
    let n_embd = req("embedding_length")? as u32;
    let n_head = scalar_or_max(g.arch_value("attention.head_count")).ok_or_else(|| ModelError::Missing(format!("{arch}.attention.head_count")))?;
    let n_head_kv = match g.arch_value("attention.head_count_kv") {
        Some(v) => match v.as_array() {
            Some(a) => a.iter().map(|x| x.as_u64().unwrap_or(0) as u32).collect(),
            None => vec![v.as_u64().unwrap_or(n_head as u64) as u32; n_layer as usize],
        },
        None => vec![n_head; n_layer as usize],
    };
    let head_dim = if n_head > 0 { n_embd / n_head } else { 0 };
    let head_dim_k = g.arch_u64("attention.key_length").map(|v| v as u32).unwrap_or(head_dim);
    let head_dim_v = g.arch_u64("attention.value_length").map(|v| v as u32).unwrap_or(head_dim);
    let n_ff = scalar_or_max(g.arch_value("feed_forward_length")).unwrap_or(0);
    let n_vocab = g
        .arch_u64("vocab_size")
        .map(|v| v as u32)
        .or_else(|| g.get("tokenizer.ggml.tokens").and_then(|v| v.as_array()).map(|a| a.len() as u32))
        .or_else(|| g.tensor("token_embd.weight").map(|t| t.dims.get(1).copied().unwrap_or(0) as u32))
        .unwrap_or(0);
    Ok(HParams {
        n_layer,
        n_embd,
        n_head,
        n_head_kv,
        head_dim_k,
        head_dim_v,
        n_ff,
        n_vocab,
        n_ctx_train: g.arch_u64("context_length").unwrap_or(2048) as u32,
        rope_freq_base: g.arch_f64("rope.freq_base").unwrap_or(10000.0) as f32,
        rope_dim: g.arch_u64("rope.dimension_count").map(|v| v as u32).unwrap_or(head_dim_k),
        rms_eps: g.arch_f64("attention.layer_norm_rms_epsilon").unwrap_or(1e-5) as f32,
    })
}

fn scalar_or_max(v: Option<&kestrel_gguf::Value>) -> Option<u32> {
    let v = v?;
    match v.as_array() {
        Some(a) => a.iter().filter_map(|x| x.as_u64()).max().map(|x| x as u32),
        None => v.as_u64().map(|x| x as u32),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kestrel_gguf::writer::{GgufWriter, TensorData};
    use kestrel_gguf::Value;

    fn tiny(dir: &std::path::Path, moe: bool) -> PathBuf {
        let p = dir.join("m.gguf");
        let mut w = GgufWriter::new();
        let arch = if moe { "qwen3moe" } else { "llama" };
        w.kv("general.architecture", Value::String(arch.into()));
        w.kv(&format!("{arch}.block_count"), Value::U32(2));
        w.kv(&format!("{arch}.embedding_length"), Value::U32(64));
        w.kv(&format!("{arch}.attention.head_count"), Value::U32(4));
        w.kv(&format!("{arch}.attention.head_count_kv"), Value::U32(2));
        w.kv(&format!("{arch}.feed_forward_length"), Value::U32(128));
        if moe {
            w.kv(&format!("{arch}.expert_count"), Value::U32(8));
            w.kv(&format!("{arch}.expert_used_count"), Value::U32(2));
        }
        w.tensor("token_embd.weight", &[64, 100], TensorData::F32(vec![0.0; 6400]));
        for l in 0..2 {
            w.tensor(&format!("blk.{l}.attn_norm.weight"), &[64], TensorData::F32(vec![1.0; 64]));
            w.tensor(&format!("blk.{l}.attn_q.weight"), &[64, 64], TensorData::F32(vec![0.0; 4096]));
            w.tensor(&format!("blk.{l}.ffn_norm.weight"), &[64], TensorData::F32(vec![1.0; 64]));
            if moe {
                w.tensor(&format!("blk.{l}.ffn_gate_inp.weight"), &[64, 8], TensorData::F32(vec![0.0; 512]));
                w.tensor(&format!("blk.{l}.ffn_up_exps.weight"), &[64, 32, 8], TensorData::F32(vec![0.0; 64 * 32 * 8]));
            } else {
                w.tensor(&format!("blk.{l}.ffn_up.weight"), &[64, 128], TensorData::F32(vec![0.0; 8192]));
            }
        }
        w.tensor("output_norm.weight", &[64], TensorData::F32(vec![1.0; 64]));
        w.write(&p).unwrap();
        p
    }

    #[test]
    fn dense_groups() {
        let d = tempfile::tempdir().unwrap();
        let m = ModelDesc::open(tiny(d.path(), false)).unwrap();
        let labels: Vec<String> = m.groups.iter().map(|g| g.label()).collect();
        assert_eq!(labels, ["embed", "blk.0.attn", "blk.0.ffn", "blk.1.attn", "blk.1.ffn", "head"]);
        // Tensors of one group are adjacent in the file: one extent each.
        assert!(m.groups.iter().all(|g| g.extents.len() == 1), "{:?}", m.groups);
        assert_eq!(m.weight_bytes(), m.tensors.iter().map(|t| t.size).sum::<u64>());
        assert_eq!(m.hparams.head_dim_k, 16);
        // 2 layers × 2 kv heads × (16+16) × 2 bytes
        assert_eq!(m.kv_bytes_per_token(GgmlType::F16), 2 * 2 * 32 * 2);
        assert!(m.moe.is_none());
    }

    #[test]
    fn moe_geometry() {
        let d = tempfile::tempdir().unwrap();
        let m = ModelDesc::open(tiny(d.path(), true)).unwrap();
        let moe = m.moe.as_ref().unwrap();
        assert_eq!(moe.n_expert, 8);
        assert_eq!(moe.moe_layers, 2);
        assert_eq!(moe.bytes_per_expert, 64 * 32 * 4);
        let e = m.groups.iter().find(|g| g.kind == GroupKind::Experts).unwrap();
        assert!((e.touch_per_token - 0.25).abs() < 1e-9);
    }

    #[test]
    fn classify_names() {
        assert_eq!(classify("blk.3.ffn_down_exps.weight"), (Some(3), GroupKind::Experts));
        assert_eq!(classify("blk.3.ffn_gate_shexp.weight"), (Some(3), GroupKind::Ffn));
        assert_eq!(classify("blk.12.attn_k_norm.weight"), (Some(12), GroupKind::Attn));
        assert_eq!(classify("output.weight"), (None, GroupKind::Head));
        assert_eq!(classify("rope_freqs.weight"), (None, GroupKind::Other));
    }
}
