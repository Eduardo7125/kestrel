//! Architecture adapters: what Kestrel knows about each GGUF architecture.
//!
//! Inspection and planning work for any architecture whose tensors follow the
//! `blk.N.*` naming convention. Native execution is opt-in per architecture,
//! which is how new architectures are added without touching the scheduler.

use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArchSupport {
    /// Runs on the native executor (and on llama.cpp).
    Native,
    /// Planned by Kestrel, executed through the llama.cpp backend only.
    LlamaCppOnly,
}

#[derive(Clone, Copy, Debug)]
pub struct ArchAdapter {
    pub arch: &'static str,
    pub support: ArchSupport,
    pub moe: bool,
    pub description: &'static str,
}

const ADAPTERS: &[ArchAdapter] = &[
    ArchAdapter { arch: "llama", support: ArchSupport::Native, moe: false, description: "Llama 2/3, Mistral, Mixtral-style dense; GQA + RoPE + SwiGLU" },
    ArchAdapter { arch: "qwen2", support: ArchSupport::Native, moe: false, description: "Qwen2/2.5: llama + QKV bias" },
    ArchAdapter { arch: "qwen3", support: ArchSupport::Native, moe: false, description: "Qwen3 dense: llama + per-head Q/K RMSNorm" },
    ArchAdapter { arch: "qwen2moe", support: ArchSupport::LlamaCppOnly, moe: true, description: "Qwen2-MoE: routed + shared experts" },
    ArchAdapter { arch: "qwen3moe", support: ArchSupport::LlamaCppOnly, moe: true, description: "Qwen3-MoE: routed experts" },
    ArchAdapter { arch: "deepseek2", support: ArchSupport::LlamaCppOnly, moe: true, description: "DeepSeek V2/V3: MLA + routed/shared experts" },
    ArchAdapter { arch: "gemma2", support: ArchSupport::LlamaCppOnly, moe: false, description: "Gemma 2" },
    ArchAdapter { arch: "gemma3", support: ArchSupport::LlamaCppOnly, moe: false, description: "Gemma 3" },
    ArchAdapter { arch: "phi3", support: ArchSupport::LlamaCppOnly, moe: false, description: "Phi-3" },
];

pub fn adapter_for(arch: &str) -> ArchAdapter {
    ADAPTERS.iter().copied().find(|a| a.arch == arch).unwrap_or(ArchAdapter {
        arch: "unknown",
        support: ArchSupport::LlamaCppOnly,
        moe: false,
        description: "unrecognized architecture: planned generically from tensor names",
    })
}

pub fn all() -> &'static [ArchAdapter] {
    ADAPTERS
}
