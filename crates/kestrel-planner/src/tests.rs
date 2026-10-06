use super::*;
use kestrel_hw::{CpuInfo, GpuBackend, GpuInfo, MemInfo, StorageInfo};
use kestrel_model::{ArchSupport, Extent, HParams, MoeInfo, TensorGroup};
use std::path::PathBuf;

const GB: u64 = 1_000_000_000;

/// A dense model description with realistic group sizes for a given bpw.
fn dense(n_layer: u32, n_embd: u64, n_ff: u64, n_vocab: u64, n_kv: u32, bpw: f64) -> ModelDesc {
    let b = |n: u64| (n as f64 * bpw / 8.0) as u64;
    let hd = 128u64;
    let n_head = (n_embd / hd) as u32;
    let mut groups = Vec::new();
    let mut push = |kind, layer, bytes, touch| {
        let id = groups.len();
        groups.push(TensorGroup { id, kind, layer, tensors: vec![], extents: vec![Extent { offset: 0, len: bytes }], bytes, touch_per_token: touch });
    };
    push(GroupKind::Embed, None, b(n_vocab * n_embd), 1.0 / n_vocab as f64);
    for l in 0..n_layer {
        push(GroupKind::Attn, Some(l), b(n_embd * n_embd * 2 + 2 * n_embd * n_kv as u64 * hd), 1.0);
        push(GroupKind::Ffn, Some(l), b(3 * n_embd * n_ff), 1.0);
    }
    push(GroupKind::Head, None, b(n_vocab * n_embd), 1.0);
    let n_params = groups.iter().map(|g| (g.bytes as f64 * 8.0 / bpw) as u64).sum();
    ModelDesc {
        path: PathBuf::from("/models/m.gguf"),
        name: "synthetic".into(),
        arch: "qwen2".into(),
        support: ArchSupport::Native,
        file_size: 0,
        fingerprint: String::new(),
        file_type: Some("Q4_K_M".into()),
        hparams: HParams {
            n_layer,
            n_embd: n_embd as u32,
            n_head,
            n_head_kv: vec![n_kv; n_layer as usize],
            head_dim_k: hd as u32,
            head_dim_v: hd as u32,
            n_ff: n_ff as u32,
            n_vocab: n_vocab as u32,
            n_ctx_train: 32768,
            rope_freq_base: 1e6,
            rope_dim: hd as u32,
            rms_eps: 1e-6,
        },
        moe: None,
        n_params,
        tensors: vec![],
        groups,
        bytes_by_type: Default::default(),
        has_chat_template: true,
    }
}

fn qwen32b() -> ModelDesc {
    dense(64, 5120, 27648, 152064, 8, 4.85)
}

fn hw(ram_total: u64, ram_avail: u64, vram: Option<(u64, u64)>) -> HardwareProfile {
    HardwareProfile {
        os: "linux".into(),
        arch: "x86_64".into(),
        cpu: CpuInfo { model: "Test CPU".into(), logical_cores: 16, physical_cores: 8, numa_nodes: 1, simd: vec![], caches: vec![] },
        ram: MemInfo { total: ram_total, available: ram_avail, swap_total: 0, cgroup_limit: None },
        gpus: vram
            .map(|(t, f)| GpuInfo {
                index: 0,
                name: "NVIDIA GeForce RTX 4060".into(),
                vram_total: t,
                vram_free: f,
                backends: vec![GpuBackend::Cuda],
                compute_capability: Some("8.9".into()),
                driver: None,
                pcie_gen: Some(4),
                pcie_width: Some(8),
                mem_bandwidth: Some(272 * GB),
                mem_bandwidth_is_estimate: true,
                pcie_bandwidth: Some(12 * GB),
                unified_memory: false,
            })
            .into_iter()
            .collect(),
        storage: vec![StorageInfo { mount_point: "/".into(), device: None, filesystem: None, capacity: 1000 * GB, available: 500 * GB, kind: "nvme".into(), model: None }],
        measured: None,
        fingerprint: "test".into(),
    }
}

#[test]
fn mvp_target_32b_on_8gb_gpu() {
    let m = qwen32b();
    let h = hw(34 * GB, 28 * GB, Some((8 * GB, 7_600_000_000)));
    let req = PlanRequest { llamacpp_available: true, ..Default::default() };
    let p = plan(&m, &h, &req).expect("feasible");
    assert_eq!(p.chosen.kind, StrategyKind::GpuRamHybrid, "{}", p.text());
    assert_eq!(p.chosen.backend, Backend::LlamaCpp);
    assert_eq!(p.chosen.disk_weights, 0, "fits in VRAM+RAM: no NVMe streaming");
    let ngl = p.chosen.n_gpu_layers.unwrap();
    assert!((10..40).contains(&ngl), "ngl {ngl}");
    assert!(p.chosen.vram_total <= p.budgets.vram.usable + p.budgets.vram.overhead);
    assert!(p.chosen.vram_weights + p.chosen.ram_weights == m.weight_bytes());
    assert!(p.llamacpp_args.contains(&"--n-gpu-layers".to_string()));
    // The hybrid plan must beat CPU-only.
    let cpu = p.alternatives.iter().find(|c| c.kind == StrategyKind::RamOnly && c.backend == Backend::LlamaCpp).unwrap();
    assert!(p.chosen.estimate.tok_s > cpu.estimate.tok_s);
    let txt = p.text();
    eprintln!("{txt}");
    assert!(txt.contains("Hybrid VRAM/RAM") && txt.contains("STRATEGIES"), "{txt}");
}

#[test]
fn native_streams_when_ram_is_short() {
    let m = qwen32b();
    let h = hw(16 * GB, 14 * GB, None);
    let p = plan(&m, &h, &PlanRequest::default()).expect("streaming makes it feasible");
    assert_eq!(p.chosen.kind, StrategyKind::RamNvme);
    assert_eq!(p.chosen.backend, Backend::Native);
    assert!(p.chosen.disk_weights > 0);
    assert!(p.chosen.ram_total <= p.budgets.ram.usable + p.budgets.ram.overhead, "{}", p.text());
    // Embeddings and head stay resident.
    for g in &m.groups {
        if g.layer.is_none() {
            assert_ne!(p.chosen.tiers[g.id], Tier::Disk);
        }
    }
    // Interleaved: streamed groups are spread, not one contiguous block.
    let streamed: Vec<usize> = m.groups.iter().filter(|g| p.chosen.tiers[g.id] == Tier::Disk).map(|g| g.id).collect();
    // Resident groups are interspersed: no long back-to-back streamed runs.
    let mut run = 1;
    let mut max_run = 1;
    for w in streamed.windows(2) {
        run = if w[1] == w[0] + 1 { run + 1 } else { 1 };
        max_run = max_run.max(run);
    }
    assert!(max_run <= 3, "streamed groups should be spread: {streamed:?}");
    assert!(p.warnings.iter().any(|w| w.contains("disk bandwidth")));
}

#[test]
fn contiguous_placement_streams_a_block() {
    let m = qwen32b();
    let h = hw(16 * GB, 14 * GB, None);
    let p = plan(&m, &h, &PlanRequest { placement: PlacementOrder::Contiguous, ..Default::default() }).unwrap();
    let streamed: Vec<usize> = m.groups.iter().filter(|g| p.chosen.tiers[g.id] == Tier::Disk).map(|g| g.id).collect();
    assert_eq!(streamed.last().unwrap() - streamed.first().unwrap() + 1, streamed.len());
}

#[test]
fn no_stream_is_infeasible_with_remedy() {
    let m = qwen32b();
    let h = hw(16 * GB, 14 * GB, None);
    let err = plan(&m, &h, &PlanRequest { no_stream: true, ..Default::default() }).unwrap_err();
    assert!(err.remedies.iter().any(|r| r.contains("NVMe tiering")), "{}", err.text());
    assert!(err.text().contains("Model cannot currently be executed"));
}

#[test]
fn tiny_machine_is_infeasible_and_explains() {
    let m = qwen32b();
    let h = hw(4 * GB, 2 * GB, None);
    let err = plan(&m, &h, &PlanRequest::default()).unwrap_err();
    let t = err.text();
    assert!(t.contains("Possible solutions") && t.contains("Close memory-heavy applications"), "{t}");
    assert!(err.required_ram > err.available_ram);
}

#[test]
fn kv_budget_limits_context() {
    let m = qwen32b();
    let h = hw(64 * GB, 60 * GB, None);
    let p = plan(&m, &h, &PlanRequest { kv_budget: Some(256 << 20), ..Default::default() }).unwrap();
    assert!(p.n_ctx * m.kv_bytes_per_token(GgmlType::F16) <= 256 << 20);
}

#[test]
fn moe_puts_experts_on_cpu() {
    // A 30B-A3B-like MoE: 48 layers, 128 experts, top-8.
    let mut m = dense(48, 2048, 768, 151936, 4, 4.85);
    let bytes_per_expert = (3.0 * 2048.0 * 768.0 * 4.85 / 8.0) as u64;
    let mut extra = Vec::new();
    for l in 0..48 {
        let id = m.groups.len() + extra.len();
        extra.push(TensorGroup { id, kind: GroupKind::Experts, layer: Some(l), tensors: vec![], extents: vec![], bytes: bytes_per_expert * 128, touch_per_token: 8.0 / 128.0 });
    }
    m.groups.extend(extra);
    m.moe = Some(MoeInfo { n_expert: 128, n_expert_used: 8, n_expert_shared: 0, bytes_per_expert, moe_layers: 48, n_ff_exp: 768, norm_topk: true, weights_scale: 0.0 });
    let h = hw(34 * GB, 28 * GB, Some((8 * GB, 7_600_000_000)));
    let p = plan(&m, &h, &PlanRequest { llamacpp_available: true, native_support: Err("moe".into()), ..Default::default() }).unwrap();
    assert!(p.chosen.experts_on_cpu, "{}", p.text());
    assert!(p.llamacpp_args.iter().any(|a| a.contains("_exps")));
    // Every layer's attention is on the GPU.
    assert_eq!(p.chosen.n_gpu_layers, Some(49));
}

#[test]
fn spread_order_is_a_spread_permutation() {
    for n in [1, 2, 7, 64, 100] {
        let o = spread_order(n);
        let mut s = o.clone();
        s.sort();
        assert_eq!(s, (0..n).collect::<Vec<_>>());
    }
    let o = spread_order(8);
    assert_eq!(&o[..4], &[0, 4, 2, 6]);
}

#[test]
fn native_moe_sizes_an_expert_cache() {
    let mut m = dense(24, 2048, 1536, 32000, 4, 4.85);
    // Replace dense FFNs with 32 routed experts per layer, top-4.
    let bpe = (3.0 * 2048.0 * 1536.0 * 4.85 / 8.0) as u64;
    for g in m.groups.iter_mut().filter(|g| g.kind == GroupKind::Ffn) {
        g.bytes = 64 << 10; // router + norm
    }
    let mut extra = Vec::new();
    for l in 0..24 {
        let id = m.groups.len() + extra.len();
        extra.push(TensorGroup { id, kind: GroupKind::Experts, layer: Some(l), tensors: vec![], extents: vec![], bytes: bpe * 32, touch_per_token: 4.0 / 32.0 });
    }
    m.groups.extend(extra);
    m.moe = Some(MoeInfo { n_expert: 32, n_expert_used: 4, n_expert_shared: 0, bytes_per_expert: bpe, moe_layers: 24, n_ff_exp: 1536, norm_topk: true, weights_scale: 0.0 });
    let experts_total = bpe * 32 * 24;

    // Plenty of RAM: every expert cached, nothing streamed.
    let p = plan(&m, &hw(32 * GB, 28 * GB, None), &PlanRequest::default()).unwrap();
    assert_eq!(p.chosen.backend, Backend::Native);
    assert_eq!(p.chosen.expert_cache, Some(32 * 24));
    assert_eq!(p.chosen.disk_weights, 0);
    assert!(p.store.external_experts);

    // Tight RAM: dense part resident, a partial expert cache, the rest on disk.
    let p = plan(&m, &hw(16 * GB, 2_500_000_000 + (1536 << 20), None), &PlanRequest::default()).unwrap();
    let cap = p.chosen.expert_cache.unwrap();
    assert!(cap > 8 && cap < 32 * 24, "cap {cap}\n{}", p.text());
    assert!(p.chosen.disk_weights > 0 && p.chosen.disk_weights < experts_total);
    // Disk reads per token reflect cache misses, not all touched expert bytes.
    let touched = 24.0 * 4.0 * bpe as f64;
    assert!(p.chosen.estimate.disk_bytes_per_token < touched, "{}", p.chosen.estimate.disk_bytes_per_token);
    assert!(p.text().contains("expert cache holds"));
}
