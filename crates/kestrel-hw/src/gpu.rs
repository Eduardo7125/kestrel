//! GPU discovery. NVIDIA GPUs are queried through `nvidia-smi`, which ships
//! with every driver; other backends are detected by the presence of their
//! runtime libraries or tools. Memory bandwidth cannot be measured without a
//! compute API, so it comes from a small table of known parts and is marked
//! as an estimate.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GpuBackend {
    Cuda,
    Hip,
    Vulkan,
    Metal,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuInfo {
    pub index: u32,
    pub name: String,
    pub vram_total: u64,
    pub vram_free: u64,
    pub backends: Vec<GpuBackend>,
    pub compute_capability: Option<String>,
    pub driver: Option<String>,
    pub pcie_gen: Option<u32>,
    pub pcie_width: Option<u32>,
    /// Device memory bandwidth in bytes/s, and whether it is an estimate.
    pub mem_bandwidth: Option<u64>,
    pub mem_bandwidth_is_estimate: bool,
    /// Host↔device bandwidth estimate in bytes/s, derived from the PCIe link.
    pub pcie_bandwidth: Option<u64>,
    pub unified_memory: bool,
}

pub fn discover() -> Vec<GpuInfo> {
    let mut gpus = nvidia();
    let vulkan = has_vulkan();
    if vulkan {
        for g in &mut gpus {
            g.backends.push(GpuBackend::Vulkan);
        }
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        // Apple Silicon: one unified pool; VRAM is a share of system RAM.
        let ram = crate::mem::discover();
        gpus.push(GpuInfo {
            index: 0,
            name: "Apple Silicon GPU".into(),
            vram_total: ram.total * 3 / 4,
            vram_free: ram.available,
            backends: vec![GpuBackend::Metal],
            compute_capability: None,
            driver: None,
            pcie_gen: None,
            pcie_width: None,
            mem_bandwidth: None,
            mem_bandwidth_is_estimate: true,
            pcie_bandwidth: None,
            unified_memory: true,
        });
    }
    gpus
}

fn nvidia() -> Vec<GpuInfo> {
    let Some(out) = crate::run_capture(
        "nvidia-smi",
        &[
            "--query-gpu=index,name,memory.total,memory.free,compute_cap,driver_version,pcie.link.gen.max,pcie.link.width.max",
            "--format=csv,noheader,nounits",
        ],
    ) else {
        return Vec::new();
    };
    parse_nvidia_smi(&out)
}

pub(crate) fn parse_nvidia_smi(out: &str) -> Vec<GpuInfo> {
    let mut v = Vec::new();
    for line in out.lines() {
        let f: Vec<&str> = line.split(',').map(str::trim).collect();
        if f.len() < 4 {
            continue;
        }
        let num = |s: &str| s.parse::<u64>().ok();
        let mib = |s: &str| num(s).map(|x| x << 20).unwrap_or(0);
        let name = f[1].to_string();
        let pcie_gen = f.get(6).and_then(|s| s.parse().ok());
        let pcie_width = f.get(7).and_then(|s| s.parse().ok());
        let pcie_bandwidth = match (pcie_gen, pcie_width) {
            // Per-lane usable GB/s after encoding: gen3 ~0.985, gen4 ~1.97, gen5 ~3.94.
            (Some(g), Some(w)) if (1..=6).contains(&g) => {
                let per_lane = [0.25, 0.5, 0.985, 1.97, 3.94, 7.88][g as usize - 1];
                Some((per_lane * w as f64 * 1e9 * 0.8) as u64) // ~80% achievable
            }
            _ => None,
        };
        let mem_bw = known_bandwidth(&name);
        v.push(GpuInfo {
            index: num(f[0]).unwrap_or(0) as u32,
            vram_total: mib(f[2]),
            vram_free: mib(f[3]),
            backends: vec![GpuBackend::Cuda],
            compute_capability: f.get(4).map(|s| s.to_string()).filter(|s| !s.is_empty() && *s != "[N/A]"),
            driver: f.get(5).map(|s| s.to_string()),
            pcie_gen,
            pcie_width,
            mem_bandwidth: Some(mem_bw.unwrap_or(300_000_000_000)),
            mem_bandwidth_is_estimate: true,
            pcie_bandwidth,
            unified_memory: false,
            name,
        });
    }
    v
}

/// Vendor-published memory bandwidth for common consumer parts (bytes/s).
/// Unknown parts fall back to a conservative 300 GB/s, flagged as an estimate.
fn known_bandwidth(name: &str) -> Option<u64> {
    const TABLE: &[(&str, u64)] = &[
        ("5090", 1792),
        ("5080", 960),
        ("5070 Ti", 896),
        ("5070", 672),
        ("5060 Ti", 448),
        ("5060", 448),
        ("4090", 1008),
        ("4080", 717),
        ("4070 Ti", 504),
        ("4070", 504),
        ("4060 Ti", 288),
        ("4060", 272),
        ("3090", 936),
        ("3080", 760),
        ("3070", 448),
        ("3060", 360),
        ("A100", 1555),
        ("H100", 3350),
        ("L4", 300),
        ("T4", 320),
    ];
    TABLE.iter().find(|(k, _)| name.contains(k)).map(|(_, gbs)| gbs * 1_000_000_000)
}

fn has_vulkan() -> bool {
    #[cfg(target_os = "linux")]
    {
        ["/usr/lib/x86_64-linux-gnu/libvulkan.so.1", "/usr/lib64/libvulkan.so.1", "/usr/lib/libvulkan.so.1", "/usr/lib/aarch64-linux-gnu/libvulkan.so.1"]
            .iter()
            .any(|p| std::path::Path::new(p).exists())
    }
    #[cfg(windows)]
    {
        std::path::Path::new(r"C:\Windows\System32\vulkan-1.dll").exists()
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn parse() {
        let out = "0, NVIDIA GeForce RTX 4060, 8188, 7590, 8.9, 555.42, 4, 8\n";
        let g = super::parse_nvidia_smi(out);
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].vram_total, 8188 << 20);
        assert_eq!(g[0].mem_bandwidth, Some(272_000_000_000));
        assert!(g[0].pcie_bandwidth.unwrap() > 10_000_000_000);
    }
}
