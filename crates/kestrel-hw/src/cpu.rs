use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CpuInfo {
    pub model: String,
    pub logical_cores: usize,
    /// Physical cores. Memory-bound quantized GEMV regresses when SMT siblings
    /// compete for memory channels, so compute thread pools size from this.
    pub physical_cores: usize,
    pub numa_nodes: usize,
    pub simd: Vec<String>,
    /// (level, kind, bytes) per cache, from CPU 0.
    pub caches: Vec<CacheLevel>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CacheLevel {
    pub level: u32,
    pub kind: String,
    pub size: u64,
}

pub fn discover() -> CpuInfo {
    let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let mut info = CpuInfo {
        model: String::from("unknown"),
        logical_cores: logical,
        physical_cores: logical,
        numa_nodes: 1,
        simd: simd_features(),
        caches: Vec::new(),
    };
    #[cfg(target_os = "linux")]
    linux(&mut info);
    #[cfg(windows)]
    {
        if let Ok(id) = std::env::var("PROCESSOR_IDENTIFIER") {
            info.model = id;
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Some(s) = crate::run_capture("sysctl", &["-n", "machdep.cpu.brand_string"]) {
            info.model = s.trim().to_string();
        }
        if let Some(s) = crate::run_capture("sysctl", &["-n", "hw.physicalcpu"]) {
            if let Ok(n) = s.trim().parse() {
                info.physical_cores = n;
            }
        }
    }
    info.physical_cores = info.physical_cores.clamp(1, info.logical_cores);
    info
}

#[cfg(target_os = "linux")]
fn linux(info: &mut CpuInfo) {
    use std::collections::BTreeSet;
    use std::fs;
    if let Ok(s) = fs::read_to_string("/proc/cpuinfo") {
        for line in s.lines() {
            let mut kv = line.splitn(2, ':');
            let k = kv.next().unwrap_or("").trim();
            let v = kv.next().unwrap_or("").trim();
            if (k == "model name" || k == "Model" || k == "cpu model") && !v.is_empty() {
                info.model = v.to_string();
                break;
            }
        }
    }
    // Physical cores = distinct (package, core) pairs among online CPUs.
    let mut cores = BTreeSet::new();
    if let Ok(rd) = fs::read_dir("/sys/devices/system/cpu") {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if !name.starts_with("cpu") || !name[3..].chars().all(|c| c.is_ascii_digit()) {
                continue;
            }
            let t = e.path().join("topology");
            let pkg = fs::read_to_string(t.join("physical_package_id")).unwrap_or_default();
            let core = fs::read_to_string(t.join("core_id")).unwrap_or_default();
            if !core.is_empty() {
                cores.insert((pkg.trim().to_string(), core.trim().to_string()));
            }
        }
    }
    if !cores.is_empty() {
        info.physical_cores = cores.len();
    }
    if let Ok(rd) = fs::read_dir("/sys/devices/system/node") {
        let n = rd
            .flatten()
            .filter(|e| {
                let n = e.file_name().to_string_lossy().to_string();
                n.starts_with("node") && n[4..].chars().all(|c| c.is_ascii_digit())
            })
            .count();
        if n > 0 {
            info.numa_nodes = n;
        }
    }
    for i in 0..8 {
        let d = format!("/sys/devices/system/cpu/cpu0/cache/index{i}");
        let Ok(level) = fs::read_to_string(format!("{d}/level")) else { break };
        let kind = fs::read_to_string(format!("{d}/type")).unwrap_or_default();
        let size = fs::read_to_string(format!("{d}/size")).unwrap_or_default();
        let size = size.trim();
        let bytes = if let Some(k) = size.strip_suffix('K') {
            k.parse::<u64>().unwrap_or(0) << 10
        } else if let Some(m) = size.strip_suffix('M') {
            m.parse::<u64>().unwrap_or(0) << 20
        } else {
            size.parse().unwrap_or(0)
        };
        info.caches.push(CacheLevel { level: level.trim().parse().unwrap_or(0), kind: kind.trim().to_string(), size: bytes });
    }
}

fn simd_features() -> Vec<String> {
    #[allow(unused_mut)]
    let mut v: Vec<String> = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        macro_rules! f {
            ($($name:tt),*) => {$(
                if std::is_x86_feature_detected!($name) { v.push($name.to_string()); }
            )*};
        }
        f!("sse4.2", "avx", "avx2", "fma", "f16c", "avx512f", "avx512bw", "avx512vnni", "avxvnni");
    }
    #[cfg(target_arch = "aarch64")]
    {
        v.push("neon".to_string());
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            v.push("dotprod".to_string());
        }
        if std::arch::is_aarch64_feature_detected!("i8mm") {
            v.push("i8mm".to_string());
        }
        if std::arch::is_aarch64_feature_detected!("sve") {
            v.push("sve".to_string());
        }
    }
    v
}
