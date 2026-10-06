//! Hardware discovery and measurement.
//!
//! Discovery (capacities, topology, devices) is cheap and runs on every
//! launch. Measurement (bandwidths) costs seconds and is cached per machine
//! fingerprint; see [`bench`]. Absence of a GPU, of `/sys`, or of a tool such
//! as `nvidia-smi` is a normal result, never an error.

pub mod bench;
mod cpu;
pub mod fileio;
mod gpu;
mod mem;
mod storage;

pub use cpu::CpuInfo;
pub use gpu::{GpuBackend, GpuInfo};
pub use mem::{process_peak_rss, process_rss, MemInfo};
pub use storage::StorageInfo;

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub os: String,
    pub arch: String,
    pub cpu: CpuInfo,
    pub ram: MemInfo,
    pub gpus: Vec<GpuInfo>,
    pub storage: Vec<StorageInfo>,
    /// Measured bandwidths; `None` until `kestrel hardware --bench` (or an
    /// automatic first-run benchmark) has filled them.
    pub measured: Option<bench::Measurements>,
    pub fingerprint: String,
}

impl HardwareProfile {
    /// Discover the machine. `storage_paths` are directories or files whose
    /// backing devices should be described (typically the model's location).
    pub fn discover(storage_paths: &[PathBuf]) -> Self {
        let cpu = cpu::discover();
        let ram = mem::discover();
        let gpus = gpu::discover();
        let mut storage = Vec::new();
        for p in storage_paths {
            if let Some(s) = storage::describe(p) {
                if !storage.iter().any(|x: &StorageInfo| x.mount_point == s.mount_point) {
                    storage.push(s);
                }
            }
        }
        let fingerprint = fingerprint(&cpu, &ram, &gpus);
        HardwareProfile {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            cpu,
            ram,
            gpus,
            storage,
            measured: None,
            fingerprint,
        }
    }

    pub fn storage_for(&self, path: &Path) -> Option<&StorageInfo> {
        let p = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        self.storage
            .iter()
            .filter(|s| p.starts_with(&s.mount_point))
            .max_by_key(|s| s.mount_point.as_os_str().len())
    }

    /// Attach cached measurements if they belong to this machine.
    pub fn load_cached_measurements(&mut self, cache: &Path) {
        if let Ok(text) = std::fs::read_to_string(cache) {
            if let Ok(m) = serde_json::from_str::<bench::CachedMeasurements>(&text) {
                if m.fingerprint == self.fingerprint {
                    self.measured = Some(m.measurements);
                }
            }
        }
    }

    pub fn save_measurements(&self, cache: &Path) -> std::io::Result<()> {
        if let Some(m) = &self.measured {
            if let Some(dir) = cache.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let c = bench::CachedMeasurements { fingerprint: self.fingerprint.clone(), measurements: m.clone() };
            let tmp = cache.with_extension("tmp");
            std::fs::write(&tmp, serde_json::to_string_pretty(&c).unwrap())?;
            std::fs::rename(tmp, cache)?;
        }
        Ok(())
    }
}

fn fingerprint(cpu: &CpuInfo, ram: &MemInfo, gpus: &[GpuInfo]) -> String {
    let mut s = format!("{}|{}|{}|{}", cpu.model, cpu.logical_cores, cpu.physical_cores, ram.total / (1 << 30));
    for g in gpus {
        s.push_str(&format!("|{}|{}", g.name, g.vram_total / (1 << 30)));
    }
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

/// `~/.cache/kestrel` (or `%LOCALAPPDATA%\kestrel`), overridable with `KESTREL_CACHE`.
pub fn cache_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("KESTREL_CACHE") {
        return PathBuf::from(d);
    }
    #[cfg(windows)]
    if let Some(d) = std::env::var_os("LOCALAPPDATA") {
        return PathBuf::from(d).join("kestrel");
    }
    if let Some(d) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(d).join("kestrel");
    }
    if let Some(h) = std::env::var_os("HOME") {
        return PathBuf::from(h).join(".cache").join("kestrel");
    }
    std::env::temp_dir().join("kestrel")
}

/// Run a command and capture stdout if it exits successfully.
pub(crate) fn run_capture(cmd: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(cmd).args(args).stderr(std::process::Stdio::null()).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

pub fn fmt_bytes(b: u64) -> String {
    let b = b as f64;
    if b >= 1e12 {
        format!("{:.2} TB", b / 1e12)
    } else if b >= 1e9 {
        format!("{:.2} GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.1} MB", b / 1e6)
    } else if b >= 1e3 {
        format!("{:.1} KB", b / 1e3)
    } else {
        format!("{b} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discover_is_sane() {
        let p = HardwareProfile::discover(&[std::env::temp_dir()]);
        assert!(p.cpu.logical_cores >= 1);
        assert!(p.cpu.physical_cores >= 1 && p.cpu.physical_cores <= p.cpu.logical_cores);
        assert!(p.ram.total > 0);
        assert!(p.ram.available <= p.ram.total);
        assert_eq!(p.fingerprint.len(), 16);
        let j = serde_json::to_string(&p).unwrap();
        let _: HardwareProfile = serde_json::from_str(&j).unwrap();
    }
}
