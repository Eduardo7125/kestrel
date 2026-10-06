use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StorageInfo {
    pub mount_point: PathBuf,
    pub device: Option<String>,
    pub filesystem: Option<String>,
    pub capacity: u64,
    pub available: u64,
    /// "nvme", "ssd", "hdd", "virtual" or "unknown".
    pub kind: String,
    pub model: Option<String>,
}

pub fn describe(path: &Path) -> Option<StorageInfo> {
    let p = std::fs::canonicalize(path).ok()?;
    let (capacity, available) = space(&p)?;
    let mut info = StorageInfo {
        mount_point: PathBuf::from("/"),
        device: None,
        filesystem: None,
        capacity,
        available,
        kind: "unknown".into(),
        model: None,
    };
    #[cfg(target_os = "linux")]
    linux(&p, &mut info);
    #[cfg(windows)]
    {
        info.mount_point = p.components().take(1).collect();
    }
    Some(info)
}

#[cfg(unix)]
fn space(p: &Path) -> Option<(u64, u64)> {
    use std::os::unix::ffi::OsStrExt;
    let c = std::ffi::CString::new(p.as_os_str().as_bytes()).ok()?;
    let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut s) } != 0 {
        return None;
    }
    let f = s.f_frsize as u64;
    Some((s.f_blocks as u64 * f, s.f_bavail as u64 * f))
}

#[cfg(windows)]
fn space(p: &Path) -> Option<(u64, u64)> {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "kernel32")]
    extern "system" {
        fn GetDiskFreeSpaceExW(dir: *const u16, avail: *mut u64, total: *mut u64, free: *mut u64) -> i32;
    }
    let w: Vec<u16> = p.as_os_str().encode_wide().chain(Some(0)).collect();
    let (mut a, mut t, mut f) = (0u64, 0u64, 0u64);
    (unsafe { GetDiskFreeSpaceExW(w.as_ptr(), &mut a, &mut t, &mut f) } != 0).then_some((t, a))
}

#[cfg(not(any(unix, windows)))]
fn space(_p: &Path) -> Option<(u64, u64)> {
    None
}

#[cfg(target_os = "linux")]
fn linux(p: &Path, info: &mut StorageInfo) {
    // Longest mount point that prefixes the path.
    if let Ok(s) = std::fs::read_to_string("/proc/self/mounts") {
        let mut best: Option<(PathBuf, String, String)> = None;
        for line in s.lines() {
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() < 3 {
                continue;
            }
            let mp = PathBuf::from(f[1].replace("\\040", " "));
            if p.starts_with(&mp) && best.as_ref().map_or(true, |b| mp.as_os_str().len() > b.0.as_os_str().len()) {
                best = Some((mp, f[0].to_string(), f[2].to_string()));
            }
        }
        if let Some((mp, dev, fs)) = best {
            info.mount_point = mp;
            info.filesystem = Some(fs.clone());
            info.device = Some(dev.clone());
            if let Some(name) = dev.strip_prefix("/dev/") {
                classify_block_device(name, info);
            } else if matches!(fs.as_str(), "overlay" | "tmpfs" | "9p" | "virtiofs" | "fuse") {
                info.kind = "virtual".into();
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn classify_block_device(name: &str, info: &mut StorageInfo) {
    // Map a partition (nvme0n1p2, sda1) to its parent disk via sysfs.
    let class = Path::new("/sys/class/block").join(name);
    let disk = match std::fs::canonicalize(&class) {
        Ok(real) if real.join("partition").exists() => {
            real.parent().and_then(|d| d.file_name()).map(|s| s.to_string_lossy().to_string()).unwrap_or(name.to_string())
        }
        _ => name.to_string(),
    };
    let base = Path::new("/sys/block").join(&disk);
    let rot = std::fs::read_to_string(base.join("queue/rotational")).ok();
    info.model = std::fs::read_to_string(base.join("device/model")).ok().map(|s| s.trim().to_string());
    info.kind = if disk.starts_with("nvme") {
        "nvme".into()
    } else if disk.starts_with("vd") || disk.starts_with("xvd") {
        // virtio disks report rotational=1 regardless of the backing device.
        "virtual".into()
    } else if rot.as_deref().map(str::trim) == Some("0") {
        "ssd".into()
    } else if rot.as_deref().map(str::trim) == Some("1") {
        "hdd".into()
    } else {
        "unknown".into()
    };
}
