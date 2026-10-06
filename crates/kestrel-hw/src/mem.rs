use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemInfo {
    pub total: u64,
    /// Memory the OS can hand out without swapping (Linux `MemAvailable`,
    /// Windows `ullAvailPhys`).
    pub available: u64,
    pub swap_total: u64,
    /// cgroup memory limit when running in a container, if lower than total.
    pub cgroup_limit: Option<u64>,
}

impl MemInfo {
    /// Available memory, also respecting a container limit.
    pub fn effective_available(&self) -> u64 {
        match self.cgroup_limit {
            Some(l) => self.available.min(l.saturating_sub(process_rss().unwrap_or(0))),
            None => self.available,
        }
    }
}

pub fn discover() -> MemInfo {
    let mut m = MemInfo { total: 0, available: 0, swap_total: 0, cgroup_limit: None };
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
            for line in s.lines() {
                let mut it = line.split_whitespace();
                let k = it.next().unwrap_or("");
                let v: u64 = it.next().and_then(|x| x.parse().ok()).unwrap_or(0) * 1024;
                match k {
                    "MemTotal:" => m.total = v,
                    "MemAvailable:" => m.available = v,
                    "SwapTotal:" => m.swap_total = v,
                    _ => {}
                }
            }
        }
        for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"] {
            if let Ok(s) = std::fs::read_to_string(p) {
                if let Ok(v) = s.trim().parse::<u64>() {
                    if v < m.total {
                        m.cgroup_limit = Some(v);
                    }
                }
                break;
            }
        }
    }
    #[cfg(windows)]
    windows(&mut m);
    #[cfg(target_os = "macos")]
    {
        if let Some(s) = crate::run_capture("sysctl", &["-n", "hw.memsize"]) {
            m.total = s.trim().parse().unwrap_or(0);
        }
        // Reclaimable = free + inactive + speculative + purgeable pages.
        if let Some(out) = crate::run_capture("vm_stat", &[]) {
            let page: u64 = out
                .split("page size of ")
                .nth(1)
                .and_then(|s| s.split_whitespace().next())
                .and_then(|s| s.parse().ok())
                .unwrap_or(16384);
            let mut pages = 0u64;
            for line in out.lines() {
                for key in ["Pages free:", "Pages inactive:", "Pages speculative:", "Pages purgeable:"] {
                    if let Some(rest) = line.strip_prefix(key) {
                        pages += rest.trim().trim_end_matches('.').parse::<u64>().unwrap_or(0);
                    }
                }
            }
            m.available = pages * page;
        }
    }
    if m.available == 0 || m.available > m.total {
        m.available = m.total / 2;
    }
    m
}

#[cfg(windows)]
fn windows(m: &mut MemInfo) {
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GlobalMemoryStatusEx(buf: *mut MemoryStatusEx) -> i32;
    }
    let mut s: MemoryStatusEx = unsafe { std::mem::zeroed() };
    s.length = std::mem::size_of::<MemoryStatusEx>() as u32;
    if unsafe { GlobalMemoryStatusEx(&mut s) } != 0 {
        m.total = s.total_phys;
        m.available = s.avail_phys;
        m.swap_total = s.total_page_file.saturating_sub(s.total_phys);
    }
}

/// Resident set size of this process in bytes.
pub fn process_rss() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let s = std::fs::read_to_string("/proc/self/statm").ok()?;
        let pages: u64 = s.split_whitespace().nth(1)?.parse().ok()?;
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
        Some(pages * page)
    }
    #[cfg(windows)]
    {
        #[repr(C)]
        struct Pmc {
            cb: u32,
            page_fault_count: u32,
            peak_working_set: usize,
            working_set: usize,
            quota_peak_paged: usize,
            quota_paged: usize,
            quota_peak_nonpaged: usize,
            quota_nonpaged: usize,
            pagefile: usize,
            peak_pagefile: usize,
        }
        #[link(name = "psapi")]
        extern "system" {
            fn GetProcessMemoryInfo(h: isize, p: *mut Pmc, cb: u32) -> i32;
        }
        #[link(name = "kernel32")]
        extern "system" {
            fn GetCurrentProcess() -> isize;
        }
        let mut p: Pmc = unsafe { std::mem::zeroed() };
        p.cb = std::mem::size_of::<Pmc>() as u32;
        if unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut p, p.cb) } != 0 {
            return Some(p.working_set as u64);
        }
        None
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        None
    }
}

/// Peak RSS (high-water mark) of this process.
pub fn process_peak_rss() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let s = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in s.lines() {
            if let Some(v) = line.strip_prefix("VmHWM:") {
                return v.trim().trim_end_matches("kB").trim().parse::<u64>().ok().map(|k| k * 1024);
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        process_rss()
    }
}
