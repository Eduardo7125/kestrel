//! The llama.cpp backend: Kestrel decides placement, llama.cpp executes.
//!
//! The plan's `llamacpp_args` already encode the decision (`--n-gpu-layers`,
//! `--override-tensor` for MoE experts, context, KV type, threads). This
//! module finds the binaries and launches them.

use anyhow::{bail, Result};
use std::path::PathBuf;
use std::process::{Child, Command};

/// Locate a llama.cpp tool: `$KESTREL_<NAME>` (e.g. `KESTREL_LLAMA_SERVER`),
/// then `$KESTREL_LLAMA_CPP/build/bin/<name>`, then `PATH`.
pub fn find_tool(name: &str) -> Option<PathBuf> {
    let env_key = format!("KESTREL_{}", name.to_uppercase().replace('-', "_"));
    if let Some(p) = std::env::var_os(&env_key) {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    let exe = if cfg!(windows) { format!("{name}.exe") } else { name.to_string() };
    if let Some(root) = std::env::var_os("KESTREL_LLAMA_CPP") {
        let p = PathBuf::from(root).join("build").join("bin").join(&exe);
        if p.is_file() {
            return Some(p);
        }
    }
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|d| d.join(&exe)).find(|p| p.is_file())
}

pub fn server_available() -> bool {
    find_tool("llama-server").is_some()
}

/// Launch `llama-server` with the plan's placement on `host:port`.
pub fn spawn_server(args: &[String], host: &str, port: u16, extra: &[String]) -> Result<Child> {
    let Some(bin) = find_tool("llama-server") else {
        bail!("llama-server not found: set KESTREL_LLAMA_SERVER, KESTREL_LLAMA_CPP, or add it to PATH");
    };
    let mut cmd = Command::new(bin);
    cmd.args(args).arg("--host").arg(host).arg("--port").arg(port.to_string()).args(extra);
    Ok(cmd.spawn()?)
}

/// Build the `llama-bench` command line equivalent to a plan (for the
/// benchmark harness's llama.cpp arms).
pub fn bench_args(plan_args: &[String], n_prompt: usize, n_gen: usize, reps: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut it = plan_args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" => out.extend(["-m".to_string(), it.next().cloned().unwrap_or_default()]),
            "--n-gpu-layers" => out.extend(["-ngl".to_string(), it.next().cloned().unwrap_or_default()]),
            "--threads" => out.extend(["-t".to_string(), it.next().cloned().unwrap_or_default()]),
            "--override-tensor" => out.extend(["-ot".to_string(), it.next().cloned().unwrap_or_default()]),
            "--cache-type-k" => out.extend(["-ctk".to_string(), it.next().cloned().unwrap_or_default()]),
            "--cache-type-v" => out.extend(["-ctv".to_string(), it.next().cloned().unwrap_or_default()]),
            "--ctx-size" => {
                it.next();
            }
            _ => {}
        }
    }
    out.extend(["-p".into(), n_prompt.to_string(), "-n".into(), n_gen.to_string(), "-r".into(), reps.to_string(), "-o".into(), "json".into()]);
    out
}
