//! Model downloads from the Hugging Face Hub, used only by `kestrel setup`
//! after the user picked a model.
//!
//! * The file list, sizes and SHA-256 come from the Hub API
//!   (`/api/models/<repo>/tree/main`), so nothing about a repository is
//!   hard-coded except its name.
//! * Downloads go to `<file>.part` and resume with an HTTP `Range` request.
//! * The SHA-256 is computed while downloading (the existing part is hashed
//!   first on resume) and checked before the file is renamed into place.
//!
//! `HF_ENDPOINT` selects a mirror, as in the Hugging Face tools, and
//! `HF_TOKEN` authenticates for gated repositories. HTTPS proxies come from
//! the usual environment variables; system certificates are trusted.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct HubFile {
    pub repo: String,
    pub path: String,
    pub size: u64,
    /// Lowercase hex SHA-256 from the Hub (LFS files).
    pub sha256: Option<String>,
}

#[derive(Deserialize)]
struct TreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    lfs: Option<Lfs>,
}

#[derive(Deserialize)]
struct Lfs {
    oid: String,
    #[serde(default)]
    size: u64,
}

pub fn endpoint() -> String {
    std::env::var("HF_ENDPOINT").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| "https://huggingface.co".into()).trim_end_matches('/').to_string()
}

fn env(keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|k| std::env::var(k).ok()).filter(|v| !v.is_empty())
}

/// Whether `host` is excluded from proxying by a `NO_PROXY` list: `*`, exact
/// names or addresses, and domain suffixes (`example.com`, `.example.com`).
fn no_proxy_matches(list: &str, host: &str) -> bool {
    let host = host.trim_start_matches('[').trim_end_matches(']').to_ascii_lowercase();
    list.split(',').map(|e| e.trim().to_ascii_lowercase()).filter(|e| !e.is_empty()).any(|e| {
        let e = e.trim_start_matches("*.").trim_start_matches('.');
        e == "*" || host == e || host.ends_with(&format!(".{e}"))
    })
}

/// The proxy for `url` from `HTTPS_PROXY`/`HTTP_PROXY`/`ALL_PROXY`, honouring
/// `NO_PROXY` (which ureq's own environment lookup ignores).
fn proxy_for(url: &str) -> Option<String> {
    let (scheme, rest) = url.split_once("://")?;
    let authority = rest.split('/').next()?;
    let host = match authority.rsplit_once(':') {
        Some((h, p)) if p.chars().all(|c| c.is_ascii_digit()) => h,
        _ => authority,
    };
    if env(&["NO_PROXY", "no_proxy"]).is_some_and(|l| no_proxy_matches(&l, host)) {
        return None;
    }
    if scheme.eq_ignore_ascii_case("https") {
        env(&["HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"])
    } else {
        env(&["HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"])
    }
}

fn agent_for(url: &str) -> Result<ureq::Agent> {
    let mut b = ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(20)).timeout_read(Duration::from_secs(60));
    if let Some(p) = proxy_for(url) {
        b = b.proxy(ureq::Proxy::new(&p).with_context(|| format!("invalid proxy setting '{p}'"))?);
    }
    Ok(b.build())
}

fn get(url: &str) -> Result<ureq::Request> {
    let r = agent_for(url)?.get(url).set("User-Agent", concat!("kestrel/", env!("CARGO_PKG_VERSION")));
    Ok(match std::env::var("HF_TOKEN") {
        Ok(t) if !t.is_empty() => r.set("Authorization", &format!("Bearer {t}")),
        _ => r,
    })
}

fn explain(e: ureq::Error, what: &str) -> anyhow::Error {
    match e {
        ureq::Error::Status(401 | 403, _) => anyhow::anyhow!("{what}: access denied. The repository may be gated: accept its license on huggingface.co and set HF_TOKEN"),
        ureq::Error::Status(404, _) => anyhow::anyhow!("{what}: not found on {}", endpoint()),
        ureq::Error::Status(code, r) => anyhow::anyhow!("{what}: HTTP {code} {}", r.status_text()),
        ureq::Error::Transport(t) => anyhow::anyhow!("{what}: cannot reach {} ({t}). Check the connection, or set HF_ENDPOINT to a mirror", endpoint()),
    }
}

/// Every `.gguf` file at the top of a repository's main branch.
pub fn list_gguf(repo: &str) -> Result<Vec<HubFile>> {
    let url = format!("{}/api/models/{repo}/tree/main", endpoint());
    let entries: Vec<TreeEntry> = get(&url)?.call().map_err(|e| explain(e, &format!("listing {repo}")))?.into_json().context("reading the file list")?;
    Ok(entries
        .into_iter()
        .filter(|e| e.kind == "file" && e.path.to_ascii_lowercase().ends_with(".gguf"))
        .map(|e| HubFile {
            repo: repo.to_string(),
            size: e.lfs.as_ref().map(|l| l.size).filter(|&s| s > 0).unwrap_or(e.size),
            sha256: e.lfs.map(|l| l.oid.to_ascii_lowercase()),
            path: e.path,
        })
        .collect())
}

/// The single-file GGUF of `quant` (e.g. `Q4_K_M`) in a listing. Split
/// files (`-00001-of-00003`) are skipped: Kestrel reads one file per model.
pub fn pick_quant<'a>(files: &'a [HubFile], quant: &str) -> Option<&'a HubFile> {
    let q = quant.to_ascii_lowercase();
    files.iter().filter(|f| !f.path.contains("-of-")).find(|f| {
        let name = f.path.to_ascii_lowercase();
        name.ends_with(&format!("-{q}.gguf")) || name.ends_with(&format!(".{q}.gguf")) || name.ends_with(&format!("_{q}.gguf"))
    })
}

fn part_path(dest: &Path) -> PathBuf {
    let mut p = dest.as_os_str().to_owned();
    p.push(".part");
    PathBuf::from(p)
}

/// Download `f` to `dest`, resuming a previous `.part`. `progress(done, total)`
/// is called as bytes arrive. Returns immediately if `dest` already exists
/// with the right size.
pub fn download(f: &HubFile, dest: &Path, mut progress: impl FnMut(u64, u64)) -> Result<()> {
    if std::fs::metadata(dest).is_ok_and(|m| m.len() == f.size) {
        return Ok(());
    }
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let part = part_path(dest);
    let mut hasher = Sha256::new();
    let mut have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    if have > f.size {
        std::fs::remove_file(&part)?;
        have = 0;
    }
    if have > 0 && f.sha256.is_some() {
        // Resume: the hash must cover the bytes already on disk.
        let mut r = std::fs::File::open(&part)?;
        let mut buf = vec![0u8; 8 << 20];
        let mut left = have;
        while left > 0 {
            let want = left.min(buf.len() as u64) as usize;
            let n = r.read(&mut buf[..want])?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            left -= n as u64;
        }
    }
    if have < f.size {
        let url = format!("{}/{}/resolve/main/{}", endpoint(), f.repo, f.path);
        let mut req = get(&url)?;
        if have > 0 {
            req = req.set("Range", &format!("bytes={have}-"));
        }
        let resp = req.call().map_err(|e| explain(e, &format!("downloading {}", f.path)))?;
        if have > 0 && resp.status() != 206 {
            // The server ignored the range: start over.
            have = 0;
            hasher = Sha256::new();
        }
        let mut out = std::fs::OpenOptions::new().create(true).write(true).append(have > 0).truncate(have == 0).open(&part).with_context(|| format!("writing {}", part.display()))?;
        let mut body = resp.into_reader();
        let mut buf = vec![0u8; 1 << 20];
        let mut last = Instant::now();
        progress(have, f.size);
        loop {
            let n = body.read(&mut buf).with_context(|| format!("downloading {} (run the same command again to resume)", f.path))?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])?;
            hasher.update(&buf[..n]);
            have += n as u64;
            if last.elapsed() > Duration::from_millis(250) {
                progress(have, f.size);
                last = Instant::now();
            }
        }
        out.sync_all()?;
        progress(have, f.size);
    }
    if have != f.size {
        bail!("download of {} stopped at {have} of {} bytes; run the same command again to resume", f.path, f.size);
    }
    if let Some(want) = &f.sha256 {
        let got = format!("{:x}", hasher.finalize());
        if &got != want {
            std::fs::remove_file(&part)?;
            bail!("{}: SHA-256 mismatch (expected {want}, got {got}); the partial file was removed, run again to download it afresh", f.path);
        }
    }
    std::fs::rename(&part, dest)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(p: &str) -> HubFile {
        HubFile { repo: "r".into(), path: p.into(), size: 1, sha256: None }
    }

    #[test]
    fn no_proxy_lists() {
        let l = "localhost,127.0.0.1,.corp.example,internal.net";
        assert!(no_proxy_matches(l, "127.0.0.1"));
        assert!(no_proxy_matches(l, "LOCALHOST"));
        assert!(no_proxy_matches(l, "hub.corp.example"));
        assert!(no_proxy_matches(l, "a.internal.net"));
        assert!(!no_proxy_matches(l, "huggingface.co"));
        assert!(!no_proxy_matches(l, "notinternal.net"));
        assert!(no_proxy_matches("*", "huggingface.co"));
    }

    #[test]
    fn picks_single_file_quant() {
        let files = [f("Model-Q4_K_S.gguf"), f("Model-Q4_K_M-00001-of-00002.gguf"), f("Model-Q4_K_M.gguf"), f("model.q8_0.gguf")];
        assert_eq!(pick_quant(&files, "Q4_K_M").unwrap().path, "Model-Q4_K_M.gguf");
        assert_eq!(pick_quant(&files, "q8_0").unwrap().path, "model.q8_0.gguf");
        assert!(pick_quant(&files, "Q6_K").is_none());
    }
}
