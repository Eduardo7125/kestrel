//! Local knowledge for retrieval-augmented answers: an Obsidian vault, any
//! folder of text files, and files uploaded from the dashboard.
//!
//! Notes are split into passages along their headings and indexed with
//! BM25. Before a chat request, the server takes the best passages for the
//! user's message and gives them to the model with the question, so answers
//! can draw on the notes and cite them. Nothing leaves the machine and no
//! extra model is needed: BM25 is lexical search, which works the same in
//! any language with spaces between words.
//!
//! The index is rebuilt from disk at most once per [`RESCAN_INTERVAL`] when a
//! file was added, removed or modified, so edits made in Obsidian show up in
//! the next answers without a manual sync.

mod text;

pub use text::{fold, terms};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

/// How often a search re-checks the sources for changed files.
pub const RESCAN_INTERVAL: Duration = Duration::from_secs(20);
/// Files larger than this are skipped (logs, exports, binaries renamed .txt).
pub const MAX_FILE_BYTES: u64 = 2 << 20;
/// Largest file the dashboard may upload.
pub const MAX_UPLOAD_BYTES: usize = 5 << 20;
/// Target passage length, in characters.
const PASSAGE_CHARS: usize = 900;

const TEXT_EXTENSIONS: &[&str] = &[
    "md", "markdown", "txt", "text", "org", "rst", "adoc", "tex", "csv", "tsv", "json", "jsonl", "yaml", "yml", "toml", "ini", "log", "html", "htm", "xml", "rs", "py", "js", "ts", "tsx", "jsx", "java", "kt", "go", "c", "h", "cpp", "hpp", "cs", "rb", "php", "sh", "ps1", "sql", "swift", "lua",
];
/// Folders never indexed: Obsidian's settings and trash, VCS, dependencies.
const SKIP_DIRS: &[&str] = &[".obsidian", ".trash", ".git", ".svn", "node_modules", "target", ".venv", "__pycache__"];

#[derive(Debug, thiserror::Error)]
pub enum KnowledgeError {
    #[error("{0}")]
    Invalid(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, KnowledgeError>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceKind {
    /// An Obsidian vault: wiki-links and front matter are understood, and
    /// results link back with `obsidian://` URIs.
    Obsidian,
    /// Any folder of text files.
    Folder,
    /// Files uploaded from the dashboard.
    Uploads,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Source {
    pub id: String,
    pub kind: SourceKind,
    pub name: String,
    pub path: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Settings {
    /// Consult the knowledge before every chat request.
    pub enabled: bool,
    /// Passages given to the model per request.
    pub top_k: usize,
    /// Upper bound for the passages' text, in characters (≈ 4 per token).
    pub max_context_chars: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { enabled: true, top_k: 4, max_context_chars: 3000 }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Config {
    sources: Vec<Source>,
    settings: Settings,
}

/// One retrieved passage.
#[derive(Clone, Debug, Serialize)]
pub struct Hit {
    pub source: String,
    pub kind: SourceKind,
    /// Path inside the source, with `/` separators.
    pub path: String,
    pub title: String,
    /// Heading path of the passage ("Project › Goals"), empty at the top.
    pub heading: String,
    pub text: String,
    pub score: f32,
    /// `obsidian://open?...` for notes in an Obsidian vault.
    pub uri: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SourceStats {
    #[serde(flatten)]
    pub source: Source,
    pub files: usize,
    pub passages: usize,
    pub bytes: u64,
    /// Why the source could not be read, if it could not.
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Status {
    pub settings: Settings,
    pub sources: Vec<SourceStats>,
    pub files: usize,
    pub passages: usize,
    /// Seconds since the index was last rebuilt.
    pub indexed_ago_s: Option<f64>,
    /// Files uploaded from the dashboard.
    pub uploads: Vec<String>,
}

struct Doc {
    source: usize,
    path: String,
    title: String,
}

struct Passage {
    doc: usize,
    heading: String,
    text: String,
    /// Term frequencies (title and heading terms count double).
    tf: HashMap<u32, u32>,
    len: u32,
}

#[derive(Default)]
struct Index {
    docs: Vec<Doc>,
    passages: Vec<Passage>,
    vocab: HashMap<String, u32>,
    /// term → passages containing it.
    postings: HashMap<u32, Vec<u32>>,
    avg_len: f32,
    /// (source, path) → (modified, size): to notice changes.
    stamps: HashMap<(usize, String), (SystemTime, u64)>,
    stats: Vec<SourceStats>,
    built: Option<Instant>,
}

pub struct KnowledgeBase {
    dir: PathBuf,
    config: Config,
    /// Modification time of config.json when last read or written, to pick
    /// up changes another process (`kestrel knowledge add`) made.
    config_stamp: Option<SystemTime>,
    index: Index,
    last_check: Option<Instant>,
}

fn load_config(dir: &Path) -> Config {
    let mut config: Config = std::fs::read_to_string(dir.join("config.json")).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
    if !config.sources.iter().any(|s| s.kind == SourceKind::Uploads) {
        config.sources.insert(0, Source { id: "uploads".into(), kind: SourceKind::Uploads, name: "Uploaded files".into(), path: dir.join("uploads") });
    }
    config
}

fn config_stamp(dir: &Path) -> Option<SystemTime> {
    std::fs::metadata(dir.join("config.json")).and_then(|m| m.modified()).ok()
}

impl KnowledgeBase {
    /// Open the knowledge base kept in `dir` (config and uploads), creating
    /// it on first use, and index its sources.
    pub fn open(dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(dir.join("uploads"))?;
        let config = load_config(&dir);
        let config_stamp = config_stamp(&dir);
        let mut kb = KnowledgeBase { dir, config, config_stamp, index: Index::default(), last_check: None };
        kb.rebuild();
        Ok(kb)
    }

    fn save(&mut self) -> Result<()> {
        let tmp = self.dir.join("config.json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(&self.config).unwrap_or_default())?;
        std::fs::rename(tmp, self.dir.join("config.json"))?;
        self.config_stamp = config_stamp(&self.dir);
        Ok(())
    }

    pub fn settings(&self) -> &Settings {
        &self.config.settings
    }

    pub fn set_settings(&mut self, s: Settings) -> Result<()> {
        if s.top_k == 0 || s.top_k > 20 {
            return Err(KnowledgeError::Invalid("passages per request must be between 1 and 20".into()));
        }
        if !(500..=20_000).contains(&s.max_context_chars) {
            return Err(KnowledgeError::Invalid("the context size must be between 500 and 20000 characters".into()));
        }
        self.config.settings = s;
        self.save()
    }

    /// Connect a folder. It is treated as an Obsidian vault when it has an
    /// `.obsidian` folder (unless `kind` says otherwise).
    pub fn add_source(&mut self, path: &Path, kind: Option<SourceKind>) -> Result<Source> {
        let path = std::fs::canonicalize(path).map_err(|_| KnowledgeError::Invalid(format!("{} does not exist", path.display())))?;
        if !path.is_dir() {
            return Err(KnowledgeError::Invalid(format!("{} is not a folder", path.display())));
        }
        if self.config.sources.iter().any(|s| s.path == path) {
            return Err(KnowledgeError::Invalid(format!("{} is already connected", path.display())));
        }
        let kind = match kind {
            Some(SourceKind::Uploads) => return Err(KnowledgeError::Invalid("uploads are managed by Kestrel".into())),
            Some(k) => k,
            None if path.join(".obsidian").is_dir() => SourceKind::Obsidian,
            None => SourceKind::Folder,
        };
        let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| path.display().to_string());
        let id = format!("{:x}", fnv(path.to_string_lossy().as_bytes()));
        let src = Source { id, kind, name, path };
        self.config.sources.push(src.clone());
        self.save()?;
        self.rebuild();
        Ok(src)
    }

    pub fn remove_source(&mut self, id: &str) -> Result<()> {
        let i = self.config.sources.iter().position(|s| s.id == id).ok_or_else(|| KnowledgeError::Invalid(format!("no source {id}")))?;
        if self.config.sources[i].kind == SourceKind::Uploads {
            return Err(KnowledgeError::Invalid("the uploads folder cannot be removed; delete its files instead".into()));
        }
        self.config.sources.remove(i);
        self.save()?;
        self.rebuild();
        Ok(())
    }

    /// Store an uploaded text file and index it. Returns the stored name.
    pub fn save_upload(&mut self, name: &str, content: &str) -> Result<String> {
        if content.len() > MAX_UPLOAD_BYTES {
            return Err(KnowledgeError::Invalid(format!("{name} is larger than {} MB", MAX_UPLOAD_BYTES >> 20)));
        }
        let safe = sanitize_name(name).ok_or_else(|| KnowledgeError::Invalid(format!("'{name}' is not a supported text file ({})", TEXT_EXTENSIONS.join(", "))))?;
        std::fs::write(self.dir.join("uploads").join(&safe), content)?;
        self.rebuild();
        Ok(safe)
    }

    pub fn delete_upload(&mut self, name: &str) -> Result<()> {
        let safe = sanitize_name(name).ok_or_else(|| KnowledgeError::Invalid(format!("no upload named {name}")))?;
        std::fs::remove_file(self.dir.join("uploads").join(safe))?;
        self.rebuild();
        Ok(())
    }

    pub fn status(&mut self) -> Status {
        self.refresh_if_changed();
        Status {
            settings: self.config.settings.clone(),
            sources: self.index.stats.clone(),
            files: self.index.docs.len(),
            passages: self.index.passages.len(),
            indexed_ago_s: self.index.built.map(|t| t.elapsed().as_secs_f64()),
            uploads: {
                let mut v: Vec<String> = std::fs::read_dir(self.dir.join("uploads")).map(|rd| rd.flatten().map(|e| e.file_name().to_string_lossy().to_string()).collect()).unwrap_or_default();
                v.sort();
                v
            },
        }
    }

    /// Re-read every source now.
    pub fn sync(&mut self) {
        self.rebuild();
    }

    /// Rebuild when files changed, checking at most every [`RESCAN_INTERVAL`].
    fn refresh_if_changed(&mut self) {
        if self.last_check.is_some_and(|t| t.elapsed() < RESCAN_INTERVAL) {
            return;
        }
        self.last_check = Some(Instant::now());
        if config_stamp(&self.dir) != self.config_stamp {
            self.config = load_config(&self.dir);
            self.config_stamp = config_stamp(&self.dir);
            self.rebuild();
            return;
        }
        let mut seen = 0usize;
        let mut changed = false;
        for (si, src) in self.config.sources.iter().enumerate() {
            walk(&src.path, &mut |rel, meta| {
                seen += 1;
                let stamp = (meta.modified().unwrap_or(SystemTime::UNIX_EPOCH), meta.len());
                if self.index.stamps.get(&(si, rel.to_string())) != Some(&stamp) {
                    changed = true;
                }
            });
        }
        if changed || seen != self.index.stamps.len() {
            self.rebuild();
        }
    }

    fn rebuild(&mut self) {
        let mut ix = Index::default();
        for (si, src) in self.config.sources.iter().enumerate() {
            let mut st = SourceStats { source: src.clone(), files: 0, passages: 0, bytes: 0, error: None };
            if !src.path.is_dir() {
                st.error = Some(format!("{} is not reachable", src.path.display()));
                ix.stats.push(st);
                continue;
            }
            let mut files = Vec::new();
            walk(&src.path, &mut |rel, meta| files.push((rel.to_string(), meta.modified().unwrap_or(SystemTime::UNIX_EPOCH), meta.len())));
            files.sort();
            for (rel, modified, size) in files {
                ix.stamps.insert((si, rel.clone()), (modified, size));
                let Ok(raw) = std::fs::read(src.path.join(&rel)) else { continue };
                let raw = String::from_utf8_lossy(&raw);
                let title = Path::new(&rel).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| rel.clone());
                let is_md = rel.ends_with(".md") || rel.ends_with(".markdown");
                let (tags, body) = if is_md { text::clean_markdown(&raw) } else { (String::new(), raw.into_owned()) };
                let sections = if is_md { text::sections(&body) } else { vec![(String::new(), body)] };
                let doc = ix.docs.len();
                ix.docs.push(Doc { source: si, path: rel, title: title.clone() });
                st.files += 1;
                st.bytes += size;
                for (heading, sec) in sections {
                    for piece in text::split(&sec, PASSAGE_CHARS) {
                        let mut tf: HashMap<u32, u32> = HashMap::new();
                        let mut len = 0u32;
                        let boosted = format!("{title} {heading} {tags}");
                        for (t, w) in terms(&piece).into_iter().map(|t| (t, 1)).chain(terms(&boosted).into_iter().map(|t| (t, 2))) {
                            let n = ix.vocab.len() as u32;
                            let id = *ix.vocab.entry(t).or_insert(n);
                            *tf.entry(id).or_insert(0) += w;
                            len += w;
                        }
                        if len == 0 {
                            continue;
                        }
                        let pid = ix.passages.len() as u32;
                        for &id in tf.keys() {
                            ix.postings.entry(id).or_default().push(pid);
                        }
                        ix.passages.push(Passage { doc, heading: heading.clone(), text: piece, tf, len });
                        st.passages += 1;
                    }
                }
            }
            ix.stats.push(st);
        }
        ix.avg_len = if ix.passages.is_empty() { 1.0 } else { ix.passages.iter().map(|p| p.len as f32).sum::<f32>() / ix.passages.len() as f32 };
        ix.built = Some(Instant::now());
        self.index = ix;
        self.last_check = Some(Instant::now());
    }

    /// The best `k` passages for `query` (BM25, k1 = 1.2, b = 0.75), at most
    /// two per note so one long note cannot crowd out the rest.
    pub fn search(&mut self, query: &str, k: usize) -> Vec<Hit> {
        self.refresh_if_changed();
        let ix = &self.index;
        let n = ix.passages.len() as f32;
        let mut scores: HashMap<u32, f32> = HashMap::new();
        let mut qterms = terms(query);
        qterms.sort();
        qterms.dedup();
        for t in &qterms {
            let Some(&id) = ix.vocab.get(t) else { continue };
            let posting = &ix.postings[&id];
            let idf = ((n - posting.len() as f32 + 0.5) / (posting.len() as f32 + 0.5) + 1.0).ln();
            for &pid in posting {
                let p = &ix.passages[pid as usize];
                let f = p.tf[&id] as f32;
                let s = idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * p.len as f32 / ix.avg_len));
                *scores.entry(pid).or_insert(0.0) += s;
            }
        }
        let mut ranked: Vec<(u32, f32)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut per_doc: HashMap<usize, usize> = HashMap::new();
        let mut hits = Vec::new();
        for (pid, score) in ranked {
            let p = &ix.passages[pid as usize];
            let c = per_doc.entry(p.doc).or_insert(0);
            if *c >= 2 {
                continue;
            }
            *c += 1;
            let d = &ix.docs[p.doc];
            let src = &self.config.sources[d.source];
            hits.push(Hit {
                source: src.name.clone(),
                kind: src.kind,
                path: d.path.clone(),
                title: d.title.clone(),
                heading: p.heading.clone(),
                text: p.text.clone(),
                score,
                uri: (src.kind == SourceKind::Obsidian).then(|| obsidian_uri(&src.name, &d.path)),
            });
            if hits.len() == k {
                break;
            }
        }
        hits
    }

    /// Passages for `query` within the configured budget, and the text to
    /// give the model. None when nothing relevant was found.
    pub fn context_for(&mut self, query: &str) -> Option<(String, Vec<Hit>)> {
        let Settings { top_k, max_context_chars, .. } = self.config.settings.clone();
        let mut hits = self.search(query, top_k);
        let mut used = 0usize;
        hits.retain(|h| {
            let take = used + h.text.len() <= max_context_chars || used == 0;
            if take {
                used += h.text.len();
            }
            take
        });
        if hits.is_empty() {
            return None;
        }
        let mut s = String::from(
            "Notes from the user's own knowledge base that may be relevant to their message. \
             Use them when they help, cite a note as [[Title]] when you rely on it, and say so \
             when the notes do not contain the answer. Do not invent note contents.\n",
        );
        for h in &hits {
            let head = if h.heading.is_empty() { String::new() } else { format!(" › {}", h.heading) };
            let text: String = h.text.chars().take(max_context_chars).collect();
            s.push_str(&format!("\n--- [[{}]]{head} ({})\n{}\n", h.title, h.path, text.trim()));
        }
        Some((s, hits))
    }
}

/// `obsidian://open?vault=…&file=…` for a note (path without `.md`).
pub fn obsidian_uri(vault: &str, path: &str) -> String {
    let file = path.strip_suffix(".md").unwrap_or(path);
    format!("obsidian://open?vault={}&file={}", url_encode(vault), url_encode(file))
}

fn url_encode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

fn fnv(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf29ce484222325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100000001b3))
}

/// A plain file name with a supported text extension, or None.
fn sanitize_name(name: &str) -> Option<String> {
    let base = name.rsplit(['/', '\\']).next()?.trim();
    let clean: String = base.chars().map(|c| if c.is_alphanumeric() || " ._-()".contains(c) { c } else { '_' }).collect();
    let clean = clean.trim_start_matches('.').trim().to_string();
    let ext = Path::new(&clean).extension()?.to_string_lossy().to_ascii_lowercase();
    (!clean.is_empty() && clean.len() <= 160 && TEXT_EXTENSIONS.contains(&ext.as_str())).then_some(clean)
}

/// Every indexable file under `root`, as (relative path with `/`, metadata).
/// Symbolic links are not followed, so a source cannot reach outside itself.
fn walk(root: &Path, f: &mut dyn FnMut(&str, &std::fs::Metadata)) {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for e in rd.flatten() {
            let Ok(meta) = e.path().symlink_metadata() else { continue };
            let name = e.file_name().to_string_lossy().to_string();
            if meta.is_dir() {
                if !SKIP_DIRS.contains(&name.as_str()) && !name.starts_with('.') {
                    stack.push(e.path());
                }
            } else if meta.is_file() && meta.len() <= MAX_FILE_BYTES {
                let ext = Path::new(&name).extension().map(|x| x.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
                if TEXT_EXTENSIONS.contains(&ext.as_str()) {
                    if let Ok(rel) = e.path().strip_prefix(root) {
                        let rel = rel.to_string_lossy().replace('\\', "/");
                        f(&rel, &meta);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
