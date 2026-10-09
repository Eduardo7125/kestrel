//! Text handling: Obsidian Markdown cleanup, sectioning, passage splitting,
//! and search terms.

/// Lower-case and strip the diacritics of Latin letters (á → a, ñ → n, ç → c).
pub fn fold(s: &str) -> String {
    s.chars()
        .flat_map(|c| c.to_lowercase())
        .map(|c| match c {
            'á' | 'à' | 'ä' | 'â' | 'ã' | 'å' => 'a',
            'é' | 'è' | 'ë' | 'ê' => 'e',
            'í' | 'ì' | 'ï' | 'î' => 'i',
            'ó' | 'ò' | 'ö' | 'ô' | 'õ' => 'o',
            'ú' | 'ù' | 'ü' | 'û' => 'u',
            'ñ' => 'n',
            'ç' => 'c',
            other => other,
        })
        .collect()
}

/// Words that carry no meaning for search (English and Spanish).
const STOP: &[&str] = &[
    "the", "and", "for", "are", "but", "not", "you", "your", "with", "this", "that", "from", "have", "has", "was", "were", "what", "which", "who", "how", "why", "when", "where", "can", "could", "would", "should", "about", "into", "than", "then", "them", "they", "their", "there", "these", "those", "its", "our", "out", "all", "any", "some", "will", "just", "does", "did", "been", "being", "also", "more", "most", "very", "her", "his", "she", "him", "my", "me", "is", "it", "in", "on", "of", "to", "a", "an", "as", "at", "be", "by", "do", "if", "or", "so", "we", "no",
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al", "y", "o", "u", "en", "que", "es", "son", "por", "para", "con", "sin", "se", "su", "sus", "lo", "le", "les", "mi", "mis", "tu", "tus", "me", "te", "nos", "como", "mas", "pero", "este", "esta", "estos", "estas", "ese", "esa", "eso", "esto", "hay", "ha", "he", "han", "fue", "era", "ser", "esta", "estan", "muy", "ya", "cual", "cuales", "donde", "cuando", "quien", "porque", "sobre", "entre", "hasta", "desde", "todo", "toda", "todos", "todas", "tiene", "tengo", "puedo", "puede", "hacer",
];

/// Search terms of a text: folded words of 2+ characters, without stop
/// words, with a trailing plural `s` removed from longer words so "notas"
/// matches "nota" and "projects" matches "project".
pub fn terms(s: &str) -> Vec<String> {
    fold(s)
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 2 && !STOP.contains(w))
        .map(|w| {
            if w.chars().count() > 4 && w.ends_with('s') && !w.ends_with("ss") {
                w[..w.len() - 1].to_string()
            } else {
                w.to_string()
            }
        })
        .collect()
}

/// Obsidian Markdown to plain text: drop front matter, comments and embeds,
/// and turn wiki-links into their visible text. Returns the front matter's
/// tags and aliases separately (they describe the whole note).
pub fn clean_markdown(src: &str) -> (String, String) {
    let mut body = src;
    let mut meta = String::new();
    if let Some(rest) = src.strip_prefix("---\n").or_else(|| src.strip_prefix("---\r\n")) {
        if let Some(end) = rest.find("\n---") {
            // Keep `tags` and `aliases` (inline or as a list); drop the rest.
            let mut in_list = false;
            for line in rest[..end].lines() {
                let l = line.trim();
                let lower = l.to_ascii_lowercase();
                if let Some(v) = ["tags:", "aliases:"].iter().find_map(|k| lower.starts_with(k).then(|| &l[k.len()..])) {
                    meta.push_str(v.trim_matches(|c: char| c == '[' || c == ']' || c.is_whitespace()));
                    meta.push(' ');
                    in_list = true;
                } else if in_list && l.starts_with("- ") {
                    meta.push_str(&l[2..]);
                    meta.push(' ');
                } else if !l.is_empty() {
                    in_list = false;
                }
            }
            body = rest[end + 4..].trim_start_matches(['-', '\r', '\n']);
        }
    }
    let mut out = String::with_capacity(body.len());
    let mut rest = body;
    while !rest.is_empty() {
        if let Some(r) = rest.strip_prefix("%%") {
            // %% comment %%
            match r.find("%%") {
                Some(end) => rest = &r[end + 2..],
                None => rest = "",
            }
        } else if let Some(r) = rest.strip_prefix("![[") {
            // Embedded file or note: skip it.
            match r.find("]]") {
                Some(end) => rest = &r[end + 2..],
                None => rest = "",
            }
        } else if let Some(r) = rest.strip_prefix("[[") {
            match r.find("]]") {
                Some(end) => {
                    let link = &r[..end];
                    let shown = match link.split_once('|') {
                        Some((_, alias)) => alias.to_string(),
                        None => link.replace('#', " "),
                    };
                    out.push_str(&shown);
                    rest = &r[end + 2..];
                }
                None => {
                    out.push_str("[[");
                    rest = r;
                }
            }
        } else {
            let c = rest.chars().next().unwrap();
            out.push(c);
            rest = &rest[c.len_utf8()..];
        }
    }
    (meta.trim().to_string(), out)
}

/// Split Markdown into (heading path, text) sections at `#` headings.
/// Headings inside fenced code blocks are ignored.
pub fn sections(md: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut path: Vec<(usize, String)> = Vec::new();
    let mut cur = String::new();
    let mut fenced = false;
    let heading_of = |path: &[(usize, String)]| path.iter().map(|(_, h)| h.as_str()).collect::<Vec<_>>().join(" › ");
    for line in md.lines() {
        if line.trim_start().starts_with("```") {
            fenced = !fenced;
        }
        let level = line.chars().take_while(|&c| c == '#').count();
        if !fenced && (1..=6).contains(&level) && line[level..].starts_with(' ') {
            if !cur.trim().is_empty() {
                out.push((heading_of(&path), std::mem::take(&mut cur)));
            }
            cur.clear();
            path.retain(|(l, _)| *l < level);
            path.push((level, line[level..].trim().to_string()));
            continue;
        }
        cur.push_str(line);
        cur.push('\n');
    }
    if !cur.trim().is_empty() {
        out.push((heading_of(&path), cur));
    }
    out
}

/// Split text into passages of about `target` characters, at paragraph
/// breaks where possible, then at line or sentence ends.
pub fn split(text: &str, target: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for para in text.split("\n\n") {
        let para = para.trim();
        if para.is_empty() {
            continue;
        }
        if !cur.is_empty() && cur.len() + para.len() > target {
            out.push(std::mem::take(&mut cur));
        }
        if para.len() > target * 2 {
            // A very long paragraph: cut it at sentence or line ends.
            let mut piece = String::new();
            for sent in para.split_inclusive(['.', '\n', '!', '?']) {
                if !piece.is_empty() && piece.len() + sent.len() > target {
                    out.push(std::mem::take(&mut piece));
                }
                piece.push_str(sent);
                while piece.len() > target * 2 {
                    let mut cut = target;
                    while !piece.is_char_boundary(cut) {
                        cut -= 1;
                    }
                    out.push(piece[..cut].to_string());
                    piece = piece[cut..].to_string();
                }
            }
            if !piece.trim().is_empty() {
                out.push(piece);
            }
            continue;
        }
        if !cur.is_empty() {
            cur.push_str("\n\n");
        }
        cur.push_str(para);
    }
    if !cur.trim().is_empty() {
        out.push(cur);
    }
    out.into_iter().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
}
