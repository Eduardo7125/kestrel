use super::*;
use std::fs;

fn vault(dir: &Path) -> PathBuf {
    let v = dir.join("MiVault");
    fs::create_dir_all(v.join(".obsidian")).unwrap();
    fs::create_dir_all(v.join("Proyectos")).unwrap();
    fs::create_dir_all(v.join(".trash")).unwrap();
    fs::write(
        v.join("Proyectos/Kestrel.md"),
        "---\ntags: [rust, ia]\naliases:\n  - runtime\nstatus: draft\n---\n# Kestrel\nUn runtime de inferencia local.\n\n## Objetivos\nLeer los expertos del disco y planificar la memoria.\n\n```\n# esto no es un encabezado\n```\n",
    )
    .unwrap();
    fs::write(v.join("Recetas.md"), "# Tortilla de patatas\nHuevos, patatas, cebolla y aceite de oliva. Ver [[Proyectos/Kestrel#Objetivos|mis objetivos]] y ![[foto.png]].\n%% nota privada %%\n").unwrap();
    fs::write(v.join(".trash/Borrada.md"), "patatas patatas patatas").unwrap();
    fs::write(v.join(".obsidian/workspace.json"), "{\"patatas\": 1}").unwrap();
    v
}

#[test]
fn markdown_cleanup_and_sections() {
    let (tags, md) = text::clean_markdown("---\ntags: [a, b]\n---\nVer [[Nota|alias]] y [[Otra#Parte]] ![[img.png]] %%oculto%% fin");
    assert_eq!(tags, "a, b");
    assert!(!md.contains("tags"), "{md}");
    assert!(md.contains("Ver alias y Otra Parte"), "{md}");
    assert!(!md.contains("img.png") && !md.contains("oculto"), "{md}");
    let secs = text::sections("# A\nuno\n## B\ndos\n```\n# no\n```\n# C\ntres\n");
    let heads: Vec<&str> = secs.iter().map(|(h, _)| h.as_str()).collect();
    assert_eq!(heads, ["A", "A › B", "C"]);
    assert!(secs[1].1.contains("# no"));
}

#[test]
fn terms_fold_accents_plurals_and_stop_words() {
    assert_eq!(terms("Las Notas de mi Canción"), ["nota", "cancion"]);
    assert_eq!(terms("The projects and Ñandú"), ["project", "nandu"]);
}

#[test]
fn indexes_an_obsidian_vault_and_finds_notes() {
    let d = tempfile::tempdir().unwrap();
    let v = vault(d.path());
    let mut kb = KnowledgeBase::open(d.path().join("kb")).unwrap();
    let src = kb.add_source(&v, None).unwrap();
    assert_eq!(src.kind, SourceKind::Obsidian);
    let st = kb.status();
    let vs = st.sources.iter().find(|s| s.source.kind == SourceKind::Obsidian).unwrap();
    assert_eq!(vs.files, 2, "trash and settings are skipped");

    let hits = kb.search("¿qué lleva la tortilla de patata?", 3);
    assert_eq!(hits[0].title, "Recetas");
    assert_eq!(hits[0].heading, "Tortilla de patatas");
    assert_eq!(hits[0].uri.as_deref(), Some("obsidian://open?vault=MiVault&file=Recetas"));

    let hits = kb.search("objetivos expertos disco", 3);
    assert_eq!(hits[0].path, "Proyectos/Kestrel.md");
    assert_eq!(hits[0].heading, "Kestrel › Objetivos");
    assert_eq!(hits[0].uri.as_deref(), Some("obsidian://open?vault=MiVault&file=Proyectos%2FKestrel"));
    // Tags and aliases from the front matter are searchable.
    assert_eq!(kb.search("runtime rust", 1)[0].title, "Kestrel");
    assert!(kb.search("zzzz", 3).is_empty());
}

#[test]
fn context_respects_the_budget_and_cites_titles() {
    let d = tempfile::tempdir().unwrap();
    let v = vault(d.path());
    let mut kb = KnowledgeBase::open(d.path().join("kb")).unwrap();
    kb.add_source(&v, None).unwrap();
    kb.set_settings(Settings { enabled: true, top_k: 4, max_context_chars: 500 }).unwrap();
    let (ctx, hits) = kb.context_for("tortilla patatas objetivos").unwrap();
    assert!(ctx.contains("[[Recetas]]"));
    assert!(hits.iter().map(|h| h.text.len()).sum::<usize>() <= 500);
    assert!(kb.context_for("zzzz").is_none());
}

#[test]
fn edits_are_picked_up_and_sources_persist() {
    let d = tempfile::tempdir().unwrap();
    let v = vault(d.path());
    let mut kb = KnowledgeBase::open(d.path().join("kb")).unwrap();
    kb.add_source(&v, None).unwrap();
    fs::write(v.join("Viajes.md"), "# Japón\nKioto en otoño.").unwrap();
    // Changes are noticed on the next check; force it instead of waiting.
    kb.last_check = None;
    assert_eq!(kb.search("kioto", 1)[0].title, "Viajes");
    // A second open of the same folder remembers the vault.
    let mut kb2 = KnowledgeBase::open(d.path().join("kb")).unwrap();
    assert_eq!(kb2.search("kioto", 1)[0].title, "Viajes");
    assert!(kb2.add_source(&v, None).is_err(), "no duplicates");
    let id = kb2.status().sources.iter().find(|s| s.source.kind == SourceKind::Obsidian).unwrap().source.id.clone();
    kb2.remove_source(&id).unwrap();
    assert!(kb2.search("kioto", 1).is_empty());
    // The first instance notices the other one's change on its next check
    // (a running server, after `kestrel knowledge remove`).
    std::thread::sleep(std::time::Duration::from_millis(20));
    kb.last_check = None;
    assert!(kb.search("kioto", 1).is_empty());
    assert!(kb2.remove_source("uploads").is_err());
}

#[test]
fn uploads_are_sanitized_and_indexed() {
    let d = tempfile::tempdir().unwrap();
    let mut kb = KnowledgeBase::open(d.path().join("kb")).unwrap();
    let name = kb.save_upload("../../etc/Contrato alquiler.md", "El alquiler vence el 30 de junio.").unwrap();
    assert_eq!(name, "Contrato alquiler.md");
    assert!(d.path().join("kb/uploads/Contrato alquiler.md").is_file());
    assert_eq!(kb.search("cuando vence el alquiler", 1)[0].title, "Contrato alquiler");
    assert!(kb.save_upload("virus.exe", "x").is_err());
    assert!(kb.save_upload(".md", "x").is_err());
    kb.delete_upload("Contrato alquiler.md").unwrap();
    assert!(kb.search("alquiler", 1).is_empty());
}

#[test]
fn long_text_is_split_into_passages() {
    let long = "Frase de prueba. ".repeat(400);
    let parts = text::split(&long, 900);
    assert!(parts.len() > 3);
    assert!(parts.iter().all(|p| p.len() <= 1800));
}
