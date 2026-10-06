//! Time promotions (disk → RAM) of streamed groups: `cargo run --release
//! --example promote -p kestrel-memory -- model.gguf [n_groups]`.
use kestrel_memory::{Ledger, StoreConfig, WeightStore};
use kestrel_model::ModelDesc;
use std::sync::Arc;

fn main() {
    let path = std::env::args().nth(1).expect("model.gguf");
    let n: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let m = Arc::new(ModelDesc::open(&path).unwrap());
    let groups = m.groups.len();
    let ledger = Ledger::new(0, u64::MAX / 4, 0);
    let store = WeightStore::new(m.clone(), &vec![false; groups], (0..groups).collect(), StoreConfig::default(), ledger).unwrap();
    for g in m.groups.iter().filter(|g| g.layer.is_some()).take(n) {
        let t = std::time::Instant::now();
        store.promote(g.id).unwrap();
        let s = t.elapsed().as_secs_f64();
        println!("group {:>3} {:>8.1} MB  {:>7.1} ms  {:>7.0} MB/s", g.id, g.bytes as f64 / 1e6, s * 1e3, g.bytes as f64 / 1e6 / s);
    }
}
