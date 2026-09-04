//! Delta example (Rust): bind a batch of keys against a directory.
//! Base: plain lookups.  With `--features delta`: an added validation pass
//! over the whole directory (the "delta"), annotated as its own region.
use perfmark::Region;
use std::collections::HashMap;

fn build_directory(n: usize) -> HashMap<u64, u64> {
    (0..n as u64).map(|k| (k, (k * 7919) % 4096)).collect()
}

#[cfg(feature = "delta")]
fn validate(directory: &HashMap<u64, u64>) -> u64 {
    let _r = Region::new("validate", "n_entries", directory.len() as i64);
    let mut checksum = 0u64;
    for (k, v) in directory {
        assert!(*v < 4096, "superblock out of range");
        checksum = checksum.wrapping_mul(31).wrapping_add(k ^ v);
    }
    checksum
}

fn bind_batch(directory: &HashMap<u64, u64>, keys: &[u64]) -> Vec<u64> {
    #[cfg(feature = "delta")]
    let _checksum = validate(directory);
    keys.iter().map(|k| directory[k]).collect()
}

fn main() {
    let st = perfmark::states(&[("n", 1000)]);
    let n = st["n"] as usize;
    let directory = build_directory(n);
    let mut x = 12345u64;
    let keys: Vec<u64> = (0..256)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x % n as u64
        })
        .collect();
    let warm = bind_batch(&directory, &keys);
    let out = {
        let _r = Region::new("bind_batch", "n_entries", n as i64);
        bind_batch(&directory, &keys)
    };
    let variant = if cfg!(feature = "delta") { "head" } else { "base" };
    println!("{} n={} sum={} warm={}", variant, n, out.iter().sum::<u64>(), warm.len());
}
