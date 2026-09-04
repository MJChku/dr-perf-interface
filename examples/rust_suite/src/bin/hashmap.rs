//! Case 3: HashMap<u64,u64> with n inserts (SipHash, table resizes).
//! Expected: hashing and probing are per-key constant work (affine blocks);
//! resizes (rehash of the whole table at power-of-two thresholds) are
//! step-like in n and come out irregular (SPEC N3).  Report the shares.
use perfmark::Region;
use std::collections::HashMap;
use std::hint::black_box;

fn main() {
    let st = perfmark::states(&[("n", 1000)]);
    let n = st["n"];
    let m = {
        let _r = Region::new("hm_insert", "n", n);
        let mut m: HashMap<u64, u64> = HashMap::new();
        for i in 0..n as u64 {
            m.insert(black_box(i).wrapping_mul(0x9E37_79B9_7F4A_7C15), i);
        }
        m
    };
    let hits = {
        let _r = Region::new("hm_lookup", "n", n);
        let mut h = 0u64;
        for i in 0..n as u64 {
            if m.contains_key(&black_box(i).wrapping_mul(0x9E37_79B9_7F4A_7C15)) {
                h += 1;
            }
        }
        h
    };
    println!("hashmap n={} len={} hits={}", n, m.len(), hits);
}
