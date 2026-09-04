//! Case 8: symbol names.  A generic, non-inlined function monomorphized for
//! u32 and u16, called under region `generic`.  Expected: derive's
//! per-function attribution shows demangled Rust names for both instances
//! (or one merged name if the hash-stripped names coincide); no garbage.
use perfmark::Region;
use std::hint::black_box;

#[inline(never)]
fn checksum<T: Copy + Into<u64>>(xs: &[T]) -> u64 {
    let mut acc = 0u64;
    for &x in xs {
        acc = black_box(acc).wrapping_mul(31).wrapping_add(x.into());
    }
    acc
}

#[inline(never)]
fn fold_pairs(xs: &[u32]) -> u64 {
    xs.chunks(2).map(|c| black_box(c[0] as u64) * 3 + c.get(1).copied().unwrap_or(0) as u64).sum()
}

fn main() {
    let st = perfmark::states(&[("n", 1000)]);
    let n = st["n"];
    let a: Vec<u32> = (0..n as u32).collect();
    let b: Vec<u16> = (0..n as u16).collect();
    let (c1, c2, c3) = {
        let _r = Region::new("generic", "n", n);
        (checksum(&a), checksum(&b), fold_pairs(&a))
    };
    println!("generics n={} {} {} {}", n, c1, c2, c3);
}
