//! Case 2: iterator chains and Vec growth, three regions in one process.
//!   map_sum  (0..n).map(..).sum()          expected: affine, 0 irregular
//!   vec_push Vec::new() + n pushes          expected: mostly affine; the
//!            reallocation copies (capacity doublings) are step-like in n and
//!            should appear as a small irregular share (SPEC N3)
//!   vec_cap  Vec::with_capacity(n) + n pushes   expected: affine, 0 irregular
use perfmark::Region;
use std::hint::black_box;

fn main() {
    let st = perfmark::states(&[("n", 1000)]);
    let n = st["n"];
    let s = {
        let _r = Region::new("map_sum", "n", n);
        (0..n as u64).map(|i| black_box(i).wrapping_mul(3)).sum::<u64>()
    };
    let v1 = {
        let _r = Region::new("vec_push", "n", n);
        let mut v: Vec<u64> = Vec::new();
        for i in 0..n as u64 {
            v.push(black_box(i));
        }
        v
    };
    let v2 = {
        let _r = Region::new("vec_cap", "n", n);
        let mut v: Vec<u64> = Vec::with_capacity(n as usize);
        for i in 0..n as u64 {
            v.push(black_box(i));
        }
        v
    };
    println!("iters n={} s={} v1={} v2={}", n, s, v1.len(), v2.len());
}
