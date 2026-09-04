//! Case 1 / 9: exact affine cost.  `outer` runs a loop of n iterations with
//! constant work per iteration, then a nested region `inner` with another
//! n-iteration loop.  Expected (SPEC P1): derive gives a*n + d exactly for
//! `inner`, and for `outer` the inclusive formula (own loop + inner) with
//! 0 irregular blocks; repeated runs are byte-identical in their counts.
use perfmark::Region;
use std::hint::black_box;

fn main() {
    let st = perfmark::states(&[("n", 1000)]);
    let n = st["n"];
    let mut acc = 0u64;
    let mut acc2 = 0u64;
    {
        let _o = Region::new("outer", "n", n);
        for i in 0..n as u64 {
            acc = black_box(acc).wrapping_add(i ^ 0x5bd1_e995);
        }
        {
            let _i = Region::new("inner", "n", n);
            for i in 0..n as u64 {
                acc2 = black_box(acc2).wrapping_mul(3).wrapping_add(i);
            }
        }
    }
    println!("affine n={} acc={} acc2={}", n, acc, acc2);
}
