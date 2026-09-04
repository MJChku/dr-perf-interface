//! Case 7 (head): same as delta_base plus a second n-iteration loop (the
//! "delta").  Expected: derived A rises by exactly the added loop's
//! instructions per iteration; D by the loop's setup.
use perfmark::Region;
use std::hint::black_box;

fn main() {
    let st = perfmark::states(&[("n", 1000)]);
    let n = st["n"];
    let mut acc = 0u64;
    let mut acc2 = 0u64;
    {
        let _r = Region::new("work", "n", n);
        for i in 0..n as u64 {
            acc = black_box(acc).wrapping_add(i);
        }
        for i in 0..n as u64 {
            acc2 = black_box(acc2) ^ i.wrapping_mul(3);
        }
    }
    println!("head n={} acc={} acc2={}", n, acc, acc2);
}
