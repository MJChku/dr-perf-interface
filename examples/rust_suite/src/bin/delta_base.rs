//! Case 7 (base): one loop of n iterations in region `work`.
//! head (delta_head.rs) adds a second loop of n iterations.  Expected: the
//! derived A of head minus base equals the instruction count of the added
//! loop's block, and `drperf diff` shows the same total difference per n.
use perfmark::Region;
use std::hint::black_box;

fn main() {
    let st = perfmark::states(&[("n", 1000)]);
    let n = st["n"];
    let mut acc = 0u64;
    {
        let _r = Region::new("work", "n", n);
        for i in 0..n as u64 {
            acc = black_box(acc).wrapping_add(i);
        }
    }
    println!("base n={} acc={}", n, acc);
}
