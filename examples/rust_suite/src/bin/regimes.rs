//! Case 5: two code paths chosen by n < 1000.  Small n: one cheap loop.
//! Large n: two loops with more work per iteration.  Expected: derive over
//! n in {200..800} U {1500..4000} splits into two regimes with different
//! (A, D) and a small irregular share in each.
use perfmark::Region;
use std::hint::black_box;

fn main() {
    let st = perfmark::states(&[("n", 500)]);
    let n = st["n"];
    let mut acc = 0u64;
    {
        let _r = Region::new("regime", "n", n);
        if n < 1000 {
            for i in 0..n as u64 {
                acc = black_box(acc).wrapping_add(i);
            }
        } else {
            for i in 0..n as u64 {
                acc = black_box(acc).wrapping_mul(6364136223846793005).wrapping_add(i);
            }
            for i in 0..n as u64 {
                acc ^= black_box(i).rotate_left(7);
            }
        }
    }
    println!("regimes n={} acc={}", n, acc);
}
