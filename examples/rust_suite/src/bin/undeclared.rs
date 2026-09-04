//! Case 4 (SPEC N2): the region declares n, but the loop runs m iterations.
//!   mode=0  m = 3000 fixed          expected: cost constant in n (D absorbs m)
//!   mode=1  m = 2*n                 expected: affine in n with the coefficient
//!                                   of m's loop attributed to n (wrong but consistent)
//!   mode=2  m = (n*7919) % 4000     expected: irregular (m is not a function
//!                                   of n that derive can express)
use perfmark::Region;
use std::hint::black_box;

fn main() {
    let st = perfmark::states(&[("n", 1000), ("mode", 0)]);
    let n = st["n"];
    let m = match st["mode"] {
        0 => 3000,
        1 => 2 * n,
        _ => (n * 7919) % 4000,
    };
    let mut acc = 0u64;
    {
        let _r = Region::new("undeclared", "n", n);
        for i in 0..m as u64 {
            acc = black_box(acc).wrapping_add(i);
        }
    }
    println!("undeclared n={} m={} acc={}", n, m, acc);
}
