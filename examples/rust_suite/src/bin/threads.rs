//! Case 6a: std::thread::scope with 4 workers, each doing n/4 iterations,
//! under the main thread's region `par`.  The workers have no region of their
//! own and follow the leader (SPEC section 0, attribution), so derive should
//! give a total of about a*n + d where a is the per-iteration cost of one
//! worker; thread creation is in d (may leave a small irregular share).
use perfmark::Region;
use std::hint::black_box;

fn main() {
    let st = perfmark::states(&[("n", 1000)]);
    let n = st["n"];
    let per = (n / 4) as u64;
    let mut sums = [0u64; 4];
    {
        let _r = Region::new("par", "n", n);
        std::thread::scope(|s| {
            for (k, slot) in sums.iter_mut().enumerate() {
                s.spawn(move || {
                    let mut acc = k as u64;
                    for i in 0..per {
                        acc = black_box(acc).wrapping_add(i);
                    }
                    *slot = acc;
                });
            }
        });
    }
    println!("threads n={} sums={:?}", n, sums);
}
