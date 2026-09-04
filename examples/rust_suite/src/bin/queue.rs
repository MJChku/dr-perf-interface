//! Case 6b (SPEC R1): producer/consumer over std::sync::mpsc.
//! The producer thread sends bursts of m items inside region `produce` (m
//! declared); every round it sends one or two bursts, then notifies the
//! consumer and waits for an ack.  The consumer drains the channel, opens
//! region `consume` with q = items in this batch, and processes them.
//! Expected: `drperf learn` finds consume.q = cum(produce.m) - cum(consume.q)
//! exactly (and cumend(produce.m) as an alias), `trace --check` confirms it.
use perfmark::Region;
use std::hint::black_box;
use std::sync::mpsc;

fn main() {
    let st = perfmark::states(&[("rounds", 12)]);
    let rounds = st["rounds"] as usize;
    let (tx, rx) = mpsc::channel::<u64>();
    let (ntx, nrx) = mpsc::channel::<()>();
    let (atx, arx) = mpsc::channel::<()>();
    let producer = std::thread::spawn(move || {
        let mut sent = 0u64;
        for i in 0..rounds {
            let bursts = if i % 3 == 2 { 2 } else { 1 };
            for j in 0..bursts {
                let m = 20 + ((i * 37 + j * 11) % 40) as i64;
                let _r = Region::new("produce", "m", m);
                for k in 0..m as u64 {
                    tx.send(black_box(k)).unwrap();
                    sent += 1;
                }
            }
            ntx.send(()).unwrap();
            arx.recv().unwrap();
        }
        sent
    });
    let mut total = 0u64;
    let mut got = 0u64;
    for _ in 0..rounds {
        nrx.recv().unwrap();
        let mut batch = Vec::new();
        while let Ok(x) = rx.try_recv() {
            batch.push(x);
        }
        {
            let _r = Region::new("consume", "q", batch.len() as i64);
            for x in &batch {
                total = black_box(total).wrapping_add(*x);
            }
        }
        got += batch.len() as u64;
        atx.send(()).unwrap();
    }
    let sent = producer.join().unwrap();
    println!("queue rounds={} sent={} got={} total={}", rounds, sent, got, total);
}
