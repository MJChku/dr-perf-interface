"""Two snippets coupled through a queue, on two threads.

  produce  (thread A): pushes a batch of `m` items, m varies per call
  consume  (thread B): drains whatever is queued; symbolic state q = queue length

The consumer's cost formula is in q; the producer decides q.  `drperf trace
--check` verifies the coupling on the recorded trace and `drperf model`
gives each snippet its own formula.

    drperf run --model produce,consume -o out/q --threads 2 -- python examples/queue/app.py
    drperf model out/q
    drperf learn out/q          # finds consume.q == cumend(produce.m) - cum(consume.q)

The producer pushes bursts of 1..3 batches before waking the consumer, so
`q == last(produce.m)` is false and only the cumulative relation holds.
`rounds` and `batch` come from the command line (K=V), e.g. batch=2 doubles
what the producer pushes per call (the "head" variant of the system).
"""
import collections
import threading

import perfmark

st = perfmark.states(rounds=24, batch=1)
ROUNDS, BATCH = int(st["rounds"]), int(st["batch"])

queue = collections.deque()
lock = threading.Lock()
produced = threading.Semaphore(0)
drained = threading.Event()
done = threading.Event()


def encode(item):
    h = 0
    for ch in item:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h


def produce(m):
    with perfmark.region("produce", m=m):
        items = ["item-%d" % i for i in range(m)]
        with lock:
            queue.extend(items)
        return len(items)


def consume():
    with lock:
        qlen = len(queue)
    with perfmark.region("consume", q=qlen):
        total = 0
        for _ in range(qlen):
            with lock:
                item = queue.popleft()
            total += encode(item)
        return total


def producer():
    produce(3)                      # warm-up, outside the measured rounds
    produced.release()
    k = 0
    while k < ROUNDS:
        burst = 1 + k % 3           # push 1..3 batches before waking the consumer
        for _ in range(burst):
            m = BATCH * (5 + 20 * (k % 4))   # 5, 25, 45, 65 items, scaled by batch
            produce(m)
            k += 1
        produced.release()
        drained.wait()              # let the consumer drain before the next burst
        drained.clear()
    done.set()
    produced.release()


def consumer():
    total = 0
    while True:
        produced.acquire()
        if done.is_set() and not queue:
            break
        total += consume()
        drained.set()
    return total


t = threading.Thread(target=producer)
t.start()
total = consumer()
t.join()
print("queue example rounds=%d batch=%d checksum=%d" % (ROUNDS, BATCH, total))
