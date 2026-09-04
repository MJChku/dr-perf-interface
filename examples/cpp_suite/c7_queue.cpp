// Case 7: std::thread producer/consumer with mutex + condition_variable.
// Producer: each round pushes k bursts (k cycles 1,2,3,2) of m items each
// (m cycles 5,9,13,7,11), one "produce" region per burst declaring m.
// Consumer: after each round drains the queue in one "consume" region
// declaring q = items popped.  Expected (R1): consume.q = cum(produce.m) - cum(consume.q)
// exactly; the relation holds because bursts end before the consumer begins.
#include "suite.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
static std::mutex mu;
static std::condition_variable cv;
static std::deque<long> queue;
static bool round_ready = false, drained = true, done = false;
int main(int argc, char **argv)
{
    long rounds = arg_value(argc, argv, "rounds", 12);
    static const long ms[5] = { 5, 9, 13, 7, 11 };
    static const int ks[4] = { 1, 2, 3, 2 };
    std::thread consumer([] {
        for (;;) {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [] { return round_ready || done; });
            if (done && !round_ready) return;
            round_ready = false;
            long q = (long)queue.size();
            perfmark_begin("consume", "q", q);
            uint64_t acc = 0;
            while (!queue.empty()) { acc += loop3(queue.front()); queue.pop_front(); }
            perfmark_end("consume");
            g_sink += acc;
            drained = true;
            cv.notify_all();
        }
    });
    int mi = 0;
    for (long r = 0; r < rounds; r++) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [] { return drained; });
        drained = false;
        for (int b = 0; b < ks[r % 4]; b++) {
            long m = ms[mi++ % 5];
            perfmark_begin("produce", "m", m);
            for (long i = 0; i < m; i++) queue.push_back(3 + i % 4);
            perfmark_end("produce");
        }
        round_ready = true;
        cv.notify_all();
    }
    { std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [] { return drained; }); done = true; }
    cv.notify_all();
    consumer.join();
    printf("c7 rounds=%ld sink=%llu\n", rounds, (unsigned long long)g_sink);
    return 0;
}
