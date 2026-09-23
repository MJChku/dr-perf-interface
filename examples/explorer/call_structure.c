/* Negative control for fixed-trace scenarios: a changed loop trip count adds
 * region calls even though each item retains the same performance interface. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "../../perfmark/perfmark.h"

static volatile uint64_t sink;

int main(int argc, char **argv) {
    long scale = argc > 1 ? strtol(argv[1], NULL, 10) : 1;
    if (scale < 1 || scale > 2) return 2;
    for (long n = 1; n <= 8; ++n) {
        long items = n * scale;
        perfmark_begin("batch", "items", items);
        for (long item = 0; item < items; ++item) {
            long bytes = 8 * (7 + item % 7);
            perfmark_begin("item", "bytes", bytes);
            uint64_t value = 0;
            for (long i = 0; i < bytes; ++i)
                __asm__ __volatile__("add %1, %0" : "+r"(value) : "r"((uint64_t)i));
            sink += value;
            perfmark_end("item");
        }
        perfmark_end("batch");
    }
    printf("call-structure control: %llu\n", (unsigned long long)sink);
    return 0;
}
