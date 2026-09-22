/* State cardinality, overflow, revisits, and many independent regions. */
#include <stdio.h>
#include <stdlib.h>
#include "../perfmark/perfmark.h"

int main(int argc, char **argv) {
    int points = argc > 1 ? atoi(argv[1]) : 5000;
    int regions = argc > 2 ? atoi(argv[2]) : 1;
    for (int r = 0; r < regions; ++r) {
        char name[32];
        snprintf(name, sizeof(name), "states_%d", r);
        for (int repeat = 0; repeat < 2; ++repeat) {
            for (int n = 0; n < points; ++n) {
                perfmark_begin(name, "n", n);
                /* A different cost beyond the old limit must stay visible. */
                for (int j = 0; j < (n < 128 ? 1 : 9); ++j)
                    __asm__ __volatile__("nop" ::: "memory");
                perfmark_end(name);
            }
        }
    }
    puts("STATE_CARDINALITY_PASS");
    return 0;
}
