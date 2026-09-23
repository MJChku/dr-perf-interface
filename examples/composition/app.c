/* Cross-function nested interfaces, including deliberately incomplete PCVs.
 * Every marked child lives in another noinline function. No symbolic source
 * inspection is needed: the checker uses observed marker nesting and PCVs.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "../../perfmark/perfmark.h"

static volatile uint64_t sink;
static int changed_calls;
#define NOINLINE __attribute__((noinline))
#define MARK2(region, a, b) do { \
    const char *names[] = {#a, #b}; \
    int64_t values[] = {a, b}; \
    perfmark_begin_v(region, 2, names, values); \
} while (0)

static NOINLINE void work(long n) {
    uint64_t value = 0;
    for (long i = 0; i < n; ++i)
        __asm__ __volatile__("add %1, %0" : "+r"(value) : "r"((uint64_t)i));
    sink += value;
}

static NOINLINE void linear_leaf(long m) {
    perfmark_begin("linear_leaf", "m", m);
    work(64 * m);
    perfmark_end("linear_leaf");
}

/* A hash-derived loop bound is not affine in m. This child remains irregular
 * even when its parent has a perfectly simple call-count interface. */
static NOINLINE void irregular_leaf(long m) {
    perfmark_begin("irregular_leaf", "m", m);
    uint32_t hash = (uint32_t)m * UINT32_C(2654435761);
    long probes = (hash ^ (hash >> 13)) % 97 + 1;
    work(128 * probes);
    perfmark_end("irregular_leaf");
}

static NOINLINE void repeated(long n, long m) {
    MARK2("repeated", n, m);
    for (long i = 0; i < 2*n+1; ++i) irregular_leaf(m);
    perfmark_end("repeated");
}

static NOINLINE void varying(long n, long m) {
    MARK2("varying", n, m);
    for (long i = 0; i < n; ++i) linear_leaf(m+i);
    perfmark_end("varying");
}

static NOINLINE void middle(long n, long m) {
    MARK2("middle", n, m);
    for (long i = 0; i < n; ++i) irregular_leaf(m);
    perfmark_end("middle");
}

static NOINLINE void three_levels(long batches, long n, long m) {
    const char *names[] = {"batches", "n", "m"};
    int64_t values[] = {batches, n, m};
    perfmark_begin_v("three_levels", 3, names, values);
    for (long i = 0; i < batches; ++i) middle(n, m);
    perfmark_end("three_levels");
}

static int selected(long i, long m) {
    uint32_t bits = (uint32_t)(i + 17*m) * UINT32_C(2654435761);
    return ((bits ^ (bits >> 11)) & 7) < 3;
}

static NOINLINE void branch_raw(long n, long m) {
    MARK2("branch_raw", n, m);
    for (long i = 0; i < n; ++i)
        if (selected(i, m)) irregular_leaf(m);
    perfmark_end("branch_raw");
}

static NOINLINE void branch_refined(long n, long m) {
    long selected_count = 0;
    for (long i = 0; i < n; ++i) selected_count += selected(i, m);
    const char *names[] = {"n", "m", "selected_count"};
    int64_t values[] = {n, m, selected_count};
    perfmark_begin_v("branch_refined", 3, names, values);
    for (long i = 0; i < n; ++i)
        if (selected(i, m)) irregular_leaf(m);
    perfmark_end("branch_refined");
}

/* The child argument is constant but the hidden selector changes the number
 * of calls for identical parent PCVs. Averaging would hide this failure. */
static NOINLINE void hidden_count(long n, long m, long secret) {
    MARK2("hidden_count", n, m);
    for (long i = 0; i < n+secret; ++i) irregular_leaf(m);
    perfmark_end("hidden_count");
}

static NOINLINE void recursive(long depth) {
    perfmark_begin("recursive", "depth", depth);
    work(16);
    if (depth > 0) recursive(depth-1);
    perfmark_end("recursive");
}

static NOINLINE void multiple_children(long n, long m) {
    MARK2("multiple_children", n, m);
    for (long i = 0; i < n; ++i) linear_leaf(m);
    for (long i = 0; i < n+2+changed_calls; ++i) irregular_leaf(m+1);
    perfmark_end("multiple_children");
}

static NOINLINE void left(long n, long m) {
    MARK2("left", n, m);
    for (long i = 0; i < n; ++i) linear_leaf(m);
    perfmark_end("left");
}

static NOINLINE void right(long n, long m) {
    MARK2("right", n, m);
    for (long i = 0; i < 2*n; ++i) linear_leaf(m);
    perfmark_end("right");
}

static NOINLINE void diamond(long n, long m) {
    MARK2("diamond", n, m);
    left(n, m);
    right(n, m);
    perfmark_end("diamond");
}

static NOINLINE void rectangular(long n, long width, long m, int refined) {
    const char *region = refined ? "rectangular_refined" : "rectangular_raw";
    const char *names[] = {"n", "width", "m", "cells"};
    int64_t values[] = {n, width, m, n*width};
    perfmark_begin_v(region, refined ? 4 : 3, names, values);
    for (long i = 0; i < n; ++i)
        for (long j = 0; j < width; ++j) linear_leaf(m);
    perfmark_end(region);
}

static NOINLINE void triangular(long n, long m, int refined) {
    const char *region = refined ? "triangular_refined" : "triangular_raw";
    const char *names[] = {"n", "m", "pairs"};
    int64_t values[] = {n, m, n*(n-1)/2};
    perfmark_begin_v(region, refined ? 3 : 2, names, values);
    for (long i = 0; i < n; ++i)
        for (long j = 0; j < i; ++j) linear_leaf(m);
    perfmark_end(region);
}

static NOINLINE void conditional_arguments(long n, long m, int refined) {
    const char *region = refined ? "arguments_refined" : "arguments_raw";
    long effective = m < 20 ? m : 2*m;
    const char *names[] = {"n", "m", "effective"};
    int64_t values[] = {n, m, effective};
    perfmark_begin_v(region, refined ? 3 : 2, names, values);
    for (long i = 0; i < n; ++i) linear_leaf(effective);
    perfmark_end(region);
}

static NOINLINE void alternating(long n, long m) {
    MARK2("alternating", n, m);
    for (long i = 0; i < n; ++i) {
        linear_leaf(m);
        linear_leaf(2*m);
    }
    perfmark_end("alternating");
}

static NOINLINE void mutual_b(long depth);
static NOINLINE void mutual_a(long depth) {
    long more = depth > 0;
    MARK2("mutual_a", depth, more);
    work(32);
    if (more) mutual_b(depth-1);
    perfmark_end("mutual_a");
}
static NOINLINE void mutual_b(long depth) {
    long more = depth > 0;
    MARK2("mutual_b", depth, more);
    work(48);
    if (more) mutual_a(depth-1);
    perfmark_end("mutual_b");
}

static NOINLINE void context_leaf(long m, int expensive, int refined) {
    const char *region = refined ? "context_leaf_refined" : "context_leaf_raw";
    const char *names[] = {"m", "units"};
    long units = m * (expensive ? 128 : 1);
    int64_t values[] = {m, units};
    perfmark_begin_v(region, refined ? 2 : 1, names, values);
    work(units);
    perfmark_end(region);
}

static NOINLINE void context_parent(long n, long m, int expensive, int refined) {
    const char *region = refined ? (expensive ? "context_expensive_refined" : "context_cheap_refined")
                                 : (expensive ? "context_expensive_raw" : "context_cheap_raw");
    MARK2(region, n, m);
    for (long i = 0; i < n; ++i) context_leaf(m, expensive, refined);
    perfmark_end(region);
}

static NOINLINE void regime_leaf(long m) {
    perfmark_begin("regime_leaf", "m", m);
    work(m * (m < 20 ? 512 : 64));
    perfmark_end("regime_leaf");
}

static NOINLINE void regime_parent(long n, long m) {
    MARK2("regime_parent", n, m);
    for (long i = 0; i < n; ++i) regime_leaf(m);
    perfmark_end("regime_parent");
}

int main(int argc, char **argv) {
    const long ns[] = {0, 1, 3, 7, 12};
    const long ms[] = {2, 5, 11, 19, 31, 47, 73, 101};
    const long new_ns[] = {2, 4, 6, 9, 14};
    const long new_ms[] = {3, 7, 13, 17, 29, 53, 79, 107};
    int holdout = argc > 1;
    changed_calls = argc > 1 && strcmp(argv[1], "changed") == 0;
    for (unsigned i = 0; i < sizeof(ns)/sizeof(*ns); ++i)
        for (unsigned j = 0; j < sizeof(ms)/sizeof(*ms); ++j) {
            long n = holdout ? new_ns[i] : ns[i], m = holdout ? new_ms[j] : ms[j];
            repeated(n, m);
            varying(n, m);
            branch_raw(n, m);
            branch_refined(n, m);
            hidden_count(n, m, 0);
            hidden_count(n, m, 3);
            for (long b = 1; b <= 3; ++b) three_levels(b, n, m);
            multiple_children(n, m);
            diamond(n, m);
            for (long width = 1; width <= 3; ++width) {
                rectangular(n, width, m, 0);
                rectangular(n, width, m, 1);
            }
            triangular(n, m, 0);
            triangular(n, m, 1);
            conditional_arguments(n, m, 0);
            conditional_arguments(n, m, 1);
            alternating(n, m);
            context_parent(n, m, 0, 0);
            context_parent(n, m, 1, 0);
            context_parent(n, m, 0, 1);
            context_parent(n, m, 1, 1);
            regime_parent(n, m);
        }
    for (long d = 1; d <= 9; ++d) recursive(d);
    for (long d = 1; d <= 9; ++d) mutual_a(d);
    printf("composition examples checksum: %llu\n", (unsigned long long)sink);
    return 0;
}
