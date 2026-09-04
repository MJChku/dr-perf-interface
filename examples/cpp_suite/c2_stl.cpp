// Case 2: STL containers.  Expected: vec_reserve affine; vec_grow affine per
// element plus reallocation blocks that are irregular (geometric growth);
// map_insert and sort_ints not affine (n log n) -> irregular share (N3);
// umap_insert affine per element plus irregular rehash blocks.
#include "suite.h"
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>

static uint32_t lcg(uint32_t &s) { s = s * 1664525u + 1013904223u; return s; }

int main(int argc, char **argv)
{
    long n = arg_value(argc, argv, "n", 1000);
    { std::vector<int> w; for (int i = 0; i < 64; i++) w.push_back(i); g_sink += w.size(); }   // warm-up

    perfmark_begin("vec_grow", "n", n);
    { std::vector<int> v; for (long i = 0; i < n; i++) v.push_back((int)i); g_sink += v[n / 2]; }
    perfmark_end("vec_grow");

    perfmark_begin("vec_reserve", "n", n);
    { std::vector<int> v; v.reserve(n); for (long i = 0; i < n; i++) v.push_back((int)i); g_sink += v[n / 2]; }
    perfmark_end("vec_reserve");

    perfmark_begin("map_insert", "n", n);
    { std::map<int, int> m; uint32_t s = 1; for (long i = 0; i < n; i++) m.emplace((int)(lcg(s) >> 1), (int)i); g_sink += m.size(); }
    perfmark_end("map_insert");

    perfmark_begin("umap_insert", "n", n);
    { std::unordered_map<int, int> m; for (long i = 0; i < n; i++) m.emplace((int)i * 7, (int)i); g_sink += m.size(); }
    perfmark_end("umap_insert");

    std::vector<int> data(n); { uint32_t s = 2; for (long i = 0; i < n; i++) data[i] = (int)(lcg(s) >> 1); }
    perfmark_begin("sort_ints", "n", n);
    std::sort(data.begin(), data.end());
    perfmark_end("sort_ints");
    g_sink += data[0];
    printf("c2 n=%ld sink=%llu\n", n, (unsigned long long)g_sink);
    return 0;
}
