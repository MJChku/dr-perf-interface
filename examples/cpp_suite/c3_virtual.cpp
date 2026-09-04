// Case 3: n virtual calls through a base pointer vs the same work through a
// template functor.  Expected: both affine in n; per-function attribution
// shows demangled C++ names (Shape::area etc.).
#include "suite.h"
struct Shape { virtual ~Shape() {} virtual uint64_t area(uint64_t i) const = 0; };
struct Square : Shape { uint64_t area(uint64_t i) const override { return i * i; } };
struct Circle : Shape { uint64_t area(uint64_t i) const override { return 3 * i * i + 1; } };
template <class F> __attribute__((noinline)) uint64_t apply_n(F f, long n)
{
    uint64_t acc = 0;
    for (long i = 0; i < n; i++) acc += f((uint64_t)i);
    return acc;
}
__attribute__((noinline)) uint64_t virtual_n(const Shape *s, long n)
{
    uint64_t acc = 0;
    for (long i = 0; i < n; i++) acc += s->area((uint64_t)i);
    return acc;
}
struct SquareF { uint64_t operator()(uint64_t i) const { return i * i; } };
int main(int argc, char **argv)
{
    long n = arg_value(argc, argv, "n", 1000);
    Square sq; Circle ci;
    const Shape *shapes[2] = { &sq, &ci };
    g_sink += virtual_n(shapes[0], 8) + apply_n(SquareF(), 8);
    perfmark_begin("virtual_calls", "n", n);
    g_sink += virtual_n(shapes[n & 1], n);     // Square for even n, Circle for odd n
    perfmark_end("virtual_calls");
    perfmark_begin("template_calls", "n", n);
    g_sink += apply_n(SquareF(), n);
    perfmark_end("template_calls");
    printf("c3 n=%ld sink=%llu\n", n, (unsigned long long)g_sink);
    return 0;
}
