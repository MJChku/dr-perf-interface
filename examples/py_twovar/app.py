"""Two declared states in one Python region.

    with perfmark.region("work", n=len(a), m=len(b)):

Both integer keywords are declared states: the client keeps block counts per
(n, m) pair and `drperf derive` gives cost = A_n*n + A_m*m + D.  Interpreter
blocks are shared by both loops, so each block's count is c1*n + c2*m + c:
affine in the pair, which a single-variable derivation could not see.

    bin/drperf run --blocks -o out/tv --state n=1000,2000,3000,5000 --state m=300,900,1500 \\
        -- python examples/py_twovar/app.py
    bin/drperf derive out/tv
"""
import perfmark

st = perfmark.states(n=1000, m=300)
n, m = int(st["n"]), int(st["m"])
a = list(range(n))
b = list(range(m))

total = 0
for rep in range(3):
    with perfmark.region("work", n=len(a), m=len(b)):
        s = 0
        for x in a:
            s += x
        for y in b:
            s += 2 * y
        total += s
print("py_twovar n=%d m=%d total=%d" % (n, m, total))
