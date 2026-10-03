# Findings

The main input-dependent cost is native expression parsing with both linear chain size and cumulative growing-prefix size. For this workload, parse-entry lexical counts suffice: A=count('+'), U=count('|'), D=count('.'), C=count('()'). The supported nonlinear expressions are A(A+1)/2, U(U+1)/2, and P(P+1)/2 with P=D+C. These are cheap computations on the input string, not parser re-execution. The expressions are restricted to the supplied generated input families.

Native cached left-recursive sum, bitwise_or, primary and t_primary rules build nested owned expressions (source/native/libcst/src/parser/grammar.rs). The cst_node derive macro generates Clone for deflated structs/enums, and BinaryOperation, Attribute and Call hold boxed child expressions. This provides a source mechanism for cumulative prefix work. cProfile cannot directly attribute native time to cloning or separate primary from t_primary; those remain inferences.

Python validation instead uses local state: Name identifier length; call argument count; and parenthesis presence/length equality. Empty calls do no argument iterations, and their validation never walks the func spine. Measured validation and constructor counts remain linear. Total __post_init__ counts, including import defaults, are exactly 5n+11 (addition), 5n+15 (union), 8n+16 (method chain). Startup import/class-generation is input-independent: 196 dataclass-processing calls in every process.

# Experiments

Submitted source-only hypotheses before measurements, then revised and submitted after each round. Three timing rounds used 18 processes, testing n=2,32,64,128,192,256 in each of the three shapes. All processes succeeded. No source or workload modifications were made. analyze.py and fit.py read saved result.json/pstats only.

Exclusive cProfile time in entrypoints._parse (milliseconds; includes opaque native parsing work, excludes Python child calls):

| n | addition | union annotation | method chain |
|---:|---:|---:|---:|
| 2 | 0.774 | 0.771 | 0.821 |
| 32 | 1.178 | 1.296 | 2.866 |
| 64 | 2.149 | 2.127 | 7.554 |
| 128 | 4.548 | 4.382 | 24.443 |
| 192 | 7.798 | 7.741 | 52.332 |
| 256 | 12.360 | 12.352 | 93.881 |

An affine fit in n and n(n+1)/2 has RMS residuals 0.069, 0.049 and 0.470 ms; linear-only residuals are 0.771, 0.816 and 8.687 ms. This is timing evidence for a nonlinear chain dependency, not an instruction-count-derived wall-time law. At n=256 method-call validation totals only 0.397 ms. Import/class generation costs about 0.13 seconds and dominates most process wall times.

# Limits

Each configuration was measured once; timer variation and cache effects remain. Counts of bytes, tokens, numbered-name lengths and final nodes are coupled by the workload, so their distinct linear contributions cannot be isolated. Attribute and call counts are also coupled. Addition and union differ in surrounding statement syntax. Native inner functions are invisible to cProfile. Fits support the restricted-family affine interface but do not establish a universal Python-source parser cost law.
