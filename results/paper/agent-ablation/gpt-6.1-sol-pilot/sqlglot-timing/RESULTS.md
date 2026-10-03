# SQLGlot timing-arm PCV discovery

Submitted source-only round 0 before measurement, revised after rounds 1 and 2, and submitted the final four claims as `measurements/submissions/round-02.json`. Used two rounds, six processes each, all successful. Source, workload, and supervisor configuration were unchanged.

The main discovery is repeated scope collection during CTE merging. `qualify(..., isolate_tables=True)` isolates each table; `eliminate_subqueries` turns those derived tables into CTEs; `merge_ctes` then merges the single-use CTEs back into the outer query. My initial source-only hypothesis placed this work in `merge_derived_tables`; profiles corrected the region to `merge_ctes`.

At `merge_ctes` entry, the useful expressions are products of simple single-use CTE count M with outer in-scope AST node count N, qualified column occurrence count C, and selected source count S. A triangular M(M-1)/2 term accounts for disappearing CTE shells and list operations; total inner projection AST size supplies a linear term. Here M=n, C=4n-2, S=n. These are computations on structural state, not workload replay.

| n tables | merge_subqueries cumulative s | _merge_expressions cumulative s | qualify cumulative s | optimize_joins cumulative s |
|---:|---:|---:|---:|---:|
| 16 | 0.0695 | 0.0348 | 0.0262 | 0.00855 |
| 32 | 0.2626 | 0.1377 | 0.0508 | 0.01714 |
| 64 | 1.0104 | 0.5429 | 0.0984 | 0.03442 |
| 128 | 3.9177 | 2.1317 | 0.1942 | 0.06924 |

Round 2 used n=24,48,72,96,112,128. Merge time at the repeated n=128 was 3.9247 s (0.18% from round 1). The merge curve supports a dominant quadratic structural product; qualification, star expansion, and join optimization are approximately linear over this envelope. At n=128 merging consumes about 83% of profiled optimizer cumulative time. Wall times include interpreter/import costs and profiling overhead: round 1 n=128 was 4.985 s. Cumulative times overlap and must not be summed.

The causal source locations are `optimizer/merge_subqueries.py:78` (`merge_ctes`), `:145` (`_mergeable`), `:399` (`_merge_expressions`), and `optimizer/scope.py:179` (`_collect`), `:353` (`columns`), `:589`/`:594` (source mutation invalidates caches). `_mergeable` reads pivots; `_merge_from` removes and adds sources, clearing caches; `_merge_expressions` subsequently reads columns; each completed merge clears caches again. The local cache-dependent PCV is invalid-cache indicator times in-scope AST node count, plus column/projection counts for `columns`. Repeated linear work explains the quadratic caller.

Supporting cProfile observations across all 12 points: `_collect` call count equals 9n+12, and generator resume count for `walk_in_scope` equals 44n^2+368n-67. These are validation evidence, not the submitted entry-state PCVs or a claim that time equals call count.

Additional claims: `_expand_stars` depends on the sum of visible column widths of sources targeted by stars, plus projection count. For this family each table has width two, and there are n+1 expansion calls. `optimize_joins` depends on join/predicate/dependency size; its `references.get(table, []) + [join]` list copying adds a source-supported triangular term in dependency fan-in. That term was too small to isolate in timing. The join DAG has two layers, so its topological-sort scans are linear here, despite quadratic whole-optimizer behavior.

`analysis.csv` extracts cumulative times and call counts from saved pstats. Each process printed the expected 2n output columns.

Limits: n_joins is the only input allowed by session.json. Schema width, column count, CTE count, and dependency fan-in covary, so timing cannot separately identify coefficients of the structural products. We could not test independent schema widths or chain-versus-star topology. Expressions are scoped to simple qualified column projections, distinct aliases, fixed equality predicates, no correlations/window/group/order/where, and bounded ancestor depth. General SQL may require further terms for correlated scopes, complex projection copies, predicate normalization, and graph depth. No claim of a universal wall-time law, and no optimization performed.
