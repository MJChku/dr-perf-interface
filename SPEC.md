# drperf: what it computes

**Measurement.** `perfmark_begin(ρ, x, v)` (`perfmark_begin_v`: up to four
states) … `perfmark_end(ρ)` around a code region yields a *trigger*
$t$: region $\rho(t)$, declared state vector $v(t) \in \mathbb{Z}^k$, thread
$\tau(t)$, sequence numbers $b(t) < e(t)$ from one atomic counter shared by
all threads. DynamoRIO attributes every executed
user-space instruction to the innermost open trigger of its thread, or the
leader thread's if it has none. $n_\beta(t)$ is the number of instructions of
basic block $\beta$ attributed to $t$ on all threads;
$\mathrm{cost}(t) = \sum_\beta n_\beta(t)$. Counts are exact and, for
deterministic single-threaded programs, reproducible to the instruction. Only
per-point means $c_\beta(v)$ over triggers with $v(t) = v$ are stored, for at
most 128 points per region; calls beyond that are counted and reported, not
placed at a point.

**Cost formulae (`derive`).** Over the observed points $V \subset \mathbb{Z}^k$,
$|V| \ge k + 2$, each block gets a least-squares plane
$a_\beta \cdot v + d_\beta$; tolerance $\theta(y) = \max(64,\ 0.05\,y)$.
$\beta$ is *affine* if $|c_\beta(v) - a_\beta \cdot v - d_\beta| \le \theta(c_\beta(v))$
for all $v \in V$ and $d_\beta \ge -\theta(\max_v c_\beta)$; it *scales* in
variable $j$ if moreover $|a_{\beta j}|(\max V_j - \min V_j) > \theta(\max_v c_\beta)$
or the points are exactly coplanar with $a_{\beta j} \ne 0$ (other slopes
are folded into the constant); a block scaling in no variable is *constant*;
every other block is *irregular*.

$$\mathrm{cost}(v) \approx A \cdot v + D,\qquad A_j = \sum_{\text{scaling in } j} a_{\beta j},\quad D = \sum_{\text{affine}} d_\beta' + \sum_{\text{constant}} \operatorname{mean}_v c_\beta(v),$$

with $I(v) = \sum_{\text{irregular}} c_\beta(v)$ tabulated at $V$ only.
OpenMP runtime blocks are excluded as waiting and the calibrated marker cost
is subtracted from $D$. Variables collinear over $V$ cannot be separated and
are reported. For $k = 1$ and an irregular share above 5%, $V$ may be split
once into two regimes of at least 3 points (3 is flagged weak), chosen by
the mean per-point irregular fraction. Guaranteed: exactly affine counts give exact $A, D$ (P1); at
$v \in V$ the error is at most $\sum_{\text{affine}} \theta$ (P2); irregular
cost never enters $A$ or $D$ (P3). Not provided: loop detection or
data flow (the only link to $v$ is co-variation across $V$: an undeclared
variable shows as irregular if independent, hides in $D$ if balanced across
$V$, is charged to $v$ if correlated), non-affine
forms ($n \log n$ over a fourfold range passes as a plane; a negative $D$ is
the sign), regimes in more than one variable, checked extrapolation beyond
$V$, min–max intervals.

**Relationships (`learn`).** For a trigger $t$ of region $A$ with state $x$,
over every region $R$ and state $u$:
$\mathrm{count}_R(t) = |\{t' : \rho(t') = R,\ b(t') < b(t)\}|$,
$\mathrm{cum}_{R,u}(t) = \sum u(t')$ over that set, $\mathrm{last}_{R,u}(t)$ =
$u$ of its latest member, $\mathrm{cumend}_{R,u}(t) = \sum u(t')$ over
$\{\rho(t') = R,\ e(t') < b(t)\}$. `learn` reports every
$x(t) = \sum_{j \in S} \gamma_j \varphi_j(t) + \delta$ with $|S| \le 3$,
$\gamma_j \in \{\pm 1, \pm 2\}$, holding exactly at every trigger of $A$;
otherwise a least-squares fit, labelled approximate. Guaranteed: an exact
relation describes the run, not an estimate (R1). Not provided: relations
through undeclared variables, causality, happens-before (the order is the
counter's), protection against coincidence when candidates outnumber
triggers, so a relation counts only if exact in every run.

**Composition.** $\mathrm{incl}(t) = \mathrm{self}(t) + \sum \mathrm{self}(t')$
over triggers nested in $t$ on its thread is an identity of the measurement,
not a model.
