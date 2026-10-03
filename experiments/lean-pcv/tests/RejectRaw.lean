import Examples
import Discover
open Pcv Examples

-- This must fail during coefficient discovery: the sample equations disagree.
perf_discover raw : threshold using [x, y]
