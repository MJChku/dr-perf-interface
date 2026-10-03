# LMCache overlay for the drperf capture

Exact copies of the LMCache modules on the store and retrieve paths, from the
source build the capture runs (upstream LMCache `8e93a34`, Apache-2.0, installed
as `pydeps_lmc` on the GX host). `drperf_lmcache.sh` copies them over the
installed package at container start, so what runs is exactly what is here.
They carry drperf regions in place, `@marked(...)` and `with region(...)` from
`src.perf`, so the exported profile maps every region to LMCache's own lines.
The first commit of this directory is the unmodified upstream text; later
commits are the annotations only.
