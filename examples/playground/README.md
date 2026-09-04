# drperf playground

A small C system to try drperf on: a log pipeline.

```
ingest -> queue -> handle (parse each line) -> output
```

`system.c` is about 250 lines with three marked regions. `bin/head` is the
same file with one change, an extra pass over every byte in `parse`.

## Build

```sh
cd /home/ubuntu/drperf
examples/playground/build.sh      # needs build/ (run ./build.sh once)
```

## Run it

```sh
bin/drperf examples/playground/bin/base
```

```
cost per call, in instructions

  handle = 1,333.7*q + 362.4   (inflight: no effect; 12% of the cost follows no declared state)
  ingest = 152*batch + 15   (bytes: moved in step with batch)
  parse  = 5*len + 486   (entries: did not vary)

relations

  handle.inflight = cum(ingest.batch) - cum(handle.q)   (holds at all 57 calls)
```

That is the whole interface: `drperf` followed by the command you would have
run anyway. One run, no options, no flags.

## Reading it

- `parse` costs five instructions per byte of the line, plus 486 fixed.
- `handle` costs about 1,334 instructions per line it takes off the queue,
  most of which is the `parse` it calls. `inflight` has no effect: handling a
  line costs the same whether the queue is short or long, and the run proves
  it, with the cost flat while `inflight` ranges from 21 to 105 at a fixed
  `q`. Twelve percent of its cost follows neither state, so it is left out of
  the formula rather than smeared into a coefficient.
- `ingest` declares `bytes` too, but here every batch has the same mix of line
  lengths, so `bytes` moved in step with `batch` and the two cannot be told
  apart. The line says so instead of splitting the cost arbitrarily.
- The relation is the queue invariant: lines waiting equals lines pushed minus
  lines popped. Nothing in the program says that. It comes from the order the
  markers fired and the values they carried.

## Now the change

```sh
bin/drperf examples/playground/bin/head
```

`parse` becomes `10*len + 488`. The added pass costs exactly five instructions
per byte, and nothing else moved.

## The three regions in the source

Each names a region and the integers that matter to it. Nothing else is added
to the program.

```c
perfmark_begin_v("ingest", 2, (const char *[]){"batch", "bytes"}, (int64_t[]){batch, bytes});
perfmark_begin_v("handle", 2, (const char *[]){"q", "inflight"}, (int64_t[]){q, inflight});
perfmark_begin_v("parse",  2, (const char *[]){"len", "entries"}, (int64_t[]){n, entries});
```

A state gets a coefficient only if it varied during the run. `entries` is
fixed by a command-line argument here, so it sits in the constant. Try

```sh
bin/drperf examples/playground/bin/base entries=16384
```

With a bigger table the hash chains get long, and `parse` reports that 66% of
its cost follows neither of its states: chain length depends on which words a
line happens to contain, and nothing declares that. The tool says so instead
of inventing a coefficient.

## Things to try

- Add work to `parse` that depends on something no region declares. It shows
  up as cost that follows none of the states, not as a wrong coefficient.
- Then declare that thing as another state and watch it become a coefficient.
- `bin/drperf examples/playground/bin/base threads=4` runs the handler on four
  threads. Work on every thread counts into the open region, so the totals
  hold.
- Put the `parse` markers inside its byte loop. A marker costs a few hundred
  instructions, more than the loop body, and the numbers stop meaning
  anything. A region of a few lines is fine; a region inside a hot inner loop
  is not.
