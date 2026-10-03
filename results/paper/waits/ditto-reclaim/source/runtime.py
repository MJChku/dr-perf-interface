"""Asynchronous KV-cache lifecycle independent of any serving engine.

Clients supply storage, optional codec adapters, and a RawTransport. This
runtime coordinates mappings, compaction, outstanding transfers, and source leases.
Scheduler policy and engine-specific request types belong to the client.
"""

import os
import sys
from dataclasses import dataclass

from src.cache.compaction import CompactionEngine, CompactionResources
from src.cache.load_engine import LoadEngine
from src.cache.store_engine import (
    StoreCapacityRefused,
    StoreEngine,
    contain_load_refusal,
    drop_refused_load,
    invalidate_store_blocks,
    refuse_store_capacity,
)
from src.cache.types import (
    CacheCompletion,
    LoadRequest,
    RawTransport,
    StoreRequest,
    TransferCompletion,
)
from src.perf import available as _perf_available
from src.perf import marked, n, pcv, region

_TRACE = os.environ.get("DITTO_DEBUG_FIN") == "1"


def _restore_concurrency(workspace):
    """Use half the workspace lanes for parallel decode streams by default.

    DITTO_DECODE_STREAMS varies concurrency for profiling while preserving
    workspace capacity. Without workspace, decodes use one stream.
    """
    slabs = int(workspace.report()["slabs"]) if workspace is not None else 0
    override = os.environ.get("DITTO_DECODE_STREAMS")
    if slabs and override:
        return max(1, min(int(override), slabs))
    return max(1, slabs // 2)


def _restore_queue(runtime):
    """Codec-restore queue accounting on the serving thread: backlog L =
    restores submitted and not yet finished on the GPU, integrals of L over
    time, and the time the queue was non-empty / at least k deep. Supply =
    submitted rows per second; service = completed rows per non-empty second;
    capacity = completed rows per saturated second (queue >= k, the decode
    path is the limiter). Arrivals are stamped at submission and departures
    at their GPU end time, so the client's polling lag is not counted as backlog;
    the events are integrated in time order by _queue_settle at each poll."""
    d = runtime.__dict__
    if "_rq" not in d:
        import time

        d["_rq"] = {
            "L": 0,
            "t": time.monotonic(),
            "max": 0,
            "events": [],
            "k": 1,  # queue depth at which reloads occupy all decode streams
        }
    return d["_rq"]


def _queue_accrue(runtime, rq, t):
    dt_us = round((t - rq["t"]) * 1e6)
    rq["t"] = max(rq["t"], t)
    if dt_us <= 0 or rq["L"] <= 0:
        return
    runtime.ftl.count("decode_backlog_us", dt_us)
    runtime.ftl.count("decode_backlog_l_us", dt_us * rq["L"])
    if rq["L"] >= rq["k"]:
        runtime.ftl.count("decode_saturated_us", dt_us)


def _queue_settle(runtime, rq, now):
    """Integrate the queue up to `now` from the events recorded since the
    last settle. After a poll every departure at or before `now` is known,
    so the events are complete up to `now`. A job's departure is stamped no
    earlier than its own arrival (the arrival is the submission time taken
    before execute, so an arena wait inside execute counts as queued), and
    a departure stamped before the last settle is clamped to it."""
    events = sorted(rq["events"], key=lambda e: (e[0], -e[1]))  # +1 before -1
    rq["events"] = []
    for t, delta, rows in events:
        _queue_accrue(runtime, rq, min(max(t, rq["t"]), now))
        if delta < 0 and rq["L"] >= rq["k"]:
            runtime.ftl.count("decode_saturated_rows", rows)
        rq["L"] = max(0, rq["L"] + delta)
        rq["max"] = max(rq["max"], rq["L"])
    _queue_accrue(runtime, rq, now)


def _trace(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


@dataclass
class PendingJob:
    """One store or load, with one completion and one refusal outcome."""

    is_load: bool
    completion: TransferCompletion | None = None
    request_id: str | None = None
    refused: bool = False
    submitted: float = 0.0  # monotonic, shared by latency and restore accounting
    codec_rows: int = 0
    codec_departed: bool = False


class _CountedFTL:
    """Measurement-only proxy over the FTL controller: ``count`` gets its own
    region (a counter's first bump inserts its key) instead of landing in
    whichever region bumps it."""

    __slots__ = ("_ftl", "_seen")

    def __init__(self, ftl):
        self._ftl = ftl
        self._seen = set()

    def __getattr__(self, name):
        return getattr(self._ftl, name)

    def count(self, name, amount=1):
        new = name not in self._seen
        if new:
            self._seen.add(name)
        with region("ftl.count", lambda: dict(new=int(new))):
            return self._ftl.count(name, amount)


class CacheRuntime:
    """One cache instance with explicit storage, transport, and codec ownership.

    Encoder and decoder must either both be provided or both be absent. The
    runtime retains all supplied resources through shutdown. Callers own GPU
    page allocation and must flush a source lease before reusing its pages.
    """

    def __init__(
        self,
        *,
        ftl,
        handle_manager,
        transport: RawTransport,
        pages_per_block: int,
        gc_by_group,
        codec=None,
        extent_encoder=None,
        extent_decoder=None,
    ):
        if (extent_encoder is None) != (extent_decoder is None):
            raise ValueError("encoder and decoder must be configured together")
        if (codec is None) != (extent_encoder is None):
            raise ValueError("codec and extent adapters must be configured together")
        if pages_per_block <= 0 or not gc_by_group:
            raise ValueError(
                "cache requires positive page geometry and at least one group"
            )
        if codec is None and any(gc_by_group):
            raise ValueError("compaction requires a codec")
        if _perf_available:
            ftl = _CountedFTL(ftl)
        self.ftl = ftl
        self.handle_manager = handle_manager
        self.carrier: RawTransport | None = transport
        self.raw_row_bytes = int(transport.raw_row_bytes)
        self.pages_per_block = int(pages_per_block)
        self.gc_by_group = tuple(gc_by_group)
        self.codec = codec
        self.extent_encoder = extent_encoder
        self.lease_by_key: dict[str, int] = {}
        self.key_by_lease: dict[int, str] = {}
        self.pending_jobs: dict[int, PendingJob] = {}
        self.compaction_engine = CompactionEngine(
            CompactionResources(
                ftl=ftl,
                handle_manager=handle_manager,
                codec=codec,
                extent_encoder=extent_encoder,
                pages_per_block=self.pages_per_block,
                lease_by_key=self.lease_by_key,
                key_by_lease=self.key_by_lease,
            )
        )
        self.store_engine = StoreEngine(handle_manager, transport, self.raw_row_bytes)
        self._codec_summary = codec.summary if codec is not None else lambda: ""
        self.load_engine = LoadEngine(
            handle_manager=handle_manager,
            carrier=transport,
            extent_decoder=extent_decoder,
            pages_per_block=self.pages_per_block,
            raw_row_bytes=self.raw_row_bytes,
            group_count=len(self.gc_by_group),
            decode_stream_count=_restore_concurrency(getattr(codec, "workspace", None)),
            register_ready_pages=self._register_ready_pages
            if codec is not None
            else None,
            count=ftl.count,
        )
        self._verify_reports = self._pool_reports = 0
        # Write opt-in stats on the serving thread: the native controller is
        # unsendable. Polling refreshes this file at most once every five seconds.
        self._stats_path = os.environ.get("DITTO_STATS_FILE")
        if self._stats_path:
            self._stats_path = self._stats_path.replace("{pid}", str(os.getpid()))
        self._stats_next = 0.0
        if self._stats_path:
            self._maybe_write_stats()  # counters are available from boot onward

    @marked("rt.depart_restore")
    def _depart_restore(self, job, completion):
        import time as _time

        ended = completion.host_end_time()
        if ended is None:
            ended = _time.monotonic()
        ended = max(ended, job.submitted)  # not before arrival
        _restore_queue(self)["events"].append((ended, -1, job.codec_rows))
        job.codec_departed = True
        completion.account_codec()  # counted at GPU end
        self._trace_restore(job, completion)

    def _trace_restore(self, job, completion):
        """One JSON line per reload in DITTO_TRACE_FILE: when the client submitted
        it, when its kernels could start (served), when they finished, when
        this poll observed it, and the per-chunk copy/kernel spans -- all in
        ms on the load engine's GPU origin. Analysis happens offline."""
        path = os.environ.get("DITTO_TRACE_FILE")
        if not path:
            return
        try:
            import json
            import time as _time

            record = completion.trace() or {}
            engine = self.load_engine
            record.update(
                t=_time.time(),  # wall clock, to slice the trace per benchmark cell
                job=job.request_id or "",
                submitted_ms=engine.host_ms(job.submitted),
                observed_ms=engine.host_ms(_time.monotonic()),
            )
            with open(path, "a") as out:
                out.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception:
            pass

    def _maybe_write_stats(self):
        import json
        import time

        path = self._stats_path
        if not path:
            return
        now = time.monotonic()
        if now < self._stats_next:
            return
        self._stats_next = now + 5.0
        try:
            rq = self.__dict__.get("_rq")  # settled by get_finished just before
            payload = {
                "t": time.time(),
                "mono": rq["t"] if rq is not None else now,  # queue settled through
                "raw_row_bytes": int(self.raw_row_bytes),
                "stats": self.ftl.stats,
                "manager": dict(self.handle_manager.stats()),
            }
            workspace = getattr(self.codec, "workspace", None)
            if workspace is not None:  # keep the historical metric key
                payload["arena"] = dict(workspace.report())
            if rq is not None:
                payload["queue"] = {
                    "L": rq["L"],
                    "max": rq["max"],
                    "k": rq["k"],
                }
            temporary = f"{path}.{os.getpid()}.tmp"
            with open(temporary, "w") as output:
                json.dump(payload, output, sort_keys=True)
            os.replace(temporary, path)
        except Exception:
            pass

    def _page_ready_callback(self, bindings):
        if self.codec is not None:
            return lambda stream: self._register_ready_pages(bindings, stream)

    @marked(
        "rt.register_ready_pages",
        lambda self, bindings_by_group, stream: dict(
            groups=n(bindings_by_group),
            blocks=sum(n(v) for v in bindings_by_group.values()),
        ),
    )
    def _register_ready_pages(self, bindings_by_group, stream):
        """Deposit each bind's page ownership and generation in the GC-owned
        table (the mapping identity; no GPU work) with a readiness object:
        an event recorded after the transfer on its stream."""
        import torch

        event = torch.cuda.Event()
        event.record(stream)
        self.compaction_engine.record_bind_batch(
            bindings_by_group, event.query, lambda g: self.gc_by_group[int(g)]
        )
        return (event,)

    def _pcv_bind(self, entries):
        """States of one FTL bind batch: entries whose logical block is
        already bound (rebind) and superblocks the batch creates (a new
        handle, allocated through the storage callback)."""
        directory = self.ftl.directory
        rebinds = sum(
            1
            for entry in entries
            if directory.lookup(int(entry.logical_block)) is not None
        )
        decoder = self.load_engine.extent_decoder
        sizes = decoder.blocks_by_group if decoder is not None else None
        keys = set()
        completes = 0
        for entry in entries:
            group = int(entry.group)
            width = sizes[group] if sizes is not None else 0
            run = int(entry.ordinal) // width if width else 0
            keys.add(f"{entry.owner}/g{group}/{run}")
            completes += int(bool(width) and (int(entry.ordinal) + 1) % width == 0)
        superblocks = [directory.superblock(key) for key in keys]
        new_count = sum(superblock is None for superblock in superblocks)
        sb_blocks = sum(
            n(superblock.blocks) for superblock in superblocks if superblock is not None
        )
        return dict(
            rebinds=rebinds,
            new_superblocks=new_count,
            sb_blocks=sb_blocks,
            completes=completes,
            directory=int(directory.stats().get("superblocks", 0)),
        )

    def _pcv_plan(self, requested):
        """Superblocks a load plan touches and how many are sealed (encoded)."""
        directory = self.ftl.directory
        keys, sealed, sb_blocks = set(), set(), 0
        for entries in requested.values():
            for block, _destination in entries:
                lookup = directory.lookup(int(block))
                if lookup is None:
                    continue
                key = lookup[0].key
                if key not in keys:
                    keys.add(key)
                    sb_blocks += n(lookup[0].blocks)
                if lookup[0].sealed:
                    sealed.add(key)
        return dict(
            superblocks=len(keys),
            sealed=len(sealed),
            sb_blocks=sb_blocks,
            directory=int(directory.stats().get("superblocks", 0)),
        )

    def _sealed_handles(self, reads):
        directory = self.ftl.directory
        out = set()
        for read in reads:
            lookup = directory.lookup(int(read.blocks[0])) if read.blocks else None
            if lookup is not None and lookup[0].sealed:
                out.add(int(read.handle))
        return out

    @marked(
        "rt.refuse_store", lambda self, job_id, job, entries: dict(blocks=n(entries))
    )
    def _refuse_store(self, job_id, job, entries):
        """Unwind a capacity-refused store to stock not-cached semantics."""
        blocks = sorted({entry.logical_block for entry in entries})
        refuse_store_capacity(
            self.ftl,
            blocks,
            compaction_engine=self.compaction_engine,
            extent_encoder=self.extent_encoder,
        )
        job.refused = True
        if _TRACE:
            _trace(
                "[DITTO-STORE] job=%d req=%s REFUSED capacity: %d "
                "block(s) not cached (refusals=%d)"
                % (
                    job_id,
                    job.request_id,
                    len(blocks),
                    self.ftl.stats.get("store_refused_capacity", 0),
                )
            )

    def _report_verification(self):
        """Periodically surface the shared codec's roundtrip verification counters."""
        summary = self._codec_summary()
        if not summary:
            return
        self._verify_reports += 1
        if self._verify_reports % 500:
            return
        _trace(f"[DITTO-VERIFY] {summary}")

    def _mapped_blocks(self):
        """Live blocks, encoded blocks and RAW backlog from one directory snapshot.

        Physical RAW rows can outlive individual mappings until their handle
        retires, so storage row counts cannot determine encoded population.
        """
        live = encoded = 0
        try:
            raw_blocks: dict[int, int] = {}
            for key, superblock in self.ftl.directory.superblocks():
                count = len(superblock.blocks)
                live += count
                if superblock.sealed:
                    encoded += count
                    continue
                try:
                    group = int(str(key).rsplit("/g", 1)[1].split("/")[0])
                except (IndexError, ValueError):
                    continue  # a key shape this split cannot attribute
                raw_blocks[group] = raw_blocks.get(group, 0) + count
        except Exception:  # diagnostics must never break serving
            return -1, -1, ""
        mb = self.raw_row_bytes / 1e6
        groups = ",".join(
            f"g{group}:{count * mb:.0f}" for group, count in sorted(raw_blocks.items())
        )
        return live, encoded, f"rawg={groups} " if groups else ""

    def _report_pool(self):
        """Report physical storage use alongside logical compaction backlog."""
        self._pool_reports += 1
        if self._pool_reports % 500:
            return
        stats = self.handle_manager.stats()
        raw = int(stats.get("raw_bytes", 0))
        extents = int(stats.get("compressed_bytes", 0))
        used = int(stats.get("used_bytes", 0))
        # used counts CHUNKS held, so used - (raw + extents) is capacity that
        # is allocated but holds no live KV: slab tails that cannot return
        # their chunks until every row in them dies, plus landing reserves.
        stranded = used - raw - extents
        # Compaction counters alongside the pool split: a backlog that will not
        # drain is either not being NOMINATED, not getting resources, or being
        # ABORTED after the work is done. Those have opposite fixes, and the
        # pool bytes alone cannot tell them apart.
        gc = {
            name: value
            for name, value in self.ftl.stats.items()
            if name.startswith("compress")
        }
        if gc:
            _trace(
                "[DITTO-GC] "
                + " ".join(f"{name}={value}" for name, value in sorted(gc.items()))
            )
        live_blocks, extent_blocks, raw_by_group = self._mapped_blocks()
        _trace(
            f"[DITTO-POOL] live_blocks={live_blocks} "
            f"{raw_by_group}"
            f"extent_blocks={extent_blocks} "
            f"raw_MB={raw / 1e6:.0f} extent_MB={extents / 1e6:.0f} "
            f"used_MB={used / 1e6:.0f} stranded_MB={stranded / 1e6:.0f} "
            f"rows={stats.get('rows', 0)} slabs={stats.get('slabs', 0)} "
            f"chains={stats.get('chains', 0)} "
            f"extents={stats.get('compressed_extents', 0)} "
            f"landing_chunks={stats.get('landing_chunks', 0)}"
        )

    def _trace_store(self, job_id, request_id, entries):
        if not _TRACE:
            return
        traced = {}
        for entry in entries:
            traced.setdefault(entry.group, []).append(entry.position)
        _trace(
            "[DITTO-STORE] job=%d req=%s blocks/grp=%s first_tok/grp=%s"
            % (
                job_id,
                request_id,
                {group: len(values) for group, values in sorted(traced.items())},
                {
                    group: (values[0], values[-1])
                    for group, values in sorted(traced.items())
                },
            )
        )

    @marked(
        "rt.arm_load",
        lambda self, job_id, destinations: dict(
            blocks=sum(n(v) for v in destinations.by_group.values())
        ),
    )
    def _arm_load_completion(self, job_id, destinations):
        self.ftl.count(
            "loads", sum(len(entries) for entries in destinations.by_group.values())
        )
        if _TRACE:
            _trace(
                "[DITTO-LOAD] job=%d blocks/grp=%s partial_io=%d %s"
                % (
                    job_id,
                    {
                        group: len(values)
                        for group, values in sorted(destinations.by_group.items())
                    },
                    destinations.partial_pages,
                    self._codec_summary(),
                )
            )

    @marked(
        "rt.submit_store",
        lambda self, job_id, request: dict(
            entries=n(request.entries), groups=n(request.bindings)
        ),
    )
    def submit_store(self, job_id: int, request: StoreRequest) -> bool:
        if job_id in self.pending_jobs:
            raise RuntimeError(f"duplicate cache job {job_id}")
        entries = request.entries
        if self.extent_encoder is not None:
            with region("enc.record_tokens", lambda: dict(blocks=n(request.tokens))):
                for block, tokens in request.tokens.items():
                    self.extent_encoder.record_block_tokens(block, tokens)
        self.compaction_engine.invalidate_blocks(
            tuple(entry.logical_block for entry in entries)
        )
        with region(
            "ftl.bind_store_batch",
            lambda: dict(entries=n(entries), **pcv(lambda: self._pcv_bind(entries))),
        ):
            plan = self.ftl.bind_store_batch(entries)
        # Stores nominate complete superblocks too (not only reloads): under
        # the lease guard an encode whose pages the client reuses mid-flight is
        # ABORTED, never waited, so nominating early costs at most a
        # discarded encode, never a stalled forward.
        self.compaction_engine.trigger(plan.trigger_keys, source=1)
        self._trace_store(job_id, request.request_id, entries)
        job = PendingJob(is_load=False, request_id=request.request_id)
        self.pending_jobs[job_id] = job
        try:
            self.store_engine.wait_for_conflicting_transfers(
                (
                    pending.completion
                    for pending in self.pending_jobs.values()
                    if not pending.is_load
                ),
                plan.placements,
            )
            job.completion = self.store_engine.execute(
                job_id,
                request.source,
                plan.placements,
                after_transfer=self._page_ready_callback(request.bindings),
            )
        except StoreCapacityRefused:
            # Allocation raced the advisory admission limit: refuse THIS job
            # (its blocks become honest misses; the client recomputes) instead of
            # dying. No bytes moved — see StoreEngine.execute's abort point.
            self._refuse_store(job_id, job, entries)
        return True

    @marked(
        "rt.submit_load",
        lambda self, job_id, destinations: dict(
            blocks=sum(n(v) for v in destinations.by_group.values()),
            groups=n(destinations.by_group),
        ),
    )
    def submit_load(self, job_id: int, destinations: LoadRequest) -> bool:
        import time as _time

        _t0 = _time.perf_counter()
        if job_id in self.pending_jobs:
            raise RuntimeError(f"duplicate load job {job_id}")
        if destinations.partial_pages:
            self.ftl.count("load_partial_io", destinations.partial_pages)
        requested = destinations.by_group
        with region(
            "ftl.plan_load",
            lambda: dict(
                blocks=sum(n(v) for v in requested.values()),
                **pcv(lambda: self._pcv_plan(requested)),
            ),
        ):
            plan = self.ftl.plan_load(requested)
        if plan.unmapped:
            raise KeyError(f"unmapped FTL load blocks: {plan.unmapped[:8]}")
        self.compaction_engine.trigger(plan.trigger_keys)
        job = PendingJob(is_load=True)
        self.pending_jobs[job_id] = job
        try:
            submitted = _time.monotonic()
            if _perf_available:
                # declared-state hint for ftl.lock: handles of sealed (encoded)
                # superblocks, known to the directory before the lock
                self.load_engine._encoded_hint = pcv(
                    lambda: self._sealed_handles(plan.reads)
                )
            completion = self.load_engine.execute(plan.reads)
            job.completion = completion
            job.submitted = submitted
            rows = completion.codec_rows
            if rows:
                _restore_queue(self)["events"].append((submitted, 1, rows))
                job.codec_rows = rows
                self.ftl.count("decode_submitted_rows", rows)
        except Exception as error:
            reason = contain_load_refusal(self.ftl, job, error)
            if reason is None:
                raise
            refused_blocks = [
                block for entries in requested.values() for block, _ in entries
            ]
            self.compaction_engine.release_load_refused(refused_blocks)
            if reason == "capacity":
                # Tier saturation: the rows are unreachable until space
                # frees, and nothing else frees it. Make them honest misses
                # (the scheduler drops the keys when it sees the failed job).
                drop_refused_load(
                    self.ftl,
                    refused_blocks,
                    compaction_engine=self.compaction_engine,
                    extent_encoder=self.extent_encoder,
                )
            if _TRACE:
                _trace(
                    f"[DITTO-LOAD] job={job_id} REFUSED {reason}: "
                    "request will recompute"
                )
        if not job.refused:
            self._arm_load_completion(job_id, destinations)
            # CPU cost of planning + submitting one load, on the engine thread.
            self.ftl.count("load_submit_us", int((_time.perf_counter() - _t0) * 1e6))
            self.ftl.count("load_jobs", 1)
        return True

    @marked(
        "rt.poll",
        lambda self, allow_compression=True: dict(
            pending=n(self.pending_jobs),
            allow_compression=int(allow_compression),
            loads=sum(
                1
                for j in self.pending_jobs.values()
                if j.is_load and j.completion is not None
            ),
            # the restore queue exists after the first codec load: one more
            # settle call per poll from then on
            settle=int(self.__dict__.get("_rq") is not None),
        ),
    )
    def get_finished(self, *, allow_compression=True):
        """Collect completions; the client controls admission of new encodes."""
        if _TRACE:
            self._report_verification()
            self._report_pool()
        out = []
        # pending_jobs is in submission order and stores complete FIFO on one
        # D2H stream, so after the first store still in flight no later store
        # is checked (stock polls 2 events per step; a full scan polled every
        # job). Loads are checked individually: a codec part may finish out
        # of order on its own stream.
        stores_blocked = False
        for job_id, job in tuple(self.pending_jobs.items()):
            if job.refused:
                # A refused transfer takes the client's failure path: the scheduler
                # drops the job's keys from the offload pool, later lookups miss.
                out.append(
                    CacheCompletion(
                        job_id=job_id,
                        success=False,
                        transfer_size=None,
                        transfer_time=None,
                    )
                )
                del self.pending_jobs[job_id]
                continue
            completion = job.completion
            if completion is None or (not job.is_load and stores_blocked):
                continue
            if job.is_load and job.codec_rows and not job.codec_departed:
                # The codec part departs independently of a mixed load's RAW copy.
                with region(
                    "rt.query",
                    lambda completion=completion: dict(
                        load=1,
                        parts=0,
                        codec=1,
                        released=int(getattr(completion, "released", False)),
                    ),
                ):
                    codec_done = completion.codec_done()
                if codec_done:
                    self._depart_restore(job, completion)
            with region(
                "rt.query",
                lambda completion=completion, job=job: dict(
                    load=int(job.is_load),
                    parts=n(getattr(completion, "raw", ())) if job.is_load else 1,
                    codec=int(getattr(completion, "codec_end", None) is not None),
                    released=int(getattr(completion, "released", False)),
                ),
            ):
                done = completion.done()
            if not done:
                stores_blocked |= not job.is_load
                continue
            submitted = job.submitted
            if job.is_load and submitted:
                import time as _time

                # submit -> observed done, as the waiting request sees it
                self.ftl.count(
                    "load_latency_us", int((_time.monotonic() - submitted) * 1e6)
                )
                self.ftl.count("load_done", 1)
            with region(
                "rt.finish",
                lambda completion=completion, job=job: dict(
                    load=int(job.is_load),
                    parts=n(getattr(completion, "raw", ())),
                    codec=int(getattr(completion, "codec_end", None) is not None),
                ),
            ):
                transfer_size, transfer_time = completion.finish()
            out.append(
                CacheCompletion(
                    job_id=job_id,
                    success=True,
                    transfer_size=transfer_size,
                    transfer_time=transfer_time,
                )
            )
            del self.pending_jobs[job_id]
        rq = self.__dict__.get("_rq")
        if rq is not None:  # every departure up to now is known after a poll
            import time as _time

            with region("rt.queue_settle", lambda: dict(events=n(rq["events"]))):
                _queue_settle(self, rq, _time.monotonic())
        self.compaction_engine.run_gc_trigger(allow_launch=allow_compression)
        if self._stats_path:  # opt-in stats file
            self._maybe_write_stats()
        return out

    @marked(
        "rt.take_lease_releases",
        lambda self: dict(releases=n(self.compaction_engine._lease_releases)),
    )
    def take_lease_releases(self):
        """Return completed source leases separately from transfer results."""
        return self.compaction_engine.take_lease_releases()

    @marked("rt.wait", lambda self, job_ids: dict(jobs=n(job_ids)))
    def wait(self, job_ids) -> None:
        for job_id in job_ids:
            job = self.pending_jobs.get(job_id)
            if job is None:
                continue
            completion = job.completion
            if completion is not None:
                completion.wait()

    def shutdown(self) -> None:
        self.wait(tuple(self.pending_jobs))
        self.compaction_engine.shutdown()
        for job in self.pending_jobs.values():
            completion = job.completion
            if completion is not None:
                completion.finish()
        self.pending_jobs.clear()
        workspace = getattr(self.codec, "workspace", None)
        if workspace is not None:
            workspace.drain()
        if self.carrier is not None:
            self.carrier.shutdown()
        self.carrier = None
        self.handle_manager = None

    @marked(
        "rt.add_lease_ids",
        lambda self, values: dict(
            leases=n(values),
            poisoned=sum(k in self.compaction_engine.poisoned for k in values),
        ),
    )
    def add_lease_ids(self, values):
        self.compaction_engine.add_lease_ids(values)

    @marked("rt.flush_source_lease")
    def flush_source_lease(self, reader_id):
        """Invalidate source reads before the client reuses leased pages."""
        self.compaction_engine.on_lease_flush(reader_id)

    @marked(
        "rt.invalidate_checked",
        lambda self, invalidations: dict(blocks=n(invalidations)),
    )
    def invalidate_checked(self, invalidations):
        """Apply client retirement before reused ids are rebound."""
        invalidate_store_blocks(
            self.ftl,
            invalidations,
            compaction_engine=self.compaction_engine,
            extent_encoder=self.extent_encoder,
        )

    @marked("rt.admission_limit")
    def admission_limit(self):
        return self.ftl.admission_limit()
