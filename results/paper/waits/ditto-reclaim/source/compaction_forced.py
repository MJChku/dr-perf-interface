"""GC-owned compaction transaction for the representation-neutral FTL."""

import os
import sys
from dataclasses import dataclass, field
from enum import Enum, auto

from _ditto_ftl_core import WorkspaceExhausted

from src.perf import marked, n, pcv, region

_TRACE = os.environ.get("DITTO_DEBUG_FIN") == "1"


def _trace(message):
    print(message, file=sys.stderr, flush=True)


class CompactionPhase(Enum):
    # Preparation owns a landing and a stable logical block-set snapshot.
    PREPARED = auto()
    # Codec source reads and deferred blob production have been enqueued.
    ENCODE_SUBMITTED = auto()
    # Opaque codec blobs are being copied into the locked landing.
    OUTPUT_SUBMITTED = auto()
    # The landing reserve-then-trim lifecycle produced a committed extent.
    COMMITTED = auto()
    # Terminal cleanup released codec and landing capabilities.
    RELEASED = auto()


@dataclass
class Compaction:
    """GC-owned mutable state for one checked compaction lifecycle."""

    # The landing handle exists before encoded sizes determine its final trim.
    storage_handle: object
    # Freeze codec units, layer ordering, and the block set used by publication.
    encode_units: tuple
    # Layer ids connect codec units to opaque extent stream keys at the boundary.
    layer_ids: tuple
    # The stable superblock handle adopts the committed landing chain.
    target_handle: object
    # GC's immutable source witness; the extent bridge does not interpret it.
    candidate: object = None
    # Phase makes asynchronous resource ownership and legal cleanup explicit.
    phase: CompactionPhase = CompactionPhase.PREPARED
    # A flushed source lease invalidates only this member's result.
    aborted: bool = False
    # Adopt-ready descriptors include codec-computed stream offsets and sizes.
    stream_descriptors: tuple = ()
    # The codec-computed trim retains precisely the packed stream bytes.
    trim_bytes: int = 0
    # The landing stays locked while D2H owns its possibly fragmented addresses.
    locked_landing: object = None

    def require(self, *allowed):
        """Reject an illegal lifecycle transition before resources move."""
        if self.phase.name not in allowed:
            raise RuntimeError(
                f"compaction is {self.phase.name}; expected {', '.join(allowed)}"
            )

    def advance(self, expected, target):
        """Move one lifecycle edge after checking its required predecessor."""
        self.require(expected)
        self.phase = CompactionPhase[target]

    def is_phase(self, phase):
        return self.phase is CompactionPhase[phase]


@dataclass(frozen=True, eq=False)
class SourceBinding:
    """One bind generation; equal contents never make two bindings identical."""

    key: str
    block: int
    group: int
    pages: tuple
    ready: object  # callable querying the bind's transfer completion


@dataclass(frozen=True)
class _Candidate:
    key: str
    group: int
    handle: object
    blocks: tuple
    pages: tuple
    sources: tuple[SourceBinding, ...]
    position_base: int | None
    # the reader lease this candidate was nominated under, so its release
    # names THAT lease even if the key has since been re-leased (a later
    # store), never the newer one.
    lease: object = None


@dataclass(eq=False)
class _Batch:
    """Superblocks sharing one codec submission to amortize submission cost.

    Each member owns its handle, landing, units, witness, and publication;
    one member can abort while its siblings publish.
    """

    members: list
    output: object
    failed: bool = False
    # GPU-event timing of the whole batch (encode + output copy); counted once
    start: object = None
    timed: bool = False
    # Batch-wide because the deferred D2H is one submission; the members'
    # publication gates stay individual.
    # None while sizes are pending, then the shared output-copy fence.
    event: object = None


def _owner(key):
    """Client owner before the FTL key's final ``/g{group}/{run}[#shadow]``.

    Owners are opaque names and may themselves contain ``/g``.
    """
    return str(key).rsplit("/g", 1)[0]


@dataclass
class CompactionResources:
    """Storage and source-lifetime capabilities needed by compaction."""

    ftl: object
    handle_manager: object
    codec: object
    extent_encoder: object
    pages_per_block: int
    lease_by_key: dict = field(default_factory=dict)
    key_by_lease: dict = field(default_factory=dict)


class CompactionEngine:
    """Own candidate selection, source ownership, and atomic publication."""

    def __init__(self, resources: CompactionResources):
        self.resources = resources
        # Values cache the complete group's size; zero waits for a later bind/trigger.
        self.to_compact: dict = {}
        self._queue_snapshot = ()
        self._queue_sorted = ()
        self.compacting: set = set()
        self.pending_compactions: list = []
        # Source residency and page ownership are GC-private, never FTL metadata.
        self.source_bindings: dict = {}
        self.page_owners: dict = {}
        self._lease_releases: list = []
        # key -> lease whose flush poisoned it: its GPU pages were reallocated,
        # so no bind is accepted and no candidate formed from them until a
        # NEW lease (a new store) arrives for the key.
        self.poisoned: dict = {}
        # key -> [attempts, aborts] under the lease guard: how often this
        # superblock's compaction was discarded because the client reused its
        # pages mid-encode. Bounded like `poisoned`.
        self.lease_stats: dict = {}
        # owner -> {lease: True} (insertion-ordered) whose release is
        # DEFERRED because another FTL child of the owner is still bound,
        # queued or in flight. Several generations can be pending at once
        # (an old candidate's abort after a rebind minted a new id), so each
        # id is tracked and removed exactly. Drained whenever children can
        # have gone (publish loop end, flush, invalidation, bind).
        self._deferred: dict = {}
        self.compaction_stream = None
        self.compaction_copy_stream = None
        # Bound codec submissions per poll. Each submission can contain multiple
        # superblocks; the landing reserve separately bounds in-flight members.
        self.resident_per_poll = max(
            0, int(os.environ.get("DITTO_XJOB_TARGETS_PER_POLL", "2"))
        )

    @staticmethod
    def _ordered_blocks(superblock):
        if superblock is None or not superblock.complete:
            return None
        blocks = superblock.blocks  # one snapshot copy, not one per offset
        try:
            return tuple(
                int(blocks[offset]) for offset in range(int(superblock.nblocks))
            )
        except KeyError:
            return None

    def _source_binding(self, key, block):
        entry = self.source_bindings.get(block)
        return entry if entry is not None and entry.key == key else None

    def _key_for_block(self, block):
        lookup = self.resources.ftl.directory.lookup(int(block))
        entry = (
            lookup[0] if lookup is not None else self.source_bindings.get(int(block))
        )
        return entry.key if entry is not None else None

    def _drop_entry(self, entry):
        if self.source_bindings.get(entry.block) is not entry:
            return
        del self.source_bindings[entry.block]
        for page in entry.pages:
            owner = (entry.group, int(page))
            if page >= 0 and self.page_owners.get(owner) is entry:
                self.page_owners.pop(owner, None)

    def _clear_source_bindings(self, key):
        for entry in tuple(self.source_bindings.values()):
            if entry.key == key:
                self._drop_entry(entry)

    def _drop_key(self, key):
        # No lease release here. A rebind drops the OLD generation's GC state,
        # but the key's current lease may already be fencing the NEW blocks
        # (same id reused, or a fresh id installed just before this bind);
        # releasing it would unfence pages a compaction can still read.
        # Leases are released only by exact-generation paths: publish, abort,
        # and flush.
        self.to_compact.pop(key, None)
        self._clear_source_bindings(key)

    @marked(
        "gc.invalidate_blocks",
        lambda self, logical_blocks, *, terminal=False: dict(
            blocks=n(logical_blocks), bindings=n(self.source_bindings)
        ),
    )
    def invalidate_blocks(self, logical_blocks, *, terminal=False):
        """Drop GC state before the bridge invalidates logical mappings.

        `terminal=False` is the REBIND path (a re-store of the same blocks
        follows at once): the owner's lease must stay held, the replacement
        pages are the next source. `terminal=True` (capacity refusal, a
        mapping that is really gone) hands back a lease left with no child."""
        if self.resources.codec is None:
            return
        self._drop_block_sources(logical_blocks, release_idle=terminal)

    def _drop_block_sources(self, logical_blocks, *, release_idle):
        keys = {self._key_for_block(block) for block in logical_blocks} - {None}
        for key in keys:
            self._drop_key(key)
        if release_idle:
            for owner in {_owner(key) for key in keys}:
                self.release_if_idle(owner)
            self._drain_deferred()

    def _pcv_trigger(self, keys):
        """Keys nominated, the blocks bound in their superblocks (a directory
        snapshot copies them) and how many are complete (only those get
        their block order read back)."""
        directory = self.resources.ftl.directory
        blocks = complete = queued = published = 0
        for key in keys:
            key = str(key)
            superblock = directory.superblock(key)
            if superblock is not None:
                blocks += n(superblock.blocks)
                complete += int(bool(superblock.complete))
            queued += int(key in self.to_compact or key in self.compacting)
            published += int(superblock is not None and superblock.sealed)
        return dict(
            keys=n(keys),
            blocks=blocks,
            complete=complete,
            queued=queued,
            published=published,
        )

    def _pcv_source_binding(self, group, logical_block, pages):
        logical_block = int(logical_block)
        lookup = self.resources.ftl.directory.lookup(logical_block)
        skip = stale = blocks = 0
        if lookup is None:
            skip = 1
        else:
            superblock = lookup[0]
            key = superblock.key
            ordered = self._ordered_blocks(superblock)
            blocks = n(ordered)
            if _owner(key) in self.poisoned or superblock.sealed:
                skip = 1
            identity = (key, logical_block)
            for page in pages:
                prior = self.page_owners.get((int(group), int(page)))
                stale += int(prior is not None and (prior.key, prior.block) != identity)
        return dict(
            pages=n(pages),
            blocks=blocks,
            rebind=int(logical_block in self.source_bindings),
            skip=skip,
            stale=stale,
        )

    def _bound_sources(self, key):
        if _owner(key) in self.poisoned:
            return None
        superblock = self.resources.ftl.directory.superblock(key)
        blocks = self._ordered_blocks(superblock)
        if not blocks or superblock.sealed:
            return None
        entries = tuple(self._source_binding(key, block) for block in blocks)
        if any(entry is None for entry in entries):
            return None
        return superblock, blocks, entries

    def _sources_ready(self, entries):
        group = entries[0].group
        width = int(self.resources.pages_per_block)
        for entry in entries:
            if entry.group != group or len(entry.pages) != width:
                return False
            if any(
                page < 0 or self.page_owners.get((group, page)) is not entry
                for page in entry.pages
            ):
                return False
        return all(entry.ready() for entry in entries)

    def _pcv_batches(self, keys, limit):
        """Count complete sources and the prefix scanned for leased, ready work."""
        complete = ready = 0
        evaluated = len(keys) if limit > 0 else 0
        for index, key in enumerate(keys):
            sources = self._bound_sources(key)
            if sources is None:
                continue
            complete += 1
            if ready >= limit:
                continue
            if (
                self._sources_ready(sources[2])
                and _owner(key) in self.resources.lease_by_key
            ):
                ready += 1
                if ready == limit:
                    evaluated = index + 1
        return dict(
            keys=len(keys),
            limit=limit,
            complete=complete,
            ready=ready,
            evaluated=evaluated,
        )

    def _pcv_candidate(self, key):
        if _owner(key) in self.poisoned:
            return dict(poisoned=1, blocks=0, entries=0)
        superblock = self.resources.ftl.directory.superblock(key)
        blocks = self._ordered_blocks(superblock) or ()
        entries = sum(
            1 for block in blocks if self._source_binding(key, block) is not None
        )
        return dict(poisoned=0, blocks=n(blocks), entries=entries)

    def _pcv_publish(self):
        """Batches by phase; members whose bytes landed get the
        per-superblock gates and the adopt (their block counts size the
        witness compare), the rest wait."""
        sizes = landed = ready = ready_blocks = aborted = untimed = 0
        for batch in self.pending_compactions:
            if batch.event is None:
                sizes += 1
            elif batch.event.query():
                landed += n(batch.members)
                # the first poll that sees a batch landed times its encode
                untimed += int(not batch.timed)
                for pending in batch.members:
                    candidate = pending.candidate
                    ready += 1
                    ready_blocks += n(candidate.blocks)
                    aborted += int(pending.aborted)
        return dict(
            batches=n(self.pending_compactions),
            members=sum(n(b.members) for b in self.pending_compactions),
            sizes=sizes,
            landed=landed,
            ready=ready,
            ready_blocks=ready_blocks,
            aborted=aborted,
            untimed=untimed,
            bindings=n(self.source_bindings),
            tracked=n(self.source_bindings),
            queued=n(self.to_compact),
        )

    def _pcv_lease_flush(self, lease):
        owner = self.resources.key_by_lease.get(lease)
        keys = self._owner_keys(owner)
        members = sum(
            1
            for batch in self.pending_compactions
            for m in batch.members
            if _owner(m.candidate.key) == owner
        )
        return dict(
            queued=n(self.to_compact),
            tracked=n(self.source_bindings),
            pending=n(self.pending_compactions),
            owner_keys=n(keys),
            owner_bindings=sum(
                1 for entry in self.source_bindings.values() if entry.key in keys
            ),
            members=members,
            bindings=n(self.source_bindings),
        )

    @marked(
        "gc.trigger",
        lambda self, keys, *, source=0: pcv(
            lambda: dict(self._pcv_trigger(keys), source=int(source))
        ),
    )
    def trigger(self, keys, *, source=0):
        """Enqueue only operation-touched keys supplied by the controller.

        `source` only names the nominating operation for the measurement
        (1 a store bind, 0 a load plan); it changes nothing here."""
        del source
        if self.resources.codec is None:
            return
        for key in keys:
            key = str(key)
            superblock = self.resources.ftl.directory.superblock(key)
            if superblock is None:
                self._drop_key(key)
                continue
            if not superblock.sealed and key not in self.compacting:
                self.to_compact[key] = (
                    superblock.nblocks
                    if superblock.complete and _owner(key) not in self.poisoned
                    else 0
                )

    @marked(
        "gc.record_source_binding",
        lambda self, group, logical_block, pages, ready: pcv(
            lambda: self._pcv_source_binding(group, logical_block, pages)
        ),
    )
    def record_source_binding(self, group, logical_block, pages, ready):
        """Install one bind generation at the transfer-stream capture point;
        `ready` answers when the bound bytes have landed (the D2H event)."""
        logical_block = int(logical_block)
        lookup = self.resources.ftl.directory.lookup(logical_block)
        if lookup is None:
            return False
        superblock, _offset = lookup
        key = superblock.key
        if _owner(key) in self.poisoned:  # pages reallocated under a flushed lease
            return False
        if superblock.sealed:
            return False
        previous = self.source_bindings.get(logical_block)
        if previous is not None:
            self._drop_entry(previous)
        pages = tuple(int(page) for page in pages)
        # A GPU page can authenticate only its latest logical owner.
        for page in pages:
            if page < 0:
                continue
            prior = self.page_owners.get((int(group), page))
            if prior is not None:
                self.to_compact.pop(prior.key, None)
                self._drop_entry(prior)
        entry = SourceBinding(key, logical_block, int(group), pages, ready)
        self.source_bindings[logical_block] = entry
        for page in pages:
            if page >= 0:
                self.page_owners[(int(group), page)] = entry
        return True

    @marked("gc.candidate", lambda self, key: pcv(lambda: self._pcv_candidate(key)))
    def _candidate(self, key):
        sources = self._bound_sources(key)
        if sources is None:
            return None
        superblock, blocks, entries = sources
        if not self._sources_ready(entries):
            return None
        lease = self.resources.lease_by_key.get(_owner(key))
        if lease is None:
            # Fail closed: a candidate with no active lease would be an
            # UNFENCED read (nothing flushes it). Wait for the next leased
            # bind instead of compacting without a fence.
            self.resources.ftl.count("compress_lease_missing")
            return None
        return _Candidate(
            key=key,
            group=entries[0].group,
            handle=superblock.handle,
            blocks=blocks,
            pages=tuple(page for entry in entries for page in entry.pages),
            sources=entries,
            position_base=superblock.position_base,
            lease=lease,
        )

    @marked(
        "gc.prepare_compaction",
        lambda self, candidate: dict(blocks=n(candidate.blocks)),
    )
    def prepare_compaction(self, candidate):
        """Freeze the witness and reserve one landing before codec work."""
        resources = self.resources
        if candidate.key in self.compacting:
            return None
        landing = resources.handle_manager.alloc_landing()
        if landing is None:
            resources.ftl.count("compress_no_space")
            return None
        self.compacting.add(candidate.key)
        self.to_compact.pop(candidate.key, None)
        try:
            prepared_encode = resources.extent_encoder.prepare_compaction(
                group=candidate.group,
                blocks=candidate.blocks,
                pages=candidate.pages,
                target_handle=candidate.handle,
                position_base=candidate.position_base,
            )
        except Exception:
            self.compacting.discard(candidate.key)
            resources.handle_manager.free(landing)
            raise
        if prepared_encode is None:
            self.compacting.discard(candidate.key)
            resources.handle_manager.free(landing)
            resources.ftl.count("compress_missing_tokens")
            return None
        encode_units, layer_ids = prepared_encode
        return Compaction(
            landing,
            tuple(encode_units),
            tuple(layer_ids),
            candidate.handle,
            candidate=candidate,
        )

    def _ensure_streams(self):
        if self.compaction_stream is None or self.compaction_copy_stream is None:
            import torch

            self.compaction_stream = torch.cuda.Stream(priority=0)
            self.compaction_copy_stream = torch.cuda.Stream(priority=0)

    @marked("gc.submit_batch", lambda self, prepared: dict(members=n(prepared)))
    def submit_batch(self, prepared):
        """Submit one encode for the batch; every member publishes behind its
        own witness (mapping, ordered blocks, bind generations) under the
        reader lease that fences its source pages."""
        resources = self.resources
        prepared = tuple(prepared)
        try:
            self._ensure_streams()
            start = self._timing_event(self.compaction_stream)
            output = resources.extent_encoder.submit_batch(
                prepared,
                self.compaction_stream,
                self.compaction_copy_stream,
            )
        except WorkspaceExhausted:
            # Deferred phase-A chunks own their arena slabs until output is
            # copied or cancelled. The codec has already unwound any partial
            # result; synchronize source work before releasing the landings
            # and treating this as an ordinary GC refusal.
            self._refuse_batch(
                prepared,
                "compress_workspace_exhausted",
                "WORKSPACE_EXHAUSTED",
                requeue=True,
            )
            return None
        except Exception:
            self._refuse_batch(prepared, None, None, requeue=False)
            raise
        for entry in prepared:
            self._note_lease_stat(_owner(entry.candidate.key), attempts=1)
        self.pending_compactions.append(_Batch(list(prepared), output, start=start))
        return True

    @marked(
        "gc.refuse_batch",
        lambda self, prepared, counter, reason, *, requeue: dict(members=n(prepared)),
    )
    def _refuse_batch(self, prepared, counter, reason, *, requeue):
        """Drain a failed submission and release every member's landing."""
        resources = self.resources
        if self.compaction_stream is not None:
            self.compaction_stream.synchronize()
        if self.compaction_copy_stream is not None:
            self.compaction_copy_stream.synchronize()
        for entry in prepared:
            key = entry.candidate.key
            self.compacting.discard(key)
            if requeue:
                self.to_compact.setdefault(key, len(entry.candidate.blocks))
            resources.extent_encoder.release(entry)
            if counter is not None:
                resources.ftl.count(counter)
            if reason is not None and _TRACE:
                _trace(f"[DITTO-COMPACT] sb={key} outcome=REJECTED reason={reason}")

    @marked(
        "gc.witness_matches", lambda self, candidate: dict(blocks=n(candidate.blocks))
    )
    def _witness_matches(self, candidate):
        superblock = self.resources.ftl.directory.superblock(candidate.key)
        if (
            superblock is None
            or superblock.handle != candidate.handle
            or superblock.position_base != candidate.position_base
        ):
            return False
        if self._ordered_blocks(superblock) != candidate.blocks:
            return False
        # Every bind creates a fresh immutable object, including a rebind to
        # identical pages. Retained identities are the generation witness.
        return candidate.sources == tuple(
            self.source_bindings.get(block) for block in candidate.blocks
        )

    def _owner_keys(self, owner):
        """Bound and queued keys; in-flight members are handled separately."""
        keys = {
            entry.key
            for entry in self.source_bindings.values()
            if _owner(entry.key) == owner
        }
        keys.update(k for k in self.to_compact if _owner(k) == owner)
        return keys

    def _owner_children(self, owner):
        """Every FTL superblock of `owner` still bound, queued or in flight."""
        keys = self._owner_keys(owner)
        for batch in self.pending_compactions:
            keys.update(
                m.candidate.key
                for m in batch.members
                if _owner(m.candidate.key) == owner
            )
        return keys

    def _release_lease(self, key, lease):
        """Queue (owner, lease) for the worker to report, once the owner has
        NO other FTL superblock still bound, queued or in flight: one lease
        fences every child of an owner (groups, shadows, a rebind under the
        same id), so releasing on the first child's publish would unfence a
        sibling mid-encode. `lease` is the exact generation being released."""
        owner = _owner(key)
        if self._owner_children(owner):
            self._deferred.setdefault(owner, {})[lease] = True
            self.resources.ftl.count("compress_lease_release_deferred")
        else:
            self._emit_releases(owner, lease)
        # Poison is NEVER cleared here: only a demonstrably fresh lease
        # (resources.add_lease_ids -> unpoison) may re-admit the owner's pages.

    def _pcv_bind_batch(self, bindings_by_group):
        """Blocks bound and the blocks already bound in the superblocks they
        map to: each directory lookup returns a snapshot of its superblock."""
        directory = self.resources.ftl.directory
        blocks = sb_blocks = rejected = 0
        accepted_by_owner = {}
        for bindings in bindings_by_group.values():
            for logical_block, _pages in bindings:
                blocks += 1
                lookup = directory.lookup(int(logical_block))
                if lookup is None:
                    rejected += 1
                    continue
                superblock = lookup[0]
                sb_blocks += n(superblock.blocks)
                owner = _owner(superblock.key)
                refused = owner in self.poisoned or superblock.sealed
                rejected += int(refused)
                accepted_by_owner[owner] = accepted_by_owner.get(owner, False) or (
                    not refused
                )
        return dict(
            groups=n(bindings_by_group),
            blocks=blocks,
            sb_blocks=sb_blocks,
            owners=len(accepted_by_owner),
            # owners none of whose bindings are accepted get their idle
            # lease handed back here (one nested release per such owner)
            idle_owners=sum(1 for ok in accepted_by_owner.values() if not ok),
            rejected=rejected,
        )

    @marked(
        "gc.record_bind_batch",
        lambda self, bindings_by_group, ready, eligible: pcv(
            lambda: self._pcv_bind_batch(bindings_by_group)
        ),
    )
    def record_bind_batch(self, bindings_by_group, ready, eligible):
        """record_source_binding for a whole bind; an owner none of whose
        bindings were accepted (a reload of a published extent) gets its
        idle lease handed back at once. Directory reads stay in GC."""
        owners = {}
        for group, bindings in bindings_by_group.items():
            ok = eligible(group)
            for logical_block, pages in bindings:
                accepted = ok and self.record_source_binding(
                    group, logical_block, pages, ready
                )
                lookup = self.resources.ftl.directory.lookup(int(logical_block))
                if lookup is not None:
                    owner = _owner(lookup[0].key)
                    owners[owner] = owners.get(owner, False) or bool(accepted)
        for owner, any_accepted in owners.items():
            if not any_accepted:  # ineligible groups or a published extent
                self.release_if_idle(owner)
        self._drain_deferred()

    @marked(
        "gc.release_if_idle",
        lambda self, owner: pcv(
            lambda: dict(
                children=n(self._owner_children(owner)),
                tracked=n(self.source_bindings),
                queued=n(self.to_compact),
                pending=n(self.pending_compactions),
            )
        ),
    )
    def release_if_idle(self, owner):
        """A bind whose every superblock was refused (already published,
        ineligible) leaves the owner's lease fencing pages that are no source:
        hand it back now instead of at page reuse."""
        if self._owner_children(owner):
            return
        current = self.resources.lease_by_key.get(owner)
        self._emit_releases(owner, current)

    def _emit_releases(self, owner, lease):
        """Report every deferred generation of `owner` plus `lease` (if any
        and not already among them), each exactly once."""
        pending = self._deferred.pop(owner, {})
        if lease is not None:
            pending[lease] = True
        for value in pending:
            self._enqueue_release(owner, value)

    def _enqueue_release(self, owner, lease):
        """The single gate to the scheduler: each (owner, lease) once."""
        if lease is None or self.resources.key_by_lease.get(lease) != owner:
            return
        del self.resources.key_by_lease[lease]
        if self.resources.lease_by_key.get(owner) == lease:
            del self.resources.lease_by_key[owner]
        self._lease_releases.append((owner, lease))

    @marked("gc.drain_deferred", lambda self: dict(owners=n(self._deferred)))
    def _drain_deferred(self):
        """Emit every deferred release whose owner has no child left."""
        for owner in tuple(self._deferred):
            if not self._owner_children(owner):
                self._emit_releases(owner, None)

    @marked(
        "gc.release_load_refused",
        lambda self, logical_blocks: dict(blocks=n(logical_blocks)),
    )
    def release_load_refused(self, logical_blocks):
        """A refused load leaves its destination pages holding nothing (or,
        for a mixed RAW+encoded load, RAW children bound before the encoded
        part failed): the client recomputes into those pages WITHOUT reallocating
        them, so no flush would ever fence them. Drop the GC bindings of
        every requested superblock, then hand back the leases left idle.
        The FTL mappings stay: the host copy is intact and a later reload
        binds afresh. Directory reads stay in GC."""
        self._drop_block_sources(logical_blocks, release_idle=True)

    def take_lease_releases(self):
        out, self._lease_releases = self._lease_releases, []
        return out

    def add_lease_ids(self, values):
        """Register live generations; releasing one removes its maps once."""
        for key, lease in values.items():
            self.resources.lease_by_key[key] = lease
            self.resources.key_by_lease[lease] = key
            self.unpoison(key, lease)

    def unpoison(self, key, lease):
        if self.poisoned.get(key) not in (None, lease):
            self.poisoned.pop(key, None)

    @marked(
        "gc.lease_flush", lambda self, lease: pcv(lambda: self._pcv_lease_flush(lease))
    )
    def on_lease_flush(self, lease):
        """the client will let a reallocated source page be written after this
        returns. Whatever state the superblock is in, its pages are no longer
        a valid source: poison the key so no bind installs them and no
        candidate forms from them until a NEW lease (new store) arrives.
        A submitted compaction is marked ABORTED and nothing waits: the
        forward overwrites the pages at once, the encode keeps reading them
        (valid device memory, garbage output), and its result is discarded
        when it completes. Inference never waits on GC. Anything else is
        dropped."""
        owner = self.resources.key_by_lease.get(lease)
        if owner is None:
            return
        current = self.resources.lease_by_key.get(owner)
        stale = current is not None and current != lease
        # every FTL superblock the scheduler's name covers (per group / run /
        # shadow); Qwen has exactly one, hybrid models one per group
        keys = self._owner_keys(owner)
        # A flush for a SUPERSEDED lease (a newer store re-leased the key --
        # e.g. old flushes dispatched after new ids were installed on reset)
        # must not poison or clear the new generation's state. It still
        # finishes any in-flight read that was nominated under it.
        if not stale:
            self._poison(owner, lease)
            for k in keys:
                self.to_compact.pop(k, None)
        # An in-flight compaction nominated under THIS lease is marked
        # aborted; its result is discarded at completion, which is also where
        # the landing and workspace come back. One nominated under an older
        # lease for the same key is left alone: the bind that created this
        # lease already bumped the key's generation, so its publication fails
        # _witness_matches on its own.
        aborted = newly = False
        for batch in self.pending_compactions:
            for m in batch.members:
                c = m.candidate
                if _owner(c.key) != owner or c.lease != lease:
                    continue
                if not m.aborted:
                    m.aborted = True
                    newly = True
                aborted = True
        if aborted:
            self.compaction_stream.synchronize()  # Deliberately restored blocking fence.
            if newly:  # a redelivered flush must not count twice
                self._note_lease_stat(owner, aborts=1)
                self.resources.ftl.count("compress_lease_abort")
            return  # the discard at completion releases its own lease
        if stale:
            # nothing of this generation is left to drop; retire the reverse
            # mapping so resets on active requests do not accumulate it
            self.resources.key_by_lease.pop(lease, None)
            return
        for k in keys:
            self._clear_source_bindings(k)  # queued OR incomplete
        if owner in self._deferred:
            self._deferred[owner].pop(lease, None)  # the flush releases it below
            if not self._deferred[owner]:
                self._deferred.pop(owner, None)
        self._enqueue_release(owner, lease)  # once, even if redelivered
        self.resources.ftl.count("compress_lease_dropped")

    _POISON_CAP = 8192

    def _poison(self, key, lease):
        """Bounded: a poison only has to outlive the flushed store's own bind
        (next step); anything older than 8192 later flushes is dead weight."""
        self.poisoned.pop(key, None)
        self.poisoned[key] = lease
        while len(self.poisoned) > self._POISON_CAP:
            self.poisoned.pop(next(iter(self.poisoned)))

    _LEASE_STATS_CAP = 8192

    def _note_lease_stat(self, key, *, attempts=0, aborts=0):
        stat = self.lease_stats.pop(key, None) or [0, 0]
        stat[0] += attempts
        stat[1] += aborts
        self.lease_stats[key] = stat
        while len(self.lease_stats) > self._LEASE_STATS_CAP:
            self.lease_stats.pop(next(iter(self.lease_stats)))

    def abort_stats(self, key):
        """(attempts, aborts) of one superblock's compactions under the lease
        guard; aborts/attempts is its abort rate."""
        attempts, aborts = self.lease_stats.get(_owner(key), (0, 0))
        return int(attempts), int(aborts)

    def _finish_member(self, batch, pending):
        """Retire ownership before diagnostics or another member can fail."""
        self.resources.extent_encoder.release(pending)
        batch.members.remove(pending)
        key = pending.candidate.key
        self.compacting.discard(key)
        self.to_compact.pop(key, None)
        for source in pending.candidate.sources:
            self._drop_entry(source)
        # A reload may have rebound every source while this result was pending.
        # Its nomination was suppressed by compacting; preserve and queue it now.
        if self._bound_sources(key) is not None:
            self.trigger((key,))
        self._release_lease(key, pending.candidate.lease)

    @marked("gc.abort")
    def _abort(self, batch, pending, reason, counter=None):
        if pending.aborted:
            reason = "LEASE_ABORTED"
        self._finish_member(batch, pending)
        if counter is not None:
            self.resources.ftl.count(counter)
        if _TRACE:
            key = pending.candidate.key
            extra = ""
            if reason == "LEASE_ABORTED":
                attempts, aborts = self.abort_stats(key)
                extra = f" aborts={aborts}/{attempts}"
            _trace(f"[DITTO-COMPACT] sb={key} outcome=REJECTED reason={reason}{extra}")

    def _release_batch(self, batch):
        # Keep ownership if cancellation fails, so the next poll can retry.
        if batch.output is not None:
            self.resources.extent_encoder.release_batch(batch.output)
            batch.output = None

    def _abort_batch(self, batch):
        self.compaction_copy_stream.synchronize()
        self._release_batch(batch)
        for pending in tuple(batch.members):
            self._abort(batch, pending, "OUTPUT_ERROR")
        self.pending_compactions.remove(batch)

    @staticmethod
    def _timing_event(stream):
        """A timing event recorded on `stream`, or None off-GPU (tests)."""
        try:
            import torch

            if not isinstance(stream, torch.cuda.Stream):
                return None
            event = torch.cuda.Event(enable_timing=True)
            event.record(stream)
            return event
        except Exception:
            return None

    def _count_encode(self, batch):
        """Achieved ENCODE throughput of one batch: io-block rows in, GPU-event
        time from submission to the landed output copy."""
        start, end = batch.start, batch.event
        if start is None or not hasattr(end, "elapsed_time"):
            return
        try:
            elapsed_ms = float(start.elapsed_time(end))
        except Exception:
            return
        rows = sum(len(m.candidate.blocks) for m in batch.members)
        count = self.resources.ftl.count
        count("encodes", 1)
        count("encode_rows", rows)
        count("encode_us", int(max(elapsed_ms, 1e-3) * 1e3))

    @marked("gc.publish", lambda self: pcv(self._pcv_publish))
    def _publish_ready_compactions(self):
        """Run every gate before the single store-private adopt operation."""
        resources = self.resources
        for batch in tuple(self.pending_compactions):
            if batch.failed:
                self._abort_batch(batch)
                continue
            if batch.event is None:
                try:
                    end = resources.extent_encoder.submit_batch_output(
                        batch.members, batch.output, self.compaction_copy_stream
                    )
                except Exception:
                    batch.failed = True
                    self._abort_batch(batch)
                    raise
                if end is None:
                    continue
                batch.event = end
                continue
            # Every gate below is PER SUPERBLOCK: the shared D2H has landed
            # each member's bytes in its OWN landing, so a member that fails
            # here aborts alone and leaves its target RAW while its siblings
            # publish.
            # The copy event is batch-wide; the gates below are per member.
            if not batch.event.query():
                continue
            if not batch.timed:
                batch.timed = True
                self._count_encode(batch)
            self._release_batch(batch)
            for pending in tuple(batch.members):
                candidate = pending.candidate
                if not pending.is_phase("COMMITTED"):
                    # All rejection gates precede adoption. Committed members
                    # can return here only to retry publication cleanup.
                    if pending.aborted:
                        self._abort(
                            batch,
                            pending,
                            "LEASE_ABORTED",
                            "compress_lease_abort_discarded",
                        )
                        continue
                    if not self._witness_matches(candidate):
                        self._abort(batch, pending, "MAPPING_CHANGED")
                        continue
                    try:
                        resources.extent_encoder.commit_encoded_extent(pending)
                    except Exception:
                        if not pending.is_phase("COMMITTED"):
                            self._abort(batch, pending, "PUBLISH_ERROR")
                        raise
                superblock = resources.ftl.directory.superblock(candidate.key)
                if superblock is not None and not superblock.sealed:
                    resources.ftl.directory.seal(candidate.key)
                resources.extent_encoder.discard_block_tokens(candidate.blocks)
                self._finish_member(batch, pending)
                resources.ftl.count("compresses")
                resources.ftl.count("compress_xjob_published")
                resources.extent_encoder.log_ratio(candidate.key, candidate.handle)
                if _TRACE:
                    _trace(
                        f"[DITTO-COMPACT] sb={candidate.key} "
                        "outcome=PUBLISHED source=hbm"
                    )
            self.pending_compactions.remove(batch)
        self._drain_deferred()  # terminal members are out of the batches now

    @marked("gc.queue_order", lambda self, keys: dict(keys=n(keys)))
    def _queue_order(self, keys):
        """Complete groups first; unchanged queues reuse their priority order."""
        snapshot = tuple((key, self.to_compact[key]) for key in keys)
        if snapshot != self._queue_snapshot:
            self._queue_snapshot = snapshot
            self._queue_sorted = tuple(
                key for key, size in sorted(snapshot, key=lambda item: -item[1]) if size
            )
        return self._queue_sorted

    @marked(
        "gc.batches",
        lambda self, keys, limit: pcv(lambda: self._pcv_batches(keys, limit)),
    )
    def _batches(self, keys, limit):
        """Group ready candidates by the codec's compatibility key and capacity.

        Stop after `limit` candidates because each needs its own landing.
        The encode and output copy are shared; publication stays per member.
        """
        encoder = self.resources.extent_encoder
        buckets, caps = {}, {}
        ready = 0
        for key in keys:
            if ready >= limit:
                break
            candidate = self._candidate(key)
            if candidate is None:
                continue
            ready += 1
            with region(
                "gc.bucket",
                lambda candidate=candidate: dict(blocks=n(candidate.blocks)),
            ):
                shape, cap = encoder.batch_hint(candidate.group, len(candidate.blocks))
                if shape is None:
                    shape = ("unbatchable", candidate.key)
                # A bucket is one (T, D, compressor) class, but its members may
                # come from groups with different head sizes and therefore
                # different slot layouts. Take the tightest cap in the bucket so
                # the batch still lands in one launch.
                caps[shape] = min(caps.get(shape, cap), cap)
                buckets.setdefault(shape, []).append(candidate)
        batches = []
        # Profiling control: cap batch width without changing workspace capacity.
        ceiling = int(os.environ.get("DITTO_XJOB_BATCH_WIDTH", "0") or 0)
        for shape, members in buckets.items():
            cap = caps[shape] if ceiling <= 0 else min(caps[shape], ceiling)
            for start in range(0, len(members), cap):
                batches.append(tuple(members[start : start + cap]))
        return batches

    @marked("gc.by_reclaimable", lambda self, batches: dict(batches=n(batches)))
    def _by_reclaimable(self, batches):
        """Prioritize total RAW rows reclaimed per codec submission.

        Sum across batch members: several short runs can reclaim more than
        one long run. Equal weights retain queue order.
        """
        return sorted(
            tuple(batches),
            key=lambda batch: -sum(len(member.blocks) for member in batch),
        )

    @marked(
        "gc.launch",
        lambda self, budget, available: dict(
            queued=n(self.to_compact), available=available
        ),
    )
    def _launch_compactions(self, budget, available):
        resources = self.resources
        if budget <= 0 or available <= 0:
            return
        keys = self._queue_order(self.to_compact)
        if not keys:
            return
        launched, superblocks = 0, 0
        for batch in self._by_reclaimable(self._batches(keys, available)):
            if launched >= int(budget) or available <= 0:
                break
            # Every member adopts its OWN extent, so every member needs its
            # own landing: the handle manager's adopt moves a whole chunk
            # chain from the landing handle to one target, and there is no
            # split. A member that cannot get one simply shrinks the batch.
            prepared = []
            try:
                for candidate in batch[:available]:
                    entry = self.prepare_compaction(candidate)
                    if entry is not None:
                        prepared.append(entry)
            except Exception:
                self._refuse_batch(prepared, None, None, requeue=False)
                raise
            if not prepared:
                continue
            available -= len(prepared)
            if self.submit_batch(prepared) is None:
                # More candidates in this poll see the same retained arena;
                # leave them queued for a later trigger after phase B drains.
                break
            launched += 1
            superblocks += len(prepared)
        if superblocks:
            resources.ftl.count("compress_xjob_nominated", superblocks)
            if superblocks > launched:
                # Transactions batching removed this poll — the number this
                # whole change exists to move.
                resources.ftl.count("compress_xjob_batched", superblocks - launched)

    @marked(
        "gc.run_trigger",
        lambda self, allow_launch=True: dict(
            pending=n(self.pending_compactions),
            queued=n(self.to_compact),
            allow_launch=int(allow_launch),
        ),
    )
    def run_gc_trigger(self, *, allow_launch=True):
        """Advance publication, then consume the operation-fed candidate set."""
        if self.resources.codec is None:
            return
        self._publish_ready_compactions()
        if not allow_launch or not any(self.to_compact.values()):
            return
        with region("ftl.reserve_landing"):
            available = self.resources.handle_manager.reserve_landing()
        self._launch_compactions(self.resident_per_poll, available)

    def shutdown(self):
        """Drain in-flight transactions without scanning or launching new work."""
        while self.pending_compactions:
            if self.compaction_stream is not None:
                self.compaction_stream.synchronize()
            if self.compaction_copy_stream is not None:
                self.compaction_copy_stream.synchronize()
            self._publish_ready_compactions()
        self.to_compact.clear()
        self._queue_snapshot = self._queue_sorted = ()
        self.poisoned.clear()
        self.source_bindings.clear()
        self.page_owners.clear()
        self.compaction_stream = None
        self.compaction_copy_stream = None
