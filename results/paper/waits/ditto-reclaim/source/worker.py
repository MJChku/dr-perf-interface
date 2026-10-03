"""vLLM buffer and request translation into the shared cache runtime."""

from dataclasses import dataclass, replace

import numpy as np
import torch
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingWorker,
)

from src.cache.builder import CacheResources
from src.cache.configuration import CacheConfiguration
from src.cache.types import (
    CacheCompletion,
    LoadRequest,
    PageBatch,
    StoreRequest,
)
from src.integration.kv_layout import logical_layer_caches, owned_store_groups
from src.perf import marked, n

from .common import layer_buffer_refs
from .spec import DittoOffloadingSpec


@dataclass
class _PendingStore:
    request_id: str | None = None
    superblock_keys: list | None = None
    tokens: list | None = None


class DittoOffloadingWorker(OffloadingWorker):
    def __init__(self, kv_caches: CanonicalKVCaches, spec: DittoOffloadingSpec):
        from .carrier import NativeCarrier

        config = spec.cache_config
        self.replicated = None
        self._completed = []
        if getattr(spec, "replicated_groups", ()):
            from .replicated import ReplicatedMLA

            self.replicated = ReplicatedMLA(spec.replicated_groups, kv_caches)
            if not all(self.replicated.owned):
                # Followers retain their rank-specific state but allocate no
                # codec/landing workspace for the owner's MLA groups.
                config = CacheConfiguration(
                    options=spec.extra_config,
                    kv_dtype="fp8"
                    if config.kv_fp8
                    else "fp16"
                    if config.kv_fp16
                    else "bf16",
                    groups=config.groups,
                    buffers=tuple(
                        replace(
                            buffer,
                            layers=tuple(
                                layer
                                for layer in buffer.layers
                                if layer.group < 0 or self.replicated.owned[layer.group]
                            ),
                        )
                        for buffer in config.buffers
                    ),
                    total_bytes=config.ftl_total_bytes,
                    pages_per_block=config.pages_per_block,
                    rope_table=config.rope_table,
                )
        logical = logical_layer_caches(
            kv_caches, [m["gpu_page_bytes"] for m in config.layer_buffer_meta]
        )
        buffers = [
            cache.tensor
            if cache.tensor.ndim == 2
            else cache.tensor.view(torch.int8).view(-1, cache.page_size_bytes)
            for cache in logical.tensors
        ]
        self.resources = CacheResources(
            config,
            layer_kv_buffers=buffers,
            layer_buffer_refs=layer_buffer_refs(logical.group_data_refs),
            transport=NativeCarrier(
                kv_caches,
                config.pages_per_block,
                layer_buffer_refs(kv_caches.group_data_refs),
            ),
            storage=spec.create_shared_storage(),
        )
        self.cache = self.resources.cache
        self.compression_allowed = False
        self.compression_enabled = spec.extra_config.get(
            "ditto_compression_enabled", True
        )
        self._shared_stores = {}
        self._shared_loads = {}
        self.pages_per_block = config.pages_per_block
        self.T_by_group = config.group_obs
        self.gc_by_group = self.resources.gc_by_group
        self.layer_kv_buffers = self.resources.layer_kv_buffers
        self._pending_stores: dict[int, _PendingStore] = {}

    @marked(
        "worker.submit_store",
        lambda self, job_id, src_spec, dst_spec: dict(
            pages=n(src_spec.block_ids), blocks=n(dst_spec.block_ids)
        ),
    )
    def submit_store(self, job_id, src_spec, dst_spec) -> bool:
        job = self._pending_stores[job_id]
        if job.request_id is None or job.tokens is None or job.superblock_keys is None:
            raise RuntimeError(f"store job {job_id} is missing scheduler metadata")
        request = StoreRequest.from_pages(
            self._page_batch(src_spec),
            dst_spec.block_ids,
            owners=job.superblock_keys,
            tokens=job.tokens,
            pages_per_block=self.pages_per_block,
            tokens_by_group=self.T_by_group,
            gc_by_group=self.gc_by_group,
            request_id=job.request_id,
        )
        if hasattr(dst_spec, "keys"):
            self.cache.ftl.register(dst_spec.block_ids, dst_spec.keys)
            self._shared_stores[job_id] = tuple(dst_spec.block_ids)
        if self.replicated is not None:
            request = owned_store_groups(request, self.replicated.owned)
        if request.entries:
            accepted = self.cache.submit_store(job_id, request)
        else:
            self._completed.append(CacheCompletion(job_id, True))
            accepted = True
        self._pending_stores.pop(job_id, None)
        return accepted

    @staticmethod
    def _page_batch(spec):
        return PageBatch(
            np.asarray(spec.block_ids, dtype=np.int64),
            tuple(int(size) for size in spec.group_sizes),
            tuple(int(start) for start in spec.block_indices),
        )

    @marked(
        "worker.submit_load",
        lambda self, job_id, src_spec, dst_spec: dict(
            pages=n(dst_spec.block_ids), blocks=n(src_spec.block_ids)
        ),
    )
    def submit_load(self, job_id, src_spec, dst_spec) -> bool:
        if hasattr(src_spec, "keys"):
            self.cache.ftl.register(src_spec.block_ids, src_spec.keys)
            self._shared_loads[job_id] = src_spec.keys
        request = self._destinations_from_spec(dst_spec, src_spec)
        if self.replicated is None or not any(
            self.replicated.replicated[group] for group in request.by_group
        ):
            return self.cache.submit_load(job_id, request)
        local = self.replicated.local_load(request)
        result, error = CacheCompletion(job_id, True), None
        try:
            if local.by_group:
                self.cache.submit_load(job_id, local)
                self.cache.wait((job_id,))
                finished = self.cache.get_finished(
                    allow_compression=self.compression_allowed
                )
                self._completed.extend(r for r in finished if r.job_id != job_id)
                result = next(r for r in finished if r.job_id == job_id)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        success = self.replicated.agree(result.success, error)
        if success:
            self.replicated.restore(request)
        self._completed.append(result if success else CacheCompletion(job_id, False))
        return True

    @marked("worker.poll", lambda self: dict(pending=n(self.cache.pending_jobs)))
    def get_finished(self):
        from vllm.v1.kv_offload.base import TransferResult

        finished = self.cache.get_finished(
            allow_compression=self.compression_allowed and self.compression_enabled
        )
        for result in finished:
            blocks = self._shared_stores.pop(result.job_id, None)
            if blocks is not None and result.success:
                self.cache.ftl.publish(blocks)
                self.cache.handle_manager.trace(
                    "store",
                    job_id=result.job_id,
                    keys=[self.cache.ftl.keys[int(block)] for block in blocks],
                    bytes=result.transfer_size,
                )
            keys = self._shared_loads.pop(result.job_id, None)
            if keys is not None and result.success:
                self.cache.handle_manager.trace(
                    "load",
                    job_id=result.job_id,
                    keys=keys,
                    bytes=result.transfer_size,
                    verification=self.resources.codec.verify_stats,
                )
        if self._completed:
            finished = self._completed + finished
            self._completed = []
        results = [
            TransferResult(
                job_id=result.job_id,
                success=result.success,
                transfer_size=result.transfer_size,
                transfer_time=result.transfer_time,
            )
            for result in finished
        ]
        # Only vLLM encodes source-reader completion as a negative job id.
        results.extend(
            TransferResult(
                job_id=lease, success=True, transfer_size=None, transfer_time=None
            )
            for _key, lease in self.cache.take_lease_releases()
            if lease is not None
        )
        return results

    @marked("worker.wait", lambda self, job_ids: dict(jobs=n(job_ids)))
    def wait(self, job_ids) -> None:
        jobs = []
        for job_id in job_ids:
            if job_id < 0:
                self.cache.flush_source_lease(job_id)
            else:
                jobs.append(job_id)
        self.cache.wait(jobs)

    def shutdown(self) -> None:
        storage = (
            self.cache.handle_manager if hasattr(self.cache.ftl, "local") else None
        )
        if storage is not None:
            storage.trace(
                "runtime",
                stats=dict(self.cache.ftl.stats),
                verification=self.resources.codec.verify_stats,
                token_blocks=len(self.cache.extent_encoder._tokens_by_block),
            )
        self.cache.shutdown()
        if self._shared_stores:
            self._shared_stores.clear()
        if storage is not None:
            storage.close()
        self._pending_stores.clear()
        self._completed.clear()
        self.layer_kv_buffers.clear()

    @marked("worker.add_lease_ids", lambda self, values: dict(leases=n(values)))
    def add_lease_ids(self, values):
        self.cache.add_lease_ids(values)
        if self.replicated is not None and not any(self.gc_by_group):
            for lease in values.values():
                self.cache.flush_source_lease(lease)

    @marked("worker.reader_flush")
    def on_reader_flush(self, reader_id):
        self.cache.flush_source_lease(reader_id)

    @marked(
        "worker.invalidate_swa",
        lambda self, invalidations: dict(blocks=n(invalidations)),
    )
    def invalidate_swa_blocks(self, invalidations):
        self.cache.invalidate_checked(invalidations)

    @marked("worker.admission_limit")
    def ftl_admission_limit(self):
        return self.cache.admission_limit()

    @marked("worker.add_pending_reqids", lambda self, values: dict(jobs=n(values)))
    def add_pending_reqids(self, values):
        for job_id, request_id in values.items():
            self._pending_stores.setdefault(
                job_id, _PendingStore()
            ).request_id = request_id

    @marked(
        "worker.add_pending_keys",
        lambda self, values: dict(
            jobs=n(values), keys=sum(n(v) for v in values.values())
        ),
    )
    def add_pending_superblock_keys(self, values):
        for job_id, keys in values.items():
            self._pending_stores.setdefault(
                job_id, _PendingStore()
            ).superblock_keys = keys

    @marked(
        "worker.add_pending_tokens",
        lambda self, values: dict(
            jobs=n(values),
            arrays=sum(n(v) for v in values.values()),
            new=sum(1 for j in values if j not in self._pending_stores),
        ),
    )
    def add_pending_tokens(self, values):
        pending = self._pending_stores
        for job_id, tokens in values.items():
            store = pending.get(job_id)
            if store is None:
                store = pending[job_id] = _PendingStore()
            store.tokens = tokens

    @marked(
        "worker.destinations",
        lambda self, gpu_spec, block_spec: dict(
            pages=n(gpu_spec.block_ids), blocks=n(block_spec.block_ids)
        ),
    )
    def _destinations_from_spec(self, gpu_spec, block_spec):
        return LoadRequest.from_pages(
            self._page_batch(gpu_spec), block_spec.block_ids, self.pages_per_block
        )
