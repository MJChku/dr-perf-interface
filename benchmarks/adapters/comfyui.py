"""Build cache keys for a generated workflow using the prepared source."""
import asyncio
import sys
import types

import bench_support


class WorkloadNode:
    NOT_IDEMPOTENT = False

    @staticmethod
    def INPUT_TYPES():
        return {"required": {}}


# These adapters avoid loading models and device-management infrastructure.
# The cache-key and graph implementations themselves are the pinned upstream code.
nodes = types.ModuleType("nodes")
nodes.NODE_CLASS_MAPPINGS = {"Source": WorkloadNode, "Transform": WorkloadNode}
sys.modules["nodes"] = nodes
patcher = types.ModuleType("comfy.model_patcher")
patcher.is_model_patcher_output = lambda value: False
sys.modules["comfy.model_patcher"] = patcher
memory = types.ModuleType("comfy.system_memory")
memory.virtual_memory_available = lambda: 1 << 40
sys.modules["comfy.system_memory"] = memory

from comfy_execution.caching import CacheKeySetInputSignature
from comfy_execution.graph import DynamicPrompt


class Unchanged:
    async def get(self, node_id):
        return False


async def main():
    args = bench_support.states(n=32, shape=0)
    n, shape = args["n"], args["shape"]
    if n < 2 or shape not in (0, 1, 2):
        raise ValueError("n must be >= 2; shape must be 0, 1 or 2")
    prompt = {"0": {"class_type": "Source", "inputs": {"seed": 0}}}
    for i in range(1, n):
        parent = 0 if shape == 1 else i - 1
        inputs = {"x": [str(parent), 0], "strength": i * 0.1}
        if shape == 2 and i > 1:
            inputs["y"] = [str(i - 2), 0]
        prompt[str(i)] = {"class_type": "Transform", "inputs": inputs}
    keys = CacheKeySetInputSignature(DynamicPrompt(prompt), list(prompt), Unchanged())
    await keys.add_keys(list(prompt))
    assert set(keys.keys) == set(prompt)
    print(f"n={n} shape={shape} keys={len(keys.keys)}")


if __name__ == "__main__":
    asyncio.run(main())
