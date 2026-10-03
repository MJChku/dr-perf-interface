# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.naive_serde.serde import Deserializer, Serializer
from lmcache._drperf import _nb, marked, n, pcv, region  # noqa: F401  (drperf overlay)


class NaiveSerializer(Serializer):
    def __init__(self):
        pass

    @marked("lmc.serde.naive.encode")
    def serialize(self, memory_obj: MemoryObj) -> MemoryObj:
        memory_obj.ref_count_up()
        return memory_obj


class NaiveDeserializer(Deserializer):
    @marked("lmc.serde.naive.decode")
    def deserialize(self, memory_obj: MemoryObj) -> MemoryObj:
        return memory_obj
