"""Summarize recorded native wait callers; never suppress coverage obligations."""
import collections
import json
from pathlib import Path
import struct
import subprocess
import sys

root = Path(sys.argv[1])
meta = json.loads(next(root.glob('run.*.json')).read_text())
modules = {m['name']: m['path'] for m in meta['modules']}
kinds = {'lock', 'condition', 'completion', 'join', 'receive', 'event_wait',
         'stream_wait', 'device_wait', 'stream_dependency'}
counts = collections.defaultdict(collections.Counter)
for path in root.glob('*.waits'):
    for line in path.open():
        e = json.loads(line)
        if e['kind'] not in kinds or not e['region']:
            continue
        key = (e['region'], e['api'], e.get('callerModule', '?'), e.get('callerOffset', '0'))
        c = counts[key]
        c['calls'] += 1
        c['syscalls'] += e['potentialSyscalls']
        c['failed'] += e['result'] != 0
        c['elapsedUs'] += e['elapsedUs']
# addr2line expects link-time VAs. DynamoRIO records offsets from the lowest
# loaded segment. Accommodate non-PIE ELF executables, whose first VA is nonzero.
def base_address(path):
    with open(path, 'rb') as f:
        h = f.read(64)
        if h[:4] != b'\x7fELF' or h[4:6] != b'\x02\x01':
            raise ValueError('expected little-endian ELF64')
        phoff = struct.unpack_from('<Q', h, 32)[0]
        size, n = struct.unpack_from('<HH', h, 54)
        addrs = []
        for i in range(n):
            f.seek(phoff + i * size)
            p = f.read(size)
            if struct.unpack_from('<I', p)[0] == 1:
                addrs.append(struct.unpack_from('<Q', p, 16)[0])
        return min(addrs)
by_module = collections.defaultdict(set)
for _, _, module, offset in counts:
    by_module[module].add(offset)
symbols = {}
for module, offsets in by_module.items():
    offsets = sorted(offsets, key=int)
    path = modules.get(module)
    if not path or not Path(path).exists():
        continue
    bias = base_address(path)
    result = subprocess.run(['addr2line', '-e', path, '-f', '-C',
        *[hex(int(o) + bias) for o in offsets]], text=True, capture_output=True, check=True)
    lines = result.stdout.splitlines()
    if len(lines) == 2 * len(offsets):
        for i, offset in enumerate(offsets):
            # Stripped binaries can yield only a nearest symbol. This is a
            # navigation hint, never sufficient to exclude a synchronization.
            symbols[module, offset] = dict(symbolHint=lines[2*i], source=lines[2*i+1])
rows = [dict(region=r, api=a, module=m, offset=o, **symbols.get((m,o), {}), **c)
        for (r,a,m,o), c in sorted(counts.items())]
json.dump(dict(capture=str(root), waitRecords=meta['drperf']['wait_records'],
    dropped=meta['drperf']['wait_dropped'],
    symbolization='addr2line hints may be approximate in stripped binaries; module and offset are recorded evidence.',
    rows=rows), sys.stdout, indent=2)
print()
