"""Make a bounded-memory Chrome activity overview from a detailed GX trace.

Counters show the fraction of each time bin covered by the union of modeled
intervals in a category, across a rank's streams. Categories can overlap;
they are separate counters, not additive device utilization measurements.
The input and output paths must differ. Full dependency records remain in
the original capture archives.
"""
import argparse
from array import array
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path


def events(path):
    decoder = json.JSONDecoder()
    with gzip.open(path, 'rt') as stream:
        buffer = stream.read(1024 * 1024)
        prefix = '{"traceEvents":['
        if not buffer.startswith(prefix):
            raise ValueError('Expected GX traceEvents array first')
        position = len(prefix)
        while True:
            while position < len(buffer) and buffer[position] in ' \r\n\t,':
                position += 1
            if position < len(buffer) and buffer[position] == ']':
                return
            try:
                event, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                more = stream.read(1024 * 1024)
                if not more:
                    raise ValueError('Truncated trace')
                buffer = buffer[position:] + more
                position = 0
                continue
            position = end
            yield event


def union_intervals(pairs):
    ordered = sorted(zip(pairs[::2], pairs[1::2]))
    if not ordered:
        return
    start, end = ordered[0]
    for left, right in ordered[1:]:
        if left <= end:
            end = max(end, right)
        else:
            yield start, end
            start, end = left, right
    yield start, end


def bin_intervals(intervals, bin_us, end_us):
    bins = [0.0] * max(1, math.ceil(end_us / bin_us))
    total = 0.0
    for start, end in intervals:
        total += end - start
        while start < end:
            index = min(int(start // bin_us), len(bins) - 1)
            right = min(end, (index + 1) * bin_us)
            assert right > start
            bins[index] += right - start
            start = right
    assert math.isclose(sum(bins), total, rel_tol=1e-9, abs_tol=1e-5)
    for index, active in enumerate(bins):
        width = min(bin_us, end_us - index * bin_us)
        assert active <= width + 1e-5, (index, active, width)
    return bins, total


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def compact(source, target, bin_ms=10):
    assert source.resolve() != target.resolve()
    assert bin_ms > 0
    phases, categories = Counter(), Counter()
    lanes, processes = {}, {}
    spans = defaultdict(lambda: array('d'))
    end_us = 0.0
    for event in events(source):
        phase = event['ph']
        phases[phase] += 1
        pid, tid = event.get('pid', 0), event.get('tid', 0)
        if phase == 'M':
            if event['name'] == 'thread_name':
                lanes[pid, tid] = event['args']['name']
            elif event['name'] == 'process_name':
                processes[pid] = event['args']['name']
            continue
        if phase != 'X':
            continue
        category = event.get('cat', '')
        categories[category] += 1
        start = float(event['ts'])
        duration = float(event.get('dur', 0))
        assert start >= 0 and duration >= 0
        end_us = max(end_us, start + duration)
        if duration == 0:
            continue
        if category == 'cpu':
            kind = 'CPU execution'
        elif category == 'cpu_wait':
            kind = 'CPU waiting for GPU'
        elif 'copy' in event.get('args', {}):
            direction = event['args']['copy']['kind']
            kind = {0: 'H2H copy', 1: 'H2D copy', 2: 'D2H copy', 3: 'D2D copy'}[direction]
        elif lanes.get((pid, tid), '').startswith('GPU '):
            kind = 'GPU kernels' if category == 'COMPUTE' else 'GPU ' + category
        else:
            kind = 'Dependency ' + category
        spans[pid, kind].extend((start, start + duration))
    output = []
    for pid in sorted({p for p, _ in spans}):
        output.append({'ph': 'M', 'name': 'process_name', 'pid': pid,
                       'args': {'name': processes.get(pid, f'rank {pid - 1}')}})
    bin_us = bin_ms * 1000.0
    integrals = {}
    for (pid, kind), pairs in sorted(spans.items()):
        bins, total = bin_intervals(union_intervals(pairs), bin_us, end_us)
        integrals[f'{pid}:{kind}'] = total
        previous = None
        for i, busy in enumerate(bins):
            width = min(bin_us, end_us - i * bin_us)
            value = round(min(100.0, 100.0 * busy / width), 2)
            if value != previous:
                output.append({'ph': 'C', 'pid': pid, 'tid': 0,
                               'name': kind + ' active (%)',
                               'ts': i * bin_us, 'args': {'percent': value}})
                previous = value
        output.append({'ph': 'C', 'pid': pid, 'tid': 0,
                       'name': kind + ' active (%)', 'ts': end_us,
                       'args': {'percent': 0}})
    provenance = {'view': 'modeled activity overview', 'bin_ms': bin_ms,
                  'percent_rounding': 2, 'source_sha256': digest(source),
                  'source_compressed_bytes': source.stat().st_size,
                  'source_events_by_phase': dict(phases),
                  'source_spans_by_category': dict(categories),
                  'source_max_end_us': end_us, 'overview_events': len(output),
                  'union_active_us_by_rank_category': integrals,
                  'interpretation': 'Each counter is interval-union occupancy per time bin, '
                  'across a rank\'s streams within one category. Categories overlap and '
                  'must not be summed as GPU utilization. No individual kernels or arrows. '
                  'Counters are rounded to 0.01 percentage points; sub-bin gaps are averaged. '
                  'Full dependency records remain in the capture archives.'}
    payload = json.dumps({'traceEvents': output, 'displayTimeUnit': 'ms',
                          'otherData': provenance}, separators=(',', ':')).encode()
    with gzip.open(target, 'wb', compresslevel=6) as stream:
        stream.write(payload)
    with gzip.open(target, 'rb') as stream:
        assert stream.read() == payload
    provenance.update(output_compressed_bytes=target.stat().st_size,
                      output_json_bytes=len(payload), output_sha256=digest(target))
    return provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path)
    parser.add_argument('--bin-ms', type=float, default=10)
    args = parser.parse_args()
    print(json.dumps(compact(args.source, args.target, args.bin_ms), indent=2))
