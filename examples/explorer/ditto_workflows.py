#!/usr/bin/env python3
"""Compose a reviewable Ditto workflow interface from an existing drperf export.

Reads profiles/source only. Does not run Ditto, refit measurements, or edit PCVs.
Source-derived branch rules are checked against every recorded parent call.
"""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib"))
import composition


def formula(region):
    if not region["regimes"]:
        return "no affine fit"
    rendered = []
    for fit in region["regimes"]:
        terms = [f"{round(a)}*{name}" for name, a in
                 zip(region["states"], fit["coefficients"]) if a]
        terms.append(str(round(fit["constant"])))
        rendered.append(" + ".join(terms).replace("+ -", "- "))
    return " / ".join(rendered)


def mean_cost(region):
    return round(sum(p["observed"] * p["calls"] for p in region["points"])
                 / region["calls"])


def render(profile, source_root):
    data = profile.read_bytes()
    model = json.loads(data)
    if model["validity"]["errors"] or model["validity"]["traceErrors"]:
        raise ValueError("Profile has invalid measurements or traces")
    if not model["trace"]["complete"] or model["composition"]["status"] != "observed":
        raise ValueError("Complete observed composition is required")
    if "no wrapper calibration subtraction" not in model["measurement"]["markerAdjustment"]:
        raise ValueError("Re-export this capture without wrapper calibration first")
    regions = {r["id"]: r for r in model["regions"]}
    nodes = composition._forest(regions, model["trace"]["events"])
    calls = defaultdict(list)
    for node in nodes:
        calls[node["region"]].append(node)

    def values(node):
        return dict(zip(regions[node["region"]]["states"], node["state"]))

    def count(node, child):
        return sum(c["region"] == child for c in node["children"])

    def one(node, child):
        found = [c for c in node["children"] if c["region"] == child]
        if len(found) != 1:
            raise ValueError(f"Expected one {child} under {node['region']}")
        return found[0]

    checks = []

    def check(parent, child, expression, expected):
        matched = calls[parent]
        if not matched:
            raise ValueError(f"No coverage of {parent}")
        failures = [n for n in matched if count(n, child) != expected(values(n))]
        if failures:
            raise ValueError(f"{parent} -> {child}: {expression} failed on {len(failures)} calls")
        checks.append((parent, child, expression, len(matched)))

    for child in ("enc.record_tokens", "ftl.bind_store_batch", "gc.invalidate_blocks",
                  "gc.trigger", "store.wait_conflicts", "store.execute"):
        check("rt.submit_store", child, "1", lambda v: 1)
    check("store.execute", "ftl.ensure_row", "placements", lambda v: v["placements"])
    check("store.execute", "carrier.transfer", "1", lambda v: 1)
    for child in ("ftl.plan_load", "gc.trigger", "load.execute", "rt.arm_load"):
        check("rt.submit_load", child, "1", lambda v: 1)
    check("load.execute", "ftl.lock", "reads (observed; unique handles in source)", lambda v: v["reads"])
    check("load.execute", "load.plan", "1", lambda v: 1)
    for child in ("carrier.raw_load", "load.raw_parts"):
        check("load.plan", child, "1", lambda v: 1)
    check("load.plan", "dec.prepare", "encoded", lambda v: v["encoded"])
    for child in ("dec.submit", "rt.acquire_stream", "rt.register_ready_pages"):
        check("load.plan", child, "has_encoded = int(encoded > 0)", lambda v: int(v["encoded"] > 0))
    check("carrier.raw_load", "carrier.h2d_part", "parts", lambda v: v["parts"])
    check("carrier.h2d_part", "carrier.transfer", "1", lambda v: 1)
    check("dec.submit", "codec.decode_units", "1 (GPU path, nonempty units)", lambda v: 1)
    check("rt.poll", "gc.run_trigger", "1", lambda v: 1)
    check("rt.poll", "rt.finish", "pending (capture correlation only)", lambda v: v["pending"])
    for child in ("gc.publish", "ftl.reserve_landing", "gc.launch"):
        check("gc.run_trigger", child, "1 (codec enabled)", lambda v: 1)

    # The flattened store equation below substitutes these child arguments.
    for node in calls["rt.submit_store"]:
        b = values(node)["entries"]
        if any(values(one(node, name))["placements"] != b
               for name in ("store.execute", "store.wait_conflicts")):
            raise ValueError("Store entries and placements differ; cannot use the simplified equation")

    paths = Counter()
    paired = Counter()
    for node in calls["load.plan"]:
        v = values(node)
        kind = "RAW-only" if not v["encoded"] else "encoded-only" if not v["raw"] else "mixed"
        paths[kind] += 1
        if v["blocks"] == 4:
            raw = one(node, "carrier.raw_load")
            paired[(kind, v["raw"], v["encoded"], values(raw)["parts"])] += 1

    source_files = {s["path"]: s["sha256"] for r in regions.values() for s in r["sources"]}
    source_match = {p: (source_root / p).is_file() and
                    sha256((source_root / p).read_bytes()).hexdigest() == digest
                    for p, digest in source_files.items()}
    core_files = [p for p in source_files if p.startswith("src/cache/")]
    if not core_files or not all(source_match[p] for p in core_files):
        raise ValueError("Core cache source differs from profile; review workflow rules against the captured source")

    raw = Path(model["provenance"]["rawDirectory"])
    manifest_path = raw.parent / "capture-source.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest_sources = manifest.get("source_sha256", {})
    manifest_mismatch = [p for p, h in manifest_sources.items()
                         if not (source_root / p).is_file() or sha256((source_root / p).read_bytes()).hexdigest() != h]
    lines = ["# Ditto: a workflow performance interface from the current Qwen capture", "",
        "This is a composed **CPU-work interface**, organized around storing KV, restoring RAW or encoded KV, "
        "and advancing background work. It is a concrete first interface, not a complete latency predictor. "
        "The codebase was read to construct it; practitioners can read the interface below without tracing the implementation.", "",
        "## Capture and scope", "",
        f"- Profile: [{profile.name}]({profile}).",
        f"- Profile SHA-256: `{sha256(data).hexdigest()}`.",
        f"- Raw capture recorded in profile: `{raw}`.",
        f"- {len(regions)} regions; {len(nodes):,} traced application calls; complete trace; no reported measurement errors.",
        f"- {sum(source_match.values())}/{len(source_match)} annotated source-file hashes match the current checkout; "
        f"all {len(core_files)} annotated cache-core files match. Changed adapter files: "
        + ", ".join(f"`{p}`" for p, ok in source_match.items() if not ok) + ".",
        f"- Capture manifest revision: `{manifest.get('git_head', 'unavailable')}`; "
        f"{len(manifest_mismatch)} of {len(manifest_sources)} manifest file hashes differ now: "
        + ", ".join(f"`{p}`" for p in manifest_mismatch) + ". "
        "Recorded numbers describe the captured binaries; source matches do not independently verify native binaries.",
        "- GX functional emulation, with CUDA emulator calls excluded. No GPU execution time, DMA duration, "
        "or end-to-end latency is inferred. Unmarked worker-thread work is not followed in this capture.",
        "- Marker-library blocks and explicit PCV regions are excluded; Python boundary overhead remains. No wrapper calibration is subtracted.",
        "- `framework.fake_output` is emulator-only synthetic metadata construction. It is shown separately, "
        "not charged to the proposed production interface. Other host paths may still depend on synthetic codec output sizes.", "",
        "## What the practitioner sees", "",
        "```text\nNew KV from model execution\n  -> STORE: place RAW rows, copy GPU -> host, register sources\n  -> POLL: finish transfers; optionally encode and publish complete extents\n\nLater cache hit\n  -> RELOAD / submit_load\n       RAW data     -> group copy runs -> copy host -> GPU\n       encoded data -> bind decode units -> upload/decode into destination pages\n       mixed data   -> both branches\n  -> POLL: collect completion, release locks, advance compaction\n```", "",
        "There is no separate public `reload` method: restoration uses `submit_load`. "
        "Initial model loading and cache-miss prefill are outside this cache interface. "
        "A store submits RAW storage first; compression is deferred. Charging every encode to the immediate store call would miss that lifecycle.", "",
        "| Semantic quantity | Meaning |",
        "| --- | --- |",
        "| B | Logical blocks / placements in this request |",
        "| P | Physical pages transferred |",
        "| R_raw, E | RAW read records and encoded read records/extents in the load plan |",
        "| H | Distinct storage handles to lock |",
        "| K | Contiguous RAW copy parts after grouping |",
        "| D | RAW destinations processed while grouping |",
        "| I_enc | `int(E > 0)`: whether decode setup is needed |",
        "| L, U | Participating layer buffers; codec units bound across encoded extents |",
        "| N_done, N_query | Completed jobs and completion queries during this poll |",
        "| Pool state | Stream/descriptor/workspace reuse, growth, and first launch |", "",
        "`O[region]` denotes its own CPU work, including its unexplained part; `F[region]` includes its child interfaces. "
        "A multiplier means that many applications at the actual child states, not that many copies of a global average. "
        "All displayed fitted coefficients are rounded to integers. Exact observed states, rather than a continuous box of inputs, define the checked domain.", "",
        "## Store", "",
        "For the successful codec-enabled store path exercised by this capture:", "",
        "```text\nF_store = O[rt.submit_store]\n        + F[enc.record_tokens] + F[gc.invalidate_blocks]\n        + F[ftl.bind_store_batch] + F[gc.trigger]\n        + F[store.wait_conflicts] + F[store.execute]\n\nF[store.execute] = O[store.execute]\n                 + H * F[ftl.write_lock]\n                 + B * F[ftl.ensure_row]\n                 + F[carrier.transfer]\n```", "",
        "`H` is the number of distinct placement handles, not generally B or a fixed fraction of B. "
        "The existing parent PCVs do not name H, which explains its unresolved write-lock multiplier. "
        "This equation also needs a separate refusal/cleanup branch for capacity failures; those were not exercised here. "
        "The conflict-wait path had no nested drain calls in these stores, so concurrent conflicting stores remain outside its measured coverage.", ""]
    names = ("rt.submit_store", "store.wait_conflicts", "store.execute")
    if all(len(regions[n]["regimes"]) == 1 for n in names):
        a = sum(regions[n]["regimes"][0]["coefficients"][0] for n in names)
        c = sum(regions[n]["regimes"][0]["constant"] for n in names)
        lines += ["The three own-work fits above combine, on this capture's observed states, to:", "",
                  f"```text\n{round(a)}*B + {round(c)} + unexplained_own\n```", "",
                  "This is only the store wrapper, conflict check, and execution bookkeeping. "
                  "Binding, token recording, locking, row materialization, transfer setup, and deferred compression remain explicit child costs. "
                  "It is not the complete store cost.", ""]
    lines += ["## RAW, encoded, and mixed reload", "",
        "```text\nF_reload = O[rt.submit_load]\n         + F[ftl.plan_load] + F[gc.trigger] + F[rt.arm_load]\n         + N_stats * F[ftl.count] + F[load.execute]\n\nF[load.execute] = O[load.execute] + H * F[ftl.lock] + F[load.plan]\n\nF[load.plan] = O[load.plan]\n            + F[load.raw_parts] + F[carrier.raw_load]\n            + E * F[dec.prepare]\n            + I_enc * (F[rt.acquire_stream] + F[dec.submit]\n                       + F[rt.register_ready_pages])\n\nF[carrier.raw_load] = O[carrier.raw_load] + K * F[carrier.h2d_part]\nF[carrier.h2d_part] = O[carrier.h2d_part] + F[carrier.transfer]\nF[dec.submit]      = O[dec.submit] + F[codec.decode_units]\n```", "",
        "The decode equation is for the exercised GPU decode path with nonempty units and source-registration callbacks enabled. "
        "Empty inputs, CPU decode, failures, and disabled callbacks need their own paths. "
        "`N_stats` counts runtime counter updates; it is not a compression parameter. "
        "In this capture the lock count equals the read count, but the source deduplicates handles, so H is the stronger semantic input.", "",
        f"Observed reloads: **{paths['RAW-only']} RAW-only, {paths['encoded-only']} encoded-only, {paths['mixed']} mixed**.", "",
        "The same four logical blocks exercised both of these paths:", "",
        "| Path for B = 4 | RAW reads | Encoded reads | RAW copy parts | Calls observed |",
        "| --- | ---: | ---: | ---: | ---: |"]
    lines += [f"| {kind} | {r} | {e} | {k} | {n} |" for (kind, r, e, k), n in sorted(paired.items())]
    lines += ["", "The encoded branch introduces decode preparation and submission even at the same request size. "
        "The RAW branch instead transfers the restored pages directly. This is a real reason to expose representation and extent counts, "
        "rather than promise a single formula in B.", "",
        "### Branches checked from source and trace", "",
        "These are source-derived workflow rules, checked against every recorded parent invocation. "
        "They are not new fitted drperf coefficients and do not modify the profile. "
        "In particular, `I_enc` resolves a Boolean branch that cannot be affine in E alone when E ranges over 0, 1, 2, 3. "
        "The RAW part-count rule is already affine, but the generic export declined it with too few distinct parent states.", "",
        "| Parent | Child | Calls per parent | Invocations checked |",
        "| --- | --- | --- | ---: |"]
    lines += [f"| `{p}` | `{c}` | `{expr}` | {n} |" for p, c, expr, n in checks
              if p in ("load.plan", "carrier.raw_load")]
    lines += ["", f"All {sum(n for _, _, _, n in checks):,} call-count checks across {len(checks)} rules passed, "
        "including the store and maintenance skeletons. Full trace nesting and per-state call counts were validated before composition.", "",
        "## Completion and deferred compression", "",
        "```text\nF_poll = O[rt.poll]\n       + N_query * F[rt.query] + N_done * F[rt.finish]\n       + N_depart * F[rt.depart_restore] + N_stats * F[ftl.count]\n       + I_settle * F[rt.queue_settle] + F_background\n\nF_background = O[gc.run_trigger]\n             + F[gc.publish] + F[ftl.reserve_landing] + F[gc.launch]\n\nencode launch -> prepare candidates -> encode submission\npublication  -> output copy -> witness check -> commit -> release\n```", "",
        "The poll coefficients above are actual event counts, not all current entry PCVs; the existing interface does not yet predict all of them. "
        "In this functional capture every pending job was completed during its poll, so the exporter learned `pending * F[rt.finish]`. "
        "Real asynchronous execution can leave jobs pending. Expose `N_done`, or a completion-state model, rather than generalize that correlation. "
        "Similarly, call frequency is controlled by the client; there is no fixed number of polls per request.", "",
        f"This run contains {len(calls['rt.poll'])} polls, {len(calls['rt.finish'])} job completions, "
        f"{len(calls['enc.submit_batch'])} encode-batch submissions, and {len(calls['enc.commit'])} commits. "
        "The run mixes initialization, pool reuse, and later restoration states.", "",
        "The measured `gc.run_trigger` interface also includes `framework.fake_output`. "
        "That node is omitted only from the proposed production equation above; its recorded cost remains visible in the evidence below. "
        "Removing that node does not turn a functional-emulation trace into a timing-accurate production capture.", "",
        "## What the measured formulas support", "",
        "Each row is **own work only**, excluding child regions. Add its reported unexplained work and compose the children above. "
        "For unfitted regions, the observed mean is evidence at the sampled states, not a constant interface or a prediction. "
        "Unexplained percentages below are the export's per-state fit summary, rounded to whole percentages.", "",
        "| Region | Own explained formula / fit status | Own observed mean | Unexplained share | Calls / states |",
        "| --- | --- | ---: | --- | ---: |"]
    selected = ["rt.submit_store", "ftl.bind_store_batch", "store.wait_conflicts", "store.execute",
                "ftl.ensure_row", "ftl.write_lock", "carrier.transfer", "carrier.swap_launch",
                "rt.submit_load", "ftl.plan_load", "load.execute", "load.plan", "ftl.lock",
                "load.raw_parts", "carrier.raw_load", "carrier.h2d_part", "dec.prepare",
                "rt.acquire_stream", "dec.submit", "codec.decode_units", "rt.poll", "gc.run_trigger",
                "gc.launch", "gc.publish", "gc.batches", "enc.submit_batch", "codec.encode_units",
                "framework.fake_output"]
    for name in selected:
        r = regions[name]
        share = "/".join(f"{round(100*f['unexplainedShare'])}%" for f in r["regimes"]) or "not fitted"
        lines.append(f"| `{name}` | `{formula(r)}` | {mean_cost(r):,} | {share} | {r['calls']} / {len(r['points'])} |")
    lines += ["", "Zero displayed percent can mean a small nonzero residual. Likewise a displayed `0*PCV` "
        "can be a tiny nonzero fitted coefficient rounded for readability. Neither display rounding nor a missing term proves zero physical cost.", "",
        "## Transfer work is a separate interface", "",
        "The transport source provides a semantic byte/descriptor model independently of CPU fits:", "",
        "```text\nRAW copy operations = sum over groups (pages_in_group * layer_buffers_in_group)\nRAW bytes           = sum over groups (pages_in_group * bytes_per_page_across_buffers)\nencoded input bytes = sum of prepared_decodes.wire_bytes\n```", "",
        "These are source-derived work counts, not measured bandwidth formulas. "
        "The RAW path computes `transfer_bytes` while constructing descriptors; the decode bridge totals stored stream bytes. "
        "Encoded input bytes are distinct from all end-to-end traffic: metadata, output placement, and other codec transfers can add traffic. "
        "This GX capture synthesizes codec output sizes, so it cannot establish a real compression ratio or GPU decode cost.", "",
        "## What a practitioner can and cannot decide yet", "",
        "- **Useful now:** distinguish RAW/encoded/mixed reloads; see that store and compression are separate lifecycle stages; "
        "identify per-request setup, per-block work, distinct-handle locking, and per-extent decode preparation; "
        "understand which costs move when the workload changes these quantities.",
        "- **Not identified here:** effects of changing layer count or page geometry. The capture fixes 24 layer references and 8 pages per stored block; "
        "pages, rows, and codec units are correlated. Those fitted coefficients cannot independently price a different architecture or geometry.",
        "- **Incomplete numerical interfaces:** decode preparation and encode submission lack sufficient state variation; "
        "load planning and compaction still have appreciable unexplained own work. Stream/workspace state also changes CPU cost. "
        "Neither an unresolved child multiplier nor a single-state mean should be presented as a complete formula.",
        "- **Next measurements:** vary encoded extent count and partial load sizes independently; exercise multiple RAW parts, "
        "store handle layouts, layer counts, page sizes, and pool cold/reuse states; separate idle polls, incomplete jobs, "
        "and completed jobs under realistic event readiness; cover contention and refusal branches.",
        "- **Next semantic PCVs:** `has_encoded`, distinct read/write handles, completed/query/departure counts for polling, "
        "and descriptor/workspace growth. Some are internal state summaries rather than knobs the practitioner sets; "
        "the interface should explain how configuration and workload produce them.", "",
        "This already gives a useful structural performance interface. It does **not yet** justify a complete numeric model of "
        "compression-method trade-offs or a latency prediction. No scalar total across concurrent CPU/GPU work is constructed here.", "",
        "## Source and observed-domain audit", "",
        "The table below retains the region arguments and observed ranges so readers do not mistake fixed quantities for independently varied inputs. "
        "Ranges summarize discrete states; combinations inside a range may never have occurred.", "",
        "| Region | Observed PCV ranges | Source |",
        "| --- | --- | --- |"]
    for name in selected:
        r = regions[name]
        bounds = []
        for i, state in enumerate(r["states"]):
            vs = [int(p["state"][i]) for p in r["points"]]
            lo, hi = min(vs), max(vs)
            bounds.append(f"{state}={lo}" if lo == hi else f"{state}={lo}..{hi}")
        s = r["sources"][0]
        lines.append(f"| `{name}` | `{', '.join(bounds)}` | [{s['path']}:{s['line']}]({source_root / s['path']}:{s['line']}) |")
    lines += ["", "Reproduce this document (reads the existing export; no GPU or new capture needed):", "",
              f"```sh\npython3 {Path(__file__).resolve()} \\\n  --profile {profile} \\\n  --source-root {source_root} \\\n  --output {Path(__file__).with_name('DITTO_WORKFLOWS.md')}\n```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("DITTO_WORKFLOWS.md"))
    args = parser.parse_args()
    report = render(args.profile.resolve(), args.source_root.resolve())
    args.output.write_text(report)
    print(f"Wrote {args.output}: workflow rules and trace/counter coverage checked")


if __name__ == "__main__":
    main()
