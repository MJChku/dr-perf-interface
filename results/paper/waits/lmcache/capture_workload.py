#!/usr/bin/env python3
"""In-process vLLM + ditto tier workload for drperf: ONE process holds the
scheduler, the worker and the tier (VLLM_ENABLE_V1_MULTIPROCESSING=0, TP=1),
so DynamoRIO injected at process start covers every region without thread
takeover or child following. Same session/turn prefix-reuse workload as
kimi_client.py, driven through LLMEngine.add_request/step.

    drperf_offline.py [--sessions S --turns T --prefix-words a,b,.. --new-words N
                       --max-tokens M --out rows.json] -- <vLLM engine CLI args>
"""

import argparse
import json
import os
import sys
import threading
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def build_prompts(sessions, prefix_words, offset=0):
    """Distinct prefixes per session; `offset` numbers another family of
    sessions (a warm-up pass) that shares no prefix with the measured ones."""
    vocab = [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "theta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "omicron",
        "pi",
        "rho",
        "sigma",
        "tau",
        "upsilon",
        "phi",
        "chi",
        "psi",
        "omega",
    ]
    sizes = [int(x) for x in str(prefix_words).split(",")]
    out = []
    for s in range(sessions):
        sid = s + offset
        words = [
            vocab[(i * 7 + sid * 13) % len(vocab)] for i in range(sizes[s % len(sizes)])
        ]
        out.append(f"session {sid} " + " ".join(words))
    return out


def main():
    argv = sys.argv[1:]
    engine_argv = []
    if "--" in argv:
        i = argv.index("--")
        argv, engine_argv = argv[:i], argv[i + 1 :]
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=12)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument(
        "--prefix-words", default="100,200,300,400,500,600,750,900,1050,1200,1350,1500"
    )
    ap.add_argument("--new-words", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=4)
    ap.add_argument(
        "--settle-polls",
        type=int,
        default=30,
        help="idle engine steps at the end so queued compactions publish",
    )
    ap.add_argument(
        "--warmup-sessions",
        type=int,
        default=0,
        help="sessions run first with every marker off (src.perf.set_enabled): lazy imports, interpreter specialization and kernel compiles happen before the measured pass and before a late DynamoRIO attach",
    )
    ap.add_argument("--warmup-turns", type=int, default=2)
    ap.add_argument("--warmup-prefix-words", default="650,850,1150")
    ap.add_argument(
        "--settle-sessions",
        type=int,
        default=2,
        help="after the warm-up, one marker triggers the late DynamoRIO attach and these sessions run under it with markers off again: the first executions after the attach cost more (interpreter caches settle) and must not be the measured ones",
    )
    ap.add_argument("--settle-turns", type=int, default=2)
    ap.add_argument("--settle-prefix-words", default="700,950")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    from vllm.engine.arg_utils import EngineArgs
    from vllm.sampling_params import SamplingParams
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.llm_engine import LLMEngine

    try:
        from vllm.utils.argparse_utils import FlexibleArgumentParser
    except ImportError:  # older layout
        from vllm.utils import FlexibleArgumentParser

    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    engine_args = EngineArgs.from_cli_args(parser.parse_args(engine_argv))
    if os.environ.get("DRPERF_WAITS") == "1":
        # Wait matching needs event creation history. Attach after heavyweight
        # imports, but before the engine creates persistent completion events.
        # This internal marker is omitted from application cost interfaces.
        import perfmark

        with perfmark.region("perf.sync_attach"):
            pass
        print("OFFLINE synchronization capture attached before engine construction", flush=True)
    t0 = time.monotonic()
    engine = LLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER
    )
    # Use the actual worker connector, including objects frozen out of gc's
    # tracked generations by vLLM. Validate the teardown handle up front.
    from vllm.distributed.kv_transfer.kv_transfer_state import get_kv_transfer_group
    capture_lookup_server = get_kv_transfer_group()._lmcache_engine.lookup_server
    assert capture_lookup_server is not None, "Worker lookup server unavailable"
    print(
        f"OFFLINE engine ready in {time.monotonic() - t0:.1f}s pid={os.getpid()}",
        flush=True,
    )
    if os.environ.get("DRPERF") and os.environ.get("DRPERF_KEEP_GC") != "1":
        # Python's cyclic collector runs at allocation-count thresholds and
        # lands in whichever region is open; per-region formulas exclude it
        # (reference-count frees stay, they follow the region's own data)
        import gc as _gc

        _gc.collect()
        _gc.disable()
        print("OFFLINE cyclic gc disabled for the measured workload", flush=True)
    params = SamplingParams(max_tokens=a.max_tokens, temperature=0, ignore_eos=True)
    if a.warmup_sessions:
        import src.perf as perf

        perf.set_enabled(False)
        t1 = time.monotonic()
        wprefixes = build_prompts(a.warmup_sessions, a.warmup_prefix_words, offset=1000)
        for t in range(a.warmup_turns):
            for s in range(a.warmup_sessions):
                prompt = wprefixes[s] + " " + " ".join(["omega"] * (a.new_words * t))
                engine.add_request(f"w{s:03d}t{t:02d}", prompt, params)
                while engine.has_unfinished_requests():
                    engine.step()
        for _ in range(a.settle_polls):
            engine.step()
            time.sleep(0.05)
        perf.set_enabled(True)
        print(
            f"OFFLINE warm-up done: {a.warmup_sessions}x{a.warmup_turns} requests in {time.monotonic() - t1:.1f}s (markers off)",
            flush=True,
        )
        if a.settle_sessions:
            t1 = time.monotonic()
            with perf.region(
                "perf.attach"
            ):  # the first marker: late attach happens here
                pass
            print(
                f"OFFLINE attach marker done in {time.monotonic() - t1:.1f}s",
                flush=True,
            )
            perf.set_enabled(False)
            t1 = time.monotonic()
            sprefixes = build_prompts(
                a.settle_sessions, a.settle_prefix_words, offset=2000
            )
            for t in range(a.settle_turns):
                for s in range(a.settle_sessions):
                    prompt = (
                        sprefixes[s] + " " + " ".join(["omega"] * (a.new_words * t))
                    )
                    engine.add_request(f"x{s:03d}t{t:02d}", prompt, params)
                    while engine.has_unfinished_requests():
                        engine.step()
            for _ in range(a.settle_polls):
                engine.step()
                time.sleep(0.05)
            perf.set_enabled(True)
            print(
                f"OFFLINE settle done: {a.settle_sessions}x{a.settle_turns} requests in {time.monotonic() - t1:.1f}s (attached, markers off)",
                flush=True,
            )
    prefixes = build_prompts(a.sessions, a.prefix_words)
    rows = []
    for t in range(a.turns):
        # Later turns start a quarter of the way in: the first loads after a
        # store-only turn run with cold interpreter caches (+20k instructions),
        # so they land on mid-size requests, and the largest prefixes from the
        # end of the previous turn are already out of the GPU cache.
        order = list(range(a.sessions))
        if t:
            cut = min(6, a.sessions // 4)
            order = order[cut:] + order[:cut]
        for s in order:
            prompt = prefixes[s] + " " + " ".join(["omega"] * (a.new_words * t))
            rid = f"s{s:03d}t{t:02d}"  # fixed width like production request ids
            t1 = time.monotonic()
            engine.add_request(rid, prompt, params)
            final = None
            while engine.has_unfinished_requests():
                for out in engine.step():
                    if out.finished:
                        final = out
            dt = time.monotonic() - t1
            row = {
                "session": s,
                "turn": t,
                "prompt_tokens": len(final.prompt_token_ids)
                if final is not None
                else None,
                "cached_tokens": getattr(final, "num_cached_tokens", None),
                "completion_tokens": len(final.outputs[0].token_ids)
                if final is not None
                else None,
                "latency_s": round(dt, 3),
            }
            rows.append(row)
            print("CLIENT " + json.dumps(row), flush=True)
    for _ in range(a.settle_polls):  # idle polls: GC publishes on polls only
        engine.step()
        time.sleep(0.2)
    summary = {
        "requests": len(rows),
        "failed": 0,
        "total_s": round(sum(r["latency_s"] for r in rows), 1),
    }
    print("CLIENT_SUMMARY " + json.dumps(summary), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"rows": rows, "summary": summary, "args": vars(a)}, f, indent=1)
    # Close the actual request server before capture termination. Leaving its
    # daemon receive invocation open would correctly fail the exit indicator.
    capture_lookup_server.close()
    capture_lookup_server.thread.join(timeout=10.0)
    assert not capture_lookup_server.thread.is_alive(), "Lookup receive did not finish"
    print("LOOKUP_SERVERS_CLOSED", 1, flush=True)
    print("OFFLINE_DONE", flush=True)
    # a normal exit lets the drperf client write its counts; a stuck
    # non-daemon thread must not turn that into a SIGKILL from the driver
    exit_timer = threading.Timer(180.0, lambda: os._exit(0))
    # The watchdog must not itself keep an otherwise finished process alive.
    exit_timer.daemon = True
    exit_timer.start()
    shutdown = getattr(engine, "shutdown", None)
    if shutdown is not None:
        shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
