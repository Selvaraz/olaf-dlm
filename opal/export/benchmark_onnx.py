#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark OPAL (ONNX) on CPU: FP32 vs FP16 vs INT8

WHAT THIS DOES
--------------
- Loads up to three ONNX models: FP32, FP16, INT8 (any subset is fine)
- Times inference on CPU (ONNX Runtime)
- Reports:
  * Latency per run (ms)
  * Throughput (tokens/sec)
  * Peak RSS (MB)
  * Average CPU% during timed runs
- Lets you pick:
  * batch size, sequence length, repetitions, warmup runs
  * number of intra/inter op threads (to respect your CPU cap)
  * “decode emulation” loop (optional) to approximate next-token generation cost if you don’t have KV cache I/O

ASSUMPTIONS
-----------
- Your model takes `input_token_ids: int64[batch, seq_len]`
- Outputs `logits: float[batch, seq_len, vocab_size]`
- You exported with dynamic axes (batch, seq_len) using the earlier export script.
- No agent memory is counted (we report the process’s RSS during the model runs only).

USAGE EXAMPLES
--------------
# Compare FP32 vs INT8 at seq_len=1024, batch=1 with controlled threads
python scripts/bench_onnx_cpu.py \
  --model-fp32 out/onnx/model.onnx \
  --model-int8 out/onnx/model.int8.onnx \
  --vocab-size 8000 \
  --seq-len 1024 --batch 1 \
  --warmup 3 --repeat 20 \
  --intra 4 --inter 1

# Include FP16 (if you exported it) and run decode emulation for 256 steps
python scripts/bench_onnx_cpu.py \
  --model-fp16 out/onnx_fp16/model.fp16.onnx \
  --model-int8 out/onnx/model.int8.onnx \
  --vocab-size 8000 \
  --seq-len 512 --batch 1 \
  --decode-steps 256 \
  --warmup 2 --repeat 10 \
  --intra 4 --inter 1
"""

import os
import time
import argparse
from pathlib import Path
import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

try:
    import onnxruntime as ort
except ImportError as e:
    raise SystemExit("onnxruntime is required: pip install onnxruntime") from e


def human_mb(bytes_val: int) -> float:
    return bytes_val / (1024 ** 2)


def cpu_mem_snapshot(proc) -> tuple[float, float]:
    """Return (cpu_percent, rss_mb). cpu_percent is instantaneous (process-wide)."""
    if psutil is None:
        return (float("nan"), float("nan"))
    cpu = proc.cpu_percent(interval=None)  # non-blocking; relies on prior call
    rss = human_mb(proc.memory_info().rss)
    return cpu, rss


def run_full_sequence(sess: ort.InferenceSession, vocab_size: int, batch: int, seq_len: int, reps: int, warmup: int):
    """
    Fast path: run BxT tokens in one call.
    Reports average latency (ms) and tokens/sec (B*T / time).
    """
    rng = np.random.default_rng(1234)
    input_ids = rng.integers(0, vocab_size, size=(batch, seq_len), dtype=np.int64)
    inputs = {"input_token_ids": input_ids}

    # Warmup (not timed)
    for _ in range(warmup):
        sess.run(["logits"], inputs)

    # Timed runs
    t0 = time.perf_counter()
    for _ in range(reps):
        sess.run(["logits"], inputs)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    avg_ms = (elapsed / reps) * 1000.0
    toks = batch * seq_len
    tps = (toks * reps) / elapsed
    return avg_ms, tps


def run_decode_emulation(sess: ort.InferenceSession, vocab_size: int, batch: int, seq_len: int, decode_steps: int, reps: int, warmup: int):
    """
    Slow path: emulate autoregressive decoding cost by repeatedly calling the model
    with a growing input sequence. NOTE: without past_key_values in the graph,
    this is pessimistic (re-computes the whole prefix every step). It’s a useful
    upper bound on latency per generated token before we add KV-cache I/O.

    We time 'decode_steps' incremental calls for one repetition; repeat 'reps' times.
    """
    rng = np.random.default_rng(1234)

    # Warmup on a shorter seq to prime kernels
    for _ in range(warmup):
        warm_ids = rng.integers(0, vocab_size, size=(batch, min(32, seq_len)), dtype=np.int64)
        sess.run(["logits"], {"input_token_ids": warm_ids})

    # Build a fixed prefix to extend
    base = rng.integers(0, vocab_size, size=(batch, seq_len), dtype=np.int64)

    total_elapsed = 0.0
    for _ in range(reps):
        cur = base.copy()
        t0 = time.perf_counter()
        for _step in range(decode_steps):
            # Append a random next token
            next_tok = rng.integers(0, vocab_size, size=(batch, 1), dtype=np.int64)
            cur = np.concatenate([cur, next_tok], axis=1)
            sess.run(["logits"], {"input_token_ids": cur})
        t1 = time.perf_counter()
        total_elapsed += (t1 - t0)

    # Average per-token ms and tokens/sec
    tokens_generated = decode_steps * reps * batch
    avg_ms_per_token = (total_elapsed / (decode_steps * reps)) * 1000.0
    tps = tokens_generated / total_elapsed
    return avg_ms_per_token, tps


def make_session(onnx_path: Path, intra: int, inter: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # You can tweak memory arenas if you see fragmentation on device:
    # so.enable_mem_pattern = True
    # so.enable_cpu_mem_arena = True

    providers = [("CPUExecutionProvider", {
        "intra_op_num_threads": intra,
        "inter_op_num_threads": inter
    })]
    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)
    return sess


def bench_one(label: str, onnx_path: Path, vocab_size: int, batch: int, seq_len: int,
              warmup: int, repeat: int, intra: int, inter: int, decode_steps: int | None):
    if not onnx_path:
        return None

    proc = psutil.Process(os.getpid()) if psutil else None
    if proc:
        proc.cpu_percent(interval=None)  # prime the internal counters

    sess = make_session(onnx_path, intra=intra, inter=inter)

    # Memory snapshot before runs
    cpu0, rss0 = cpu_mem_snapshot(proc) if proc else (float("nan"), float("nan"))

    # Full sequence pass
    avg_ms_full, tps_full = run_full_sequence(sess, vocab_size, batch, seq_len, reps=repeat, warmup=warmup)

    # Optional decode emulation
    decode_ms, decode_tps = (None, None)
    if decode_steps and decode_steps > 0:
        decode_ms, decode_tps = run_decode_emulation(sess, vocab_size, batch, seq_len, decode_steps, reps=max(1, repeat // 2), warmup=warmup)

    # Memory snapshot after runs
    cpu1, rss1 = cpu_mem_snapshot(proc) if proc else (float("nan"), float("nan"))
    peak_rss = max(rss0, rss1)

    return {
        "label": label,
        "onnx": str(onnx_path),
        "batch": batch,
        "seq_len": seq_len,
        "threads": f"intra={intra}, inter={inter}",
        "avg_ms_full": round(avg_ms_full, 3),
        "tps_full": round(tps_full, 2),
        "decode_ms_token": (None if decode_ms is None else round(decode_ms, 3)),
        "decode_tps": (None if decode_tps is None else round(decode_tps, 2)),
        "rss_mb": (None if np.isnan(peak_rss) else round(peak_rss, 1)),
        "cpu_percent": (None if np.isnan(cpu1) else round(cpu1, 1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-fp32", type=str, default="", help="Path to FP32 ONNX (e.g., out/onnx/model.onnx)")
    ap.add_argument("--model-fp16", type=str, default="", help="Path to FP16 ONNX (e.g., out/onnx_fp16/model.fp16.onnx)")
    ap.add_argument("--model-int8", type=str, default="", help="Path to INT8 ONNX (e.g., out/onnx/model.int8.onnx)")
    ap.add_argument("--vocab-size", type=int, required=True, help="Vocab size used by the model (e.g., 8000)")
    ap.add_argument("--batch", type=int, default=1, help="Batch size for the benchmark")
    ap.add_argument("--seq-len", type=int, default=1024, help="Sequence length for the benchmark")
    ap.add_argument("--warmup", type=int, default=3, help="Warmup runs before timing")
    ap.add_argument("--repeat", type=int, default=20, help="Number of timed runs")
    ap.add_argument("--decode-steps", type=int, default=0, help="If >0, run decode emulation loop for this many steps")
    ap.add_argument("--intra", type=int, default=4, help="intra_op_num_threads for ORT")
    ap.add_argument("--inter", type=int, default=1, help="inter_op_num_threads for ORT")
    args = ap.parse_args()

    rows = []
    for label, path in [
        ("FP32", Path(args.model_fp32) if args.model_fp32 else None),
        ("FP16", Path(args.model_fp16) if args.model_fp16 else None),
        ("INT8", Path(args.model_int8) if args.model_int8 else None),
    ]:
        if path:
            res = bench_one(
                label=label, onnx_path=path, vocab_size=args.vocab_size,
                batch=args.batch, seq_len=args.seq_len,
                warmup=args.warmup, repeat=args.repeat,
                intra=args.intra, inter=args.inter,
                decode_steps=(args.decode_steps if args.decode_steps > 0 else None),
            )
            if res:
                rows.append(res)

    # Pretty print
    if not rows:
        print("No models provided. Pass at least one of --model-fp32/--model-fp16/--model-int8")
        return

    print("\n=== ONNX CPU Benchmark Results ===")
    for r in rows:
        print(f"[{r['label']}] {r['onnx']}")
        print(f"  threads:     {r['threads']}")
        print(f"  batch x len: {r['batch']} x {r['seq_len']}")
        print(f"  FULL  avg_ms: {r['avg_ms_full']} ms    throughput: {r['tps_full']} tok/s")
        if r['decode_ms_token'] is not None:
            print(f"  DECODE ms/tok: {r['decode_ms_token']} ms     throughput: {r['decode_tps']} tok/s")
        if r['rss_mb'] is not None:
            print(f"  RSS (peak approx): {r['rss_mb']} MB")
        if r['cpu_percent'] is not None:
            print(f"  CPU% (process snapshot): {r['cpu_percent']}%")
        print()

    print("TIP: Tune --intra/--inter to keep CPU ≤ 20% on your Xeon D-1548. Start with --intra 4 --inter 1.")
    print("NOTE: 'decode' loop is pessimistic without KV cache I/O. We can export a streaming graph later to measure true per-token cost.")


if __name__ == "__main__":
    main()
