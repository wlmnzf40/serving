#!/usr/bin/env python3
"""
GEMM benchmark via TF Serving REST API.

Dependencies: pip install requests numpy
(No TensorFlow installation required.)

Usage:
  python3 benchmark_rest.py --M 512 --K 512 --N 512 --iters 100 --warmup 10
"""
import argparse
import json
import time

import numpy as np
import requests


def make_request_body(A: np.ndarray, B: np.ndarray) -> dict:
    # TF Serving accepts nested lists as tensor values.
    return {
        "inputs": {
            "A": A.tolist(),
            "B": B.tolist(),
        }
    }


def run_benchmark(host: str, port: int, M: int, K: int, N: int,
                  iters: int, warmup: int, dtype: str):
    url = f"http://{host}:{port}/v1/models/gemm:predict"
    np_dtype = np.float32 if dtype == 'float32' else np.float16

    rng = np.random.default_rng(42)
    A = rng.standard_normal((M, K)).astype(np_dtype)
    B = rng.standard_normal((K, N)).astype(np_dtype)
    body = json.dumps(make_request_body(A, B))
    headers = {"Content-Type": "application/json"}

    # --- Connectivity check ---
    try:
        r = requests.get(f"http://{host}:{port}/v1/models/gemm", timeout=5)
        r.raise_for_status()
    except Exception as e:
        raise SystemExit(f"Cannot reach TF Serving at {url}: {e}")

    # --- Warmup ---
    print(f"Warmup ({warmup} iters) ...", flush=True)
    for _ in range(warmup):
        resp = requests.post(url, data=body, headers=headers, timeout=30)
        resp.raise_for_status()

    # --- Benchmark ---
    print(f"Benchmarking ({iters} iters) ...", flush=True)
    latencies_ms: list[float] = []

    for _ in range(iters):
        t0 = time.perf_counter()
        resp = requests.post(url, data=body, headers=headers, timeout=30)
        resp.raise_for_status()
        latencies_ms.append((time.perf_counter() - t0) * 1e3)

    lat = np.array(latencies_ms)
    flops = 2 * M * N * K                   # multiply-add counted as 2 ops
    gflops = flops / (lat.mean() / 1e3) / 1e9

    print()
    print("=" * 50)
    print(f"  Matrix sizes : ({M}x{K}) x ({K}x{N})")
    print(f"  dtype        : {dtype}")
    print(f"  FLOPs/call   : {flops/1e9:.3f} GFLOP")
    print(f"  Latency avg  : {lat.mean():.2f} ms")
    print(f"  Latency p50  : {np.percentile(lat, 50):.2f} ms")
    print(f"  Latency p99  : {np.percentile(lat, 99):.2f} ms")
    print(f"  Throughput   : {gflops:.2f} GFLOPS")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GEMM benchmark via TF Serving REST')
    parser.add_argument('--host',   default='localhost')
    parser.add_argument('--port',   type=int, default=8501)
    parser.add_argument('--M',      type=int, default=512, help='Rows of A')
    parser.add_argument('--K',      type=int, default=512, help='Cols of A / Rows of B')
    parser.add_argument('--N',      type=int, default=512, help='Cols of B')
    parser.add_argument('--iters',  type=int, default=50)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--dtype',  choices=['float32', 'float16'], default='float32')
    args = parser.parse_args()

    run_benchmark(args.host, args.port, args.M, args.K, args.N,
                  args.iters, args.warmup, args.dtype)
