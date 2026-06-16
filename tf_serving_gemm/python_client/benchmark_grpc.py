#!/usr/bin/env python3
"""
GEMM benchmark via TF Serving gRPC API.

Dependencies: pip install grpcio tensorflow-serving-api numpy
(tensorflow-serving-api is a standalone package, no full TF needed.)

Usage:
  python3 benchmark_grpc.py --M 512 --K 512 --N 512 --iters 100 --warmup 10
"""
import argparse
import time

import grpc
import numpy as np
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from tensorflow.core.framework import tensor_pb2, types_pb2
from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto


def ndarray_to_proto(arr: np.ndarray) -> tensor_pb2.TensorProto:
    proto = tensor_pb2.TensorProto()
    proto.dtype = types_pb2.DT_FLOAT
    shape = TensorShapeProto()
    for dim in arr.shape:
        shape.dim.add().size = dim
    proto.tensor_shape.CopyFrom(shape)
    proto.float_val.extend(arr.flatten().tolist())
    return proto


def run_benchmark(host: str, port: int, M: int, K: int, N: int,
                  iters: int, warmup: int):
    channel = grpc.insecure_channel(
        f'{host}:{port}',
        options=[
            ('grpc.max_send_message_length',    256 * 1024 * 1024),
            ('grpc.max_receive_message_length', 256 * 1024 * 1024),
        ],
    )
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    rng = np.random.default_rng(42)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'gemm'
    request.model_spec.signature_name = 'serving_default'
    request.inputs['A'].CopyFrom(ndarray_to_proto(A))
    request.inputs['B'].CopyFrom(ndarray_to_proto(B))

    TIMEOUT = 30.0

    # --- Warmup ---
    print(f"Warmup ({warmup} iters) ...", flush=True)
    for _ in range(warmup):
        stub.Predict(request, TIMEOUT)

    # --- Benchmark ---
    print(f"Benchmarking ({iters} iters) ...", flush=True)
    latencies_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        stub.Predict(request, TIMEOUT)
        latencies_ms.append((time.perf_counter() - t0) * 1e3)

    lat = np.array(latencies_ms)
    flops = 2 * M * N * K
    gflops = flops / (lat.mean() / 1e3) / 1e9

    print()
    print("=" * 50)
    print(f"  Matrix sizes : ({M}x{K}) x ({K}x{N})")
    print(f"  dtype        : float32")
    print(f"  FLOPs/call   : {flops/1e9:.3f} GFLOP")
    print(f"  Latency avg  : {lat.mean():.2f} ms")
    print(f"  Latency p50  : {np.percentile(lat, 50):.2f} ms")
    print(f"  Latency p99  : {np.percentile(lat, 99):.2f} ms")
    print(f"  Throughput   : {gflops:.2f} GFLOPS")
    print("=" * 50)

    channel.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GEMM benchmark via TF Serving gRPC')
    parser.add_argument('--host',   default='localhost')
    parser.add_argument('--port',   type=int, default=8500)
    parser.add_argument('--M',      type=int, default=512)
    parser.add_argument('--K',      type=int, default=512)
    parser.add_argument('--N',      type=int, default=512)
    parser.add_argument('--iters',  type=int, default=50)
    parser.add_argument('--warmup', type=int, default=10)
    args = parser.parse_args()

    run_benchmark(args.host, args.port, args.M, args.K, args.N,
                  args.iters, args.warmup)
